// SPDX-FileCopyrightText: QPrompt voice tracking
// SPDX-License-Identifier: GPL-3.0-or-later
//
// macOS microphone capture, delivering 16 kHz mono float32 to a consumer.
//
// The implementation is Objective-C++ (audiocapture.mm, AVFoundation), but this
// header is PLAIN C++17 with no Obj-C types, so it can be included from ordinary
// .cpp translation units and from tests. All Obj-C state lives behind an opaque
// pimpl.
//
// ============================================================================
// THREADING CONTRACT
// ============================================================================
// AVAudioEngine calls its input tap on a real-time audio thread. That thread
// must never block, never allocate, and never wait on a slow consumer. This
// file therefore provides two pieces:
//
//   AudioRingBuffer  — a lock-free single-producer/single-consumer float ring.
//                      The audio thread only ever does a bounded memcpy plus a
//                      release store. On overflow it DROPS THE OLDEST samples
//                      and increments a counter; it never blocks and never
//                      grows. Dropping is the correct policy here: this is a
//                      live-tracking feature, and stale audio is worthless.
//
//   AudioCapture     — owns the engine, the format conversion (device rate ->
//                      16 kHz mono float32 via AVAudioConverter, which does
//                      proper band-limited sample-rate conversion, not
//                      decimation), and the permission handling.
//
// The consumer (SpeechRecognizer) polls read() from its own thread. Producer and
// consumer indices are separate atomics; only one thread writes each.
// ============================================================================

#ifndef QPROMPT_VOICETRACKING_AUDIOCAPTURE_H
#define QPROMPT_VOICETRACKING_AUDIOCAPTURE_H

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace qprompt {
namespace voicetracking {

// The one and only sample rate whisper accepts.
inline constexpr int kWhisperSampleRate = 16000;

// ---------------------------------------------------------------------------
// AudioRingBuffer
// ---------------------------------------------------------------------------
// Lock-free SPSC ring of float samples. Capacity is fixed at construction and
// rounded up to a power of two so index wrapping is a mask rather than a modulo.
//
// Overflow policy: DROP THE NEWEST samples that do not fit, and count them.
//
// The obvious alternative — "writer wins", where the producer advances the read
// index to overwrite the oldest data — is WRONG for a true SPSC ring, because it
// makes both threads writers of m_readIndex. That is a data race, and it shows
// up as the consumer observing duplicated or reordered samples. This was caught
// by testRingConcurrent().
//
// So: exactly one thread writes each index. The producer owns m_writeIndex and
// only ever advances it by however much actually fit; the consumer owns
// m_readIndex. When a block does not fit, the tail of that block is discarded
// and added to droppedSamples().
//
// The practical consequence for capture is that a stalled consumer loses the
// audio that arrives during the stall rather than the audio already buffered.
// The SpeechRecognizer compensates: it watches droppedSamples(), and on any drop
// it discards its rolling window rather than transcribe a spliced buffer, and
// adds the dropped count to its audio clock so timestamps stay honest.
class AudioRingBuffer {
public:
    explicit AudioRingBuffer(size_t capacitySamples);

    // Producer side. Called from the real-time audio thread.
    // Never allocates, never blocks, never locks. Returns the number of samples
    // dropped to make room (0 in the normal case).
    size_t write(const float *samples, size_t count);

    // Consumer side. Copies up to `maxCount` samples out and advances the read
    // index. Returns the number actually copied.
    size_t read(float *out, size_t maxCount);

    // Consumer side. Copies up to `maxCount` samples WITHOUT advancing the read
    // index. Used by the sliding window, which wants to peek at a window while
    // only consuming one hop's worth.
    size_t peek(float *out, size_t maxCount) const;

    // Consumer side. Drops up to `count` samples from the front.
    size_t discard(size_t count);

    size_t available() const;
    size_t capacity() const { return m_capacity; }
    bool empty() const { return available() == 0; }

    // Total samples ever destroyed by overflow. Monotonic.
    uint64_t droppedSamples() const
    {
        return m_dropped.load(std::memory_order_relaxed);
    }

    // Consumer side only; not safe concurrently with write().
    void clear();

private:
    std::vector<float> m_data;
    size_t m_capacity = 0;  // power of two
    size_t m_mask = 0;

    // Monotonically increasing counts, masked on use. 64-bit so wraparound is
    // not a practical concern (2^64 samples at 16 kHz is ~36 million years).
    std::atomic<uint64_t> m_writeIndex{0};
    std::atomic<uint64_t> m_readIndex{0};
    std::atomic<uint64_t> m_dropped{0};
};

// ---------------------------------------------------------------------------
// Microphone authorization
// ---------------------------------------------------------------------------
enum class MicPermission {
    NotDetermined,  // the user has not been asked yet
    Granted,
    Denied,         // user said no, or the app is restricted by policy
    Restricted,
    Unavailable,    // no input device, or AVFoundation refused entirely
};

const char *micPermissionName(MicPermission p);

// ---------------------------------------------------------------------------
// AudioCapture
// ---------------------------------------------------------------------------
class AudioCapture {
public:
    struct Config {
        // Ring capacity in seconds of 16 kHz audio. 8s comfortably spans the
        // 5s analysis window plus scheduling slop, while staying small enough
        // (512 KB) that a stalled consumer cannot hoard memory.
        double ringSeconds = 8.0;

        // Requested tap block size at the DEVICE rate. AVAudioEngine treats this
        // as a hint. 4096 frames at 48 kHz is ~85 ms — large enough to keep
        // per-callback overhead negligible, small enough that the 500 ms hop is
        // not noticeably quantized.
        uint32_t tapFrames = 4096;
    };

    AudioCapture();
    explicit AudioCapture(const Config &config);
    ~AudioCapture();

    AudioCapture(const AudioCapture &) = delete;
    AudioCapture &operator=(const AudioCapture &) = delete;

    // Current authorization, without prompting.
    static MicPermission permission();

    // Prompts if and only if the state is NotDetermined; otherwise reports the
    // existing state immediately. The callback may fire on an arbitrary thread.
    // Never blocks, never throws.
    static void requestPermission(std::function<void(MicPermission)> done);

    // Synchronous convenience for non-GUI contexts (tests, CLI harnesses).
    // Waits up to `timeoutSeconds` for the user to answer the prompt, then
    // returns whatever state is known. Never hangs indefinitely.
    static MicPermission requestPermissionBlocking(double timeoutSeconds = 30.0);

    // Starts capture. Returns false and sets lastError() if permission is not
    // granted, no input device exists, or the engine refuses to start. Never
    // throws, never crashes on a denied mic.
    bool start();
    void stop();
    bool isRunning() const;

    // The ring the audio thread feeds. Valid for the lifetime of this object,
    // whether or not capture is running.
    AudioRingBuffer &ring() { return *m_ring; }
    const AudioRingBuffer &ring() const { return *m_ring; }

    // Native rate of the input device, in Hz. 0 until start() succeeds.
    double deviceSampleRate() const;

    // Human-readable reason the last start() failed. Empty on success.
    const std::string &lastError() const { return m_lastError; }

private:
    struct Impl;                    // Obj-C state, defined in the .mm
    std::unique_ptr<Impl> m_impl;
    std::unique_ptr<AudioRingBuffer> m_ring;
    Config m_config;
    std::string m_lastError;
};

// ---------------------------------------------------------------------------
// Offline sources — needed for the replay harness and for testing without a mic
// ---------------------------------------------------------------------------

// Reads a RIFF/WAVE file and returns mono float32 resampled to 16 kHz.
// Supports PCM 8/16/24/32-bit integer and 32/64-bit IEEE float, any channel
// count (channels are averaged down to mono), any sample rate.
// Returns false and fills `error` on any problem; never throws.
bool loadWavAsMono16k(const std::string &path,
                      std::vector<float> &outSamples,
                      std::string &error);

// Band-limited-ish sample rate conversion used by the WAV loader and by tests.
//
// This is a windowed-sinc resampler, not a decimator: dropping samples to go
// from 48 kHz to 16 kHz aliases everything above 8 kHz back down into the speech
// band, which is exactly the band whisper cares about. The kernel is a
// Blackman-windowed sinc with a cutoff at the lower of the two Nyquist limits,
// so downsampling low-pass filters before it decimates.
//
// The live capture path does NOT use this — AVAudioConverter does the same job
// in optimized vDSP code. This exists so the offline path and the tests do not
// depend on AVFoundation.
std::vector<float> resampleLinear(const std::vector<float> &input,
                                  double inputRate,
                                  double outputRate);

}  // namespace voicetracking
}  // namespace qprompt

#endif  // QPROMPT_VOICETRACKING_AUDIOCAPTURE_H
