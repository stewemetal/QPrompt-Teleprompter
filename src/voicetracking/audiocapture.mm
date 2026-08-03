// SPDX-FileCopyrightText: QPrompt voice tracking
// SPDX-License-Identifier: GPL-3.0-or-later
//
// AVFoundation microphone capture for macOS. See audiocapture.h for the
// threading contract.
//
// Deliberately NOT SDL2: the project must not gain that dependency, and
// AVFoundation is what carries the TCC microphone permission that attaches to
// the app bundle.

#include "audiocapture.h"

#import <AVFoundation/AVFoundation.h>
#import <Foundation/Foundation.h>

#include <algorithm>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <fstream>
#include <mutex>
#include <vector>

namespace qprompt {
namespace voicetracking {

// ===========================================================================
// AudioRingBuffer
// ===========================================================================

namespace {

size_t roundUpPow2(size_t v)
{
    if (v < 2) return 2;
    size_t p = 2;
    while (p < v) {
        const size_t next = p << 1;
        if (next < p) return p;  // overflow guard
        p = next;
    }
    return p;
}

}  // namespace

AudioRingBuffer::AudioRingBuffer(size_t capacitySamples)
{
    m_capacity = roundUpPow2(capacitySamples);
    m_mask = m_capacity - 1;
    m_data.assign(m_capacity, 0.0f);
}

size_t AudioRingBuffer::write(const float *samples, size_t count)
{
    if (!samples || count == 0) return 0;

    // Only this thread writes m_writeIndex, so a relaxed load of our own index
    // is fine. m_readIndex is written only by the consumer, so acquire it.
    const uint64_t w = m_writeIndex.load(std::memory_order_relaxed);
    const uint64_t r = m_readIndex.load(std::memory_order_acquire);
    const size_t used = static_cast<size_t>(w - r);

    // Leave one slot empty so full and empty are distinguishable.
    const size_t freeSpace = (m_capacity - 1) - used;

    size_t dropped = 0;
    if (count > freeSpace) {
        dropped = count - freeSpace;
        count = freeSpace;
        m_dropped.fetch_add(dropped, std::memory_order_relaxed);
    }
    if (count == 0) return dropped;

    // Bounded memcpy, at most two segments. No allocation, no lock.
    const size_t start = static_cast<size_t>(w) & m_mask;
    const size_t firstChunk = std::min(count, m_capacity - start);
    std::memcpy(m_data.data() + start, samples, firstChunk * sizeof(float));
    if (firstChunk < count) {
        std::memcpy(m_data.data(), samples + firstChunk,
                    (count - firstChunk) * sizeof(float));
    }

    m_writeIndex.store(w + count, std::memory_order_release);
    return dropped;
}

size_t AudioRingBuffer::peek(float *out, size_t maxCount) const
{
    if (!out || maxCount == 0) return 0;
    const uint64_t r = m_readIndex.load(std::memory_order_relaxed);
    const uint64_t w = m_writeIndex.load(std::memory_order_acquire);
    const size_t used = static_cast<size_t>(w - r);
    const size_t n = std::min(maxCount, used);
    if (n == 0) return 0;

    const size_t start = static_cast<size_t>(r) & m_mask;
    const size_t firstChunk = std::min(n, m_capacity - start);
    std::memcpy(out, m_data.data() + start, firstChunk * sizeof(float));
    if (firstChunk < n) {
        std::memcpy(out + firstChunk, m_data.data(),
                    (n - firstChunk) * sizeof(float));
    }
    return n;
}

size_t AudioRingBuffer::read(float *out, size_t maxCount)
{
    const size_t n = peek(out, maxCount);
    if (n > 0) {
        const uint64_t r = m_readIndex.load(std::memory_order_relaxed);
        m_readIndex.store(r + n, std::memory_order_release);
    }
    return n;
}

size_t AudioRingBuffer::discard(size_t count)
{
    const uint64_t r = m_readIndex.load(std::memory_order_relaxed);
    const uint64_t w = m_writeIndex.load(std::memory_order_acquire);
    const size_t used = static_cast<size_t>(w - r);
    const size_t n = std::min(count, used);
    if (n > 0) m_readIndex.store(r + n, std::memory_order_release);
    return n;
}

size_t AudioRingBuffer::available() const
{
    const uint64_t r = m_readIndex.load(std::memory_order_acquire);
    const uint64_t w = m_writeIndex.load(std::memory_order_acquire);
    if (w <= r) return 0;
    return static_cast<size_t>(w - r);
}

void AudioRingBuffer::clear()
{
    const uint64_t w = m_writeIndex.load(std::memory_order_acquire);
    m_readIndex.store(w, std::memory_order_release);
}

// ===========================================================================
// Permission
// ===========================================================================

const char *micPermissionName(MicPermission p)
{
    switch (p) {
    case MicPermission::NotDetermined: return "NotDetermined";
    case MicPermission::Granted:       return "Granted";
    case MicPermission::Denied:        return "Denied";
    case MicPermission::Restricted:    return "Restricted";
    case MicPermission::Unavailable:   return "Unavailable";
    }
    return "Unknown";
}

MicPermission AudioCapture::permission()
{
    @autoreleasepool {
        const AVAuthorizationStatus s =
            [AVCaptureDevice authorizationStatusForMediaType:AVMediaTypeAudio];
        switch (s) {
        case AVAuthorizationStatusNotDetermined: return MicPermission::NotDetermined;
        case AVAuthorizationStatusAuthorized:    return MicPermission::Granted;
        case AVAuthorizationStatusDenied:        return MicPermission::Denied;
        case AVAuthorizationStatusRestricted:    return MicPermission::Restricted;
        }
        return MicPermission::Unavailable;
    }
}

void AudioCapture::requestPermission(std::function<void(MicPermission)> done)
{
    if (!done) return;

    const MicPermission current = permission();
    if (current != MicPermission::NotDetermined) {
        // Never re-prompt: macOS will not show the dialog a second time, so a
        // request here would silently resolve to the stored answer anyway.
        done(current);
        return;
    }

    @autoreleasepool {
        // Heap-copy the std::function so the block can own it across threads.
        auto *holder = new std::function<void(MicPermission)>(std::move(done));
        [AVCaptureDevice requestAccessForMediaType:AVMediaTypeAudio
                                 completionHandler:^(BOOL granted) {
            (*holder)(granted ? MicPermission::Granted : MicPermission::Denied);
            delete holder;
        }];
    }
}

MicPermission AudioCapture::requestPermissionBlocking(double timeoutSeconds)
{
    const MicPermission current = permission();
    if (current != MicPermission::NotDetermined) return current;

    // Shared state is refcounted through a shared_ptr captured by both sides so
    // a late callback after the timeout cannot touch a dead stack frame. This is
    // the "never hang, never crash" requirement made concrete.
    struct Shared {
        std::mutex mutex;
        std::condition_variable cv;
        bool done = false;
        MicPermission result = MicPermission::NotDetermined;
    };
    auto shared = std::make_shared<Shared>();

    requestPermission([shared](MicPermission p) {
        std::lock_guard<std::mutex> lock(shared->mutex);
        shared->result = p;
        shared->done = true;
        shared->cv.notify_all();
    });

    std::unique_lock<std::mutex> lock(shared->mutex);
    shared->cv.wait_for(
        lock,
        std::chrono::milliseconds(static_cast<long long>(timeoutSeconds * 1000.0)),
        [&] { return shared->done; });

    return shared->done ? shared->result : permission();
}

// ===========================================================================
// AudioCapture
// ===========================================================================

struct AudioCapture::Impl {
    AVAudioEngine *engine = nil;
    AVAudioConverter *converter = nil;
    AVAudioFormat *outputFormat = nil;
    double deviceRate = 0.0;
    bool running = false;

    // Scratch buffer reused by the audio thread. Allocated once at start().
    AVAudioPCMBuffer *scratch = nil;
};

AudioCapture::AudioCapture() : AudioCapture(Config{}) {}

AudioCapture::AudioCapture(const Config &config)
    : m_impl(new Impl), m_config(config)
{
    const size_t cap = static_cast<size_t>(
        std::max(1.0, m_config.ringSeconds) * kWhisperSampleRate);
    m_ring.reset(new AudioRingBuffer(cap));
}

AudioCapture::~AudioCapture()
{
    stop();
}

bool AudioCapture::isRunning() const
{
    return m_impl && m_impl->running;
}

double AudioCapture::deviceSampleRate() const
{
    return m_impl ? m_impl->deviceRate : 0.0;
}

bool AudioCapture::start()
{
    m_lastError.clear();
    if (isRunning()) return true;

    const MicPermission p = permission();
    if (p != MicPermission::Granted) {
        // Fail cleanly. Do NOT prompt from here: prompting has UI consequences
        // and must be an explicit caller decision.
        m_lastError = std::string("microphone permission is ") + micPermissionName(p);
        return false;
    }

    @autoreleasepool {
        @try {
            m_impl->engine = [[AVAudioEngine alloc] init];
            AVAudioInputNode *input = m_impl->engine.inputNode;
            if (!input) {
                m_lastError = "no audio input node available";
                m_impl->engine = nil;
                return false;
            }

            AVAudioFormat *inFormat = [input outputFormatForBus:0];
            if (!inFormat || inFormat.sampleRate <= 0 || inFormat.channelCount == 0) {
                m_lastError = "input device reported an unusable format "
                              "(no microphone connected?)";
                m_impl->engine = nil;
                return false;
            }
            m_impl->deviceRate = inFormat.sampleRate;

            // Target: 16 kHz, mono, float32, non-interleaved. This is exactly
            // what whisper_full() wants, so no further conversion downstream.
            m_impl->outputFormat = [[AVAudioFormat alloc]
                initWithCommonFormat:AVAudioPCMFormatFloat32
                          sampleRate:static_cast<double>(kWhisperSampleRate)
                            channels:1
                         interleaved:NO];

            // AVAudioConverter does proper band-limited SRC (and channel
            // downmix) in vDSP. This is the "resample properly, do not
            // decimate" requirement.
            m_impl->converter = [[AVAudioConverter alloc]
                initFromFormat:inFormat toFormat:m_impl->outputFormat];
            if (!m_impl->converter) {
                m_lastError = "could not create AVAudioConverter for the device format";
                m_impl->engine = nil;
                return false;
            }

            // Pre-allocate the output scratch buffer so the audio thread never
            // allocates. Sized for the worst case: a tap block at the device
            // rate converted up... it can only shrink going to 16 kHz, but size
            // generously anyway to tolerate AVAudioEngine handing us bigger
            // blocks than requested.
            const double ratio =
                static_cast<double>(kWhisperSampleRate) / m_impl->deviceRate;
            const AVAudioFrameCount outCap = static_cast<AVAudioFrameCount>(
                std::ceil(m_config.tapFrames * 4 * std::max(1.0, ratio)) + 1024);
            m_impl->scratch = [[AVAudioPCMBuffer alloc]
                initWithPCMFormat:m_impl->outputFormat frameCapacity:outCap];

            AudioRingBuffer *ring = m_ring.get();
            AVAudioConverter *converter = m_impl->converter;
            AVAudioPCMBuffer *scratch = m_impl->scratch;

            [input installTapOnBus:0
                        bufferSize:m_config.tapFrames
                            format:inFormat
                             block:^(AVAudioPCMBuffer *buffer, AVAudioTime *) {
                // ============ REAL-TIME AUDIO THREAD ============
                // No locks, no allocation, no logging, no Obj-C object creation
                // in the steady state. AVAudioConverter's convertToBuffer uses
                // the pre-allocated scratch buffer.
                if (!buffer || buffer.frameLength == 0) return;

                __block BOOL consumed = NO;
                AVAudioConverterInputBlock inputBlock =
                    ^AVAudioBuffer *(AVAudioPacketCount, AVAudioConverterInputStatus *status) {
                        if (consumed) {
                            *status = AVAudioConverterInputStatus_NoDataNow;
                            return nil;
                        }
                        consumed = YES;
                        *status = AVAudioConverterInputStatus_HaveData;
                        return buffer;
                    };

                scratch.frameLength = scratch.frameCapacity;
                NSError *err = nil;
                const AVAudioConverterOutputStatus st =
                    [converter convertToBuffer:scratch
                                         error:&err
                            withInputFromBlock:inputBlock];

                if (st == AVAudioConverterOutputStatus_Error) {
                    // Drop the block. Never throw or log from here.
                    return;
                }
                const AVAudioFrameCount n = scratch.frameLength;
                if (n == 0) return;
                const float *const *ch = scratch.floatChannelData;
                if (!ch || !ch[0]) return;

                ring->write(ch[0], static_cast<size_t>(n));
            }];

            NSError *startError = nil;
            [m_impl->engine prepare];
            if (![m_impl->engine startAndReturnError:&startError]) {
                m_lastError = startError
                    ? std::string([[startError localizedDescription] UTF8String])
                    : std::string("AVAudioEngine failed to start");
                [input removeTapOnBus:0];
                m_impl->engine = nil;
                m_impl->converter = nil;
                m_impl->scratch = nil;
                return false;
            }

            m_impl->running = true;
            return true;
        } @catch (NSException *ex) {
            // AVFoundation throws NSExceptions for format mismatches. Catch
            // them so a missing/odd device can never take down the app.
            m_lastError = ex.reason
                ? std::string([ex.reason UTF8String])
                : std::string("AVFoundation exception during start");
            m_impl->engine = nil;
            m_impl->converter = nil;
            m_impl->scratch = nil;
            m_impl->running = false;
            return false;
        }
    }
}

void AudioCapture::stop()
{
    if (!m_impl || !m_impl->running) return;
    @autoreleasepool {
        @try {
            AVAudioInputNode *input = m_impl->engine.inputNode;
            if (input) [input removeTapOnBus:0];
            [m_impl->engine stop];
        } @catch (NSException *) {
            // Nothing useful to do; stopping must never throw.
        }
        m_impl->engine = nil;
        m_impl->converter = nil;
        m_impl->scratch = nil;
        m_impl->outputFormat = nil;
        m_impl->running = false;
    }
}

// ===========================================================================
// Offline resampling + WAV loading
// ===========================================================================

std::vector<float> resampleLinear(const std::vector<float> &input,
                                  double inputRate,
                                  double outputRate)
{
    std::vector<float> out;
    if (input.empty() || inputRate <= 0.0 || outputRate <= 0.0) return out;
    if (std::fabs(inputRate - outputRate) < 1e-9) return input;

    const double ratio = outputRate / inputRate;
    const size_t outCount =
        static_cast<size_t>(std::floor(input.size() * ratio));
    if (outCount == 0) return out;
    out.resize(outCount);

    // Windowed-sinc kernel. Cutoff sits at the lower Nyquist so downsampling
    // low-pass filters first (this is what stops 48k->16k aliasing everything
    // above 8 kHz straight into the speech band).
    // Pull the cutoff slightly below Nyquist (0.45 rather than 0.5) so the
    // transition band lands inside the guard region instead of straddling
    // Nyquist, where a finite-length kernel would leak.
    const double cutoff = std::min(1.0, ratio) * 0.45;  // cycles/input-sample
    const int halfWidth = 32;  // 65-tap Blackman-windowed sinc. 33 taps left a
                               // 12 kHz tone at -16 dB after 48k->16k, which is
                               // audible aliasing in the speech band; 65 taps
                               // push it well below -30 dB. Offline path only,
                               // so the extra cost is irrelevant.

    const double step = 1.0 / ratio;  // input samples per output sample
    const int n = static_cast<int>(input.size());

    for (size_t i = 0; i < outCount; ++i) {
        const double center = i * step;
        const int base = static_cast<int>(std::floor(center));
        double acc = 0.0;
        double norm = 0.0;
        for (int k = -halfWidth; k <= halfWidth; ++k) {
            const int idx = base + k;
            if (idx < 0 || idx >= n) continue;
            const double t = (idx - center);            // in input samples
            const double x = 2.0 * cutoff * t;
            double sinc;
            if (std::fabs(x) < 1e-9) {
                sinc = 1.0;
            } else {
                sinc = std::sin(M_PI * x) / (M_PI * x);
            }
            // Blackman window over the kernel support.
            const double wpos = (t + halfWidth) / (2.0 * halfWidth);
            const double w = (wpos < 0.0 || wpos > 1.0)
                ? 0.0
                : 0.42 - 0.5 * std::cos(2.0 * M_PI * wpos)
                       + 0.08 * std::cos(4.0 * M_PI * wpos);
            const double coeff = sinc * w;
            acc += input[static_cast<size_t>(idx)] * coeff;
            norm += coeff;
        }
        out[i] = static_cast<float>(norm > 1e-12 ? acc / norm : 0.0);
    }
    return out;
}

namespace {

uint32_t rd32(const unsigned char *p)
{
    return static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8)
         | (static_cast<uint32_t>(p[2]) << 16) | (static_cast<uint32_t>(p[3]) << 24);
}
uint16_t rd16(const unsigned char *p)
{
    return static_cast<uint16_t>(static_cast<uint32_t>(p[0])
                                 | (static_cast<uint32_t>(p[1]) << 8));
}

}  // namespace

bool loadWavAsMono16k(const std::string &path,
                      std::vector<float> &outSamples,
                      std::string &error)
{
    outSamples.clear();
    error.clear();

    std::ifstream f(path, std::ios::binary);
    if (!f) { error = "cannot open " + path; return false; }

    std::vector<unsigned char> bytes((std::istreambuf_iterator<char>(f)),
                                     std::istreambuf_iterator<char>());
    if (bytes.size() < 44) { error = "file too short to be a WAV"; return false; }
    if (std::memcmp(bytes.data(), "RIFF", 4) != 0
        || std::memcmp(bytes.data() + 8, "WAVE", 4) != 0) {
        error = "not a RIFF/WAVE file";
        return false;
    }

    uint16_t format = 0, channels = 0, bitsPerSample = 0;
    uint32_t sampleRate = 0;
    size_t dataOffset = 0, dataSize = 0;

    size_t pos = 12;
    while (pos + 8 <= bytes.size()) {
        const char *id = reinterpret_cast<const char *>(bytes.data() + pos);
        const uint32_t size = rd32(bytes.data() + pos + 4);
        const size_t body = pos + 8;
        if (std::memcmp(id, "fmt ", 4) == 0 && size >= 16 && body + 16 <= bytes.size()) {
            format        = rd16(bytes.data() + body);
            channels      = rd16(bytes.data() + body + 2);
            sampleRate    = rd32(bytes.data() + body + 4);
            bitsPerSample = rd16(bytes.data() + body + 14);
            if (format == 0xFFFE && size >= 40 && body + 26 <= bytes.size()) {
                format = rd16(bytes.data() + body + 24);  // WAVE_FORMAT_EXTENSIBLE
            }
        } else if (std::memcmp(id, "data", 4) == 0) {
            dataOffset = body;
            dataSize = std::min<size_t>(size, bytes.size() - body);
        }
        pos = body + size + (size & 1u);  // chunks are word-aligned
    }

    if (channels == 0 || sampleRate == 0 || dataSize == 0) {
        error = "WAV missing fmt or data chunk";
        return false;
    }
    if (format != 1 && format != 3) {
        error = "unsupported WAV encoding (only PCM and IEEE float)";
        return false;
    }

    const size_t bytesPerSample = bitsPerSample / 8;
    if (bytesPerSample == 0) { error = "invalid bits per sample"; return false; }
    const size_t frames = dataSize / (bytesPerSample * channels);
    if (frames == 0) { error = "WAV contains no frames"; return false; }

    std::vector<float> mono(frames, 0.0f);
    const unsigned char *d = bytes.data() + dataOffset;

    for (size_t i = 0; i < frames; ++i) {
        double acc = 0.0;
        for (size_t c = 0; c < channels; ++c) {
            const unsigned char *s = d + (i * channels + c) * bytesPerSample;
            double v = 0.0;
            if (format == 3 && bitsPerSample == 32) {
                float tmp; std::memcpy(&tmp, s, 4); v = tmp;
            } else if (format == 3 && bitsPerSample == 64) {
                double tmp; std::memcpy(&tmp, s, 8); v = tmp;
            } else if (bitsPerSample == 8) {
                v = (static_cast<int>(s[0]) - 128) / 128.0;   // 8-bit WAV is unsigned
            } else if (bitsPerSample == 16) {
                int16_t tmp; std::memcpy(&tmp, s, 2); v = tmp / 32768.0;
            } else if (bitsPerSample == 24) {
                int32_t tmp = (static_cast<int32_t>(s[0]) << 8)
                            | (static_cast<int32_t>(s[1]) << 16)
                            | (static_cast<int32_t>(s[2]) << 24);
                v = (tmp >> 8) / 8388608.0;
            } else if (bitsPerSample == 32) {
                int32_t tmp; std::memcpy(&tmp, s, 4); v = tmp / 2147483648.0;
            } else {
                error = "unsupported bit depth";
                return false;
            }
            acc += v;
        }
        mono[i] = static_cast<float>(acc / channels);
    }

    if (sampleRate == static_cast<uint32_t>(kWhisperSampleRate)) {
        outSamples = std::move(mono);
    } else {
        outSamples = resampleLinear(mono, sampleRate, kWhisperSampleRate);
    }
    return true;
}

}  // namespace voicetracking
}  // namespace qprompt
