// SPDX-FileCopyrightText: QPrompt voice tracking
// SPDX-License-Identifier: GPL-3.0-or-later
//
// whisper.cpp sliding-window driver. Turns a stream of 16 kHz mono float32 into
// a stream of `Observation`s for the Aligner.
//
// Plain C++17 header: no Qt, no Obj-C, no whisper.h leakage (the context is a
// forward-declared opaque pointer), so tests and the Qt adapter both include it
// freely.
//
// ============================================================================
// WHY A SLIDING WINDOW, AND WHAT THE TIMESTAMPS MEAN
// ============================================================================
// whisper.cpp is not a streaming recognizer. It transcribes a fixed block of
// audio. To approximate streaming, we keep a rolling window of the most recent
// `windowSec` of audio and re-transcribe the WHOLE window every `hopSec`.
//
// Consecutive results therefore overlap heavily and freely contradict each
// other. That is expected and correct — the Aligner is built for it. Our job is
// only to label each result honestly:
//
//   Observation.audioTimeSec = the START TIME OF THE WINDOW, in AUDIO time
//                              (samples consumed / 16000), never wall clock.
//   Observation.isVolatile   = true, always, for live windows.
//
// audioTimeSec is what drives the Aligner's checkpoint/rollback. Because the
// window start advances by exactly one hop per emission, successive
// observations carry monotonically increasing start times spaced `hopSec`
// apart, and each one re-describes `windowSec - hopSec` of audio that a previous
// observation already covered. The Aligner rolls back to the checkpoint at that
// audio time and re-applies, so the same audio moves the position exactly once.
//
// Getting this wrong — e.g. using window END time, or wall-clock time — breaks
// revision handling silently and the prompter runs away from the speaker.
//
// isVolatile is always true for live audio: whisper may revise any part of a
// window that is still within `windowSec` of the present. The one case that
// emits isVolatile=false is the final flush of a FILE source, where no further
// audio can ever arrive.
// ============================================================================

#ifndef QPROMPT_VOICETRACKING_SPEECHRECOGNIZER_H
#define QPROMPT_VOICETRACKING_SPEECHRECOGNIZER_H

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "aligner.h"      // Observation
#include "audiocapture.h" // AudioRingBuffer, AudioCapture

struct whisper_context;  // opaque; whisper.h is included only in the .cpp

namespace qprompt {
namespace voicetracking {

class SpeechRecognizer {
public:
    struct Config {
        // Absolute path to a MULTILINGUAL ggml model. The `.en` models cannot
        // do Hungarian and are rejected at load time by whisper itself for a
        // non-"en" language; we additionally warn.
        std::string modelPath;

        // ISO-639-1 code. "hu" and "en" are both first-class. Never hardcode.
        // "auto" lets whisper detect, which is NOT recommended here: detection
        // flips between languages mid-session on short windows.
        std::string language = "en";

        // Rolling window length. 5.0 s: long enough that whisper has real
        // context (it degrades badly below ~2 s and starts hallucinating on
        // near-silent short buffers), short enough that a 500 ms hop only
        // re-does 10x the work of the new audio, and short enough that the
        // Aligner's rollback depth stays small.
        double windowSec = 5.0;

        // How often we re-transcribe. 0.5 s per the design doc. This sets the
        // observation rate the control loop sees; going lower buys latency you
        // cannot use because whisper's own encode already costs ~50-150 ms.
        double hopSec = 0.5;

        // Do not run inference on a window with less than this much audio.
        // Raised from 1.0 to 2.0 after the Hungarian end-to-end run: the 1.0 s
        // and 1.5 s ramp-up windows were exactly the ones where whisper base
        // fell into a degenerate repetition loop. 2.0 s costs one extra hop of
        // startup latency (~500 ms, once per session) and removes a whole class
        // of hallucinated observation.
        double minWindowSec = 2.0;

        // 0 = auto (hardware_concurrency capped at 8). whisper scales poorly
        // past ~8 threads and we must leave cores for the UI thread and Metal.
        int nThreads = 0;

        // Skip windows whose RMS is below this. Cheap VAD: whisper will invent
        // text ("Köszönöm a figyelmet!", "Thank you.") for silence, and a
        // hallucinated observation is worse than no observation at all — the
        // design doc's "prefer failing safe over emitting a wrong one".
        float silenceRms = 0.004f;

        // Use the Metal backend. Off makes startup deterministic in tests.
        bool useGpu = true;
    };

    // Emitted for every window that produced text. Called on the INFERENCE
    // thread, never the audio thread and never the caller's thread. Keep it
    // fast; do not block on it.
    using ObservationCallback = std::function<void(const Observation &)>;

    SpeechRecognizer();
    ~SpeechRecognizer();

    SpeechRecognizer(const SpeechRecognizer &) = delete;
    SpeechRecognizer &operator=(const SpeechRecognizer &) = delete;

    // Loads the model. Returns false and sets lastError() on a missing file, an
    // unreadable file, or a whisper init failure. Never throws, never crashes.
    // Safe to call again to swap models (stops first).
    bool loadModel(const Config &config);
    bool isModelLoaded() const;

    const Config &config() const { return m_config; }

    // Changing language does not require a model reload (it is a per-call
    // param). Takes effect on the next window.
    void setLanguage(const std::string &lang);

    void setObservationCallback(ObservationCallback cb);

    // --- Live microphone ----------------------------------------------------
    // Starts capture and the inference thread. Returns false (with lastError())
    // if the model is not loaded or the mic is unavailable/denied.
    bool startLive(AudioCapture &capture);

    // --- Offline file source (replay harness) -------------------------------
    // Feeds a WAV file through EXACTLY the same sliding-window path as live
    // audio, synchronously, on the calling thread. Observations arrive with the
    // same timestamps the live path would produce. The final partial window is
    // emitted with isVolatile=false because no more audio can arrive.
    //
    // Returns false on a load failure or no model.
    bool runFile(const std::string &wavPath);

    // Feeds raw 16 kHz mono float32 through the same path. Same semantics as
    // runFile(); this is what the tests use.
    bool runSamples(const std::vector<float> &samples16k);

    // Stops the inference thread and (if we started it) capture. Safe to call
    // from any thread, safe while inference is running, safe twice, safe when
    // never started. Blocks until the inference thread has exited.
    void stop();
    bool isRunning() const { return m_running.load(std::memory_order_acquire); }

    // --- Diagnostics --------------------------------------------------------
    const std::string &lastError() const { return m_lastError; }

    // Windows skipped because inference was still busy when the hop came due.
    // Per the brief: falling behind is worse than queueing stale work, so we
    // drop. A steadily rising count means the model is too big for the machine.
    uint64_t skippedHops() const { return m_skippedHops.load(std::memory_order_relaxed); }
    // Windows skipped by the silence gate.
    uint64_t silentHops() const { return m_silentHops.load(std::memory_order_relaxed); }
    // Windows rejected because whisper produced a degenerate repetition loop
    // ("aki aki aki ..."). Observed in real Hungarian output; see the .cpp.
    uint64_t degenerateHops() const { return m_degenerateHops.load(std::memory_order_relaxed); }
    uint64_t emittedObservations() const { return m_emitted.load(std::memory_order_relaxed); }

    // Last raw transcript, for logging/debugging. Thread-safe-ish: a snapshot.
    std::string lastTranscript() const;

private:
    // Runs one whisper pass over `window`, emits at most one Observation.
    // `windowStartSec` is audio time of window[0].
    void transcribeWindow(const std::vector<float> &window,
                          double windowStartSec,
                          bool isVolatile);

    void liveLoop(AudioCapture *capture);

    whisper_context *m_ctx = nullptr;
    Config m_config;

    ObservationCallback m_callback;

    std::thread m_thread;
    std::atomic<bool> m_running{false};
    std::atomic<bool> m_stopRequested{false};

    // Guards m_callback and m_lastTranscript against stop()/setter races.
    mutable std::mutex m_mutex;
    std::string m_lastTranscript;

    std::atomic<uint64_t> m_skippedHops{0};
    std::atomic<uint64_t> m_silentHops{0};
    std::atomic<uint64_t> m_degenerateHops{0};
    std::atomic<uint64_t> m_emitted{0};

    std::string m_lastError;

    struct Detail;
    std::unique_ptr<Detail> m_detail;
};

}  // namespace voicetracking
}  // namespace qprompt

#endif  // QPROMPT_VOICETRACKING_SPEECHRECOGNIZER_H
