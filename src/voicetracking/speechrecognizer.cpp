// SPDX-FileCopyrightText: QPrompt voice tracking
// SPDX-License-Identifier: GPL-3.0-or-later
//
// whisper.cpp sliding-window driver. See speechrecognizer.h for the timestamp
// contract, which is the part that is easy to get silently wrong.

#include "speechrecognizer.h"

#include "textnormalize.h"

#include "whisper.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <sys/stat.h>

namespace qprompt {
namespace voicetracking {

struct SpeechRecognizer::Detail {
    // Reused across windows so the steady state does not reallocate.
    std::vector<float> window;
    std::string languageBuffer;  // whisper_full_params holds a raw const char*
};

namespace {

// whisper logs verbosely to stderr by default, including per-call model dumps.
// Route it to nothing unless someone asks; a teleprompter must not spew.
void silentLog(enum ggml_log_level, const char *, void *) {}

bool fileExists(const std::string &path)
{
    struct stat st;
    return !path.empty() && ::stat(path.c_str(), &st) == 0 && S_ISREG(st.st_mode);
}

float rms(const float *s, size_t n)
{
    if (n == 0) return 0.0f;
    double acc = 0.0;
    for (size_t i = 0; i < n; ++i) acc += static_cast<double>(s[i]) * s[i];
    return static_cast<float>(std::sqrt(acc / static_cast<double>(n)));
}

int resolveThreads(int requested)
{
    if (requested > 0) return requested;
    unsigned hc = std::thread::hardware_concurrency();
    if (hc == 0) hc = 4;
    // Cap at 8: whisper's scaling flattens past that, and on Apple Silicon the
    // extra threads land on efficiency cores where they slow the batch down.
    // Also leave headroom for the Qt UI thread and the Metal command queue.
    return static_cast<int>(std::min(hc > 1 ? hc - 1 : 1u, 8u));
}

// whisper concatenates segment text with leading spaces and sometimes emits
// bracketed non-speech markers ("[BLANK_AUDIO]", "(music)"). Those are not
// words and must not reach the aligner.
bool isNoiseMarker(const std::string &t)
{
    if (t.size() < 2) return false;
    const char a = t.front(), b = t.back();
    return (a == '[' && b == ']') || (a == '(' && b == ')')
        || (a == '*' && b == '*');
}

// ---------------------------------------------------------------------------
// Degenerate-repetition guard.
//
// OBSERVED, not hypothesized: on the Hungarian end-to-end run, whisper base
// emitted windows like
//     "ilyen aki aki aki aki aki aki ... aki"   (73 tokens, 72 identical)
//     "ilyen a kis kis kis ... kis k"           (108 tokens)
// This is the classic greedy-decoder repetition loop, and it happens on windows
// that are short or start mid-word. Such an observation is actively harmful: 72
// copies of one token would hammer the aligner's beam with spurious evidence and
// could drag the posterior far off position — precisely the "confidently jumps
// to the wrong place" failure the design doc forbids.
//
// The cheap, reliable signal is the type/token ratio: real speech in a 5 s
// window has many distinct words; a repetition loop has almost none. Reject the
// whole observation rather than try to repair it — failing safe means emitting
// nothing, per the brief.
bool isDegenerateRepetition(const std::vector<std::string> &tokens)
{
    if (tokens.size() < 8) return false;  // too short to judge

    std::vector<std::string> uniq(tokens);
    std::sort(uniq.begin(), uniq.end());
    uniq.erase(std::unique(uniq.begin(), uniq.end()), uniq.end());

    const double ratio =
        static_cast<double>(uniq.size()) / static_cast<double>(tokens.size());

    // Real Hungarian/English speech in a 5 s window sits well above 0.5 even
    // with repeated function words. 0.34 leaves generous headroom while still
    // catching the observed loops (which score under 0.05).
    if (ratio < 0.34) return true;

    // Also catch a long run of one identical token even when the rest of the
    // window is varied, e.g. "... szavakat aki aki aki aki aki aki aki".
    size_t run = 1, maxRun = 1;
    for (size_t i = 1; i < tokens.size(); ++i) {
        run = (tokens[i] == tokens[i - 1]) ? run + 1 : 1;
        maxRun = std::max(maxRun, run);
    }
    return maxRun >= 6;
}

}  // namespace

// ===========================================================================

SpeechRecognizer::SpeechRecognizer() : m_detail(new Detail) {}

SpeechRecognizer::~SpeechRecognizer()
{
    stop();
    if (m_ctx) {
        whisper_free(m_ctx);
        m_ctx = nullptr;
    }
}

bool SpeechRecognizer::isModelLoaded() const { return m_ctx != nullptr; }

bool SpeechRecognizer::loadModel(const Config &config)
{
    stop();
    m_lastError.clear();

    if (m_ctx) {
        whisper_free(m_ctx);
        m_ctx = nullptr;
    }

    if (config.modelPath.empty()) {
        m_lastError = "no model path configured";
        return false;
    }
    if (!fileExists(config.modelPath)) {
        m_lastError = "model file not found: " + config.modelPath;
        return false;
    }

    // Hungarian is a hard requirement, so refuse the English-only models rather
    // than let them silently emit garbage for "hu".
    const std::string &mp = config.modelPath;
    const bool looksEnglishOnly =
        mp.size() > 7 && mp.find(".en.") != std::string::npos;
    const bool enOnlySuffix =
        mp.size() > 7 && mp.compare(mp.size() - 7, 7, ".en.bin") == 0;
    if ((looksEnglishOnly || enOnlySuffix)
        && config.language != "en" && !config.language.empty()) {
        m_lastError = "model '" + mp + "' is English-only but language is '"
                    + config.language + "'; use a multilingual model";
        return false;
    }

    whisper_log_set(silentLog, nullptr);

    whisper_context_params cparams = whisper_context_default_params();
    cparams.use_gpu = config.useGpu;

    m_ctx = whisper_init_from_file_with_params(config.modelPath.c_str(), cparams);
    if (!m_ctx) {
        m_lastError = "whisper failed to initialize from " + config.modelPath
                    + " (corrupt or wrong-format model?)";
        return false;
    }

    if (!whisper_is_multilingual(m_ctx)
        && !config.language.empty() && config.language != "en") {
        m_lastError = "loaded model is not multilingual; cannot do language '"
                    + config.language + "'";
        whisper_free(m_ctx);
        m_ctx = nullptr;
        return false;
    }

    m_config = config;
    if (m_config.language.empty()) m_config.language = "en";
    m_detail->languageBuffer = m_config.language;
    return true;
}

void SpeechRecognizer::setLanguage(const std::string &lang)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    m_config.language = lang.empty() ? std::string("en") : lang;
    m_detail->languageBuffer = m_config.language;
}

void SpeechRecognizer::setObservationCallback(ObservationCallback cb)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    m_callback = std::move(cb);
}

std::string SpeechRecognizer::lastTranscript() const
{
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_lastTranscript;
}

// ===========================================================================
// The one place whisper is actually called
// ===========================================================================

void SpeechRecognizer::transcribeWindow(const std::vector<float> &window,
                                        double windowStartSec,
                                        bool isVolatile)
{
    if (!m_ctx || window.empty()) return;

    // Cheap energy VAD. whisper reliably hallucinates stock phrases on silence
    // ("Thank you.", "Köszönöm a figyelmet!"), and a hallucinated observation is
    // strictly worse than none: it would move the aligner's posterior on
    // evidence that does not exist. Fail safe, emit nothing.
    if (rms(window.data(), window.size()) < m_config.silenceRms) {
        m_silentHops.fetch_add(1, std::memory_order_relaxed);
        return;
    }

    std::string language;
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_detail->languageBuffer = m_config.language;
        language = m_detail->languageBuffer;
    }

    whisper_full_params wp =
        whisper_full_default_params(WHISPER_SAMPLING_GREEDY);

    // --- Parameter rationale -------------------------------------------------
    // GREEDY, best_of = 1:
    //   Beam search costs several times more per window for accuracy gains that
    //   do not matter here — we align against a KNOWN script, so the aligner
    //   recovers from token-level errors that beam search would fix. Latency is
    //   the scarce resource; a 500 ms hop budget does not survive beam search.
    wp.greedy.best_of = 1;

    // no_context = true is REQUIRED, not an optimization.
    //   Each window overlaps the previous one by windowSec - hopSec. If whisper
    //   were primed with the previous window's tokens it would see the same
    //   audio twice, once as prompt and once as audio, and start looping or
    //   "completing" the prompt instead of transcribing. Independent windows are
    //   the whole premise of "each result is a fresh observation of the recent
    //   past".
    wp.no_context = true;

    // single_segment = true:
    //   We want one text blob per window, not whisper's internal segmentation.
    //   The aligner tokenizes the blob itself, and segment boundaries inside a
    //   5 s window carry no information we use.
    wp.single_segment = true;

    // translate = false: we want Hungarian OUT for Hungarian IN. Translation
    // would silently turn the whole feature into English-only.
    wp.translate = false;

    wp.language = language.c_str();
    wp.detect_language = false;  // explicit language; detection flips on short
                                 // windows and would break mid-sentence.

    // Suppress non-speech tokens and blank outputs. Directly reduces the
    // "[BLANK_AUDIO]" / "(music)" class of junk observation.
    wp.suppress_blank = true;
    wp.suppress_nst = true;

    // Timestamps: we do NOT need whisper's token timestamps — the design doc
    // says they are less trustworthy than Apple's and the control loop must not
    // depend on them. Our audioTimeSec comes from sample counting, which is
    // exact. Turning them off saves the DTW/alignment work.
    wp.token_timestamps = false;
    wp.no_timestamps = true;

    // Silence all of whisper's own printing; the callback is our only output.
    wp.print_progress = false;
    wp.print_realtime = false;
    wp.print_timestamps = false;
    wp.print_special = false;

    wp.n_threads = resolveThreads(m_config.nThreads);

    // temperature 0 with no fallback: a deterministic single pass. The
    // temperature-increase fallback path re-decodes the window up to 5 more
    // times when entropy is high, which blows the hop budget for output we
    // would rather simply skip.
    wp.temperature = 0.0f;
    wp.temperature_inc = 0.0f;

    // Abort promptly if stop() lands mid-inference — this is what makes
    // "no crash on stop-during-inference" also mean "no multi-second hang".
    wp.abort_callback = [](void *user) -> bool {
        auto *self = static_cast<SpeechRecognizer *>(user);
        return self->m_stopRequested.load(std::memory_order_acquire);
    };
    wp.abort_callback_user_data = this;

    const int rc = whisper_full(m_ctx, wp,
                                window.data(),
                                static_cast<int>(window.size()));
    if (rc != 0) return;  // aborted or failed: emit nothing. Fail safe.
    if (m_stopRequested.load(std::memory_order_acquire)) return;

    std::string text;
    const int n = whisper_full_n_segments(m_ctx);
    for (int i = 0; i < n; ++i) {
        const char *seg = whisper_full_get_segment_text(m_ctx, i);
        if (seg) text += seg;
    }

    Observation obs;
    // THE CONTRACT: start of the audio window, in audio time. Not wall clock,
    // not window end. See the header.
    obs.audioTimeSec = windowStartSec;
    obs.isVolatile = isVolatile;

    for (const std::string &t : tokenizeToStrings(text)) {
        if (isNoiseMarker(t)) continue;
        obs.tokens.push_back(t);
    }

    {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_lastTranscript = text;
    }

    if (obs.tokens.empty()) return;  // nothing usable; do not emit an empty obs

    if (isDegenerateRepetition(obs.tokens)) {
        // Fail safe: a hallucinated loop is worse than no observation at all.
        m_degenerateHops.fetch_add(1, std::memory_order_relaxed);
        return;
    }

    ObservationCallback cb;
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        cb = m_callback;
    }
    m_emitted.fetch_add(1, std::memory_order_relaxed);
    if (cb) cb(obs);
}

// ===========================================================================
// Live path
// ===========================================================================

bool SpeechRecognizer::startLive(AudioCapture &capture)
{
    m_lastError.clear();
    if (!m_ctx) {
        m_lastError = "no model loaded";
        return false;
    }
    if (m_running.load(std::memory_order_acquire)) return true;

    if (!capture.isRunning() && !capture.start()) {
        m_lastError = "audio capture failed: " + capture.lastError();
        return false;
    }

    m_stopRequested.store(false, std::memory_order_release);
    m_running.store(true, std::memory_order_release);
    m_thread = std::thread(&SpeechRecognizer::liveLoop, this, &capture);
    return true;
}

void SpeechRecognizer::liveLoop(AudioCapture *capture)
{
    const size_t windowSamples = static_cast<size_t>(
        m_config.windowSec * kWhisperSampleRate);
    const size_t hopSamples = static_cast<size_t>(
        m_config.hopSec * kWhisperSampleRate);
    const size_t minSamples = static_cast<size_t>(
        m_config.minWindowSec * kWhisperSampleRate);

    AudioRingBuffer &ring = capture->ring();

    // Rolling window owned by this thread. The ring is drained one hop at a
    // time; the window keeps the last `windowSamples`.
    std::vector<float> window;
    window.reserve(windowSamples + hopSamples);
    std::vector<float> hopBuf(hopSamples);

    // Audio time of window[0], in seconds. Derived purely from sample counts so
    // it is exact and independent of scheduling jitter. Dropped samples are
    // added in so time does not silently compress on overflow.
    uint64_t consumedSamples = 0;
    uint64_t droppedAtLastCheck = 0;

    while (!m_stopRequested.load(std::memory_order_acquire)) {
        if (ring.available() < hopSamples) {
            // Idle wait. Sleeping a fraction of a hop keeps latency bounded
            // without spinning a core.
            std::this_thread::sleep_for(std::chrono::milliseconds(
                std::max<long long>(5,
                    static_cast<long long>(m_config.hopSec * 1000.0 / 8.0))));
            continue;
        }

        // Account for anything the audio thread had to drop, so audio time
        // stays honest about how much real time has elapsed.
        const uint64_t nowDropped = ring.droppedSamples();
        if (nowDropped > droppedAtLastCheck) {
            consumedSamples += (nowDropped - droppedAtLastCheck);
            droppedAtLastCheck = nowDropped;
            // A drop means the window is no longer contiguous. Discard it
            // rather than transcribe a spliced buffer.
            window.clear();
        }

        // If we have fallen far behind (consumer starved for a while), throw
        // away everything but the most recent window. Falling behind is worse
        // than dropping — the brief is explicit.
        const size_t avail = ring.available();
        if (avail > windowSamples + hopSamples) {
            const size_t excess = avail - (windowSamples + hopSamples);
            consumedSamples += ring.discard(excess);
            m_skippedHops.fetch_add(1, std::memory_order_relaxed);
            window.clear();
        }

        const size_t got = ring.read(hopBuf.data(), hopSamples);
        if (got == 0) continue;

        window.insert(window.end(), hopBuf.begin(),
                      hopBuf.begin() + static_cast<long>(got));
        if (window.size() > windowSamples) {
            window.erase(window.begin(),
                         window.begin()
                             + static_cast<long>(window.size() - windowSamples));
        }

        // audio time of window[0] = (total consumed) - (window length)
        consumedSamples += got;
        const double windowStartSec =
            (static_cast<double>(consumedSamples) - static_cast<double>(window.size()))
            / static_cast<double>(kWhisperSampleRate);

        if (window.size() < minSamples) continue;

        // Live windows are ALWAYS volatile: more audio can still arrive that
        // changes how whisper reads this window's tail.
        transcribeWindow(window, std::max(0.0, windowStartSec), /*isVolatile=*/true);

        // Note on "skip rather than queue": we never queue. Inference happens
        // inline on this thread, so while it runs the ring simply accumulates.
        // The `avail > window + hop` check above is what turns that accumulation
        // into a DROP on the next iteration instead of a growing backlog.
    }

    m_running.store(false, std::memory_order_release);
}

// ===========================================================================
// Offline path — identical windowing, so the replay harness is faithful
// ===========================================================================

bool SpeechRecognizer::runFile(const std::string &wavPath)
{
    m_lastError.clear();
    std::vector<float> samples;
    std::string err;
    if (!loadWavAsMono16k(wavPath, samples, err)) {
        m_lastError = err;
        return false;
    }
    return runSamples(samples);
}

bool SpeechRecognizer::runSamples(const std::vector<float> &samples16k)
{
    if (!m_ctx) {
        m_lastError = "no model loaded";
        return false;
    }
    m_stopRequested.store(false, std::memory_order_release);

    const size_t windowSamples = static_cast<size_t>(
        m_config.windowSec * kWhisperSampleRate);
    const size_t hopSamples = std::max<size_t>(1, static_cast<size_t>(
        m_config.hopSec * kWhisperSampleRate));
    const size_t minSamples = static_cast<size_t>(
        m_config.minWindowSec * kWhisperSampleRate);

    const size_t total = samples16k.size();
    size_t consumed = 0;

    while (consumed < total && !m_stopRequested.load(std::memory_order_acquire)) {
        consumed = std::min(total, consumed + hopSamples);

        const size_t windowLen = std::min(consumed, windowSamples);
        const size_t windowStart = consumed - windowLen;

        if (windowLen < minSamples && consumed < total) continue;
        if (windowLen == 0) continue;

        std::vector<float> window(samples16k.begin() + static_cast<long>(windowStart),
                                  samples16k.begin() + static_cast<long>(consumed));

        // Exactly the same audio-time definition as the live path.
        const double windowStartSec =
            static_cast<double>(windowStart) / static_cast<double>(kWhisperSampleRate);

        // The last window of a file is final: no more audio can ever revise it.
        const bool finalWindow = (consumed >= total);
        transcribeWindow(window, windowStartSec, /*isVolatile=*/!finalWindow);
    }
    return true;
}

// ===========================================================================

void SpeechRecognizer::stop()
{
    m_stopRequested.store(true, std::memory_order_release);
    if (m_thread.joinable()) {
        // whisper_full's abort_callback sees m_stopRequested and unwinds, so
        // this join is bounded by roughly one encoder step, not a whole window.
        m_thread.join();
    }
    m_running.store(false, std::memory_order_release);
    m_stopRequested.store(false, std::memory_order_release);
}

}  // namespace voicetracking
}  // namespace qprompt
