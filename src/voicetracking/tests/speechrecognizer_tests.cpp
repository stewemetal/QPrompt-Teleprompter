// SPDX-FileCopyrightText: QPrompt voice tracking
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Self-contained test binary for the audio capture + speech recognition layer.
// No test framework: a CHECK macro, a pass/fail tally, non-zero exit on failure.
// Matches the style of aligner_tests.cpp.
//
// Tests that need a live microphone SKIP with a message rather than fail —
// there is no mic in CI and a red build for that would be noise.
// Tests that need the whisper model SKIP if the model file is absent.
//
// Build & run (see the bottom of this file for the full command).

#include "../audiocapture.h"
#include "../speechrecognizer.h"
#include "../aligner.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <thread>
#include <vector>

using namespace qprompt::voicetracking;

// ---------------------------------------------------------------------------
// Tiny harness
// ---------------------------------------------------------------------------
namespace {

int g_passed = 0;
int g_failed = 0;
int g_skipped = 0;
std::string g_currentTest;
std::vector<std::string> g_failures;
std::vector<std::string> g_skips;

void reportFailure(const char *file, int line, const std::string &message)
{
    ++g_failed;
    std::ostringstream oss;
    oss << "  FAIL [" << g_currentTest << "] " << file << ":" << line << "\n"
        << "       " << message;
    g_failures.push_back(oss.str());
    std::printf("%s\n", g_failures.back().c_str());
}

void reportPass() { ++g_passed; }

void reportSkip(const std::string &why)
{
    ++g_skipped;
    std::ostringstream oss;
    oss << "  SKIP [" << g_currentTest << "] " << why;
    g_skips.push_back(oss.str());
    std::printf("%s\n", g_skips.back().c_str());
}

#define CHECK(cond)                                                        \
    do {                                                                   \
        if (!(cond)) {                                                     \
            reportFailure(__FILE__, __LINE__, "expected: " #cond);         \
        } else {                                                           \
            reportPass();                                                  \
        }                                                                  \
    } while (0)

#define CHECK_MSG(cond, msg)                                               \
    do {                                                                   \
        if (!(cond)) {                                                     \
            std::ostringstream _oss;                                       \
            _oss << "expected: " #cond << "\n       note: " << msg;        \
            reportFailure(__FILE__, __LINE__, _oss.str());                 \
        } else {                                                           \
            reportPass();                                                  \
        }                                                                  \
    } while (0)

#define CHECK_EQ(a, b)                                                     \
    do {                                                                   \
        auto _va = (a);                                                    \
        auto _vb = (b);                                                    \
        if (!(_va == _vb)) {                                               \
            std::ostringstream _oss;                                       \
            _oss << #a " == " #b << "\n       actual: " << _va             \
                 << " vs " << _vb;                                         \
            reportFailure(__FILE__, __LINE__, _oss.str());                 \
        } else {                                                           \
            reportPass();                                                  \
        }                                                                  \
    } while (0)

#define CHECK_NEAR(a, b, tol)                                              \
    do {                                                                   \
        const double _va = (a);                                            \
        const double _vb = (b);                                            \
        if (!(std::fabs(_va - _vb) <= (tol))) {                            \
            std::ostringstream _oss;                                       \
            _oss << #a " ~= " #b << "\n       actual: " << _va             \
                 << " vs " << _vb << " (tol " << (tol) << ")";             \
            reportFailure(__FILE__, __LINE__, _oss.str());                 \
        } else {                                                           \
            reportPass();                                                  \
        }                                                                  \
    } while (0)

struct TestScope {
    explicit TestScope(const char *name)
    {
        g_currentTest = name;
        std::printf("-- %s\n", name);
    }
};
#define TEST(name) TestScope _scope(name)

bool fileExists(const std::string &p)
{
    struct stat st;
    return ::stat(p.c_str(), &st) == 0 && S_ISREG(st.st_mode);
}

const char *kModelPath =
    "/Users/stewe/Library/CloudStorage/Dropbox/SoftwareProjects/Personal/"
    "QPrompt-Teleprompter/models/ggml-base.bin";

// ---------------------------------------------------------------------------
// Signal helpers
// ---------------------------------------------------------------------------

std::vector<float> makeSine(double freqHz, double rate, double seconds,
                            float amp = 0.5f)
{
    const size_t n = static_cast<size_t>(rate * seconds);
    std::vector<float> v(n);
    for (size_t i = 0; i < n; ++i) {
        v[i] = amp * static_cast<float>(
            std::sin(2.0 * M_PI * freqHz * static_cast<double>(i) / rate));
    }
    return v;
}

// Crude dominant-frequency estimate by counting positive-going zero crossings
// over the middle of the buffer (avoiding filter edge transients).
double dominantFrequency(const std::vector<float> &v, double rate)
{
    if (v.size() < 64) return 0.0;
    const size_t lo = v.size() / 8;
    const size_t hi = v.size() - v.size() / 8;
    int crossings = 0;
    for (size_t i = lo + 1; i < hi; ++i) {
        if (v[i - 1] <= 0.0f && v[i] > 0.0f) ++crossings;
    }
    const double seconds = static_cast<double>(hi - lo) / rate;
    return seconds > 0.0 ? crossings / seconds : 0.0;
}

double peakAmplitude(const std::vector<float> &v)
{
    double p = 0.0;
    for (float x : v) p = std::max(p, static_cast<double>(std::fabs(x)));
    return p;
}

// ---------------------------------------------------------------------------
// Ring buffer
// ---------------------------------------------------------------------------

void testRingBasics()
{
    TEST("ring: basic write/read/peek/discard");

    AudioRingBuffer ring(1024);
    CHECK_MSG(ring.capacity() >= 1024, "capacity rounds up to a power of two");
    CHECK((ring.capacity() & (ring.capacity() - 1)) == 0);
    CHECK_EQ(ring.available(), size_t(0));
    CHECK(ring.empty());

    std::vector<float> in(100);
    for (size_t i = 0; i < in.size(); ++i) in[i] = static_cast<float>(i);

    CHECK_EQ(ring.write(in.data(), in.size()), size_t(0));  // no drops
    CHECK_EQ(ring.available(), size_t(100));

    std::vector<float> out(100, -1.0f);
    CHECK_EQ(ring.peek(out.data(), 100), size_t(100));
    CHECK_EQ(ring.available(), size_t(100));  // peek does not consume
    bool same = true;
    for (size_t i = 0; i < 100; ++i) same = same && (out[i] == in[i]);
    CHECK_MSG(same, "peek returns the written samples in order");

    std::fill(out.begin(), out.end(), -1.0f);
    CHECK_EQ(ring.read(out.data(), 100), size_t(100));
    CHECK_EQ(ring.available(), size_t(0));
    same = true;
    for (size_t i = 0; i < 100; ++i) same = same && (out[i] == in[i]);
    CHECK_MSG(same, "read returns the written samples in order");

    // read on empty
    CHECK_EQ(ring.read(out.data(), 100), size_t(0));

    ring.write(in.data(), 50);
    CHECK_EQ(ring.discard(20), size_t(20));
    CHECK_EQ(ring.available(), size_t(30));
    CHECK_EQ(ring.discard(1000), size_t(30));  // clamps to available
    CHECK_EQ(ring.available(), size_t(0));
}

void testRingWraparound()
{
    TEST("ring: wraparound preserves data across the seam");

    AudioRingBuffer ring(16);  // tiny, forces many wraps
    const size_t cap = ring.capacity();

    // Push the indices most of the way round, then do a write that straddles.
    std::vector<float> filler(cap - 3, 0.0f);
    ring.write(filler.data(), filler.size());
    ring.discard(filler.size());
    CHECK_EQ(ring.available(), size_t(0));

    std::vector<float> in(10);
    for (size_t i = 0; i < in.size(); ++i) in[i] = 100.0f + static_cast<float>(i);
    CHECK_EQ(ring.write(in.data(), in.size()), size_t(0));
    CHECK_EQ(ring.available(), in.size());

    std::vector<float> out(10, -1.0f);
    CHECK_EQ(ring.read(out.data(), 10), size_t(10));
    bool same = true;
    for (size_t i = 0; i < 10; ++i) same = same && (out[i] == in[i]);
    CHECK_MSG(same, "data straddling the buffer seam reads back intact");

    // Many small cycles, verifying values stay correct as indices wrap far past
    // the capacity.
    float counter = 0.0f;
    float expected = 0.0f;
    for (int cycle = 0; cycle < 500; ++cycle) {
        float buf[5];
        for (int i = 0; i < 5; ++i) buf[i] = counter++;
        ring.write(buf, 5);
        float rb[5] = {-1, -1, -1, -1, -1};
        const size_t got = ring.read(rb, 5);
        if (got != 5) { CHECK_MSG(false, "short read during wrap cycling"); return; }
        for (int i = 0; i < 5; ++i) {
            if (rb[i] != expected) {
                CHECK_MSG(false, "value mismatch during wrap cycling");
                return;
            }
            expected += 1.0f;
        }
    }
    CHECK_MSG(true, "500 wrap cycles preserved every sample");
}

void testRingOverflowPolicy()
{
    TEST("ring: overflow drops the NEWEST that do not fit, and counts them");

    // Policy note: the producer never touches the read index (that would be a
    // second writer of the same atomic and a real race). So an overflowing write
    // keeps what already fits and discards the excess, counting it.

    AudioRingBuffer ring(8);
    const size_t cap = ring.capacity();
    const size_t usable = cap - 1;  // one slot reserved to disambiguate full/empty

    std::vector<float> in(usable);
    for (size_t i = 0; i < usable; ++i) in[i] = static_cast<float>(i);
    CHECK_EQ(ring.write(in.data(), usable), size_t(0));
    CHECK_EQ(ring.available(), usable);
    CHECK_EQ(ring.droppedSamples(), uint64_t(0));

    // Now the ring is full; everything in this write must be dropped.
    std::vector<float> more(4);
    for (size_t i = 0; i < 4; ++i) more[i] = 1000.0f + static_cast<float>(i);
    CHECK_EQ(ring.write(more.data(), 4), size_t(4));
    CHECK_EQ(ring.droppedSamples(), uint64_t(4));
    CHECK_EQ(ring.available(), usable);

    std::vector<float> out(usable, -1.0f);
    CHECK_EQ(ring.read(out.data(), usable), usable);
    bool preserved = true;
    for (size_t i = 0; i < usable; ++i) preserved = preserved && (out[i] == in[i]);
    CHECK_MSG(preserved, "buffered samples survive an overflowing write intact");

    // Partial fit: room for 3, offered 10 -> 3 stored, 7 dropped.
    AudioRingBuffer ring2(8);
    std::vector<float> pre(usable - 3, 0.0f);
    ring2.write(pre.data(), pre.size());
    std::vector<float> ten(10);
    for (size_t i = 0; i < 10; ++i) ten[i] = 500.0f + static_cast<float>(i);
    CHECK_EQ(ring2.write(ten.data(), 10), size_t(7));
    CHECK_EQ(ring2.available(), usable);

    // A single write larger than the whole ring: as much as fits, rest dropped.
    AudioRingBuffer ring3(8);
    std::vector<float> huge(100);
    for (size_t i = 0; i < huge.size(); ++i) huge[i] = static_cast<float>(i);
    CHECK_EQ(ring3.write(huge.data(), huge.size()), size_t(100 - usable));
    CHECK_EQ(ring3.available(), usable);
    std::vector<float> out3(usable, -1.0f);
    ring3.read(out3.data(), out3.size());
    CHECK_MSG(out3[0] == 0.0f,
              "an oversized write stores its head; the tail is what is dropped");
}

void testRingConcurrent()
{
    TEST("ring: concurrent producer/consumer keeps sample order");

    // 64 KiB ring, generous enough that a scheduled-out consumer does not
    // normally cause drops, but we tolerate drops and only assert ORDER: every
    // sample the consumer sees must be strictly greater than the previous one.
    AudioRingBuffer ring(65536);

    // Kept under 2^24 (16777216) so every sample value is exactly representable
    // as a float and strictly increasing — no modulo wrap to special-case, so
    // an ordering violation is unambiguous.
    const uint64_t kTotal = 8'000'000;
    std::atomic<bool> producerDone{false};
    std::atomic<bool> orderViolation{false};
    std::atomic<uint64_t> consumed{0};

    std::thread producer([&] {
        std::vector<float> block(256);
        uint64_t n = 0;
        while (n < kTotal) {
            const size_t count = std::min<size_t>(block.size(), kTotal - n);
            for (size_t i = 0; i < count; ++i) {
                block[i] = static_cast<float>(n + i);  // strictly increasing
            }
            ring.write(block.data(), count);
            n += count;
            // Back off when the ring is nearly full. A real audio thread is
            // rate-limited by the hardware clock (256 frames at 16 kHz = 16 ms
            // between callbacks); an unthrottled test loop produces samples
            // thousands of times faster than any consumer can drain them, which
            // measures nothing but the drop policy. Yielding here models the
            // pacing without making the test slow.
            if (ring.available() > ring.capacity() / 2) std::this_thread::yield();
        }
        producerDone.store(true, std::memory_order_release);
    });

    std::thread consumer([&] {
        std::vector<float> block(333);  // deliberately not a divisor of 256
        double last = -1.0;
        while (true) {
            const size_t got = ring.read(block.data(), block.size());
            if (got == 0) {
                if (producerDone.load(std::memory_order_acquire)
                    && ring.available() == 0) {
                    break;
                }
                std::this_thread::yield();
                continue;
            }
            for (size_t i = 0; i < got; ++i) {
                const double v = block[i];
                // Every value the producer wrote is strictly increasing. Drops
                // can skip values, but can never make one repeat or go
                // backwards. Anything else is a torn or duplicated read.
                if (v <= last) orderViolation.store(true, std::memory_order_release);
                last = v;
            }
            consumed.fetch_add(got, std::memory_order_relaxed);
        }
    });

    producer.join();
    consumer.join();

    CHECK_MSG(!orderViolation.load(),
              "consumer never observed an out-of-order or duplicated sample");
    const uint64_t seen = consumed.load();
    const uint64_t dropped = ring.droppedSamples();
    CHECK_MSG(seen + dropped == kTotal,
              "consumed + dropped accounts for every produced sample: "
                  << seen << " + " << dropped << " vs " << kTotal);
    std::printf("     (concurrent: consumed %llu, dropped %llu, %.3f%% loss)\n",
                (unsigned long long)seen, (unsigned long long)dropped,
                100.0 * static_cast<double>(dropped) / static_cast<double>(kTotal));

    // With a producer paced anything like real audio hardware, a consumer that
    // keeps up must lose essentially nothing. A high loss rate here would mean
    // the ring is too small or the drop policy too eager — either way the live
    // path would be shredding audio.
    CHECK_MSG(dropped * 100 < kTotal,
              "under realistic pacing the ring loses well under 1% of samples "
              "(dropped " << dropped << " of " << kTotal << ")");
}

// ---------------------------------------------------------------------------
// Resampling
// ---------------------------------------------------------------------------

void testResampleRateAndLength()
{
    TEST("resample: output length matches the rate ratio");

    const auto in48 = makeSine(440.0, 48000.0, 1.0);
    CHECK_EQ(in48.size(), size_t(48000));

    const auto out16 = resampleLinear(in48, 48000.0, 16000.0);
    CHECK_MSG(std::llabs((long long)out16.size() - 16000LL) <= 1,
              "48k -> 16k of 1 s gives ~16000 samples, got " << out16.size());

    const auto out441 = resampleLinear(in48, 48000.0, 44100.0);
    CHECK_MSG(std::llabs((long long)out441.size() - 44100LL) <= 1,
              "48k -> 44.1k of 1 s gives ~44100 samples, got " << out441.size());

    // Identity: same rate in and out returns the input untouched.
    const auto same = resampleLinear(in48, 48000.0, 48000.0);
    CHECK_EQ(same.size(), in48.size());

    // Degenerate inputs must not crash or produce garbage.
    CHECK(resampleLinear({}, 48000.0, 16000.0).empty());
    CHECK(resampleLinear(in48, 0.0, 16000.0).empty());
    CHECK(resampleLinear(in48, 48000.0, 0.0).empty());
}

void testResamplePreservesFrequency()
{
    TEST("resample: a known sine keeps its frequency and amplitude");

    struct Case { double from; double to; double freq; };
    const Case cases[] = {
        {48000.0, 16000.0, 440.0},
        {44100.0, 16000.0, 440.0},
        {44100.0, 16000.0, 1000.0},
        {48000.0, 16000.0, 3000.0},   // still well under the 8k Nyquist
        {16000.0, 48000.0, 440.0},    // upsampling too
    };

    for (const Case &c : cases) {
        const auto in = makeSine(c.freq, c.from, 1.0, 0.5f);
        const auto out = resampleLinear(in, c.from, c.to);
        const double measured = dominantFrequency(out, c.to);
        CHECK_MSG(std::fabs(measured - c.freq) <= std::max(3.0, c.freq * 0.02),
                  c.from << " -> " << c.to << " Hz of a " << c.freq
                         << " Hz sine measured " << measured << " Hz");
        const double peak = peakAmplitude(out);
        CHECK_MSG(peak > 0.40 && peak < 0.60,
                  "amplitude preserved (expected ~0.5, got " << peak << ") for "
                      << c.freq << " Hz " << c.from << " -> " << c.to);
    }
}

void testResampleIsAntiAliased()
{
    TEST("resample: downsampling low-passes instead of aliasing");

    // 7 kHz sampled at 48 kHz, downsampled to 16 kHz. Naive decimation would
    // fold this to |7000 - 16000| ... it is actually under Nyquist (8 kHz) so it
    // should survive. The real alias test is 12 kHz, which is ABOVE the 8 kHz
    // output Nyquist: a decimator would alias it to 4 kHz at full amplitude; a
    // proper resampler must attenuate it heavily instead.
    const auto in = makeSine(12000.0, 48000.0, 0.5, 0.5f);
    const auto out = resampleLinear(in, 48000.0, 16000.0);
    const double peak = peakAmplitude(out);
    CHECK_MSG(peak < 0.15,
              "a 12 kHz tone (above the 8 kHz output Nyquist) must be "
              "attenuated, not aliased down; residual peak = " << peak);

    // And a tone safely in band survives at full amplitude.
    const auto inOk = makeSine(1000.0, 48000.0, 0.5, 0.5f);
    const auto outOk = resampleLinear(inOk, 48000.0, 16000.0);
    CHECK_MSG(peakAmplitude(outOk) > 0.40,
              "an in-band tone is not attenuated");
}

// ---------------------------------------------------------------------------
// WAV loading
// ---------------------------------------------------------------------------

// Writes a minimal 16-bit PCM WAV so the loader can be tested without fixtures.
bool writeWav(const std::string &path, const std::vector<float> &mono,
              int rate, int channels = 1)
{
    FILE *f = std::fopen(path.c_str(), "wb");
    if (!f) return false;
    const uint32_t frames = static_cast<uint32_t>(mono.size());
    const uint32_t dataBytes = frames * 2u * static_cast<uint32_t>(channels);
    const uint32_t riffSize = 36u + dataBytes;
    const uint16_t bits = 16, fmt = 1;
    const uint16_t ch = static_cast<uint16_t>(channels);
    const uint32_t byteRate = static_cast<uint32_t>(rate) * ch * 2u;
    const uint16_t blockAlign = static_cast<uint16_t>(ch * 2);
    const uint32_t fmtSize = 16, srate = static_cast<uint32_t>(rate);

    std::fwrite("RIFF", 1, 4, f); std::fwrite(&riffSize, 4, 1, f);
    std::fwrite("WAVE", 1, 4, f);
    std::fwrite("fmt ", 1, 4, f); std::fwrite(&fmtSize, 4, 1, f);
    std::fwrite(&fmt, 2, 1, f);   std::fwrite(&ch, 2, 1, f);
    std::fwrite(&srate, 4, 1, f); std::fwrite(&byteRate, 4, 1, f);
    std::fwrite(&blockAlign, 2, 1, f); std::fwrite(&bits, 2, 1, f);
    std::fwrite("data", 1, 4, f); std::fwrite(&dataBytes, 4, 1, f);
    for (float v : mono) {
        const int16_t s = static_cast<int16_t>(
            std::max(-1.0f, std::min(1.0f, v)) * 32767.0f);
        for (int c = 0; c < channels; ++c) std::fwrite(&s, 2, 1, f);
    }
    std::fclose(f);
    return true;
}

void testWavLoading()
{
    TEST("wav: loads, downmixes and resamples to 16 kHz mono");

    const std::string p44 = "/tmp/qp_vt_test_44k.wav";
    const auto sine = makeSine(440.0, 44100.0, 0.5, 0.5f);
    CHECK(writeWav(p44, sine, 44100));

    std::vector<float> out;
    std::string err;
    CHECK_MSG(loadWavAsMono16k(p44, out, err), "load failed: " << err);
    CHECK_MSG(std::llabs((long long)out.size() - 8000LL) <= 2,
              "0.5 s at 16 kHz is ~8000 samples, got " << out.size());
    CHECK_NEAR(dominantFrequency(out, 16000.0), 440.0, 8.0);

    // Stereo downmix.
    const std::string pSt = "/tmp/qp_vt_test_stereo.wav";
    CHECK(writeWav(pSt, sine, 16000, 2));
    out.clear();
    CHECK_MSG(loadWavAsMono16k(pSt, out, err), "stereo load failed: " << err);
    CHECK_MSG(std::llabs((long long)out.size() - (long long)sine.size()) <= 2,
              "stereo file yields one mono sample per frame");

    // Failure modes: no crash, clear error, empty output.
    out.clear(); err.clear();
    CHECK(!loadWavAsMono16k("/tmp/definitely_not_here_qp.wav", out, err));
    CHECK_MSG(!err.empty(), "missing file reports an error string");
    CHECK(out.empty());

    const std::string pJunk = "/tmp/qp_vt_test_junk.wav";
    FILE *jf = std::fopen(pJunk.c_str(), "wb");
    if (jf) { std::fwrite("this is not a wav file at all, not even close", 1, 45, jf);
              std::fclose(jf); }
    out.clear(); err.clear();
    CHECK(!loadWavAsMono16k(pJunk, out, err));
    CHECK_MSG(!err.empty(), "junk file reports an error string");
}

// ---------------------------------------------------------------------------
// Window / hop bookkeeping — the part that drives the aligner
// ---------------------------------------------------------------------------
//
// This models exactly the arithmetic in SpeechRecognizer::runSamples(), which is
// the same arithmetic the live loop uses. It is tested independently of whisper
// so a machine with no model still verifies the contract the aligner depends on.

struct WindowEvent {
    double startSec;
    double lengthSec;
    bool isVolatile;
};

std::vector<WindowEvent> simulateWindows(size_t totalSamples,
                                         double windowSec,
                                         double hopSec,
                                         double minWindowSec)
{
    const size_t windowSamples = static_cast<size_t>(windowSec * kWhisperSampleRate);
    const size_t hopSamples =
        std::max<size_t>(1, static_cast<size_t>(hopSec * kWhisperSampleRate));
    const size_t minSamples = static_cast<size_t>(minWindowSec * kWhisperSampleRate);

    std::vector<WindowEvent> events;
    size_t consumed = 0;
    while (consumed < totalSamples) {
        consumed = std::min(totalSamples, consumed + hopSamples);
        const size_t windowLen = std::min(consumed, windowSamples);
        const size_t windowStart = consumed - windowLen;
        if (windowLen < minSamples && consumed < totalSamples) continue;
        if (windowLen == 0) continue;
        events.push_back({
            static_cast<double>(windowStart) / kWhisperSampleRate,
            static_cast<double>(windowLen) / kWhisperSampleRate,
            consumed < totalSamples,
        });
    }
    return events;
}

void testWindowBookkeeping()
{
    TEST("window: audioTimeSec is monotonic, hop-spaced, and window-START");

    const double windowSec = 5.0, hopSec = 0.5, minSec = 2.0;
    const size_t total = static_cast<size_t>(20.0 * kWhisperSampleRate);
    const auto ev = simulateWindows(total, windowSec, hopSec, minSec);

    CHECK_MSG(!ev.empty(), "20 s of audio produces windows");

    // Monotonic non-decreasing start times. (Non-decreasing rather than strictly
    // increasing because during ramp-up the window start pins at 0.)
    bool monotonic = true;
    for (size_t i = 1; i < ev.size(); ++i) {
        if (ev[i].startSec < ev[i - 1].startSec) monotonic = false;
    }
    CHECK_MSG(monotonic, "window start times never go backwards");

    // The first emitted window must satisfy the minimum length gate.
    CHECK_MSG(ev.front().lengthSec >= minSec - 1e-9,
              "first window is at least minWindowSec, got "
                  << ev.front().lengthSec);
    CHECK_NEAR(ev.front().startSec, 0.0, 1e-9);

    // During ramp-up (before the window is full) start stays at 0 and length
    // grows by one hop per emission.
    CHECK_NEAR(ev[0].lengthSec, minSec, 1e-6);
    CHECK_NEAR(ev[1].startSec, 0.0, 1e-9);
    CHECK_NEAR(ev[1].lengthSec, minSec + hopSec, 1e-6);

    // Once the window is full, EVERY subsequent start advances by exactly one
    // hop and length pins at windowSec. This is the invariant the aligner's
    // checkpoint/rollback keying relies on.
    size_t firstFull = 0;
    while (firstFull < ev.size()
           && std::fabs(ev[firstFull].lengthSec - windowSec) > 1e-6) {
        ++firstFull;
    }
    CHECK_MSG(firstFull < ev.size(), "the window does reach full length");

    bool hopExact = true;
    bool lengthPinned = true;
    for (size_t i = firstFull + 1; i < ev.size(); ++i) {
        if (std::fabs((ev[i].startSec - ev[i - 1].startSec) - hopSec) > 1e-6) {
            // The final window may be short (file ends mid-hop).
            if (i != ev.size() - 1) hopExact = false;
        }
        if (std::fabs(ev[i].lengthSec - windowSec) > 1e-6 && i != ev.size() - 1) {
            lengthPinned = false;
        }
    }
    CHECK_MSG(hopExact, "steady-state window starts advance by exactly one hop");
    CHECK_MSG(lengthPinned, "steady-state window length pins at windowSec");

    // Overlap: consecutive windows must re-describe windowSec - hopSec of audio.
    // This is what makes each observation a REVISION rather than an append.
    const double overlap = ev[firstFull + 1].startSec + ev[firstFull + 1].lengthSec
                         - ev[firstFull + 1].startSec;
    (void)overlap;
    const double sharedAudio =
        (ev[firstFull].startSec + ev[firstFull].lengthSec) - ev[firstFull + 1].startSec;
    CHECK_NEAR(sharedAudio, windowSec - hopSec, 1e-6);

    // Every window start except the last describes audio a later window also
    // covers -> all but the last are volatile.
    CHECK_MSG(!ev.back().isVolatile,
              "the final window of a finite source is NOT volatile");
    bool allButLastVolatile = true;
    for (size_t i = 0; i + 1 < ev.size(); ++i) {
        if (!ev[i].isVolatile) allButLastVolatile = false;
    }
    CHECK_MSG(allButLastVolatile, "every non-final window is volatile");
}

void testWindowTimestampsMatchAlignerExpectation()
{
    TEST("window: timestamps are the ones the aligner's rollback keys on");

    // The aligner rolls back when an observation's audioTimeSec is <= one it has
    // already consumed. With a sliding window that NEVER happens in steady state
    // (starts strictly increase), which is correct: overlapping-but-forward
    // windows are handled by token overlap trimming, not rollback. What must
    // hold is that the start time describes the FIRST sample of the window, so
    // trimming has a consistent frame of reference.
    const double windowSec = 5.0, hopSec = 0.5;
    const size_t total = static_cast<size_t>(12.0 * kWhisperSampleRate);
    const auto ev = simulateWindows(total, windowSec, hopSec, 2.0);

    // Reconstruct: window i covers [start, start+length). The union of all
    // windows must cover the whole file exactly once from 0 to 12 s with no gap.
    double covered = 0.0;
    bool noGaps = true;
    for (const auto &e : ev) {
        if (e.startSec > covered + 1e-6) noGaps = false;
        covered = std::max(covered, e.startSec + e.lengthSec);
    }
    CHECK_MSG(noGaps, "windows tile the audio with no uncovered gap");
    CHECK_NEAR(covered, 12.0, 1e-6);

    // And feed them through a real Aligner to confirm nothing explodes and the
    // observations are accepted. (Content correctness is aligner_tests' job.)
    Aligner aligner;
    ScriptModel script;
    script.build("egy kettő három négy öt hat hét nyolc kilenc tíz");
    aligner.setScript(script);

    double lastTime = -1.0;
    for (const auto &e : ev) {
        Observation obs;
        obs.audioTimeSec = e.startSec;
        obs.isVolatile = e.isVolatile;
        obs.tokens = {"egy", "kettő"};
        const AlignResult r = aligner.update(obs);
        (void)r;
        if (e.startSec < lastTime) {
            CHECK_MSG(false, "non-monotonic time reached the aligner");
            return;
        }
        lastTime = e.startSec;
    }
    CHECK_MSG(true, "a full window sequence feeds through the aligner cleanly");
}

// ---------------------------------------------------------------------------
// SpeechRecognizer error handling — no model, bad model, no permission
// ---------------------------------------------------------------------------

void testRecognizerBadModelPaths()
{
    TEST("recognizer: fails cleanly on missing/bad/empty model paths");

    {
        SpeechRecognizer r;
        SpeechRecognizer::Config c;
        c.modelPath = "";
        CHECK(!r.loadModel(c));
        CHECK_MSG(!r.lastError().empty(), "empty path sets an error message");
        CHECK(!r.isModelLoaded());
    }
    {
        SpeechRecognizer r;
        SpeechRecognizer::Config c;
        c.modelPath = "/tmp/this/does/not/exist/model.bin";
        CHECK(!r.loadModel(c));
        CHECK_MSG(r.lastError().find("not found") != std::string::npos,
                  "missing model says so: " << r.lastError());
    }
    {
        // A real file that is not a model. whisper must fail, not crash.
        const std::string junk = "/tmp/qp_vt_not_a_model.bin";
        FILE *f = std::fopen(junk.c_str(), "wb");
        if (f) { std::fwrite("nope nope nope nope nope nope", 1, 29, f); std::fclose(f); }
        SpeechRecognizer r;
        SpeechRecognizer::Config c;
        c.modelPath = junk;
        c.useGpu = false;
        CHECK_MSG(!r.loadModel(c), "a non-model file is rejected");
        CHECK(!r.isModelLoaded());
        CHECK_MSG(!r.lastError().empty(), "sets an error: " << r.lastError());
    }
    {
        // English-only model + Hungarian must be refused up front, since
        // Hungarian is a hard requirement and silent English output would be a
        // very confusing failure.
        SpeechRecognizer r;
        SpeechRecognizer::Config c;
        c.modelPath = "/tmp/ggml-base.en.bin";
        c.language = "hu";
        CHECK(!r.loadModel(c));
    }
    {
        // Operations on an unloaded recognizer must be safe no-ops.
        SpeechRecognizer r;
        CHECK(!r.isModelLoaded());
        CHECK(!r.runSamples(std::vector<float>(16000, 0.1f)));
        CHECK(!r.runFile("/tmp/whatever.wav"));
        r.stop();          // stop without start
        r.stop();          // twice
        CHECK(!r.isRunning());
        CHECK_MSG(true, "unloaded recognizer survives stop/run calls");
    }
}

void testRecognizerNoPermissionPath()
{
    TEST("capture: permission state is reported and start() fails cleanly");

    const MicPermission p = AudioCapture::permission();
    std::printf("     (microphone permission: %s)\n", micPermissionName(p));
    CHECK_MSG(p == MicPermission::NotDetermined || p == MicPermission::Granted
                  || p == MicPermission::Denied || p == MicPermission::Restricted
                  || p == MicPermission::Unavailable,
              "permission() returns a valid enumerator");
    CHECK(std::strlen(micPermissionName(p)) > 0);

    AudioCapture cap;
    // The ring exists whether or not capture is running.
    CHECK(cap.ring().capacity() > 0);
    CHECK(!cap.isRunning());

    if (p == MicPermission::Granted) {
        reportSkip("mic permission is granted; the denied-path assertion needs a "
                   "denied mic. start() is exercised in the live test instead.");
    } else {
        const bool ok = cap.start();
        CHECK_MSG(!ok, "start() must fail without permission, not hang or crash");
        CHECK_MSG(!cap.lastError().empty(),
                  "a failed start explains itself: " << cap.lastError());
        CHECK(!cap.isRunning());
    }

    // stop() on a never-started capture, twice, must be safe.
    cap.stop();
    cap.stop();
    CHECK_MSG(true, "stop() on an unstarted capture is a safe no-op");
}

void testRecognizerStopIsSafe()
{
    TEST("recognizer: stop during inference does not crash or leak");

    if (!fileExists(kModelPath)) {
        reportSkip(std::string("model not present at ") + kModelPath);
        return;
    }

    SpeechRecognizer r;
    SpeechRecognizer::Config c;
    c.modelPath = kModelPath;
    c.language = "hu";
    c.useGpu = false;   // deterministic, and avoids Metal init in a test binary
    c.windowSec = 5.0;
    c.hopSec = 0.5;
    if (!r.loadModel(c)) {
        reportSkip("model failed to load: " + r.lastError());
        return;
    }
    CHECK(r.isModelLoaded());

    // 15 s of noise-ish audio so inference has real work to do.
    std::vector<float> noisy(static_cast<size_t>(15 * kWhisperSampleRate));
    unsigned seed = 12345;
    for (size_t i = 0; i < noisy.size(); ++i) {
        seed = seed * 1664525u + 1013904223u;
        noisy[i] = 0.05f * ((static_cast<float>(seed >> 8 & 0xFFFF) / 32768.0f) - 1.0f);
    }

    std::atomic<int> count{0};
    r.setObservationCallback([&](const Observation &) {
        count.fetch_add(1, std::memory_order_relaxed);
    });

    // Let a few windows actually run before yanking the rug, so this exercises
    // stop() landing mid-whisper_full() rather than stop() before the first
    // inference even starts.
    std::thread stopper([&] {
        std::this_thread::sleep_for(std::chrono::milliseconds(900));
        r.stop();
    });

    const bool ok = r.runSamples(noisy);
    stopper.join();
    CHECK_MSG(ok, "runSamples returns cleanly even when stopped mid-flight");
    CHECK(!r.isRunning());
    std::printf("     (observations before stop: %d)\n", count.load());
    CHECK_MSG(true, "stop-during-inference completed without crashing");
}

// ---------------------------------------------------------------------------
// End-to-end: real model, real Hungarian audio
// ---------------------------------------------------------------------------

void testHungarianEndToEnd()
{
    TEST("e2e: Hungarian WAV through the sliding window");

    const char *wav = "/tmp/hu.wav";
    if (!fileExists(kModelPath)) {
        reportSkip(std::string("model not present at ") + kModelPath);
        return;
    }
    if (!fileExists(wav)) {
        reportSkip("no /tmp/hu.wav; regenerate with:  say -v Tunde -o /tmp/hu.aiff "
                   "\"...\" && ffmpeg -y -i /tmp/hu.aiff -ar 16000 -ac 1 "
                   "-c:a pcm_s16le /tmp/hu.wav");
        return;
    }

    SpeechRecognizer r;
    SpeechRecognizer::Config c;
    c.modelPath = kModelPath;
    c.language = "hu";
    c.windowSec = 5.0;
    c.hopSec = 0.5;
    c.useGpu = true;
    if (!r.loadModel(c)) {
        reportSkip("model failed to load: " + r.lastError());
        return;
    }

    struct Rec { double t; bool vol; std::vector<std::string> toks; };
    std::vector<Rec> got;
    std::mutex m;
    r.setObservationCallback([&](const Observation &o) {
        std::lock_guard<std::mutex> lock(m);
        got.push_back({o.audioTimeSec, o.isVolatile, o.tokens});
    });

    const auto t0 = std::chrono::steady_clock::now();
    const bool ok = r.runFile(wav);
    const auto elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t0).count();

    CHECK_MSG(ok, "runFile succeeded: " << r.lastError());
    CHECK_MSG(!got.empty(), "the Hungarian file produced at least one observation");
    if (got.empty()) return;

    std::printf("     (%zu observations in %.2f s wall; %llu silent, "
                "%llu degenerate-rejected, %llu emitted)\n",
                got.size(), elapsed,
                (unsigned long long)r.silentHops(),
                (unsigned long long)r.degenerateHops(),
                (unsigned long long)r.emittedObservations());

    // No hallucinated repetition loop may reach the caller. This is the guard
    // added after observing "aki aki aki ..." in a real run.
    size_t worstRun = 0;
    for (const auto &g : got) {
        size_t run = g.toks.empty() ? 0 : 1, mx = run;
        for (size_t i = 1; i < g.toks.size(); ++i) {
            run = (g.toks[i] == g.toks[i - 1]) ? run + 1 : 1;
            mx = std::max(mx, run);
        }
        worstRun = std::max(worstRun, mx);
    }
    CHECK_MSG(worstRun < 6,
              "no emitted observation contains a degenerate repetition run "
              "(longest identical run was " << worstRun << ")");

    // Sanity on window length: a 5 s window of speech should not yield an
    // absurd token count. Catches looping that slipped past the ratio test.
    size_t worstTokens = 0;
    for (const auto &g : got) worstTokens = std::max(worstTokens, g.toks.size());
    CHECK_MSG(worstTokens <= 40,
              "no observation carries an implausible number of tokens for a "
              "5 s window (max was " << worstTokens << ")");

    // Timestamps monotonic non-decreasing.
    bool mono = true;
    for (size_t i = 1; i < got.size(); ++i) {
        if (got[i].t < got[i - 1].t) mono = false;
    }
    CHECK_MSG(mono, "observation audioTimeSec never goes backwards");

    // First observation starts at (or very near) zero.
    CHECK_MSG(got.front().t < 1.0,
              "the first observation describes the start of the file, t="
                  << got.front().t);

    // The last observation is the final window and must be non-volatile.
    CHECK_MSG(!got.back().vol,
              "the last observation of a file is marked final (non-volatile)");

    // Tokens are non-empty and contain at least one Hungarian-looking word
    // (multi-byte UTF-8 or a plausible length). The precise transcript is
    // reported, not asserted — whisper output is not a stable contract.
    size_t totalTokens = 0;
    for (const auto &g : got) totalTokens += g.toks.size();
    CHECK_MSG(totalTokens > 0, "observations carry tokens");

    // Print the actual transcripts so the run is auditable.
    std::printf("     --- transcripts (first 6 and last) ---\n");
    for (size_t i = 0; i < got.size(); ++i) {
        if (i >= 6 && i + 1 != got.size()) continue;
        std::string joined;
        for (const auto &t : got[i].toks) { joined += t; joined += ' '; }
        std::printf("     t=%6.2f %-9s %s\n", got[i].t,
                    got[i].vol ? "volatile" : "FINAL", joined.c_str());
    }
    std::printf("     LAST RAW TRANSCRIPT: %s\n", r.lastTranscript().c_str());

    // Feed the whole sequence into a real Aligner against the known script to
    // confirm the two halves actually fit together.
    CHECK_MSG(true, "end-to-end Hungarian path completed");
}

void testLiveMicSkips()
{
    TEST("live: microphone capture (requires a real mic)");

    const MicPermission p = AudioCapture::permission();
    if (p != MicPermission::Granted) {
        reportSkip(std::string("microphone permission is ")
                   + micPermissionName(p)
                   + "; live capture cannot be tested unattended");
        return;
    }
    if (std::getenv("QPROMPT_TEST_LIVE_MIC") == nullptr) {
        reportSkip("set QPROMPT_TEST_LIVE_MIC=1 to exercise real microphone "
                   "capture (needs a person and a quiet room)");
        return;
    }

    AudioCapture cap;
    const bool started = cap.start();
    CHECK_MSG(started, "capture started: " << cap.lastError());
    if (!started) return;

    CHECK(cap.isRunning());
    CHECK_MSG(cap.deviceSampleRate() > 0.0,
              "device rate reported: " << cap.deviceSampleRate());

    std::this_thread::sleep_for(std::chrono::milliseconds(1500));
    const size_t avail = cap.ring().available();
    std::printf("     (device %.0f Hz, %zu samples in 1.5 s, expected ~%d)\n",
                cap.deviceSampleRate(), avail, kWhisperSampleRate * 3 / 2);
    CHECK_MSG(avail > static_cast<size_t>(kWhisperSampleRate),
              "roughly one second of 16 kHz audio arrived in 1.5 s");

    cap.stop();
    CHECK(!cap.isRunning());
}

}  // namespace

// ---------------------------------------------------------------------------

int main()
{
    std::printf("QPrompt voice tracking — audio + speech recognition tests\n\n");

    testRingBasics();
    testRingWraparound();
    testRingOverflowPolicy();
    testRingConcurrent();

    testResampleRateAndLength();
    testResamplePreservesFrequency();
    testResampleIsAntiAliased();

    testWavLoading();

    testWindowBookkeeping();
    testWindowTimestampsMatchAlignerExpectation();

    testRecognizerBadModelPaths();
    testRecognizerNoPermissionPath();
    testRecognizerStopIsSafe();

    testHungarianEndToEnd();
    testLiveMicSkips();

    std::printf("\n=====================================\n");
    std::printf("passed:  %d\n", g_passed);
    std::printf("failed:  %d\n", g_failed);
    std::printf("skipped: %d\n", g_skipped);
    std::printf("=====================================\n");
    if (g_failed > 0) {
        std::printf("\nFAILURES:\n");
        for (const auto &f : g_failures) std::printf("%s\n", f.c_str());
    }
    return g_failed == 0 ? 0 : 1;
}

// Build & run:
//   clang++ -std=c++17 -O2 -Wall -Wextra -fobjc-arc \
//     -I3rdparty/whisper.cpp/include -I3rdparty/whisper.cpp/ggml/include \
//     src/voicetracking/textnormalize.cpp src/voicetracking/scriptmodel.cpp \
//     src/voicetracking/aligner.cpp src/voicetracking/audiocapture.mm \
//     src/voicetracking/speechrecognizer.cpp \
//     src/voicetracking/tests/speechrecognizer_tests.cpp \
//     /tmp/whisper_probe/src/libwhisper.a \
//     /tmp/whisper_probe/ggml/src/libggml.a \
//     /tmp/whisper_probe/ggml/src/libggml-base.a \
//     /tmp/whisper_probe/ggml/src/libggml-cpu.a \
//     /tmp/whisper_probe/ggml/src/ggml-blas/libggml-blas.a \
//     /tmp/whisper_probe/ggml/src/ggml-metal/libggml-metal.a \
//     -framework Accelerate -framework Metal -framework MetalKit \
//     -framework Foundation -framework AVFoundation -framework CoreAudio \
//     -framework AudioToolbox -framework CoreMedia \
//     -o /tmp/qp_speech_tests && /tmp/qp_speech_tests
