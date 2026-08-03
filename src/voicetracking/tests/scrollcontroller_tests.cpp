// SPDX-FileCopyrightText: QPrompt voice tracking
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Self-contained test binary for the scroll control law. No test framework:
// a small CHECK macro, a pass/fail tally, and a non-zero exit on failure.
// Matches the style of aligner_tests.cpp.
//
// Build & run:
//   clang++ -std=c++17 -O2 -Wall -Wextra \
//       src/voicetracking/scrollcontroller.cpp \
//       src/voicetracking/tests/scrollcontroller_tests.cpp \
//       -o /tmp/scrollcontroller_tests && /tmp/scrollcontroller_tests

#include "../scrollcontroller.h"

#include <cmath>
#include <cstdio>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

using namespace qprompt::voicetracking;

// ---------------------------------------------------------------------------
// Tiny harness
// ---------------------------------------------------------------------------

namespace {

int g_passed = 0;
int g_failed = 0;
std::string g_currentTest;

void reportFailure(const char *file, int line, const std::string &message)
{
    ++g_failed;
    std::printf("  FAIL [%s] %s:%d\n       %s\n",
                g_currentTest.c_str(), file, line, message.c_str());
}

void reportPass() { ++g_passed; }

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
            reportFailure(__FILE__, __LINE__, (msg));                      \
        } else {                                                           \
            reportPass();                                                  \
        }                                                                  \
    } while (0)

void beginTest(const std::string &name)
{
    g_currentTest = name;
    std::printf("running: %s\n", name.c_str());
}

// A controller set up the way the adapter will use it: 30px lines, and a
// user-configured constant speed of 60 px/sec to fall back to.
ScrollController makeController()
{
    ScrollController c;
    c.setLineHeight(30.0);
    c.setFallbackVelocity(60.0);
    return c;
}

// Runs the controller to steady state at a fixed error, so tests can assert on
// settled behaviour rather than on the acceleration-limited first step.
ScrollCommand settle(ScrollController &c,
                     bool confident,
                     double targetPixel,
                     double currentPixel,
                     double speechPxPerSec,
                     double startSec,
                     int steps = 60)
{
    ScrollCommand cmd;
    double t = startSec;
    for (int i = 0; i < steps; ++i) {
        t += 0.1;
        cmd = c.update(confident, targetPixel, currentPixel, speechPxPerSec, t);
    }
    return cmd;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

// On-pace reading: the prompter should simply move at the speech rate.
void testTracksSpeechRateWhenAligned()
{
    beginTest("controller: matches speech rate when perfectly aligned");

    ScrollController c = makeController();
    // Target already includes where the reader is; no error beyond the lead.
    const double speech = 80.0;
    const double current = 1000.0;
    const double target = current;

    const ScrollCommand cmd = settle(c, true, target, current, speech, 0.0);

    CHECK_MSG(!cmd.coasting, "reports tracking, not coasting");
    CHECK_MSG(!cmd.seek, "never seeks during routine tracking");
    // With zero positional error the command is the speech rate plus the
    // lead-driven correction, which is bounded by the velocity clamp.
    CHECK_MSG(cmd.velocityPxPerSec > 0.0, "moves forward");
    CHECK_MSG(cmd.velocityPxPerSec <= speech * 1.9 + 1e-6,
              "respects the upper velocity clamp");
}

// Reader ahead of the text: speed up, but never beyond the clamp.
void testSpeedsUpWhenBehind()
{
    beginTest("controller: speeds up when the prompter is behind the reader");

    ScrollController c = makeController();
    const double speech = 80.0;
    const double current = 1000.0;
    const double target = 1300.0;  // reader is well ahead

    const ScrollCommand cmd = settle(c, true, target, current, speech, 0.0);

    CHECK_MSG(cmd.velocityPxPerSec > speech,
              "commands more than the base speech rate");
    CHECK_MSG(cmd.velocityPxPerSec <= speech * 1.9 + 1e-6,
              "clamped so the text stays readable while catching up");
    CHECK_MSG(!cmd.seek, "catches up by scrolling, never by jumping");
}

// Reader behind the text: slow down, and never scroll backwards.
void testSlowsDownWhenAhead()
{
    beginTest("controller: slows down when the prompter is ahead of the reader");

    ScrollController c = makeController();
    const double speech = 80.0;
    const double current = 1400.0;
    const double target = 1000.0;  // prompter overshot

    const ScrollCommand cmd = settle(c, true, target, current, speech, 0.0);

    CHECK_MSG(cmd.velocityPxPerSec < speech,
              "commands less than the base speech rate");
    CHECK_MSG(cmd.velocityPxPerSec >= 0.0,
              "never commands a negative velocity: rewinding under the reader "
              "is far more disorienting than waiting for them to catch up");
}

// The deadband exists so small errors do not produce visible fidget.
void testDeadbandSuppressesTinyCorrections()
{
    beginTest("controller: ignores errors smaller than the deadband");

    ScrollController c = makeController();
    const double speech = 0.0;   // isolate the correction term
    c.setFallbackVelocity(0.0);
    const double current = 1000.0;
    // Deadband is 0.35 * 30px = 10.5px. 6px must be ignored.
    const ScrollCommand cmd = settle(c, true, current + 6.0, current, speech, 0.0);

    CHECK_MSG(std::fabs(cmd.velocityPxPerSec) < 1e-6,
              "a sub-deadband error produces no motion (got "
                  + std::to_string(cmd.velocityPxPerSec) + ")");
}

// The core safety property: an unconfident aligner must never cause a jump.
void testUnconfidentNeverSeeks()
{
    beginTest("controller: unconfident alignment never seeks or jumps");

    ScrollController c = makeController();

    // Establish tracking first.
    settle(c, true, 1000.0, 1000.0, 80.0, 0.0);

    // Now go unconfident while claiming a wildly distant target. The controller
    // must ignore the target entirely.
    double t = 10.0;
    for (int i = 0; i < 40; ++i) {
        t += 0.1;
        const ScrollCommand cmd =
            c.update(false, 99999.0, 1000.0, 80.0, t);
        CHECK_MSG(!cmd.seek, "no seek while unconfident");
        CHECK_MSG(cmd.coasting, "reports coasting while unconfident");
        CHECK_MSG(cmd.velocityPxPerSec >= 0.0
                      && cmd.velocityPxPerSec < 400.0,
                  "velocity stays sane despite an absurd target");
    }
}

// Coasting should hold, then blend to the user's speed -- not stop dead.
void testCoastBlendsToFallback()
{
    beginTest("controller: coasting blends to the configured constant speed");

    ScrollController c = makeController();
    c.setFallbackVelocity(60.0);

    // Track at a rate clearly different from the fallback so the blend is
    // observable.
    settle(c, true, 1000.0, 1000.0, 120.0, 0.0);

    // Immediately after losing confidence, hold near the last trusted rate.
    ScrollCommand held = c.update(false, 0.0, 1000.0, 0.0, 10.5);
    CHECK_MSG(held.velocityPxPerSec > 60.0,
              "holds the last trusted velocity briefly rather than dropping "
              "straight to the fallback");

    // Well past the blend window, it should have converged on the fallback.
    double t = 11.0;
    ScrollCommand blended;
    for (int i = 0; i < 80; ++i) {
        t += 0.1;
        blended = c.update(false, 0.0, 1000.0, 0.0, t);
    }
    CHECK_MSG(std::fabs(blended.velocityPxPerSec - 60.0) < 5.0,
              "converges on the fallback velocity (got "
                  + std::to_string(blended.velocityPxPerSec) + ")");
    CHECK_MSG(blended.velocityPxPerSec > 0.0,
              "keeps moving while coasting rather than freezing");
}

// Acceleration limiting: no instantaneous velocity steps.
void testAccelerationIsLimited()
{
    beginTest("controller: velocity changes are acceleration limited");

    ScrollController c = makeController();
    const double speech = 80.0;

    // First update with a huge error; the response must be gradual.
    const ScrollCommand first =
        c.update(true, 5000.0, 1000.0, speech, 0.1);

    CHECK_MSG(first.velocityPxPerSec < speech * 1.9,
              "does not reach the clamp in a single step (got "
                  + std::to_string(first.velocityPxPerSec) + ")");

    // And successive steps must each be bounded.
    double previous = first.velocityPxPerSec;
    double t = 0.1;
    for (int i = 0; i < 5; ++i) {
        t += 0.1;
        const ScrollCommand cmd = c.update(true, 5000.0, 1000.0, speech, t);
        const double step = std::fabs(cmd.velocityPxPerSec - previous);
        CHECK_MSG(step <= speech * 1.0 * 0.1 + 1e-6,
                  "per-step velocity change respects the acceleration limit "
                  "(step " + std::to_string(step) + ")");
        previous = cmd.velocityPxPerSec;
    }
}

// Re-anchoring is the one case where an immediate seek is correct.
void testReanchorSeeks()
{
    beginTest("controller: explicit re-anchor seeks immediately");

    ScrollController c = makeController();
    settle(c, true, 1000.0, 1000.0, 80.0, 0.0);

    const ScrollCommand cmd = c.reanchor(4321.0, 20.0);

    CHECK_MSG(cmd.seek, "re-anchor requests a seek");
    CHECK_MSG(std::fabs(cmd.seekToPixel - 4321.0) < 1e-9,
              "seeks to exactly the requested position");
    CHECK_MSG(cmd.coasting,
              "treats the position as user-supplied until alignment "
              "re-establishes itself");
}

// A reset must not leave stale velocity or timing behind.
void testResetClearsState()
{
    beginTest("controller: reset clears accumulated state");

    ScrollController c = makeController();
    settle(c, true, 3000.0, 1000.0, 120.0, 0.0);
    c.reset();

    // After reset, an unconfident update should behave like a cold start:
    // fall back rather than resume the old trusted velocity.
    double t = 0.0;
    ScrollCommand cmd;
    for (int i = 0; i < 60; ++i) {
        t += 0.1;
        cmd = c.update(false, 0.0, 0.0, 0.0, t);
    }
    CHECK_MSG(std::fabs(cmd.velocityPxPerSec - 60.0) < 5.0,
              "cold-starts to the fallback velocity after reset (got "
                  + std::to_string(cmd.velocityPxPerSec) + ")");
}

// Garbage in must not become garbage out: NaN/inf must never reach the view,
// where they would corrupt the scroll animation rather than just look wrong.
void testNonFiniteInputsAreContained()
{
    beginTest("controller: non-finite inputs never reach the output");

    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double inf = std::numeric_limits<double>::infinity();

    ScrollController c = makeController();

    const ScrollCommand a = c.update(true, nan, 1000.0, 80.0, 1.0);
    CHECK_MSG(std::isfinite(a.velocityPxPerSec), "NaN target contained");

    const ScrollCommand b = c.update(true, 1000.0, inf, 80.0, 1.1);
    CHECK_MSG(std::isfinite(b.velocityPxPerSec), "inf position contained");

    const ScrollCommand d = c.update(true, 1000.0, 1000.0, nan, 1.2);
    CHECK_MSG(std::isfinite(d.velocityPxPerSec), "NaN speech rate contained");

    const ScrollCommand e = c.update(true, 1000.0, 1000.0, 80.0, nan);
    CHECK_MSG(std::isfinite(e.velocityPxPerSec), "NaN clock contained");

    const ScrollCommand f = c.reanchor(nan, 2.0);
    CHECK_MSG(std::isfinite(f.seekToPixel), "NaN re-anchor target contained");
}

// A clock that goes backwards (or repeats) must not produce a wild step.
void testNonMonotonicClockIsSafe()
{
    beginTest("controller: a backwards clock does not destabilise the loop");

    ScrollController c = makeController();
    settle(c, true, 1000.0, 1000.0, 80.0, 0.0);

    const ScrollCommand before = c.update(true, 1200.0, 1000.0, 80.0, 10.0);
    // Same timestamp, then an earlier one.
    const ScrollCommand same = c.update(true, 1200.0, 1000.0, 80.0, 10.0);
    const ScrollCommand back = c.update(true, 1200.0, 1000.0, 80.0, 5.0);

    CHECK(std::isfinite(same.velocityPxPerSec));
    CHECK(std::isfinite(back.velocityPxPerSec));
    CHECK_MSG(std::fabs(same.velocityPxPerSec - before.velocityPxPerSec) < 200.0,
              "repeated timestamp does not cause a large jump");
    CHECK_MSG(std::fabs(back.velocityPxPerSec - same.velocityPxPerSec) < 200.0,
              "backwards timestamp does not cause a large jump");
}

// A speaker who stops talking entirely must not freeze the prompter.
void testSilenceDoesNotFreeze()
{
    beginTest("controller: silence falls back rather than stopping dead");

    ScrollController c = makeController();
    c.setFallbackVelocity(60.0);
    settle(c, true, 1000.0, 1000.0, 90.0, 0.0);

    // Speech rate collapses to zero and confidence is lost.
    double t = 10.0;
    ScrollCommand cmd;
    for (int i = 0; i < 80; ++i) {
        t += 0.1;
        cmd = c.update(false, 0.0, 1000.0, 0.0, t);
    }

    CHECK_MSG(cmd.velocityPxPerSec > 1.0,
              "keeps scrolling at the fallback rate instead of freezing (got "
                  + std::to_string(cmd.velocityPxPerSec) + ")");
}

} // namespace

int main()
{
    testTracksSpeechRateWhenAligned();
    testSpeedsUpWhenBehind();
    testSlowsDownWhenAhead();
    testDeadbandSuppressesTinyCorrections();
    testUnconfidentNeverSeeks();
    testCoastBlendsToFallback();
    testAccelerationIsLimited();
    testReanchorSeeks();
    testResetClearsState();
    testNonFiniteInputsAreContained();
    testNonMonotonicClockIsSafe();
    testSilenceDoesNotFreeze();

    std::printf("\n=== summary ===\npassed: %d\nfailed: %d\n\n",
                g_passed, g_failed);
    if (g_failed == 0) {
        std::printf("RESULT: ALL TESTS PASSED\n");
        return 0;
    }
    std::printf("RESULT: FAILURES\n");
    return 1;
}
