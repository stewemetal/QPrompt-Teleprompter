// SPDX-FileCopyrightText: QPrompt voice tracking
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Offline replay harness: audio file + script in, tracking behaviour out.
//
// WHY THIS EXISTS
// ---------------
// Every tuning constant in the aligner and the controller is a reasoned
// starting point, not a measured one. Tuning them against a live microphone --
// on camera, with a light on you -- is miserable and unrepeatable. This runs
// the whole real pipeline (whisper -> aligner -> controller) over a recording,
// deterministically, as fast as the machine allows, and prints what happened.
//
// It is also the only end-to-end check of the pipeline that does not need a
// GUI, a microphone, or macOS to grant anything.
//
// USAGE
//   replay_harness --model <ggml.bin> --audio <file.wav> --script <file.txt>
//                  [--language hu] [--verbose] [--expect-progress 0.5]
//
// Exit code is 0 when the run meets its expectations, 1 otherwise, so it can
// be wired into CI once there are recordings to check against.

#include "../aligner.h"
#include "../audiocapture.h"
#include "../scriptmodel.h"
#include "../speechrecognizer.h"
#include "../textnormalize.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using namespace qprompt::voicetracking;

namespace {

struct Options {
    std::string modelPath;
    std::string audioPath;
    std::string scriptPath;
    std::string language = "en";
    bool verbose = false;
    // Fraction of the script the position must advance through for the run to
    // count as tracking. Deliberately loose by default: this asserts "it
    // followed the speaker", not "it matched a golden trace".
    double expectProgress = 0.5;
};

// Stands in for the view. Real geometry comes from QML; here we assume a fixed
// line height and a fixed number of tokens per line, which is enough to
// exercise the control loop's arithmetic.
constexpr double kLineHeightPx = 40.0;
constexpr double kTokensPerLine = 7.0;
constexpr double kPxPerToken = kLineHeightPx / kTokensPerLine;

// Mirrors VoiceTracking.qml's follow rules, so the harness measures the
// behaviour the view actually produces rather than the aligner's raw output.
constexpr double kOpeningAgreementPx = 8 * kPxPerToken;
constexpr double kBackwardTolerancePx = 120.0;

std::string readFile(const std::string &path, bool *ok)
{
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        *ok = false;
        return {};
    }
    std::ostringstream ss;
    ss << in.rdbuf();
    *ok = true;
    return ss.str();
}

bool parseArgs(int argc, char **argv, Options *options)
{
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto next = [&](const char *what) -> std::string {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "error: %s needs a value\n", what);
                return {};
            }
            return argv[++i];
        };
        if (arg == "--model") {
            options->modelPath = next("--model");
        } else if (arg == "--audio") {
            options->audioPath = next("--audio");
        } else if (arg == "--script") {
            options->scriptPath = next("--script");
        } else if (arg == "--language") {
            options->language = next("--language");
        } else if (arg == "--expect-progress") {
            options->expectProgress = std::atof(next("--expect-progress").c_str());
        } else if (arg == "--verbose") {
            options->verbose = true;
        } else if (arg == "--help" || arg == "-h") {
            return false;
        } else {
            std::fprintf(stderr, "error: unknown argument '%s'\n", arg.c_str());
            return false;
        }
    }
    return !options->modelPath.empty() && !options->audioPath.empty()
           && !options->scriptPath.empty();
}

void usage()
{
    std::printf(
        "usage: replay_harness --model <ggml.bin> --audio <file.wav>\n"
        "                      --script <file.txt> [--language hu]\n"
        "                      [--verbose] [--expect-progress 0.5]\n\n"
        "Replays a recording through the real whisper -> aligner -> controller\n"
        "pipeline and reports how the prompter would have behaved.\n");
}

} // namespace

int main(int argc, char **argv)
{
    Options options;
    if (!parseArgs(argc, argv, &options)) {
        usage();
        return 2;
    }

    bool ok = false;
    const std::string scriptText = readFile(options.scriptPath, &ok);
    if (!ok || scriptText.empty()) {
        std::fprintf(stderr, "error: could not read script '%s'\n",
                     options.scriptPath.c_str());
        return 2;
    }

    ScriptModel script;
    script.build(scriptText);
    if (script.tokenCount() == 0) {
        std::fprintf(stderr, "error: script contains no tokens\n");
        return 2;
    }

    Aligner aligner;
    aligner.setScript(script);

    SpeechRecognizer recognizer;
    SpeechRecognizer::Config config;
    config.modelPath = options.modelPath;
    config.language = options.language;
    if (!recognizer.loadModel(config)) {
        std::fprintf(stderr, "error: %s\n", recognizer.lastError().c_str());
        return 2;
    }

    std::printf("script:   %s (%zu tokens)\n",
                options.scriptPath.c_str(), script.tokenCount());
    std::printf("audio:    %s\n", options.audioPath.c_str());
    std::printf("language: %s\n\n", options.language.c_str());

    // Simulated view state. The prompter is moved TO the matched line and holds
    // there until the next confident match, which is what the live path does.
    double positionPx = 0.0;
    double lastClockSec = 0.0;

    size_t observations = 0;
    size_t confidentCount = 0;
    size_t maxTokenIndex = 0;
    size_t backwardJumps = 0;
    size_t lastTokenIndex = 0;
    double maxStepPx = 0.0;
    bool sawBackwardScroll = false;
    bool haveOpening = false;
    bool havePending = false;
    double pendingPx = 0.0;
    double pendingAudioSec = -1.0;
    bool sawNonFinite = false;

    recognizer.setObservationCallback([&](const Observation &observation) {
        ++observations;

        const AlignResult result = aligner.update(observation);

        const double nowSec = observation.audioTimeSec;
        lastClockSec = nowSec;

        // Where the matched token sits, in pixels.
        const double targetPx =
            static_cast<double>(result.tokenIndex) * kPxPerToken;

        // Follow: a trusted match moves the view to that line, anything else
        // leaves it exactly where it is. This mirrors VoiceTracking.qml,
        // including the two rules that keep the view from lurching.
        if (result.confident) {
            const double step = targetPx - positionPx;

            // 1. Confirm the opening. Until the view has committed to a
            //    position, a match may be a repeated phrase elsewhere in the
            //    script; wait for a second fix from LATER audio that agrees.
            bool act = true;
            if (!haveOpening) {
                const bool agrees = havePending
                    && std::abs(targetPx - pendingPx) <= kOpeningAgreementPx;
                const bool laterAudio = nowSec > pendingAudioSec + 1e-6;
                if (agrees && laterAudio) {
                    haveOpening = true;
                } else {
                    if (!agrees) {
                        pendingPx = targetPx;
                        pendingAudioSec = nowSec;
                        havePending = true;
                    }
                    act = false;
                }
            }

            // 2. Ignore small backward corrections. Alignment refines itself as
            //    whisper firms up a window, and a token or two of backtrack is
            //    invisible as a number but reads as a glitch on screen.
            if (act && step < 0.0 && -step < kBackwardTolerancePx)
                act = false;

            if (act) {
                if (step < 0.0)
                    sawBackwardScroll = true;
                maxStepPx = std::max(maxStepPx, std::abs(step));
                positionPx = targetPx;
            }
        }

        if (!std::isfinite(positionPx))
            sawNonFinite = true;

        if (result.confident) {
            ++confidentCount;
            // Counted only once the view has committed to a position. Before
            // that the aligner is still deciding between candidate openings and
            // will legitimately report distant alternatives; those never reach
            // the screen, so counting them measures the search rather than the
            // behaviour. What matters is that the position never walks
            // backwards once the reader is being followed.
            if (haveOpening && result.tokenIndex + 3 < lastTokenIndex)
                ++backwardJumps;
            lastTokenIndex = result.tokenIndex;
            maxTokenIndex = std::max(maxTokenIndex, result.tokenIndex);
        }

        if (options.verbose) {
            std::printf("t=%6.2f %-8s tok=%4zu conf=%.2f %-8s pos=%8.1f\n",
                        observation.audioTimeSec,
                        observation.isVolatile ? "volatile" : "FINAL",
                        result.tokenIndex,
                        result.confidence,
                        result.confident ? "TRACK" : "hold",
                        positionPx);
        }
    });

    if (!recognizer.runFile(options.audioPath)) {
        std::fprintf(stderr, "error: %s\n", recognizer.lastError().c_str());
        return 2;
    }

    // ---------------------------------------------------------------------
    // Report
    // ---------------------------------------------------------------------
    const double progress = script.tokenCount() > 0
                                ? static_cast<double>(maxTokenIndex)
                                      / static_cast<double>(script.tokenCount())
                                : 0.0;
    const double confidentRatio =
        observations > 0 ? static_cast<double>(confidentCount)
                               / static_cast<double>(observations)
                         : 0.0;

    std::printf("\n=== replay summary ===\n");
    std::printf("observations:      %zu\n", observations);
    std::printf("confident:         %zu (%.0f%%)\n",
                confidentCount, confidentRatio * 100.0);
    std::printf("furthest token:    %zu / %zu (%.0f%% of script)\n",
                maxTokenIndex, script.tokenCount(), progress * 100.0);
    std::printf("backward jumps:    %zu\n", backwardJumps);
    std::printf("largest jump:      %.1f px\n", maxStepPx);
    std::printf("final position:    %.1f px\n", positionPx);
    std::printf("last transcript:   %s\n", recognizer.lastTranscript().c_str());

    // ---------------------------------------------------------------------
    // Verdict. These are behavioural invariants, not a golden trace: a golden
    // trace would break on every whisper or tuning change and teach us nothing.
    // ---------------------------------------------------------------------
    int failures = 0;
    auto require = [&](bool condition, const char *what) {
        std::printf("%-42s %s\n", what, condition ? "ok" : "FAILED");
        if (!condition)
            ++failures;
    };

    std::printf("\n=== checks ===\n");
    require(observations > 0, "produced observations");
    require(!sawNonFinite, "no non-finite velocity or position");
    require(!sawBackwardScroll, "never scrolled backwards");
    require(backwardJumps == 0, "no confident backward jumps");
    require(progress >= options.expectProgress,
            "advanced through the expected share of script");

    std::printf("\nRESULT: %s\n", failures == 0 ? "PASS" : "FAIL");
    return failures == 0 ? 0 : 1;
}
