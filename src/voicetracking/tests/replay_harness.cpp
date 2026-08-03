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
#include "../scrollcontroller.h"
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

    ScrollController controller;
    controller.setLineHeight(kLineHeightPx);
    // A plausible constant speed to fall back to, so coasting is observable.
    controller.setFallbackVelocity(kPxPerToken * 2.6);

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

    // Simulated view state. The scroll position integrates the commanded
    // velocity, which is what the real prompter's animation does.
    double positionPx = 0.0;
    double lastClockSec = 0.0;

    size_t observations = 0;
    size_t confidentCount = 0;
    size_t maxTokenIndex = 0;
    size_t backwardJumps = 0;
    size_t lastTokenIndex = 0;
    double maxVelocity = 0.0;
    bool sawNegativeVelocity = false;
    bool sawNonFinite = false;

    recognizer.setObservationCallback([&](const Observation &observation) {
        ++observations;

        const AlignResult result = aligner.update(observation);

        // Advance the simulated clock to this observation's audio time, so the
        // control loop sees the same dt the live path would.
        const double nowSec = observation.audioTimeSec;
        const double dt = std::max(0.0, nowSec - lastClockSec);
        lastClockSec = nowSec;

        // Where the matched token sits, in pixels.
        const double targetPx =
            static_cast<double>(result.tokenIndex) * kPxPerToken;

        const ScrollCommand command =
            controller.update(result.confident,
                              targetPx,
                              positionPx,
                              result.estimatedTokensPerSec * kPxPerToken,
                              nowSec);

        // Integrate, exactly as the view's animation would.
        positionPx += command.velocityPxPerSec * dt;

        if (!std::isfinite(command.velocityPxPerSec) || !std::isfinite(positionPx))
            sawNonFinite = true;
        if (command.velocityPxPerSec < 0.0)
            sawNegativeVelocity = true;
        maxVelocity = std::max(maxVelocity, command.velocityPxPerSec);

        if (result.confident) {
            ++confidentCount;
            if (result.tokenIndex + 3 < lastTokenIndex)
                ++backwardJumps;
            lastTokenIndex = result.tokenIndex;
            maxTokenIndex = std::max(maxTokenIndex, result.tokenIndex);
        }

        if (options.verbose) {
            std::printf("t=%6.2f %-8s tok=%4zu conf=%.2f %-8s v=%7.1f px/s  pos=%8.1f\n",
                        observation.audioTimeSec,
                        observation.isVolatile ? "volatile" : "FINAL",
                        result.tokenIndex,
                        result.confidence,
                        result.confident ? "TRACK" : "coast",
                        command.velocityPxPerSec,
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
    std::printf("peak velocity:     %.1f px/s\n", maxVelocity);
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
    require(!sawNegativeVelocity, "never scrolled backwards");
    require(backwardJumps == 0, "no confident backward jumps");
    require(progress >= options.expectProgress,
            "advanced through the expected share of script");

    std::printf("\nRESULT: %s\n", failures == 0 ? "PASS" : "FAIL");
    return failures == 0 ? 0 : 1;
}
