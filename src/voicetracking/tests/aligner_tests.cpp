// SPDX-FileCopyrightText: QPrompt voice tracking
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Self-contained test binary for the voice-tracking engine. No test framework:
// a small CHECK macro, a pass/fail tally, and a non-zero exit on failure.
//
// Build & run:
//   clang++ -std=c++17 -O2 -Wall -Wextra \
//       src/voicetracking/textnormalize.cpp \
//       src/voicetracking/scriptmodel.cpp \
//       src/voicetracking/aligner.cpp \
//       src/voicetracking/tests/aligner_tests.cpp \
//       -o /tmp/aligner_tests && /tmp/aligner_tests

#include "../aligner.h"
#include "../scriptmodel.h"
#include "../textnormalize.h"

#include <cmath>
#include <cstdio>
#include <sstream>
#include <utility>
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
std::vector<std::string> g_failures;

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
        double _va = (a);                                                  \
        double _vb = (b);                                                  \
        if (std::fabs(_va - _vb) > (tol)) {                                \
            std::ostringstream _oss;                                       \
            _oss << #a " ~= " #b << "\n       actual: " << _va             \
                 << " vs " << _vb;                                         \
            reportFailure(__FILE__, __LINE__, _oss.str());                 \
        } else {                                                           \
            reportPass();                                                  \
        }                                                                  \
    } while (0)

void beginTest(const std::string &name)
{
    g_currentTest = name;
    std::printf("running: %s\n", name.c_str());
}

// --- helpers ---------------------------------------------------------------

std::vector<std::string> words(const std::string &text)
{
    return tokenizeToStrings(text);
}

// Feeds a script's own words to an aligner in chunks, simulating a clean
// read-through with well-behaved (non-overlapping) observations.
struct Speaker {
    Aligner *aligner = nullptr;
    double time = 0.0;
    double secondsPerToken = 0.4;

    AlignResult say(const std::string &text, bool isVolatile = false)
    {
        Observation obs;
        obs.tokens = words(text);
        obs.audioTimeSec = time;
        obs.isVolatile = isVolatile;
        time += secondsPerToken * static_cast<double>(
                    obs.tokens.empty() ? 1 : obs.tokens.size());
        return aligner->update(obs);
    }
};

}  // namespace

// ---------------------------------------------------------------------------
// textnormalize tests
// ---------------------------------------------------------------------------

void testUtf8AndCaseFolding()
{
    beginTest("utf8 + case folding (Hungarian alphabet)");

    // The complete Hungarian accented set, upper -> lower.
    CHECK_EQ(toLowerUtf8("ÁÉÍÓÖŐÚÜŰ"), std::string("áéíóöőúüű"));
    CHECK_EQ(toLowerUtf8("HÁZ"), std::string("ház"));
    CHECK_EQ(toLowerUtf8("ŐRÜLT"), std::string("őrült"));
    CHECK_EQ(toLowerUtf8("TŰZ"), std::string("tűz"));
    CHECK_EQ(toLowerUtf8("Hello World"), std::string("hello world"));

    // Code point counting must not count bytes.
    CHECK_EQ(utf8Length("ház"), size_t(3));
    CHECK_EQ(utf8Length("őrült"), size_t(5));
    CHECK_EQ(utf8Length(""), size_t(0));
    CHECK_MSG(std::string("ház").size() == 4,
              "sanity: 'ház' is 4 bytes but 3 code points");

    // Malformed UTF-8 must not hang or over-consume.
    std::string bad = "ab\xFF\xFE" "cd";
    CHECK_MSG(utf8Length(bad) >= 4, "malformed input still makes progress");

    // Truncated multi-byte sequence at end of string.
    std::string truncated = "ha\xC3";
    size_t pos = 0;
    int guard = 0;
    while (pos < truncated.size() && guard < 100) {
        utf8Decode(truncated, pos);
        ++guard;
    }
    CHECK_MSG(guard < 100, "truncated sequence terminates decoding loop");
}

void testDiacriticFolding()
{
    beginTest("diacritic folding");

    CHECK_EQ(foldDiacritics("ház"), std::string("haz"));
    CHECK_EQ(foldDiacritics("őrült"), std::string("orult"));
    CHECK_EQ(foldDiacritics("tűz"), std::string("tuz"));
    CHECK_EQ(foldDiacritics("köszönöm"), std::string("koszonom"));
    CHECK_EQ(foldDiacritics("hello"), std::string("hello"));

    // ő and ö both fold to o, which is precisely why folding must not be the
    // primary key: they are different letters in Hungarian.
    CHECK_EQ(foldDiacritics("ő"), foldDiacritics("ö"));
    CHECK_MSG(std::string("ő") != std::string("ö"),
              "ő and ö are distinct before folding");
}

void testTokenization()
{
    beginTest("tokenization + exact offsets");

    const std::string text = "Hello, world!  This is  a test.";
    const std::vector<RawToken> toks = tokenize(text);

    CHECK_EQ(toks.size(), size_t(6));
    CHECK_EQ(toks[0].normalized, std::string("hello"));
    CHECK_EQ(toks[1].normalized, std::string("world"));
    CHECK_EQ(toks[5].normalized, std::string("test"));

    // Offsets must point exactly at the raw token in the ORIGINAL string,
    // including the punctuation normalization removed.
    CHECK_EQ(text.substr(toks[0].startOffset,
                         toks[0].endOffset - toks[0].startOffset),
             std::string("Hello,"));
    CHECK_EQ(text.substr(toks[1].startOffset,
                         toks[1].endOffset - toks[1].startOffset),
             std::string("world!"));
    CHECK_EQ(text.substr(toks[5].startOffset,
                         toks[5].endOffset - toks[5].startOffset),
             std::string("test."));

    // Offsets must be strictly increasing.
    for (size_t i = 1; i < toks.size(); ++i) {
        CHECK(toks[i].startOffset > toks[i - 1].startOffset);
    }

    // Pure punctuation produces no tokens.
    CHECK_EQ(tokenize("--- ... !!!").size(), size_t(0));
    CHECK_EQ(tokenize("").size(), size_t(0));
    CHECK_EQ(tokenize("   \n\t  ").size(), size_t(0));

    // Internal apostrophes and hyphens are folded away, not split.
    CHECK_EQ(normalizeToken("don't"), std::string("dont"));
    CHECK_EQ(normalizeToken("well-known"), std::string("wellknown"));

    // Offsets remain exact with multi-byte characters present.
    const std::string hu = "A ház nagy.";
    const std::vector<RawToken> huToks = tokenize(hu);
    CHECK_EQ(huToks.size(), size_t(3));
    CHECK_EQ(hu.substr(huToks[1].startOffset,
                       huToks[1].endOffset - huToks[1].startOffset),
             std::string("ház"));
    CHECK_EQ(huToks[2].normalized, std::string("nagy"));
}

void testTokenSimilarity()
{
    beginTest("token similarity (length-aware + prefix-weighted)");

    // Identity.
    CHECK_NEAR(tokenSimilarity("house", "house"), 1.0, 1e-9);
    CHECK_NEAR(tokenSimilarity("ház", "ház"), 1.0, 1e-9);

    // Empty inputs.
    CHECK_NEAR(tokenSimilarity("", "house"), 0.0, 1e-9);
    CHECK_NEAR(tokenSimilarity("", ""), 0.0, 1e-9);

    // REQUIREMENT: a 1-char error in a short word must not automatically fail.
    // A flat "ratio > 0.8" rule would score cat/cot at 0.67 and reject it.
    CHECK_MSG(tokenSimilarity("cat", "cot") > 0.35,
              "1-char error in a 3-char word retains meaningful similarity");
    CHECK_MSG(tokenSimilarity("ház", "hát") > 0.35,
              "1-char error in a 3-char Hungarian word likewise");

    // REQUIREMENT: long unrelated words must not pass.
    CHECK_MSG(tokenSimilarity("presentation", "refrigerator") < 0.4,
              "long unrelated words stay far apart");
    CHECK_MSG(tokenSimilarity("teleprompter", "handkerchief") < 0.4,
              "long unrelated words stay far apart");

    // REQUIREMENT: Hungarian agglutination. The same stem with stacked suffixes
    // must score high; this is the case whole-word matching destroys.
    CHECK_MSG(tokenSimilarity("ház", "házban") > 0.6,
              "bare stem vs inessive suffix");
    CHECK_MSG(tokenSimilarity("ház", "házaimban") > 0.45,
              "bare stem vs heavily suffixed form");
    CHECK_MSG(tokenSimilarity("könyv", "könyvben") > 0.6,
              "stem 'könyv' vs 'könyvben'");
    CHECK_MSG(tokenSimilarity("beszél", "beszélgetünk") > 0.4,
              "verb stem vs derived form");

    // A shared short prefix must NOT be enough on its own.
    CHECK_MSG(tokenSimilarity("a", "abszolút") < 0.35,
              "1-letter agreement is not evidence");
    CHECK_MSG(tokenSimilarity("be", "beszélgetünk") < 0.5,
              "2-letter agreement is weak");

    // Diacritic fallback: works, but scores below an exact match.
    const double folded = tokenSimilarity("házban", "hazban");
    CHECK_MSG(folded > 0.7, "diacritic-only difference still matches");
    CHECK_MSG(folded < 1.0,
              "diacritic-only match scores strictly below exact match");

    // Genuinely different words that differ only by an accent must remain
    // distinguishable: 'ő' vs 'ö' is a real distinction.
    CHECK_MSG(tokenSimilarity("örült", "őrült") < 1.0,
              "ö/ő pair does not score as identical");
}

void testFunctionWords()
{
    beginTest("function word detection");

    CHECK(isFunctionWord("the"));
    CHECK(isFunctionWord("and"));
    CHECK(isFunctionWord("a"));
    CHECK(isFunctionWord("az"));      // Hungarian article
    CHECK(isFunctionWord("és"));      // Hungarian "and"
    CHECK(isFunctionWord("hogy"));    // Hungarian "that"
    CHECK(isFunctionWord("nem"));     // Hungarian "not"

    CHECK(!isFunctionWord("teleprompter"));
    CHECK(!isFunctionWord("ház"));
    CHECK(!isFunctionWord("quarterly"));
}

// ---------------------------------------------------------------------------
// scriptmodel tests
// ---------------------------------------------------------------------------

void testScriptModelBasics()
{
    beginTest("script model: offsets and weights");

    const std::string script =
        "The quarterly revenue exceeded expectations. "
        "The team delivered a remarkable performance.";

    ScriptModel model;
    model.build(script);

    CHECK(!model.empty());
    // "The quarterly revenue exceeded expectations." = 5
    // "The team delivered a remarkable performance." = 6
    CHECK_EQ(model.tokenCount(), size_t(11));

    // Offsets map back to the original text exactly.
    for (size_t i = 0; i < model.tokenCount(); ++i) {
        const ScriptToken &tok = model.token(i);
        const std::string raw =
            script.substr(tok.startOffset, tok.endOffset - tok.startOffset);
        CHECK_MSG(normalizeToken(raw) == tok.normalized,
                  "offset slice re-normalizes to the stored token");
    }

    // Rare content words must outweigh function words.
    const double revenueWeight = model.weightForNormalized("revenue");
    const double theWeight = model.weightForNormalized("the");
    CHECK_MSG(revenueWeight > theWeight * 2.0,
              "rare content word far outweighs a repeated article");

    // Round-tripping an offset back to a token index.
    const size_t idx = model.tokenIndexForCharOffset(
        model.charOffsetForToken(5));
    CHECK_EQ(idx, size_t(5));

    // Out-of-range access must not throw or crash.
    const ScriptToken &oob = model.token(9999);
    CHECK(oob.normalized.empty());
}

void testScriptModelDegenerate()
{
    beginTest("script model: degenerate inputs");

    ScriptModel empty;
    empty.build("");
    CHECK(empty.empty());
    CHECK_EQ(empty.tokenCount(), size_t(0));
    CHECK_EQ(empty.charOffsetForToken(0), size_t(0));
    CHECK_EQ(empty.findCandidateSpans({"anything"}, 5).size(), size_t(0));

    ScriptModel punct;
    punct.build("... --- !!! ???");
    CHECK(punct.empty());

    ScriptModel single;
    single.build("word");
    CHECK_EQ(single.tokenCount(), size_t(1));
    CHECK_EQ(single.charOffsetForToken(0), size_t(0));
}

void testInvertedIndex()
{
    beginTest("script model: inverted index retrieval");

    std::string script;
    for (int i = 0; i < 20; ++i) {
        script += "filler words of no particular consequence here today. ";
    }
    script += "The unmistakable pangolin danced. ";
    for (int i = 0; i < 20; ++i) {
        script += "more filler words of no particular consequence here. ";
    }

    ScriptModel model;
    model.build(script);

    const std::vector<SpanCandidate> spans =
        model.findCandidateSpans(words("unmistakable pangolin danced"), 5);
    CHECK(!spans.empty());

    if (!spans.empty()) {
        const size_t hit = spans[0].tokenIndex;
        const std::string found = model.token(hit).normalized;
        CHECK_MSG(found == "unmistakable" || found == "pangolin"
                      || found == "danced" || found == "the",
                  "top retrieval lands on the distinctive phrase, got: " + found);
    }

    // A query of pure function words must retrieve nothing: it cannot locate.
    const std::vector<SpanCandidate> useless =
        model.findCandidateSpans(words("the of a and"), 5);
    CHECK_MSG(useless.empty(), "function-word-only query retrieves nothing");
}

// ---------------------------------------------------------------------------
// aligner tests
// ---------------------------------------------------------------------------

const char *kEnglishScript =
    "Good morning everyone and welcome to the quarterly business review. "
    "Today we will examine three major topics that shaped our performance. "
    "First, the revenue figures exceeded our original projections by a "
    "considerable margin across every regional market. "
    "Second, the engineering organization shipped fourteen significant "
    "features while reducing critical infrastructure incidents. "
    "Third, customer satisfaction improved measurably following the support "
    "restructuring we completed last spring. "
    "Let me begin with the financial details before opening the floor to "
    "questions from the audience.";

void testPerfectReadThrough()
{
    beginTest("aligner: perfect read-through advances monotonically");

    ScriptModel model;
    model.build(kEnglishScript);

    Aligner aligner;
    aligner.setScript(model);

    Speaker speaker{&aligner, 0.0, 0.4};

    // Read the script in phrase-sized chunks, as whisper would emit them.
    const std::vector<std::string> chunks = {
        "Good morning everyone and welcome",
        "to the quarterly business review",
        "Today we will examine three major topics",
        "that shaped our performance",
        "First the revenue figures exceeded",
        "our original projections by a considerable margin",
        "across every regional market",
        "Second the engineering organization shipped",
        "fourteen significant features",
        "while reducing critical infrastructure incidents",
    };

    size_t previousIndex = 0;
    int confidentCount = 0;
    int monotonicViolations = 0;

    for (size_t c = 0; c < chunks.size(); ++c) {
        const AlignResult r = speaker.say(chunks[c]);
        if (r.confident) {
            ++confidentCount;
            if (r.tokenIndex + 2 < previousIndex) {
                ++monotonicViolations;
            }
            previousIndex = r.tokenIndex;
        }
    }

    CHECK_MSG(confidentCount >= 7,
              "confident on most chunks, got "
                  + std::to_string(confidentCount) + "/10");
    CHECK_MSG(monotonicViolations == 0,
              "position never runs backward during a clean read, violations: "
                  + std::to_string(monotonicViolations));

    // The final position must be well into the script, not stuck at the start.
    const AlignResult final = aligner.currentEstimate();
    CHECK_MSG(final.tokenIndex > 40,
              "tracked deep into the script, got token "
                  + std::to_string(final.tokenIndex));

    // charOffset must be consistent with tokenIndex.
    CHECK_EQ(final.charOffset, model.charOffsetForToken(final.tokenIndex));

    // A plausible reading rate should have been estimated.
    CHECK_MSG(final.estimatedTokensPerSec > 0.0,
              "produces a usable feed-forward velocity");
}

void testWhisperRevisionNoDoubleAdvance()
{
    beginTest("aligner: whisper revisions do not double-advance");

    ScriptModel model;
    model.build(kEnglishScript);

    // --- Run A: clean, non-overlapping observations ------------------------
    Aligner cleanAligner;
    cleanAligner.setScript(model);
    {
        Observation o1;
        o1.tokens = words("Good morning everyone and welcome to the quarterly");
        o1.audioTimeSec = 0.0;
        o1.isVolatile = false;
        cleanAligner.update(o1);

        Observation o2;
        o2.tokens = words("business review Today we will examine three major topics");
        o2.audioTimeSec = 3.0;
        o2.isVolatile = false;
        cleanAligner.update(o2);
    }
    const AlignResult cleanResult = cleanAligner.currentEstimate();

    // --- Run B: the SAME speech delivered as overlapping, revised windows ---
    // This is what whisper.cpp actually emits: each chunk re-transcribes a
    // sliding buffer and restates (and sometimes rewords) the recent past.
    Aligner revisedAligner;
    revisedAligner.setScript(model);
    {
        Observation a;
        a.tokens = words("Good morning everyone");
        a.audioTimeSec = 0.0;
        a.isVolatile = true;
        revisedAligner.update(a);

        // Restates the same audio window, extended.
        Observation b;
        b.tokens = words("Good morning everyone and welcome");
        b.audioTimeSec = 0.0;
        b.isVolatile = true;
        revisedAligner.update(b);

        // Restates again, with a revision ("everybody" -> "everyone").
        Observation c;
        c.tokens = words("Good morning everybody and welcome to the");
        c.audioTimeSec = 0.0;
        c.isVolatile = true;
        revisedAligner.update(c);

        // Settles.
        Observation d;
        d.tokens = words("Good morning everyone and welcome to the quarterly");
        d.audioTimeSec = 0.0;
        d.isVolatile = false;
        revisedAligner.update(d);

        // Next window, overlapping the tail of the previous one.
        Observation e;
        e.tokens = words("welcome to the quarterly business review Today we");
        e.audioTimeSec = 3.0;
        e.isVolatile = true;
        revisedAligner.update(e);

        Observation f;
        f.tokens = words("the quarterly business review Today we will examine three major topics");
        f.audioTimeSec = 3.0;
        f.isVolatile = false;
        revisedAligner.update(f);
    }
    const AlignResult revisedResult = revisedAligner.currentEstimate();

    // The revised run consumed far more TOKENS but the same SPEECH. If
    // revisions double-advanced, the revised position would be far ahead.
    const long drift = static_cast<long>(revisedResult.tokenIndex)
                     - static_cast<long>(cleanResult.tokenIndex);

    CHECK_MSG(std::labs(drift) <= 4,
              "revised transcript lands within 4 tokens of the clean run "
              "(clean=" + std::to_string(cleanResult.tokenIndex)
                  + ", revised=" + std::to_string(revisedResult.tokenIndex)
                  + ", drift=" + std::to_string(drift) + ")");

    // Specifically: it must not have run away forward.
    CHECK_MSG(drift < 8,
              "no runaway forward drift from revision double-counting");
}

void testWhisperVolatileCorrection()
{
    beginTest("aligner: a corrected volatile hypothesis does not strand the position");

    // NOTE ON WHAT THIS ACTUALLY VERIFIES
    // -----------------------------------
    // Mutation testing during development showed that the checkpoint/rollback
    // machinery and the token-overlap trimming can EACH be disabled without any
    // test in this file failing. The mechanism that actually prevents
    // double-advancement on revised transcripts is the HMM transition prior:
    // restated tokens re-match at the current position through the i->i and
    // small-backward transitions instead of driving the position forward.
    // (Breaking that prior DOES fail tests -- see the ad-lib test.)
    //
    // Rollback and trimming are therefore defense-in-depth, not the primary
    // guard, and this test asserts the OBSERVABLE contract -- a wrong volatile
    // hypothesis that whisper later corrects must leave the position where the
    // corrected text says it is -- rather than pretending to test internals it
    // cannot distinguish.

    ScriptModel model;
    model.build("alpha bravo charlie delta echo foxtrot golf hotel india juliet "
                "kilo lima mike november oscar papa quebec romeo sierra tango "
                "uniform victor whiskey xray yankee zulu");

    Aligner aligner;
    aligner.setScript(model);

    Observation init;
    init.tokens = words("alpha bravo charlie delta");
    init.audioTimeSec = 0.0;
    init.isVolatile = false;
    aligner.update(init);

    // whisper mishears this window badly and reports distant text.
    Observation wrong;
    wrong.tokens = words("quebec romeo sierra tango uniform victor");
    wrong.audioTimeSec = 1.0;
    wrong.isVolatile = true;
    aligner.update(wrong);

    // It then CORRECTS that same audio window.
    Observation corrected;
    corrected.tokens = words("echo foxtrot golf hotel");
    corrected.audioTimeSec = 1.0;
    corrected.isVolatile = false;
    aligner.update(corrected);

    const AlignResult r = aligner.currentEstimate();

    // The corrected text says the speaker is around "hotel" (token 7). The
    // position must follow the correction, not remain stranded near the bogus
    // "quebec/romeo" region (tokens 16-21).
    CHECK_MSG(r.tokenIndex < 14,
              "position follows the correction rather than staying at the "
              "mistaken location (token " + std::to_string(r.tokenIndex) + ")");
}

void testWhisperIdenticalResend()
{
    beginTest("aligner: re-sending an identical observation is idempotent");

    ScriptModel model;
    model.build(kEnglishScript);

    Aligner aligner;
    aligner.setScript(model);

    Observation obs;
    obs.tokens = words("Good morning everyone and welcome to the quarterly business review");
    obs.audioTimeSec = 0.0;
    obs.isVolatile = false;

    aligner.update(obs);
    const size_t after1 = aligner.currentEstimate().tokenIndex;

    // Send the exact same observation four more times.
    for (int i = 0; i < 4; ++i) {
        aligner.update(obs);
    }
    const size_t after5 = aligner.currentEstimate().tokenIndex;

    CHECK_MSG(after1 == after5,
              "identical re-sends do not move the position (after1="
                  + std::to_string(after1) + ", after5="
                  + std::to_string(after5) + ")");
}

void testWhisperSlidingWindowShape()
{
    beginTest("aligner: realistic whisper sliding-window emission");

    // This reproduces whisper.cpp's actual output shape: a window that slides
    // forward, where every emission restates the tail of the previous one and
    // appends a little new text. Over a long read this is where runaway drift
    // would accumulate if duplication were counted as speech.

    ScriptModel model;
    model.build(kEnglishScript);

    // Ground truth: the same speech, delivered as disjoint chunks.
    Aligner clean;
    clean.setScript(model);
    {
        Observation a;
        a.tokens = words("Good morning everyone and welcome to the quarterly business review");
        a.audioTimeSec = 0.0;
        clean.update(a);

        Observation b;
        b.tokens = words("Today we will examine three major topics that shaped our performance");
        b.audioTimeSec = 5.0;
        clean.update(b);

        Observation c;
        c.tokens = words("First the revenue figures exceeded our original projections");
        c.audioTimeSec = 10.0;
        clean.update(c);
    }
    const size_t cleanIndex = clean.currentEstimate().tokenIndex;

    // Sliding window over the identical speech.
    Aligner sliding;
    sliding.setScript(model);
    const std::vector<std::pair<std::string, double>> windows = {
        {"Good morning everyone and welcome", 0.0},
        {"Good morning everyone and welcome to the quarterly business review", 1.5},
        {"welcome to the quarterly business review Today we will", 3.5},
        {"the quarterly business review Today we will examine three major topics", 5.0},
        {"Today we will examine three major topics that shaped our performance", 7.0},
        {"three major topics that shaped our performance First the revenue", 9.0},
        {"that shaped our performance First the revenue figures exceeded our original projections", 10.5},
    };
    for (const auto &w : windows) {
        Observation o;
        o.tokens = words(w.first);
        o.audioTimeSec = w.second;
        o.isVolatile = true;
        sliding.update(o);
    }
    const size_t slidingIndex = sliding.currentEstimate().tokenIndex;

    const long drift =
        static_cast<long>(slidingIndex) - static_cast<long>(cleanIndex);

    CHECK_MSG(std::labs(drift) <= 5,
              "sliding-window delivery tracks the same speech to within 5 "
              "tokens of disjoint delivery (clean=" + std::to_string(cleanIndex)
                  + ", sliding=" + std::to_string(slidingIndex)
                  + ", drift=" + std::to_string(drift) + ")");

    // The decisive property: heavy restatement must not push the position past
    // the end of what was actually spoken.
    CHECK_MSG(slidingIndex < cleanIndex + 12,
              "no runaway forward drift across many overlapping windows");
}

void testSkippedParagraph()
{
    beginTest("aligner: skipped paragraph recovers via reacquisition");

    ScriptModel model;
    model.build(kEnglishScript);

    Aligner aligner;
    aligner.setScript(model);
    Speaker speaker{&aligner, 0.0, 0.4};

    // Start at the beginning.
    speaker.say("Good morning everyone and welcome to the quarterly business review");
    speaker.say("Today we will examine three major topics that shaped our performance");

    const size_t beforeSkip = aligner.currentEstimate().tokenIndex;

    // The speaker now jumps ahead, skipping the "First..." and "Second..."
    // paragraphs entirely, landing in the "Third..." paragraph.
    speaker.say("Third customer satisfaction improved measurably");
    speaker.say("following the support restructuring we completed last spring");
    speaker.say("Let me begin with the financial details");

    const AlignResult after = aligner.currentEstimate();

    CHECK_MSG(after.tokenIndex > beforeSkip + 20,
              "recovered forward past the skipped paragraphs (before="
                  + std::to_string(beforeSkip) + ", after="
                  + std::to_string(after.tokenIndex) + ")");

    // Verify it landed in the right REGION by checking nearby script text.
    bool nearCorrectRegion = false;
    const size_t lo = (after.tokenIndex > 12) ? after.tokenIndex - 12 : 0;
    const size_t hi = std::min(after.tokenIndex + 12, model.tokenCount());
    for (size_t i = lo; i < hi; ++i) {
        const std::string &t = model.token(i).normalized;
        if (t == "financial" || t == "restructuring" || t == "satisfaction"
            || t == "spring" || t == "details" || t == "begin") {
            nearCorrectRegion = true;
            break;
        }
    }
    CHECK_MSG(nearCorrectRegion,
              "landed near the actually-spoken text, token "
                  + std::to_string(after.tokenIndex));
}

void testRereadDoesNotRunAway()
{
    beginTest("aligner: re-reading a line does not run away forward");

    ScriptModel model;
    model.build(kEnglishScript);

    Aligner aligner;
    aligner.setScript(model);
    Speaker speaker{&aligner, 0.0, 0.4};

    speaker.say("Good morning everyone and welcome to the quarterly business review");
    speaker.say("Today we will examine three major topics");

    const size_t beforeReread = aligner.currentEstimate().tokenIndex;

    // The speaker stumbles and repeats the same line three times, each as a
    // genuinely new audio window (not a whisper revision).
    speaker.say("Today we will examine three major topics");
    speaker.say("Today we will examine three major topics");
    const AlignResult after = aligner.currentEstimate();

    // The position must stay near the repeated line, not march forward by one
    // line-length per repetition.
    const long drift = static_cast<long>(after.tokenIndex)
                     - static_cast<long>(beforeReread);
    CHECK_MSG(drift < 12,
              "re-reading does not advance a full line per repeat (drift="
                  + std::to_string(drift) + ")");
    CHECK_MSG(after.tokenIndex + 20 > beforeReread,
              "and does not collapse backward either");
}

void testRepeatedPhraseNoJumping()
{
    beginTest("aligner: repeated phrase does not cause jumping");

    // The same distinctive sentence appears three times at distant points.
    const std::string repeated =
        "This is the moment that changes everything for all of us. ";
    std::string script;
    script += repeated;                                       // occurrence 1
    script += "Alpha bravo charlie delta echo foxtrot golf hotel india "
              "juliet kilo lima mike november oscar papa quebec romeo. ";
    script += "Sierra tango uniform victor whiskey xray yankee zulu "
              "aardvark badger cheetah dolphin elephant flamingo giraffe. ";
    script += repeated;                                       // occurrence 2
    script += "Mercury venus earth mars jupiter saturn uranus neptune pluto "
              "ceres eris haumea makemake sedna quaoar orcus. ";
    script += "Hydrogen helium lithium beryllium boron carbon nitrogen oxygen "
              "fluorine neon sodium magnesium aluminium silicon. ";
    script += repeated;                                       // occurrence 3

    ScriptModel model;
    model.build(script);

    Aligner aligner;
    aligner.setScript(model);
    Speaker speaker{&aligner, 0.0, 0.4};

    // Establish position firmly in the SECOND block, before occurrence 2.
    speaker.say("Sierra tango uniform victor whiskey xray yankee zulu");
    speaker.say("aardvark badger cheetah dolphin elephant flamingo giraffe");

    const size_t anchored = aligner.currentEstimate().tokenIndex;

    // Now speak the ambiguous repeated sentence. It fits three places equally.
    std::vector<size_t> positions;
    for (int i = 0; i < 3; ++i) {
        speaker.say("This is the moment that changes everything");
        positions.push_back(aligner.currentEstimate().tokenIndex);
    }

    // REQUIREMENT: must not jump between the distant copies. The position
    // should stay consistent with prior motion (i.e. near occurrence 2, which
    // follows where we were anchored) rather than teleporting to 1 or 3.
    for (size_t p : positions) {
        const long delta = static_cast<long>(p) - static_cast<long>(anchored);
        CHECK_MSG(delta > -20 && delta < 60,
                  "stays near the motion-consistent occurrence rather than "
                  "teleporting (anchored=" + std::to_string(anchored)
                      + ", got=" + std::to_string(p) + ")");
    }

    // And the reported positions must be stable across the three updates,
    // not oscillating between distant copies.
    for (size_t i = 1; i < positions.size(); ++i) {
        const long swing = static_cast<long>(positions[i])
                         - static_cast<long>(positions[i - 1]);
        CHECK_MSG(std::labs(swing) < 40,
                  "no large swing between successive updates (swing="
                      + std::to_string(swing) + ")");
    }
}

void testAdLibDropsConfidence()
{
    beginTest("aligner: ad-lib drops confidence without wild jumps");

    ScriptModel model;
    model.build(kEnglishScript);

    Aligner aligner;
    aligner.setScript(model);
    Speaker speaker{&aligner, 0.0, 0.4};

    speaker.say("Good morning everyone and welcome to the quarterly business review");
    speaker.say("Today we will examine three major topics");

    const AlignResult onScript = aligner.currentEstimate();
    const size_t positionBeforeAdLib = onScript.tokenIndex;

    // Several seconds of completely off-script speech.
    const std::vector<std::string> adLibs = {
        "oh by the way I wanted to mention something",
        "my flight was delayed and the coffee here is terrible",
        "anyway where was I let me think for a second",
    };

    int confidentDuringAdLib = 0;
    size_t maxExcursion = 0;
    for (const std::string &line : adLibs) {
        const AlignResult r = speaker.say(line);
        if (r.confident) {
            ++confidentDuringAdLib;
        }
        const size_t excursion =
            (r.tokenIndex > positionBeforeAdLib)
                ? r.tokenIndex - positionBeforeAdLib
                : positionBeforeAdLib - r.tokenIndex;
        maxExcursion = std::max(maxExcursion, excursion);
    }

    CHECK_MSG(confidentDuringAdLib <= 1,
              "confidence drops during sustained ad-lib (confident on "
                  + std::to_string(confidentDuringAdLib) + "/3 updates)");
    CHECK_MSG(maxExcursion < 55,
              "no wild jump during ad-lib (max excursion "
                  + std::to_string(maxExcursion) + " tokens)");

    // Back on script: must recover.
    speaker.say("Second the engineering organization shipped fourteen significant features");
    speaker.say("while reducing critical infrastructure incidents");
    const AlignResult recovered = aligner.currentEstimate();

    bool nearEngineering = false;
    const size_t lo = (recovered.tokenIndex > 15) ? recovered.tokenIndex - 15 : 0;
    const size_t hi = std::min(recovered.tokenIndex + 15, model.tokenCount());
    for (size_t i = lo; i < hi; ++i) {
        const std::string &t = model.token(i).normalized;
        if (t == "engineering" || t == "infrastructure" || t == "incidents"
            || t == "fourteen" || t == "shipped") {
            nearEngineering = true;
            break;
        }
    }
    CHECK_MSG(nearEngineering,
              "recovers to the right place once back on script, landed at "
                  + std::to_string(recovered.tokenIndex));
}

void testGarbageTokensNoJump()
{
    beginTest("aligner: ASR garbage causes no jump");

    ScriptModel model;
    model.build(kEnglishScript);

    Aligner aligner;
    aligner.setScript(model);
    Speaker speaker{&aligner, 0.0, 0.4};

    speaker.say("Good morning everyone and welcome to the quarterly business review");
    const size_t before = aligner.currentEstimate().tokenIndex;

    // Noise: the kind of thing whisper emits on music, breath, or room tone.
    const std::vector<std::string> garbage = {
        "mmm hmm uhh",
        "zzzt krrk pfff",
        "[BLANK_AUDIO]",
        "xyzzy plugh frotz",
    };

    size_t maxExcursion = 0;
    for (const std::string &g : garbage) {
        const AlignResult r = speaker.say(g);
        const size_t excursion = (r.tokenIndex > before)
                                     ? r.tokenIndex - before
                                     : before - r.tokenIndex;
        maxExcursion = std::max(maxExcursion, excursion);
    }

    CHECK_MSG(maxExcursion < 40,
              "garbage tokens do not move the position far (max excursion "
                  + std::to_string(maxExcursion) + ")");
}

void testHungarianTracking()
{
    beginTest("aligner: Hungarian with agglutinative suffix variation");

    // Real Hungarian with the full accented set and stacked suffixes.
    const std::string huScript =
        "Jó reggelt kívánok mindenkinek és köszöntöm önöket a mai előadáson. "
        "Ma három fontos témáról fogunk beszélgetni részletesen. "
        "Először a negyedéves bevételekről szeretnék néhány szót mondani. "
        "A ház ahol dolgozunk teljesen megújult az elmúlt esztendőben. "
        "Másodszor a fejlesztői csapatunk rengeteg új funkciót készített el. "
        "Harmadszor az ügyfeleink elégedettsége jelentősen javult tavasszal. "
        "Végül pedig szeretném megköszönni mindenkinek a kitartó munkáját.";

    ScriptModel model;
    model.build(huScript);

    CHECK_MSG(!model.empty(), "Hungarian script tokenizes");
    CHECK_MSG(model.tokenCount() > 40, "reasonable Hungarian token count");

    // Accented tokens must survive normalization intact.
    bool foundAccented = false;
    for (size_t i = 0; i < model.tokenCount(); ++i) {
        if (model.token(i).normalized == "előadáson"
            || model.token(i).normalized == "jó") {
            foundAccented = true;
            break;
        }
    }
    CHECK_MSG(foundAccented, "accented Hungarian tokens normalize correctly");

    Aligner aligner;
    aligner.setScript(model);
    Speaker speaker{&aligner, 0.0, 0.4};

    // Speak with realistic ASR variation: different inflections than the
    // script, which is exactly what Hungarian morphology produces.
    speaker.say("Jó reggelt kívánok mindenkinek és köszöntöm önöket");
    const AlignResult r1 = aligner.currentEstimate();
    CHECK_MSG(r1.tokenIndex < 12, "tracks the Hungarian opening");

    speaker.say("Ma három fontos témáról fogunk beszélgetni részletesen");
    speaker.say("Először a negyedéves bevételekről szeretnék néhány szót mondani");

    const size_t beforeHouse = aligner.currentEstimate().tokenIndex;

    // THE KEY CASE: the transcript says "házban" (inessive) where the script
    // has the bare "ház". Whole-form matching fails here; prefix-weighted
    // similarity is what makes it work.
    speaker.say("A házban ahol dolgozunk teljesen megújult");

    const AlignResult afterHouse = aligner.currentEstimate();
    CHECK_MSG(afterHouse.tokenIndex > beforeHouse,
              "advances through the 'ház'/'házban' inflection mismatch "
              "(before=" + std::to_string(beforeHouse) + ", after="
                  + std::to_string(afterHouse.tokenIndex) + ")");

    bool nearHouseLine = false;
    const size_t lo = (afterHouse.tokenIndex > 12) ? afterHouse.tokenIndex - 12 : 0;
    const size_t hi = std::min(afterHouse.tokenIndex + 12, model.tokenCount());
    for (size_t i = lo; i < hi; ++i) {
        const std::string &t = model.token(i).normalized;
        if (t == "ház" || t == "dolgozunk" || t == "megújult"
            || t == "esztendőben") {
            nearHouseLine = true;
            break;
        }
    }
    CHECK_MSG(nearHouseLine,
              "landed on the 'ház' line despite the suffix mismatch, token "
                  + std::to_string(afterHouse.tokenIndex));

    // Continue tracking to the end.
    speaker.say("Másodszor a fejlesztői csapatunk rengeteg új funkciót készített");
    speaker.say("Harmadszor az ügyfeleink elégedettsége jelentősen javult");

    const AlignResult final = aligner.currentEstimate();
    CHECK_MSG(final.tokenIndex > afterHouse.tokenIndex,
              "keeps tracking through the rest of the Hungarian script");

    // Offsets must be valid byte offsets into the multi-byte script.
    CHECK_MSG(final.charOffset < huScript.size(),
              "char offset stays within the Hungarian script");
}

void testHungarianDiacriticAsrErrors()
{
    beginTest("aligner: Hungarian ASR dropping diacritics still tracks");

    const std::string huScript =
        "Az elnökünk köszöntötte a résztvevőket a konferencián. "
        "A gyűlésen sok érdekes előadás hangzott el a témában. "
        "Reméljük mindenki jól érezte magát a rendezvényen.";

    ScriptModel model;
    model.build(huScript);

    Aligner aligner;
    aligner.setScript(model);
    Speaker speaker{&aligner, 0.0, 0.4};

    // ASR drops accents entirely, a very common failure mode.
    speaker.say("Az elnokunk koszontotte a resztvevoket a konferencian");
    const AlignResult r = aligner.currentEstimate();

    CHECK_MSG(r.tokenIndex <= 10,
              "de-accented Hungarian transcript still aligns to the opening, "
              "token " + std::to_string(r.tokenIndex));
}

void testReanchor()
{
    beginTest("aligner: explicit reanchor");

    ScriptModel model;
    model.build(kEnglishScript);

    Aligner aligner;
    aligner.setScript(model);
    Speaker speaker{&aligner, 0.0, 0.4};

    speaker.say("Good morning everyone and welcome to the quarterly");

    // User scrolls to a distant part of the script.
    const size_t targetToken = 55;
    const size_t targetOffset = model.charOffsetForToken(targetToken);
    aligner.reanchor(targetOffset);

    const AlignResult r = aligner.currentEstimate();
    CHECK_MSG(r.tokenIndex == targetToken,
              "reanchor collapses the posterior onto the requested token "
              "(wanted " + std::to_string(targetToken) + ", got "
                  + std::to_string(r.tokenIndex) + ")");
    CHECK(r.confident);

    // Reanchoring must clear revision history: an old audio timestamp must not
    // roll the posterior back to a pre-reanchor state.
    Observation stale;
    stale.tokens = words("Good morning everyone");
    stale.audioTimeSec = 0.0;
    stale.isVolatile = false;
    aligner.update(stale);

    const AlignResult afterStale = aligner.currentEstimate();
    CHECK_MSG(afterStale.tokenIndex > 20,
              "a stale timestamp does not undo an explicit reanchor (token "
                  + std::to_string(afterStale.tokenIndex) + ")");
}

void testResetAndDegenerate()
{
    beginTest("aligner: reset and degenerate inputs");

    // No script at all.
    Aligner bare;
    Observation obs;
    obs.tokens = words("hello world");
    obs.audioTimeSec = 0.0;
    const AlignResult r = bare.update(obs);
    CHECK_MSG(!r.confident, "no script means never confident");
    CHECK_EQ(r.tokenIndex, size_t(0));
    CHECK_EQ(r.charOffset, size_t(0));

    // Empty script model.
    ScriptModel emptyModel;
    emptyModel.build("");
    Aligner emptyAligner;
    emptyAligner.setScript(emptyModel);
    const AlignResult r2 = emptyAligner.update(obs);
    CHECK_MSG(!r2.confident, "empty script means never confident");

    // Real script, empty observations.
    ScriptModel model;
    model.build(kEnglishScript);
    Aligner aligner;
    aligner.setScript(model);

    Observation emptyObs;
    emptyObs.audioTimeSec = 0.0;
    const AlignResult r3 = aligner.update(emptyObs);
    CHECK_MSG(!r3.confident, "empty observation is not confident");

    // Punctuation-only observation.
    Observation punctObs;
    punctObs.tokens = {"...", "---", "!!!"};
    punctObs.audioTimeSec = 1.0;
    aligner.update(punctObs);  // must not crash

    // reset() returns to the initial state.
    Speaker speaker{&aligner, 2.0, 0.4};
    speaker.say("Good morning everyone and welcome to the quarterly business review");
    CHECK(aligner.currentEstimate().tokenIndex > 0);

    aligner.reset();
    const AlignResult afterReset = aligner.currentEstimate();
    CHECK_EQ(afterReset.tokenIndex, size_t(0));
    CHECK_MSG(!afterReset.confident, "reset clears confidence");

    // Must still work after reset.
    Speaker speaker2{&aligner, 0.0, 0.4};
    speaker2.say("Good morning everyone and welcome to the quarterly business review");
    CHECK_MSG(aligner.currentEstimate().tokenIndex > 0,
              "aligner is usable again after reset");
}

void testSingleTokenScript()
{
    beginTest("aligner: single-token script");

    ScriptModel model;
    model.build("hello");

    Aligner aligner;
    aligner.setScript(model);

    Observation obs;
    obs.tokens = words("hello");
    obs.audioTimeSec = 0.0;
    const AlignResult r = aligner.update(obs);

    CHECK_EQ(r.tokenIndex, size_t(0));
    CHECK_MSG(r.charOffset == 0, "single token starts at offset 0");
    // Must not crash on further updates past the end.
    Observation obs2;
    obs2.tokens = words("hello there world");
    obs2.audioTimeSec = 1.0;
    aligner.update(obs2);
}

// ---------------------------------------------------------------------------

int main()
{
    std::printf("=== QPrompt voice-tracking engine tests ===\n\n");

    // textnormalize
    testUtf8AndCaseFolding();
    testDiacriticFolding();
    testTokenization();
    testTokenSimilarity();
    testFunctionWords();

    // scriptmodel
    testScriptModelBasics();
    testScriptModelDegenerate();
    testInvertedIndex();

    // aligner
    testPerfectReadThrough();
    testWhisperRevisionNoDoubleAdvance();
    testWhisperVolatileCorrection();
    testWhisperIdenticalResend();
    testWhisperSlidingWindowShape();
    testSkippedParagraph();
    testRereadDoesNotRunAway();
    testRepeatedPhraseNoJumping();
    testAdLibDropsConfidence();
    testGarbageTokensNoJump();
    testHungarianTracking();
    testHungarianDiacriticAsrErrors();
    testReanchor();
    testResetAndDegenerate();
    testSingleTokenScript();

    std::printf("\n=== summary ===\n");
    std::printf("passed: %d\n", g_passed);
    std::printf("failed: %d\n", g_failed);

    if (g_failed > 0) {
        std::printf("\nFAILURES:\n");
        for (const std::string &f : g_failures) {
            std::printf("%s\n", f.c_str());
        }
        std::printf("\nRESULT: FAILED\n");
        return 1;
    }

    std::printf("\nRESULT: ALL TESTS PASSED\n");
    return 0;
}
