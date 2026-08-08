// SPDX-FileCopyrightText: QPrompt voice tracking
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Pure C++17, no Qt, no external dependencies, no exceptions across the API.
//
// Token-position HMM that answers: "given what the speaker just said, where in
// the script are they?" This implements Decision 2 of the voice-tracking design
// document.
//
// ============================================================================
// WHISPER REVISION SEMANTICS — READ THIS FIRST
// ============================================================================
//
// whisper.cpp is NOT a streaming recognizer. It re-transcribes a sliding audio
// buffer (~3-5s window, ~0.5s hop). Every chunk it emits is a FRESH OBSERVATION
// OF THE RECENT PAST, not new text appended to a growing string. Consecutive
// chunks overlap heavily, and whisper freely REVISES what it previously said
// about the same audio.
//
// Concretely, three successive calls might carry:
//
//     t=1.0   "welcome to the show"
//     t=1.5   "welcome to the show today we"      <- re-states the first four
//     t=2.0   "welcome to the show today we are" <- re-states again, revised
//
// The naive implementation feeds all three into the HMM as if they were 4, 7
// and 8 new words. The position estimate then advances ~19 tokens for what was
// really 8 words of speech, and the prompter runs away from the speaker. This
// is the single easiest way to get this feature wrong, and the design document
// calls it out explicitly.
//
// HOW THIS CLASS SOLVES IT
// ------------------------
// Every Observation carries `audioTimeSec`, the start time of the audio window
// it was transcribed from, and `isVolatile`, saying whether whisper may still
// revise it.
//
//   1. CHECKPOINTING. Before applying a volatile observation, the aligner saves
//      its full posterior state into a checkpoint keyed by audioTimeSec.
//
//   2. ROLLBACK. When an observation arrives whose audioTimeSec is at or before
//      one already consumed, it is a REVISION of audio already accounted for.
//      The aligner rolls the posterior back to the checkpoint taken before that
//      audio was first applied, then re-applies the revised text. The net
//      effect is that the same audio moves the position exactly once, no matter
//      how many times whisper restates it.
//
//   3. TOKEN-LEVEL OVERLAP TRIMMING. Even within a forward-moving observation,
//      whisper usually repeats the tail of the previous one. After rollback the
//      aligner trims the leading tokens of the new observation that duplicate
//      the tail of what was already consumed, so only genuinely new words are
//      scored. This is a suffix/prefix overlap match, not an exact-equality
//      test, so it tolerates whisper changing a word or two in the overlap.
//
// The caller may therefore re-send overlapping and revised transcripts freely.
// The contract is only that audioTimeSec is monotonically meaningful — it must
// describe when the audio happened, not when the callback fired.
// ============================================================================

#ifndef QPROMPT_VOICETRACKING_ALIGNER_H
#define QPROMPT_VOICETRACKING_ALIGNER_H

#include <cstddef>
#include <deque>
#include <string>
#include <vector>

#include "scriptmodel.h"

namespace qprompt {
namespace voicetracking {

// One ASR update.
struct Observation {
    // Normalized-or-raw tokens from the recognizer. The aligner normalizes
    // them internally, so passing raw whisper output is fine.
    std::vector<std::string> tokens;

    // Start time, in seconds, of the audio window these tokens describe.
    // Drives checkpoint/rollback. Must reflect AUDIO time, not wall-clock
    // arrival time.
    double audioTimeSec = 0.0;

    // True while whisper may still revise this text (i.e. it is a partial
    // hypothesis over the tail of the buffer). Volatile observations are
    // checkpointed so they can be rolled back and replaced.
    bool isVolatile = true;
};

// The aligner's answer.
struct AlignResult {
    // False means "I do not know where the speaker is; keep doing what you were
    // doing." The caller must coast, never jump, when this is false.
    bool confident = false;

    // Best-estimate script token index. Only meaningful when confident is true,
    // but always populated with the current best guess for diagnostics.
    size_t tokenIndex = 0;

    // Byte offset into the script text corresponding to tokenIndex. This is
    // what the Qt adapter maps to a scroll position.
    size_t charOffset = 0;

    // Posterior mass concentrated near the winning mode, in [0, 1].
    double confidence = 0.0;

    // Filtered speech rate in script tokens per second, for the control loop's
    // feed-forward term. Falls back to a rolling estimate when timestamps look
    // unreliable, per Decision 3.
    double estimatedTokensPerSec = 0.0;
};

class Aligner {
public:
    Aligner();

    // Sets the script. Copies the model, resets all tracking state. Calling
    // this with an empty model leaves the aligner permanently unconfident,
    // which is the safe degenerate behaviour.
    void setScript(const ScriptModel &script);

    // Consumes one ASR update and returns the current best estimate.
    // Never throws. Safe to call with empty tokens (treated as silence, which
    // decays confidence without moving the position).
    AlignResult update(const Observation &observation);

    // Explicit re-anchor: the user scrolled manually, or jumped to a marker.
    // Collapses the posterior onto the given offset immediately and clears the
    // revision history, per the design doc's "manual scroll resets the
    // posterior immediately".
    void reanchor(size_t charOffset);

    // Returns to the initial state: no position, no confidence, no history.
    void reset();

    // Current estimate without consuming an observation.
    AlignResult currentEstimate() const;

    // --- Tuning knobs, exposed for the offline replay harness ---------------
    // All have defaults chosen in the constructor; see aligner.cpp for the
    // rationale behind each value.
    struct Config {
        // Beam / banding
        size_t beamWidth = 48;          // surviving states carried forward
        size_t bandBackward = 25;       // local band extent behind each mode
        size_t bandForward = 130;       // local band extent ahead of each mode
        size_t maxModes = 5;            // distinct location modes kept
        // Script tokens per "region" for the prune's diversity rule. One
        // representative of each region survives, so a repeated phrase cannot
        // crowd its rivals out of the beam and manufacture a false margin.
        // Roughly a long sentence: small enough to separate distant repeats,
        // large enough not to bloat the beam.
        size_t regionTokens = 40;

        // Transition log-probabilities (unnormalized; relative values matter)
        double logStay = -1.6;          // i -> i     (filler, ASR insertion)
        double logAdvance = -0.05;      // i -> i+1   (the expected case)
        double logSkipBase = -2.2;      // i -> i+k   base cost
        double logSkipPerToken = -0.28; // geometric growth per skipped token
        double logBackward = -3.4;      // small backward step (re-reading)
        double logRecovery = -11.0;     // distant jump, gated on low confidence

        // Cold-start tilt towards the top of the script, applied across the
        // whole document (0 at the first token, this value at the last). Readers
        // usually start at the beginning, so this breaks ties between repeated
        // phrases in favour of the earliest copy without forbidding a mid-script
        // start.
        double logStartBias = -2.0;


        // Extra cost for landing in script that has already been read aloud.
        // Scripts repeat words -- "QPrompt" appears eleven times in the demo
        // text -- and a repeat behind the reader scores just as well as the
        // real position, so the view snaps backwards to a line already spoken.
        // Text before the high-water mark is not where the speaker is, unless
        // they have genuinely gone back, which this is small enough to allow.
        double logAlreadySpoken = -2.6;
        // Re-reading within this many tokens of the high-water mark is normal
        // (correcting a stumble, repeating a phrase for a second take), so it
        // is not penalised at all.
        size_t spokenGraceTokens = 12;

        // Observation model
        double matchBonus = 2.4;        // log-LR for an exact normalized match
        double similarityFloor = 0.62;  // below this, similarity is not a match
        double mismatchPenalty = -0.85; // log-LR for a confident non-match

        // Confidence gating (design doc "starting gates")
        double minPosteriorMass = 0.55; // mass within +/- massRadius of winner
        size_t massRadius = 3;          // tokens
        double minMarginNats = 1.1;     // best vs runner-up cluster
        double minInformativeWeight = 1.05;  // summed weight of matched tokens

        // Reacquisition
        double lowConfidenceSeconds = 1.2;   // sustained low confidence trigger
        size_t reacquireCandidates = 6;      // spans retrieved per attempt
        size_t reacquireContext = 16;        // tokens scored around each span
        double reacquireMarginNats = 1.6;    // margin a challenger must win by
        int reacquireWinsRequired = 2;       // successive wins before moving

        // Velocity estimation
        double velocitySmoothing = 0.3;      // alpha for the EMA
        double defaultTokensPerSec = 2.6;    // fallback reading rate
    };

    const Config &config() const { return m_config; }
    void setConfig(const Config &config) { m_config = config; }

private:
    // One hypothesis: "the speaker is at script token `index`".
    struct State {
        size_t index = 0;
        double logProb = 0.0;
    };

    // Saved posterior, for rollback of revised whisper hypotheses.
    struct Checkpoint {
        double audioTimeSec = 0.0;
        std::vector<State> beam;
        std::vector<std::string> consumedTail;  // recent tokens already scored
        size_t bestIndex = 0;
        double confidence = 0.0;
        double tokensPerSec = 0.0;
        double lowConfidenceSince = -1.0;
        size_t reacquireCandidate = 0;
        int reacquireWins = 0;
        bool hasPosition = false;
        double lastAudioTimeSec = 0.0;
    };

    // Applies the tokens of one observation to the beam. Assumes rollback and
    // overlap trimming have already happened.
    void applyTokens(const std::vector<std::string> &tokens,
                     double audioTimeSec);

    // One HMM update step for a single observed token.
    void stepToken(const std::string &observed);

    // Observation log-likelihood ratio of `observed` against script token
    // `scriptIndex`, versus the background model.
    double observationLogLR(const std::string &observed,
                            size_t scriptIndex) const;

    // Normalizes the beam with log-sum-exp and prunes to beamWidth.
    void normalizeAndPrune();

    // Recomputes m_bestIndex / m_confidence from the current beam.
    void recomputeEstimate();

    // Attempts global reacquisition. Returns true if the posterior relocated.
    bool tryReacquire(const std::vector<std::string> &recentTokens,
                      double audioTimeSec);

    // Scores a candidate span against recent tokens, for reacquisition.
    double scoreSpan(size_t spanStart,
                     const std::vector<std::string> &recentTokens) const;

    // Trims leading tokens of `incoming` that duplicate the tail of
    // m_consumedTail. Returns the index of the first genuinely new token.
    size_t trimOverlap(const std::vector<std::string> &incoming) const;

    void saveCheckpoint(double audioTimeSec);
    bool rollbackTo(double audioTimeSec);

    Config m_config;
    ScriptModel m_script;

    std::vector<State> m_beam;

    // Rolling record of tokens already scored, used for overlap trimming and
    // as the query for reacquisition.
    std::vector<std::string> m_consumedTail;

    std::deque<Checkpoint> m_checkpoints;

    size_t m_bestIndex = 0;
    double m_confidence = 0.0;
    double m_tokensPerSec = 0.0;
    bool m_hasPosition = false;



    // Furthest point the speaker has been confidently placed at. Script before
    // this has already been read aloud, so a later match landing behind it is
    // far more likely to be a repeated phrase than the reader going back.
    // See kSpokenPenalty in aligner.cpp.
    size_t m_spokenThrough = 0;

    double m_lastAudioTimeSec = 0.0;
    double m_lowConfidenceSince = -1.0;

    // Reacquisition candidate that must win repeatedly before we relocate.
    size_t m_reacquireCandidate = 0;
    int m_reacquireWins = 0;

    // Tokens matched in the current update, for the informative-evidence gate.
    double m_lastInformativeWeight = 0.0;
};

}  // namespace voicetracking
}  // namespace qprompt

#endif  // QPROMPT_VOICETRACKING_ALIGNER_H
