// SPDX-FileCopyrightText: QPrompt voice tracking
// SPDX-License-Identifier: GPL-3.0-or-later

#include "aligner.h"

#include "textnormalize.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_map>

namespace qprompt {
namespace voicetracking {

namespace {

constexpr double kNegInf = -std::numeric_limits<double>::infinity();

// How many recently consumed tokens to retain for overlap trimming and as the
// reacquisition query. Must comfortably exceed whisper's window overlap
// (~3-5s of speech, so ~15 words) without becoming a stale query.
constexpr size_t kConsumedTailLimit = 40;

// How many checkpoints to retain. whisper hops ~0.5s, so 24 covers ~12s of
// revisable history, far more than it ever revises in practice.
constexpr size_t kCheckpointLimit = 24;

// Numerically stable log(sum(exp(v))).
double logSumExp(const std::vector<double> &values)
{
    if (values.empty()) {
        return kNegInf;
    }
    const double maxVal = *std::max_element(values.begin(), values.end());
    if (!std::isfinite(maxVal)) {
        return maxVal;  // all -inf
    }
    double sum = 0.0;
    for (double v : values) {
        sum += std::exp(v - maxVal);
    }
    return maxVal + std::log(sum);
}

}  // namespace

Aligner::Aligner() = default;

void Aligner::setScript(const ScriptModel &script)
{
    m_script = script;
    reset();
}

void Aligner::reset()
{
    m_beam.clear();
    m_consumedTail.clear();
    m_checkpoints.clear();
    m_bestIndex = 0;
    m_confidence = 0.0;
    m_tokensPerSec = 0.0;
    m_hasPosition = false;
    m_lastAudioTimeSec = 0.0;
    m_lowConfidenceSince = -1.0;
    m_reacquireCandidate = 0;
    m_reacquireWins = 0;
    m_lastInformativeWeight = 0.0;
    m_spokenThrough = 0;
}

void Aligner::reanchor(size_t charOffset)
{
    if (m_script.empty()) {
        return;
    }
    const size_t index =
        std::min(m_script.tokenIndexForCharOffset(charOffset),
                 m_script.tokenCount() - 1);

    // Collapse the posterior onto a tight distribution around the anchor. We do
    // not use a delta spike: the user's scroll lands on a line, not a word, so a
    // few tokens of uncertainty is honest.
    m_beam.clear();
    const size_t lo = (index > 3) ? index - 3 : 0;
    const size_t hi = std::min(index + 4, m_script.tokenCount());
    for (size_t i = lo; i < hi; ++i) {
        const double distance = static_cast<double>(
            (i > index) ? i - index : index - i);
        m_beam.push_back(State{i, -0.9 * distance});
    }
    normalizeAndPrune();

    // A manual re-anchor invalidates all revision history: the user has
    // asserted a position, and old checkpoints describe a different timeline.
    m_checkpoints.clear();
    m_consumedTail.clear();
    m_lowConfidenceSince = -1.0;
    m_reacquireCandidate = 0;
    m_reacquireWins = 0;

    m_bestIndex = index;
    m_hasPosition = true;
    m_confidence = 1.0;
    // An explicit re-anchor is the user overruling us. If they went back, the
    // script after this point has NOT been read yet, so the high-water mark
    // follows them down; otherwise it would penalise everything ahead of them.
    m_spokenThrough = index;
}

// ---------------------------------------------------------------------------
// Observation model
// ---------------------------------------------------------------------------

double Aligner::observationLogLR(const std::string &observed,
                                 size_t scriptIndex) const
{
    if (scriptIndex >= m_script.tokenCount()) {
        return m_config.mismatchPenalty;
    }
    const ScriptToken &tok = m_script.token(scriptIndex);

    // Weight scales how much this token's evidence counts. Rare content words
    // dominate; function words barely move the posterior, which is what stops a
    // run of "and the of a" from manufacturing confidence.
    const double weight = tok.weight;

    if (observed == tok.normalized) {
        // Exact normalized match: the strongest signal available.
        return m_config.matchBonus * weight;
    }

    const double sim = tokenSimilarity(observed, tok.normalized);
    if (sim >= m_config.similarityFloor) {
        // Softer, length-aware similarity. Scaled by how far above the floor it
        // sits, so a marginal match earns markedly less than a near-exact one.
        const double strength =
            (sim - m_config.similarityFloor) / (1.0 - m_config.similarityFloor);
        return m_config.matchBonus * weight * strength;
    }

    // A confident non-match. Penalized in proportion to weight: failing to
    // match a rare word here is strong evidence we are in the wrong place,
    // while failing to match "the" means very little.
    return m_config.mismatchPenalty * weight;
}

// ---------------------------------------------------------------------------
// HMM step
// ---------------------------------------------------------------------------

void Aligner::stepToken(const std::string &observed)
{
    if (m_script.empty()) {
        return;
    }
    const size_t tokenCount = m_script.tokenCount();

    // Cold start: no beam yet. Seed across the whole script so the first
    // observations can land anywhere -- a speaker may well start from the
    // middle -- but tilt the prior towards the beginning.
    //
    // Without the tilt every occurrence of a repeated phrase starts out equally
    // likely, and on a script that names itself eleven times the first window
    // lands on whichever copy whisper's partial transcript happens to favour.
    // That is what made the prompter open deep in the text and then snap back
    // to the top once more words arrived. The tilt is small: a few tokens of
    // real evidence overrides it, so starting mid-script still works, it just
    // has to be earned rather than assumed.
    if (m_beam.empty()) {
        m_beam.reserve(tokenCount);
        const double denominator = std::max<double>(1.0, static_cast<double>(tokenCount));
        for (size_t i = 0; i < tokenCount; ++i) {
            const double fraction = static_cast<double>(i) / denominator;
            m_beam.push_back(State{i, m_config.logStartBias * fraction});
        }
    }

    // Accumulate the new distribution in a map so several source states landing
    // on the same destination combine correctly.
    std::unordered_map<size_t, double> next;
    next.reserve(m_beam.size() * 8);

    // Everything already read aloud is discounted, whichever transition leads
    // there. Applying it here rather than on the backward transition alone
    // matters: a repeated phrase is usually reached by a RECOVERY jump or a
    // long skip, not by stepping back one token at a time.
    const size_t spokenLimit =
        m_spokenThrough > m_config.spokenGraceTokens
            ? m_spokenThrough - m_config.spokenGraceTokens
            : 0;
    auto addTo = [&next, spokenLimit, this](size_t index, double logProb) {
        if (index < spokenLimit)
            logProb += m_config.logAlreadySpoken;
        auto it = next.find(index);
        if (it == next.end()) {
            next.emplace(index, logProb);
        } else {
            // Combine two paths into the same state with log-sum-exp. Using a
            // sum rather than a max makes this a forward algorithm rather than
            // pure Viterbi, which gives better-calibrated posterior mass — and
            // posterior mass is exactly what the confidence gate reads.
            const double a = std::max(it->second, logProb);
            const double b = std::min(it->second, logProb);
            it->second = a + std::log1p(std::exp(b - a));
        }
    };

    const bool allowRecovery = !m_hasPosition
        || m_confidence < m_config.minPosteriorMass;

    for (const State &state : m_beam) {
        const double base = state.logProb;

        // --- i -> i : ASR insertion, filler word, ad-lib ---------------------
        // The observed token is not consumed from the script, so no observation
        // term is applied for the script token here.
        addTo(state.index, base + m_config.logStay);

        // --- i -> i+1 : the normal case -------------------------------------
        if (state.index + 1 < tokenCount) {
            addTo(state.index + 1,
                  base + m_config.logAdvance
                      + observationLogLR(observed, state.index + 1));
        }

        // --- i -> i+k : skipped script words, geometric penalty --------------
        // Bounded by the forward band; the penalty grows per skipped token so
        // long skips need correspondingly strong observation evidence.
        const size_t maxSkip =
            std::min(m_config.bandForward, tokenCount - state.index - 1);
        for (size_t k = 2; k <= maxSkip; ++k) {
            const double skipCost = m_config.logSkipBase
                + m_config.logSkipPerToken * static_cast<double>(k - 1);
            // Once the transition cost alone is hopeless, further k only get
            // worse; stop early rather than filling the map with dead states.
            if (skipCost < m_config.logRecovery) {
                break;
            }
            addTo(state.index + k,
                  base + skipCost
                      + observationLogLR(observed, state.index + k));
        }

        // --- small backward : immediate correction, re-reading a line --------
        const size_t maxBack = std::min(m_config.bandBackward, state.index);
        for (size_t k = 1; k <= maxBack; ++k) {
            const double backCost = m_config.logBackward
                + m_config.logSkipPerToken * static_cast<double>(k - 1);
            if (backCost < m_config.logRecovery) {
                break;
            }
            addTo(state.index - k,
                  base + backCost
                      + observationLogLR(observed, state.index - k));
        }
    }

    // --- low-prior recovery transitions -------------------------------------
    // Enabled ONLY once confidence has fallen, per the design doc. Without this
    // gate, a script with repeated phrases would teleport between them; with it,
    // distant relocation requires that local tracking has already failed.
    //
    // We seed recovery only at positions that actually match the observed token,
    // found through the inverted index, rather than over the whole script.
    if (allowRecovery) {
        std::vector<std::string> query{observed};
        const std::vector<SpanCandidate> spans =
            m_script.findCandidateSpans(query, m_config.reacquireCandidates);
        for (const SpanCandidate &span : spans) {
            addTo(span.tokenIndex,
                  m_config.logRecovery
                      + observationLogLR(observed, span.tokenIndex));
        }
    }

    m_beam.clear();
    m_beam.reserve(next.size());
    for (const auto &kv : next) {
        m_beam.push_back(State{kv.first, kv.second});
    }

    normalizeAndPrune();
}

void Aligner::normalizeAndPrune()
{
    if (m_beam.empty()) {
        return;
    }

    // Keep the strongest states. Sorting by log-probability then truncating is
    // the beam prune; it bounds cost per update to beamWidth * band.
    if (m_beam.size() > m_config.beamWidth) {
        std::partial_sort(m_beam.begin(),
                          m_beam.begin() + static_cast<long>(m_config.beamWidth),
                          m_beam.end(),
                          [](const State &a, const State &b) {
                              return a.logProb > b.logProb;
                          });

        // Before truncating, keep the best surviving state from each distinct
        // region of the script.
        //
        // A plain top-N prune is what let a repeated phrase report false
        // certainty. Matching a word that occurs eleven times seeds a cluster
        // around EVERY occurrence, but the winning cluster's neighbours crowd
        // the beam and the rival occurrences fall off the end. The margin gate
        // below then compares the winner against its own neighbours -- which is
        // no contest -- and reports near-total confidence for a position chosen
        // essentially at random among the copies.
        //
        // Retaining one representative per region keeps those rivals in play,
        // so the margin measures what it is meant to: how much better this
        // location is than the best ALTERNATIVE location. When several copies
        // fit equally the margin collapses, the fix is reported unconfident,
        // and the view holds still until the wording diverges -- which is the
        // honest answer, because until then the speaker's position genuinely
        // is ambiguous.
        std::vector<State> survivors(m_beam.begin(),
                                     m_beam.begin() + static_cast<long>(m_config.beamWidth));
        if (m_config.regionTokens > 0) {
            std::unordered_map<size_t, double> keptBest;
            for (const State &s : survivors)
                keptBest[s.index / m_config.regionTokens] =
                    std::max(keptBest.count(s.index / m_config.regionTokens)
                                 ? keptBest[s.index / m_config.regionTokens]
                                 : kNegInf,
                             s.logProb);

            std::unordered_map<size_t, State> rescued;
            for (auto it = m_beam.begin() + static_cast<long>(m_config.beamWidth);
                 it != m_beam.end(); ++it) {
                const size_t region = it->index / m_config.regionTokens;
                if (keptBest.count(region))
                    continue;  // that region already has a representative
                auto found = rescued.find(region);
                if (found == rescued.end() || it->logProb > found->second.logProb)
                    rescued[region] = *it;
            }
            for (const auto &entry : rescued)
                survivors.push_back(entry.second);
        }
        m_beam.swap(survivors);
    }

    // Normalize with log-sum-exp so log-probabilities stay bounded and
    // comparable across updates. Without this they drift toward -inf and the
    // confidence readings become meaningless.
    std::vector<double> logs;
    logs.reserve(m_beam.size());
    for (const State &s : m_beam) {
        logs.push_back(s.logProb);
    }
    const double total = logSumExp(logs);
    if (std::isfinite(total)) {
        for (State &s : m_beam) {
            s.logProb -= total;
        }
    }

    // Keep the beam ordered by script index; several routines below rely on it
    // for clustering and for deterministic behaviour.
    std::sort(m_beam.begin(), m_beam.end(),
              [](const State &a, const State &b) { return a.index < b.index; });
}

void Aligner::recomputeEstimate()
{
    if (m_beam.empty()) {
        m_confidence = 0.0;
        return;
    }

    // Winner = highest posterior single state.
    size_t bestIdx = m_beam.front().index;
    double bestLog = m_beam.front().logProb;
    for (const State &s : m_beam) {
        if (s.logProb > bestLog) {
            bestLog = s.logProb;
            bestIdx = s.index;
        }
    }

    // Posterior mass within massRadius of the winner. This is the design doc's
    // primary gate: a sharp peak means we know where we are, a diffuse posterior
    // means we do not, even if some single state happens to lead.
    double nearMass = 0.0;
    for (const State &s : m_beam) {
        const size_t diff =
            (s.index > bestIdx) ? s.index - bestIdx : bestIdx - s.index;
        if (diff <= m_config.massRadius) {
            nearMass += std::exp(s.logProb);
        }
    }

    m_bestIndex = bestIdx;
    m_confidence = std::min(1.0, std::max(0.0, nearMass));
}

// ---------------------------------------------------------------------------
// Whisper revision handling
// ---------------------------------------------------------------------------

void Aligner::saveCheckpoint(double audioTimeSec)
{
    Checkpoint cp;
    cp.audioTimeSec = audioTimeSec;
    cp.beam = m_beam;
    cp.consumedTail = m_consumedTail;
    cp.bestIndex = m_bestIndex;
    cp.confidence = m_confidence;
    cp.tokensPerSec = m_tokensPerSec;
    cp.lowConfidenceSince = m_lowConfidenceSince;
    cp.reacquireCandidate = m_reacquireCandidate;
    cp.reacquireWins = m_reacquireWins;
    cp.hasPosition = m_hasPosition;
    cp.lastAudioTimeSec = m_lastAudioTimeSec;

    m_checkpoints.push_back(std::move(cp));
    while (m_checkpoints.size() > kCheckpointLimit) {
        m_checkpoints.pop_front();
    }
}

bool Aligner::rollbackTo(double audioTimeSec)
{
    // Find the newest checkpoint taken at or before this audio time: that is
    // the state of the world before this audio was ever applied.
    // Small epsilon absorbs floating-point noise in whisper's timestamps.
    constexpr double kEpsilon = 1e-6;

    long found = -1;
    for (long i = static_cast<long>(m_checkpoints.size()) - 1; i >= 0; --i) {
        if (m_checkpoints[static_cast<size_t>(i)].audioTimeSec
            <= audioTimeSec + kEpsilon) {
            found = i;
            break;
        }
    }
    if (found < 0) {
        return false;
    }

    const Checkpoint &cp = m_checkpoints[static_cast<size_t>(found)];
    m_beam = cp.beam;
    m_consumedTail = cp.consumedTail;
    m_bestIndex = cp.bestIndex;
    m_confidence = cp.confidence;
    m_tokensPerSec = cp.tokensPerSec;
    m_lowConfidenceSince = cp.lowConfidenceSince;
    m_reacquireCandidate = cp.reacquireCandidate;
    m_reacquireWins = cp.reacquireWins;
    m_hasPosition = cp.hasPosition;
    m_lastAudioTimeSec = cp.lastAudioTimeSec;

    // Discard this checkpoint and everything after it: that timeline is being
    // replaced by the revised transcript we are about to apply.
    m_checkpoints.resize(static_cast<size_t>(found));
    return true;
}

size_t Aligner::trimOverlap(const std::vector<std::string> &incoming) const
{
    if (m_consumedTail.empty() || incoming.empty()) {
        return 0;
    }

    // Find the longest suffix of m_consumedTail that is also a prefix of
    // `incoming`, allowing a small number of mismatches. whisper repeats the
    // tail of the previous window but often rewords part of it, so an
    // exact-equality test would fail exactly when it matters most.
    const size_t maxOverlap = std::min(m_consumedTail.size(), incoming.size());

    size_t bestOverlap = 0;
    double bestQuality = 0.0;

    for (size_t len = maxOverlap; len >= 1; --len) {
        double matched = 0.0;
        for (size_t k = 0; k < len; ++k) {
            const std::string &a = m_consumedTail[m_consumedTail.size() - len + k];
            const std::string &b = incoming[k];
            if (a == b) {
                matched += 1.0;
            } else if (tokenSimilarity(a, b) >= m_config.similarityFloor) {
                matched += 0.75;  // reworded but recognizably the same word
            }
        }
        const double quality = matched / static_cast<double>(len);

        // Require most of the candidate overlap to agree. Longer confirmed
        // overlaps win, which is why we scan from the longest downward and
        // weight quality by length.
        if (quality >= 0.7) {
            const double score = quality * static_cast<double>(len);
            if (score > bestQuality) {
                bestQuality = score;
                bestOverlap = len;
            }
        }
        if (len == 1) {
            break;
        }
    }

    return bestOverlap;
}

// ---------------------------------------------------------------------------
// Reacquisition
// ---------------------------------------------------------------------------

double Aligner::scoreSpan(size_t spanStart,
                          const std::vector<std::string> &recentTokens) const
{
    if (m_script.empty() || recentTokens.empty()) {
        return 0.0;
    }

    // Score the recent tokens against the script starting at spanStart, using a
    // small local alignment that permits insertions and deletions on both
    // sides. This is the Smith-Waterman-style recovery the design doc keeps for
    // exactly this purpose, restricted to a short window.
    const size_t contextLen =
        std::min(m_config.reacquireContext,
                 m_script.tokenCount() - std::min(spanStart, m_script.tokenCount()));
    if (contextLen == 0) {
        return 0.0;
    }

    const size_t obsLen = std::min(recentTokens.size(), m_config.reacquireContext);
    const size_t obsStart = recentTokens.size() - obsLen;

    // Local alignment DP, tracking the best cell value anywhere in the matrix.
    std::vector<double> prev(contextLen + 1, 0.0);
    std::vector<double> curr(contextLen + 1, 0.0);
    double best = 0.0;

    constexpr double kGap = -0.5;

    for (size_t i = 1; i <= obsLen; ++i) {
        const std::string &obs = recentTokens[obsStart + i - 1];
        curr[0] = 0.0;
        for (size_t j = 1; j <= contextLen; ++j) {
            const size_t scriptIdx = spanStart + j - 1;
            const ScriptToken &tok = m_script.token(scriptIdx);

            double matchScore;
            if (obs == tok.normalized) {
                matchScore = tok.weight;
            } else {
                const double sim = tokenSimilarity(obs, tok.normalized);
                matchScore = (sim >= m_config.similarityFloor)
                                 ? tok.weight * sim * 0.8
                                 : -0.35;
            }

            const double diag = prev[j - 1] + matchScore;
            const double up = prev[j] + kGap;
            const double left = curr[j - 1] + kGap;
            curr[j] = std::max({0.0, diag, up, left});
            best = std::max(best, curr[j]);
        }
        prev.swap(curr);
    }

    return best;
}

bool Aligner::tryReacquire(const std::vector<std::string> &recentTokens,
                           double audioTimeSec)
{
    (void)audioTimeSec;

    if (m_script.empty() || recentTokens.empty()) {
        return false;
    }

    const std::vector<SpanCandidate> spans =
        m_script.findCandidateSpans(recentTokens, m_config.reacquireCandidates);
    if (spans.empty()) {
        m_reacquireWins = 0;
        return false;
    }

    // Rescore each retrieved span properly; index retrieval is recall-oriented
    // and its ranking is too crude to relocate on.
    size_t bestSpan = 0;
    double bestScore = kNegInf;
    double runnerUp = kNegInf;

    for (const SpanCandidate &span : spans) {
        // Score a window starting a little before the hit, since the retrieved
        // token is usually mid-phrase rather than at the phrase start.
        const size_t start =
            (span.tokenIndex > m_config.reacquireContext / 2)
                ? span.tokenIndex - m_config.reacquireContext / 2
                : 0;
        const double score = scoreSpan(start, recentTokens);
        if (score > bestScore) {
            runnerUp = bestScore;
            bestScore = score;
            bestSpan = start;
        } else if (score > runnerUp) {
            runnerUp = score;
        }
    }

    if (!std::isfinite(bestScore) || bestScore <= 0.0) {
        m_reacquireWins = 0;
        return false;
    }

    // The challenger must beat its nearest rival by a clear margin. If two
    // distant places in the script fit equally well — the repeated-phrase case —
    // neither wins, and we keep coasting instead of guessing.
    if (std::isfinite(runnerUp)
        && (bestScore - runnerUp) < m_config.reacquireMarginNats) {
        m_reacquireWins = 0;
        return false;
    }

    // Also require the winner to beat where we currently think we are; without
    // this we would relocate on evidence that our present position explains
    // just as well.
    if (m_hasPosition) {
        const size_t currentStart =
            (m_bestIndex > m_config.reacquireContext / 2)
                ? m_bestIndex - m_config.reacquireContext / 2
                : 0;
        const double currentScore = scoreSpan(currentStart, recentTokens);
        if (bestScore - currentScore < m_config.reacquireMarginNats) {
            m_reacquireWins = 0;
            return false;
        }
    }

    // TWO SUCCESSIVE WINS. A single strong update is not enough to relocate:
    // ASR noise produces convincing one-shot coincidences, and a wrong jump is
    // the worst failure mode this feature has.
    if (m_reacquireWins > 0 && m_reacquireCandidate == bestSpan) {
        ++m_reacquireWins;
    } else {
        m_reacquireCandidate = bestSpan;
        m_reacquireWins = 1;
    }

    if (m_reacquireWins < m_config.reacquireWinsRequired) {
        return false;
    }

    // Relocate. Seed a fresh local posterior over the winning span.
    m_beam.clear();
    const size_t center =
        std::min(bestSpan + m_config.reacquireContext / 2,
                 m_script.tokenCount() - 1);
    const size_t lo = (center > 6) ? center - 6 : 0;
    const size_t hi = std::min(center + 7, m_script.tokenCount());
    for (size_t i = lo; i < hi; ++i) {
        const double distance =
            static_cast<double>((i > center) ? i - center : center - i);
        m_beam.push_back(State{i, -0.6 * distance});
    }
    normalizeAndPrune();
    recomputeEstimate();

    m_hasPosition = true;
    m_reacquireWins = 0;
    m_lowConfidenceSince = -1.0;
    return true;
}

// ---------------------------------------------------------------------------
// update()
// ---------------------------------------------------------------------------

void Aligner::applyTokens(const std::vector<std::string> &tokens,
                          double audioTimeSec)
{
    m_lastInformativeWeight = 0.0;

    for (const std::string &raw : tokens) {
        const std::string norm = normalizeToken(raw);
        if (norm.empty()) {
            continue;
        }
        stepToken(norm);
        recomputeEstimate();

        // Track how much genuinely informative evidence this update carried.
        // A stream of function words must not be able to satisfy the confidence
        // gate on its own, no matter how peaked the posterior looks.
        if (!isFunctionWord(norm)) {
            m_lastInformativeWeight += m_script.weightForNormalized(norm);
        }

        m_consumedTail.push_back(norm);
    }

    if (m_consumedTail.size() > kConsumedTailLimit) {
        m_consumedTail.erase(
            m_consumedTail.begin(),
            m_consumedTail.begin()
                + static_cast<long>(m_consumedTail.size() - kConsumedTailLimit));
    }

    m_lastAudioTimeSec = audioTimeSec;
}

AlignResult Aligner::update(const Observation &observation)
{
    // Degenerate script: never claim to know anything.
    if (m_script.empty()) {
        return AlignResult{};
    }

    const size_t previousIndex = m_bestIndex;
    const bool hadPosition = m_hasPosition;
    const double previousAudioTime = m_lastAudioTimeSec;

    // --- Step 1: revision handling ------------------------------------------
    // An observation at or before audio time we already consumed is a REVISION
    // of that audio, not new speech. Roll back so the same audio cannot advance
    // the position twice.
    const bool isRevision =
        hadPosition && observation.audioTimeSec <= m_lastAudioTimeSec + 1e-6;
    if (isRevision) {
        rollbackTo(observation.audioTimeSec);
    }

    // --- Step 2: checkpoint --------------------------------------------------
    // Save the pre-application state so a later revision of THIS audio can undo
    // it. Only volatile hypotheses need this, but checkpointing stable ones too
    // is cheap and makes the rollback search simpler.
    saveCheckpoint(observation.audioTimeSec);

    // --- Step 3: normalize and trim the overlap -----------------------------
    std::vector<std::string> normalized;
    normalized.reserve(observation.tokens.size());
    for (const std::string &raw : observation.tokens) {
        std::string n = normalizeToken(raw);
        if (!n.empty()) {
            normalized.push_back(std::move(n));
        }
    }

    // Even a forward-moving whisper window repeats the tail of the previous
    // one. Drop the duplicated prefix so only new words are scored.
    const size_t skip = trimOverlap(normalized);
    std::vector<std::string> fresh(normalized.begin() + static_cast<long>(skip),
                                   normalized.end());

    // --- Step 4: silence ------------------------------------------------------
    if (fresh.empty()) {
        // No new speech. Decay confidence toward zero rather than holding a
        // stale high reading, but do NOT move the position.
        if (normalized.empty()) {
            m_confidence *= 0.85;
            if (m_lowConfidenceSince < 0.0) {
                m_lowConfidenceSince = observation.audioTimeSec;
            }
        }
        m_lastAudioTimeSec = std::max(m_lastAudioTimeSec, observation.audioTimeSec);
        return currentEstimate();
    }

    // --- Step 5: the HMM update ---------------------------------------------
    applyTokens(fresh, observation.audioTimeSec);
    recomputeEstimate();

    // --- Step 6: confidence gating -------------------------------------------
    // All three gates from the design doc must pass together.
    const bool massOk = m_confidence >= m_config.minPosteriorMass;
    const bool evidenceOk =
        m_lastInformativeWeight >= m_config.minInformativeWeight;

    // Margin between the winning cluster and the best cluster elsewhere. This
    // is what keeps a repeated phrase from being reported as confident: when
    // three copies of a sentence fit equally, the margin collapses.
    double bestLog = kNegInf;
    double bestOtherLog = kNegInf;
    for (const State &s : m_beam) {
        const size_t diff = (s.index > m_bestIndex) ? s.index - m_bestIndex
                                                    : m_bestIndex - s.index;
        if (diff <= m_config.massRadius) {
            bestLog = std::max(bestLog, s.logProb);
        } else {
            bestOtherLog = std::max(bestOtherLog, s.logProb);
        }
    }
    const bool marginOk = !std::isfinite(bestOtherLog)
        || (bestLog - bestOtherLog) >= m_config.minMarginNats;

    const bool confident = massOk && evidenceOk && marginOk;

    if (confident) {
        m_hasPosition = true;
        m_lowConfidenceSince = -1.0;
    } else if (m_lowConfidenceSince < 0.0) {
        m_lowConfidenceSince = observation.audioTimeSec;
    }

    // --- Step 7: global reacquisition ----------------------------------------
    // Only after confidence has been low for a sustained period, per the design
    // doc. A brief dip is normal and must not trigger a search.
    if (!confident && m_lowConfidenceSince >= 0.0
        && (observation.audioTimeSec - m_lowConfidenceSince)
               >= m_config.lowConfidenceSeconds) {
        if (tryReacquire(m_consumedTail, observation.audioTimeSec)) {
            m_lowConfidenceSince = -1.0;
        }
    }

    // --- Step 8: velocity estimate -------------------------------------------
    // Feed-forward term for the control loop. Derived from how far the position
    // moved over how much audio time elapsed, smoothed with an EMA.
    //
    // Per Decision 3 we do NOT trust whisper's per-token timestamps. We use only
    // the elapsed audio time between observations, and fall back to a nominal
    // reading rate when the measurement looks implausible.
    const double elapsed = observation.audioTimeSec - previousAudioTime;
    if (confident && hadPosition && elapsed > 0.05 && elapsed < 10.0
        && m_bestIndex >= previousIndex) {
        const double advanced =
            static_cast<double>(m_bestIndex - previousIndex);
        const double instantaneous = advanced / elapsed;
        // Reject implausible rates rather than letting one bad update poison
        // the filter: nobody reads at 20 tokens/sec.
        if (instantaneous >= 0.0 && instantaneous <= 12.0) {
            if (m_tokensPerSec <= 0.0) {
                m_tokensPerSec = instantaneous;
            } else {
                m_tokensPerSec = (1.0 - m_config.velocitySmoothing) * m_tokensPerSec
                    + m_config.velocitySmoothing * instantaneous;
            }
        }
    }

    AlignResult result = currentEstimate();
    result.confident = confident;

    // Advance the high-water mark on a trusted fix. A doubtful one must not move
    // it: if it were wrong, the penalty would then defend a position the speaker
    // never reached and fight every correct match after it.
    if (confident)
        m_spokenThrough = std::max(m_spokenThrough, result.tokenIndex);

    return result;
}

AlignResult Aligner::currentEstimate() const
{
    AlignResult result;
    if (m_script.empty()) {
        return result;
    }
    result.tokenIndex = m_bestIndex;
    result.charOffset = m_script.charOffsetForToken(m_bestIndex);
    result.confidence = m_confidence;
    result.estimatedTokensPerSec =
        (m_tokensPerSec > 0.0) ? m_tokensPerSec : m_config.defaultTokensPerSec;
    result.confident = m_hasPosition
        && m_confidence >= m_config.minPosteriorMass;
    return result;
}

}  // namespace voicetracking
}  // namespace qprompt
