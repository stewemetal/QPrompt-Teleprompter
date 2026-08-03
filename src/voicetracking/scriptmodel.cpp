// SPDX-FileCopyrightText: QPrompt voice tracking
// SPDX-License-Identifier: GPL-3.0-or-later

#include "scriptmodel.h"

#include "textnormalize.h"

#include <algorithm>
#include <cmath>

namespace qprompt {
namespace voicetracking {

namespace {

// --- Tuning constants -------------------------------------------------------

// A token occurring more often than this is not worth putting in the unigram
// index: retrieving every "the" in a long script costs time and buys nothing.
// Shingles remain indexed regardless, since a repeated 3-gram is still a useful
// locator even when its individual words are common.
constexpr size_t kMaxIndexedOccurrences = 12;

// Function words are clamped to at most this weight no matter how rare they
// happen to be in a short script. In a 20-word script "the" might occur once
// and look rare by pure IDF, but it is still not evidence of position.
constexpr double kFunctionWordWeightCap = 0.25;

// Floor on any token's weight, so a very common content word still contributes
// something rather than dropping out of the likelihood entirely.
constexpr double kMinWeight = 0.15;

// Shingle sizes built into the index.
constexpr size_t kMinShingle = 2;
constexpr size_t kMaxShingle = 3;

// Retrieval scoring: a matching shingle is worth more than a matching unigram
// because word order agreement is much stronger evidence of location.
constexpr double kShingleScoreMultiplier = 2.5;

// Candidate spans closer together than this are merged into one, so a single
// true location does not occupy every slot in the result list.
constexpr size_t kCandidateMergeDistance = 8;

}  // namespace

void ScriptModel::clear()
{
    m_text.clear();
    m_tokens.clear();
    m_documentFrequency.clear();
    m_unigramIndex.clear();
    m_shingleIndex.clear();
    m_maxWeight = 1.0;
}

void ScriptModel::build(const std::string &scriptText)
{
    clear();
    m_text = scriptText;

    // Tokenize, preserving exact byte offsets into m_text.
    const std::vector<RawToken> raw = tokenize(m_text);
    m_tokens.reserve(raw.size());
    for (const RawToken &r : raw) {
        ScriptToken tok;
        tok.normalized = r.normalized;
        tok.startOffset = r.startOffset;
        tok.endOffset = r.endOffset;
        tok.functionWord = isFunctionWord(r.normalized);
        m_tokens.push_back(std::move(tok));
    }

    if (m_tokens.empty()) {
        return;
    }

    // Count occurrences.
    for (const ScriptToken &tok : m_tokens) {
        ++m_documentFrequency[tok.normalized];
    }

    // IDF-style weight.
    //
    // Standard IDF is log(N / df). Here N is the token count of this single
    // script rather than a corpus size, which is the right choice: we care how
    // distinctive a word is WITHIN this script, because that is exactly what
    // makes it useful for locating a position in it. A word appearing once in a
    // 500-token script pins a location; a word appearing 40 times does not.
    //
    // The +1 smoothing keeps the weight finite and positive for a token that
    // occupies the entire script.
    const double totalTokens = static_cast<double>(m_tokens.size());
    m_maxWeight = kMinWeight;
    for (ScriptToken &tok : m_tokens) {
        const double df =
            static_cast<double>(m_documentFrequency[tok.normalized]);
        double w = std::log((totalTokens + 1.0) / (df + 1.0)) + kMinWeight;

        // Function words are capped regardless of measured rarity.
        if (tok.functionWord) {
            w = std::min(w, kFunctionWordWeightCap);
        }
        // Very short tokens carry little information and are frequently ASR
        // noise, so they are also capped.
        if (utf8Length(tok.normalized) <= 2) {
            w = std::min(w, kFunctionWordWeightCap * 1.5);
        }

        tok.weight = std::max(w, kMinWeight);
        m_maxWeight = std::max(m_maxWeight, tok.weight);
    }

    buildIndex();
}

void ScriptModel::buildIndex()
{
    // Unigram index over tokens rare enough to be worth retrieving.
    for (size_t i = 0; i < m_tokens.size(); ++i) {
        const ScriptToken &tok = m_tokens[i];
        if (tok.functionWord) {
            continue;
        }
        const size_t df = m_documentFrequency[tok.normalized];
        if (df > kMaxIndexedOccurrences) {
            continue;
        }
        m_unigramIndex[tok.normalized].push_back(i);
    }

    // Shingle index over 2- and 3-token windows. Shingles are indexed even when
    // their component words are common: "in the beginning" locates a position
    // that none of its three words could locate alone.
    std::vector<std::string> norm;
    norm.reserve(m_tokens.size());
    for (const ScriptToken &tok : m_tokens) {
        norm.push_back(tok.normalized);
    }

    for (size_t n = kMinShingle; n <= kMaxShingle; ++n) {
        if (m_tokens.size() < n) {
            break;
        }
        for (size_t i = 0; i + n <= m_tokens.size(); ++i) {
            m_shingleIndex[shingleKey(norm, i, n)].push_back(i);
        }
    }
}

std::string ScriptModel::shingleKey(const std::vector<std::string> &toks,
                                    size_t start, size_t n)
{
    std::string key;
    for (size_t k = 0; k < n; ++k) {
        if (k > 0) {
            key.push_back('\x1f');  // unit separator, cannot occur in a token
        }
        key += toks[start + k];
    }
    return key;
}

const ScriptToken &ScriptModel::token(size_t index) const
{
    static const ScriptToken kEmpty{};
    if (index >= m_tokens.size()) {
        return kEmpty;
    }
    return m_tokens[index];
}

size_t ScriptModel::charOffsetForToken(size_t tokenIndex) const
{
    if (m_tokens.empty()) {
        return 0;
    }
    if (tokenIndex >= m_tokens.size()) {
        // Past the end: report the end of the final token so a caller that
        // scrolls to this offset lands at the bottom of the script.
        return m_tokens.back().endOffset;
    }
    return m_tokens[tokenIndex].startOffset;
}

size_t ScriptModel::tokenIndexForCharOffset(size_t charOffset) const
{
    if (m_tokens.empty()) {
        return 0;
    }
    // Binary search for the first token starting at or after charOffset.
    size_t lo = 0;
    size_t hi = m_tokens.size();
    while (lo < hi) {
        const size_t mid = lo + (hi - lo) / 2;
        if (m_tokens[mid].startOffset < charOffset) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return lo;
}

double ScriptModel::weightForNormalized(const std::string &normalized) const
{
    auto it = m_documentFrequency.find(normalized);
    if (it == m_documentFrequency.end()) {
        // Not in the script at all. An unknown word is not evidence for any
        // position, but it is also not evidence against one; the aligner treats
        // it via the background model. Returning the max weight here would be
        // wrong, so return a neutral mid weight.
        return std::max(kMinWeight, m_maxWeight * 0.5);
    }
    if (isFunctionWord(normalized)) {
        return std::min(kFunctionWordWeightCap,
                        std::max(kMinWeight, m_maxWeight));
    }
    const double totalTokens = static_cast<double>(m_tokens.size());
    const double df = static_cast<double>(it->second);
    const double w = std::log((totalTokens + 1.0) / (df + 1.0)) + kMinWeight;
    return std::max(w, kMinWeight);
}

std::vector<SpanCandidate> ScriptModel::findCandidateSpans(
    const std::vector<std::string> &observedTokens, size_t maxCandidates) const
{
    std::vector<SpanCandidate> result;
    if (m_tokens.empty() || observedTokens.empty() || maxCandidates == 0) {
        return result;
    }

    // Accumulate evidence per script token index.
    std::unordered_map<size_t, double> scores;

    // --- Unigram evidence ---------------------------------------------------
    for (const std::string &obs : observedTokens) {
        if (obs.empty() || isFunctionWord(obs)) {
            continue;  // function words cannot locate anything
        }
        auto it = m_unigramIndex.find(obs);
        if (it == m_unigramIndex.end()) {
            continue;
        }
        const double weight = weightForNormalized(obs);
        // A word occurring in many places gives each of those places weaker
        // evidence; divide by the number of hits so a common word cannot
        // out-vote a genuinely rare one.
        const double perHit = weight / static_cast<double>(it->second.size());
        for (size_t idx : it->second) {
            scores[idx] += perHit;
        }
    }

    // --- Shingle evidence ---------------------------------------------------
    for (size_t n = kMinShingle; n <= kMaxShingle; ++n) {
        if (observedTokens.size() < n) {
            break;
        }
        for (size_t i = 0; i + n <= observedTokens.size(); ++i) {
            bool allFunction = true;
            for (size_t k = 0; k < n; ++k) {
                if (!isFunctionWord(observedTokens[i + k])) {
                    allFunction = false;
                    break;
                }
            }
            if (allFunction) {
                continue;
            }

            const std::string key = shingleKey(observedTokens, i, n);
            auto it = m_shingleIndex.find(key);
            if (it == m_shingleIndex.end()) {
                continue;
            }

            // Weight the shingle by the summed rarity of its members, scaled up
            // because ordered agreement is much stronger evidence than a bag of
            // words, and divided across its occurrences as above.
            double shingleWeight = 0.0;
            for (size_t k = 0; k < n; ++k) {
                shingleWeight += weightForNormalized(observedTokens[i + k]);
            }
            shingleWeight *= kShingleScoreMultiplier;
            const double perHit =
                shingleWeight / static_cast<double>(it->second.size());

            for (size_t idx : it->second) {
                scores[idx] += perHit;
            }
        }
    }

    if (scores.empty()) {
        return result;
    }

    // Flatten and sort by descending score.
    std::vector<SpanCandidate> all;
    all.reserve(scores.size());
    for (const auto &kv : scores) {
        all.push_back(SpanCandidate{kv.first, kv.second});
    }
    std::sort(all.begin(), all.end(),
              [](const SpanCandidate &a, const SpanCandidate &b) {
                  if (a.score != b.score) {
                      return a.score > b.score;
                  }
                  return a.tokenIndex < b.tokenIndex;  // deterministic ties
              });

    // Merge nearby candidates: neighbouring tokens of one true location would
    // otherwise fill every slot and crowd out genuinely distinct alternatives,
    // which is exactly the repeated-phrase case we must keep visible.
    for (const SpanCandidate &cand : all) {
        bool merged = false;
        for (const SpanCandidate &kept : result) {
            const size_t diff = (cand.tokenIndex > kept.tokenIndex)
                                    ? cand.tokenIndex - kept.tokenIndex
                                    : kept.tokenIndex - cand.tokenIndex;
            if (diff <= kCandidateMergeDistance) {
                merged = true;
                break;
            }
        }
        if (!merged) {
            result.push_back(cand);
            if (result.size() >= maxCandidates) {
                break;
            }
        }
    }

    return result;
}

}  // namespace voicetracking
}  // namespace qprompt
