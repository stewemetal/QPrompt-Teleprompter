// SPDX-FileCopyrightText: QPrompt voice tracking
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Pure C++17, no Qt, no external dependencies.
//
// ScriptModel turns the teleprompter's plain-text script into the immutable
// structure the aligner searches: a token sequence with exact character offsets
// back into the original text, a rarity weight per token, and an inverted index
// used for global reacquisition after tracking is lost.

#ifndef QPROMPT_VOICETRACKING_SCRIPTMODEL_H
#define QPROMPT_VOICETRACKING_SCRIPTMODEL_H

#include <cstddef>
#include <string>
#include <unordered_map>
#include <vector>

namespace qprompt {
namespace voicetracking {

// One script token.
struct ScriptToken {
    // Normalized form (lowercased, punctuation stripped). Never empty.
    std::string normalized;

    // Half-open byte offsets [startOffset, endOffset) into the ORIGINAL script
    // string passed to build(). These are the contract with the caller: they
    // map a token back to a cursor position and therefore to a scroll Y.
    // They must be exact, and they refer to the raw token including any
    // punctuation that normalization removed.
    size_t startOffset = 0;
    size_t endOffset = 0;

    // IDF-style rarity weight. Rare tokens are informative about position;
    // function words are not. See ScriptModel::build() for the formula.
    double weight = 1.0;

    // True if this token is a known English/Hungarian function word. Cached to
    // avoid a hash lookup per token per beam state in the inner loop.
    bool functionWord = false;
};

// A candidate span returned by global reacquisition search.
struct SpanCandidate {
    size_t tokenIndex = 0;  // start of the retrieved span
    double score = 0.0;     // retrieval score, higher is better
};

class ScriptModel {
public:
    ScriptModel() = default;

    // Builds the model from UTF-8 plain text. Replaces any previous content.
    // Never throws; an empty or punctuation-only script yields an empty model,
    // which the aligner handles by reporting "not confident" forever.
    void build(const std::string &scriptText);

    // Clears all state back to an empty model.
    void clear();

    bool empty() const { return m_tokens.empty(); }
    size_t tokenCount() const { return m_tokens.size(); }

    const std::vector<ScriptToken> &tokens() const { return m_tokens; }

    // Bounds-checked accessors. token(i) on an out-of-range index returns a
    // shared empty token rather than throwing, keeping the no-exceptions
    // contract of the public API.
    const ScriptToken &token(size_t index) const;

    // The original text this model was built from.
    const std::string &text() const { return m_text; }

    // Character (byte) offset of a token's start. Returns 0 for an empty model
    // and clamps an out-of-range index to the last token's end offset.
    size_t charOffsetForToken(size_t tokenIndex) const;

    // Inverse mapping: the index of the token that starts at or after
    // `charOffset`. Used by reanchor() when the user scrolls manually.
    // Returns tokenCount() if the offset is past the last token.
    size_t tokenIndexForCharOffset(size_t charOffset) const;

    // Rarity weight of a normalized token as it appears in THIS script.
    // Unknown tokens (present in the transcript but not the script) get the
    // maximum weight, since an unknown word is not evidence for any position.
    double weightForNormalized(const std::string &normalized) const;

    // --- Global reacquisition ------------------------------------------------

    // Retrieves candidate script positions matching `observedTokens`, using the
    // inverted index over rare unigrams and 2-3 token shingles.
    //
    // Returns at most `maxCandidates` spans ordered by descending score. Only
    // spans supported by genuinely informative evidence are returned; if the
    // observation is all function words the result is empty, because such an
    // observation cannot identify a location.
    std::vector<SpanCandidate> findCandidateSpans(
        const std::vector<std::string> &observedTokens,
        size_t maxCandidates) const;

private:
    // Builds the inverted index after m_tokens is populated.
    void buildIndex();

    // Joins tokens [start, start+n) with '\x1f' into a shingle key.
    static std::string shingleKey(const std::vector<std::string> &toks,
                                  size_t start, size_t n);

    std::string m_text;
    std::vector<ScriptToken> m_tokens;

    // normalized token -> number of occurrences in the script.
    std::unordered_map<std::string, size_t> m_documentFrequency;

    // normalized token -> token indices where it occurs. Only populated for
    // tokens rare enough to be worth indexing (see kMaxIndexedOccurrences).
    std::unordered_map<std::string, std::vector<size_t>> m_unigramIndex;

    // 2- and 3-token shingle key -> token indices where the shingle starts.
    std::unordered_map<std::string, std::vector<size_t>> m_shingleIndex;

    double m_maxWeight = 1.0;
};

}  // namespace voicetracking
}  // namespace qprompt

#endif  // QPROMPT_VOICETRACKING_SCRIPTMODEL_H
