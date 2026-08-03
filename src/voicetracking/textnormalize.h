// SPDX-FileCopyrightText: QPrompt voice tracking
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Pure C++17, no Qt, no external dependencies, UTF-8 throughout.
//
// This header holds the lowest layer of the voice-tracking engine: turning raw
// UTF-8 text (script text or ASR transcript text) into comparable tokens, and
// scoring how similar two tokens are.
//
// Everything here is deliberately free of Qt types so the alignment engine can
// be unit tested without a GUI or an event loop.

#ifndef QPROMPT_VOICETRACKING_TEXTNORMALIZE_H
#define QPROMPT_VOICETRACKING_TEXTNORMALIZE_H

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace qprompt {
namespace voicetracking {

// ---------------------------------------------------------------------------
// UTF-8 primitives
// ---------------------------------------------------------------------------

// Decodes the UTF-8 sequence starting at `bytes[pos]`. On success returns the
// code point and advances `pos` past the sequence. On malformed input, returns
// U+FFFD and advances exactly one byte, so callers always make progress and
// never loop forever on bad data. Never throws.
uint32_t utf8Decode(const std::string &bytes, size_t &pos);

// Appends `cp` to `out` as UTF-8. Code points above U+10FFFF and surrogates are
// written as U+FFFD.
void utf8Append(std::string &out, uint32_t cp);

// Number of code points in a UTF-8 string (not bytes).
size_t utf8Length(const std::string &bytes);

// Decodes an entire UTF-8 string into code points.
std::vector<uint32_t> utf8ToCodepoints(const std::string &bytes);

// ---------------------------------------------------------------------------
// Case folding and diacritics
// ---------------------------------------------------------------------------

// Lowercases a single code point. Covers ASCII, Latin-1 Supplement,
// Latin Extended-A and Latin Extended-B ranges relevant to European scripts,
// which includes the full Hungarian alphabet:
//     a b c d e f g h i j k l m n o p q r s t u v w x y z
//     á é í ó ö ő ú ü ű  (and A-Z, Á É Í Ó Ö Ő Ú Ü Ű)
// The two Hungarian-specific letters ő (U+0151) and ű (U+0171) live in Latin
// Extended-A and are handled by the even/odd pairing rule there.
uint32_t foldCaseCodepoint(uint32_t cp);

// UTF-8 aware lowercase of a whole string.
std::string toLowerUtf8(const std::string &text);

// Maps an accented Latin code point to its unaccented base ('á' -> 'a',
// 'ő' -> 'o', 'ű' -> 'u'). Code points with no mapping are returned unchanged.
uint32_t foldDiacriticCodepoint(uint32_t cp);

// Strips diacritics from a whole string.
//
// IMPORTANT: this is used ONLY as a fallback similarity signal, never as the
// primary matching key. In Hungarian the difference between 'ő' and 'ö' (and
// between 'ú'/'ü'/'ű') is a real phonemic distinction that separates distinct
// words, so folding it away as the primary key would create false matches.
// See tokenSimilarity() for how the fallback is weighted down.
std::string foldDiacritics(const std::string &text);

// ---------------------------------------------------------------------------
// Classification
// ---------------------------------------------------------------------------

// True for code points that carry lexical content: letters and digits. Used to
// decide what to strip as punctuation.
bool isWordCodepoint(uint32_t cp);

// True for space, tab, newline and the Unicode separator code points we care
// about (NBSP, various spaces, line/paragraph separators).
bool isSpaceCodepoint(uint32_t cp);

// ---------------------------------------------------------------------------
// Token normalization
// ---------------------------------------------------------------------------

// Normalizes a single token: UTF-8 lowercase, then drop every code point that
// is not a word code point. Internal apostrophes and hyphens are dropped rather
// than split, so "don't" -> "dont" and "well-known" -> "wellknown". This keeps
// ASR output (which is inconsistent about both) comparable with script text.
//
// Returns an empty string for tokens that are pure punctuation.
std::string normalizeToken(const std::string &token);

// A token produced by tokenization, carrying both its normalized form and the
// exact byte offsets of the ORIGINAL substring it came from.
//
// The offsets are half-open [startOffset, endOffset) byte indices into the
// string that was passed to tokenize(). They are how a caller maps a matched
// token back to a scroll position, so they must be exact: they span the raw
// token including any attached punctuation that normalization removed.
struct RawToken {
    std::string normalized;  // may be empty if the token was pure punctuation
    size_t startOffset = 0;  // byte offset into the source string, inclusive
    size_t endOffset = 0;    // byte offset into the source string, exclusive
};

// Splits UTF-8 `text` on whitespace and normalizes each piece. Tokens whose
// normalized form is empty (pure punctuation, e.g. a lone em dash) are dropped
// from the result, so every returned token has a non-empty `normalized`.
//
// Collapsing runs of whitespace is implicit: consecutive separators simply
// produce no token between them.
std::vector<RawToken> tokenize(const std::string &text);

// Convenience: normalized strings only, offsets discarded. Used for transcript
// text, where offsets into the ASR string are meaningless.
std::vector<std::string> tokenizeToStrings(const std::string &text);

// ---------------------------------------------------------------------------
// Similarity
// ---------------------------------------------------------------------------

// Levenshtein edit distance measured in CODE POINTS, not bytes. Measuring in
// bytes would make a one-letter difference between 'ház' and 'hát' cost the
// same as a two-letter difference in ASCII, because the accented letters are
// multi-byte. Capped internally for very long inputs.
size_t editDistanceCodepoints(const std::string &a, const std::string &b);

// Number of leading code points that `a` and `b` share.
size_t longestCommonPrefixCodepoints(const std::string &a, const std::string &b);

// Similarity of two normalized tokens in [0, 1]. 1.0 means identical.
//
// FORMULA AND RATIONALE
// ---------------------
// The naive approach — a flat "Levenshtein ratio > 0.8" threshold — is wrong in
// both directions, and the design doc calls this out explicitly:
//
//   * For SHORT words it is too strict. In a 3-character word a single
//     character error gives a ratio of 0.67 and fails, yet a 1-character ASR
//     error on a short word is extremely common and usually still the right
//     word.
//   * For LONG words it is too loose. Two unrelated 12-character words can
//     easily share enough characters to clear 0.8 by accident.
//
// So the edit-distance component here is LENGTH-AWARE: instead of comparing a
// fixed ratio, we compare the observed distance against an allowance that grows
// sublinearly with word length, roughly one permitted edit per 4 code points
// plus a constant. Short words get their one free error; long words are held to
// a proportionally tighter standard than a flat ratio would impose.
//
// The second and, for Hungarian, more important component is a LONGEST COMMON
// PREFIX ratio. Hungarian is agglutinative: suffixes stack onto an invariant
// stem, so the same lexical word appears in the script and the transcript in
// different surface forms.
//
//     ház        (house)
//     házban     (in the house)
//     házaimban  (in my houses)
//
// Whole-form matching treats "ház" and "házaimban" as almost unrelated — the
// edit distance is 6 on a 3-character stem, which no ratio test will accept.
// But they share a complete 3-code-point prefix, and that prefix IS the word.
// This is why prefix weight dominates the formula: the design doc's open item 3
// proposed exactly this ("match on a longest-common-prefix ratio rather than
// full form"), and it is the single change that makes Hungarian tracking work.
//
// The prefix score is the shared prefix length divided by the length of the
// SHORTER token, then damped by a mild penalty for large length mismatch. Using
// the shorter token as the denominator is what lets a bare stem in the script
// match a heavily suffixed form in the transcript. The damping stops a
// 2-character prefix from making "a" and "abszolút" look identical.
//
// A prefix shorter than 3 code points earns no prefix credit at all, because in
// both languages a 1-2 letter agreement is far too common to be evidence.
//
// The final score takes the MAXIMUM of the two components rather than a
// weighted sum, since they capture different, individually sufficient kinds of
// evidence: "this is the same word, misheard" and "this is the same stem,
// differently inflected".
//
// A diacritic-folded retry is applied last, and only as a FALLBACK, scaled down
// by kDiacriticFallbackScale. ASR does confuse ö/ő, but so do genuinely
// different Hungarian words, so a diacritic-only agreement is weaker evidence
// than an exact one and must never score as high.
double tokenSimilarity(const std::string &a, const std::string &b);

// ---------------------------------------------------------------------------
// Function words
// ---------------------------------------------------------------------------

// True if `normalizedToken` is a very common function word in English or
// Hungarian (articles, conjunctions, pronouns, auxiliaries, common
// postpositions). These carry little positional information: matching "a" or
// "the" tells you almost nothing about WHERE in a script you are.
//
// The alignment engine uses this as a floor on the IDF weight so that a passage
// made mostly of function words cannot manufacture confidence on its own.
bool isFunctionWord(const std::string &normalizedToken);

}  // namespace voicetracking
}  // namespace qprompt

#endif  // QPROMPT_VOICETRACKING_TEXTNORMALIZE_H
