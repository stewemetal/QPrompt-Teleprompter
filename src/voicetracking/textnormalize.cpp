// SPDX-FileCopyrightText: QPrompt voice tracking
// SPDX-License-Identifier: GPL-3.0-or-later

#include "textnormalize.h"

#include <algorithm>
#include <cmath>
#include <unordered_set>

namespace qprompt {
namespace voicetracking {

namespace {

constexpr uint32_t kReplacementChar = 0xFFFDu;

// --- Similarity tuning constants -------------------------------------------
// These are starting values chosen by reasoning about the failure modes, not by
// fitting to data. They are expected to need field tuning against real ASR
// output; see the README for what each one trades off.

// Edit-distance allowance: permitted edits = kEditBaseAllowance +
// len / kEditLengthDivisor. With base 1.0 and divisor 4.0 a 3-letter word may
// differ by 1.75 edits before scoring zero, while a 12-letter word may differ
// by 4. This is the "length-aware, not a flat ratio" requirement.
constexpr double kEditBaseAllowance = 1.0;
constexpr double kEditLengthDivisor = 4.0;

// Shared prefixes shorter than this earn no credit. Two tokens agreeing on one
// or two leading letters is noise, not evidence, in both target languages.
constexpr size_t kMinPrefixCodepoints = 3;

// A diacritic-only agreement scores at most this fraction of an exact match.
// Below 1.0 because ö/ő and ú/ü/ű separate real Hungarian words.
constexpr double kDiacriticFallbackScale = 0.88;

// Length-mismatch damping on the prefix score. A shared stem is strong evidence
// but weakens as the two surface forms diverge in length; without this a short
// function word would match every long word starting with the same letters.
constexpr double kPrefixLengthMismatchDamping = 0.55;

// Function words for English and Hungarian. Hungarian entries are stored
// already normalized (lowercase, punctuation stripped). The list covers
// articles, the copula, common conjunctions, pronouns, demonstratives and the
// high-frequency postpositions that behave like English prepositions.
const std::unordered_set<std::string> &functionWordSet()
{
    static const std::unordered_set<std::string> words = {
        // --- English ---
        "a", "an", "the", "and", "or", "but", "if", "then", "than", "that",
        "this", "these", "those", "of", "to", "in", "on", "at", "by", "for",
        "with", "from", "as", "is", "are", "was", "were", "be", "been", "being",
        "am", "do", "does", "did", "have", "has", "had", "will", "would",
        "shall", "should", "can", "could", "may", "might", "must", "not", "no",
        "so", "it", "its", "he", "she", "they", "them", "we", "us", "you",
        "your", "his", "her", "their", "our", "my", "me", "i", "him", "there",
        "here", "what", "which", "who", "when", "where", "how", "all", "any",
        "some", "just", "very", "too", "also", "into", "out", "up", "down",
        "over", "about", "after", "before", "more", "most", "such", "own",
        // --- Hungarian ---
        "a", "az", "egy", "es", "és", "vagy", "de", "hogy", "mert", "ha",
        "akkor", "mint", "is", "nem", "igen", "van", "vannak", "volt", "voltak",
        "lesz", "lenne", "legyen", "kell", "lehet", "ez", "az", "ezek", "azok",
        "ezt", "azt", "itt", "ott", "ide", "oda", "en", "én", "te", "o", "ő",
        "mi", "ti", "ok", "ők", "engem", "teged", "téged", "minket", "titeket",
        "meg", "még", "mar", "már", "csak", "nagyon", "is", "sem", "se",
        "pedig", "tehat", "tehát", "azonban", "viszont", "ugy", "úgy", "igy",
        "így", "ami", "aki", "amely", "amikor", "ahol", "ahogy", "minden",
        "valami", "valaki", "semmi", "senki", "sok", "kevés", "keves", "ban",
        "ben", "ra", "re", "rol", "ról", "bol", "ből", "tol", "tól", "nak",
        "nek", "val", "vel", "ert", "ért", "ig", "hoz", "hez", "hoz",
        "utan", "után", "elott", "előtt", "alatt", "folott", "fölött", "kozott",
        "között", "nelkul", "nélkül", "szerint", "miatt", "altal", "által",
    };
    return words;
}

}  // namespace

// ---------------------------------------------------------------------------
// UTF-8 primitives
// ---------------------------------------------------------------------------

uint32_t utf8Decode(const std::string &bytes, size_t &pos)
{
    if (pos >= bytes.size()) {
        return kReplacementChar;
    }

    const auto b0 = static_cast<unsigned char>(bytes[pos]);

    // 1-byte sequence: 0xxxxxxx
    if (b0 < 0x80u) {
        ++pos;
        return b0;
    }

    // Determine the sequence length from the leading byte.
    size_t extraBytes = 0;
    uint32_t cp = 0;
    if ((b0 & 0xE0u) == 0xC0u) {
        extraBytes = 1;
        cp = b0 & 0x1Fu;
    } else if ((b0 & 0xF0u) == 0xE0u) {
        extraBytes = 2;
        cp = b0 & 0x0Fu;
    } else if ((b0 & 0xF8u) == 0xF0u) {
        extraBytes = 3;
        cp = b0 & 0x07u;
    } else {
        // Continuation byte or invalid leader in leading position.
        ++pos;
        return kReplacementChar;
    }

    if (pos + extraBytes >= bytes.size()) {
        // Truncated sequence at end of string.
        ++pos;
        return kReplacementChar;
    }

    for (size_t k = 1; k <= extraBytes; ++k) {
        const auto bk = static_cast<unsigned char>(bytes[pos + k]);
        if ((bk & 0xC0u) != 0x80u) {
            // Invalid continuation: consume only the leader so we resynchronize
            // on the next byte instead of swallowing valid characters.
            ++pos;
            return kReplacementChar;
        }
        cp = (cp << 6) | (bk & 0x3Fu);
    }

    pos += extraBytes + 1;

    // Reject overlong encodings, surrogates and out-of-range values.
    if (cp > 0x10FFFFu || (cp >= 0xD800u && cp <= 0xDFFFu)) {
        return kReplacementChar;
    }
    return cp;
}

void utf8Append(std::string &out, uint32_t cp)
{
    if (cp > 0x10FFFFu || (cp >= 0xD800u && cp <= 0xDFFFu)) {
        cp = kReplacementChar;
    }
    if (cp < 0x80u) {
        out.push_back(static_cast<char>(cp));
    } else if (cp < 0x800u) {
        out.push_back(static_cast<char>(0xC0u | (cp >> 6)));
        out.push_back(static_cast<char>(0x80u | (cp & 0x3Fu)));
    } else if (cp < 0x10000u) {
        out.push_back(static_cast<char>(0xE0u | (cp >> 12)));
        out.push_back(static_cast<char>(0x80u | ((cp >> 6) & 0x3Fu)));
        out.push_back(static_cast<char>(0x80u | (cp & 0x3Fu)));
    } else {
        out.push_back(static_cast<char>(0xF0u | (cp >> 18)));
        out.push_back(static_cast<char>(0x80u | ((cp >> 12) & 0x3Fu)));
        out.push_back(static_cast<char>(0x80u | ((cp >> 6) & 0x3Fu)));
        out.push_back(static_cast<char>(0x80u | (cp & 0x3Fu)));
    }
}

size_t utf8Length(const std::string &bytes)
{
    size_t pos = 0;
    size_t count = 0;
    while (pos < bytes.size()) {
        utf8Decode(bytes, pos);
        ++count;
    }
    return count;
}

std::vector<uint32_t> utf8ToCodepoints(const std::string &bytes)
{
    std::vector<uint32_t> out;
    out.reserve(bytes.size());
    size_t pos = 0;
    while (pos < bytes.size()) {
        out.push_back(utf8Decode(bytes, pos));
    }
    return out;
}

// ---------------------------------------------------------------------------
// Case folding
// ---------------------------------------------------------------------------

uint32_t foldCaseCodepoint(uint32_t cp)
{
    // ASCII
    if (cp >= 'A' && cp <= 'Z') {
        return cp + 32;
    }
    if (cp < 0x80u) {
        return cp;
    }

    // Latin-1 Supplement: U+00C0..U+00DE lowercase to +0x20, with U+00D7
    // (multiplication sign) excluded because it is not a letter.
    // This range covers Hungarian Á É Í Ó Ö Ú Ü.
    if (cp >= 0x00C0u && cp <= 0x00DEu && cp != 0x00D7u) {
        return cp + 0x20u;
    }

    // Latin Extended-A: U+0100..U+017F. Almost all of this block is arranged as
    // consecutive (uppercase, lowercase) pairs where the uppercase code point is
    // even. This is the rule that gives us Hungarian Ő (U+0150 -> U+0151) and
    // Ű (U+0170 -> U+0171) without a lookup table.
    if (cp >= 0x0100u && cp <= 0x0137u) {
        return (cp % 2 == 0) ? cp + 1 : cp;
    }
    // U+0139..U+0148 pairs are offset by one (odd uppercase).
    if (cp >= 0x0139u && cp <= 0x0148u) {
        return (cp % 2 == 1) ? cp + 1 : cp;
    }
    // U+014A..U+0177 returns to even-uppercase pairing. Ő and Ű live here.
    if (cp >= 0x014Au && cp <= 0x0177u) {
        return (cp % 2 == 0) ? cp + 1 : cp;
    }
    // U+0178 (Ÿ) maps back down to U+00FF.
    if (cp == 0x0178u) {
        return 0x00FFu;
    }
    // U+0179..U+017E pairs are offset by one (odd uppercase).
    if (cp >= 0x0179u && cp <= 0x017Eu) {
        return (cp % 2 == 1) ? cp + 1 : cp;
    }

    // Greek and Cyrillic basic ranges, included so that stray non-Latin text
    // does not case-fold inconsistently. Not required for EN/HU.
    if (cp >= 0x0391u && cp <= 0x03A9u && cp != 0x03A2u) {
        return cp + 0x20u;
    }
    if (cp >= 0x0410u && cp <= 0x042Fu) {
        return cp + 0x20u;
    }
    if (cp >= 0x0400u && cp <= 0x040Fu) {
        return cp + 0x50u;
    }

    return cp;
}

std::string toLowerUtf8(const std::string &text)
{
    std::string out;
    out.reserve(text.size());
    size_t pos = 0;
    while (pos < text.size()) {
        utf8Append(out, foldCaseCodepoint(utf8Decode(text, pos)));
    }
    return out;
}

// ---------------------------------------------------------------------------
// Diacritics
// ---------------------------------------------------------------------------

uint32_t foldDiacriticCodepoint(uint32_t cp)
{
    if (cp < 0x80u) {
        return cp;
    }

    // Latin-1 Supplement lowercase letters.
    switch (cp) {
    case 0x00E0u: case 0x00E1u: case 0x00E2u: case 0x00E3u:
    case 0x00E4u: case 0x00E5u:
        return 'a';
    case 0x00E7u:
        return 'c';
    case 0x00E8u: case 0x00E9u: case 0x00EAu: case 0x00EBu:
        return 'e';
    case 0x00ECu: case 0x00EDu: case 0x00EEu: case 0x00EFu:
        return 'i';
    case 0x00F1u:
        return 'n';
    case 0x00F2u: case 0x00F3u: case 0x00F4u: case 0x00F5u:
    case 0x00F6u: case 0x00F8u:
        return 'o';
    case 0x00F9u: case 0x00FAu: case 0x00FBu: case 0x00FCu:
        return 'u';
    case 0x00FDu: case 0x00FFu:
        return 'y';
    default:
        break;
    }

    // Latin Extended-A, lowercase members of each pair. Grouped by base letter.
    // The Hungarian long-umlaut letters ő (U+0151) and ű (U+0171) are here.
    if (cp >= 0x0101u && cp <= 0x0105u && cp % 2 == 1) return 'a';
    if (cp >= 0x0107u && cp <= 0x010Du && cp % 2 == 1) return 'c';
    if (cp >= 0x010Fu && cp <= 0x0111u && cp % 2 == 1) return 'd';
    if (cp >= 0x0113u && cp <= 0x011Bu && cp % 2 == 1) return 'e';
    if (cp >= 0x011Du && cp <= 0x0123u && cp % 2 == 1) return 'g';
    if (cp >= 0x0125u && cp <= 0x0127u && cp % 2 == 1) return 'h';
    if (cp >= 0x0129u && cp <= 0x0131u && cp % 2 == 1) return 'i';
    if (cp == 0x0135u) return 'j';
    if (cp == 0x0137u || cp == 0x0138u) return 'k';
    if (cp == 0x013Au || cp == 0x013Cu || cp == 0x013Eu
        || cp == 0x0140u || cp == 0x0142u) return 'l';
    if (cp == 0x0144u || cp == 0x0146u || cp == 0x0148u
        || cp == 0x0149u || cp == 0x014Bu) return 'n';
    if (cp >= 0x014Du && cp <= 0x0151u && cp % 2 == 1) return 'o';  // includes ő
    if (cp == 0x0155u || cp == 0x0157u || cp == 0x0159u) return 'r';
    if (cp == 0x015Bu || cp == 0x015Du || cp == 0x015Fu
        || cp == 0x0161u) return 's';
    if (cp == 0x0163u || cp == 0x0165u || cp == 0x0167u) return 't';
    if (cp >= 0x0169u && cp <= 0x0173u && cp % 2 == 1) return 'u';  // includes ű
    if (cp == 0x0175u) return 'w';
    if (cp == 0x0177u || cp == 0x00FFu) return 'y';
    if (cp == 0x017Au || cp == 0x017Cu || cp == 0x017Eu) return 'z';

    return cp;
}

std::string foldDiacritics(const std::string &text)
{
    std::string out;
    out.reserve(text.size());
    size_t pos = 0;
    while (pos < text.size()) {
        utf8Append(out, foldDiacriticCodepoint(utf8Decode(text, pos)));
    }
    return out;
}

// ---------------------------------------------------------------------------
// Classification
// ---------------------------------------------------------------------------

bool isWordCodepoint(uint32_t cp)
{
    if (cp >= '0' && cp <= '9') return true;
    if (cp >= 'a' && cp <= 'z') return true;
    if (cp >= 'A' && cp <= 'Z') return true;
    if (cp < 0x80u) return false;

    // Latin-1 Supplement letters, excluding the two mathematical operators
    // U+00D7 and U+00F7 that sit inside the letter ranges.
    if (cp >= 0x00C0u && cp <= 0x00FFu && cp != 0x00D7u && cp != 0x00F7u) {
        return true;
    }
    // Latin Extended-A and Extended-B are entirely letters.
    if (cp >= 0x0100u && cp <= 0x024Fu) return true;
    // Greek and Cyrillic.
    if (cp >= 0x0370u && cp <= 0x04FFu) return true;

    return false;
}

bool isSpaceCodepoint(uint32_t cp)
{
    switch (cp) {
    case ' ':
    case '\t':
    case '\n':
    case '\r':
    case '\v':
    case '\f':
    case 0x00A0u:  // no-break space
    case 0x1680u:  // ogham space mark
    case 0x2028u:  // line separator
    case 0x2029u:  // paragraph separator
    case 0x202Fu:  // narrow no-break space
    case 0x205Fu:  // medium mathematical space
    case 0x3000u:  // ideographic space
        return true;
    default:
        // U+2000..U+200A general punctuation spaces.
        return cp >= 0x2000u && cp <= 0x200Au;
    }
}

// ---------------------------------------------------------------------------
// Token normalization
// ---------------------------------------------------------------------------

std::string normalizeToken(const std::string &token)
{
    std::string out;
    out.reserve(token.size());
    size_t pos = 0;
    while (pos < token.size()) {
        const uint32_t cp = utf8Decode(token, pos);
        const uint32_t lowered = foldCaseCodepoint(cp);
        if (isWordCodepoint(lowered)) {
            utf8Append(out, lowered);
        }
        // Non-word code points (punctuation, apostrophes, hyphens) are dropped
        // rather than treated as separators, so "don't" -> "dont".
    }
    return out;
}

std::vector<RawToken> tokenize(const std::string &text)
{
    std::vector<RawToken> tokens;

    size_t pos = 0;
    while (pos < text.size()) {
        // Skip any run of whitespace. This is what collapses repeated
        // separators without a separate pass.
        size_t probe = pos;
        uint32_t cp = utf8Decode(text, probe);
        if (isSpaceCodepoint(cp)) {
            pos = probe;
            continue;
        }

        // Consume one whitespace-delimited chunk, remembering exact bounds.
        const size_t start = pos;
        size_t end = pos;
        while (pos < text.size()) {
            probe = pos;
            cp = utf8Decode(text, probe);
            if (isSpaceCodepoint(cp)) {
                break;
            }
            pos = probe;
            end = pos;
        }

        RawToken tok;
        tok.startOffset = start;
        tok.endOffset = end;
        tok.normalized = normalizeToken(text.substr(start, end - start));
        if (!tok.normalized.empty()) {
            tokens.push_back(std::move(tok));
        }
    }

    return tokens;
}

std::vector<std::string> tokenizeToStrings(const std::string &text)
{
    std::vector<std::string> out;
    for (auto &tok : tokenize(text)) {
        out.push_back(std::move(tok.normalized));
    }
    return out;
}

// ---------------------------------------------------------------------------
// Similarity
// ---------------------------------------------------------------------------

size_t editDistanceCodepoints(const std::string &a, const std::string &b)
{
    const std::vector<uint32_t> ca = utf8ToCodepoints(a);
    const std::vector<uint32_t> cb = utf8ToCodepoints(b);

    if (ca.empty()) return cb.size();
    if (cb.empty()) return ca.size();

    // Guard against pathological inputs; tokens this long are not real words.
    constexpr size_t kMaxLen = 128;
    const size_t n = std::min(ca.size(), kMaxLen);
    const size_t m = std::min(cb.size(), kMaxLen);

    // Two-row dynamic programming; we only ever need the previous row.
    std::vector<size_t> prev(m + 1);
    std::vector<size_t> curr(m + 1);
    for (size_t j = 0; j <= m; ++j) {
        prev[j] = j;
    }

    for (size_t i = 1; i <= n; ++i) {
        curr[0] = i;
        for (size_t j = 1; j <= m; ++j) {
            const size_t cost = (ca[i - 1] == cb[j - 1]) ? 0u : 1u;
            curr[j] = std::min({prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost});
        }
        prev.swap(curr);
    }

    return prev[m];
}

size_t longestCommonPrefixCodepoints(const std::string &a, const std::string &b)
{
    size_t pa = 0;
    size_t pb = 0;
    size_t shared = 0;
    while (pa < a.size() && pb < b.size()) {
        size_t na = pa;
        size_t nb = pb;
        const uint32_t cpa = utf8Decode(a, na);
        const uint32_t cpb = utf8Decode(b, nb);
        if (cpa != cpb) {
            break;
        }
        ++shared;
        pa = na;
        pb = nb;
    }
    return shared;
}

namespace {

// Core similarity on two already-normalized strings, without the
// diacritic-folded fallback. Split out so tokenSimilarity() can call it twice.
double similarityCore(const std::string &a, const std::string &b)
{
    if (a.empty() || b.empty()) {
        return 0.0;
    }
    if (a == b) {
        return 1.0;
    }

    const size_t lenA = utf8Length(a);
    const size_t lenB = utf8Length(b);
    const size_t shortLen = std::min(lenA, lenB);
    const size_t longLen = std::max(lenA, lenB);

    // --- Component 1: length-aware edit distance -----------------------------
    // The allowance grows with the SHORTER length so that a long transcript word
    // cannot buy extra tolerance against a short script word.
    const double allowance =
        kEditBaseAllowance + static_cast<double>(shortLen) / kEditLengthDivisor;
    const double distance = static_cast<double>(editDistanceCodepoints(a, b));

    double editScore = 0.0;
    if (distance <= allowance) {
        // Linear falloff from 1.0 at zero distance to 0.0 at the allowance.
        editScore = 1.0 - (distance / (allowance + 1.0));
    }

    // --- Component 2: longest common prefix (the Hungarian workhorse) --------
    double prefixScore = 0.0;
    const size_t prefix = longestCommonPrefixCodepoints(a, b);
    if (prefix >= kMinPrefixCodepoints) {
        // Denominator is the SHORTER token: a bare stem in the script should
        // match a suffixed form in the transcript at full strength.
        const double coverage =
            static_cast<double>(prefix) / static_cast<double>(shortLen);

        // Damp by how far apart the two lengths are. Without this, a short word
        // would match every longer word sharing its opening letters.
        const double lengthRatio =
            static_cast<double>(shortLen) / static_cast<double>(longLen);
        const double damping =
            1.0 - kPrefixLengthMismatchDamping * (1.0 - lengthRatio);

        prefixScore = coverage * damping;
    }

    // The two components are individually sufficient kinds of evidence, so take
    // the better one rather than averaging them away.
    double score = std::max(editScore, prefixScore);

    // Never let a non-identical pair reach a perfect score; exact match must
    // remain strictly the strongest signal available.
    return std::min(score, 0.99);
}

}  // namespace

double tokenSimilarity(const std::string &a, const std::string &b)
{
    if (a.empty() || b.empty()) {
        return 0.0;
    }
    if (a == b) {
        return 1.0;
    }

    const double direct = similarityCore(a, b);

    // FALLBACK ONLY: retry with diacritics folded away, scaled down. This
    // catches ASR writing "hazban" for "házban" without ever letting a
    // diacritic-only difference score as high as a true match, because in
    // Hungarian ö/ő and ú/ü/ű distinguish real words.
    const std::string fa = foldDiacritics(a);
    const std::string fb = foldDiacritics(b);
    if (fa != a || fb != b) {
        const double folded = similarityCore(fa, fb) * kDiacriticFallbackScale;
        return std::max(direct, folded);
    }

    return direct;
}

// ---------------------------------------------------------------------------
// Function words
// ---------------------------------------------------------------------------

bool isFunctionWord(const std::string &normalizedToken)
{
    return functionWordSet().count(normalizedToken) > 0;
}

}  // namespace voicetracking
}  // namespace qprompt
