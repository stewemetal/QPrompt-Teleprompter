# Voice tracking engine

Standalone, dependency-free C++17 script-alignment engine for QPrompt's
speech-following scroll. Implements Decision 2 of
`agents/plans/2026-08-03/voice-tracking-design.md`.

**No Qt types appear anywhere in this directory.** The engine is pure C++17 so
it can be unit tested without a GUI or an event loop. The Qt/QML adapter is a
separate layer that owns `QString` conversion and the scroll integration.

## Files

| File | Purpose |
|---|---|
| `textnormalize.h/.cpp` | UTF-8 decoding, case folding, diacritic folding, tokenization, token similarity |
| `scriptmodel.h/.cpp` | Script tokenization, exact character offsets, IDF weights, inverted index |
| `aligner.h/.cpp` | The token-position HMM |
| `tests/aligner_tests.cpp` | Self-contained test binary, no framework |

## Build and test

```sh
clang++ -std=c++17 -O2 -Wall -Wextra \
    src/voicetracking/textnormalize.cpp \
    src/voicetracking/scriptmodel.cpp \
    src/voicetracking/aligner.cpp \
    src/voicetracking/tests/aligner_tests.cpp \
    -o /tmp/aligner_tests && /tmp/aligner_tests
```

Nothing here is wired into CMake yet; that integration is deliberately left to
the caller.

## The three ideas that make it work

### 1. Position is a distribution, not a number

State `i` means "the speaker is around script token `i`". The engine carries a
beam of weighted hypotheses, not a single index. Transitions encode the prior
that *continuing from where you are is overwhelmingly more likely than
teleporting*: `i->i` (filler), `i->i+1` (normal), `i->i+k` (skip, geometrically
penalized), a small backward step (re-reading), and a very low-prior recovery
transition **enabled only once confidence has already fallen**.

Keeping multiple modes is what makes repeated phrases survivable. When the same
sentence appears three times in a script, all three modes stay alive, the margin
between them collapses, and the engine reports `confident = false` and coasts
instead of guessing. It waits for a rare distinguishing token.

### 2. Rarity decides how much a word is allowed to move you

Every script token carries an IDF-style weight computed within *this* script,
because what matters is how distinctive a word is here — a word appearing once
in a 500-token script pins a location; a word appearing forty times does not.
Function words are hard-capped regardless of measured rarity.

This is why a passage of "and the of a" cannot manufacture confidence: the
`minInformativeWeight` gate counts summed rarity of matched tokens, not token
count.

### 3. Hungarian needs prefix matching, not whole-word matching

Hungarian is agglutinative — suffixes stack onto an invariant stem:

```
ház        house
házban     in the house
házaimban  in my houses
```

Whole-form matching treats `ház` and `házaimban` as unrelated (edit distance 6
on a 3-character stem). They share a complete 3-code-point prefix, and that
prefix *is* the word. `tokenSimilarity()` therefore takes the **maximum** of a
length-aware edit-distance score and a longest-common-prefix ratio, where the
prefix ratio divides by the length of the *shorter* token so a bare stem in the
script matches a suffixed form in the transcript at full strength.

Diacritic folding is a **fallback only**, scaled to 0.88. In Hungarian `ő` vs
`ö` and `ú`/`ü`/`ű` are real phonemic distinctions separating real words, so a
diacritic-only agreement must never score as high as an exact match.

Case folding covers the full Hungarian alphabet via the Latin Extended-A
even/odd pairing rule, which is what gives `Ő`→`ő` (U+0150→U+0151) and `Ű`→`ű`
(U+0170→U+0171) without a lookup table.

## Whisper revision semantics

whisper.cpp is **not** a streaming recognizer. It re-transcribes a sliding
buffer, so every chunk is a *fresh observation of the recent past*, not new text
appended. Consecutive chunks overlap heavily and freely revise each other.

`update()` defends against double-advancement three ways:

1. **Checkpoint/rollback keyed on `audioTimeSec`** — an observation at or before
   already-consumed audio time is a revision; the posterior rolls back to the
   state before that audio was first applied.
2. **Token-level overlap trimming** — the leading tokens of a new observation
   that duplicate the tail of what was already consumed are dropped, using a
   similarity-tolerant suffix/prefix match so whisper rewording the overlap does
   not defeat it.
3. **The transition prior itself** — restated tokens re-match at the current
   position through `i->i` and backward transitions rather than driving forward.

### An honest note on which of these is load-bearing

Mutation testing during development found that mechanisms (1) and (2) can each
be disabled individually **without any test in this suite failing**. Breaking
the transition prior (3), by contrast, *does* fail tests.

The conclusion is that the transition prior is the primary guard against
revision double-counting in the scenarios exercised here, and that rollback and
trimming are defense-in-depth whose independent contribution this suite does not
demonstrate. They are retained because they cost little and cover cases the
tests may not reach (very long restated tails, out-of-order timestamps), but
nobody should assume they are verified. If this engine misbehaves in the field
on revision handling, that is where to look first, and the replay harness from
open item 4 of the design doc is the right tool for it.

## Measured against real whisper output

`tokenSimilarity()` scored against errors whisper.cpp v1.9.1 (`ggml-base`)
actually made on Hungarian speech in testing, rather than synthetic ones:

| Script token | What whisper emitted | Score | Error type |
|---|---|---|---|
| `kívánok` | `kívánok` | 1.00 | exact |
| `felolvasott` | `felolvasot` | 0.95 | suffix truncated |
| `súgógép` | `sugo` | 0.67 | diacritics stripped + split |
| `teszt` | `test` | 0.67 | spelling |
| `ház` | `házaimban` | 0.63 | agglutination (by design) |
| `beszédfelismeréshez` | `beszét` | 0.52 | one word split into three |
| `napfényben` | `őszínapfényben` | **0.00** | **two words merged** |

**Known gap — merged words.** When the recognizer merges two script words into
one token, the *second* word scores zero: prefix matching only rewards agreement
at the start of a token, and `napfényben` appears at the end of
`őszínapfényben`. This was observed in real output, not hypothesized.

It is survivable rather than fatal — the merged token still matches the *first*
word strongly, and the HMM's `i->i+k` skip transition absorbs the orphaned
second word as a single missed token, which is exactly the case skip transitions
exist for. But it does mean a merge costs the rarity weight of the swallowed
word, so confidence dips slightly. If field testing shows merges are frequent,
the fix is a substring/containment term in `similarityCore()` for the case where
one token wholly contains the other; it was left out here to avoid the false
positives that unrestricted substring matching invites on short tokens.

## Safety posture

The engine is built to **coast, not guess**. `confident = false` means "keep
doing what you were doing"; it is never an instruction to jump. Three gates must
pass together before a position is reported as confident:

- posterior mass concentrated within `massRadius` of the winner,
- a clear log-margin over the best cluster elsewhere,
- a minimum summed rarity weight of matched tokens.

Global reacquisition additionally requires a candidate to win **two successive
updates** with a clear margin, and to beat the current position, before the
posterior relocates.

## Tuning constants

All aligner knobs live in `Aligner::Config` and are settable from an offline
replay harness. The values in the header are reasoned starting points, not
measured optima — the design doc is explicit that these need field tuning
against recorded audio before they mean anything.
