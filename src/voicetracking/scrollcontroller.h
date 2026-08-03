/****************************************************************************
 **
 ** QPrompt
 ** Copyright (C) 2026 Javier O. Cordero Pérez
 **
 ** This file is part of QPrompt.
 **
 ** This program is free software: you can redistribute it and/or modify
 ** it under the terms of the GNU General Public License as published by
 ** the Free Software Foundation, version 3 of the License.
 **
 ** This program is distributed in the hope that it will be useful,
 ** but WITHOUT ANY WARRANTY; without even the implied warranty of
 ** MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 ** GNU General Public License for more details.
 **
 ** You should have received a copy of the GNU General Public License
 ** along with this program.  If not, see <http://www.gnu.org/licenses/>.
 **
 ****************************************************************************/

// Turns alignment estimates into a scroll velocity.
//
// This implements Decision 3 of the design doc. It is deliberately free of Qt
// types, like the rest of this directory, so the control law can be tested
// without a GUI: the Qt adapter feeds it numbers and applies the result.
//
// The control law is proportional-plus-feed-forward:
//
//     velocity = speechVelocity + Kp * positionError
//
// and NOT a PID. A derivative term is omitted on purpose. Scroll position is
// already the integral of velocity, so proportional feedback on position gives
// a stable first-order catch-up on its own. Differentiating an ASR-derived
// target would amplify timestamp jitter and produce a kick every time whisper
// revises a hypothesis -- which, with a sliding re-transcription window, is
// constantly. The useful part of derivative action is available more cleanly as
// the aligner's filtered speech rate, which enters as the feed-forward term.

#ifndef QPROMPT_VOICETRACKING_SCROLLCONTROLLER_H
#define QPROMPT_VOICETRACKING_SCROLLCONTROLLER_H

#include <cstddef>

namespace qprompt {
namespace voicetracking {

// What the controller wants the prompter to do next.
struct ScrollCommand {
    // Desired scroll velocity in PIXELS PER SECOND. Always finite, always
    // clamped. The adapter converts this into whatever the view needs.
    double velocityPxPerSec = 0.0;

    // True when the controller wants an immediate, non-animated seek --
    // currently only after an explicit re-anchor. Routine tracking corrections
    // are folded into velocity instead, because a teleprompter that jumps is
    // unreadable even when the jump is "correct".
    bool seek = false;
    double seekToPixel = 0.0;

    // True while running on the fallback rate because alignment is not
    // trusted. Purely informational -- for a UI indicator.
    bool coasting = true;
};

class ScrollController {
public:
    struct Config {
        // Proportional gain, in inverse seconds. 0.5 corresponds to roughly a
        // two-second correction time constant: fast enough to catch up within a
        // sentence, slow enough that a single bad estimate is not visible as a
        // lurch.
        double kp = 0.5;

        // Errors smaller than this are treated as zero. Expressed as a
        // fraction of a line height: below about a third of a line, correcting
        // is more visible than the error itself.
        double deadbandLines = 0.35;

        // Hard bounds on commanded velocity, as a multiple of the current
        // estimated speech rate. The lower bound still allows slowing down;
        // the upper bound stops a large error from producing a scroll so fast
        // the text is unreadable while it catches up.
        double minVelocityFactor = 0.0;
        double maxVelocityFactor = 1.9;

        // Maximum change in commanded velocity per second, as a fraction of
        // the speech rate. Limits acceleration so corrections ease in.
        double maxAccelFactorPerSec = 1.0;

        // ASR latency compensation. whisper reports what was said a moment
        // ago, so aiming at the matched word puts the prompter behind where
        // the reader's eyes are. Aim this far ahead instead.
        double leadSeconds = 0.45;

        // Coasting behaviour. Below coastHoldSeconds the last trusted velocity
        // is held; between there and coastBlendSeconds it blends to the user's
        // configured constant speed; past that it is fully manual speed.
        double coastHoldSeconds = 1.0;
        double coastBlendSeconds = 3.0;
    };

    ScrollController();

    // The prompter's own constant-speed setting, in px/sec. Used as the
    // fallback the controller blends to when alignment is not trusted, so
    // turning voice tracking off mid-read is not a visible discontinuity.
    void setFallbackVelocity(double pxPerSec);

    // Pixels per line of rendered text. Sets the scale for the deadband.
    void setLineHeight(double px);

    // Feeds one alignment estimate.
    //
    //   confident        - from AlignResult; false means coast, never jump
    //   targetPixel      - y position the matched word should occupy
    //   currentPixel     - where the view actually is now
    //   speechPxPerSec   - aligner's filtered speech rate, converted to pixels
    //   nowSec           - monotonic clock, seconds
    ScrollCommand update(bool confident,
                         double targetPixel,
                         double currentPixel,
                         double speechPxPerSec,
                         double nowSec);

    // User scrolled or jumped to a marker: adopt that position without
    // fighting it, and emit a seek so the view lands exactly there.
    ScrollCommand reanchor(double toPixel, double nowSec);

    // Back to the initial state.
    void reset();

    const Config &config() const { return m_config; }
    void setConfig(const Config &config) { m_config = config; }

private:
    Config m_config;

    double m_fallbackVelocity = 0.0;
    double m_lineHeight = 24.0;

    double m_commandedVelocity = 0.0;   // last velocity actually emitted
    double m_lastTrustedVelocity = 0.0; // velocity when alignment last held
    double m_lastConfidentSec = -1.0;   // when alignment last held
    double m_lastUpdateSec = -1.0;      // for dt
    bool m_hasRun = false;
};

} // namespace voicetracking
} // namespace qprompt

#endif // QPROMPT_VOICETRACKING_SCROLLCONTROLLER_H
