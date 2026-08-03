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

#include "scrollcontroller.h"

#include <algorithm>
#include <cmath>

namespace qprompt {
namespace voicetracking {
namespace {

// Guards against NaN/inf reaching the view, where they would corrupt the
// animation's state rather than merely looking wrong for one frame.
double sanitize(double value, double fallback = 0.0)
{
    return std::isfinite(value) ? value : fallback;
}

} // namespace

ScrollController::ScrollController() = default;

void ScrollController::setFallbackVelocity(double pxPerSec)
{
    m_fallbackVelocity = std::max(0.0, sanitize(pxPerSec));
}

void ScrollController::setLineHeight(double px)
{
    // A zero or negative line height would collapse the deadband and make the
    // controller chase sub-pixel errors.
    m_lineHeight = std::max(1.0, sanitize(px, 24.0));
}

void ScrollController::reset()
{
    m_commandedVelocity = 0.0;
    m_lastTrustedVelocity = 0.0;
    m_lastConfidentSec = -1.0;
    m_lastUpdateSec = -1.0;
    m_hasRun = false;
}

ScrollCommand ScrollController::reanchor(double toPixel, double nowSec)
{
    // An explicit re-anchor is the one case where seeking is right: the user
    // has just told us where they are, so easing there would be slower and
    // more confusing than simply being there.
    ScrollCommand command;
    command.seek = true;
    command.seekToPixel = sanitize(toPixel);
    command.velocityPxPerSec = m_fallbackVelocity;
    command.coasting = true;

    m_commandedVelocity = m_fallbackVelocity;
    m_lastTrustedVelocity = m_fallbackVelocity;
    m_lastConfidentSec = -1.0;
    m_lastUpdateSec = sanitize(nowSec);
    m_hasRun = true;
    return command;
}

ScrollCommand ScrollController::update(bool confident,
                                       double targetPixel,
                                       double currentPixel,
                                       double speechPxPerSec,
                                       double nowSec)
{
    targetPixel = sanitize(targetPixel);
    currentPixel = sanitize(currentPixel);
    speechPxPerSec = std::max(0.0, sanitize(speechPxPerSec));
    nowSec = sanitize(nowSec);

    // dt drives both acceleration limiting and the coast blend. On the first
    // call, and if the clock ever goes backwards, fall back to a nominal step
    // rather than letting a wild dt scale the limits.
    double dt = 0.1;
    if (m_hasRun && nowSec > m_lastUpdateSec) {
        dt = std::min(nowSec - m_lastUpdateSec, 1.0);
    }
    m_lastUpdateSec = nowSec;
    m_hasRun = true;

    ScrollCommand command;

    // The rate the text should move at when tracking holds. The aligner's
    // estimate is preferred; its absence means we have no speech-derived rate
    // and the user's setting is all we have.
    const double baseVelocity = speechPxPerSec > 0.0 ? speechPxPerSec
                                                     : m_fallbackVelocity;

    double desired;
    if (confident) {
        // Aim ahead of the matched word by the ASR latency, otherwise the
        // prompter sits permanently behind the reader by that latency.
        const double lead = baseVelocity * m_config.leadSeconds;
        double error = (targetPixel + lead) - currentPixel;

        // Ignore errors too small to be worth a visible correction.
        const double deadband = m_config.deadbandLines * m_lineHeight;
        if (std::fabs(error) < deadband) {
            error = 0.0;
        }

        desired = baseVelocity + m_config.kp * error;
        m_lastConfidentSec = nowSec;
        m_lastTrustedVelocity = desired;
        command.coasting = false;
    } else {
        // Not trusted: coast. Hold the last trusted velocity briefly, then
        // blend to the user's constant speed. Never seek, never jump -- an
        // unconfident aligner is exactly when a jump would be wrong.
        const double sinceConfident = (m_lastConfidentSec < 0.0)
                                          ? m_config.coastBlendSeconds
                                          : (nowSec - m_lastConfidentSec);

        if (sinceConfident <= m_config.coastHoldSeconds) {
            desired = m_lastTrustedVelocity;
        } else if (sinceConfident >= m_config.coastBlendSeconds) {
            desired = m_fallbackVelocity;
        } else {
            const double span = std::max(1e-6,
                                         m_config.coastBlendSeconds
                                             - m_config.coastHoldSeconds);
            const double t = (sinceConfident - m_config.coastHoldSeconds) / span;
            desired = m_lastTrustedVelocity
                      + (m_fallbackVelocity - m_lastTrustedVelocity) * t;
        }
        command.coasting = true;
    }

    // Clamp before smoothing, so the acceleration limit governs the approach to
    // the clamped value rather than to an unreachable one.
    //
    // The bounds scale with the speech rate, but a speaker who has fallen
    // silent would collapse them to zero and freeze the prompter, so the user's
    // configured speed sets a floor for the scale.
    const double scale = std::max(baseVelocity, m_fallbackVelocity);
    if (scale > 0.0) {
        desired = std::clamp(desired,
                             m_config.minVelocityFactor * scale,
                             m_config.maxVelocityFactor * scale);
    } else {
        desired = std::max(0.0, desired);
    }

    // Acceleration limit: ease into the new velocity instead of stepping to it.
    // Visible jitter is worse than being slightly behind.
    const double maxDelta = m_config.maxAccelFactorPerSec
                            * std::max(scale, 1.0) * dt;
    const double delta = std::clamp(desired - m_commandedVelocity,
                                    -maxDelta, maxDelta);
    m_commandedVelocity = m_commandedVelocity + delta;

    command.velocityPxPerSec = sanitize(m_commandedVelocity);
    command.seek = false;
    command.seekToPixel = 0.0;
    return command;
}

} // namespace voicetracking
} // namespace qprompt
