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

// Binds the voice tracker to the prompter's scroll.
//
// HOW THIS DRIVES THE PROMPTER
// ----------------------------
// The prompter does not run a per-frame scroll loop. It assigns
// `position = __destination` -- the far end of the document -- and lets a
// single Linear NumberAnimation carry contentY there over minutes. Speed is
// therefore expressed as the animation's DURATION, not as a velocity.
//
// That is why this file converts px/sec into a duration rather than writing a
// velocity anywhere:
//
//     duration_ms = remaining_pixels / velocity_px_per_sec * 1000
//
// `timeToArival` is declared non-readonly in Prompter.qml precisely so it can
// be overridden this way. Writing to it breaks its binding to __timeToArival,
// which is intended here: while tracking, this component owns the scroll rate.
// restoreBinding() puts the original binding back when tracking stops.
//
// The alternative -- driving the integer speed step __i -- was rejected: __i is
// an integer mapped through a power curve, so at low speeds a single step
// nearly doubles the rate. That quantisation is far too coarse for smooth
// tracking.

import QtQuick
import com.cuperino.qprompt

QtObject {
    id: voiceTracking

    // The Prompter this drives. Required.
    property var prompter: null

    // The editor holding the script, used to map character offsets to y.
    property var editor: null

    required property bool enabled

    readonly property bool available: tracker.available
    readonly property bool active: tracker.active
    readonly property bool coasting: tracker.coasting
    readonly property double confidence: tracker.confidence
    readonly property string errorString: tracker.errorString
    property alias language: tracker.language
    property alias modelPath: tracker.modelPath
    readonly property bool modelAvailable: tracker.modelAvailable

    // True once we have taken over the prompter's scroll rate, so we know
    // whether there is a binding to restore.
    property bool __drivingScroll: false

    // Last velocity received, in px/sec.
    property double __velocity: 0

    readonly property VoiceTracker __tracker: VoiceTracker {
        id: tracker

        onVelocityChanged: function (pixelsPerSecond) {
            voiceTracking.__velocity = pixelsPerSecond;
            voiceTracking.__applyVelocity(pixelsPerSecond);
        }

        onSeekRequested: function (pixel) {
            voiceTracking.__seekTo(pixel);
        }

        // The aligner reports WHERE IN THE TEXT the speaker is; the controller
        // needs a y coordinate. Only this side can do that mapping, because it
        // lives in the text document's layout. Resolve the offset here and hand
        // the pixel target back, which is what runs the control step.
        onTrackingUpdated: function (characterOffset, confidence) {
            if (!voiceTracking.enabled || !prompter || !editor)
                return;
            tracker.applyTarget(voiceTracking.__pixelForOffset(characterOffset),
                                prompter.position);
        }

        onErrorOccurred: function (message) {
            if (typeof root !== "undefined" && root.passiveNotifications)
                showPassiveNotification(message);
        }
    }

    // Pushes the script and the view geometry the controller needs. Cheap
    // enough to call whenever the document or font changes.
    function refresh() {
        if (!prompter || !editor)
            return;
        tracker.setScript(editor.getText(0, editor.length));
        tracker.setLineHeight(editor.cursorRectangle.height > 0
                              ? editor.cursorRectangle.height
                              : prompter.fontSize);
        // The prompter's own constant speed, in px/sec, so that losing
        // alignment blends back to the speed the user chose rather than to an
        // arbitrary default. __relativeSpeed is px/sec doubled -- see
        // __timeToEnd in Prompter.qml, which divides a doubled distance by it.
        tracker.setFallbackVelocity(prompter.__relativeSpeed / 2);
    }

    function start() {
        if (!available || !prompter || !editor)
            return;
        refresh();
        tracker.start();
    }

    function stop() {
        tracker.stop();
        __restoreScrollBinding();
    }

    // Character offset -> scroll position, so the matched word sits in the read
    // region.
    //
    // This is the same conversion Prompter.qml's goTo() performs: move the
    // cursor, read back cursorRectangle.y, and subtract the read-region offset
    // (which is what topMargin holds). Reusing it keeps voice tracking aligned
    // with markers and manual navigation instead of inventing a second, subtly
    // different idea of where "the current line" is.
    //
    // Moving the cursor is a side effect, so this deliberately does NOT run
    // while the user is editing -- see the guard in the caller.
    function __pixelForOffset(characterOffset) {
        if (!prompter || !editor)
            return 0;
        const clamped = Math.max(0, Math.min(characterOffset, editor.length));
        const previous = editor.cursorPosition;
        editor.cursorPosition = clamped;
        const y = editor.cursorRectangle.y;
        // Put the cursor back: leaving it parked on the tracked word would
        // fight the user and move the visible caret around mid-read.
        editor.cursorPosition = previous;
        return y - prompter.topMargin + 1;
    }

    // Converts a velocity into the animation duration the prompter runs on.
    function __applyVelocity(pixelsPerSecond) {
        if (!prompter || !enabled || !prompter.__play)
            return;

        // Below this the duration explodes towards infinity; treat it as a
        // stop and let the prompter's own logic hold position.
        if (pixelsPerSecond <= 0.01) {
            prompter.timeToArival = 0;
            return;
        }

        const remaining = prompter.__destination - prompter.position;
        if (Math.abs(remaining) < 1) {
            return;
        }

        if (!__drivingScroll) {
            __drivingScroll = true;
        }

        // Duration for the animation still heading to __destination.
        prompter.timeToArival = Math.abs(remaining) / pixelsPerSecond * 1000;

        // Re-arm the ride. Every speed change in Prompter.qml ends by
        // re-assigning position to __destination, because the Behavior only
        // re-fires when the destination value actually changes -- that is what
        // the __jitterMargin/__tikTok parity hack exists for.
        prompter.position = prompter.__destination;
    }

    // Immediate, non-animated move. Used only for explicit re-anchors.
    function __seekTo(pixel) {
        if (!prompter)
            return;
        // Suppress the Behavior so this lands instantly instead of animating
        // over the current (possibly minutes-long) duration.
        const wasEnabled = prompter.motion.enabled;
        prompter.motion.enabled = false;
        prompter.position = pixel;
        prompter.motion.enabled = wasEnabled;
        if (prompter.__play && prompter.__i !== 0)
            prompter.position = prompter.__destination;
    }

    // Hands the scroll rate back to the prompter's own bindings.
    function __restoreScrollBinding() {
        if (!prompter || !__drivingScroll)
            return;
        prompter.timeToArival = Qt.binding(function () {
            return prompter.__timeToArival;
        });
        __drivingScroll = false;
        if (prompter.__play && prompter.__i !== 0)
            prompter.position = prompter.__destination;
    }

    // Feeds the current scroll position back so the controller can measure its
    // own error.
    //
    // Sampled on a timer rather than bound to contentY: the position changes
    // every frame while scrolling, and the controller only needs it at the rate
    // new alignment estimates arrive. A binding would push thousands of updates
    // a second across the C++ boundary for no benefit.
    property Timer __positionFeed: Timer {
        interval: 100
        repeat: true
        running: voiceTracking.enabled && voiceTracking.active && prompter !== null
        onTriggered: tracker.setCurrentPosition(prompter.position)
    }

    // A manual scroll or a marker jump means the user has told us where they
    // are; adopt it rather than fighting them.
    property Connections __userMoved: Connections {
        target: voiceTracking.enabled && voiceTracking.active ? prompter : null
        // onMovementEnded covers both dragging and flicking, and fires after
        // the Flickable has settled, which is when the position is finally
        // meaningful to anchor on.
        function onMovementEnded() {
            voiceTracking.__reanchorHere();
        }
    }

    function __reanchorHere() {
        if (!prompter || !editor)
            return;
        // Map the current scroll position back to a character offset the
        // aligner can anchor on, using the same read-region maths the
        // prompter's own goTo()/setCursorAtCurrentPosition() use.
        const offset = editor.positionAt(0, prompter.position + prompter.topMargin);
        tracker.reanchor(prompter.position, offset);
    }

    onEnabledChanged: {
        if (enabled)
            start();
        else
            stop();
    }
}
