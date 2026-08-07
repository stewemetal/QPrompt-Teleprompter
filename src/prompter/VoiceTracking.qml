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
// The recognizer says WHICH WORDS were heard; the aligner says WHERE in the
// script those words are. That is a position, not a speed -- so this drives the
// prompter the same way every other "go to a place in the script" feature does:
// by moving to that position and letting the prompter animate the rest.
//
// `scrollToOffset()` is the seam (Prompter.qml). It is what markers, search
// results and manual navigation already use: set the cursor, read back the
// y coordinate of that line, and assign `position`, which the existing
// `Behavior on position` animates smoothly. Voice tracking is therefore just
// another caller of a path the prompter already understands, rather than a
// second, parallel scroll driver.
//
// An earlier design instead overrode the scroll RATE -- computing a velocity
// from the alignment error and expressing it as the animation's duration. It
// worked, but only by fighting the scroll engine: forcing `__i` non-zero so a
// destination existed, respecting the `__jitterMargin`/`__tikTok` parity hack
// to make writes re-fire, breaking and restoring the `timeToArival` binding,
// and adding a seek-on-start state machine to escape the case where the view
// had already run past the speaker. All of that disappears when the alignment
// is treated as a position.

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

    // Ignore a match that would move the view less than this. Alignment jitters
    // by a few pixels between updates even while the speaker holds still, and
    // re-targeting on that jitter restarts the animation constantly, which reads
    // as a shimmer rather than a scroll.
    readonly property int __deadbandPx: 8

    // How long a follow move takes. Roughly one recogniser hop, so consecutive
    // matches join into continuous motion instead of a series of visible jumps.
    readonly property int __followMs: 500

    readonly property VoiceTracker __tracker: VoiceTracker {
        id: tracker

        // Alignment moved. This is the whole integration: resolve the character
        // offset the aligner reported to a scroll position, and go there.
        //
        // Only confident matches move the view. When the aligner is unsure --
        // the speaker ad-libbed, went quiet, or the audio was noisy -- the right
        // behaviour is to leave the prompter exactly where it is. Guessing would
        // send the reader somewhere they are not, which is far worse than
        // briefly not following.
        onTrackingUpdated: function (characterOffset, confidence) {
            if (!voiceTracking.enabled || !prompter || !editor)
                return;
            if (voiceTracking.coasting)
                return;
            voiceTracking.__followTo(characterOffset);
        }

        onErrorOccurred: function (message) {
            if (typeof root !== "undefined" && root.passiveNotifications)
                showPassiveNotification(message);
        }
    }

    // Pushes the script to the recognizer. Cheap enough to call whenever the
    // document changes.
    function refresh() {
        if (!prompter || !editor)
            return;
        tracker.setScript(editor.getText(0, editor.length));
    }

    function start() {
        if (!available || !prompter || !editor)
            return;
        refresh();
        tracker.start();
    }

    function stop() {
        tracker.stop();
    }

    // Hold the timed scroll at a standstill for as long as tracking is running.
    //
    // While following a voice, the voice is the only thing that should move the
    // text: left running, the constant-speed scroll drifts on through every
    // pause and the two sources fight over the same position. A Binding rather
    // than a one-off assignment because entering prompting mode sets __i itself,
    // after this component has already started -- so the value has to be held,
    // not just set once. Restoring the binding on `when` going false hands the
    // user's own speed straight back.
    // Hold the timed scroll at a standstill for as long as tracking is running.
    //
    // While following a voice, the voice is the only thing that should move the
    // text: left running, the constant-speed scroll drifts on through every
    // pause and the two sources fight over the same position.
    //
    // Tell the prompter that something else is driving the scroll, so it stops
    // arming its own timed ride to the end of the document. Without this the
    // constant-speed scroll keeps carrying the view through every pause and the
    // two sources fight over the same position. Clearing it hands the user's own
    // speed straight back -- __i was never touched.
    property Binding __ownScroll: Binding {
        target: prompter
        property: "scrollDrivenExternally"
        value: voiceTracking.active
        restoreMode: Binding.RestoreBindingOrValue
    }

    // Moves the prompter so the matched words sit in the read region.
    //
    // Deliberately delegates to the prompter rather than computing a position
    // here: the prompter owns where the read region is and how the view eases
    // there, and duplicating that would give voice tracking its own subtly
    // different idea of where "the current line" sits.
    function __followTo(characterOffset) {
        const clamped = Math.max(0, Math.min(characterOffset, editor.length));
        const target = prompter.positionForCursor(clamped);
        if (Math.abs(target - prompter.position) < __deadbandPx)
            return;
        prompter.scrollToOffset(clamped, __followMs);
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
        // aligner can anchor on, using the prompter's own inverse mapping.
        tracker.reanchor(prompter.position, prompter.cursorForPosition());
    }

    onEnabledChanged: {
        if (enabled)
            start();
        else
            stop();
    }
}
