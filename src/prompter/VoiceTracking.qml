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

    // Backward moves smaller than this are treated as alignment jitter and
    // ignored; larger ones are taken as the reader genuinely going back. About
    // two lines at a typical prompting size.
    readonly property int __backwardTolerancePx: 120

    // How far behind the speaker recognition runs, in seconds. whisper reports
    // on a window of audio, so a match describes speech from roughly this long
    // ago. Used with the measured reading rate to lead the view.
    readonly property real __latencySeconds: 1.2

    // Rough number of script tokens on a rendered line. Only used to size the
    // lead, where being off by a word costs a few pixels of aim.
    readonly property real __tokensPerLine: 7.0

    // How close the confirming match must be for the opening to be accepted,
    // and how much later it must arrive. The delay is what makes it independent
    // evidence: whisper restates the same window several times as it firms up,
    // and those restatements agree with each other by construction.
    readonly property int __openingAgreementPx: 60
    readonly property int __openingAgreementMs: 700

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
        __opened = false;
        __pendingAt = -1;
    }

    // Take over as the prompter's scroll driver while listening, and hand it
    // back the moment we stop. Selecting Voice suspends the timed ride; without
    // that the constant-speed scroll keeps carrying the view through every pause
    // and the two sources fight over the same position. Returning to Timed gives
    // the user's own speed straight back -- __i was never touched.
    property Binding __scrollDriver: Binding {
        target: prompter
        property: "scrollDriver"
        value: voiceTracking.active ? Prompter.ScrollDrivers.Voice
                                    : Prompter.ScrollDrivers.Timed
        restoreMode: Binding.RestoreBindingOrValue
    }

    // Moves the prompter so the matched words sit in the read region.
    //
    // Deliberately delegates to the prompter rather than computing a position
    // here: the prompter owns where the read region is and how the view eases
    // there, and duplicating that would give voice tracking its own subtly
    // different idea of where "the current line" sits.
    // Acquisition state. Until the view has committed to a position, a match may
    // just as well be a repeated phrase elsewhere in the script, so the first
    // one is held until a later one agrees. See the gate in __followTo().
    property bool __opened: false
    property real __pendingPx: 0
    property real __pendingAt: -1

    function __followTo(characterOffset) {
        const clamped = Math.max(0, Math.min(characterOffset, editor.length));

        // Aim ahead of the words just recognised.
        //
        // Recognition is unavoidably behind the speaker: whisper needs a window
        // of audio before it can report anything, so by the time a match lands
        // the reader has already moved on. Putting the matched word in the read
        // region therefore shows the line just FINISHED, and the reader is
        // permanently reading below it.
        //
        // Lead by however far they travel during that latency -- their measured
        // rate times the delay -- so the read region holds the words being
        // spoken now. The rate is measured rather than assumed, so a fast reader
        // gets a proportionally larger lead. This is the "some autoscrolling in
        // voice mode" effect: motion between matches still comes from the voice,
        // it is just aimed where the voice will be rather than where it was.
        const leadTokens = tracker.tokensPerSecond * __latencySeconds;
        const target = prompter.positionForCursor(clamped)
                     + leadTokens * __pixelsPerToken();
        const delta = target - prompter.position;

        if (Math.abs(delta) < __deadbandPx)
            return;

        // Confirm the opening before moving at all.
        //
        // At the start of a read nothing constrains where the speaker might be,
        // so a script that repeats a distinctive phrase -- the demo text names
        // itself eleven times -- offers several equally good matches, and
        // whisper's first partial transcript picks one essentially at random.
        // Acting on it makes the prompter open in the wrong paragraph and then
        // lurch across the document once more words arrive.
        //
        // Waiting for a second match that agrees, taken a beat later so it rests
        // on speech the first one had not heard, removes that entirely: a
        // spurious copy rarely wins twice. It costs a moment of stillness at the
        // very start, which is a good trade for never opening in the wrong place.
        if (!__opened) {
            const now = Date.now();
            const agrees = __pendingAt >= 0
                        && Math.abs(target - __pendingPx) <= __openingAgreementPx;
            if (agrees && now - __pendingAt >= __openingAgreementMs) {
                __opened = true;
            } else {
                if (!agrees) {
                    __pendingPx = target;
                    __pendingAt = now;
                }
                return;
            }
        }

        // Never walk the text backwards by a line or two.
        //
        // The aligner refines its estimate as whisper firms up a window, so a
        // fix can land a token or two behind the last one even while the reader
        // is moving steadily forward. Correcting for that is invisible as a
        // number and very visible on screen: the text twitches upward mid-line,
        // which reads as a glitch and costs the reader their place.
        //
        // A genuine backward move -- the reader restarting a paragraph -- is far
        // larger than this, so it still gets through. Anything smaller is
        // jitter, and holding position is the honest response.
        if (delta < 0 && -delta < __backwardTolerancePx)
            return;

        prompter.scrollToPosition(target, __followMs);
    }

    // Rendered width of one script token, used to convert the lead from tokens
    // into pixels. Measured from the document rather than assumed: font size and
    // wrap width are the view's business and the user changes both.
    function __pixelsPerToken() {
        const lineHeight = editor.cursorRectangle.height > 0
                         ? editor.cursorRectangle.height
                         : prompter.fontSize;
        return lineHeight / __tokensPerLine;
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
        __opened = true;
        __pendingAt = -1;
    }

    onEnabledChanged: {
        if (enabled)
            start();
        else
            stop();
    }
}
