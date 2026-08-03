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

// QML-facing front end for voice tracking.
//
// This is the only file in the feature that knows about Qt. It owns the
// microphone, the recognizer and the aligner, and publishes a scroll velocity
// that Prompter.qml binds to. The alignment and control logic live in
// voicetracking/ as plain C++17 so they can be tested without a GUI; this class
// is the adapter between the two worlds and deliberately holds no algorithm of
// its own.
//
// Threading: audio capture and whisper inference run on their own threads. All
// signals reach QML on the GUI thread via queued connections, so QML only ever
// sees consistent state.
//
// The class exists even when voice tracking is compiled out, so QML does not
// need conditional imports; in that build it reports unavailable and every
// operation is a no-op.

#ifndef QPROMPT_VOICETRACKER_H
#define QPROMPT_VOICETRACKER_H

#include <QObject>
#include <QQmlEngine>
#include <QString>

#include <memory>

#ifdef QPROMPT_VOICE_TRACKING
#include "voicetracking/aligner.h"
#include "voicetracking/scriptmodel.h"
#include "voicetracking/scrollcontroller.h"
#endif

class VoiceTrackerPrivate;

class VoiceTracker : public QObject
{
    Q_OBJECT
    QML_ELEMENT

    // False in builds without voice tracking, or where the platform backend is
    // missing. QML uses this to hide the feature rather than offer a control
    // that cannot work.
    Q_PROPERTY(bool available READ available CONSTANT)

    // True once capture and recognition are actually running.
    Q_PROPERTY(bool active READ isActive NOTIFY activeChanged)

    // Recognition language: "en", "hu", ... Changing it while active restarts
    // recognition with the new language.
    Q_PROPERTY(QString language READ language WRITE setLanguage NOTIFY languageChanged)

    // Path to the whisper model. Empty until one is present; the feature
    // refuses to start without it rather than failing obscurely at runtime.
    Q_PROPERTY(QString modelPath READ modelPath WRITE setModelPath NOTIFY modelPathChanged)
    Q_PROPERTY(bool modelAvailable READ modelAvailable NOTIFY modelPathChanged)

    // True while running on the fallback rate because alignment is not
    // trusted. Drives a UI indicator so the reader can see that the prompter
    // has lost them, rather than wondering why it feels wrong.
    Q_PROPERTY(bool coasting READ isCoasting NOTIFY coastingChanged)

    // Posterior confidence in [0, 1], for diagnostics and a UI meter.
    Q_PROPERTY(double confidence READ confidence NOTIFY trackingUpdated)

    // Last error, empty when fine. Shown to the user rather than logged and
    // forgotten -- a silent microphone failure is indistinguishable from the
    // feature not working.
    Q_PROPERTY(QString errorString READ errorString NOTIFY errorOccurred)

public:
    explicit VoiceTracker(QObject *parent = nullptr);
    ~VoiceTracker() override;

    bool available() const;
    bool isActive() const;
    QString language() const;
    void setLanguage(const QString &language);
    QString modelPath() const;
    void setModelPath(const QString &path);
    bool modelAvailable() const;
    bool isCoasting() const;
    double confidence() const;
    QString errorString() const;

public slots:
    // Sets the script to follow. Plain text, exactly as the editor holds it:
    // character offsets in the results index into THIS string, so the caller
    // must pass the same text the view is displaying.
    void setScript(const QString &plainText);

    // Starts capture and recognition. Asks for microphone permission if it has
    // not been granted; emits errorOccurred and stays inactive if refused.
    void start();

    // Stops capture and recognition. Safe to call when not running.
    void stop();

    // Geometry the controller needs, pushed from QML because only the view
    // knows it.
    void setLineHeight(double px);
    void setFallbackVelocity(double pxPerSec);

    // How many script tokens fit on one rendered line. QML owns the font
    // metrics and wrap width, so only it can measure this. Used to convert the
    // aligner's tokens/second into pixels/second.
    void setTokensPerLine(double tokens);

    // Current scroll position, pushed each time QML asks for a velocity.
    void setCurrentPosition(double pixel);

    // Runs one control step and emits a new velocity.
    //
    // Split from alignment on purpose: the aligner reports a CHARACTER OFFSET,
    // but the controller needs a PIXEL target, and only QML can map between
    // them because the mapping lives in the text document's layout. So QML
    // receives trackingUpdated(offset), resolves the offset to a y coordinate,
    // and calls this with the result.
    void applyTarget(double targetPixel, double currentPixel);

    // The user scrolled or jumped: adopt that position instead of fighting it.
    void reanchor(double pixel, int characterOffset);

signals:
    // A new scroll velocity is available, in pixels per second. Prompter.qml
    // applies this to its animation.
    void velocityChanged(double pixelsPerSecond);

    // An immediate, non-animated seek is wanted. Only emitted after an explicit
    // re-anchor; routine corrections never seek.
    void seekRequested(double pixel);

    // Alignment moved. characterOffset indexes the script passed to
    // setScript(); QML maps it to a y position.
    void trackingUpdated(int characterOffset, double confidence);

    void activeChanged();
    void languageChanged();
    void modelPathChanged();
    void coastingChanged();
    void errorOccurred(const QString &message);

private:
    std::unique_ptr<VoiceTrackerPrivate> d;
};

#endif // QPROMPT_VOICETRACKER_H
