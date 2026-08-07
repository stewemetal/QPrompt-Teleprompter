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

#include "voicetracker.h"

#include <QFileInfo>
#include <QLoggingCategory>
#include <QStandardPaths>
#include <QTimer>

#ifdef QPROMPT_VOICE_TRACKING
#include "voicetracking/audiocapture.h"
#include "voicetracking/speechrecognizer.h"

#include <mutex>

using namespace qprompt::voicetracking;
#endif

// Off by default; enable with QT_LOGGING_RULES="qprompt.voicetracking.debug=true".
// Worth having permanently: every failure mode here (missing model, denied
// microphone, model that will not load) otherwise surfaces only as a line of
// text in a menu, which is impossible to inspect from a terminal or a log file.
Q_LOGGING_CATEGORY(lcVoiceTracking, "qprompt.voicetracking")

// ---------------------------------------------------------------------------
// Private
// ---------------------------------------------------------------------------

class VoiceTrackerPrivate
{
public:
    QString language = QStringLiteral("en");
    QString modelPath;
    QString errorString;
    bool active = false;
    bool coasting = true;
    double confidence = 0.0;

#ifdef QPROMPT_VOICE_TRACKING
    AudioCapture capture;
    SpeechRecognizer recognizer;
    ScriptModel script;
    Aligner aligner;

    // The script text whose character offsets the aligner reports. Held so the
    // adapter can validate offsets before handing them to QML.
    std::string scriptText;

    // Guards the aligner, which is touched from the inference thread (on each
    // observation) and from the GUI thread (re-anchors).
    std::mutex mutex;

    // Last alignment result. Kept for the UI-facing properties.
    bool lastConfident = false;
    double lastTokensPerSec = 0.0;
#endif
};

// ---------------------------------------------------------------------------

VoiceTracker::VoiceTracker(QObject *parent)
    : QObject(parent)
    , d(std::make_unique<VoiceTrackerPrivate>())
{
#ifdef QPROMPT_VOICE_TRACKING
    // Look for a model in the usual per-user data location. Not fatal when
    // absent: `modelAvailable` stays false and QML can offer to fetch one,
    // which is better than starting and failing at the first window.
    const QString dataDir =
        QStandardPaths::writableLocation(QStandardPaths::AppDataLocation);
    for (const QString &name : {QStringLiteral("ggml-base.bin"),
                                QStringLiteral("ggml-small.bin"),
                                QStringLiteral("ggml-medium.bin")}) {
        const QString candidate = dataDir + QStringLiteral("/models/") + name;
        if (QFileInfo::exists(candidate)) {
            d->modelPath = candidate;
            break;
        }
    }

    // Observations arrive on the inference thread, so this must not touch
    // QObject state directly: it aligns there, then hands the resulting
    // character offset to the GUI thread through a queued emit. QML resolves
    // that offset to a scroll position, because only it can -- the mapping
    // lives in the text document's layout.
    d->recognizer.setObservationCallback([this](const Observation &observation) {
        AlignResult result;
        {
            std::lock_guard<std::mutex> lock(d->mutex);
            result = d->aligner.update(observation);
        }

        QMetaObject::invokeMethod(this, [this, result]() {
            d->confidence = result.confidence;
            d->lastConfident = result.confident;
            d->lastTokensPerSec = result.estimatedTokensPerSec;

            // `coasting` is simply "the aligner does not trust this fix". The
            // view follows confident fixes and holds still otherwise, so this
            // is both the UI indicator and the gate on moving the prompter.
            if (d->coasting != !result.confident) {
                d->coasting = !result.confident;
                emit coastingChanged();
            }

            qCDebug(lcVoiceTracking)
                << "aligned: offset" << result.charOffset
                << "confidence" << result.confidence
                << (result.confident ? "confident" : "coasting");
            emit trackingUpdated(static_cast<int>(result.charOffset),
                                 result.confidence);
        }, Qt::QueuedConnection);
    });
#endif
}

VoiceTracker::~VoiceTracker()
{
    stop();
}

bool VoiceTracker::available() const
{
#ifdef QPROMPT_VOICE_TRACKING
    return true;
#else
    return false;
#endif
}

bool VoiceTracker::isActive() const { return d->active; }
QString VoiceTracker::language() const { return d->language; }
QString VoiceTracker::modelPath() const { return d->modelPath; }
bool VoiceTracker::isCoasting() const { return d->coasting; }
double VoiceTracker::confidence() const { return d->confidence; }
QString VoiceTracker::errorString() const { return d->errorString; }

bool VoiceTracker::modelAvailable() const
{
    return !d->modelPath.isEmpty() && QFileInfo::exists(d->modelPath);
}

void VoiceTracker::setLanguage(const QString &language)
{
    if (d->language == language)
        return;
    d->language = language;
#ifdef QPROMPT_VOICE_TRACKING
    d->recognizer.setLanguage(language.toStdString());
#endif
    emit languageChanged();
}

void VoiceTracker::setModelPath(const QString &path)
{
    if (d->modelPath == path)
        return;
    d->modelPath = path;
    emit modelPathChanged();
}

void VoiceTracker::setScript(const QString &plainText)
{
#ifdef QPROMPT_VOICE_TRACKING
    std::lock_guard<std::mutex> lock(d->mutex);
    d->scriptText = plainText.toStdString();
    d->script.build(d->scriptText);
    d->aligner.setScript(d->script);
#else
    Q_UNUSED(plainText)
#endif
}

void VoiceTracker::reanchor(double pixel, int characterOffset)
{
#ifdef QPROMPT_VOICE_TRACKING
    // The pixel is the view's business, not ours: QML has already moved there.
    // All this needs to do is tell the aligner to believe it, so the next match
    // is scored from the new position instead of the abandoned one.
    Q_UNUSED(pixel)
    std::lock_guard<std::mutex> lock(d->mutex);
    d->aligner.reanchor(static_cast<size_t>(std::max(0, characterOffset)));
#else
    Q_UNUSED(pixel)
    Q_UNUSED(characterOffset)
#endif
}

void VoiceTracker::start()
{
#ifdef QPROMPT_VOICE_TRACKING
    if (d->active)
        return;

    qCDebug(lcVoiceTracking) << "start requested; language" << d->language
                             << "model" << d->modelPath
                             << "script tokens" << d->script.tokenCount();

    if (!modelAvailable()) {
        d->errorString = tr("No speech model found. Voice tracking needs a "
                            "whisper model to recognise speech.");
        qCWarning(lcVoiceTracking) << "no speech model at" << d->modelPath;
        emit errorOccurred(d->errorString);
        return;
    }

    // Permission first: starting capture without it fails in a way that is
    // hard to distinguish from a broken microphone.
    if (AudioCapture::permission() == MicPermission::Denied) {
        d->errorString = tr("Microphone access is denied. Grant it in System "
                            "Settings > Privacy & Security > Microphone.");
        qCWarning(lcVoiceTracking) << "microphone permission denied";
        emit errorOccurred(d->errorString);
        return;
    }

    SpeechRecognizer::Config config;
    config.modelPath = d->modelPath.toStdString();
    config.language = d->language.toStdString();
    if (!d->recognizer.loadModel(config)) {
        d->errorString = tr("Could not load the speech model: %1")
                             .arg(QString::fromStdString(d->recognizer.lastError()));
        qCWarning(lcVoiceTracking) << "model load failed:" << d->errorString;
        emit errorOccurred(d->errorString);
        return;
    }

    {
        std::lock_guard<std::mutex> lock(d->mutex);
        d->aligner.reset();
        d->aligner.setScript(d->script);
    }

    if (!d->recognizer.startLive(d->capture)) {
        d->errorString = tr("Could not start listening: %1")
                             .arg(QString::fromStdString(d->recognizer.lastError()));
        qCWarning(lcVoiceTracking) << "startLive failed:" << d->errorString;
        emit errorOccurred(d->errorString);
        return;
    }

    d->errorString.clear();
    d->active = true;
    qCDebug(lcVoiceTracking) << "listening";
    emit activeChanged();
#endif
}

void VoiceTracker::stop()
{
#ifdef QPROMPT_VOICE_TRACKING
    if (!d->active)
        return;
    d->recognizer.stop();
    d->capture.stop();
    d->active = false;
    emit activeChanged();
#endif
}
