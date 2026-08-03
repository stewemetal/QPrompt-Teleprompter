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

#include <QElapsedTimer>
#include <QFileInfo>
#include <QStandardPaths>
#include <QTimer>

#ifdef QPROMPT_VOICE_TRACKING
#include "voicetracking/audiocapture.h"
#include "voicetracking/speechrecognizer.h"

#include <mutex>

using namespace qprompt::voicetracking;
#endif

namespace {
// Used only until QML reports real metrics. Roughly the number of words on a
// teleprompter line at a typical font size and wrap width; wrong by a factor of
// two costs some feed-forward accuracy, which the proportional term absorbs.
constexpr double kFallbackTokensPerLine = 7.0;
} // namespace

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
    ScrollController controller;

    // The script text whose character offsets the aligner reports. Held so the
    // adapter can validate offsets before handing them to QML.
    std::string scriptText;

    // Guards the aligner and controller, which are touched from the inference
    // thread (on each observation) and from the GUI thread (geometry updates,
    // re-anchors).
    std::mutex mutex;

    // Monotonic clock for the control loop. Wall-clock time would let a system
    // clock change produce an absurd dt.
    QElapsedTimer clock;

    double currentPosition = 0.0;
    double lineHeight = 24.0;
    double fallbackVelocity = 0.0;

    // Last alignment result, kept so the control step can run when QML supplies
    // the pixel target that corresponds to the reported character offset.
    bool lastConfident = false;
    double lastTokensPerSec = 0.0;

    // How many script tokens occupy one rendered line. Supplied by QML, which
    // is the only side that knows the font metrics and wrap width.
    double tokensPerLine = 0.0;
#endif
};

// ---------------------------------------------------------------------------

VoiceTracker::VoiceTracker(QObject *parent)
    : QObject(parent)
    , d(std::make_unique<VoiceTrackerPrivate>())
{
#ifdef QPROMPT_VOICE_TRACKING
    d->clock.start();

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

    // Observations arrive on the inference thread. Everything below runs there,
    // so it must not touch QObject state directly; results reach the GUI thread
    // through a queued emit.
    // Observations arrive on the inference thread. Alignment happens there;
    // the control step does NOT, because it needs a target in pixels and only
    // QML can turn a character offset into a y coordinate (via the text
    // document's layout). So this callback aligns, then hands the offset to the
    // GUI thread, which resolves it to a pixel and calls applyTarget().
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

void VoiceTracker::setLineHeight(double px)
{
#ifdef QPROMPT_VOICE_TRACKING
    std::lock_guard<std::mutex> lock(d->mutex);
    d->lineHeight = px > 0.0 ? px : 24.0;
    d->controller.setLineHeight(d->lineHeight);
#else
    Q_UNUSED(px)
#endif
}

void VoiceTracker::setFallbackVelocity(double pxPerSec)
{
#ifdef QPROMPT_VOICE_TRACKING
    std::lock_guard<std::mutex> lock(d->mutex);
    d->fallbackVelocity = pxPerSec;
    d->controller.setFallbackVelocity(pxPerSec);
#else
    Q_UNUSED(pxPerSec)
#endif
}

void VoiceTracker::setCurrentPosition(double pixel)
{
#ifdef QPROMPT_VOICE_TRACKING
    std::lock_guard<std::mutex> lock(d->mutex);
    d->currentPosition = pixel;
#else
    Q_UNUSED(pixel)
#endif
}

void VoiceTracker::applyTarget(double targetPixel, double currentPixel)
{
#ifdef QPROMPT_VOICE_TRACKING
    ScrollCommand command;
    bool coastingChangedNow = false;
    {
        std::lock_guard<std::mutex> lock(d->mutex);
        d->currentPosition = currentPixel;

        // The aligner counts script TOKENS; the view moves PIXELS. Converting
        // between them needs the rendered size of a token, which is a view
        // property. Derive it from the distance the text has to travel per
        // token rather than assuming a fixed words-per-line: pixels per token
        // is the line height divided by however many tokens fit on a line.
        const double tokensPerLine = d->tokensPerLine > 0.0 ? d->tokensPerLine
                                                            : kFallbackTokensPerLine;
        const double pxPerToken = d->lineHeight / tokensPerLine;
        const double speechPxPerSec = d->lastTokensPerSec * pxPerToken;

        command = d->controller.update(d->lastConfident,
                                       targetPixel,
                                       currentPixel,
                                       speechPxPerSec,
                                       d->clock.elapsed() / 1000.0);

        if (d->coasting != command.coasting) {
            d->coasting = command.coasting;
            coastingChangedNow = true;
        }
    }

    if (coastingChangedNow)
        emit coastingChanged();
    if (command.seek)
        emit seekRequested(command.seekToPixel);
    emit velocityChanged(command.velocityPxPerSec);
#else
    Q_UNUSED(targetPixel)
    Q_UNUSED(currentPixel)
#endif
}

void VoiceTracker::setTokensPerLine(double tokens)
{
#ifdef QPROMPT_VOICE_TRACKING
    std::lock_guard<std::mutex> lock(d->mutex);
    d->tokensPerLine = tokens;
#else
    Q_UNUSED(tokens)
#endif
}

void VoiceTracker::reanchor(double pixel, int characterOffset)
{
#ifdef QPROMPT_VOICE_TRACKING
    ScrollCommand command;
    {
        std::lock_guard<std::mutex> lock(d->mutex);
        d->aligner.reanchor(static_cast<size_t>(std::max(0, characterOffset)));
        d->currentPosition = pixel;
        command = d->controller.reanchor(pixel, d->clock.elapsed() / 1000.0);
    }
    emit velocityChanged(command.velocityPxPerSec);
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

    if (!modelAvailable()) {
        d->errorString = tr("No speech model found. Voice tracking needs a "
                            "whisper model to recognise speech.");
        emit errorOccurred(d->errorString);
        return;
    }

    // Permission first: starting capture without it fails in a way that is
    // hard to distinguish from a broken microphone.
    if (AudioCapture::permission() == MicPermission::Denied) {
        d->errorString = tr("Microphone access is denied. Grant it in System "
                            "Settings > Privacy & Security > Microphone.");
        emit errorOccurred(d->errorString);
        return;
    }

    SpeechRecognizer::Config config;
    config.modelPath = d->modelPath.toStdString();
    config.language = d->language.toStdString();
    if (!d->recognizer.loadModel(config)) {
        d->errorString = tr("Could not load the speech model: %1")
                             .arg(QString::fromStdString(d->recognizer.lastError()));
        emit errorOccurred(d->errorString);
        return;
    }

    {
        std::lock_guard<std::mutex> lock(d->mutex);
        d->controller.reset();
        d->aligner.reset();
        d->aligner.setScript(d->script);
    }

    if (!d->recognizer.startLive(d->capture)) {
        d->errorString = tr("Could not start listening: %1")
                             .arg(QString::fromStdString(d->recognizer.lastError()));
        emit errorOccurred(d->errorString);
        return;
    }

    d->errorString.clear();
    d->active = true;
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
