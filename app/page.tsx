"use client"

import { useCallback, useEffect, useRef, useState } from "react"
import { DashboardHeader } from "@/components/dashboard-header"
import { CameraFeed } from "@/components/camera-feed"
import { MetricsSidebar } from "@/components/metrics-sidebar"
import { AiFeedback, type Feedback } from "@/components/ai-feedback"
import { ExerciseAnalyzer, CalibrationController, type ExerciseMode, type FormAlert } from "@/lib/exercise-analyzer"
import type { NormalizedLandmark } from "@mediapipe/tasks-vision"

const MODES = ["Shadowboxing", "Squats", "Lunges", "Jab-Cross", "Push-ups", "Plank"] as const

// How many recent rep scores feed the displayed accuracy number. A short
// window keeps the metric responsive to how the last few reps actually
// looked, rather than a lifetime average that barely moves after rep 30.
const ACCURACY_WINDOW = 5

// How long a flashed form-alert ("Push your knees outward.") stays on
// screen before auto-clearing. This is presentation, not analysis — the
// analyzer itself doesn't have a concept of "for how long," it just reports
// a fault on the frame it's detected; the UI decides how long to flash it.
const FORM_ALERT_DISPLAY_MS = 2600

/**
 * Exercise phase, separate from whether the camera itself is on/off:
 *   - "idle": camera off, nothing happening.
 *   - "calibrating": camera on, CalibrationController is driving — getting
 *     the person into frame and into the starting position before anything
 *     is scored. See lib/exercise-analyzer.ts CalibrationController.
 *   - "tracking": calibration finished, ExerciseAnalyzer is now receiving
 *     frames and counting reps / running the plank timer.
 */
type Phase = "idle" | "calibrating" | "tracking"

export default function Page() {
  const [cameraOn, setCameraOn] = useState(false)
  const [phase, setPhase] = useState<Phase>("idle")
  const [mode, setMode] = useState<ExerciseMode>(MODES[0])
  const [reps, setReps] = useState(0)
  const [accuracy, setAccuracy] = useState(0)
  const [feedback, setFeedback] = useState<Feedback[]>([])
  const [tracking, setTracking] = useState(false) // true once a pose has actually been seen this session
  const [formAlert, setFormAlert] = useState<FormAlert | null>(null)
  const [calibrationMessage, setCalibrationMessage] = useState("")
  const [countdownValue, setCountdownValue] = useState<number | null>(null)

  // Plank-only state — a held timer has no equivalent in the rep-counting
  // UI, so it gets its own fields rather than being forced through `reps`.
  const [holdSeconds, setHoldSeconds] = useState(0)
  const [bestHoldSeconds, setBestHoldSeconds] = useState(0)

  const fbId = useRef(0)
  const formAlertTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  // Lives for the whole session — owns the per-mode state machines (squat
  // phase, wrist velocity, plank timer, etc). A ref because it's mutated
  // every frame and must NOT be recreated on re-render.
  const analyzerRef = useRef<ExerciseAnalyzer | null>(null)
  if (!analyzerRef.current) analyzerRef.current = new ExerciseAnalyzer()

  // Drives the pre-exercise calibration phase. Separate instance/ref from
  // the analyzer above so the two state machines (get-into-position vs.
  // count-the-movement) don't tangle — see module note in exercise-analyzer.ts.
  const calibrationRef = useRef<CalibrationController | null>(null)
  if (!calibrationRef.current) calibrationRef.current = new CalibrationController()

  // Recent rep form-scores, used to compute the displayed accuracy as a
  // short rolling average (see ACCURACY_WINDOW above) rather than an
  // ever-more-sluggish lifetime average.
  const recentScoresRef = useRef<number[]>([])

  // Tracks the current mode inside the per-frame callback without making
  // the callback's identity change every time mode changes (see
  // handlePoseLandmarks below — it's wrapped in useCallback with an empty
  // dep array on purpose, reading mode via this ref instead).
  const modeRef = useRef(mode)
  modeRef.current = mode

  const phaseRef = useRef(phase)
  phaseRef.current = phase

  function clearFormAlertTimer() {
    if (formAlertTimeoutRef.current !== null) {
      clearTimeout(formAlertTimeoutRef.current)
      formAlertTimeoutRef.current = null
    }
  }

  function flashFormAlert(alert: FormAlert) {
    clearFormAlertTimer()
    setFormAlert(alert)
    formAlertTimeoutRef.current = setTimeout(() => setFormAlert(null), FORM_ALERT_DISPLAY_MS)
  }

  // Form alerts are surfaced two ways: an immediate flash over the video
  // (handled by flashFormAlert above) AND an entry in the persistent
  // feedback list, so there's also a retrospective record to scroll back
  // through after the flash fades.
  function flashAndLogFormAlert(alert: FormAlert) {
    flashFormAlert(alert)
    fbId.current += 1
    setFeedback((prev) => [{ id: fbId.current, text: alert.text, tone: "warn" }, ...prev].slice(0, 6))
  }

  // Restart calibration whenever the exercise mode changes or the camera
  // (re)starts — a half-completed squat rep, or a calibration step for the
  // PREVIOUS exercise, should never bleed into a freshly selected mode.
  function resetForNewSession() {
    analyzerRef.current?.reset()
    calibrationRef.current?.reset()
    recentScoresRef.current = []
    setReps(0)
    setAccuracy(0)
    setTracking(false)
    setHoldSeconds(0)
    setBestHoldSeconds(0)
    setCalibrationMessage("")
    setCountdownValue(null)
    clearFormAlertTimer()
    setFormAlert(null)
  }

  useEffect(() => {
    if (cameraOn) {
      resetForNewSession()
      setPhase("calibrating")
      fbId.current += 1
      setFeedback([{ id: fbId.current, text: `Get ready for ${mode}…`, tone: "info" }])
    } else {
      setPhase("idle")
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [mode, cameraOn])

  // Fired on every detection frame (~30x/sec) via CameraFeed -> usePoseLandmarker.
  // Branches on phaseRef: during calibration, frames go to
  // CalibrationController; once that reports "done", we capture this
  // person's baseline and switch to feeding ExerciseAnalyzer instead.
  // Deliberately does NOT setState on every call where avoidable — only
  // when something user-visible actually changed — so the dashboard isn't
  // re-rendering 30 times a second.
  //
  // useCallback with an empty dep array on purpose: this needs a STABLE
  // identity so usePoseLandmarker (in camera-feed.tsx) doesn't tear down
  // and rebuild its detection loop every render. Everything it reads that
  // could change over time (mode, phase) comes from a ref, not a closed-
  // over variable — see modeRef/phaseRef above. Everything it calls
  // (setState setters, clearFormAlertTimer, analyzer/calibration methods)
  // is either a React-guaranteed-stable setter or a ref-backed class
  // instance, so none of it goes stale despite the frozen closure.
  const handlePoseLandmarks = useCallback((landmarks: NormalizedLandmark[] | undefined, timestampMs: number) => {
    const analyzer = analyzerRef.current
    const calibration = calibrationRef.current
    if (!analyzer || !calibration) return

    const currentMode = modeRef.current

    if (phaseRef.current === "calibrating") {
      const status = calibration.process(landmarks, currentMode, timestampMs)
      setCalibrationMessage(status.message)
      setCountdownValue(status.countdownValue ?? null)

      if (status.step === "done") {
        if (landmarks) analyzer.captureBaseline(landmarks, currentMode)
        phaseRef.current = "tracking"
        setPhase("tracking")
      }
      return
    }

    // phase === "tracking"
    const result = analyzer.processFrame(landmarks, currentMode, timestampMs)

    if (result.trackingOk) {
      setTracking((prev) => (prev ? prev : true))
    }

    if (currentMode === "Plank") {
      // Held-position timer, not a rep counter — see FrameResult's plank
      // fields in lib/exercise-analyzer.ts for why this is a separate path.
      setHoldSeconds(result.holdSeconds)
      setBestHoldSeconds(result.bestHoldSeconds) // primitive state — React
      // already skips the re-render if this is the same number as last time,
      // so no manual "did it change" check is needed here.
      if (result.formAlert) flashAndLogFormAlert(result.formAlert)
      return
    }

    // All other modes: rep/strike counting.
    if (result.repCompleted) {
      setReps((r) => r + 1)

      if (result.formScore !== null) {
        const scores = recentScoresRef.current
        scores.push(result.formScore)
        if (scores.length > ACCURACY_WINDOW) scores.shift()
        const avg = Math.round(scores.reduce((a, b) => a + b, 0) / scores.length)
        setAccuracy(avg)
      }

      if (result.cue) {
        fbId.current += 1
        setFeedback((prev) => [{ id: fbId.current, text: result.cue!.text, tone: result.cue!.tone }, ...prev].slice(0, 6))
      }
    }

    // formAlert can fire independent of rep completion (e.g. knee valgus
    // flagged mid-squat, well before the rep finishes) — checked
    // unconditionally, not nested inside the repCompleted branch above.
    if (result.formAlert) flashAndLogFormAlert(result.formAlert)

    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  function handleModeChange(next: string) {
    setMode(next as ExerciseMode)
    // resetForNewSession + re-entering "calibrating" happens in the
    // cameraOn/mode effect above, since mode is one of its dependencies.
  }

  function handleToggle(next: boolean) {
    setCameraOn(next)
    if (!next) {
      setFeedback([])
      resetForNewSession()
    }
  }

  // Punching modes count a fundamentally different, less precise signal
  // (wrist-velocity spikes) than the angle-based squat/lunge rep counter —
  // see lib/exercise-analyzer.ts. Labeling it "Strikes" instead of "Reps"
  // reflects that difference instead of presenting both as equally exact.
  const repLabel = mode === "Shadowboxing" || mode === "Jab-Cross" ? "strikes" : "reps"
  const isHoldMode = mode === "Plank"

  return (
    <main className="min-h-dvh bg-background">
      <div className="mx-auto flex min-h-dvh max-w-7xl flex-col gap-6 px-4 py-6 sm:px-6 lg:px-8">
        <DashboardHeader active={cameraOn} />

        <div className="grid flex-1 grid-cols-1 gap-6 lg:grid-cols-[1fr_22rem]">
          {/* Center stage: webcam */}
          <div className="min-h-[60vh] lg:min-h-0">
            <CameraFeed
              active={cameraOn}
              mode={mode}
              reps={reps}
              holdSeconds={isHoldMode ? holdSeconds : null}
              onToggle={handleToggle}
              onLandmarks={handlePoseLandmarks}
              calibrating={phase === "calibrating"}
              calibrationMessage={calibrationMessage}
              countdownValue={countdownValue}
              formAlert={formAlert}
            />
          </div>

          {/* Right sidebar */}
          <div className="flex min-h-0 flex-col gap-6">
            <MetricsSidebar
              active={cameraOn}
              reps={reps}
              repLabel={repLabel}
              accuracy={accuracy}
              tracking={tracking}
              mode={mode}
              modes={MODES}
              onModeChange={handleModeChange}
              isHoldMode={isHoldMode}
              holdSeconds={holdSeconds}
              bestHoldSeconds={bestHoldSeconds}
            />
            <AiFeedback active={cameraOn} items={feedback} />
          </div>
        </div>
      </div>
    </main>
  )
}
