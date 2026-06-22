# main.py
"""
Gesture Canvas — full integrated pipeline.

Prompts 9 + 10: hand detection → gesture mode → draw / erase / predict
+ UI polish (palette bar, pinch thickness, thumbnail, countdown ring, HUD).

Controls
--------
    q         – quit
    c         – clear canvas manually
    Esc       – quit
"""

import time
import math
import cv2          # type: ignore
import numpy as np

from config          import CONFIG
from gesture_detector import HandDetector, get_gesture_mode, draw_mode_badge
from canvas           import Canvas, PALETTE
from shape_classifier import ShapeClassifier


# ---------------------------------------------------------------------------
# FPS counter
# ---------------------------------------------------------------------------

class FPSCounter:
    def __init__(self, smoothing: int = 15):
        self._times: list[float] = []
        self._n = smoothing

    def tick(self) -> float:
        now = time.perf_counter()
        self._times.append(now)
        if len(self._times) > self._n:
            self._times.pop(0)
        if len(self._times) < 2:
            return 0.0
        dt = self._times[-1] - self._times[0]
        return (len(self._times) - 1) / dt if dt > 0 else 0.0


# ---------------------------------------------------------------------------
# Countdown ring (Prompt 10)
# ---------------------------------------------------------------------------

PREDICT_HOLD_S = 3.0   # seconds the ✌ gesture must be held to trigger predict


def draw_countdown(frame, elapsed: float, total: float = PREDICT_HOLD_S) -> None:
    """Draw a circular fill-up countdown centred on the frame."""
    progress = min(elapsed / total, 1.0)
    h, w     = frame.shape[:2]
    cx, cy   = w // 2, h // 2
    radius   = 50
    angle    = int(360 * progress)

    # Background circle
    cv2.circle(frame, (cx, cy), radius, (50, 50, 50), 4)
    # Arc (drawn as series of points for cv2 compatibility)
    axes = (radius, radius)
    cv2.ellipse(frame, (cx, cy), axes, -90, 0, angle,
                (0, 220, 255), 4, cv2.LINE_AA)

    pct_text = f"{int(progress * 100)}%"
    cv2.putText(frame, pct_text,
                (cx - 18, cy + 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                (0, 220, 255), 2, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# HUD panel (Prompt 10)
# ---------------------------------------------------------------------------

def draw_hud(frame, mode: str, color_bgr, thickness: int,
             last_result: str | None, last_conf: float) -> None:
    """Semi-transparent info panel in the top-right corner."""
    h, w    = frame.shape[:2]
    panel_w = 220
    panel_h = 120
    x1      = w - panel_w - 10
    y1      = 55       # below the mode badge
    x2      = w - 10
    y2      = y1 + panel_h

    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (20, 20, 20), cv2.FILLED)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    # Border
    cv2.rectangle(frame, (x1, y1), (x2, y2), (100, 100, 100), 1)

    font  = cv2.FONT_HERSHEY_SIMPLEX
    lh    = 24   # line height
    tx    = x1 + 8
    ty    = y1 + lh

    # Mode
    cv2.putText(frame, f"Mode : {mode}", (tx, ty),
                font, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
    ty += lh

    # Active colour swatch  (small filled rectangle)
    cv2.rectangle(frame, (tx, ty - 12), (tx + 20, ty + 4), color_bgr, cv2.FILLED)
    cv2.putText(frame, "Color", (tx + 26, ty),
                font, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
    ty += lh

    # Thickness
    cv2.putText(frame, f"Thick: {thickness}px", (tx, ty),
                font, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
    ty += lh

    # Last prediction
    if last_result:
        result_txt = f"{last_result}  {last_conf:.0f}%"
    else:
        result_txt = "–"
    cv2.putText(frame, f"Shape: {result_txt}", (tx, ty),
                font, 0.5, (0, 255, 180), 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # ---- Initialise components -------------------------------------------
    detector   = HandDetector(
        min_detection_confidence=CONFIG["min_detection_confidence"],
        min_tracking_confidence=CONFIG["min_tracking_confidence"],
    )
    canvas     = Canvas(
        width=CONFIG["canvas_width"],
        height=CONFIG["canvas_height"],
        color=CONFIG["drawing_color"],
    )
    classifier = ShapeClassifier()
    fps_ctr    = FPSCounter()

    # ---- Open camera ------------------------------------------------------
    cap = cv2.VideoCapture(CONFIG["camera_index"])
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {CONFIG['camera_index']}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CONFIG["canvas_width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG["canvas_height"])

    WIN = "Gesture Canvas"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, CONFIG["canvas_width"], CONFIG["canvas_height"])

    # ---- State -----------------------------------------------------------
    active_palette_idx = 1           # Green by default (index into PALETTE)
    canvas.color       = PALETTE[active_palette_idx][1]

    mode               = "idle"
    prev_mode          = "idle"
    cooldown_active    = False       # True after a predict fires, until mode changes

    predict_hold_start = 0.0         # when ✌ gesture first held
    predict_triggered  = False

    last_class: str | None = None
    last_conf:  float      = 0.0

    print("[INFO] Gesture Canvas running.  Press 'q' or Esc to quit, 'c' to clear.")

    # ---- Capture loop ----------------------------------------------------
    while True:
        ok, frame = cap.read()
        if not ok:
            continue

        # 1. Mirror --------------------------------------------------------
        frame = cv2.flip(frame, 1)

        # 2. Hand detection ------------------------------------------------
        frame = detector.find_hands(frame, draw=True)

        # 3. Resolve gesture mode -----------------------------------------
        fingers = detector.fingers_up()
        mode    = get_gesture_mode(fingers) if fingers else "idle"

        # 4. Get index fingertip (landmark 8) pixel position ---------------
        tip_pos = detector.get_landmark_position(8)

        # 5. Colour palette hover (Prompt 10) ------------------------------
        active_palette_idx, hover_progress = canvas.update_palette(
            tip_pos, active_palette_idx
        )
        # Draw hover progress arc over the hovered swatch
        if hover_progress > 0.0 and tip_pos is not None:
            tx, ty  = tip_pos
            hovered = tx // 60   # _SWATCH_W = 60
            if 0 <= hovered < len(PALETTE) and ty < 40:
                cx = hovered * 60 + 30
                angle = int(360 * hover_progress)
                cv2.ellipse(frame, (cx, 20), (18, 18), -90, 0, angle,
                            (255, 255, 255), 2, cv2.LINE_AA)

        # 6. Pen thickness from pinch distance (Prompt 10) -----------------
        pinch_dist = detector.distance_between(4, 8)
        canvas.update_thickness(pinch_dist)

        # 7. Act on gesture mode -------------------------------------------
        if mode == "draw" and tip_pos:
            canvas.draw(tip_pos[0], tip_pos[1])
        else:
            canvas.stop_stroke()

        if mode == "erase":
            canvas.clear()

        # 8. Countdown + prediction trigger (Prompt 10) -------------------
        if mode == "predict":
            now = time.perf_counter()
            if prev_mode != "predict":
                predict_hold_start = now
                predict_triggered  = False
                cooldown_active    = False

            hold_elapsed = now - predict_hold_start
            draw_countdown(frame, hold_elapsed, PREDICT_HOLD_S)

            if hold_elapsed >= PREDICT_HOLD_S and not predict_triggered:
                pred_class, pred_conf = classifier.predict(canvas.get_canvas())
                if pred_class:
                    last_class, last_conf = pred_class, pred_conf
                predict_triggered = True
                cooldown_active   = True
        else:
            # Reset countdown when gesture changes
            predict_hold_start = time.perf_counter()
            predict_triggered  = False

        prev_mode = mode

        # 9. Overlay canvas -----------------------------------------------
        frame = canvas.overlay(frame)

        # 10. UI chrome ---------------------------------------------------
        canvas.draw_palette(frame, active_palette_idx)
        draw_mode_badge(frame, mode)
        canvas.draw_thumbnail(frame)
        classifier.draw_result(frame)
        draw_hud(frame, mode, canvas.color, canvas.thickness, last_class, last_conf)

        # FPS counter
        fps = fps_ctr.tick()
        cv2.putText(frame, f"FPS {fps:.0f}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.75, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, f"FPS {fps:.0f}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.75, (0, 255, 0), 2, cv2.LINE_AA)

        # 11. Display -------------------------------------------------
        cv2.imshow(WIN, frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):           # q or Esc
            break
        if key == ord("c"):
            canvas.clear()
            last_class = None

        if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
            break

    # ---- Cleanup --------------------------------------------------------
    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Exited cleanly.")


if __name__ == "__main__":
    main()
