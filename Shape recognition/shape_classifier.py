# shape_classifier.py
"""
Real-time shape predictor.

Class:
    ShapeClassifier – loads the Keras model and predicts shape + confidence
                      with a cooldown timer and fade-out display helper.

Prompt 8 implementation.
"""

import time
import cv2  # type: ignore
import numpy as np
from config import SHAPE_CLASSES


# ---------------------------------------------------------------------------
# Optional TensorFlow import
# ---------------------------------------------------------------------------

try:
    import tensorflow as tf  # type: ignore
    _TF_AVAILABLE = True
except ImportError:
    _TF_AVAILABLE = False


# ---------------------------------------------------------------------------
# ShapeClassifier
# ---------------------------------------------------------------------------

class ShapeClassifier:
    """Load a trained Keras model and predict shape labels from canvas images.

    Falls back gracefully to a contour-based heuristic when TensorFlow is
    not installed (e.g., Python 3.14) or the model file is missing.

    Cooldown:
        Predictions are suppressed for ``cooldown_s`` seconds after each
        successful call to avoid spamming the model.

    Fade-out display:
        Call ``draw_result(frame)`` every frame; the overlay fades away
        ``display_s`` seconds after the last prediction.
    """

    COOLDOWN_S  = 1.5   # minimum seconds between predictions
    DISPLAY_S   = 2.0   # seconds the result stays visible before fading

    def __init__(self,
                 model_path:  str   = "models/shape_classifier.h5",
                 cooldown_s:  float = COOLDOWN_S,
                 display_s:   float = DISPLAY_S):
        self.model_path  = model_path
        self.cooldown_s  = cooldown_s
        self.display_s   = display_s

        self._model          = None
        self._last_pred_time = 0.0

        # Most-recent prediction result
        self._last_class:  str | None = None
        self._last_conf:   float      = 0.0
        self._result_time: float      = 0.0   # when the last prediction was made

    # ------------------------------------------------------------------
    # Model loading (lazy)
    # ------------------------------------------------------------------

    @property
    def model(self):
        if self._model is not None:
            return self._model
        if not _TF_AVAILABLE:
            return None
        try:
            self._model = tf.keras.models.load_model(self.model_path)
            print(f"[INFO] Model loaded from '{self.model_path}'")
        except Exception as e:
            print(f"[WARN] Could not load model: {e}  →  using heuristic.")
        return self._model

    # ------------------------------------------------------------------
    # Preprocessing  (Prompt 8)
    # ------------------------------------------------------------------

    def preprocess(self, canvas: np.ndarray) -> np.ndarray | None:
        """Crop, resize, and normalise the canvas for the CNN.

        1. Convert BGR → grayscale.
        2. Find bounding box of non-zero pixels.
        3. Crop to that bounding box.
        4. Resize to 28 × 28.
        5. Normalise to [0, 1].
        6. Reshape to (1, 28, 28, 1) for Keras.

        Returns:
            Float32 array of shape ``(1, 28, 28, 1)``, or ``None`` when the
            canvas is blank.
        """
        gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)

        # Edge case: blank canvas
        if cv2.countNonZero(gray) == 0:
            return None

        # Bounding box of drawn pixels
        x, y, w, h = cv2.boundingRect(gray)
        if w == 0 or h == 0:
            return None

        crop   = gray[y:y + h, x:x + w]
        resized = cv2.resize(crop, (28, 28), interpolation=cv2.INTER_AREA)
        normed  = resized.astype(np.float32) / 255.0
        return normed.reshape(1, 28, 28, 1)

    # ------------------------------------------------------------------
    # Prediction  (Prompt 8)
    # ------------------------------------------------------------------

    def predict(self, canvas: np.ndarray) -> tuple[str | None, float]:
        """Run inference and return ``(class_name, confidence_percent)``.

        Returns ``(None, 0)`` when:
        * The canvas is blank.
        * The cooldown period has not elapsed yet.

        Args:
            canvas: Raw BGR numpy array from ``Canvas.get_canvas()``.
        """
        now = time.perf_counter()
        if now - self._last_pred_time < self.cooldown_s:
            return self._last_class, self._last_conf   # still in cooldown

        processed = self.preprocess(canvas)
        if processed is None:
            return None, 0.0

        # --- TF model path ---
        m = self.model
        if m is not None:
            preds = m.predict(processed, verbose=0)[0]
            idx   = int(np.argmax(preds))
            conf  = float(preds[idx]) * 100.0
            name  = SHAPE_CLASSES[idx] if idx < len(SHAPE_CLASSES) else "unknown"
        else:
            # --- Heuristic fallback ---
            name, conf = self._heuristic(canvas)

        self._last_class     = name
        self._last_conf      = conf
        self._last_pred_time = now
        self._result_time    = now
        return name, conf

    # ------------------------------------------------------------------
    # Heuristic fallback (no model needed)
    # ------------------------------------------------------------------

    def _heuristic(self, canvas: np.ndarray) -> tuple[str, float]:
        gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh,
                                       cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return "unknown", 0.0

        cnt        = max(contours, key=cv2.contourArea)
        perimeter  = cv2.arcLength(cnt, True)
        if perimeter == 0:
            return "unknown", 0.0

        area          = cv2.contourArea(cnt)
        circularity   = 4 * np.pi * area / (perimeter ** 2)
        approx        = cv2.approxPolyDP(cnt, 0.04 * perimeter, True)
        n             = len(approx)

        if circularity > 0.75:
            return "circle",    round(circularity * 100, 1)
        if n == 3:
            return "triangle",  70.0
        if n == 4:
            return "square",    70.0
        if n >= 5:
            return "star",      65.0
        return "unknown", 0.0

    # ------------------------------------------------------------------
    # Fade-out display  (Prompt 8 / 10)
    # ------------------------------------------------------------------

    def draw_result(self, frame) -> None:
        """Overlay the last prediction result on *frame* with a fade-out.

        The label is shown for ``display_s`` seconds after the prediction,
        then fades away.  Designed to be called every frame.
        """
        if self._last_class is None:
            return

        elapsed = time.perf_counter() - self._result_time
        if elapsed > self.display_s:
            return

        # Alpha fades from 1.0 → 0.0 over the last 50 % of display time
        fade_start = self.display_s * 0.5
        if elapsed > fade_start:
            alpha = 1.0 - (elapsed - fade_start) / (self.display_s - fade_start)
        else:
            alpha = 1.0

        h, w = frame.shape[:2]
        overlay = frame.copy()

        # Background panel
        px, py = w // 2 - 130, h // 2 - 60
        cv2.rectangle(overlay, (px, py), (px + 260, py + 110),
                      (30, 30, 30), cv2.FILLED)
        cv2.rectangle(overlay, (px, py), (px + 260, py + 110),
                      (200, 200, 200), 2)

        # Class name (large)
        name = self._last_class.capitalize()
        cv2.putText(overlay, name,
                    (px + 10, py + 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4,
                    (0, 255, 180), 3, cv2.LINE_AA)

        # Confidence (smaller)
        conf_text = f"{self._last_conf:.1f}% confidence"
        cv2.putText(overlay, conf_text,
                    (px + 10, py + 95),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (180, 180, 180), 1, cv2.LINE_AA)

        cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)
