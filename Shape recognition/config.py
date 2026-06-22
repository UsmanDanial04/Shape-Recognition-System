# config.py
"""
Central configuration — all tunable constants in one place.
Import individual names or the CONFIG dict from any module.
"""

# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------
CAMERA_INDEX = 0               # webcam device index

# ---------------------------------------------------------------------------
# Canvas / window
# ---------------------------------------------------------------------------
CANVAS_WIDTH  = 640
CANVAS_HEIGHT = 480

# ---------------------------------------------------------------------------
# Drawing defaults
# ---------------------------------------------------------------------------
DRAWING_COLOR = (0, 255, 0)    # BGR green
DRAW_COLOR    = DRAWING_COLOR  # backward-compat alias
MIN_THICKNESS = 3
MAX_THICKNESS = 20

# ---------------------------------------------------------------------------
# MediaPipe thresholds
# ---------------------------------------------------------------------------
MIN_DETECTION_CONFIDENCE = 0.8
MIN_TRACKING_CONFIDENCE  = 0.7

# ---------------------------------------------------------------------------
# Shape classes  (must match the order used during training)
# ---------------------------------------------------------------------------
SHAPE_CLASSES = ["circle", "square", "triangle", "star"]

# ---------------------------------------------------------------------------
# Convenience dict
# ---------------------------------------------------------------------------
CONFIG = {
    "camera_index":              CAMERA_INDEX,
    "canvas_width":              CANVAS_WIDTH,
    "canvas_height":             CANVAS_HEIGHT,
    "drawing_color":             DRAWING_COLOR,
    "min_detection_confidence":  MIN_DETECTION_CONFIDENCE,
    "min_tracking_confidence":   MIN_TRACKING_CONFIDENCE,
    "shape_classes":             SHAPE_CLASSES,
}
