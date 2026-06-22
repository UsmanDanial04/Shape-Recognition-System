import cv2  # type: ignore
import math
import numpy as np
from config import CANVAS_WIDTH, CANVAS_HEIGHT, DRAWING_COLOR


def euclidean_distance(p1, p2) -> float:
    """Return Euclidean distance between two (x, y) points."""
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def landmark_to_pixel(landmark, frame_width: int, frame_height: int):
    """Convert a MediaPipe NormalizedLandmark to pixel (x, y) coordinates."""
    return int(landmark.x * frame_width), int(landmark.y * frame_height)


def put_text_with_bg(frame, text: str, origin, font_scale=0.7, thickness=2,
                     text_color=(255, 255, 255), bg_color=(0, 0, 0)):
    """Draw text with a filled background rectangle for readability."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = origin
    cv2.rectangle(frame, (x - 4, y - th - 4), (x + tw + 4, y + baseline + 4),
                  bg_color, cv2.FILLED)
    cv2.putText(frame, text, (x, y), font, font_scale, text_color, thickness,
                cv2.LINE_AA)


def preprocess_canvas(image: np.ndarray, size=(64, 64)) -> np.ndarray:
    """Resize, convert to grayscale, and normalise a canvas image for ML input.

    Returns a float32 array of shape (1, H, W, 1) with values in [0, 1].
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    resized = cv2.resize(gray, size).astype(np.float32) / 255.0
    return resized[np.newaxis, :, :, np.newaxis]  # (1, H, W, 1)
