# gesture_detector.py
"""
Hand landmark detection and gesture mode classification.

Classes:
    HandDetector – wraps MediaPipe Hands; detects landmarks and classifies fingers.

Functions:
    get_gesture_mode(fingers) – maps a fingers-up list to a named mode string.
    draw_mode_badge(frame, mode) – overlays a colored mode badge on the frame.
"""

import cv2  # type: ignore
import math
from config import CONFIG

try:
    import mediapipe as mp  # type: ignore
    if not hasattr(mp, "solutions"):
        raise ImportError(
            "Installed mediapipe package does not expose mp.solutions. "
            "Use Python 3.10 / 3.11 / 3.12 with mediapipe==0.10.35."
        )
except Exception as exc:
    raise SystemExit(
        "MediaPipe import failed. This project requires the classic MediaPipe solutions API.\n"
        "Please install a supported Python version and MediaPipe package:\n"
        "    python -m pip install mediapipe==0.10.35\n"
        "If you are using Python 3.14, switch to Python 3.12 or lower."
    ) from exc


# ---------------------------------------------------------------------------
# Gesture mode helpers
# ---------------------------------------------------------------------------

# Colour palette for badge overlay  (BGR)
_MODE_COLOURS = {
    "draw":    (0,   200,  0),    # green
    "erase":   (0,   0,   220),   # red
    "predict": (180, 0,   180),   # purple
    "idle":    (120, 120, 120),   # grey
}


def get_gesture_mode(fingers: list[bool]) -> str:
    """Map a five-element fingers-up list to a gesture mode string.

    Args:
        fingers: [thumb, index, middle, ring, pinky] — True = finger raised.

    Returns:
        One of: ``"draw"``, ``"erase"``, ``"predict"``, ``"idle"``.
    """
    thumb, index, middle, ring, pinky = fingers

    if index and not middle and not ring and not pinky:
        return "draw"                      # ☝  index only
    if not index and not middle and not ring and not pinky:
        return "erase"                     # ✊  fist / all down
    if index and middle and not ring and not pinky:
        return "predict"                   # ✌  index + middle
    return "idle"


def draw_mode_badge(frame, mode: str) -> None:
    """Draw a coloured rounded-rectangle badge showing the current mode.

    Placed in the top-right corner of *frame* (in-place).
    """
    colour = _MODE_COLOURS.get(mode, _MODE_COLOURS["idle"])
    label  = mode.upper()
    font   = cv2.FONT_HERSHEY_SIMPLEX
    scale  = 0.75
    thick  = 2

    (tw, th), _ = cv2.getTextSize(label, font, scale, thick)
    pad = 10
    h, w = frame.shape[:2]

    x1 = w - tw - pad * 2 - 10
    y1 = 10
    x2 = w - 10
    y2 = y1 + th + pad * 2

    # Filled badge
    cv2.rectangle(frame, (x1, y1), (x2, y2), colour, cv2.FILLED)
    # Border
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 1)
    # Text (white)
    cv2.putText(frame, label,
                (x1 + pad, y2 - pad),
                font, scale, (255, 255, 255), thick, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# HandDetector
# ---------------------------------------------------------------------------

class HandDetector:
    """Wraps MediaPipe Hands for landmark detection and finger-state analysis.

    Usage::

        detector = HandDetector()
        while True:
            success, frame = cap.read()
            frame = detector.find_hands(frame)
            pos = detector.get_landmark_position(frame, 8)   # index tip
            fingers = detector.fingers_up()
            mode = get_gesture_mode(fingers)
    """

    # MediaPipe landmark IDs for each fingertip and its PIP (knuckle) joint
    _TIP_IDS = [4,  8,  12, 16, 20]
    _PIP_IDS = [3,  6,  10, 14, 18]   # one joint below each tip (MCP for thumb)

    def __init__(
        self,
        max_hands: int = 1,
        min_detection_confidence: float = 0.8,
        min_tracking_confidence:  float = 0.7,
    ):
        self._mp_hands = mp.solutions.hands
        self._mp_draw  = mp.solutions.drawing_utils
        self._mp_style = mp.solutions.drawing_styles

        self.hands = self._mp_hands.Hands(
            max_num_hands=max_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

        # Populated by find_hands(); consumed by fingers_up() / get_landmark_position()
        self._landmarks = []   # raw landmark list for the first hand
        self._h = 0
        self._w = 0

    # ------------------------------------------------------------------
    # Core detection
    # ------------------------------------------------------------------

    def find_hands(self, frame, draw: bool = True):
        """Run MediaPipe hand detection and optionally draw all 21 landmarks.

        Args:
            frame: BGR numpy array from OpenCV.
            draw:  Whether to draw landmarks and connections on the frame.

        Returns:
            The annotated frame (or the original if ``draw=False``).
        """
        self._h, self._w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self.hands.process(rgb)
        rgb.flags.writeable = True

        self._landmarks = []

        if results.multi_hand_landmarks:
            # Use only the first detected hand
            hand_lms = results.multi_hand_landmarks[0]
            self._landmarks = hand_lms.landmark

            if draw:
                self._mp_draw.draw_landmarks(
                    frame,
                    hand_lms,
                    self._mp_hands.HAND_CONNECTIONS,
                    self._mp_style.get_default_hand_landmarks_style(),
                    self._mp_style.get_default_hand_connections_style(),
                )

        return frame

    # ------------------------------------------------------------------
    # Landmark access
    # ------------------------------------------------------------------

    def get_landmark_position(self, landmark_id: int):
        """Return the (x, y) pixel position of a landmark, or None if no hand.

        Args:
            landmark_id: 0–20 (MediaPipe hand landmark index).

        Returns:
            ``(x, y)`` tuple of ints, or ``None`` when no hand is detected.
        """
        if not self._landmarks:
            return None
        lm = self._landmarks[landmark_id]
        return int(lm.x * self._w), int(lm.y * self._h)

    def hand_detected(self) -> bool:
        """Return True if at least one hand was found in the last frame."""
        return bool(self._landmarks)

    # ------------------------------------------------------------------
    # Finger state
    # ------------------------------------------------------------------

    def fingers_up(self) -> list[bool]:
        """Return a list of five booleans indicating which fingers are raised.

        Order: ``[thumb, index, middle, ring, pinky]``.
        A finger is considered "up" when its tip is above its PIP joint
        (lower y-value in image coordinates).

        Returns an empty list when no hand is detected.
        """
        if not self._landmarks:
            return []

        lm = self._landmarks
        fingers = []

        # Thumb: compare x-coordinates (tip must be to the LEFT of IP joint
        # for a right hand facing the camera in mirror mode)
        fingers.append(lm[self._TIP_IDS[0]].x < lm[self._PIP_IDS[0]].x)

        # Index → Pinky: compare y-coordinates (tip y < PIP y → finger up)
        for tip, pip in zip(self._TIP_IDS[1:], self._PIP_IDS[1:]):
            fingers.append(lm[tip].y < lm[pip].y)

        return fingers

    # ------------------------------------------------------------------
    # Derived measurements
    # ------------------------------------------------------------------

    def distance_between(self, id1: int, id2: int):
        """Return the Euclidean pixel distance between two landmarks, or None."""
        p1 = self.get_landmark_position(id1)
        p2 = self.get_landmark_position(id2)
        if p1 is None or p2 is None:
            return None
        return math.hypot(p2[0] - p1[0], p2[1] - p1[1])
