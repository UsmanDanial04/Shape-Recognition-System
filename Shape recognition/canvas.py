# canvas.py
"""
Drawing canvas engine.

Class:
    Canvas – numpy-backed drawing surface with overlay, clear, and thumbnail support.

Prompt 5 implementation + Prompt 10 polish (color palette, thickness, thumbnail).
"""

import cv2  # type: ignore
import numpy as np
import time


# ---------------------------------------------------------------------------
# Color palette (BGR tuples shown to the user)
# ---------------------------------------------------------------------------

PALETTE: list[tuple[str, tuple]] = [
    ("Red",    (0,   0,   255)),
    ("Green",  (0,   200, 0  )),
    ("Blue",   (255, 80,  0  )),
    ("Yellow", (0,   220, 220)),
    ("White",  (255, 255, 255)),
]

_SWATCH_W  = 60   # width of each colour swatch in pixels
_SWATCH_H  = 40   # height of the palette bar


class Canvas:
    """Transparent drawing canvas that lives on top of the webcam feed.

    Features
    --------
    * Continuous stroke drawing with ``draw()``.
    * Full-canvas clear with ``clear()``.
    * ``cv2.addWeighted`` blending via ``overlay()``.
    * Raw canvas access via ``get_canvas()``.
    * Live colour-palette bar (5 swatches) — hover the index finger for 1 s
      to select a colour — via ``update_palette()``.
    * Dynamic pen thickness tied to thumb–index distance via
      ``update_thickness()``.
    * Bottom-right thumbnail via ``draw_thumbnail()``.
    """

    THUMB_SIZE = 150           # thumbnail side length in pixels
    MIN_THICKNESS = 3
    MAX_THICKNESS = 20
    HOVER_DWELL_S = 1.0        # seconds to hover before colour changes

    def __init__(self, width: int, height: int,
                 color=(0, 255, 0), thickness: int = 5, alpha: float = 0.4):
        self.width     = width
        self.height    = height
        self.color     = color        # active drawing colour (BGR)
        self.thickness = thickness    # active pen thickness
        self.alpha     = alpha        # canvas blend weight

        # Internal drawing surface
        self._canvas = np.zeros((height, width, 3), np.uint8)

        # Stroke continuity
        self.prev_x: int | None = None
        self.prev_y: int | None = None

        # Palette hover tracking
        self._hover_idx:   int | None = None
        self._hover_start: float      = 0.0

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def draw(self, x: int, y: int,
             color=None, thickness: int | None = None) -> None:
        """Draw a continuous line from the previous point to (x, y).

        If there is no previous point (start of stroke) a single dot is drawn.
        """
        c = color     or self.color
        t = thickness or self.thickness

        if self.prev_x is not None and self.prev_y is not None:
            cv2.line(self._canvas,
                     (self.prev_x, self.prev_y), (x, y),
                     c, t, cv2.LINE_AA)
        else:
            # First point of a new stroke — draw a dot
            cv2.circle(self._canvas, (x, y), max(t // 2, 1), c, cv2.FILLED)

        self.prev_x, self.prev_y = x, y

    def stop_stroke(self) -> None:
        """Call when the drawing gesture ends to break stroke continuity."""
        self.prev_x = None
        self.prev_y = None

    def clear(self) -> None:
        """Reset the canvas to black and clear stroke state."""
        self._canvas[:] = 0
        self.stop_stroke()

    # ------------------------------------------------------------------
    # Compositing
    # ------------------------------------------------------------------

    def overlay(self, frame):
        """Blend the canvas onto *frame* and return the composited result.

        Uses ``cv2.addWeighted`` so the webcam feed shows through wherever
        the canvas is dark.
        """
        return cv2.addWeighted(frame, 1.0,
                               self._canvas, self.alpha, 0)

    def get_canvas(self) -> np.ndarray:
        """Return the raw canvas array (for preprocessing / saving)."""
        return self._canvas.copy()

    # ------------------------------------------------------------------
    # Palette bar (Prompt 10)
    # ------------------------------------------------------------------

    def draw_palette(self, frame, active_idx: int | None = None) -> None:
        """Draw the colour palette bar at the top of *frame* (in-place).

        Args:
            frame:      BGR frame to annotate.
            active_idx: Index of the currently selected colour (highlighted).
        """
        for i, (name, bgr) in enumerate(PALETTE):
            x1 = i * _SWATCH_W
            y1 = 0
            x2 = x1 + _SWATCH_W
            y2 = _SWATCH_H

            cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, cv2.FILLED)

            # White border for active swatch
            border_c = (255, 255, 255) if i == active_idx else (60, 60, 60)
            border_t = 3               if i == active_idx else 1
            cv2.rectangle(frame, (x1, y1), (x2, y2), border_c, border_t)

    def update_palette(self, tip_pos: tuple | None,
                       active_idx: int) -> tuple[int, float]:
        """Check whether the index fingertip hovers over a colour swatch.

        Args:
            tip_pos:    (x, y) pixel position of the index fingertip, or None.
            active_idx: Currently active palette index.

        Returns:
            ``(new_active_idx, hover_progress_0_to_1)``
        """
        if tip_pos is None:
            self._hover_idx   = None
            self._hover_start = 0.0
            return active_idx, 0.0

        tx, ty = tip_pos
        if ty > _SWATCH_H:
            # Finger not over palette bar
            self._hover_idx   = None
            self._hover_start = 0.0
            return active_idx, 0.0

        hovered = tx // _SWATCH_W
        if hovered < 0 or hovered >= len(PALETTE):
            return active_idx, 0.0

        now = time.perf_counter()
        if hovered != self._hover_idx:
            self._hover_idx   = hovered
            self._hover_start = now

        elapsed  = now - self._hover_start
        progress = min(elapsed / self.HOVER_DWELL_S, 1.0)

        if progress >= 1.0:
            # Commit colour change
            self.color = PALETTE[hovered][1]
            return hovered, 1.0

        return active_idx, progress

    # ------------------------------------------------------------------
    # Thickness from pinch (Prompt 10)
    # ------------------------------------------------------------------

    def update_thickness(self, pinch_dist: float | None) -> int:
        """Map thumb–index distance to pen thickness and store it.

        Args:
            pinch_dist: Pixel distance between landmark 4 (thumb) and 8
                        (index), or None if no hand detected.

        Returns:
            The new ``self.thickness`` value.
        """
        if pinch_dist is None:
            return self.thickness

        # Clamp and map distance [20, 150] → thickness [MIN, MAX]
        clamped = max(20.0, min(pinch_dist, 150.0))
        t = int(self.MIN_THICKNESS +
                (clamped - 20) / (150 - 20) *
                (self.MAX_THICKNESS - self.MIN_THICKNESS))
        self.thickness = t
        return t

    # ------------------------------------------------------------------
    # Thumbnail (Prompt 10)
    # ------------------------------------------------------------------

    def draw_thumbnail(self, frame) -> None:
        """Draw a small canvas preview in the bottom-right corner of *frame*."""
        thumb = cv2.resize(self._canvas,
                           (self.THUMB_SIZE, self.THUMB_SIZE))
        h, w = frame.shape[:2]
        x1 = w - self.THUMB_SIZE - 10
        y1 = h - self.THUMB_SIZE - 10
        x2 = x1 + self.THUMB_SIZE
        y2 = y1 + self.THUMB_SIZE

        # Semi-transparent border
        cv2.rectangle(frame, (x1 - 2, y1 - 2), (x2 + 2, y2 + 2),
                      (200, 200, 200), 1)
        frame[y1:y2, x1:x2] = thumb
