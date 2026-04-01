"""
utils/iris_preprocessor.py
All OpenCV-based iris detection and preprocessing steps.
"""

import cv2
import numpy as np
from typing import Optional, Tuple


# ── Constants ──────────────────────────────────────────────────────────────
TARGET_SIZE  = (64, 64)      # Input size expected by the CNN model
CLAHE_CLIP   = 2.0           # Contrast Limited Adaptive Histogram Equalisation
CLAHE_GRID   = (8, 8)


class IrisPreprocessor:
    """
    Detects, segments, and normalises an iris region from a captured frame.

    Pipeline
    --------
    1. Convert to grayscale
    2. CLAHE contrast enhancement
    3. Detect pupil (innermost dark circle) with HoughCircles
    4. Detect iris boundary (outer circle) with HoughCircles
    5. Doughnut-mask everything outside the iris
    6. Resize to TARGET_SIZE and normalise to [0, 1]
    """

    def __init__(self):
        self.clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_GRID)

    # ── Public API ──────────────────────────────────────────────────────────

    def preprocess(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Full preprocessing pipeline.

        Parameters
        ----------
        image : BGR or grayscale numpy array (H x W [x 3])

        Returns
        -------
        np.ndarray of shape (64, 64, 1) with values in [0, 1], or None on failure.
        """
        gray = self._to_gray(image)
        enhanced = self.clahe.apply(gray)

        iris_region = self._segment_iris(enhanced)
        if iris_region is None:
            # Fallback: use centre crop if circle detection fails
            iris_region = self._centre_crop(enhanced)

        resized    = cv2.resize(iris_region, TARGET_SIZE)
        normalised = resized.astype(np.float32) / 255.0
        return normalised.reshape(*TARGET_SIZE, 1)      # (64, 64, 1)

    def detect_eye_region(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Detect and crop the eye region from a full face frame using Haar cascade.

        Returns the cropped eye ROI (BGR) or None.
        """
        eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_eye.xml"
        )
        gray   = self._to_gray(frame)
        eyes   = eye_cascade.detectMultiScale(gray, scaleFactor=1.1,
                                               minNeighbors=5, minSize=(40, 40))
        if len(eyes) == 0:
            return None

        # Pick the largest detected eye
        eyes = sorted(eyes, key=lambda e: e[2] * e[3], reverse=True)
        x, y, w, h = eyes[0]
        return frame[y: y + h, x: x + w]

    def draw_iris_overlay(self, frame: np.ndarray) -> np.ndarray:
        """
        Return a copy of the frame with detected iris circles drawn on it.
        Useful for the live webcam feed.
        """
        gray     = self._to_gray(frame)
        enhanced = self.clahe.apply(gray)
        output   = frame.copy()

        iris_circles = self._find_circles(enhanced, minR=40, maxR=100)
        if iris_circles is not None:
            for (x, y, r) in iris_circles:
                cv2.circle(output, (x, y), r, (0, 255, 100), 2)    # iris  — green
                cv2.circle(output, (x, y), 3, (0, 0, 255),  -1)    # centre — red

        pupil_circles = self._find_circles(enhanced, minR=10, maxR=40)
        if pupil_circles is not None:
            for (x, y, r) in pupil_circles:
                cv2.circle(output, (x, y), r, (255, 165, 0), 2)    # pupil — orange

        return output

    # ── Private helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _to_gray(image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image

    def _segment_iris(self, gray: np.ndarray) -> Optional[np.ndarray]:
        """Detect iris boundary and return the masked, cropped iris region."""
        circles = self._find_circles(gray, minR=40, maxR=110)
        if circles is None:
            return None

        x, y, r = circles[0]
        mask     = np.zeros_like(gray)
        cv2.circle(mask, (x, y), r, 255, -1)

        # Optional: subtract pupil to get pure iris band
        pupil = self._find_circles(gray, minR=10, maxR=r - 5)
        if pupil is not None:
            px, py, pr = pupil[0]
            cv2.circle(mask, (px, py), pr, 0, -1)

        masked = cv2.bitwise_and(gray, gray, mask=mask)

        # Crop to bounding square around iris
        x1, y1 = max(x - r, 0), max(y - r, 0)
        x2, y2 = min(x + r, gray.shape[1]), min(y + r, gray.shape[0])
        return masked[y1:y2, x1:x2]

    @staticmethod
    def _find_circles(gray: np.ndarray,
                      minR: int, maxR: int) -> Optional[np.ndarray]:
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1.2, minDist=50,
            param1=100, param2=30,
            minRadius=minR, maxRadius=maxR,
        )
        if circles is None:
            return None
        circles = np.round(circles[0, :]).astype(int)
        return circles

    @staticmethod
    def _centre_crop(gray: np.ndarray) -> np.ndarray:
        """Return the central square of the image as a fallback."""
        h, w  = gray.shape
        side  = min(h, w)
        y_off = (h - side) // 2
        x_off = (w - side) // 2
        return gray[y_off: y_off + side, x_off: x_off + side]


# ── Convenience function ────────────────────────────────────────────────────

def preprocess_image_file(filepath: str) -> Optional[np.ndarray]:
    """Load an image from disk and run the full preprocessing pipeline."""
    img = cv2.imread(filepath)
    if img is None:
        return None
    return IrisPreprocessor().preprocess(img)


def preprocess_base64_frame(b64_data: str) -> Optional[np.ndarray]:
    """
    Decode a base64-encoded JPEG/PNG frame (from webcam) and preprocess it.

    b64_data may start with a data-URL prefix such as:
        'data:image/jpeg;base64,/9j/4AAQ...'
    or be raw base64.
    """
    import base64

    if "," in b64_data:
        b64_data = b64_data.split(",", 1)[1]

    try:
        raw   = base64.b64decode(b64_data)
        arr   = np.frombuffer(raw, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            return None
        return IrisPreprocessor().preprocess(frame)
    except Exception:
        return None
