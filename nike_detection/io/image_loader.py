"""Decode images once."""

from __future__ import annotations

import logging
from typing import Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class ImageLoadError(ValueError):
    pass


def load_bgr_gray(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load an image once and return (BGR, gray)."""
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ImageLoadError(f"Could not read image: {path}")

    if image.ndim == 2:
        gray = image
        bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.ndim == 3 and image.shape[2] == 4:
        bgr = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    elif image.ndim == 3:
        bgr = image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        raise ImageLoadError(f"Unsupported image shape {image.shape} for {path}")

    logger.debug("Loaded %s shape=%s", path, bgr.shape)
    return bgr, gray


def ensure_bgr(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image


def apply_clahe(gray: np.ndarray, clip_limit: float = 2.0,
                grid_size: tuple[int, int] = (8, 8)) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    return clahe.apply(gray)
