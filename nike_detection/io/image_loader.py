"""Decode images once."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Above this file size a full scan is memory-mapped instead of decoded: a
# 2400 DPI four-colour sheet is ~6.7 GB of pixels, which no amount of RAM
# makes pleasant and which OpenCV refuses outright past 2^30 pixels.
MEMMAP_MIN_BYTES = 1_500_000_000


class ImageLoadError(ValueError):
    pass


@dataclass
class Scan:
    """A full press scan, which may be far too large to hold in memory.

    ``data`` is either a decoded array or a read-only memory map, so callers
    must slice it before doing anything dense. ``rgb`` records that the
    channels arrived in R,G,B order (as TIFF stores them) rather than
    OpenCV's B,G,R; ``crop_bgr`` is the only place that has to care.
    """

    data: np.ndarray
    rgb: bool = False
    memmap: bool = False
    path: str = ""

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.data.shape

    def crop_bgr(self, x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
        piece = np.ascontiguousarray(self.data[y0:y1, x0:x1])
        if piece.ndim == 2:
            return cv2.cvtColor(piece, cv2.COLOR_GRAY2BGR)
        if piece.shape[2] == 4:
            piece = piece[:, :, :3]
        if self.rgb:
            piece = piece[:, :, ::-1]
        return np.ascontiguousarray(piece)


def open_scan(path: str) -> Scan:
    """Open a full scan, memory-mapping it when it is too big to decode."""
    size = os.path.getsize(path)
    if size >= MEMMAP_MIN_BYTES:
        mapped = _memmap_tiff(path)
        if mapped is not None:
            logger.info(
                "Memory-mapped %s (%.1f GB, %s)",
                os.path.basename(path), size / 1e9, mapped.data.shape,
            )
            return mapped
        logger.warning(
            "%s is %.1f GB but could not be memory-mapped; decoding it in full",
            os.path.basename(path), size / 1e9,
        )
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ImageLoadError(
            f"Could not read image: {path}"
            + (
                " -- it may exceed OpenCV's pixel limit; set the environment "
                "variable OPENCV_IO_MAX_IMAGE_PIXELS before running"
                if size >= MEMMAP_MIN_BYTES else ""
            )
        )
    return Scan(data=image, rgb=False, memmap=False, path=path)


def _memmap_tiff(path: str) -> Optional[Scan]:
    """Map an uncompressed TIFF's pixels straight off disk."""
    if not path.lower().endswith((".tif", ".tiff")):
        return None
    try:
        import tifffile
    except ImportError:
        logger.warning("tifffile is not installed, so large scans cannot be mapped")
        return None
    try:
        data = tifffile.memmap(path, mode="r")
    except Exception as exc:  # compressed, tiled or otherwise not contiguous
        logger.warning("Cannot memory-map %s: %s", os.path.basename(path), exc)
        return None
    if data.ndim == 2:
        return Scan(data=data, rgb=False, memmap=True, path=path)
    if data.ndim == 3 and data.shape[2] in (3, 4):
        return Scan(data=data, rgb=True, memmap=True, path=path)
    logger.warning("Unsupported memory-mapped shape %s for %s", data.shape, path)
    return None


def load_bgr_gray(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load an image once and return (BGR, gray).

    Full press sheets must go through ``open_scan`` instead: OpenCV refuses
    images past ``CV_IO_MAX_IMAGE_PIXELS`` (~1.07 GP), which a 2400 DPI
    four-colour scan exceeds.
    """
    size = os.path.getsize(path)
    if size >= MEMMAP_MIN_BYTES:
        raise ImageLoadError(
            f"{os.path.basename(path)} is {size / 1e9:.1f} GB — too large for "
            "OpenCV. It will be treated as a full scan and memory-mapped."
        )
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
