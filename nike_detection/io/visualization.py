"""Visualization writers. Detectors score; this module draws and saves."""

from __future__ import annotations

import logging
from typing import Any, List, Optional

import cv2
import numpy as np

from nike_detection.io.image_saver import save_image
from nike_detection.pipeline.context import ImageContext
from nike_detection.pipeline.types import Defect

logger = logging.getLogger(__name__)

VOID_TIFF_KEYS = {"void"}


def downscale(image: np.ndarray, max_edge: int = 8000) -> np.ndarray:
    h, w = image.shape[:2]
    longest = max(h, w)
    if longest <= max_edge:
        return image
    scale = max_edge / float(longest)
    return cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def save_visualization(
    ctx: ImageContext,
    detector: Any,
    defects: List[Defect],
    output_dir: str,
    detector_key: str,
    write: bool = True,
    downscale_vis: bool = False,
) -> Optional[str]:
    if not write:
        return None
    vis = detector.render(ctx, defects)
    if vis is None:
        return None
    if downscale_vis:
        vis = downscale(vis)
    if detector_key in VOID_TIFF_KEYS:
        path = f"{output_dir}/{detector_key}_visualization.tiff"
        ok = cv2.imwrite(path, vis)
        return path if ok else None
    return save_image(output_dir, detector_key, vis, "visualization") or None


def save_debug(detector: Any, output_dir: str, base_name: str, enabled: bool) -> None:
    if not enabled:
        return
    impl = getattr(detector, "_impl", None)
    if impl is not None and hasattr(impl, "save_debug_images"):
        try:
            impl.save_debug_images(output_dir, base_name)
        except Exception:
            logger.exception("Failed to save debug images for %s", base_name)
