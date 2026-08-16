"""
Material profile helpers for the clear scan material.

The clear material scans have a mid-gray background (instead of white paper),
lighter/fainter ink, and a lower signal-to-noise ratio. Fixed grayscale
thresholds tuned for white paper (e.g. ink < 127, background > 180) are
meaningless there, so clear-mode detectors derive every threshold from the
measured background level of the actual image.
"""

import numpy as np


def estimate_background_level(gray, subsample=8):
    """Estimate the background gray level of a scan.

    The background dominates the image area (ink covers only a small
    fraction), so the median of a subsampled view is a robust estimate that
    ignores ink, debris, and speckle noise.

    Args:
        gray: Grayscale image (uint8).
        subsample: Row/column subsampling step for speed on large scans.

    Returns:
        Background gray level as a float.
    """
    return float(np.median(gray[::subsample, ::subsample]))


def ink_threshold_below_background(gray, offset, minimum=5, maximum=250):
    """Threshold `offset` gray levels below the measured background.

    Args:
        gray: Grayscale image (uint8).
        offset: How far below the background level the threshold sits.
        minimum: Lower clamp for the returned threshold.
        maximum: Upper clamp for the returned threshold.

    Returns:
        Integer threshold; pixels below it should be treated as ink/defect.
    """
    background = estimate_background_level(gray)
    return int(np.clip(background - offset, minimum, maximum))
