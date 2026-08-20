"""Shared stripe geometry: edge walk, LAB bounds, CLAHE. Built once per region."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np


@dataclass
class StripeBounds:
    x_left: int
    x_right: int


@dataclass
class StripeEdges:
    col_mean: np.ndarray
    mid: float
    ink_mask: np.ndarray
    raw_left: np.ndarray
    raw_right: np.ndarray
    subpixel_left: np.ndarray
    subpixel_right: np.ndarray
    median_left: np.ndarray
    median_right: np.ndarray
    gx_left: int
    gx_right: int


def median_filter_1d(values: np.ndarray, window: int) -> np.ndarray:
    n = len(values)
    win = max(3, int(window) | 1)
    valid = ~np.isnan(values)
    if valid.sum() <= win:
        return values.copy()
    filled = values.copy()
    xs = np.flatnonzero(valid)
    filled[~valid] = np.interp(np.flatnonzero(~valid), xs, values[valid])
    pad = win // 2
    padded = np.pad(filled, pad, mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, win)
    smoothed = np.median(windows, axis=1).astype(np.float64)
    out = np.full(n, np.nan, dtype=np.float64)
    out[valid] = smoothed[valid]
    return out


def find_stripe_bounds(lab: np.ndarray) -> Optional[StripeBounds]:
    """Column chroma bounds with a dark fallback (void / debris_stripe)."""
    l_chan, a_chan, b_chan = cv2.split(lab)
    a_f = a_chan.astype(np.float32) - 128.0
    b_f = b_chan.astype(np.float32) - 128.0
    col_chroma = np.sqrt(a_f * a_f + b_f * b_f).mean(axis=0)
    chroma_threshold = max(float(col_chroma.max()) * 0.40, 15.0)
    stripe_cols = col_chroma > chroma_threshold

    if stripe_cols.sum() < max(20, int(0.10 * len(col_chroma))):
        col_l = l_chan.mean(axis=0)
        l_threshold = np.median(col_l) - 0.35 * (np.median(col_l) - np.min(col_l))
        stripe_cols = col_l < l_threshold

    idx = np.where(stripe_cols)[0]
    if len(idx) == 0:
        return None
    x_left, x_right = int(idx.min()), int(idx.max())
    if x_right - x_left < 30:
        return None
    return StripeBounds(x_left=x_left, x_right=x_right)


def _subpixel_refine(gray: np.ndarray, integer_edge: np.ndarray,
                     mid: float, side: str) -> np.ndarray:
    h, w = gray.shape
    ys = np.arange(h)
    valid = integer_edge >= 0
    x = np.clip(integer_edge, 1, w - 2)
    if side == "left":
        g0 = gray[ys, x - 1].astype(np.float64)
        g1 = gray[ys, x].astype(np.float64)
        denom = g1 - g0
        t = (mid - g0) / np.where(np.abs(denom) < 1e-3, np.nan, denom)
        t = np.clip(t, -1.5, 1.5)
        refined = (x - 1) + t
    else:
        g0 = gray[ys, x].astype(np.float64)
        g1 = gray[ys, x + 1].astype(np.float64)
        denom = g1 - g0
        t = (mid - g0) / np.where(np.abs(denom) < 1e-3, np.nan, denom)
        t = np.clip(t, -1.5, 1.5)
        refined = x + t
    out = np.full(h, np.nan, dtype=np.float64)
    ok = valid & ~np.isnan(refined)
    out[ok] = refined[ok]
    fallback = valid & np.isnan(refined)
    out[fallback] = integer_edge[fallback].astype(np.float64)
    return out


def extract_stripe_edges(gray: np.ndarray, median_window: int = 31) -> Optional[StripeEdges]:
    """One interior-anchored edge walk feeding misalignment and roughness."""
    col_mean = gray.mean(axis=0)
    lo, hi = float(col_mean.min()), float(col_mean.max())
    if hi - lo < 20:
        return None
    mid = (lo + hi) / 2.0

    dark_cols = np.flatnonzero(col_mean < mid)
    if dark_cols.size < 10:
        return None
    gx_left, gx_right = int(dark_cols.min()), int(dark_cols.max())
    stripe_w = gx_right - gx_left
    inset = max(30, stripe_w // 6)
    anchor_l = min(gx_left + inset, gray.shape[1] - 1)
    anchor_r = max(gx_right - inset, 0)

    mask = gray < mid

    corridor = mask[:, :anchor_l + 1][:, ::-1]
    run = np.logical_and.accumulate(corridor, axis=1).sum(axis=1)
    left_i = (anchor_l - run + 1).astype(np.int32)
    left_i[run == 0] = -1

    corridor = mask[:, anchor_r:]
    run = np.logical_and.accumulate(corridor, axis=1).sum(axis=1)
    right_i = (anchor_r + run - 1).astype(np.int32)
    right_i[run == 0] = -1

    raw_left = np.where(left_i >= 0, left_i.astype(np.float64), np.nan)
    raw_right = np.where(right_i >= 0, right_i.astype(np.float64), np.nan)
    subpixel_left = _subpixel_refine(gray, left_i, mid, "left")
    subpixel_right = _subpixel_refine(gray, right_i, mid, "right")
    median_left = median_filter_1d(raw_left, median_window).astype(np.float32)
    median_right = median_filter_1d(raw_right, median_window).astype(np.float32)

    return StripeEdges(
        col_mean=col_mean,
        mid=mid,
        ink_mask=mask,
        raw_left=raw_left,
        raw_right=raw_right,
        subpixel_left=subpixel_left,
        subpixel_right=subpixel_right,
        median_left=median_left,
        median_right=median_right,
        gx_left=gx_left,
        gx_right=gx_right,
    )


def rolling_step_profile(edge: np.ndarray, window: int, guard: int,
                         stride: int) -> Tuple[np.ndarray, np.ndarray]:
    n = len(edge)
    ys = np.arange(window + guard, n - window - guard, stride)
    steps = np.full(len(ys), np.nan)
    for i, y in enumerate(ys):
        above = edge[y - guard - window:y - guard]
        below = edge[y + guard:y + guard + window]
        above = above[~np.isnan(above)]
        below = below[~np.isnan(below)]
        if above.size < window // 4 or below.size < window // 4:
            continue
        steps[i] = np.median(below) - np.median(above)
    return ys, steps
