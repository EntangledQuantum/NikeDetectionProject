"""
Void Detection Algorithm (Stripe Images)

Detects small "void" regions inside solid-color vertical stripe images.
A void is a compact region inside the stripe whose color has drifted toward
the paper/white background (i.e. missing or very faint ink coverage), often
with a few residual ink spots sprinkled inside. Voids are small compared to
the stripe; huge regions are usually other defect classes (e.g. stripe
fade-outs at the image boundary) and are filtered out.

Algorithm outline:
  1. Convert image to LAB color space.
  2. Find the stripe's left/right boundaries from per-column chroma (for
     colored stripes) or per-column lightness (for black stripes).
  3. Compute a robust stripe reference color (median LAB of the interior)
     and a paper reference color (median LAB of columns outside the stripe).
  4. Per-pixel "void score" is the signed projection of
     (pixel_LAB - stripe_ref_LAB) onto the unit vector
     (paper_ref_LAB - stripe_ref_LAB), normalized to [0, 1]. Stripe pixels
     score ~0, paper-colored pixels score ~1.
  5. Restrict scoring to the stripe's interior (shrunk inward by a margin
     to ignore ragged stripe edges).
  6. Hysteresis-threshold the (lightly blurred) score using robust
     statistics: strong seeds at median + k * MAD * 1.4826, grown into a
     weaker level (a fraction of k), keeping only weak components that
     contain a strong seed. A small floor handles nearly-uniform stripes.
  7. Morphological close to merge void pixels through residual ink spots,
     then open to suppress single-pixel noise. Kernel size scales with the
     stripe width so the detector adapts to DPI/crops.
  8. Keep connected components whose area, extent, and aspect ratio are
     consistent with a "void"; reject huge regions (likely stripe fades).
  9. Draw a black bounding box around each surviving region.
"""

from __future__ import annotations

import os
from typing import List, Dict, Optional, Tuple

import cv2
import numpy as np


class VoidDetector:
    """Detect small voids inside solid-color vertical stripe images.

    The detector is intended for pre-extracted stripe sub-images (e.g.
    ``blueStripe.tiff``, ``blackStripe.tiff``, ``pinkStripe.tiff``). It is
    robust to varying brightness/contrast because the decision threshold
    is derived from the stripe's own color statistics (median + MAD) and
    because the classification direction is the axis connecting the
    stripe color to the paper color measured from the same image.
    """

    def __init__(
        self,
        sensitivity: str = "medium",
        min_area: Optional[int] = None,
        max_area_frac: float = 0.10,
        max_dim_ratio_to_stripe_w: float = 1.0,
        inner_pad_frac: float = 0.05,
        inner_pad_min: int = 20,
        debug: bool = False,
    ):
        """Configure the detector.

        Args:
            sensitivity: 'low' | 'medium' | 'high'. Controls the score
                threshold (how many robust std-devs above the stripe median
                qualifies as a void) and the minimum area.
            min_area: Override for minimum void area in pixels. If None, a
                value is derived from image size by sensitivity.
            max_area_frac: Upper area cap expressed as a fraction of the
                image size. Components larger than this are discarded as
                global/stripe-fade artifacts rather than voids.
            max_dim_ratio_to_stripe_w: Reject components whose bounding
                box height or width exceeds this multiple of the stripe
                width. A value near 1.0 keeps the detector focused on
                compact voids rather than broad low-ink regions.
            inner_pad_frac: Shrink the stripe detection region by this
                fraction of the stripe width on each side to avoid ragged
                edges. A minimum of `inner_pad_min` pixels is always used.
            inner_pad_min: Minimum inner padding in pixels.
            debug: When True, stores intermediate artifacts accessible via
                `save_debug_images` and prints detailed progress.
        """
        self.sensitivity = sensitivity
        self.min_area_override = min_area
        self.max_area_frac = max_area_frac
        self.max_dim_ratio_to_stripe_w = max_dim_ratio_to_stripe_w
        self.inner_pad_frac = inner_pad_frac
        self.inner_pad_min = inner_pad_min
        self.debug = debug

        if sensitivity == "high":
            self._mad_k = 5.0
            self._min_area_floor = 80
            self._min_area_frac = 1e-4
        elif sensitivity == "low":
            self._mad_k = 10.0
            self._min_area_floor = 300
            self._min_area_frac = 4e-4
        else:
            self._mad_k = 7.0
            self._min_area_floor = 150
            self._min_area_frac = 2e-4

        self._score_floor = 0.12
        # Hysteresis: weak threshold is this fraction of the strong one
        # (measured above the stripe median). Weak-mask components are kept
        # only when they contain at least one strong pixel.
        self._weak_fraction = 0.55
        self._debug_artifacts: Dict[str, np.ndarray] = {}

    def detect(
        self,
        image: np.ndarray,
        image_path: Optional[str] = None,
    ) -> Tuple[np.ndarray, List[Dict]]:
        """Run void detection and produce a visualization.

        Args:
            image: Input stripe image (BGR or grayscale uint8).
            image_path: Optional path of the source image. Currently unused
                by this detector but accepted for interface parity with the
                rest of the pipeline.

        Returns:
            Tuple ``(visualization_bgr, defects)`` where ``defects`` is a
            list of dicts with keys: ``type``, ``bbox`` [x, y, w, h],
            ``area``, ``centroid`` [cx, cy], ``mean_voidness``, ``threshold``.
        """
        if image is None or image.size == 0:
            return np.zeros((1, 1, 3), dtype=np.uint8), []

        bgr = image if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        h, w = bgr.shape[:2]

        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        bounds = self._find_stripe_bounds(lab)
        if bounds is None:
            if self.debug:
                print("  [void] No stripe detected; skipping.")
            return bgr.copy(), []

        x_left, x_right = bounds
        stripe_w = max(1, x_right - x_left)

        pad = max(self.inner_pad_min, int(self.inner_pad_frac * stripe_w))
        ix1 = min(x_left + pad, w - 1)
        ix2 = max(x_right - pad, 0)
        if ix2 - ix1 < 10:
            if self.debug:
                print("  [void] Stripe interior too narrow after padding; skipping.")
            return bgr.copy(), []

        voidness = self._compute_voidness(lab, (x_left, x_right), (ix1, ix2), w)
        if voidness is None:
            return bgr.copy(), []

        stripe_inner_mask = np.zeros((h, w), dtype=np.uint8)
        stripe_inner_mask[:, ix1:ix2] = 255

        binmask, threshold = self._threshold_voidness(voidness, stripe_inner_mask)
        cleaned = self._morph_cleanup(binmask, stripe_w)

        if self.debug:
            self._debug_artifacts["voidness"] = (voidness * 255).astype(np.uint8)
            self._debug_artifacts["binmask_raw"] = binmask
            self._debug_artifacts["binmask_cleaned"] = cleaned

        defects = self._filter_components(
            cleaned, voidness, threshold, stripe_w, (h, w)
        )
        vis = self._draw_visualization(bgr, defects, stripe_w)

        if self.debug:
            print(
                f"  [void] stripe x=[{x_left},{x_right}] inner=[{ix1},{ix2}] "
                f"threshold={threshold:.3f} defects={len(defects)}"
            )

        return vis, defects

    def _find_stripe_bounds(self, lab: np.ndarray) -> Optional[Tuple[int, int]]:
        """Locate the stripe's left/right column indices.

        Uses per-column mean chroma (distance of (a,b) from neutral). For
        near-monochrome/black stripes where chroma is too low, falls back
        to selecting columns that are substantially darker than average.
        """
        L, A, B = cv2.split(lab)
        a_f = A.astype(np.float32) - 128.0
        b_f = B.astype(np.float32) - 128.0
        col_chroma = np.sqrt(a_f * a_f + b_f * b_f).mean(axis=0)
        col_L = L.mean(axis=0)

        chroma_thresh = max(col_chroma.max() * 0.4, 15.0)
        stripe_cols = col_chroma > chroma_thresh
        total_cols = len(col_chroma)

        if stripe_cols.sum() < total_cols * 0.1:
            mean_L = col_L.mean()
            L_thresh = mean_L - (mean_L - col_L.min()) * 0.5
            stripe_cols = col_L < L_thresh

        idx = np.where(stripe_cols)[0]
        if len(idx) == 0:
            return None

        x_left, x_right = int(idx.min()), int(idx.max())
        if x_right - x_left < 30:
            return None
        return x_left, x_right

    def _compute_voidness(
        self,
        lab: np.ndarray,
        outer_bounds: Tuple[int, int],
        inner_bounds: Tuple[int, int],
        image_width: int,
    ) -> Optional[np.ndarray]:
        """Compute the per-pixel void score in [0, 1]."""
        x_left, x_right = outer_bounds
        ix1, ix2 = inner_bounds
        L, A, B = cv2.split(lab)

        inner = lab[:, ix1:ix2]
        ref_L = float(np.median(inner[..., 0]))
        ref_a = float(np.median(inner[..., 1]))
        ref_b = float(np.median(inner[..., 2]))

        non_stripe = np.ones(image_width, dtype=bool)
        non_stripe[x_left : x_right + 1] = False
        if non_stripe.sum() > 50:
            paper_L = float(np.median(L[:, non_stripe]))
            paper_a = float(np.median(A[:, non_stripe]))
            paper_b = float(np.median(B[:, non_stripe]))
        else:
            paper_L, paper_a, paper_b = 240.0, 128.0, 128.0

        vx, vy, vz = paper_L - ref_L, paper_a - ref_a, paper_b - ref_b
        vnorm2 = vx * vx + vy * vy + vz * vz
        if vnorm2 < 1.0:
            if self.debug:
                print("  [void] Stripe ~ paper; cannot define void direction.")
            return None
        vnorm = vnorm2 ** 0.5

        Lf = L.astype(np.float32) - ref_L
        Af = A.astype(np.float32) - ref_a
        Bf = B.astype(np.float32) - ref_b
        proj = (Lf * vx + Af * vy + Bf * vz) / vnorm
        voidness = np.clip(proj / vnorm, 0.0, 1.0).astype(np.float32)

        if self.debug:
            print(
                f"  [void] stripe LAB=({ref_L:.1f},{ref_a:.1f},{ref_b:.1f}) "
                f"paper LAB=({paper_L:.1f},{paper_a:.1f},{paper_b:.1f}) "
                f"|v|={vnorm:.2f}"
            )
        return voidness

    def _threshold_voidness(
        self, voidness: np.ndarray, stripe_inner_mask: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """Hysteresis threshold (median + k*MAD), restricted to inner stripe.

        The voidness map is lightly blurred first so per-pixel sensor noise
        does not inflate the MAD estimate (real voids are blobs and survive
        the blur). Strong seeds are found at ``median + k*MAD``; the mask is
        then grown into the weaker ``median + weak_fraction*k*MAD`` level,
        keeping only weak components that contain a strong seed. This
        recovers the faint outskirts of genuine voids without admitting
        isolated faint noise.
        """
        smoothed = cv2.blur(voidness, (5, 5))
        inside = smoothed[stripe_inner_mask > 0]
        if inside.size == 0:
            return np.zeros_like(stripe_inner_mask), self._score_floor

        med = float(np.median(inside))
        mad = float(np.median(np.abs(inside - med))) + 1e-6
        sigma_robust = 1.4826 * mad
        strong_t = max(med + self._mad_k * sigma_robust, self._score_floor)
        weak_t = max(med + self._weak_fraction * self._mad_k * sigma_robust,
                     self._score_floor * 0.7)

        inner = stripe_inner_mask > 0
        strong = (smoothed > strong_t) & inner
        weak = ((smoothed > weak_t) & inner).astype(np.uint8)

        # Hysteresis: keep weak components seeded by at least 1 strong pixel
        n_lab, labels = cv2.connectedComponents(weak, connectivity=8)
        if n_lab > 1:
            seeded = np.unique(labels[strong])
            seeded = seeded[seeded > 0]
            keep = np.zeros(n_lab, dtype=bool)
            keep[seeded] = True
            binmask = (keep[labels]).astype(np.uint8) * 255
        else:
            binmask = np.zeros_like(weak)

        if self.debug:
            print(
                f"  [void] med={med:.3f} sigma={sigma_robust:.4f} "
                f"strong_t={strong_t:.3f} weak_t={weak_t:.3f}"
            )
        return binmask, weak_t

    def _morph_cleanup(self, binmask: np.ndarray, stripe_w: int) -> np.ndarray:
        """Close to fill ink-spot gaps inside voids; open to drop tiny noise."""
        base = max(5, stripe_w // 200)
        close_sz = base * 2 + 1
        open_sz = max(3, base)

        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_sz, close_sz))
        k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_sz, open_sz))

        closed = cv2.morphologyEx(binmask, cv2.MORPH_CLOSE, k_close, iterations=2)
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, k_open, iterations=1)
        return opened

    def _filter_components(
        self,
        cleaned: np.ndarray,
        voidness: np.ndarray,
        threshold: float,
        stripe_w: int,
        image_shape: Tuple[int, int],
    ) -> List[Dict]:
        """Keep only compact regions with consistent void scores."""
        h, w = image_shape
        nlab, labels, stats, centroids = cv2.connectedComponentsWithStats(
            cleaned, connectivity=8
        )

        if self.min_area_override is not None:
            min_area = int(self.min_area_override)
        else:
            # Scale with stripe width squared (a DPI proxy) so the minimum
            # void size does not depend on how tall the crop is. The old
            # `frac * h * stripe_w` formula ballooned to thousands of pixels
            # for full-height stripes and silently dropped small voids.
            size_based = int(self._min_area_frac * stripe_w * stripe_w)
            min_area = max(self._min_area_floor, size_based)

        max_area = int(self.max_area_frac * h * w)
        max_dim = int(self.max_dim_ratio_to_stripe_w * stripe_w)

        defects: List[Dict] = []
        for i in range(1, nlab):
            x, y, cw, ch, area = stats[i]
            if area < min_area:
                continue
            if area > max_area:
                continue
            if cw > max_dim or ch > max_dim:
                continue

            ar = max(cw, ch) / max(1, min(cw, ch))
            if ar > 8.0 and area < min_area * 3:
                continue

            extent = area / max(1, cw * ch)
            if extent < 0.15:
                continue

            region = labels == i
            mean_v = float(voidness[region].mean())
            if mean_v < threshold * 1.05:
                continue

            defects.append({
                "type": "void",
                "bbox": [int(x), int(y), int(cw), int(ch)],
                "area": int(area),
                "centroid": [float(centroids[i][0]), float(centroids[i][1])],
                "mean_voidness": mean_v,
                "threshold": float(threshold),
            })

        defects.sort(key=lambda d: -d["area"])
        return defects

    def _draw_visualization(
        self, bgr: np.ndarray, defects: List[Dict], stripe_w: int
    ) -> np.ndarray:
        """Draw a black bounding box around each detected void."""
        vis = bgr.copy()
        h, w = vis.shape[:2]
        base = max(5, stripe_w // 200)
        pad = base
        thickness = max(2, base // 2)
        for d in defects:
            x, y, cw, ch = d["bbox"]
            pt1 = (max(0, x - pad), max(0, y - pad))
            pt2 = (min(w - 1, x + cw + pad), min(h - 1, y + ch + pad))
            cv2.rectangle(vis, pt1, pt2, (0, 0, 0), thickness)
        return vis

    def save_debug_images(self, output_dir: str, base_name: str) -> None:
        """Persist debug artifacts (voidness map, masks) when ``debug=True``."""
        if not self.debug or not self._debug_artifacts:
            return
        os.makedirs(output_dir, exist_ok=True)
        for tag, img in self._debug_artifacts.items():
            path = os.path.join(output_dir, f"{base_name}_void_{tag}.png")
            cv2.imwrite(path, img)
