"""
Stripe debris detection for colored stripe images.

This detector targets dark debris spots on solid-color stripe images such as
`blueStripe.tiff` and `pinkStripe.tiff`. It is designed to be color-agnostic:
the stripe may be blue, pink, or another saturated color, but debris is
expected to be significantly darker and closer to black than the surrounding
stripe.

Algorithm outline:
  1. Detect the colored stripe bounds from per-column chroma.
  2. Restrict analysis to the stripe interior so paper/background is ignored.
  3. Build three complementary feature maps inside the stripe:
     - absolute darkness relative to the stripe's own luminance statistics
     - chroma drop relative to the stripe's own colorfulness statistics
     - multi-scale black-hat contrast to highlight dark spots of varying size
  4. Combine those maps into a debris score and threshold it using robust
     image-derived statistics rather than fixed color thresholds.
  5. Grow strong detections through connected weak pixels so irregular shapes
     are preserved.
  6. Morphologically clean the mask and keep connected components consistent
     with true dark debris.
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


class DebrisStripeDetector:
    """Detect dark debris spots inside colored stripe images."""

    def __init__(
        self,
        sensitivity: str = "medium",
        min_area: Optional[int] = None,
        inner_pad_frac: float = 0.035,
        inner_pad_min: int = 8,
        max_area_frac: float = 0.20,
        debug: bool = False,
    ):
        self.sensitivity = sensitivity
        self.min_area_override = min_area
        self.inner_pad_frac = inner_pad_frac
        self.inner_pad_min = inner_pad_min
        self.max_area_frac = max_area_frac
        self.debug = debug

        if sensitivity == "high":
            self._score_k = 4.5
            self._strong_floor = 0.34
            self._weak_floor = 0.22
            self._min_area_floor = 4
            self._min_area_frac = 2e-5
        elif sensitivity == "low":
            self._score_k = 7.0
            self._strong_floor = 0.48
            self._weak_floor = 0.32
            self._min_area_floor = 20
            self._min_area_frac = 1.2e-4
        else:
            self._score_k = 5.5
            self._strong_floor = 0.40
            self._weak_floor = 0.26
            self._min_area_floor = 8
            self._min_area_frac = 5e-5

        self._debug_artifacts: Dict[str, np.ndarray] = {}

    def detect(
        self,
        image: np.ndarray,
        image_path: Optional[str] = None,
    ) -> Tuple[np.ndarray, List[Dict]]:
        """Run debris detection and return `(visualization, defects)`."""
        if image is None or image.size == 0:
            return np.zeros((1, 1, 3), dtype=np.uint8), []

        bgr = image if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        h, w = bgr.shape[:2]

        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        stripe_bounds = self._find_stripe_bounds(lab)
        if stripe_bounds is None:
            if self.debug:
                print("  [debris_stripe] No stripe bounds found; skipping.")
            return bgr.copy(), []

        x_left, x_right = stripe_bounds
        stripe_w = max(1, x_right - x_left)

        pad = max(self.inner_pad_min, int(round(self.inner_pad_frac * stripe_w)))
        ix1 = min(x_left + pad, w - 1)
        ix2 = max(x_right - pad, 0)
        if ix2 - ix1 < 10:
            if self.debug:
                print("  [debris_stripe] Stripe interior too narrow after padding; skipping.")
            return bgr.copy(), []

        inner_mask = np.zeros((h, w), dtype=np.uint8)
        inner_mask[:, ix1:ix2] = 255

        score, feature_maps = self._compute_debris_score(bgr, lab, inner_mask, stripe_w)
        strong_mask, weak_mask, thresholds = self._build_candidate_masks(score, inner_mask)
        promoted = self._promote_connected_weak_regions(strong_mask, weak_mask)
        cleaned = self._morph_cleanup(promoted, stripe_w)

        defects = self._extract_defects(cleaned, score, feature_maps, inner_mask, stripe_w)
        vis = self._draw_visualization(bgr, defects, stripe_w)

        if self.debug:
            self._store_debug_artifacts(
                score=score,
                strong_mask=strong_mask,
                weak_mask=weak_mask,
                final_mask=cleaned,
                stripe_bounds=(x_left, x_right, ix1, ix2),
                thresholds=thresholds,
            )
            print(
                f"  [debris_stripe] stripe x=[{x_left},{x_right}] inner=[{ix1},{ix2}] "
                f"strong={thresholds['strong']:.3f} weak={thresholds['weak']:.3f} "
                f"defects={len(defects)}"
            )

        return vis, defects

    def _find_stripe_bounds(self, lab: np.ndarray) -> Optional[Tuple[int, int]]:
        """Locate the colored stripe from column chroma, with a dark fallback."""
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

        x_left = int(idx.min())
        x_right = int(idx.max())
        if x_right - x_left < 30:
            return None

        return x_left, x_right

    def _compute_debris_score(
        self,
        bgr: np.ndarray,
        lab: np.ndarray,
        inner_mask: np.ndarray,
        stripe_w: int,
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Combine absolute darkness, chroma loss, and local contrast."""
        l_chan, a_chan, b_chan = cv2.split(lab)
        l_float = l_chan.astype(np.float32)
        a_float = a_chan.astype(np.float32) - 128.0
        b_float = b_chan.astype(np.float32) - 128.0
        chroma = np.sqrt(a_float * a_float + b_float * b_float)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        hsv_s = hsv[:, :, 1].astype(np.float32)
        hsv_v = hsv[:, :, 2].astype(np.float32)

        inside_l = l_float[inner_mask > 0]
        inside_chroma = chroma[inner_mask > 0]

        l_med = float(np.median(inside_l))
        l_q05 = float(np.quantile(inside_l, 0.05))
        l_scale = max(l_med - l_q05, 18.0)
        absolute_dark = np.clip((l_med - l_float) / l_scale, 0.0, 1.0)

        chroma_med = float(np.median(inside_chroma))
        chroma_q05 = float(np.quantile(inside_chroma, 0.05))
        chroma_scale = max(chroma_med - chroma_q05, 6.0)
        chroma_drop = np.clip((chroma_med - chroma) / chroma_scale, 0.0, 1.0)

        inside_s = hsv_s[inner_mask > 0]
        s_med = float(np.median(inside_s))
        s_q05 = float(np.quantile(inside_s, 0.05))
        saturation_scale = max(s_med - s_q05, 20.0)
        saturation_drop = np.clip((s_med - hsv_s) / saturation_scale, 0.0, 1.0)

        local_dark = self._multiscale_blackhat_score(l_chan, hsv_v.astype(np.uint8), stripe_w)
        score = (
            0.40 * local_dark
            + 0.35 * absolute_dark
            + 0.20 * saturation_drop
            + 0.05 * chroma_drop
        ).astype(np.float32)
        score[inner_mask == 0] = 0.0

        feature_maps = {
            "absolute_dark": absolute_dark,
            "chroma_drop": chroma_drop,
            "saturation_drop": saturation_drop,
            "local_dark": local_dark,
        }
        return score, feature_maps

    def _multiscale_blackhat_score(
        self,
        l_chan: np.ndarray,
        value_chan: np.ndarray,
        stripe_w: int,
    ) -> np.ndarray:
        """Highlight dark spots over multiple spatial scales."""
        kernels = []
        for divisor in (180, 90, 45):
            size = max(9, stripe_w // divisor)
            if size % 2 == 0:
                size += 1
            kernels.append(size)

        responses = []
        for kernel_size in sorted(set(kernels)):
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (kernel_size, kernel_size),
            )
            l_bh = cv2.morphologyEx(l_chan, cv2.MORPH_BLACKHAT, kernel).astype(np.float32)
            v_bh = cv2.morphologyEx(value_chan, cv2.MORPH_BLACKHAT, kernel).astype(np.float32)
            responses.append(np.maximum(l_bh, v_bh))

        local_raw = np.maximum.reduce(responses)
        positive = local_raw[local_raw > 0]
        if positive.size == 0:
            return np.zeros_like(local_raw, dtype=np.float32)

        scale = max(float(np.quantile(positive, 0.9995)), 20.0)
        return np.clip(local_raw / scale, 0.0, 1.0).astype(np.float32)

    def _build_candidate_masks(
        self,
        score: np.ndarray,
        inner_mask: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
        """Create strong/weak masks from robust score thresholds."""
        inside_scores = score[inner_mask > 0]
        if inside_scores.size == 0:
            empty = np.zeros_like(inner_mask)
            return empty, empty, {"strong": 1.0, "weak": 1.0}

        median = float(np.median(inside_scores))
        mad = float(np.median(np.abs(inside_scores - median))) + 1e-6
        sigma = 1.4826 * mad

        strong_threshold = max(median + self._score_k * sigma, self._strong_floor)
        strong_threshold = min(strong_threshold, 0.90)
        weak_threshold = max(strong_threshold * 0.65, self._weak_floor)

        strong_mask = ((score >= strong_threshold) & (inner_mask > 0)).astype(np.uint8) * 255
        weak_mask = ((score >= weak_threshold) & (inner_mask > 0)).astype(np.uint8) * 255

        return strong_mask, weak_mask, {
            "strong": float(strong_threshold),
            "weak": float(weak_threshold),
        }

    def _promote_connected_weak_regions(
        self,
        strong_mask: np.ndarray,
        weak_mask: np.ndarray,
    ) -> np.ndarray:
        """Keep weak components that contain at least one strong pixel."""
        if not np.any(strong_mask):
            return strong_mask.copy()

        kept = np.zeros_like(weak_mask)
        n_labels, labels, _, _ = cv2.connectedComponentsWithStats(weak_mask, connectivity=8)
        for idx in range(1, n_labels):
            component = labels == idx
            if np.any(strong_mask[component] > 0):
                kept[component] = 255
        return kept

    def _morph_cleanup(self, mask: np.ndarray, stripe_w: int) -> np.ndarray:
        """Clean noise while preserving compact or irregular debris shapes."""
        open_size = max(3, stripe_w // 350)
        if open_size % 2 == 0:
            open_size += 1
        close_size = max(open_size, stripe_w // 140)
        if close_size % 2 == 0:
            close_size += 1

        k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_size, open_size))
        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size))

        opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open, iterations=1)
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, k_close, iterations=1)
        return closed

    def _extract_defects(
        self,
        mask: np.ndarray,
        score: np.ndarray,
        feature_maps: Dict[str, np.ndarray],
        inner_mask: np.ndarray,
        stripe_w: int,
    ) -> List[Dict]:
        """Convert the cleaned debris mask into structured defect records."""
        h, w = mask.shape
        if self.min_area_override is not None:
            min_area = int(self.min_area_override)
        else:
            min_area = max(self._min_area_floor, int(self._min_area_frac * h * stripe_w))
        max_area = int(self.max_area_frac * h * w)

        n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        defects: List[Dict] = []

        absolute_dark = feature_maps["absolute_dark"]
        local_dark = feature_maps["local_dark"]
        chroma_drop = feature_maps["chroma_drop"]
        saturation_drop = feature_maps["saturation_drop"]

        for idx in range(1, n_labels):
            x, y, cw, ch, area = stats[idx]
            if area < min_area or area > max_area:
                continue

            region = labels == idx
            if not np.any(inner_mask[region] > 0):
                continue

            mean_score = float(score[region].mean())
            max_score = float(score[region].max())
            mean_abs_dark = float(absolute_dark[region].mean())
            mean_local_dark = float(local_dark[region].mean())
            mean_chroma_drop = float(chroma_drop[region].mean())
            max_abs_dark = float(absolute_dark[region].max())
            max_saturation_drop = float(saturation_drop[region].max())

            if max_score < self._strong_floor:
                continue
            if mean_abs_dark < 0.20 and mean_local_dark < 0.20:
                continue
            if max_saturation_drop < 0.25 and max_abs_dark < 0.80:
                continue

            defects.append(
                {
                    "type": "debris_stripe",
                    "bbox": [int(x), int(y), int(cw), int(ch)],
                    "area": int(area),
                    "centroid": [float(centroids[idx][0]), float(centroids[idx][1])],
                    "mean_score": mean_score,
                    "max_score": max_score,
                    "mean_absolute_darkness": mean_abs_dark,
                    "mean_local_darkness": mean_local_dark,
                    "max_absolute_darkness": max_abs_dark,
                    "max_saturation_drop": max_saturation_drop,
                    "mean_chroma_drop": mean_chroma_drop,
                }
            )

        defects.sort(key=lambda defect: (-defect["max_score"], -defect["area"]))
        return defects

    def _draw_visualization(self, bgr: np.ndarray, defects: List[Dict], stripe_w: int) -> np.ndarray:
        """Draw black bounding boxes around debris detections."""
        vis = bgr.copy()
        h, w = vis.shape[:2]
        pad = max(2, stripe_w // 250)
        thickness = max(2, stripe_w // 400)

        for defect in defects:
            x, y, bw, bh = defect["bbox"]
            pt1 = (max(0, x - pad), max(0, y - pad))
            pt2 = (min(w - 1, x + bw + pad), min(h - 1, y + bh + pad))
            cv2.rectangle(vis, pt1, pt2, (0, 0, 0), thickness)

        return vis

    def _store_debug_artifacts(
        self,
        score: np.ndarray,
        strong_mask: np.ndarray,
        weak_mask: np.ndarray,
        final_mask: np.ndarray,
        stripe_bounds: Tuple[int, int, int, int],
        thresholds: Dict[str, float],
    ) -> None:
        """Store debug artifacts as uint8 images for later saving."""
        x_left, x_right, ix1, ix2 = stripe_bounds
        bounds_vis = np.zeros_like(score, dtype=np.uint8)
        bounds_vis[:, x_left:x_right] = 80
        bounds_vis[:, ix1:ix2] = 170

        score_vis = np.clip(score * 255.0, 0, 255).astype(np.uint8)
        self._debug_artifacts = {
            "score": score_vis,
            "strong_mask": strong_mask,
            "weak_mask": weak_mask,
            "final_mask": final_mask,
            "stripe_bounds": bounds_vis,
        }
        self._debug_thresholds = thresholds

    def save_debug_images(self, output_dir: str, base_name: str) -> None:
        """Persist debug artifacts when `debug=True`."""
        if not self.debug or not self._debug_artifacts:
            return

        os.makedirs(output_dir, exist_ok=True)
        for tag, image in self._debug_artifacts.items():
            path = os.path.join(output_dir, f"{base_name}_debris_stripe_{tag}.png")
            cv2.imwrite(path, image)


def _main() -> None:
    parser = argparse.ArgumentParser(description="Run stripe debris detection on one image.")
    parser.add_argument("image", help="Path to a stripe image")
    parser.add_argument("--output", "-o", help="Optional visualization output path")
    parser.add_argument(
        "--sensitivity",
        choices=["low", "medium", "high"],
        default="medium",
        help="Detection sensitivity",
    )
    parser.add_argument("--debug", action="store_true", help="Save debug artifacts next to output")
    args = parser.parse_args()

    image = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if image is None:
        raise SystemExit(f"Could not read image: {args.image}")

    detector = DebrisStripeDetector(sensitivity=args.sensitivity, debug=args.debug)
    visualization, defects = detector.detect(image, args.image)

    output_path = args.output
    if not output_path:
        stem, _ = os.path.splitext(args.image)
        output_path = f"{stem}_debris_stripe.jpg"

    cv2.imwrite(output_path, visualization, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"Saved visualization: {output_path}")
    print(f"Detected {len(defects)} debris regions")
    for idx, defect in enumerate(defects, start=1):
        print(
            f"  {idx}. bbox={defect['bbox']} area={defect['area']} "
            f"score={defect['max_score']:.3f}"
        )

    if args.debug:
        debug_dir = os.path.dirname(output_path) or "."
        base_name = os.path.splitext(os.path.basename(output_path))[0]
        detector.save_debug_images(debug_dir, base_name)


if __name__ == "__main__":
    _main()
