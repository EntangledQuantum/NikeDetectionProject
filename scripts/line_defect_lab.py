"""
Development harness for island line-defect detection.

Runs the detector over a set of crops, writes annotated overlays and prints a
compact per-image summary (including the measured ink calibration, which is
usually the first thing to check when a colour behaves oddly).

    python scripts/line_defect_lab.py data/test_data/line_defect
    python scripts/line_defect_lab.py <folder-or-image> -s high -o /tmp/out
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nike_detection.detectors.island_new.line_defect import (  # noqa: E402
    NewPatternLineDefectDetector,
)

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def collect(target: Path):
    if target.is_file():
        return [target]
    return sorted(
        path for path in target.rglob("*")
        if path.suffix.lower() in IMAGE_SUFFIXES and "_overlay" not in path.stem
    )


def summarize(defects):
    header = next((d for d in defects if d.get("type") == "bands_detected"), {})
    buckets = {"missing_line": [], "misaligned_line": [], "stitch_error": [],
               "high_density_region": []}
    for defect in defects:
        if defect.get("type") in buckets:
            buckets[defect["type"]].append(defect)
    return header, buckets


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", help="Image file or folder of crops")
    parser.add_argument("-s", "--sensitivity", default="medium",
                        choices=("low", "medium", "high"))
    parser.add_argument("-o", "--output", help="Overlay folder (default: <input>/overlays)")
    parser.add_argument("--clear", action="store_true", help="Clear scan material")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args(argv)

    target = Path(args.input).resolve()
    paths = collect(target)
    if not paths:
        print(f"No images under {target}")
        return 1

    out_dir = Path(args.output) if args.output else (
        target if target.is_dir() else target.parent) / "overlays"
    out_dir.mkdir(parents=True, exist_ok=True)

    for path in paths:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            print(f"!! could not read {path}")
            continue

        detector = NewPatternLineDefectDetector(
            sensitivity=args.sensitivity, debug=args.debug, clear=args.clear)
        start = time.perf_counter()
        vis, defects = detector.detect(image, str(path))
        elapsed = (time.perf_counter() - start) * 1000.0

        header, buckets = summarize(defects)
        ink = header.get("ink", {})
        bands = header.get("bands", [])
        spacing = np.median([b["spacing"] for b in bands]) if bands else 0.0

        print(f"\n=== {path.name}  ({image.shape[1]}x{image.shape[0]})  {elapsed:.0f} ms")
        print(f"    ink axis BGR {ink.get('axis_bgr')}  paper {ink.get('paper_bgr')}  "
              f"contrast {ink.get('contrast')}  noise {ink.get('noise')}")
        print(f"    bands {header.get('band_count')}  lines {header.get('total_lines')}  "
              f"spacing {spacing:.1f}px")
        print(f"    missing {len(buckets['missing_line']):4d} gaps / "
              f"{header.get('total_missing_pixels', 0)} px")

        hazy = buckets["misaligned_line"]
        if hazy:
            sev = np.array([d["severity"] for d in hazy])
            grades = {g: sum(1 for d in hazy if d["grade"] == g)
                      for g in ("mild", "moderate", "severe")}
            print(f"    hazy    {len(hazy):4d}  severity mean {sev.mean():.2f} "
                  f"max {sev.max():.2f}  {grades}")
        else:
            print("    hazy       0")

        for defect in buckets["stitch_error"]:
            print(f"    stitch  line {defect['line_index']:3d}  "
                  f"waviness {defect['waviness']:.4f} "
                  f"({defect['amplitude_px']:.2f} px)  "
                  f"severity {defect['severity']:.2f} [{defect['grade']}]")
        if not buckets["stitch_error"]:
            print("    stitch     0")

        dest = out_dir / f"{path.stem}_overlay.png"
        cv2.imwrite(str(dest), vis)
        print(f"    -> {dest}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
