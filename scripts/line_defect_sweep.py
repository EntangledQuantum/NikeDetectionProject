"""
Synthetic sweep for island line-defect detection.

Scores the detector against generated ground truth across all four inks. The
point of the sweep is not a single pass/fail number but the two things that
broke the previous implementation:

* whether a threshold means the same thing on Key as on Yellow, and
* how much healthy stipple leaks through as false haze.

    python scripts/line_defect_sweep.py
    python scripts/line_defect_sweep.py -s high --colors cyan yellow
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from digital_air_cv.detectors.island_new.line_defect import (  # noqa: E402
    NewPatternLineDefectDetector,
)
from digital_air_cv.testing.scoring import merge, score_sample  # noqa: E402
from digital_air_cv.testing.synthetic_island import IslandBuilder  # noqa: E402

COLORS = ("key", "cyan", "magenta", "yellow")
MARGIN = 60.0


def run(detector, sample):
    return detector.detect(sample.image)[1]


def build_mixed(color: str, seed: int) -> "IslandBuilder":
    """One island carrying every defect type at several strengths."""
    builder = IslandBuilder(color=color, n_lines=34, width=1800, seed=seed)
    # Missing runs comfortably above the ~90 px single-jet gap.
    for i, (line, width) in enumerate(((3, 120), (8, 200), (14, 420), (25, 150))):
        start = 200 + 260 * i
        builder.add_missing(line=line, x0=start, x1=start + width)
    builder.add_whole_line_missing(line=19)
    # Haze from barely visible to heavy smear.
    for i, (line, strength) in enumerate(((5, 0.35), (11, 0.55), (17, 0.75), (28, 0.9))):
        start = 300 + 300 * i
        builder.add_haze(line=line, x0=start, x1=start + 320, strength=strength)
    # Two joins, as a 3-head press produces.
    builder.add_stitch(line=11, amplitude=3.2)
    builder.add_stitch(line=22, amplitude=3.2)
    return builder


def sweep_false_positives(detector, colors, seeds):
    print("\n-- healthy print (every report here is a false positive) --")
    print(f"{'ink':9s} {'missing':>9s} {'hazy':>7s} {'stitch':>7s}   per 1000 line-px")
    worst = 0.0
    for color in colors:
        totals = np.zeros(3)
        columns = 0
        for seed in seeds:
            sample = IslandBuilder(color=color, n_lines=34, width=1800,
                                   seed=seed).build()
            defects = run(detector, sample)
            counts = [
                sum(1 for d in defects if d.get("type") == "missing_line"),
                sum(1 for d in defects if d.get("type") == "misaligned_line"),
                sum(1 for d in defects if d.get("type") == "stitch_error"),
            ]
            totals += counts
            columns += 34 * 1800
        rate = 1000.0 * totals[1] / columns
        worst = max(worst, rate)
        print(f"{color:9s} {totals[0]:9.0f} {totals[1]:7.0f} {totals[2]:7.0f}   "
              f"haze {rate:.4f}")
    return worst


def sweep_haze_curve(detector, colors, seeds):
    print("\n-- haze detection rate vs smear strength --")
    strengths = (0.2, 0.3, 0.4, 0.55, 0.7, 0.85)
    header = "  ".join(f"{s:>4.2f}" for s in strengths)
    print(f"{'ink':9s} {header}")
    for color in colors:
        cells = []
        for strength in strengths:
            found = trials = 0
            for seed in seeds:
                builder = IslandBuilder(color=color, n_lines=12, width=1400,
                                        seed=seed)
                builder.add_haze(line=6, x0=400, x1=1000, strength=strength)
                sample = builder.build()
                scores = score_sample(sample, run(detector, sample), MARGIN)
                found += scores["haze"].true_positives
                trials += 1
            cells.append(found / max(1, trials))
        print(f"{color:9s} " + "  ".join(f"{c:4.2f}" for c in cells))


def sweep_stitch_curve(detector, colors, seeds):
    print("\n-- stitch: detection rate and reported severity vs amplitude --")
    amplitudes = (0.0, 1.0, 2.0, 3.0, 5.0)
    print(f"{'ink':9s} " + "  ".join(f"{a:>10.1f}px" for a in amplitudes))
    for color in colors:
        cells = []
        for amplitude in amplitudes:
            found = trials = 0
            severities = []
            for seed in seeds:
                builder = IslandBuilder(color=color, n_lines=20, width=1600,
                                        seed=seed)
                if amplitude > 0:
                    builder.add_stitch(line=10, amplitude=amplitude)
                sample = builder.build()
                defects = run(detector, sample)
                stitches = [d for d in defects if d.get("type") == "stitch_error"]
                if amplitude > 0:
                    scores = score_sample(sample, defects, MARGIN)
                    found += scores["stitch"].true_positives
                    severities.extend(
                        d["severity"] for d in stitches
                        if abs(d["y"] - (MARGIN + 10 * sample.spacing
                                         + sample.slope * 800)) < 1.5 * sample.spacing)
                else:
                    found += len(stitches)  # any report is spurious
                trials += 1
            mean_sev = np.mean(severities) if severities else 0.0
            cells.append(f"{found / max(1, trials):4.2f}/{mean_sev:4.2f}")
        print(f"{color:9s} " + "  ".join(f"{c:>12s}" for c in cells))
    print("           (rate/severity; amplitude 0.0 column counts false alarms)")


def sweep_mixed(detector, colors, seeds):
    print("\n-- mixed island: precision / recall per ink --")
    all_scores = []
    for color in colors:
        per_color = []
        elapsed = 0.0
        for seed in seeds:
            sample = build_mixed(color, seed).build()
            start = time.perf_counter()
            defects = run(detector, sample)
            elapsed += time.perf_counter() - start
            per_color.append(score_sample(sample, defects, MARGIN))
        totals = merge(per_color)
        all_scores.extend(per_color)
        summary = "  ".join(str(totals[k]) for k in ("missing", "haze", "stitch"))
        print(f"{color:9s} {summary}   {1000 * elapsed / len(seeds):.0f} ms/img")
    print("\noverall:")
    for _kind, score in merge(all_scores).items():
        print(f"    {score}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-s", "--sensitivity", default="medium",
                        choices=("low", "medium", "high"))
    parser.add_argument("--colors", nargs="+", default=list(COLORS), choices=COLORS)
    parser.add_argument("--seeds", type=int, default=3)
    args = parser.parse_args(argv)

    detector = NewPatternLineDefectDetector(sensitivity=args.sensitivity)
    seeds = list(range(args.seeds))

    print(f"sensitivity={args.sensitivity}  seeds={args.seeds}")
    sweep_false_positives(detector, args.colors, seeds)
    sweep_haze_curve(detector, args.colors, seeds)
    sweep_stitch_curve(detector, args.colors, seeds)
    sweep_mixed(detector, args.colors, seeds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
