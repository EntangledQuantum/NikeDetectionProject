"""
Match detector output against synthetic ground truth.

Detections cannot be matched by line index: the detector derives its own line
numbering from the print and may insert rows for fully missing lines, so index
``7`` on either side need not be the same line. Matching therefore happens in
image space -- a detection belongs to a truth defect when it sits on the same
trajectory (within half a line spacing) and overlaps it along x.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

from nike_detection.testing.synthetic_island import Sample

# Fraction of the shorter span that must overlap for a match. Deliberately
# forgiving on extent: the useful question is "was this defect found and
# localized", not whether the edges agree to the pixel.
MIN_OVERLAP = 0.30


@dataclass
class Score:
    """Precision / recall for one defect type."""

    kind: str
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    matched_severity: List[float] = None

    def __post_init__(self):
        if self.matched_severity is None:
            self.matched_severity = []

    @property
    def precision(self) -> float:
        denominator = self.true_positives + self.false_positives
        return self.true_positives / denominator if denominator else 1.0

    @property
    def recall(self) -> float:
        denominator = self.true_positives + self.false_negatives
        return self.true_positives / denominator if denominator else 1.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    def __str__(self) -> str:
        return (f"{self.kind:8s} P={self.precision:5.2f} R={self.recall:5.2f} "
                f"F1={self.f1:5.2f}  (tp={self.true_positives} "
                f"fp={self.false_positives} fn={self.false_negatives})")


def _truth_y(sample: Sample, line: int, x: float, margin: float = 60.0) -> float:
    return margin + line * sample.spacing + sample.slope * x


def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    span = min(a1, b1) - max(a0, b0)
    if span <= 0:
        return 0.0
    shorter = min(a1 - a0, b1 - b0)
    return span / shorter if shorter > 0 else 0.0


def _of_type(defects: Iterable[dict], type_name: str) -> List[dict]:
    return [d for d in defects if d.get("type") == type_name]


def score_spans(
    sample: Sample,
    detections: Sequence[dict],
    truths: Sequence[dict],
    kind: str,
    margin: float = 60.0,
) -> Score:
    """Score span-like defects (missing runs, haze runs) by position overlap."""
    score = Score(kind=kind)
    used = set()

    for truth in truths:
        best, best_overlap = None, 0.0
        for index, detection in enumerate(detections):
            if index in used:
                continue
            mid = 0.5 * (detection["start_x"] + detection["end_x"])
            expected = _truth_y(sample, truth["line"], mid, margin)
            if abs(detection["y"] - expected) > 0.5 * sample.spacing:
                continue
            overlap = _overlap(truth["x0"], truth["x1"],
                               detection["start_x"], detection["end_x"])
            if overlap > best_overlap:
                best, best_overlap = index, overlap
        if best is not None and best_overlap >= MIN_OVERLAP:
            used.add(best)
            score.true_positives += 1
            score.matched_severity.append(
                float(detections[best].get("severity", 1.0)))
        else:
            score.false_negatives += 1

    score.false_positives = len(detections) - len(used)
    return score


def score_stitch(
    sample: Sample,
    detections: Sequence[dict],
    margin: float = 60.0,
) -> Score:
    """Score stitch zones by which line they landed on."""
    score = Score(kind="stitch")
    used = set()

    for truth in sample.stitch:
        expected = _truth_y(sample, truth["line"], 0.5 * sample.image.shape[1], margin)
        best = None
        for index, detection in enumerate(detections):
            if index in used:
                continue
            if abs(detection["y"] - expected) <= 1.5 * sample.spacing:
                best = index
                break
        if best is not None:
            used.add(best)
            score.true_positives += 1
            score.matched_severity.append(float(detections[best].get("severity", 0.0)))
        else:
            score.false_negatives += 1

    score.false_positives = len(detections) - len(used)
    return score


def score_sample(
    sample: Sample, defects: Sequence[dict], margin: float = 60.0
) -> Dict[str, Score]:
    """Score every defect type of one synthetic sample."""
    return {
        "missing": score_spans(
            sample, _of_type(defects, "missing_line"), sample.missing,
            "missing", margin),
        "haze": score_spans(
            sample, _of_type(defects, "misaligned_line"), sample.haze,
            "haze", margin),
        "stitch": score_stitch(sample, _of_type(defects, "stitch_error"), margin),
    }


def merge(scores: Iterable[Dict[str, Score]]) -> Dict[str, Score]:
    """Accumulate per-sample scores into one totals table."""
    totals: Dict[str, Score] = {}
    for entry in scores:
        for kind, score in entry.items():
            target = totals.setdefault(kind, Score(kind=kind))
            target.true_positives += score.true_positives
            target.false_positives += score.false_positives
            target.false_negatives += score.false_negatives
            target.matched_severity.extend(score.matched_severity)
    return totals
