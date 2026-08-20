"""
Island line-defect tests on synthetic islands with known ground truth.

The properties asserted here are the ones that actually broke before: that a
threshold means the same thing on every ink, that healthy stipple is not
reported as haze, and that a head join is separated from print that merely
looks textured.
"""

from __future__ import annotations

import numpy as np
import pytest

from digital_air_cv.detectors.island_new.line_defect import NewPatternLineDefectDetector
from digital_air_cv.geometry.ink_density import SCALE, measure_ink_field
from digital_air_cv.testing.scoring import score_sample
from digital_air_cv.testing.synthetic_island import (
    INK_ABSORBANCE,
    IslandBuilder,
    healthy,
)

COLORS = tuple(INK_ABSORBANCE)
MARGIN = 60.0


@pytest.fixture(scope="module")
def detector():
    return NewPatternLineDefectDetector(sensitivity="medium")


def run(detector, sample):
    return detector.detect(sample.image)[1]


def of_type(defects, name):
    return [d for d in defects if d.get("type") == name]


# ----------------------------------------------------------------------
# Ink normalization
# ----------------------------------------------------------------------

@pytest.mark.parametrize("color", COLORS)
def test_ink_axis_matches_the_absorbing_channel(color):
    """Each ink's axis must point at the channel it actually absorbs."""
    field = measure_ink_field(healthy(color=color, n_lines=10, width=800).image)
    expected = np.asarray(INK_ABSORBANCE[color], dtype=np.float32)
    expected /= np.linalg.norm(expected)
    assert float(np.dot(field.axis, expected)) > 0.97


@pytest.mark.parametrize("color", COLORS)
def test_healthy_core_normalizes_to_alpha_one(color):
    """A saturated line reads ~1.0 whatever colour it is, and never clips."""
    field = measure_ink_field(healthy(color=color, n_lines=10, width=800).image)
    alpha = field.alpha.astype(np.float32) / SCALE
    assert 0.85 <= float(np.percentile(alpha, 99.5)) <= 1.20
    assert float((field.alpha == 255).mean()) < 1e-4


def test_yellow_is_not_penalized_by_low_luminance():
    """Yellow has ~21 levels of grayscale contrast but separates fine in alpha.

    This is the case that made grayscale thresholding unusable: in luminance
    yellow is nearly invisible, so any shared gray threshold either misses it
    entirely or floods every other ink with false positives.
    """
    import cv2

    sample = healthy(color="yellow", n_lines=10, width=800)
    gray = cv2.cvtColor(sample.image, cv2.COLOR_BGR2GRAY)
    gray_contrast = float(np.percentile(gray, 95) - np.percentile(gray, 1))

    field = measure_ink_field(sample.image)
    alpha = field.alpha.astype(np.float32) / SCALE
    alpha_contrast = float(np.percentile(alpha, 99) - np.percentile(alpha, 50))

    assert gray_contrast < 60.0
    assert alpha_contrast > 0.7


# ----------------------------------------------------------------------
# False positives
# ----------------------------------------------------------------------

@pytest.mark.parametrize("color", COLORS)
def test_healthy_print_reports_nothing(detector, color):
    """Stipple must not be mistaken for haze, on any ink."""
    sample = healthy(color=color, n_lines=24, width=1400, seed=3)
    defects = run(detector, sample)
    assert of_type(defects, "misaligned_line") == []
    assert of_type(defects, "missing_line") == []
    assert of_type(defects, "stitch_error") == []


def test_thin_but_crisp_line_is_not_haze(detector):
    """Less ink is not a defect; ink that lost its core is.

    Distinguishing these two is the whole reason haze keys off core density
    rather than ink thickness.
    """
    sample = IslandBuilder(color="cyan", n_lines=20, width=1400,
                           thickness=2.8, seed=5).build()
    assert of_type(run(detector, sample), "misaligned_line") == []


# ----------------------------------------------------------------------
# Detection
# ----------------------------------------------------------------------

@pytest.mark.parametrize("color", COLORS)
def test_missing_nozzle_run_is_found(detector, color):
    builder = IslandBuilder(color=color, n_lines=20, width=1400, seed=7)
    builder.add_missing(line=9, x0=500, x1=800)
    sample = builder.build()
    score = score_sample(sample, run(detector, sample), MARGIN)["missing"]
    assert score.recall == 1.0
    assert score.false_positives == 0


def test_gap_below_one_jet_width_is_ignored(detector):
    """A gap shorter than a single jet's footprint is print, not a defect."""
    builder = IslandBuilder(color="key", n_lines=20, width=1400, seed=8)
    builder.add_missing(line=9, x0=600, x1=630)
    sample = builder.build()
    assert of_type(run(detector, sample), "missing_line") == []


def test_whole_missing_line_is_reported(detector):
    builder = IslandBuilder(color="magenta", n_lines=20, width=1400, seed=9)
    builder.add_whole_line_missing(line=11)
    defects = run(detector, builder.build())
    assert any(d.get("whole_line") for d in of_type(defects, "missing_line"))


@pytest.mark.parametrize("color", COLORS)
def test_haze_is_found_on_every_ink(detector, color):
    builder = IslandBuilder(color=color, n_lines=16, width=1400, seed=11)
    builder.add_haze(line=8, x0=400, x1=1000, strength=0.6)
    sample = builder.build()
    score = score_sample(sample, run(detector, sample), MARGIN)["haze"]
    assert score.recall == 1.0


def test_haze_severity_tracks_smear_strength(detector):
    """Severity must be ordered, so a heavy smear outranks a faint one."""
    severities = []
    for strength in (0.3, 0.6, 0.9):
        builder = IslandBuilder(color="cyan", n_lines=16, width=1400, seed=13)
        builder.add_haze(line=8, x0=400, x1=1000, strength=strength)
        hazy = of_type(run(detector, builder.build()), "misaligned_line")
        assert hazy
        severities.append(max(d["severity"] for d in hazy))
    assert severities[0] < severities[1] < severities[2]


# ----------------------------------------------------------------------
# Stitch
# ----------------------------------------------------------------------

def test_stitch_is_found_and_localized(detector):
    builder = IslandBuilder(color="cyan", n_lines=24, width=1600, seed=17)
    builder.add_stitch(line=12, amplitude=4.0)
    sample = builder.build()
    stitches = of_type(run(detector, sample), "stitch_error")
    assert len(stitches) == 1
    expected = MARGIN + 12 * sample.spacing + sample.slope * 800
    assert abs(stitches[0]["y"] - expected) < sample.spacing


def test_stitch_severity_orders_by_amplitude(detector):
    """A mild join must score below a badly calibrated one."""
    severities = []
    for amplitude in (2.5, 5.0):
        builder = IslandBuilder(color="yellow", n_lines=24, width=1600, seed=19)
        builder.add_stitch(line=12, amplitude=amplitude)
        stitches = of_type(run(detector, builder.build()), "stitch_error")
        assert stitches
        severities.append(stitches[0]["severity"])
    assert severities[0] < severities[1]


def test_stitch_count_cannot_exceed_head_joins(detector):
    """Three heads can only join twice, however wavy the band looks."""
    builder = IslandBuilder(color="magenta", n_lines=30, width=1600, seed=23)
    for line in range(4, 28, 3):
        builder.add_stitch(line=line, amplitude=4.0)
    stitches = of_type(run(detector, builder.build()), "stitch_error")
    assert len(stitches) <= detector.num_heads - 1


def test_haze_does_not_masquerade_as_a_stitch(detector):
    """Smeared ink drags the centroid about without the printed core moving.

    Measuring wander across smeared columns is what previously turned a hazy
    band into a phantom head join.
    """
    builder = IslandBuilder(color="cyan", n_lines=24, width=1600, seed=29)
    for line in (6, 10, 14, 18):
        builder.add_haze(line=line, x0=300, x1=1300, strength=0.8)
    assert of_type(run(detector, builder.build()), "stitch_error") == []


# ----------------------------------------------------------------------
# Cross-ink consistency
# ----------------------------------------------------------------------

def test_identical_defect_scores_identically_across_inks(detector):
    """The same physical defect must produce the same verdict on every ink.

    Colour independence is the point of the whole ink-density layer, so it is
    asserted directly rather than inferred from per-ink recall.
    """
    results = {}
    for color in COLORS:
        builder = IslandBuilder(color=color, n_lines=20, width=1500, seed=31)
        builder.add_missing(line=5, x0=400, x1=700)
        builder.add_haze(line=12, x0=500, x1=1000, strength=0.65)
        defects = run(detector, builder.build())
        results[color] = (len(of_type(defects, "missing_line")),
                          len(of_type(defects, "misaligned_line")))
    assert len(set(results.values())) == 1, results
