"""Fast region-detection tests on geometric canvases and overlay ops."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from nike_detection.config.loader import load_config
from nike_detection.geometry.full_region_detector import detect_full_regions
from nike_detection.geometry.region_metrics import (
    INWARD_FAIL_PX,
    boxes_from_debug,
    max_inward,
    score_region,
)
from nike_detection.geometry.synthetic_overlays import (
    GEO_LINE_SPACING,
    SYNTHETIC_DIR,
    GeometricLayout,
    apply_ops_inplace,
    covering_xyxy,
    render_geometric_sheet,
)
from nike_detection.tools.eval_regions import geometric_config, score_debug_against_gt


def _detect(layout: GeometricLayout):
    image, gt, ref = render_geometric_sheet(layout)
    cfg = geometric_config(ref)
    specs, debug = detect_full_regions(image, cfg, image_path="geometric.png")
    return image, gt, ref, specs, debug


def test_clean_geometric_sheet_hits_print_edges():
    _, gt, _, _, debug = _detect(GeometricLayout())
    scored = score_debug_against_gt(debug, gt, iou_min=0.98, outward_cap=24)
    assert scored["pass"], scored


def test_missing_top_lines_do_not_shrink_y():
    layout = GeometricLayout(ops=[{
        "op": "mask_band_h",
        "region": "island",
        "which": "top",
        "n_lines": 20,
        "spacing": GEO_LINE_SPACING,
    }])
    _, gt, _, _, debug = _detect(layout)
    pred = boxes_from_debug(debug)
    color = next(iter(pred.values()))
    gt_box = covering_xyxy(gt["colors"][0]["island"]["corners"])
    scored = score_region(color["island"], gt_box, iou_min=0.97, outward_cap=24)
    assert scored["max_inward"] <= INWARD_FAIL_PX, scored


def test_missing_corner_rect_does_not_clip_print():
    layout = GeometricLayout(ops=[{
        "op": "mask_rect", "region": "island", "corner": "tl", "w": 80, "h": 80,
    }])
    _, gt, _, _, debug = _detect(layout)
    scored = score_debug_against_gt(debug, gt, iou_min=0.97, outward_cap=24)
    assert scored["island"]["max_inward"] <= INWARD_FAIL_PX, scored["island"]


def test_overspray_blob_does_not_expand_y_past_cap():
    layout = GeometricLayout(ops=[{
        "op": "overspray_blob",
        "region": "island",
        "which": "below",
        "past": 40,
        "w": 28,
        "h": 22,
    }])
    _, gt, _, _, debug = _detect(layout)
    scored = score_debug_against_gt(debug, gt, iou_min=0.97, outward_cap=24)
    assert scored["island"]["max_outward"] <= 24, scored["island"]


def test_slant_covering_aabb_contains_print():
    _, gt, _, _, debug = _detect(GeometricLayout(rotation_deg=0.8))
    scored = score_debug_against_gt(debug, gt, iou_min=0.95, outward_cap=40)
    assert scored["island"]["coverage"] >= 0.995, scored["island"]
    assert scored["island"]["max_inward"] <= INWARD_FAIL_PX, scored["island"]


def test_stitch_dogleg_is_covered():
    _, gt, _, _, debug = _detect(GeometricLayout(stitch_offset=80))
    scored = score_debug_against_gt(debug, gt, iou_min=0.95, outward_cap=50)
    assert scored["island"]["max_inward"] <= INWARD_FAIL_PX, scored["island"]
    assert scored["island"]["coverage"] >= 0.995, scored["island"]


def test_feeble_outer_vertical_keeps_left_edge():
    layout = GeometricLayout(ops=[{
        "op": "feeble_vertical",
        "region": "island",
        "which": "outer",
        "alpha": 0.2,
        "y_frac": [0.0, 0.2],
        "w": 12,
    }])
    _, gt, _, _, debug = _detect(layout)
    scored = score_debug_against_gt(debug, gt, iou_min=0.97, outward_cap=24)
    assert scored["island"]["max_inward"] <= INWARD_FAIL_PX, scored["island"]


def test_debug_writes_four_corners_and_stations():
    _, _, _, specs, debug = _detect(GeometricLayout())
    color = debug["colors"][0]
    assert "tl" in color["corners"]["island"]
    assert "br" in color["corners"]["island"]
    assert color["edge_stations"]["island_left"]
    assert specs[0].corners is not None
    assert "tr" in specs[0].corners


def test_overlay_mask_paints_paper():
    image, gt, _ = render_geometric_sheet(GeometricLayout())
    x0, y0 = gt["colors"][0]["island"]["corners"]["tl"]
    mutated = apply_ops_inplace(
        image,
        [{"op": "mask_rect", "region": "island", "corner": "tl", "w": 40, "h": 40}],
        gt,
        rgb=False,
    )
    assert np.all(mutated[int(y0) + 5, int(x0) + 5] == (255, 255, 255))
    assert not np.array_equal(mutated, image)


def test_metrics_inward_and_outward():
    gt = (10, 10, 100, 100)
    clipped = (20, 10, 100, 100)
    wide = (0, 10, 100, 100)
    assert max_inward({"left": 10, "top": 0, "right": 0, "bottom": 0}) == 10
    from nike_detection.geometry.region_metrics import inward_offsets
    assert inward_offsets(clipped, gt)["left"] == 10
    assert inward_offsets(wide, gt)["left"] == -10


@pytest.mark.slow
def test_unmodified_key_full_does_not_regress():
    gt_path = Path(SYNTHETIC_DIR) / "ground_truth" / "key_full.json"
    tiff = Path(SYNTHETIC_DIR).parent / "key_full.tif"
    if not gt_path.exists() or not tiff.exists():
        pytest.skip("WhitePaper TIFF / GT not present")
    from nike_detection.geometry.synthetic_overlays import load_gt, open_memmap_scan
    gt = load_gt(str(gt_path))
    scan = open_memmap_scan(str(tiff))
    cfg = load_config()
    _, debug = detect_full_regions(scan, cfg, image_path=str(tiff))
    scored = score_debug_against_gt(debug, gt, iou_min=0.97, outward_cap=80)
    assert scored["pass"], scored
