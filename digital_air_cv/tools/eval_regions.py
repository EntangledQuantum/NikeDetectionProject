"""Evaluate region detection against the synthetic catalog.

  python -m digital_air_cv.tools.eval_regions
  python -m digital_air_cv.tools.eval_regions --full
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import replace
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import cv2

from digital_air_cv.config.loader import load_config
from digital_air_cv.config.schema import AppConfig, RegionReference
from digital_air_cv.geometry.full_region_detector import detect_full_regions
from digital_air_cv.geometry.region_metrics import (
    COVERAGE_MIN,
    INWARD_FAIL_PX,
    IOU_MISSING_INK,
    IOU_SLANT,
    OUTWARD_CAP_PX,
    boxes_from_debug,
    score_region,
)
from digital_air_cv.geometry.synthetic_overlays import (
    SYNTHETIC_DIR,
    WHITEPAPER_DIR,
    apply_recipe,
    covering_xyxy,
    load_gt,
    open_memmap_scan,
    render_geometric_sheet,
    resolve_source_tiff,
)


def geometric_config(ref: RegionReference, config: Optional[AppConfig] = None) -> AppConfig:
    base = config or load_config()
    geometry = replace(
        base.geometry,
        buffer={"horizontal": 8, "vertical": 8, "stripe_island": 8},
        num_heads=3,
        head_height=max(1, ref.height // 3),
        island_front=True,
    )
    return replace(base, region_reference=ref, geometry=geometry)


def _color_boxes(debug: Dict[str, Any]) -> Dict[str, Tuple[int, int, int, int]]:
    grouped = boxes_from_debug(debug)
    if not grouped:
        return {}
    color = next(iter(grouped.values()))
    return color


def score_debug_against_gt(
    debug: Dict[str, Any],
    gt: Dict[str, Any],
    *,
    iou_min: float,
    outward_cap: float,
    coverage_min: float = COVERAGE_MIN,
) -> Dict[str, Any]:
    pred = _color_boxes(debug)
    color = gt["colors"][0]
    island_gt = covering_xyxy(color["island"]["corners"])
    stripe_gt = covering_xyxy(color["stripe"]["corners"])
    results = {}
    for kind, gt_box in (("island", island_gt), ("stripe", stripe_gt)):
        pred_box = pred.get(kind)
        if pred_box is None:
            results[kind] = {"pass": False, "reasons": ["missing predicted box"]}
            continue
        results[kind] = score_region(
            pred_box, gt_box, iou_min=iou_min,
            inward_fail=INWARD_FAIL_PX, outward_cap=outward_cap,
            coverage_min=coverage_min,
        )
    results["pass"] = all(item.get("pass") for item in results.values() if isinstance(item, dict) and "pass" in item)
    results["warnings"] = debug.get("warnings") or []
    return results


def run_geometric_case(case_json: str, config: Optional[AppConfig] = None) -> Dict[str, Any]:
    with open(case_json, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    image = cv2.imread(os.path.join(os.path.dirname(case_json), os.path.basename(payload["file"]).replace("geometric/", "")), cv2.IMREAD_COLOR)
    if image is None:
        # file field is relative to synthetic/
        synthetic = os.path.dirname(os.path.dirname(case_json))
        image = cv2.imread(os.path.join(synthetic, payload["file"]), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(payload["file"])
    ref = RegionReference(**payload["reference"])
    cfg = geometric_config(ref, config)
    _, debug = detect_full_regions(image, cfg, image_path=payload["id"] + ".png")
    iou_min = float(payload.get("iou_min") or IOU_MISSING_INK)
    cov_min = COVERAGE_MIN if iou_min >= 0.98 else 0.95
    scored = score_debug_against_gt(
        debug, payload["gt"], iou_min=iou_min, outward_cap=24, coverage_min=cov_min,
    )
    scored["id"] = payload["id"]
    scored["family"] = payload.get("family")
    scored["kind"] = "geometric"
    return scored


def run_rendered_layout(layout, iou_min: float = IOU_MISSING_INK, outward_cap: float = 24) -> Dict[str, Any]:
    image, gt, ref = render_geometric_sheet(layout)
    cfg = geometric_config(ref)
    _, debug = detect_full_regions(image, cfg, image_path="geometric.png")
    scored = score_debug_against_gt(debug, gt, iou_min=iou_min, outward_cap=outward_cap)
    return scored


def run_recipe_case(
    recipe: Dict[str, Any],
    synthetic_dir: str,
    config: Optional[AppConfig] = None,
) -> Dict[str, Any]:
    gt = load_gt(os.path.join(synthetic_dir, "ground_truth", f"{recipe['source']}.json"))
    scan = open_memmap_scan(resolve_source_tiff(gt, WHITEPAPER_DIR))
    overlaid = apply_recipe(scan, recipe, gt)
    cfg = config or load_config()
    _, debug = detect_full_regions(overlaid, cfg, image_path=gt["source_tiff"])
    iou_min = IOU_SLANT if "slant" in recipe.get("family", "") else IOU_MISSING_INK
    scored = score_debug_against_gt(
        debug, gt, iou_min=iou_min, outward_cap=OUTWARD_CAP_PX,
    )
    scored["id"] = recipe["id"]
    scored["family"] = recipe.get("family")
    scored["kind"] = "recipe"
    return scored


def run_baseline(stem: str, synthetic_dir: str, config: Optional[AppConfig] = None) -> Dict[str, Any]:
    gt = load_gt(os.path.join(synthetic_dir, "ground_truth", f"{stem}.json"))
    scan = open_memmap_scan(resolve_source_tiff(gt, WHITEPAPER_DIR))
    cfg = config or load_config()
    _, debug = detect_full_regions(scan, cfg, image_path=gt["source_tiff"])
    scored = score_debug_against_gt(
        debug, gt, iou_min=0.97, outward_cap=OUTWARD_CAP_PX,
    )
    scored["id"] = f"{stem}_unmodified"
    scored["family"] = "baseline"
    scored["kind"] = "baseline"
    return scored


def _flatten_row(scored: Dict[str, Any]) -> Dict[str, Any]:
    row = {
        "id": scored.get("id"),
        "kind": scored.get("kind"),
        "family": scored.get("family"),
        "pass": scored.get("pass"),
        "warnings": "; ".join(scored.get("warnings") or []),
    }
    for kind in ("island", "stripe"):
        item = scored.get(kind) or {}
        row[f"{kind}_pass"] = item.get("pass")
        row[f"{kind}_inward"] = (item.get("max_inward") if isinstance(item.get("max_inward"), (int, float)) else "")
        row[f"{kind}_outward"] = item.get("max_outward", "")
        row[f"{kind}_iou"] = item.get("iou", "")
        row[f"{kind}_coverage"] = item.get("coverage", "")
        row[f"{kind}_reasons"] = "; ".join(item.get("reasons") or [])
    return row


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Score region detection on synthetic cases")
    parser.add_argument("--synthetic", default=SYNTHETIC_DIR)
    parser.add_argument("--full", action="store_true", help="Also run recipes on the three TIFFs")
    parser.add_argument("--baselines", action="store_true", help="Run unmodified key/cyan/magenta TIFFs")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    synthetic = args.synthetic
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out or os.path.join(synthetic, "results", stamp)
    os.makedirs(out_dir, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    geo_dir = os.path.join(synthetic, "geometric")
    if os.path.isdir(geo_dir):
        for name in sorted(os.listdir(geo_dir)):
            if not name.endswith(".json"):
                continue
            scored = run_geometric_case(os.path.join(geo_dir, name))
            rows.append(_flatten_row(scored))
            status = "PASS" if scored["pass"] else "FAIL"
            print(f"{status}  {scored['id']}")

    if args.baselines or args.full:
        for stem in ("key_full", "cyan_full", "magenta_full"):
            print(f"baseline {stem}...")
            scored = run_baseline(stem, synthetic)
            rows.append(_flatten_row(scored))
            print(("PASS" if scored["pass"] else "FAIL"), scored["id"])

    if args.full:
        catalog_path = os.path.join(synthetic, "recipes", "catalog.json")
        with open(catalog_path, "r", encoding="utf-8") as handle:
            recipes = json.load(handle)
        for recipe in recipes:
            print(f"recipe {recipe['id']}...")
            scored = run_recipe_case(recipe, synthetic)
            rows.append(_flatten_row(scored))
            print(("PASS" if scored["pass"] else "FAIL"), recipe["id"])

    csv_path = os.path.join(out_dir, "summary.csv")
    if rows:
        with open(csv_path, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as handle:
            json.dump(rows, handle, indent=2)
        failed = [r for r in rows if not r.get("pass")]
        print(f"Wrote {csv_path}  ({len(rows) - len(failed)}/{len(rows)} passed)")
    return 0 if all(r.get("pass") for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
