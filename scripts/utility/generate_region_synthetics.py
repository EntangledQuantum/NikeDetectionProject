"""Build the WhitePaper synthetic region-detection catalog.

Writes recipes, geometric canvases, corner-crop previews and a manifest
under data/20260617_P1_WhitePaper_KCM-updated-folder/synthetic/. Full
mutated TIFFs are never materialised.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from digital_air_cv.geometry.synthetic_overlays import (  # noqa: E402
    GEO_LINE_SPACING,
    SYNTHETIC_DIR,
    WHITEPAPER_DIR,
    GeometricLayout,
    apply_ops_inplace,
    covering_xyxy,
    load_gt,
    open_memmap_scan,
    region_box,
    render_geometric_sheet,
    resolve_source_tiff,
)

STEMS = ("key_full", "cyan_full", "magenta_full")
CROP = 1800
LINE_SPACING = 100


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def build_recipes(stem: str) -> list:
    recipes = []

    def add(rid, family, failure, patches, notes=""):
        recipes.append({
            "id": f"{stem}_{rid}",
            "family": family,
            "source": stem,
            "expected_failure_mode": failure,
            "notes": notes,
            "patches": patches,
        })

    for corner in ("tl", "tr", "bl", "br"):
        add(
            f"missing_{corner}_rect_400",
            "missing_ink",
            "inward_shrink",
            [{"op": "mask_rect", "region": "island", "corner": corner, "w": 400, "h": 400}],
            "paper rectangle over an island corner",
        )
        add(
            f"missing_{corner}_tri_300",
            "missing_ink",
            "inward_shrink",
            [{"op": "mask_triangle", "region": "island", "corner": corner, "w": 300, "h": 300}],
        )
    add(
        "missing_tl_wide_band",
        "missing_ink",
        "inward_shrink",
        [{"op": "mask_rect", "region": "island", "corner": "tl", "w": 800, "h": 200}],
        "wide missing horizontals at the top-left",
    )
    add(
        "missing_bl_tall_vertical",
        "missing_ink",
        "inward_shrink",
        [{"op": "mask_rect", "region": "island", "corner": "bl", "w": 80, "h": 1200}],
        "tall missing outer vertical at the bottom-left",
    )
    for n_lines, which in ((1, "top"), (5, "top"), (20, "top"), (50, "top"), (20, "bottom"), (50, "bottom")):
        add(
            f"missing_{which}_{n_lines}_lines",
            "missing_ink",
            "inward_y_shrink",
            [{"op": "mask_band_h", "region": "island", "which": which, "n_lines": n_lines, "spacing": LINE_SPACING}],
            f"erase {n_lines} dashed line(s) at the {which} of the island",
        )
    for frac, tag in ((0.10, "10pct"), (0.30, "30pct"), (1.0, "all")):
        add(
            f"missing_outer_vertical_top_{tag}",
            "missing_ink",
            "inward_x_shrink",
            [{"op": "mask_band_v", "region": "island", "which": "outer", "y_frac": frac, "w": 40}],
        )
    add(
        "missing_tl_blob",
        "missing_ink",
        "inward_shrink",
        [{"op": "mask_blob", "region": "island", "corner": "tl", "w": 500, "h": 400, "seed": 11}],
    )

    add(
        "feeble_outer_top",
        "feeble_vertical",
        "inward_x_shrink",
        [{"op": "feeble_vertical", "region": "island", "which": "outer", "alpha": 0.25, "y_frac": [0.0, 0.15], "w": 50}],
    )
    add(
        "feeble_inner_top",
        "feeble_vertical",
        "inward_x_shrink",
        [{"op": "feeble_vertical", "region": "island", "which": "inner", "alpha": 0.25, "y_frac": [0.0, 0.15], "w": 50}],
    )
    add(
        "sparse_outer_top",
        "feeble_vertical",
        "inward_x_shrink",
        [{"op": "sparse_vertical", "region": "island", "which": "outer", "drop": 0.7, "y_frac": [0.0, 0.2], "w": 50, "seed": 21}],
    )
    add(
        "thin_outer_top",
        "feeble_vertical",
        "inward_x_shrink",
        [{"op": "thin_vertical", "region": "island", "which": "outer", "erode": 2, "y_frac": [0.0, 0.25], "w": 50}],
    )

    for corner in ("tl", "br"):
        add(
            f"overspray_dots_{corner}",
            "overspray",
            "outward_expand",
            [{"op": "overspray_dots", "region": "island", "corner": corner, "w": 220, "h": 220, "n": 50, "r": 5, "seed": 30}],
        )
    add(
        "overspray_blob_below",
        "overspray",
        "outward_y_expand",
        [{"op": "overspray_blob", "region": "island", "which": "below", "past": 200, "w": 90, "h": 70}],
        "large blob 200px past the true bottom, the case that drags a single y-snap",
    )
    add(
        "overspray_blob_above",
        "overspray",
        "outward_y_expand",
        [{"op": "overspray_blob", "region": "island", "which": "above", "past": 200, "w": 90, "h": 70}],
    )
    add(
        "overspray_gap",
        "overspray",
        "outward_x_expand",
        [{"op": "overspray_edge", "region": "island", "which": "right", "thick": 22, "n": 90, "seed": 41}],
        "spray in the island-stripe gap",
    )
    add(
        "overspray_outer_left",
        "overspray",
        "outward_x_expand",
        [{"op": "overspray_edge", "region": "island", "which": "left", "thick": 22, "n": 90, "seed": 42}],
    )

    add(
        "combo_missing_tl_overspray_br",
        "combined",
        "inward_and_outward",
        [
            {"op": "mask_rect", "region": "island", "corner": "tl", "w": 400, "h": 400},
            {"op": "overspray_dots", "region": "island", "corner": "br", "w": 220, "h": 220, "n": 50, "seed": 51},
        ],
    )
    add(
        "combo_missing_top_lines_overspray_below",
        "combined",
        "inward_and_outward",
        [
            {"op": "mask_band_h", "region": "island", "which": "top", "n_lines": 20, "spacing": LINE_SPACING},
            {"op": "overspray_blob", "region": "island", "which": "below", "past": 200, "w": 90, "h": 70},
        ],
    )
    add(
        "combo_feeble_outer_overspray_left",
        "combined",
        "inward_and_outward",
        [
            {"op": "feeble_vertical", "region": "island", "which": "outer", "alpha": 0.2, "y_frac": [0.0, 0.2], "w": 50},
            {"op": "overspray_edge", "region": "island", "which": "left", "thick": 20, "n": 70, "seed": 61},
        ],
    )
    return recipes


def geometric_cases() -> list:
    cases = []

    def add(rid, layout: GeometricLayout, family, failure, iou_min):
        cases.append({
            "id": rid,
            "family": family,
            "expected_failure_mode": failure,
            "iou_min": iou_min,
            "layout": layout,
        })

    add("geo_clean", GeometricLayout(), "baseline", "none", 0.98)
    for deg in (-1.0, -0.5, -0.2, 0.2, 0.5, 1.0):
        tag = str(deg).replace(".", "p").replace("-", "m")
        add(f"geo_rotate_{tag}", GeometricLayout(rotation_deg=deg), "geometric", "slant_clip", 0.95)
    add("geo_shear", GeometricLayout(shear_x=0.02), "geometric", "slant_clip", 0.95)
    for offset in (30, 80, 150, -80):
        tag = f"p{offset}" if offset > 0 else f"m{abs(offset)}"
        add(f"geo_stitch_{tag}", GeometricLayout(stitch_offset=offset), "geometric", "stitch_clip", 0.95)
    add(
        "geo_missing_tl",
        GeometricLayout(ops=[{"op": "mask_rect", "region": "island", "corner": "tl", "w": 80, "h": 80}]),
        "missing_ink",
        "inward_shrink",
        0.98,
    )
    add(
        "geo_missing_top_20",
        GeometricLayout(ops=[{
            "op": "mask_band_h", "region": "island", "which": "top",
            "n_lines": 20, "spacing": GEO_LINE_SPACING,
        }]),
        "missing_ink",
        "inward_y_shrink",
        0.98,
    )
    add(
        "geo_overspray_below",
        GeometricLayout(ops=[{
            "op": "overspray_blob", "region": "island", "which": "below",
            "past": 40, "w": 28, "h": 22,
        }]),
        "overspray",
        "outward_y_expand",
        0.98,
    )
    add(
        "geo_combo_slant_missing_top",
        GeometricLayout(
            rotation_deg=0.5,
            ops=[{
                "op": "mask_band_h", "region": "island", "which": "top",
                "n_lines": 12, "spacing": GEO_LINE_SPACING,
            }],
        ),
        "combined",
        "slant_and_missing",
        0.95,
    )
    add(
        "geo_feeble_outer",
        GeometricLayout(ops=[{
            "op": "feeble_vertical", "region": "island", "which": "outer",
            "alpha": 0.2, "y_frac": [0.0, 0.2], "w": 12,
        }]),
        "feeble_vertical",
        "inward_x_shrink",
        0.98,
    )
    return cases


def _save_bgr_jpeg(path: Path, image, rgb: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    bgr = image[:, :, ::-1] if rgb else image
    cv2.imwrite(str(path), bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])


def extract_corner_crops(synthetic: Path) -> list:
    records = []
    crop_dir = synthetic / "corner_crops"
    crop_dir.mkdir(parents=True, exist_ok=True)
    half = CROP // 2
    for stem in STEMS:
        gt = load_gt(str(synthetic / "ground_truth" / f"{stem}.json"))
        scan = open_memmap_scan(resolve_source_tiff(gt, WHITEPAPER_DIR))
        rgb = bool(scan.rgb)
        h, w = scan.shape[:2]
        for kind in ("island", "stripe"):
            x0, y0, x1, y1 = region_box(gt, kind)
            corners = {
                "tl": (x0, y0), "tr": (x1, y0),
                "bl": (x0, y1), "br": (x1, y1),
            }
            for name, (cx, cy) in corners.items():
                xa = int(np.clip(cx - half, 0, max(0, w - 1)))
                ya = int(np.clip(cy - half, 0, max(0, h - 1)))
                xb = int(np.clip(xa + CROP, 1, w))
                yb = int(np.clip(ya + CROP, 1, h))
                piece = np.asarray(scan.data[ya:yb, xa:xb])
                fname = f"{stem}_{kind}_{name}.jpg"
                _save_bgr_jpeg(crop_dir / fname, piece, rgb)
                rec = {
                    "id": f"{stem}_{kind}_{name}",
                    "source": stem,
                    "region": kind,
                    "corner": name,
                    "file": f"corner_crops/{fname}",
                    "origin_xy": [xa, ya],
                    "gt_xy_full": [cx, cy],
                    "gt_xy_local": [cx - xa, cy - ya],
                }
                _write_json(crop_dir / f"{stem}_{kind}_{name}.json", rec)
                records.append(rec)
        del scan
    return records


def write_recipe_previews(synthetic: Path, recipes: list) -> None:
    preview_dir = synthetic / "previews"
    preview_dir.mkdir(parents=True, exist_ok=True)
    by_source = {}
    for recipe in recipes:
        by_source.setdefault(recipe["source"], []).append(recipe)
    half = 700
    for stem, items in by_source.items():
        gt = load_gt(str(synthetic / "ground_truth" / f"{stem}.json"))
        scan = open_memmap_scan(resolve_source_tiff(gt, WHITEPAPER_DIR))
        rgb = bool(scan.rgb)
        h, w = scan.shape[:2]
        x0, y0, x1, y1 = region_box(gt, "island")
        # One preview per family using the island top-left (or the op corner).
        seen = set()
        for recipe in items:
            family = recipe["family"]
            if family in seen:
                continue
            seen.add(family)
            cx, cy = x0, y0
            for op in recipe["patches"]:
                if op.get("corner") in ("tr",):
                    cx, cy = x1, y0
                elif op.get("corner") in ("bl",):
                    cx, cy = x0, y1
                elif op.get("corner") in ("br",) or op.get("which") in ("below", "bottom"):
                    cx, cy = x1, y1
            xa = int(np.clip(cx - half, 0, max(0, w - 1)))
            ya = int(np.clip(cy - half, 0, max(0, h - 1)))
            xb = int(np.clip(xa + 2 * half, 1, w))
            yb = int(np.clip(ya + 2 * half, 1, h))
            local_gt = json.loads(json.dumps(gt))
            piece = np.array(scan.data[ya:yb, xa:xb], copy=True)
            # Shift recipe ops into crop coordinates via a translated GT.
            for color in local_gt["colors"]:
                for kind in ("island", "stripe"):
                    color[kind]["x"] = [color[kind]["x"][0] - xa, color[kind]["x"][1] - xa]
                    color[kind]["y"] = [color[kind]["y"][0] - ya, color[kind]["y"][1] - ya]
                    for cname, pt in color[kind]["corners"].items():
                        color[kind]["corners"][cname] = [pt[0] - xa, pt[1] - ya]
            mutated = apply_ops_inplace(piece, recipe["patches"], local_gt, rgb=rgb)
            _save_bgr_jpeg(preview_dir / f"{recipe['id']}.jpg", mutated, rgb)
        del scan


def write_geometric(synthetic: Path, cases: list) -> list:
    geo_dir = synthetic / "geometric"
    geo_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for case in cases:
        image, gt, ref = render_geometric_sheet(case["layout"])
        png = geo_dir / f"{case['id']}.png"
        cv2.imwrite(str(png), image)
        payload = {
            "id": case["id"],
            "family": case["family"],
            "expected_failure_mode": case["expected_failure_mode"],
            "iou_min": case["iou_min"],
            "file": f"geometric/{case['id']}.png",
            "reference": {
                "island_width": ref.island_width,
                "island_stripe_gap": ref.island_stripe_gap,
                "stripe_width": ref.stripe_width,
                "height": ref.height,
            },
            "gt": gt,
            "covering": {
                "island": list(covering_xyxy(gt["colors"][0]["island"]["corners"])),
                "stripe": list(covering_xyxy(gt["colors"][0]["stripe"]["corners"])),
            },
        }
        _write_json(geo_dir / f"{case['id']}.json", payload)
        records.append({k: payload[k] for k in ("id", "family", "expected_failure_mode", "iou_min", "file")})
    return records


def main() -> None:
    synthetic = Path(SYNTHETIC_DIR)
    synthetic.mkdir(parents=True, exist_ok=True)
    recipes_dir = synthetic / "recipes"
    recipes_dir.mkdir(parents=True, exist_ok=True)

    all_recipes = []
    for stem in STEMS:
        recipes = build_recipes(stem)
        all_recipes.extend(recipes)
        for recipe in recipes:
            _write_json(recipes_dir / f"{recipe['id']}.json", recipe)
    _write_json(recipes_dir / "catalog.json", all_recipes)

    geo_records = write_geometric(synthetic, geometric_cases())
    print(f"Wrote {len(geo_records)} geometric canvases")
    crop_records = extract_corner_crops(synthetic)
    print(f"Wrote {len(crop_records)} corner crops")
    write_recipe_previews(synthetic, all_recipes)
    print("Wrote family preview JPEGs")

    manifest = {
        "whitepaper_dir": WHITEPAPER_DIR,
        "stems": list(STEMS),
        "baselines": [
            {
                "id": f"{stem}_unmodified",
                "family": "baseline",
                "source": stem,
                "expected_failure_mode": "none",
                "gt": f"ground_truth/{stem}.json",
                "tiff": f"{stem}.tif" if stem != "key_full" else "key_full.tif",
            }
            for stem in STEMS
        ],
        "recipes": [
            {
                "id": r["id"],
                "family": r["family"],
                "source": r["source"],
                "expected_failure_mode": r["expected_failure_mode"],
                "file": f"recipes/{r['id']}.json",
            }
            for r in all_recipes
        ],
        "geometric": geo_records,
        "corner_crops": crop_records,
        "gt_rule": (
            "Paper masks and overspray do not change ground-truth print corners. "
            "Geometric transforms do; covering AABB is the min/max of the transformed corners."
        ),
    }
    # Fix tiff names
    for item in manifest["baselines"]:
        item["tiff"] = f"{item['source']}.tif"
    _write_json(synthetic / "manifest.json", manifest)
    print(f"Manifest: {len(all_recipes)} recipes, {len(geo_records)} geometric, {len(STEMS)} baselines")


if __name__ == "__main__":
    main()
