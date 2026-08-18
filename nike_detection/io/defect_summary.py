"""Readable per-region defect summary (txt + csv). Detection math is unchanged."""

from __future__ import annotations

import csv
import logging
import os
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from nike_detection.config.schema import RunSettings
from nike_detection.pipeline.types import ImageResult

logger = logging.getLogger(__name__)

_SUMMARY_KINDS = {"bands_detected", "edge_roughness_summary"}


def _items(result: ImageResult, detector: str) -> List[Dict[str, Any]]:
    det = result.detections.get(detector)
    if det is None:
        return []
    return [item for item in (det.defects or []) if isinstance(item, dict)]


def _of_type(items: Iterable[Dict[str, Any]], kind: str) -> List[Dict[str, Any]]:
    return [item for item in items if str(item.get("type")) == kind]


def _num(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if abs(number - round(number)) < 1e-9:
        return str(int(round(number)))
    return f"{number:.{digits}f}"


def _csv_row(
    rows: List[Dict[str, str]],
    region: str,
    region_type: str,
    detector: str,
    kind: str,
    metric: str,
    value: Any,
    unit: str = "",
    notes: str = "",
) -> None:
    rows.append({
        "region": region,
        "region_type": region_type,
        "detector": detector,
        "kind": kind,
        "metric": metric,
        "value": _fmt(value),
        "unit": unit,
        "notes": notes,
    })


def _island_line_block(
    result: ImageResult,
    lines: List[str],
    rows: List[Dict[str, str]],
) -> None:
    items = _items(result, "line_defect")
    if not items and "line_defect" not in result.detections:
        return
    summary = (_of_type(items, "bands_detected") or [{}])[0]
    missing = _of_type(items, "missing_line")
    stitch = _of_type(items, "stitch_error")
    hazy = _of_type(items, "misaligned_line")
    dense = _of_type(items, "high_density_region")
    nozzles = summary.get("estimated_missing_nozzles", sum(
        int(_num(item.get("missing_pixels"))) for item in missing
    ))
    gap_count = summary.get("total_missing_defects", len(missing))
    stitch_count = summary.get("total_stitch_defects", len(stitch))
    hazy_count = summary.get("total_misaligned_defects", len(hazy))
    lines.append("  Line defect (missing nozzles + calibration / stitch)")
    lines.append(f"    Missing-nozzle gaps:           {gap_count}")
    lines.append(f"    Estimated missing nozzles:     {nozzles}  (ink-free columns)")
    lines.append(f"    Stitch / calibration errors:   {stitch_count}")
    lines.append(f"    Hazy / misaligned segments:    {hazy_count}")
    if dense:
        lines.append(f"    High-density missing clusters: {len(dense)}")
    region = result.image_name
    kind = result.image_type.value
    _csv_row(rows, region, kind, "line_defect", "missing_line",
             "missing_nozzle_gaps", gap_count, "count")
    _csv_row(rows, region, kind, "line_defect", "missing_line",
             "estimated_missing_nozzles", nozzles, "px columns",
             "sum of ink-free columns in scored gaps")
    _csv_row(rows, region, kind, "line_defect", "stitch_error",
             "stitch_calibration_errors", stitch_count, "count")
    _csv_row(rows, region, kind, "line_defect", "misaligned_line",
             "hazy_misaligned_segments", hazy_count, "count")
    if dense:
        _csv_row(rows, region, kind, "line_defect", "high_density_region",
                 "high_density_clusters", len(dense), "count")
    for i, item in enumerate(stitch, start=1):
        loc = item.get("location") or (item.get("start_x"), item.get("y"))
        _csv_row(
            rows, region, kind, "line_defect", "stitch_error",
            f"stitch_{i}_y", item.get("y"), "px",
            f"location={loc}",
        )


def _stripe_misalignment_block(
    result: ImageResult,
    lines: List[str],
    rows: List[Dict[str, str]],
) -> None:
    items = _items(result, "stripe_misalignment")
    if not items and "stripe_misalignment" not in result.detections:
        return
    stitches = _of_type(items, "stripe_misalignment")
    rolls = _of_type(items, "roll_error")
    steps = [abs(_num(item.get("step_px", item.get("x_delta")))) for item in stitches]
    drifts = [abs(_num(item.get("drift_px"))) for item in rolls]
    lines.append("  Stripe misalignment (calibration / stitch)")
    if steps:
        lines.append(
            f"    Stitch steps:                   {len(stitches)}  "
            f"(max |step| {_fmt(max(steps))} px, mean {_fmt(sum(steps) / len(steps))} px)"
        )
    else:
        lines.append("    Stitch steps:                   0")
    if drifts:
        lines.append(
            f"    Roll / drift segments:          {len(rolls)}  "
            f"(max |drift| {_fmt(max(drifts))} px)"
        )
    else:
        lines.append("    Roll / drift segments:          0")
    region = result.image_name
    kind = result.image_type.value
    _csv_row(rows, region, kind, "stripe_misalignment", "stripe_misalignment",
             "stitch_count", len(stitches), "count")
    if steps:
        _csv_row(rows, region, kind, "stripe_misalignment", "stripe_misalignment",
                 "max_abs_step_px", max(steps), "px")
        _csv_row(rows, region, kind, "stripe_misalignment", "stripe_misalignment",
                 "mean_abs_step_px", sum(steps) / len(steps), "px")
    _csv_row(rows, region, kind, "stripe_misalignment", "roll_error",
             "roll_count", len(rolls), "count")
    if drifts:
        _csv_row(rows, region, kind, "stripe_misalignment", "roll_error",
                 "max_abs_drift_px", max(drifts), "px")
    for i, item in enumerate(stitches, start=1):
        _csv_row(
            rows, region, kind, "stripe_misalignment", "stripe_misalignment",
            f"stitch_{i}_step_px", item.get("step_px"), "px",
            f"y={item.get('y')}",
        )


def _edge_metrics(side: Dict[str, Any]) -> str:
    if not side:
        return "n/a"
    return (
        f"sigma={_fmt(side.get('sigma_px'))} px, "
        f"MAD={_fmt(side.get('mad_px'))} px, "
        f"P95={_fmt(side.get('p95_px'))} px, "
        f"RMS={_fmt(side.get('rms_px'))} px, "
        f"peak-to-peak={_fmt(side.get('peak_to_peak_px'))} px"
    )


def _roughness_block(
    result: ImageResult,
    lines: List[str],
    rows: List[Dict[str, str]],
) -> None:
    items = _items(result, "edge_roughness")
    if not items and "edge_roughness" not in result.detections:
        return
    summary = (_of_type(items, "edge_roughness_summary") or [{}])[0]
    spans = _of_type(items, "edge_roughness")
    left = dict(summary.get("left") or {})
    right = dict(summary.get("right") or {})
    left_flag = bool(summary.get("left_flagged"))
    right_flag = bool(summary.get("right_flagged"))
    lines.append("  Edge roughness")
    lines.append(
        f"    Left edge:                      {_edge_metrics(left)}"
        f"{'  FLAGGED' if left_flag else ''}"
    )
    lines.append(
        f"    Right edge:                     {_edge_metrics(right)}"
        f"{'  FLAGGED' if right_flag else ''}"
    )
    lines.append(f"    Flagged spans:                  {len(spans)}")
    region = result.image_name
    kind = result.image_type.value
    for edge_name, metrics, flagged in (
        ("left", left, left_flag),
        ("right", right, right_flag),
    ):
        _csv_row(rows, region, kind, "edge_roughness", "edge_roughness_summary",
                 f"{edge_name}_sigma_px", metrics.get("sigma_px"), "px")
        _csv_row(rows, region, kind, "edge_roughness", "edge_roughness_summary",
                 f"{edge_name}_mad_px", metrics.get("mad_px"), "px")
        _csv_row(rows, region, kind, "edge_roughness", "edge_roughness_summary",
                 f"{edge_name}_p95_px", metrics.get("p95_px"), "px")
        _csv_row(rows, region, kind, "edge_roughness", "edge_roughness_summary",
                 f"{edge_name}_rms_px", metrics.get("rms_px"), "px")
        _csv_row(rows, region, kind, "edge_roughness", "edge_roughness_summary",
                 f"{edge_name}_peak_to_peak_px", metrics.get("peak_to_peak_px"), "px")
        _csv_row(rows, region, kind, "edge_roughness", "edge_roughness_summary",
                 f"{edge_name}_flagged", int(flagged), "",
                 "1 = over MAD/P95 thresholds")
    _csv_row(rows, region, kind, "edge_roughness", "edge_roughness",
             "flagged_span_count", len(spans), "count")
    if spans:
        _csv_row(
            rows, region, kind, "edge_roughness", "edge_roughness",
            "max_span_p95_px",
            max(_num(item.get("p95_px")) for item in spans),
            "px",
        )


def _void_block(
    result: ImageResult,
    lines: List[str],
    rows: List[Dict[str, str]],
) -> None:
    items = _items(result, "void")
    if not items and "void" not in result.detections:
        return
    voids = _of_type(items, "void")
    areas = [_num(item.get("area")) for item in voids]
    total = sum(areas)
    lines.append("  Voids")
    lines.append(f"    Count:                          {len(voids)}")
    if areas:
        lines.append(f"    Total area:                     {_fmt(total)} px^2")
        lines.append(f"    Largest void:                   {_fmt(max(areas))} px^2")
        lines.append(f"    Mean area:                      {_fmt(total / len(areas))} px^2")
    region = result.image_name
    kind = result.image_type.value
    _csv_row(rows, region, kind, "void", "void", "void_count", len(voids), "count")
    _csv_row(rows, region, kind, "void", "void", "total_area_px2", total, "px^2")
    if areas:
        _csv_row(rows, region, kind, "void", "void", "max_area_px2", max(areas), "px^2")
        _csv_row(rows, region, kind, "void", "void", "mean_area_px2", total / len(areas), "px^2")


def _region_lines(result: ImageResult, rows: List[Dict[str, str]]) -> List[str]:
    header = f"=== {result.image_name} ({result.image_type.value}) ==="
    block = [header]
    if result.error:
        block.append(f"  ERROR: {result.error}")
        return block
    used = ", ".join(result.detectors_used) or "(none)"
    block.append(f"  Detectors run: {used}")
    _island_line_block(result, block, rows)
    _stripe_misalignment_block(result, block, rows)
    _roughness_block(result, block, rows)
    _void_block(result, block, rows)
    other = [
        name for name in result.detections
        if name not in {
            "line_defect", "stripe_misalignment", "edge_roughness", "void",
        }
    ]
    for name in other:
        det = result.detections[name]
        scored = [
            item for item in (det.defects or [])
            if isinstance(item, dict) and str(item.get("type")) not in _SUMMARY_KINDS
        ]
        block.append(f"  {name}: {len(scored)} finding(s)")
        _csv_row(
            rows, result.image_name, result.image_type.value, name, name,
            "finding_count", len(scored), "count",
        )
    if len(block) == 2:
        block.append("  No enabled-detector findings.")
    return block


def _totals(results: Sequence[ImageResult]) -> List[str]:
    missing_nozzles = 0
    missing_gaps = 0
    stitches_island = 0
    stitches_stripe = 0
    voids = 0
    roughness_flagged = 0
    for result in results:
        for item in _of_type(_items(result, "line_defect"), "bands_detected"):
            missing_nozzles += int(_num(item.get("estimated_missing_nozzles")))
            missing_gaps += int(_num(item.get("total_missing_defects")))
            stitches_island += int(_num(item.get("total_stitch_defects")))
        stitches_stripe += len(_of_type(_items(result, "stripe_misalignment"), "stripe_misalignment"))
        voids += len(_of_type(_items(result, "void"), "void"))
        summary = _of_type(_items(result, "edge_roughness"), "edge_roughness_summary")
        if summary and (summary[0].get("left_flagged") or summary[0].get("right_flagged")):
            roughness_flagged += 1
    return [
        "Totals across all regions",
        f"  Estimated missing nozzles:     {missing_nozzles}",
        f"  Missing-nozzle gaps:           {missing_gaps}",
        f"  Island stitch / calibration:   {stitches_island}",
        f"  Stripe stitch steps:           {stitches_stripe}",
        f"  Stripes with roughness flag:   {roughness_flagged}",
        f"  Voids:                         {voids}",
    ]


def write_defect_summary(
    results: Sequence[ImageResult],
    output_dir: str,
    settings: Optional[RunSettings] = None,
    generated_at: Optional[datetime] = None,
) -> Tuple[str, str]:
    """Write `defect_summary.txt` and `defect_summary.csv` next to the JSON report."""
    os.makedirs(output_dir, exist_ok=True)
    when = generated_at or datetime.now()
    stamp = when.strftime("%Y-%m-%d %H:%M:%S")
    enabled = []
    if settings is not None:
        sets = settings.config.detector_sets
        enabled = list(sets.island) + list(sets.stripe) + [
            key for key in sets.unknown if key not in sets.island and key not in sets.stripe
        ]
        if settings.only_detectors:
            enabled = [key for key in enabled if key in set(settings.only_detectors)]
    rows: List[Dict[str, str]] = []
    lines = [
        "Nike print-defect summary",
        f"Generated:        {stamp}",
        f"Result folder:    {os.path.basename(os.path.abspath(output_dir))}",
        f"Sensitivity:      {getattr(settings, 'sensitivity', 'n/a')}",
        f"Island pattern:   {getattr(settings, 'pattern', 'n/a')}",
        f"Detectors enabled: {', '.join(enabled) if enabled else '(see per-region list)'}",
        "",
    ]
    lines.extend(_totals(results))
    lines.append("")
    for result in results:
        lines.extend(_region_lines(result, rows))
        lines.append("")

    txt_path = os.path.join(output_dir, "defect_summary.txt")
    with open(txt_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines).rstrip() + "\n")

    csv_path = os.path.join(output_dir, "defect_summary.csv")
    fieldnames = [
        "region", "region_type", "detector", "kind",
        "metric", "value", "unit", "notes",
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    logger.info("Wrote defect summary to %s and %s", txt_path, csv_path)
    return txt_path, csv_path
