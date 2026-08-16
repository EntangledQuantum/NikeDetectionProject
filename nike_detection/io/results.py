"""JSON / PDF result writers."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np

from nike_detection.pipeline.types import ImageResult

logger = logging.getLogger(__name__)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, tuple):
        return list(obj)
    if isinstance(obj, Enum):
        return obj.value
    if hasattr(obj, "__dict__"):
        return obj.__dict__
    return str(obj)


def save_image_result(result: ImageResult, output_dir: str) -> str:
    path = f"{output_dir}/{result.image_name}_results.json"
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(asdict(result), handle, indent=2, default=_json_default)
    return path


def _timing(results: List[ImageResult], batch_elapsed: Optional[float]) -> Dict[str, Any]:
    by_type: Dict[str, Dict[str, float]] = {}
    by_detector: Dict[str, Dict[str, float]] = {}
    per_image = []
    for result in results:
        kind = result.image_type.value
        entry = by_type.setdefault(kind, {"total_seconds": 0.0, "image_count": 0})
        entry["total_seconds"] += float(result.elapsed_seconds or 0.0)
        entry["image_count"] += 1
        per_image.append({
            "image_name": result.image_name,
            "image_type": kind,
            "elapsed_seconds": float(result.elapsed_seconds or 0.0),
            "load_seconds": float(result.load_seconds or 0.0),
            "context_seconds": float(result.context_seconds or 0.0),
            "detect_seconds": float(result.detect_seconds or 0.0),
            "write_seconds": float(result.write_seconds or 0.0),
            "detectors": {
                name: float(det.elapsed_seconds or 0.0)
                for name, det in result.detections.items()
            },
        })
        for name, det in result.detections.items():
            d = by_detector.setdefault(name, {"total_seconds": 0.0, "call_count": 0})
            d["total_seconds"] += float(det.elapsed_seconds or 0.0)
            d["call_count"] += 1
    for entry in by_type.values():
        n = max(1, int(entry["image_count"]))
        entry["total_seconds"] = round(entry["total_seconds"], 3)
        entry["average_seconds_per_image"] = round(entry["total_seconds"] / n, 3)
    for entry in by_detector.values():
        n = max(1, int(entry["call_count"]))
        entry["total_seconds"] = round(entry["total_seconds"], 3)
        entry["average_seconds"] = round(entry["total_seconds"] / n, 3)
    out: Dict[str, Any] = {
        "by_image_type": by_type,
        "by_detector": by_detector,
        "per_image": per_image,
        "sum_of_image_seconds": round(
            sum(float(r.elapsed_seconds or 0.0) for r in results), 3
        ),
    }
    if batch_elapsed is not None:
        out["batch_elapsed_seconds"] = round(float(batch_elapsed), 3)
    return out


def _stats(results: List[ImageResult]) -> Dict[str, Any]:
    type_dist: Dict[str, int] = {}
    defect_stats: Dict[str, Dict[str, Any]] = {}
    summary = {
        "successful_images": 0,
        "failed_images": 0,
        "total_processing_time": 0.0,
        "average_file_size_mb": 0.0,
    }
    total_size = 0.0
    for result in results:
        type_dist[result.image_type.value] = type_dist.get(result.image_type.value, 0) + 1
        if result.error:
            summary["failed_images"] += 1
        else:
            summary["successful_images"] += 1
        total_size += result.file_size_mb
        summary["total_processing_time"] += float(result.elapsed_seconds or 0.0)
        for name, det in result.detections.items():
            slot = defect_stats.setdefault(name, {
                "total_defects": 0, "affected_images": 0, "total_seconds": 0.0,
            })
            slot["total_defects"] += det.defect_count
            slot["total_seconds"] += float(det.elapsed_seconds or 0.0)
            if det.defect_count > 0:
                slot["affected_images"] += 1
    for slot in defect_stats.values():
        slot["total_seconds"] = round(slot["total_seconds"], 3)
    summary["total_processing_time"] = round(summary["total_processing_time"], 3)
    summary["average_file_size_mb"] = total_size / len(results) if results else 0.0
    return {
        "image_type_distribution": type_dist,
        "defect_statistics": defect_stats,
        "processing_summary": summary,
    }


def save_summary_report(
    results: List[ImageResult],
    output_dir: str,
    sensitivity: str,
    pattern: str,
    generate_pdf: bool = False,
    batch_elapsed: Optional[float] = None,
) -> str:
    stats = _stats(results)
    timing = _timing(results, batch_elapsed)
    payload = {
        "timestamp": datetime.now().isoformat(),
        "total_images": len(results),
        "detection_sensitivity": sensitivity,
        "island_pattern": pattern,
        "image_type_distribution": stats["image_type_distribution"],
        "defect_statistics": stats["defect_statistics"],
        "processing_summary": stats["processing_summary"],
        "timing": timing,
        "detailed_results": [asdict(result) for result in results],
    }
    path = f"{output_dir}/defect_report.json"
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=_json_default)
    logger.info("Summary report saved to %s", path)
    if generate_pdf:
        _write_pdf(results, stats, timing, output_dir)
    return path


def _write_pdf(results, stats, timing, output_dir: str) -> None:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
    except Exception:
        logger.warning("matplotlib not available; skipping PDF report")
        return
    pdf_path = f"{output_dir}/defect_detection_report.pdf"
    with PdfPages(pdf_path) as pdf:
        fig, ax = plt.subplots(figsize=(8.5, 11))
        fig.suptitle("Defect Detection Summary Report", fontsize=20, fontweight="bold")
        text = f"Total Images Processed: {len(results)}\n\n"
        text += "Image Type Distribution:\n" + "-" * 40 + "\n"
        for kind, count in stats["image_type_distribution"].items():
            text += f"  {kind.title()}: {count} images\n"
        text += "\nDefect Summary:\n" + "-" * 40 + "\n"
        for name, det in stats["defect_statistics"].items():
            text += f"\n{name.replace('_', ' ').title()}:\n"
            text += f"  Total Defects Found: {det['total_defects']}\n"
            text += f"  Images with Defects: {det['affected_images']}\n"
            text += f"  Total Time: {det.get('total_seconds', 0):.2f}s\n"
        text += "\nTiming (wall-clock):\n" + "-" * 40 + "\n"
        if timing.get("batch_elapsed_seconds") is not None:
            text += f"  Batch total: {timing['batch_elapsed_seconds']:.2f}s\n"
        ax.text(0.1, 0.9, text, transform=ax.transAxes, fontsize=10,
                verticalalignment="top", fontfamily="monospace")
        ax.axis("off")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close()
    logger.info("PDF report saved to %s", pdf_path)
