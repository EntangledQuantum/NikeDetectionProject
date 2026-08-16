"""Load once, shared context, parallel detectors, then write results."""

from __future__ import annotations

import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from nike_detection.config.loader import load_regions_file
from nike_detection.config.schema import RegionSpec, RunSettings
from nike_detection.io.image_loader import load_bgr_gray
from nike_detection.io.region_views import RegionView, slice_views
from nike_detection.io.results import save_image_result, save_summary_report
from nike_detection.io.visualization import save_debug, save_visualization
from nike_detection.pipeline.context import ImageContext
from nike_detection.pipeline.preprocess_island import GeometryError
from nike_detection.pipeline.registry import (
    ALL_DETECTOR_KEYS,
    create_detectors,
    filter_keys,
    keys_for_type,
)
from nike_detection.pipeline.types import (
    DetectionResult,
    ImageResult,
    ImageType,
    classify_image,
)

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp")
BYTES_PER_BGR_PIXEL = 3
MAX_RAM_BUDGET = 8 * 1024 ** 3


def cap_image_workers(requested: int, sample_shape: Optional[Tuple[int, ...]] = None) -> int:
    requested = max(1, int(requested))
    if sample_shape is None:
        return requested
    h, w = int(sample_shape[0]), int(sample_shape[1])
    per_image = h * w * BYTES_PER_BGR_PIXEL * 2  # bgr + gray-ish working set
    if per_image <= 0:
        return requested
    fit = max(1, int(MAX_RAM_BUDGET // per_image))
    if fit < requested:
        logger.warning(
            "Capping image workers from %s to %s (estimated %0.1f GB per image)",
            requested, fit, per_image / (1024 ** 3),
        )
    return min(requested, fit)


def resolve_regions(settings: RunSettings, image_path: str) -> List[RegionSpec]:
    if settings.regions_path:
        specs = load_regions_file(settings.regions_path)
        if not specs:
            raise ValueError(f"--regions file has no regions: {settings.regions_path}")
        return specs
    sibling = Path(image_path).with_suffix(".json")
    if sibling.exists():
        try:
            specs = load_regions_file(str(sibling))
            if specs:
                return specs
        except Exception:
            logger.debug("Sibling JSON %s is not a regions file", sibling)
    if settings.config.regions:
        return list(settings.config.regions)
    raise ValueError(
        "A 'full' image requires region boxes via --regions, a sibling JSON, "
        "or a 'regions' block in detection_2400.json"
    )


def collect_image_files(
    folder: str, recursive: bool = True, include_unknown: bool = False
) -> List[str]:
    found: List[str] = []
    if recursive:
        for root, _dirs, files in os.walk(folder):
            for name in files:
                if name.lower().endswith(IMAGE_EXTENSIONS):
                    found.append(os.path.join(root, name))
    else:
        for name in os.listdir(folder):
            path = os.path.join(folder, name)
            if os.path.isfile(path) and name.lower().endswith(IMAGE_EXTENSIONS):
                found.append(path)
    kept = []
    skipped = []
    for path in found:
        kind = classify_image(path)
        if kind in (ImageType.STRIPE, ImageType.ISLAND, ImageType.FULL) or include_unknown:
            kept.append(path)
        else:
            skipped.append(os.path.basename(path))
    if skipped:
        logger.info("Skipping %s non-stripe/island/full file(s)", len(skipped))
    return sorted(kept)


def _make_output_dir(settings: RunSettings, parent: str) -> str:
    if settings.output_dir:
        out = settings.output_dir
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = os.path.join(parent, f"output_{stamp}")
    os.makedirs(out, exist_ok=True)
    return out


def _run_detectors(
    ctx: ImageContext,
    detectors: Dict[str, Any],
    settings: RunSettings,
) -> Tuple[Dict[str, DetectionResult], float]:
    results: Dict[str, DetectionResult] = {}
    t0 = time.perf_counter()
    workers = max(1, min(settings.max_detector_threads, len(detectors) or 1))

    def _one(name: str, detector: Any) -> Tuple[str, DetectionResult]:
        det_t0 = time.perf_counter()
        try:
            defects = detector.detect(ctx)
            elapsed = time.perf_counter() - det_t0
            payload = [d.to_dict() for d in defects]
            return name, DetectionResult(
                detector=name,
                defect_count=len(defects),
                defects=payload,
                elapsed_seconds=round(elapsed, 3),
            ), detector, defects
        except Exception as exc:
            elapsed = time.perf_counter() - det_t0
            logger.exception("Detector %s failed on %s", name, ctx.name)
            return name, DetectionResult(
                detector=name,
                defect_count=0,
                defects=[],
                error=str(exc),
                elapsed_seconds=round(elapsed, 3),
            ), detector, []

    rendered = []
    if workers == 1 or len(detectors) <= 1:
        for name, detector in detectors.items():
            name, result, detector, defects = _one(name, detector)
            results[name] = result
            rendered.append((name, detector, defects))
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [
                pool.submit(_one, name, detector)
                for name, detector in detectors.items()
            ]
            for fut in as_completed(futures):
                name, result, detector, defects = fut.result()
                results[name] = result
                rendered.append((name, detector, defects))
    return results, time.perf_counter() - t0, rendered


def process_region(
    ctx: ImageContext,
    output_root: str,
    settings: RunSettings,
) -> ImageResult:
    image_output = os.path.join(output_root, ctx.name)
    os.makedirs(image_output, exist_ok=True)

    default_keys = keys_for_type(ctx.image_type, settings)
    try:
        keys = filter_keys(ctx.image_type, default_keys, settings.only_detectors)
    except ValueError as exc:
        return ImageResult(
            image_name=ctx.name,
            image_path=ctx.path,
            image_type=ctx.image_type,
            processing_time=datetime.now().isoformat(),
            file_size_mb=0.0,
            detectors_used=[],
            detections={},
            error=str(exc),
            origin_xy=ctx.origin_xy,
            parent_image=ctx.parent_image,
        )
    if settings.only_detectors:
        ignored = set(settings.only_detectors) - set(keys)
        if ignored:
            logger.warning(
                "Ignoring detectors not available for %s/%s: %s",
                ctx.image_type.value, settings.pattern, sorted(ignored),
            )
    if not keys:
        return ImageResult(
            image_name=ctx.name,
            image_path=ctx.path,
            image_type=ctx.image_type,
            processing_time=datetime.now().isoformat(),
            file_size_mb=0.0,
            detectors_used=[],
            detections={},
            error="No detectors available for this image type",
            origin_xy=ctx.origin_xy,
            parent_image=ctx.parent_image,
        )

    detectors = create_detectors(ctx.image_type, keys, settings)
    needed = set()
    for det in detectors.values():
        needed |= set(det.required_layers())

    ctx_t0 = time.perf_counter()
    try:
        ctx.ensure_layers(needed)
    except GeometryError as exc:
        logger.error("Geometry failed for %s: %s", ctx.name, exc)
        return ImageResult(
            image_name=ctx.name,
            image_path=ctx.path,
            image_type=ctx.image_type,
            processing_time=datetime.now().isoformat(),
            file_size_mb=0.0,
            detectors_used=keys,
            detections={},
            error=str(exc),
            origin_xy=ctx.origin_xy,
            parent_image=ctx.parent_image,
        )
    context_seconds = time.perf_counter() - ctx_t0

    detections, detect_seconds, rendered = _run_detectors(ctx, detectors, settings)

    write_t0 = time.perf_counter()
    for name, detector, defects in rendered:
        vis_path = save_visualization(
            ctx, detector, defects, image_output, name,
            write=settings.write_visualizations,
            downscale_vis=settings.downscale_vis,
        )
        detections[name].visualization_path = vis_path
        save_debug(detector, image_output, ctx.name, settings.debug)
        logger.info(
            "%s / %s: %.2fs, %s defect(s)",
            ctx.name, name, detections[name].elapsed_seconds, detections[name].defect_count,
        )

    result = ImageResult(
        image_name=ctx.name,
        image_path=ctx.path,
        image_type=ctx.image_type,
        processing_time=datetime.now().isoformat(),
        file_size_mb=0.0,
        detectors_used=list(detectors.keys()),
        detections=detections,
        elapsed_seconds=round(context_seconds + detect_seconds, 3),
        context_seconds=round(context_seconds, 3),
        detect_seconds=round(detect_seconds, 3),
        write_seconds=round(time.perf_counter() - write_t0, 3),
        origin_xy=ctx.origin_xy,
        parent_image=ctx.parent_image,
    )
    save_image_result(result, image_output)
    return result


def process_image_path(
    image_path: str,
    output_root: str,
    settings: RunSettings,
) -> List[ImageResult]:
    image_path = os.path.abspath(image_path)
    kind = classify_image(image_path)
    logger.info("Processing %s (%s)", os.path.basename(image_path), kind.value)

    load_t0 = time.perf_counter()
    bgr, gray = load_bgr_gray(image_path)
    load_seconds = time.perf_counter() - load_t0
    file_size_mb = os.path.getsize(image_path) / (1024 * 1024)
    stem = Path(image_path).stem

    if kind == ImageType.FULL:
        specs = resolve_regions(settings, image_path)
        views = slice_views(bgr, gray, specs)
        if settings.write_crops:
            crop_dir = os.path.join(output_root, f"{stem}_crops")
            os.makedirs(crop_dir, exist_ok=True)
            import cv2
            for view in views:
                cv2.imwrite(os.path.join(crop_dir, f"{view.name}.tiff"), view.bgr)
        results: List[ImageResult] = []
        for view in views:
            region_type = ImageType.STRIPE if view.region_type == "stripe" else ImageType.ISLAND
            ctx = ImageContext(
                bgr=view.bgr,
                gray=view.gray,
                path=image_path,
                name=view.name,
                image_type=region_type,
                settings=settings,
                origin_xy=view.origin_xy,
                parent_image=stem,
            )
            result = process_region(ctx, output_root, settings)
            result.load_seconds = round(load_seconds, 3)
            result.file_size_mb = file_size_mb
            result.elapsed_seconds = round(
                (result.elapsed_seconds or 0) + load_seconds + (result.write_seconds or 0), 3
            )
            results.append(result)
        if not results:
            raise ValueError(f"No valid region views produced for {image_path}")
        return results

    ctx = ImageContext(
        bgr=bgr,
        gray=gray,
        path=image_path,
        name=stem,
        image_type=kind,
        settings=settings,
    )
    result = process_region(ctx, output_root, settings)
    result.load_seconds = round(load_seconds, 3)
    result.file_size_mb = file_size_mb
    result.elapsed_seconds = round(
        (result.elapsed_seconds or 0) + load_seconds + (result.write_seconds or 0), 3
    )
    return [result]


def _process_image_job(args: Tuple[str, str, RunSettings]) -> List[ImageResult]:
    image_path, output_root, settings = args
    return process_image_path(image_path, output_root, settings)


def run_paths(
    paths: Sequence[str],
    settings: RunSettings,
    output_parent: Optional[str] = None,
) -> Tuple[List[ImageResult], str]:
    if not paths:
        raise ValueError("No input images to process")
    parent = output_parent or os.path.dirname(os.path.abspath(paths[0])) or "."
    output_root = _make_output_dir(settings, parent)
    batch_t0 = time.perf_counter()
    results: List[ImageResult] = []

    workers = cap_image_workers(settings.max_image_workers)
    if len(paths) == 1 or workers == 1:
        for path in paths:
            results.extend(process_image_path(path, output_root, settings))
    else:
        jobs = [(path, output_root, settings) for path in paths]
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_process_image_job, job) for job in jobs]
            for fut in as_completed(futures):
                results.extend(fut.result())

    batch_elapsed = time.perf_counter() - batch_t0
    save_summary_report(
        results, output_root, settings.sensitivity, settings.pattern,
        generate_pdf=settings.generate_report,
        batch_elapsed=batch_elapsed,
    )
    logger.info("Processed %s region(s) in %.2fs -> %s", len(results), batch_elapsed, output_root)
    return results, output_root
