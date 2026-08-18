"""Load once, shared context, parallel detectors, then write results."""

from __future__ import annotations

import json
import logging
import multiprocessing as mp
import os
import re
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import cv2

from nike_detection.config.loader import load_regions_file
from nike_detection.config.schema import RegionSpec, RunSettings
from nike_detection.geometry.full_region_detector import (
    color_prefix_from_path,
    detect_full_regions,
    draw_corner_montage,
    draw_region_overlay,
)
from nike_detection.io.defect_summary import write_defect_summary
from nike_detection.io.image_loader import MEMMAP_MIN_BYTES, Scan, load_bgr_gray, open_scan
from nike_detection.io.region_views import slice_view
from nike_detection.io.results import save_image_result, save_summary_report
from nike_detection.io.visualization import (
    draw_full_defect_overlay,
    save_debug,
    save_visualization,
)
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
RUN_OUTPUT_DIR_RE = re.compile(r"^output_\d{8}_\d{6}$", re.IGNORECASE)
NAMED_RUN_DIR_RE = re.compile(r"_\d{2}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2}$")
GENERATED_NAME_TOKENS = ("_full_regions", "_visualization", "_corners")
_WINDOWS_BAD_CHARS = re.compile(r'[<>:"/\\|?*]')


def _cpu_count() -> int:
    return max(1, int(os.cpu_count() or 1))


def cap_workers(
    requested: int,
    n_jobs: int,
    sample_shapes: Optional[Sequence[Tuple[int, int]]] = None,
) -> int:
    """Clamp process workers to jobs, CPU, and an 8 GB working-set budget."""
    requested = max(1, int(requested))
    n_jobs = max(1, int(n_jobs))
    n = min(requested, n_jobs, _cpu_count())
    if sample_shapes:
        worst = 0
        for height, width in sample_shapes:
            worst = max(worst, int(height) * int(width) * BYTES_PER_BGR_PIXEL * 2)
        if worst > 0:
            fit = max(1, int(MAX_RAM_BUDGET // worst))
            if fit < n:
                logger.warning(
                    "Capping workers from %s to %s (estimated %.1f GB per job)",
                    n, fit, worst / (1024 ** 3),
                )
            n = min(n, fit)
    return max(1, n)


def cap_image_workers(requested: int, sample_shape: Optional[Tuple[int, ...]] = None) -> int:
    shapes = None
    if sample_shape is not None and len(sample_shape) >= 2:
        shapes = [(int(sample_shape[0]), int(sample_shape[1]))]
    return cap_workers(requested, requested, shapes)


def _safe_stem(stem: str) -> str:
    cleaned = _WINDOWS_BAD_CHARS.sub("_", stem).strip(" ._")
    return cleaned or "image"


def run_output_stamp(when: Optional[datetime] = None) -> str:
    """``MM_DD_YY_HH_MM_SS`` used in result folder names."""
    return (when or datetime.now()).strftime("%m_%d_%y_%H_%M_%S")


def output_dir_for_image(
    image_path: str,
    settings: RunSettings,
    stamp: str,
    *,
    nest_under_override: bool = False,
) -> str:
    """Results live next to the TIFF as ``{stem}_{MM}_{DD}_{YY}_{HH}_{MM}_{SS}``.

    ``--output`` overrides that location. When one ``-o`` is shared across
    several inputs, each image is nested as ``{output}/{stem}_{stamp}``.
    """
    stem = _safe_stem(Path(image_path).stem)
    named = f"{stem}_{stamp}"
    if settings.output_dir:
        if nest_under_override:
            return os.path.join(os.path.abspath(settings.output_dir), named)
        return os.path.abspath(settings.output_dir)
    parent = os.path.dirname(os.path.abspath(image_path)) or "."
    return os.path.join(parent, named)


def manual_regions(settings: RunSettings, image_path: str) -> Optional[List[RegionSpec]]:
    """Operator-supplied boxes for a full scan, if any.

    Only ``--regions`` or a sibling JSON override detection; there are no
    seed boxes, so without an override the regions are measured from the
    print itself.
    """
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
                logger.info("Using operator regions from %s", sibling)
                return specs
        except Exception:
            logger.debug("Sibling JSON %s is not a regions file", sibling)
    return None


def _is_run_output_dir(name: str) -> bool:
    """True for a directory this pipeline itself created."""
    if bool(RUN_OUTPUT_DIR_RE.match(name)) or name.lower().endswith("_crops"):
        return True
    return bool(NAMED_RUN_DIR_RE.search(name))


def _is_generated_artifact(name: str) -> bool:
    lowered = name.lower()
    return any(token in lowered for token in GENERATED_NAME_TOKENS)


def collect_image_files(
    folder: str, recursive: bool = True, include_unknown: bool = False
) -> List[str]:
    """Input scans under ``folder``, never this pipeline's own output.

    A previous run's overlays and visualizations are images sitting in a
    sub-folder, so without this filter a second run would "detect" regions
    on its own JPEGs.
    """
    found: List[str] = []
    if recursive:
        for root, dirs, files in os.walk(folder):
            dirs[:] = [d for d in dirs if not _is_run_output_dir(d)]
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
    generated = []
    for path in found:
        name = os.path.basename(path)
        if _is_generated_artifact(name):
            generated.append(name)
            continue
        kind = classify_image(path)
        if kind in (ImageType.STRIPE, ImageType.ISLAND, ImageType.FULL) or include_unknown:
            kept.append(path)
        else:
            skipped.append(name)
    if generated:
        logger.info("Skipping %s previously generated artifact(s)", len(generated))
    if skipped:
        logger.info("Skipping %s non-stripe/island/full file(s)", len(skipped))
    return sorted(kept)


def _spec_shape(spec: RegionSpec) -> Tuple[int, int]:
    x0, y0, x1, y1 = spec.bounding_box_pixels.as_int_xyxy()
    return max(1, y1 - y0), max(1, x1 - x0)


def _detector_threads_for_pool(settings: RunSettings, region_workers: int) -> int:
    """Leave cores for sibling region processes."""
    per = max(1, _cpu_count() // max(1, region_workers))
    return max(1, min(int(settings.max_detector_threads), per))


def _init_worker() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        force=True,
    )


def _run_one_full_region(
    image_path: str,
    spec: RegionSpec,
    output_root: str,
    settings: RunSettings,
    load_seconds: float,
    file_size_mb: float,
    stem: str,
    crop_dir: Optional[str],
) -> ImageResult:
    """Crop one colour region from a (possibly memory-mapped) sheet and detect."""
    scan = open_scan(image_path)
    try:
        view = slice_view(scan, spec)
    finally:
        del scan
    if view is None:
        return ImageResult(
            image_name=spec.name,
            image_path=image_path,
            image_type=ImageType.ISLAND if spec.type == "island" else ImageType.STRIPE,
            processing_time=datetime.now().isoformat(),
            file_size_mb=file_size_mb,
            detectors_used=[],
            detections={},
            error=f"Region '{spec.name}' has no overlap with the scan",
            parent_image=stem,
        )
    if crop_dir is not None:
        os.makedirs(crop_dir, exist_ok=True)
        cv2.imwrite(os.path.join(crop_dir, f"{view.name}.tiff"), view.bgr)
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
    return result


def _full_region_job(job: Tuple[Any, ...]) -> ImageResult:
    (
        image_path, spec, output_root, settings, load_seconds,
        file_size_mb, stem, crop_dir, detector_threads,
    ) = job
    local = replace(settings, max_detector_threads=int(detector_threads))
    try:
        return _run_one_full_region(
            image_path, spec, output_root, local,
            load_seconds, file_size_mb, stem, crop_dir,
        )
    except Exception as exc:
        logger.exception("Region %s failed on %s", spec.name, image_path)
        return ImageResult(
            image_name=spec.name,
            image_path=image_path,
            image_type=ImageType.ISLAND if spec.type == "island" else ImageType.STRIPE,
            processing_time=datetime.now().isoformat(),
            file_size_mb=file_size_mb,
            detectors_used=[],
            detections={},
            error=str(exc),
            parent_image=stem,
        )


def _map_jobs(fn, jobs: Sequence[Any], workers: int) -> List[Any]:
    """Run ``fn`` over jobs, preserving order.

    Uses processes from the parent, and threads if we are already inside a
    worker process (Windows spawn workers are daemonic and cannot nest pools).
    """
    if not jobs:
        return []
    if workers <= 1 or len(jobs) == 1:
        return [fn(job) for job in jobs]
    ordered: List[Any] = [None] * len(jobs)
    use_threads = bool(mp.current_process().daemon)
    executor_cls = ThreadPoolExecutor if use_threads else ProcessPoolExecutor
    kwargs = {"max_workers": workers}
    if not use_threads:
        kwargs["initializer"] = _init_worker
    logger.info(
        "Dispatching %s job(s) on %s %s",
        len(jobs), workers, "threads" if use_threads else "processes",
    )
    with executor_cls(**kwargs) as pool:
        future_index = {pool.submit(fn, job): index for index, job in enumerate(jobs)}
        for fut in as_completed(future_index):
            ordered[future_index[fut]] = fut.result()
    return ordered


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
    write_folders = bool(settings.write_region_folders)
    image_output = os.path.join(output_root, ctx.name)
    if write_folders:
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
    if write_folders:
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
                ctx.name, name, detections[name].elapsed_seconds,
                detections[name].defect_count,
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

    for name, detector, defects in rendered:
        logger.info(
            "%s / %s: %.2fs, %s defect(s)",
            ctx.name, name, detections[name].elapsed_seconds,
            detections[name].defect_count,
        )
    return ImageResult(
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


def process_image_path(
    image_path: str,
    output_root: str,
    settings: RunSettings,
) -> List[ImageResult]:
    image_path = os.path.abspath(image_path)
    kind = classify_image(image_path)
    file_size = os.path.getsize(image_path)
    # --regions-only already promoted unknown TIFFs to a full sheet. Detection
    # on a 4-colour scan must do the same: the file is too large for OpenCV
    # and the name often has no "full" token.
    if kind == ImageType.UNKNOWN and (
        settings.regions_only
        or settings.regions_path
        or file_size >= MEMMAP_MIN_BYTES
    ):
        kind = ImageType.FULL
        logger.info(
            "Treating %s as a full scan (%s)",
            os.path.basename(image_path),
            "regions override" if settings.regions_path or settings.regions_only
            else f"{file_size / 1e9:.1f} GB",
        )
    logger.info("Processing %s (%s)", os.path.basename(image_path), kind.value)

    load_t0 = time.perf_counter()
    file_size_mb = file_size / (1024 * 1024)
    stem = Path(image_path).stem

    if kind == ImageType.FULL:
        # A full sheet is opened as a Scan and cropped per region, so a
        # multi-gigabyte scan never has to be decoded whole.
        scan = open_scan(image_path)
        load_seconds = time.perf_counter() - load_t0
        override = manual_regions(settings, image_path)
        if override is not None:
            specs = override
            region_debug = {
                "prefix": color_prefix_from_path(image_path),
                "sources": {"regions": "operator-supplied"},
            }
        else:
            specs, region_debug = detect_full_regions(
                scan, settings.config, image_path=image_path,
            )
        _write_region_artifacts(scan, specs, region_debug, output_root, stem)
        del scan
        if settings.regions_only:
            prefix = region_debug.get("prefix") or stem
            return [ImageResult(
                image_name=f"{prefix}_full_regions",
                image_path=image_path,
                image_type=ImageType.FULL,
                processing_time=datetime.now().isoformat(),
                file_size_mb=file_size_mb,
                detectors_used=[],
                detections={},
                load_seconds=round(load_seconds, 3),
                elapsed_seconds=round(load_seconds, 3),
            )]
        crop_dir = None
        if settings.write_crops:
            crop_dir = os.path.join(output_root, f"{stem}_crops")
            os.makedirs(crop_dir, exist_ok=True)
        shapes = [_spec_shape(spec) for spec in specs]
        region_workers = cap_workers(settings.max_image_workers, len(specs), shapes)
        det_threads = _detector_threads_for_pool(settings, region_workers)
        logger.info(
            "Detecting %s region(s) on %s with up to %s parallel workers, "
            "%s detector thread(s) each",
            len(specs), os.path.basename(image_path), region_workers, det_threads,
        )
        jobs = [
            (
                image_path, spec, output_root, settings, load_seconds,
                file_size_mb, stem, crop_dir, det_threads,
            )
            for spec in specs
        ]
        results = _map_jobs(_full_region_job, jobs, region_workers)
        results = [item for item in results if item is not None]
        if not results:
            raise ValueError(f"No valid region views produced for {image_path}")
        logger.info("Finished %s region(s) on %s", len(results), os.path.basename(image_path))
        _write_full_defect_overlay(
            image_path, results, specs, region_debug, output_root, stem, settings,
        )
        return results

    bgr, gray = load_bgr_gray(image_path)
    load_seconds = time.perf_counter() - load_t0
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


def _write_region_artifacts(
    scan: Scan,
    specs: List[RegionSpec],
    debug: Dict[str, Any],
    output_root: str,
    stem: str,
) -> None:
    os.makedirs(output_root, exist_ok=True)
    prefix = str(debug.get("prefix") or Path(stem).stem)
    payload = {
        "image_stem": stem,
        "reference": debug.get("reference"),
        "warnings": debug.get("warnings") or [],
        "colors": debug.get("colors"),
        "regions": [
            {
                "name": spec.name,
                "type": spec.type,
                "bounding_box_pixels": {
                    "top_x": spec.bounding_box_pixels.top_x,
                    "top_y": spec.bounding_box_pixels.top_y,
                    "bottom_x": spec.bounding_box_pixels.bottom_x,
                    "bottom_y": spec.bounding_box_pixels.bottom_y,
                },
            }
            for spec in specs
        ],
    }
    json_path = os.path.join(output_root, f"{prefix}_full_regions.json")
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    overlay_path = os.path.join(output_root, f"{prefix}_full_regions.jpg")
    cv2.imwrite(
        overlay_path, draw_region_overlay(scan, specs, debug=debug),
        [cv2.IMWRITE_JPEG_QUALITY, 90],
    )
    corners = draw_corner_montage(scan, specs)
    if corners is not None:
        cv2.imwrite(
            os.path.join(output_root, f"{prefix}_full_regions_corners.jpg"),
            corners, [cv2.IMWRITE_JPEG_QUALITY, 92],
        )
    logger.info("Wrote region boxes to %s and %s", json_path, overlay_path)


def _write_full_defect_overlay(
    image_path: str,
    results: List[ImageResult],
    specs: List[RegionSpec],
    debug: Dict[str, Any],
    output_root: str,
    stem: str,
    settings: RunSettings,
) -> None:
    if not settings.write_full_defect_overlay or not settings.write_visualizations:
        return
    if settings.regions_only:
        return
    os.makedirs(output_root, exist_ok=True)
    prefix = str(debug.get("prefix") or Path(stem).stem)
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    scan = open_scan(image_path)
    try:
        overlay = draw_full_defect_overlay(
            scan, results, specs=specs, stamp=stamp,
        )
    finally:
        del scan
    overlay_path = os.path.join(output_root, f"{prefix}_full_defects.jpg")
    cv2.imwrite(overlay_path, overlay, [cv2.IMWRITE_JPEG_QUALITY, 90])
    logger.info("Wrote full-scan defect overlay to %s", overlay_path)


def _process_image_job(args: Tuple[str, str, RunSettings]) -> Tuple[str, List[ImageResult]]:
    image_path, output_root, settings = args
    results = process_image_path(image_path, output_root, settings)
    return output_root, results


def _save_image_summary(
    results: List[ImageResult],
    output_root: str,
    settings: RunSettings,
    batch_elapsed: Optional[float] = None,
) -> None:
    os.makedirs(output_root, exist_ok=True)
    save_summary_report(
        results, output_root, settings.sensitivity, settings.pattern,
        generate_pdf=settings.generate_report,
        batch_elapsed=batch_elapsed,
    )
    if not settings.regions_only:
        write_defect_summary(results, output_root, settings)


def run_paths(
    paths: Sequence[str],
    settings: RunSettings,
    output_parent: Optional[str] = None,
) -> Tuple[List[ImageResult], str]:
    if not paths:
        raise ValueError("No input images to process")
    stamp = run_output_stamp()
    nest = bool(settings.output_dir) and len(paths) > 1
    batch_t0 = time.perf_counter()
    all_results: List[ImageResult] = []
    output_roots: List[str] = []

    def dest(path: str) -> str:
        out = output_dir_for_image(path, settings, stamp, nest_under_override=nest)
        os.makedirs(out, exist_ok=True)
        return out

    image_workers = cap_workers(settings.max_image_workers, len(paths))
    if len(paths) == 1 or image_workers == 1:
        for path in paths:
            out = dest(path)
            part = process_image_path(path, out, settings)
            _save_image_summary(part, out, settings)
            all_results.extend(part)
            output_roots.append(out)
    else:
        logger.info("Processing %s images with %s process worker(s)", len(paths), image_workers)
        jobs = [(path, dest(path), settings) for path in paths]
        mapped = _map_jobs(_process_image_job, jobs, image_workers)
        for out, part in mapped:
            _save_image_summary(part, out, settings)
            all_results.extend(part)
            output_roots.append(out)

    batch_elapsed = time.perf_counter() - batch_t0
    output_root = output_roots[0]
    if len(output_roots) == 1:
        _save_image_summary(all_results, output_root, settings, batch_elapsed=batch_elapsed)
    logger.info(
        "Processed %s region(s) in %.2fs -> %s",
        len(all_results), batch_elapsed,
        output_root if len(output_roots) == 1 else f"{len(output_roots)} folders",
    )
    return all_results, output_root
