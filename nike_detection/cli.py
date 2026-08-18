"""Single CLI for 2400-DPI extraction + detection."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from multiprocessing import freeze_support
from pathlib import Path
from typing import List, Optional

from nike_detection.config.loader import DEFAULT_CONFIG_PATH, load_config
from nike_detection.config.schema import PATTERN_VALUES, SENSITIVITY_LEVELS, RunSettings
from nike_detection.io.extract import extract_legacy, extract_new_pattern
from nike_detection.pipeline.registry import ALL_DETECTOR_KEYS
from nike_detection.pipeline.runner import collect_image_files, run_paths
from nike_detection.pipeline.types import ImageType, classify_image

logger = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nike_detection",
        description="2400-DPI print defect detection (shared context, parallel detectors)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m nike_detection -i Cyan_full.tiff --pattern new --regions-only
  python -m nike_detection -i Cyan_full.tiff --pattern new
  python -m nike_detection -i Cyan_full.tiff --pattern new --regions regions.json
  python -m nike_detection -i KeyIsland.tiff --pattern new --only line_defect
  python -m nike_detection -i CyanStripe.tiff --only void stripe_misalignment -s high
  python -m nike_detection -i extracted_folder --pattern new --workers 2 --no-vis
  python -m nike_detection -i scan.tif --extract --pattern new
""",
    )
    parser.add_argument("-i", "--input", dest="input_path", required=True,
                        help="Image file, 'full' TIFF, or folder of crops")
    parser.add_argument("-o", "--output",
                        help="Output folder (default: {image_name}_MM_DD_YY_HH_MM_SS next to the TIFF)")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH),
                        help="Unified detection_2400.json (all thresholds live here)")
    parser.add_argument("-s", "--sensitivity", choices=list(SENSITIVITY_LEVELS),
                        default=None, help="Sensitivity preset (default from config)")
    parser.add_argument("--pattern", choices=list(PATTERN_VALUES), default=None,
                        help="Island pattern: legacy single-band or new dual-band")
    parser.add_argument("--clear", action="store_true",
                        help="Clear scan material (requires --pattern new)")
    parser.add_argument("--only", nargs="+", metavar="DETECTOR",
                        help="Run only these detector keys")
    parser.add_argument("--regions", help="JSON with stripe/island bounding boxes for a 'full' TIFF")
    parser.add_argument("--regions-only", action="store_true",
                        help="Detect island/stripe boxes and write <color>_full_regions.jpg; skip detectors")
    parser.add_argument("--extract", action="store_true",
                        help="Extract regions from a full press scan before detection")
    parser.add_argument("--extract-config",
                        help="Legacy bbox JSON used when --extract --pattern legacy")
    parser.add_argument("--generate_report", action="store_true", help="Write PDF summary")
    parser.add_argument("--no-vis", action="store_true", help="Skip visualization images")
    parser.add_argument("--downscale-vis", action="store_true",
                        help="Downscale overlays (faster writes)")
    parser.add_argument("--debug", action="store_true", help="Save detector debug artifacts")
    parser.add_argument("--workers", type=int, default=None,
                        help="Parallel processes: regions of a full scan, or images in a folder")
    parser.add_argument("--detector-threads", type=int, default=None,
                        help="Threads per region after shared geometry (capped by CPU / workers)")
    parser.add_argument("--write-crops", action="store_true",
                        help="When processing a 'full' TIFF, also write region crops")
    parser.add_argument("--region-folders", action="store_true",
                        help="Also write per-region subfolders with visualizations (off by default)")
    parser.add_argument("--no-full-overlay", action="store_true",
                        help="Skip the full-scan annotated defect image at the result-folder root")
    parser.add_argument("--no-recursive", action="store_true",
                        help="Folder input: only the top level")
    parser.add_argument("--include-unknown", action="store_true",
                        help="Folder input: also process files that are not stripe/island/full")
    parser.add_argument("--dpi", choices=["2400"], default="2400",
                        help="Must be 2400 (other DPIs are not supported)")
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser


def _settings_from_args(args, config) -> RunSettings:
    defaults = config.defaults
    pattern = args.pattern or defaults.pattern
    clear = bool(args.clear or defaults.clear)
    if clear and pattern != "new":
        raise SystemExit("--clear is only supported with --pattern new")
    only = args.only
    if only:
        bad = set(only) - set(ALL_DETECTOR_KEYS)
        if bad:
            raise SystemExit(f"Unknown --only keys: {sorted(bad)}")
    return RunSettings(
        config=config,
        sensitivity=args.sensitivity or defaults.sensitivity,
        pattern=pattern,
        clear=clear,
        only_detectors=only,
        output_dir=args.output,
        generate_report=args.generate_report,
        write_visualizations=defaults.write_visualizations and not args.no_vis,
        downscale_vis=bool(args.downscale_vis or defaults.downscale_vis),
        debug=args.debug,
        max_image_workers=args.workers or defaults.max_image_workers,
        max_detector_threads=args.detector_threads or defaults.max_detector_threads,
        write_crops=bool(args.write_crops or defaults.write_crops),
        write_region_folders=bool(args.region_folders or defaults.write_region_folders),
        write_full_defect_overlay=(
            defaults.write_full_defect_overlay
            and not args.no_full_overlay
            and not args.no_vis
        ),
        regions_path=args.regions,
        recursive=not args.no_recursive,
        include_unknown=bool(args.include_unknown or args.regions_only),
        extract_config_path=args.extract_config,
        regions_only=bool(args.regions_only),
    )


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    config = load_config(args.config)
    settings = _settings_from_args(args, config)

    input_path = os.path.abspath(args.input_path)
    if not os.path.exists(input_path):
        logger.error("Input path does not exist: %s", input_path)
        return 1

    paths: List[str]
    if os.path.isdir(input_path):
        paths = collect_image_files(
            input_path, recursive=settings.recursive,
            include_unknown=settings.include_unknown,
        )
        if not paths:
            logger.error("No matching images in %s", input_path)
            return 1
        output_parent = input_path
    else:
        kind = classify_image(input_path)
        if args.extract or (
            kind == ImageType.UNKNOWN
            and input_path.lower().endswith((".tif", ".tiff"))
            and args.extract
        ):
            if settings.pattern == "new":
                extracted = extract_new_pattern(input_path, config, args.output)
            else:
                extract_cfg = args.extract_config
                if not extract_cfg:
                    extract_cfg = str(
                        Path(__file__).resolve().parents[1]
                        / "regions_json" / "template-2400-configs.json"
                    )
                extracted = extract_legacy(input_path, extract_cfg)
            paths = collect_image_files(extracted, recursive=True, include_unknown=False)
            output_parent = extracted
        else:
            paths = [input_path]
            output_parent = os.path.dirname(input_path) or "."

        if kind == ImageType.FULL and not args.extract:
            try:
                from nike_detection.pipeline.runner import manual_regions
                manual_regions(settings, input_path)
            except ValueError as exc:
                logger.error("%s", exc)
                return 1

    logger.info(
        "Run: sensitivity=%s pattern=%s clear=%s detectors=%s images=%s",
        settings.sensitivity, settings.pattern, settings.clear,
        settings.only_detectors or "default", len(paths),
    )
    results, output_root = run_paths(paths, settings, output_parent=output_parent)
    print(f"Results saved to: {output_root}")
    print(f"Processed {len(results)} region(s)")
    return 0


if __name__ == "__main__":
    freeze_support()
    sys.exit(main())
