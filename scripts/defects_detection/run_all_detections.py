#!/usr/bin/env python3
"""Shim around ``python -m nike_detection`` for already-extracted crops."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nike_detection.cli import main as package_main
from nike_detection.pipeline.registry import ALL_DETECTOR_KEYS


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run defect detection on a single island/stripe/full image or a folder",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input", "-i", dest="input_path", default=None)
    parser.add_argument("--input_folder", dest="input_path_alias", default=None,
                        help=argparse.SUPPRESS)
    parser.add_argument("--output", "-o")
    parser.add_argument("--generate_report", action="store_true")
    parser.add_argument("--sensitivity", choices=["low", "medium", "high"], default="medium")
    parser.add_argument("--pattern", choices=["legacy", "new"], default="new")
    parser.add_argument("--clear", action="store_true")
    parser.add_argument("--only", nargs="+", metavar="DETECTOR", choices=ALL_DETECTOR_KEYS)
    parser.add_argument("--regions")
    parser.add_argument("--no-recursive", action="store_true")
    parser.add_argument("--include-unknown", action="store_true")
    parser.add_argument("--no-vis", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--workers", type=int)
    args = parser.parse_args()

    input_path = args.input_path or args.input_path_alias
    if not input_path:
        parser.error("one of -i/--input or --input_folder is required")

    argv = [
        "-i", input_path,
        "--sensitivity", args.sensitivity,
        "--pattern", args.pattern,
    ]
    if args.output:
        argv.extend(["-o", args.output])
    if args.generate_report:
        argv.append("--generate_report")
    if args.clear:
        argv.append("--clear")
    if args.only:
        argv.extend(["--only", *args.only])
    if args.regions:
        argv.extend(["--regions", args.regions])
    if args.no_recursive:
        argv.append("--no-recursive")
    if args.include_unknown:
        argv.append("--include-unknown")
    if args.no_vis:
        argv.append("--no-vis")
    if args.debug:
        argv.append("--debug")
    if args.workers is not None:
        argv.extend(["--workers", str(args.workers)])
    return package_main(argv)


if __name__ == "__main__":
    sys.exit(main())
