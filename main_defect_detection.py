#!/usr/bin/env python3
"""Shim: extract a 2400-DPI scan then run the modular detection package."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from digital_air_cv.cli import main as package_main


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract 2400-DPI regions then run defect detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="This entry point now delegates to `python -m digital_air_cv --extract`.",
    )
    parser.add_argument("--image", "-i", required=True, help="Path to the TIFF scan")
    parser.add_argument("--dpi", "-d", choices=["2400"], default="2400",
                        help="Must be 2400")
    parser.add_argument("--config", "-c", help="Custom JSON (legacy bbox, or ignored for --pattern new)")
    parser.add_argument("--sensitivity", "-s", choices=["low", "medium", "high"], default="medium")
    parser.add_argument("--generate_report", action="store_true")
    parser.add_argument("--pattern", choices=["legacy", "new"], default="new")
    parser.add_argument("--clear", action="store_true")
    parser.add_argument("--only", nargs="+")
    parser.add_argument("-o", "--output")
    args = parser.parse_args()

    argv = [
        "-i", args.image,
        "--extract",
        "--pattern", args.pattern,
        "--sensitivity", args.sensitivity,
        "--dpi", "2400",
    ]
    if args.clear:
        argv.append("--clear")
    if args.generate_report:
        argv.append("--generate_report")
    if args.output:
        argv.extend(["-o", args.output])
    if args.only:
        argv.extend(["--only", *args.only])
    if args.config:
        argv.extend(["--extract-config", args.config])
    return package_main(argv)


if __name__ == "__main__":
    sys.exit(main())
