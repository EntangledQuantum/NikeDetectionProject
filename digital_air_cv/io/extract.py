"""In-process region extraction (no subprocess)."""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from digital_air_cv.config.schema import AppConfig, GeometryConfig

logger = logging.getLogger(__name__)


def _utility_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "scripts" / "utility"


def _ensure_utility_path() -> None:
    util = str(_utility_dir())
    if util not in sys.path:
        sys.path.insert(0, util)


def geometry_as_extractor_config(geometry: GeometryConfig) -> dict:
    return {
        "dpi": 2400,
        "colors": geometry.colors,
        "color_width": geometry.color_width,
        "x_offset": geometry.x_offset,
        "num_heads": geometry.num_heads,
        "head_height": geometry.head_height,
        "y_offset": geometry.y_offset,
        "island_front": geometry.island_front,
        "island_width": geometry.island_width,
        "stripe_width": geometry.stripe_width,
        "buffer": geometry.buffer,
    }


def extract_new_pattern(image_path: str, config: AppConfig, output_dir: Optional[str] = None) -> str:
    """Extract ColorStripe / ColorIsland crops using the unified 2400 geometry."""
    _ensure_utility_path()
    from new_pattern_tiff_extractor import extract_colors_from_tiff

    image = Path(image_path).resolve()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = Path(output_dir) if output_dir else image.parent / f"{image.stem}_extracted_regions_{stamp}"
    extractor_cfg = geometry_as_extractor_config(config.geometry)
    extract_colors_from_tiff(str(image), extractor_cfg, base, split_stripe_island=True)
    extracted = base / "extracted"
    if not extracted.exists():
        raise RuntimeError(f"Extraction did not create {extracted}")
    logger.info("Extracted new-pattern regions to %s", extracted)
    return str(extracted)


def extract_legacy(image_path: str, regions_json: str) -> str:
    """Legacy bbox extraction via tiff_extractor (in-process)."""
    import json

    _ensure_utility_path()
    import tiff_extractor as extractor

    image = Path(image_path).resolve()
    with open(regions_json, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    data["original_image_path"] = str(image)
    extractor.process_tiff_with_config(data)

    expected = image.parent / f"{image.stem}_output"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = image.parent / f"{image.stem}_extracted_regions_{stamp}"
    if expected.exists():
        expected.rename(dest)
        return str(dest)
    raise RuntimeError("Legacy extraction output directory was not created")
