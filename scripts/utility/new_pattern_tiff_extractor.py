#!/usr/bin/env python3
"""
New Pattern TIFF Color Extractor

Extracts vertically stacked color columns from large TIFF files using a
parameter-driven region config (for example, regions_json/new_pattern_2400.json).

Bounding boxes are derived from color count, column width, head count, and
offsets rather than hard-coded coordinates.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from tiff_extractor import extract_region_from_tiff

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _get_config_value(config: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    """Read a config value supporting snake_case and camelCase keys."""
    for key in keys:
        if key in config:
            return config[key]
    return default


def derive_color_regions(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Derive one bounding box per configured color from pattern parameters.

    Horizontal layout:
      - island_front=true:  [leading gap/island][color][gap][color]...
      - island_front=false: [color][gap][color][gap]...

    The first color always starts at x_offset. When island_width > 0, it is
    treated as the horizontal gap between adjacent color columns.
    """
    colors = config.get("colors", [])
    if not colors:
        raise ValueError("Config must include a non-empty 'colors' list")

    color_width = int(_get_config_value(config, "color_width", "colorWidth"))
    x_offset = int(_get_config_value(config, "x_offset", "xOffset"))
    num_heads = int(_get_config_value(config, "num_heads", "numHeads", default=3))
    head_height = int(_get_config_value(config, "head_height", "headHeight"))
    y_offset = int(_get_config_value(config, "y_offset", "yOffset"))
    island_front = bool(_get_config_value(config, "island_front", "islandFront", default=True))
    island_width = int(_get_config_value(config, "island_width", "islandWidth", default=0))

    buffer = _get_config_value(config, "buffer", default={}) or {}
    horizontal_buffer = int(buffer.get("horizontal", 0))
    vertical_buffer = int(buffer.get("vertical", 0))

    total_height = num_heads * head_height + y_offset
    regions: List[Dict[str, Any]] = []

    for color_index, color_name in enumerate(colors):
        if island_front:
            # Layout: [leading gap][color][island][color][island]...
            x_start = x_offset + color_index * (color_width + island_width)
        else:
            # Layout: [color][island][color][island]...
            if color_index == 0:
                x_start = x_offset
            else:
                x_start = x_offset + color_index * color_width + (color_index - 1) * island_width

        x_end = x_start + color_width
        y_start = y_offset
        y_end = total_height

        regions.append(
            {
                "name": color_name,
                "color_index": color_index,
                "bounding_box_pixels": {
                    "top_x": x_start - horizontal_buffer,
                    "top_y": y_start - vertical_buffer,
                    "bottom_x": x_end + horizontal_buffer,
                    "bottom_y": y_end + vertical_buffer,
                },
            }
        )

    return regions


def clip_bbox_to_image(
    bbox: Dict[str, int],
    image_width: int,
    image_height: int,
    region_name: str = "",
) -> Optional[Dict[str, int]]:
    """
    Clip a bounding box to valid image bounds.

    When the requested region extends beyond the image, it is trimmed to the
    nearest edge and a warning is logged instead of raising an error.
    Returns None when the region has no overlap with the image.
    """
    requested = {
        "top_x": int(bbox["top_x"]),
        "top_y": int(bbox["top_y"]),
        "bottom_x": int(bbox["bottom_x"]),
        "bottom_y": int(bbox["bottom_y"]),
    }

    x1 = max(0, requested["top_x"])
    y1 = max(0, requested["top_y"])
    x2 = min(image_width, requested["bottom_x"])
    y2 = min(image_height, requested["bottom_y"])

    if x2 <= x1 or y2 <= y1:
        logger.warning(
            "Region '%s' lies outside the image bounds (%dx%d); skipping",
            region_name or "unnamed",
            image_width,
            image_height,
        )
        return None

    clipped = {
        "top_x": x1,
        "top_y": y1,
        "bottom_x": x2,
        "bottom_y": y2,
    }

    if clipped != requested:
        logger.warning(
            "Region '%s' exceeds image size (%dx%d); clipping from "
            "(%d, %d)-(%d, %d) to (%d, %d)-(%d, %d)",
            region_name or "unnamed",
            image_width,
            image_height,
            requested["top_x"],
            requested["top_y"],
            requested["bottom_x"],
            requested["bottom_y"],
            x1,
            y1,
            x2,
            y2,
        )

    return clipped


def get_image_size(tiff_path: str) -> Tuple[int, int]:
    """Return (width, height) for the first page of a TIFF file."""
    import tifffile

    with tifffile.TiffFile(tiff_path) as tif:
        shape = tif.pages[0].shape

    if len(shape) == 2:
        height, width = shape
    elif len(shape) == 3:
        height, width = shape[0], shape[1]
    else:
        raise ValueError(f"Unsupported TIFF shape: {shape}")

    return width, height


def load_config(config_path: str) -> Dict[str, Any]:
    """Load pattern config JSON from disk."""
    with open(config_path, "r", encoding="utf-8") as config_file:
        return json.load(config_file)


def extract_colors_from_tiff(
    tiff_path: str,
    config: Dict[str, Any],
    output_dir: Path,
) -> Dict[str, bool]:
    """
    Extract one TIFF per configured color and return success by color name.
    """
    tiff_path = str(Path(tiff_path).resolve())
    if not Path(tiff_path).is_file():
        raise FileNotFoundError(f"Input TIFF not found: {tiff_path}")

    output_dir = output_dir / "extracted"
    output_dir.mkdir(parents=True, exist_ok=True)

    regions = derive_color_regions(config)
    image_width, image_height = get_image_size(tiff_path)

    logger.info("Input image: %s (%dx%d)", tiff_path, image_width, image_height)
    logger.info("Extracting %d color region(s) to %s", len(regions), output_dir)

    results: Dict[str, bool] = {}
    for region in tqdm(regions, desc="Extracting colors", unit="color"):
        name = region["name"]
        clipped_bbox = clip_bbox_to_image(
            region["bounding_box_pixels"],
            image_width,
            image_height,
            region_name=name,
        )
        if clipped_bbox is None:
            results[name] = False
            continue

        output_path = output_dir / f"{name}.tiff"

        logger.info(
            "Color '%s': (%d, %d) to (%d, %d)",
            name,
            clipped_bbox["top_x"],
            clipped_bbox["top_y"],
            clipped_bbox["bottom_x"],
            clipped_bbox["bottom_y"],
        )

        success = extract_region_from_tiff(
            tiff_path,
            clipped_bbox["top_x"],
            clipped_bbox["top_y"],
            clipped_bbox["bottom_x"],
            clipped_bbox["bottom_y"],
            str(output_path),
        )
        results[name] = success

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Extract color columns from a large TIFF using a parameter-driven "
            "new-pattern config JSON"
        )
    )
    parser.add_argument("input_tiff", help="Path to the input TIFF image")
    parser.add_argument("config", help="Path to the new-pattern config JSON file")
    parser.add_argument(
        "-o",
        "--output-dir",
        help=(
            "Base output directory. Extracted TIFF files are saved in an "
            "'extracted' subfolder here. Defaults to the input TIFF directory."
        ),
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    input_path = Path(args.input_tiff).resolve()
    config = load_config(args.config)

    if args.output_dir:
        output_dir = Path(args.output_dir).resolve()
    else:
        output_dir = input_path.parent

    try:
        results = extract_colors_from_tiff(str(input_path), config, output_dir)
    except Exception as exc:
        logger.error("%s", exc)
        sys.exit(1)

    print("\nExtraction Results:")
    print("-" * 50)
    for name, success in results.items():
        status = "Success" if success else "Failed"
        print(f"{name}: {status}")

    if all(results.values()):
        sys.exit(0)
    sys.exit(1)


if __name__ == "__main__":
    main()
