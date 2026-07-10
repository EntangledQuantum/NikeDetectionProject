#!/usr/bin/env python3
"""
New Pattern TIFF Color Extractor

Extracts vertically stacked color columns from large TIFF files using a
parameter-driven region config (for example, regions_json/new_pattern_2400.json).

Bounding boxes are derived from color count, column width, head count, and
offsets rather than hard-coded coordinates.

Optionally, each extracted color TIFF can be further split into Stripe and
Island regions via --split-stripe-island.
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


def derive_stripe_island_regions(
    config: Dict[str, Any],
    color_image_width: int,
    color_image_height: int,
) -> List[Dict[str, Any]]:
    """
    Derive Stripe and Island bounding boxes within an extracted color TIFF.

    Coordinates are relative to the color TIFF. The color content is assumed to
    sit between the color horizontal buffers applied during color extraction:

      [horizontal_buffer][color_width content][horizontal_buffer]

    Within the color content:
      - island_front=true:  [island][stripe]
      - island_front=false: [stripe][island]

    A stripe_island buffer expands each region horizontally.
    """
    color_width = int(_get_config_value(config, "color_width", "colorWidth"))
    stripe_width = int(_get_config_value(config, "stripe_width", "stripeWidth", default=1030))
    island_front = bool(_get_config_value(config, "island_front", "islandFront", default=True))

    buffer = _get_config_value(config, "buffer", default={}) or {}
    color_h_buffer = int(buffer.get("horizontal", 0))
    split_buffer = int(
        buffer.get(
            "stripe_island",
            _get_config_value(config, "stripe_island_buffer", "stripeIslandBuffer", default=50),
        )
    )

    if stripe_width <= 0:
        raise ValueError("stripe_width must be a positive integer")
    if stripe_width >= color_width:
        raise ValueError(
            f"stripe_width ({stripe_width}) must be less than color_width ({color_width})"
        )

    island_width = color_width - stripe_width
    content_x0 = color_h_buffer

    if island_front:
        island_x0 = content_x0
        island_x1 = content_x0 + island_width
        stripe_x0 = content_x0 + island_width
        stripe_x1 = content_x0 + color_width
    else:
        stripe_x0 = content_x0
        stripe_x1 = content_x0 + stripe_width
        island_x0 = content_x0 + stripe_width
        island_x1 = content_x0 + color_width

    regions = [
        {
            "suffix": "Stripe",
            "bounding_box_pixels": {
                "top_x": stripe_x0 - split_buffer,
                "top_y": 0,
                "bottom_x": stripe_x1 + split_buffer,
                "bottom_y": color_image_height,
            },
        },
        {
            "suffix": "Island",
            "bounding_box_pixels": {
                "top_x": island_x0 - split_buffer,
                "top_y": 0,
                "bottom_x": island_x1 + split_buffer,
                "bottom_y": color_image_height,
            },
        },
    ]

    # Keep derived sizes available for logging
    for region in regions:
        region["image_width"] = color_image_width
        region["image_height"] = color_image_height

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


def _extract_named_region(
    source_tiff: str,
    name: str,
    bbox: Dict[str, int],
    image_width: int,
    image_height: int,
    output_path: Path,
) -> bool:
    """Clip bbox to image bounds and extract; returns True on success."""
    clipped_bbox = clip_bbox_to_image(
        bbox,
        image_width,
        image_height,
        region_name=name,
    )
    if clipped_bbox is None:
        return False

    logger.info(
        "Region '%s': (%d, %d) to (%d, %d)",
        name,
        clipped_bbox["top_x"],
        clipped_bbox["top_y"],
        clipped_bbox["bottom_x"],
        clipped_bbox["bottom_y"],
    )

    return extract_region_from_tiff(
        source_tiff,
        clipped_bbox["top_x"],
        clipped_bbox["top_y"],
        clipped_bbox["bottom_x"],
        clipped_bbox["bottom_y"],
        str(output_path),
    )


def split_color_into_stripe_island(
    color_tiff_path: Path,
    color_name: str,
    config: Dict[str, Any],
    output_dir: Path,
) -> Dict[str, bool]:
    """
    Split one extracted color TIFF into Stripe and Island images.

    Output names follow ColorStripe.tiff / ColorIsland.tiff.
    """
    if not color_tiff_path.is_file():
        logger.error("Color TIFF not found for split: %s", color_tiff_path)
        return {f"{color_name}Stripe": False, f"{color_name}Island": False}

    image_width, image_height = get_image_size(str(color_tiff_path))
    sub_regions = derive_stripe_island_regions(config, image_width, image_height)

    results: Dict[str, bool] = {}
    for sub_region in sub_regions:
        name = f"{color_name}{sub_region['suffix']}"
        output_path = output_dir / f"{name}.tiff"
        success = _extract_named_region(
            str(color_tiff_path),
            name,
            sub_region["bounding_box_pixels"],
            image_width,
            image_height,
            output_path,
        )
        results[name] = success

    return results


def extract_colors_from_tiff(
    tiff_path: str,
    config: Dict[str, Any],
    output_dir: Path,
    split_stripe_island: bool = False,
) -> Dict[str, bool]:
    """
    Extract one TIFF per configured color and return success by color name.

    When split_stripe_island is True, each successful color TIFF is further
    split into Stripe and Island images in the same extracted folder.
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
        output_path = output_dir / f"{name}.tiff"

        success = _extract_named_region(
            tiff_path,
            name,
            region["bounding_box_pixels"],
            image_width,
            image_height,
            output_path,
        )
        results[name] = success

        if success and split_stripe_island:
            split_results = split_color_into_stripe_island(
                output_path,
                name,
                config,
                output_dir,
            )
            results.update(split_results)

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
        "--split-stripe-island",
        action="store_true",
        help=(
            "After extracting each color TIFF, further split it into Stripe "
            "and Island images (e.g. CyanStripe.tiff, CyanIsland.tiff)"
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
        results = extract_colors_from_tiff(
            str(input_path),
            config,
            output_dir,
            split_stripe_island=args.split_stripe_island,
        )
    except Exception as exc:
        logger.error("%s", exc)
        sys.exit(1)

    print("\nExtraction Results:")
    print("-" * 50)
    for name, success in results.items():
        status = "Success" if success else "Failed"
        print(f"{name}: {status}")

    if results and all(results.values()):
        sys.exit(0)
    sys.exit(1)


if __name__ == "__main__":
    main()
