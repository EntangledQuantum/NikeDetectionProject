"""Load and validate detection_2400.json."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from digital_air_cv.config.schema import (
    PATTERN_VALUES,
    REQUIRED_DETECTOR_PARAM_KEYS,
    SENSITIVITY_LEVELS,
    AppConfig,
    BoundingBox,
    Defaults,
    DetectorSets,
    GeometryConfig,
    IdealReference,
    MaterialConfig,
    RegionReference,
    RegionSpec,
    VerticalBandConfig,
)

logger = logging.getLogger(__name__)

_PACKAGE_CONFIG = Path(__file__).resolve().parent / "detection_2400.json"
_PROJECT_CONFIG = Path(__file__).resolve().parents[2] / "config" / "detection_2400.json"
DEFAULT_CONFIG_PATH = _PROJECT_CONFIG if _PROJECT_CONFIG.exists() else _PACKAGE_CONFIG


class ConfigError(ValueError):
    """Raised when the operator config is missing or invalid."""


# Catalog order is what `keys_for_type` emits when a detector is enabled.
_DETECTOR_CATALOG = {
    "stripe": [
        "stripe_misalignment",
        "edge_roughness",
        "void",
        "debris_stripe",
        "overspray",
        "surface_treatment",
    ],
    "island": [
        "line_defect",
        "debris_island",
        "overspray_island",
    ],
    "unknown": [
        "surface_treatment",
    ],
}


def _is_enabled_flag(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _enabled_detector_keys(raw: Any, kind: str) -> List[str]:
    """Accept `{key: true/false}` (preferred) or a list of enabled keys."""
    catalog = _DETECTOR_CATALOG.get(kind, [])
    if raw is None:
        raise ConfigError(f"detector_sets.{kind} is required")
    if isinstance(raw, dict):
        enabled: List[str] = []
        seen = set()
        for key in catalog:
            if key in raw and _is_enabled_flag(raw[key]):
                enabled.append(key)
                seen.add(key)
        for key, value in raw.items():
            if key.startswith("_") or key in seen:
                continue
            if _is_enabled_flag(value):
                if key not in catalog:
                    logger.warning(
                        "Unknown detector key %r in detector_sets.%s", key, kind
                    )
                enabled.append(str(key))
                seen.add(key)
        return enabled
    if isinstance(raw, list):
        return [str(key) for key in raw if not str(key).startswith("_")]
    raise ConfigError(
        f"detector_sets.{kind} must be an object of true/false flags "
        "or a list of detector keys"
    )


def _require(data: Dict[str, Any], key: str, ctx: str = "") -> Any:
    if key not in data:
        where = f" in {ctx}" if ctx else ""
        raise ConfigError(f"Missing required config key '{key}'{where}")
    return data[key]


def _bbox_from_dict(raw: Dict[str, Any]) -> BoundingBox:
    return BoundingBox(
        top_x=float(_require(raw, "top_x", "bounding_box_pixels")),
        top_y=float(_require(raw, "top_y", "bounding_box_pixels")),
        bottom_x=float(_require(raw, "bottom_x", "bounding_box_pixels")),
        bottom_y=float(_require(raw, "bottom_y", "bounding_box_pixels")),
    )


def _parse_regions(raw_regions: Any) -> List[RegionSpec]:
    if not raw_regions:
        return []
    if not isinstance(raw_regions, list):
        raise ConfigError("'regions' must be a list")
    regions: List[RegionSpec] = []
    for i, item in enumerate(raw_regions):
        ctx = f"regions[{i}]"
        bbox = _bbox_from_dict(_require(item, "bounding_box_pixels", ctx))
        regions.append(
            RegionSpec(
                name=str(_require(item, "name", ctx)),
                type=str(_require(item, "type", ctx)).lower(),
                bounding_box_pixels=bbox,
                corners=item.get("corners"),
            )
        )
    return regions


def _validate_sensitivity(table: Dict[str, Any]) -> Dict[str, Dict[str, Dict[str, Any]]]:
    if not isinstance(table, dict):
        raise ConfigError("'sensitivity' must be an object keyed by low/medium/high")
    missing_levels = [level for level in SENSITIVITY_LEVELS if level not in table]
    if missing_levels:
        raise ConfigError(f"sensitivity is missing levels: {missing_levels}")
    for level in SENSITIVITY_LEVELS:
        params = table[level]
        if not isinstance(params, dict):
            raise ConfigError(f"sensitivity.{level} must be an object")
        missing = [key for key in REQUIRED_DETECTOR_PARAM_KEYS if key not in params]
        if missing:
            raise ConfigError(
                f"sensitivity.{level} is missing detector keys: {missing}"
            )
    return table  # type: ignore[return-value]


def load_config(path: Optional[str] = None) -> AppConfig:
    """Load the unified 2400-DPI config. Fails loudly on missing keys."""
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not config_path.exists():
        raise ConfigError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    dpi = int(_require(data, "dpi"))
    if dpi != 2400:
        raise ConfigError(f"This pipeline is frozen at 2400 DPI; got dpi={dpi}")

    raw_defaults = _require(data, "defaults")
    defaults = Defaults(
        pattern=str(raw_defaults.get("pattern", "new")),
        sensitivity=str(raw_defaults.get("sensitivity", "medium")),
        clear=bool(raw_defaults.get("clear", False)),
        max_image_workers=int(raw_defaults.get("max_image_workers", 2)),
        max_detector_threads=int(raw_defaults.get("max_detector_threads", 4)),
        write_visualizations=bool(raw_defaults.get("write_visualizations", True)),
        downscale_vis=bool(raw_defaults.get("downscale_vis", False)),
        write_crops=bool(raw_defaults.get("write_crops", False)),
        write_region_folders=bool(raw_defaults.get("write_region_folders", False)),
        write_full_defect_overlay=bool(
            raw_defaults.get("write_full_defect_overlay", True)
        ),
    )
    if defaults.pattern not in PATTERN_VALUES:
        raise ConfigError(f"defaults.pattern must be one of {PATTERN_VALUES}")
    if defaults.sensitivity not in SENSITIVITY_LEVELS:
        raise ConfigError(f"defaults.sensitivity must be one of {SENSITIVITY_LEVELS}")

    raw_sets = _require(data, "detector_sets")
    detector_sets = DetectorSets(
        stripe=_enabled_detector_keys(raw_sets.get("stripe"), "stripe"),
        island=_enabled_detector_keys(raw_sets.get("island"), "island"),
        unknown=_enabled_detector_keys(
            raw_sets.get("unknown", ["surface_treatment"]), "unknown"
        ),
    )

    raw_geo = _require(data, "geometry")
    geometry = GeometryConfig(
        colors=list(_require(raw_geo, "colors", "geometry")),
        color_width=int(_require(raw_geo, "color_width", "geometry")),
        x_offset=int(_require(raw_geo, "x_offset", "geometry")),
        num_heads=int(_require(raw_geo, "num_heads", "geometry")),
        head_height=int(_require(raw_geo, "head_height", "geometry")),
        y_offset=int(_require(raw_geo, "y_offset", "geometry")),
        island_front=bool(_require(raw_geo, "island_front", "geometry")),
        island_width=int(_require(raw_geo, "island_width", "geometry")),
        stripe_width=int(_require(raw_geo, "stripe_width", "geometry")),
        buffer=dict(_require(raw_geo, "buffer", "geometry")),
    )

    raw_region_ref = data.get("region_reference") or {}
    defaults_region = RegionReference()
    region_reference = RegionReference(
        island_width=int(raw_region_ref.get("island_width", defaults_region.island_width)),
        island_stripe_gap=int(
            raw_region_ref.get("island_stripe_gap", defaults_region.island_stripe_gap)
        ),
        stripe_width=int(raw_region_ref.get("stripe_width", defaults_region.stripe_width)),
        height=int(raw_region_ref.get("height", defaults_region.height)),
        color_gap=int(raw_region_ref.get("color_gap", defaults_region.color_gap)),
        color_y_tolerance=int(
            raw_region_ref.get("color_y_tolerance", defaults_region.color_y_tolerance)
        ),
        tolerance=float(raw_region_ref.get("tolerance", defaults_region.tolerance)),
    )
    for field_name in ("island_width", "island_stripe_gap", "stripe_width", "height",
                       "color_gap"):
        if getattr(region_reference, field_name) <= 0:
            raise ConfigError(f"region_reference.{field_name} must be positive")

    raw_ideal = _require(data, "ideal_reference")
    ideal_reference = IdealReference(
        width=int(_require(raw_ideal, "width", "ideal_reference")),
        height=int(_require(raw_ideal, "height", "ideal_reference")),
        y_delta_min=int(_require(raw_ideal, "y_delta_min", "ideal_reference")),
        y_delta_max=int(_require(raw_ideal, "y_delta_max", "ideal_reference")),
        line_spacing=float(_require(raw_ideal, "line_spacing", "ideal_reference")),
        slope_min=float(_require(raw_ideal, "slope_min", "ideal_reference")),
        slope_max=float(_require(raw_ideal, "slope_max", "ideal_reference")),
    )

    raw_material = _require(data, "material")
    material = MaterialConfig(
        clear_debris_offsets=dict(_require(raw_material, "clear_debris_offsets", "material")),
        clear_overspray_offsets=dict(
            _require(raw_material, "clear_overspray_offsets", "material")
        ),
        ink_clamps=dict(_require(raw_material, "ink_clamps", "material")),
        clear_ink_offsets=dict(_require(raw_material, "clear_ink_offsets", "material")),
        line_paint_extension=int(raw_material.get("line_paint_extension", 20)),
    )

    raw_vband = data.get("vertical_band") or {}
    vertical_band = VerticalBandConfig(
        binary_threshold=int(raw_vband.get("binary_threshold", 170)),
        content_binary_threshold=int(raw_vband.get("content_binary_threshold", 127)),
        min_coverage=float(raw_vband.get("min_coverage", 0.25)),
        inner_margin=int(raw_vband.get("inner_margin", 5)),
    )

    sensitivity = _validate_sensitivity(_require(data, "sensitivity"))
    regions = _parse_regions(data.get("regions", []))

    logger.info("Loaded config %s (dpi=%s, pattern default=%s)", config_path, dpi, defaults.pattern)
    return AppConfig(
        dpi=dpi,
        defaults=defaults,
        detector_sets=detector_sets,
        geometry=geometry,
        ideal_reference=ideal_reference,
        material=material,
        vertical_band=vertical_band,
        sensitivity=sensitivity,
        region_reference=region_reference,
        regions=regions,
        source_path=str(config_path),
    )


def load_regions_file(path: str) -> List[RegionSpec]:
    """Load a sidecar / --regions JSON with a top-level ``regions`` list."""
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, list):
        return _parse_regions(data)
    return _parse_regions(data.get("regions", []))
