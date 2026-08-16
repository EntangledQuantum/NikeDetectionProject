"""Detector registry: keys, required layers, pattern-specific implementations."""

from __future__ import annotations

from typing import Dict, List, Type

from nike_detection.config.schema import RunSettings
from nike_detection.detectors.adapters import (
    DebrisStripeAdapter,
    LegacyDebrisAdapter,
    LegacyLineDefectAdapter,
    LegacyOversprayIslandAdapter,
    MisalignmentAdapter,
    NewDebrisAdapter,
    NewLineDefectAdapter,
    NewOversprayIslandAdapter,
    OversprayAdapter,
    RoughnessAdapter,
    SurfaceTreatmentAdapter,
    VoidAdapter,
    _BaseAdapter,
)
from nike_detection.pipeline.types import ImageType

ALL_DETECTOR_KEYS = [
    "stripe_misalignment",
    "overspray",
    "surface_treatment",
    "void",
    "debris_stripe",
    "edge_roughness",
    "debris_island",
    "overspray_island",
    "line_defect",
]

STRIPE_KEYS = {
    "stripe_misalignment",
    "overspray",
    "surface_treatment",
    "void",
    "debris_stripe",
    "edge_roughness",
}
ISLAND_KEYS = {"debris_island", "overspray_island", "line_defect"}

_STRIPE: Dict[str, Type[_BaseAdapter]] = {
    "stripe_misalignment": MisalignmentAdapter,
    "edge_roughness": RoughnessAdapter,
    "void": VoidAdapter,
    "debris_stripe": DebrisStripeAdapter,
    "overspray": OversprayAdapter,
    "surface_treatment": SurfaceTreatmentAdapter,
}

_ISLAND_LEGACY: Dict[str, Type[_BaseAdapter]] = {
    "debris_island": LegacyDebrisAdapter,
    "overspray_island": LegacyOversprayIslandAdapter,
    "line_defect": LegacyLineDefectAdapter,
}

_ISLAND_NEW: Dict[str, Type[_BaseAdapter]] = {
    "debris_island": NewDebrisAdapter,
    "overspray_island": NewOversprayIslandAdapter,
    "line_defect": NewLineDefectAdapter,
}


def keys_for_type(image_type: ImageType, settings: RunSettings) -> List[str]:
    cfg = settings.config.detector_sets
    if image_type == ImageType.STRIPE:
        return list(cfg.stripe)
    if image_type == ImageType.ISLAND:
        return list(cfg.island)
    if image_type == ImageType.UNKNOWN:
        return list(cfg.unknown)
    if image_type == ImageType.FULL:
        return list(cfg.stripe) + list(cfg.island)
    return []


def filter_keys(
    image_type: ImageType,
    requested: List[str],
    only: List[str] | None,
) -> List[str]:
    if image_type == ImageType.STRIPE:
        valid = STRIPE_KEYS
    elif image_type == ImageType.ISLAND:
        valid = ISLAND_KEYS
    elif image_type == ImageType.UNKNOWN:
        valid = {"surface_treatment"}
    else:
        valid = STRIPE_KEYS | ISLAND_KEYS
    if only:
        unknown = set(only) - set(ALL_DETECTOR_KEYS)
        if unknown:
            raise ValueError(f"Unknown detector keys: {sorted(unknown)}")
        return [key for key in only if key in valid]
    return [key for key in requested if key in valid]


def create_detectors(
    image_type: ImageType,
    keys: List[str],
    settings: RunSettings,
) -> Dict[str, _BaseAdapter]:
    if image_type == ImageType.ISLAND and settings.pattern == "new":
        table = _ISLAND_NEW
    elif image_type == ImageType.ISLAND:
        table = _ISLAND_LEGACY
    else:
        table = _STRIPE
    detectors: Dict[str, _BaseAdapter] = {}
    for key in keys:
        cls = table.get(key) or _STRIPE.get(key)
        if cls is None:
            continue
        detectors[key] = cls(settings)
    return detectors
