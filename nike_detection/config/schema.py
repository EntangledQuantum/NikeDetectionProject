"""Typed configuration schema for the 2400-DPI detection pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


REQUIRED_DETECTOR_PARAM_KEYS = (
    "line_detector",
    "debris_island",
    "overspray_island",
    "line_defect_legacy",
    "line_defect_new",
    "stripe_misalignment",
    "edge_roughness",
    "void",
    "debris_stripe",
    "overspray",
    "surface_treatment",
)

SENSITIVITY_LEVELS = ("low", "medium", "high")
PATTERN_VALUES = ("legacy", "new")
REGION_TYPES = ("stripe", "island")


@dataclass
class BoundingBox:
    top_x: float
    top_y: float
    bottom_x: float
    bottom_y: float

    def sanitized(self) -> "BoundingBox":
        x1 = min(abs(self.top_x), abs(self.bottom_x))
        y1 = min(abs(self.top_y), abs(self.bottom_y))
        x2 = max(abs(self.top_x), abs(self.bottom_x))
        y2 = max(abs(self.top_y), abs(self.bottom_y))
        return BoundingBox(top_x=x1, top_y=y1, bottom_x=x2, bottom_y=y2)

    def as_int_xyxy(self) -> tuple[int, int, int, int]:
        box = self.sanitized()
        return (
            int(box.top_x),
            int(box.top_y),
            int(box.bottom_x),
            int(box.bottom_y),
        )


@dataclass
class RegionSpec:
    name: str
    type: str
    bounding_box_pixels: BoundingBox

    def __post_init__(self) -> None:
        if self.type not in REGION_TYPES:
            raise ValueError(
                f"Region '{self.name}' has invalid type '{self.type}'; "
                f"expected one of {REGION_TYPES}"
            )
        self.bounding_box_pixels = self.bounding_box_pixels.sanitized()


@dataclass
class Defaults:
    pattern: str = "new"
    sensitivity: str = "medium"
    clear: bool = False
    max_image_workers: int = 2
    max_detector_threads: int = 4
    write_visualizations: bool = True
    downscale_vis: bool = False
    write_crops: bool = False


@dataclass
class DetectorSets:
    stripe: List[str]
    island: List[str]
    unknown: List[str] = field(default_factory=lambda: ["surface_treatment"])


@dataclass
class GeometryConfig:
    colors: List[str]
    color_width: int
    x_offset: int
    num_heads: int
    head_height: int
    y_offset: int
    island_front: bool
    island_width: int
    stripe_width: int
    buffer: Dict[str, int]


@dataclass
class IdealReference:
    width: int
    height: int
    y_delta_min: int
    y_delta_max: int
    line_spacing: float
    slope_min: float
    slope_max: float


@dataclass
class MaterialConfig:
    clear_debris_offsets: Dict[str, int]
    clear_overspray_offsets: Dict[str, int]
    ink_clamps: Dict[str, int]
    clear_ink_offsets: Dict[str, int]
    line_paint_extension: int = 20


@dataclass
class VerticalBandConfig:
    binary_threshold: int = 170
    content_binary_threshold: int = 127
    min_coverage: float = 0.25
    inner_margin: int = 5


@dataclass
class AppConfig:
    """Fully validated operator config loaded from detection_2400.json."""

    dpi: int
    defaults: Defaults
    detector_sets: DetectorSets
    geometry: GeometryConfig
    ideal_reference: IdealReference
    material: MaterialConfig
    vertical_band: VerticalBandConfig
    sensitivity: Dict[str, Dict[str, Dict[str, Any]]]
    regions: List[RegionSpec] = field(default_factory=list)
    source_path: Optional[str] = None

    def params_for(self, detector_key: str, level: str) -> Dict[str, Any]:
        if level not in self.sensitivity:
            raise KeyError(f"Unknown sensitivity '{level}'")
        table = self.sensitivity[level]
        if detector_key not in table:
            raise KeyError(
                f"Sensitivity '{level}' is missing detector params for '{detector_key}'"
            )
        return dict(table[detector_key])


@dataclass
class RunSettings:
    """CLI-resolved runtime settings layered on AppConfig."""

    config: AppConfig
    sensitivity: str
    pattern: str
    clear: bool
    only_detectors: Optional[List[str]]
    output_dir: Optional[str]
    generate_report: bool
    write_visualizations: bool
    downscale_vis: bool
    debug: bool
    max_image_workers: int
    max_detector_threads: int
    write_crops: bool
    regions_path: Optional[str]
    recursive: bool
    include_unknown: bool
    extract_config_path: Optional[str] = None

    def detector_params(self, key: str) -> Dict[str, Any]:
        return self.config.params_for(key, self.sensitivity)
