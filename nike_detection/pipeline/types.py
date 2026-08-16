"""Shared pipeline types."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

BBox = Tuple[int, int, int, int]


class ImageType(Enum):
    STRIPE = "stripe"
    ISLAND = "island"
    FULL = "full"
    UNKNOWN = "unknown"


@dataclass
class Defect:
    """One scored finding. Visualization is handled separately."""

    detector: str
    kind: str
    bbox: Optional[BBox] = None
    bbox_full: Optional[BBox] = None
    metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = dict(self.metrics)
        payload.setdefault("type", self.kind)
        payload.setdefault("detector", self.detector)
        if self.bbox is not None:
            payload.setdefault("bbox", list(self.bbox))
        if self.bbox_full is not None:
            payload["bbox_full"] = list(self.bbox_full)
        return payload


@dataclass
class DetectionResult:
    detector: str
    defect_count: int
    defects: List[Dict[str, Any]]
    visualization_path: Optional[str] = None
    error: Optional[str] = None
    elapsed_seconds: float = 0.0
    context_seconds: float = 0.0


@dataclass
class ImageResult:
    image_name: str
    image_path: str
    image_type: ImageType
    processing_time: str
    file_size_mb: float
    detectors_used: List[str]
    detections: Dict[str, DetectionResult]
    error: Optional[str] = None
    elapsed_seconds: float = 0.0
    load_seconds: float = 0.0
    context_seconds: float = 0.0
    detect_seconds: float = 0.0
    write_seconds: float = 0.0
    origin_xy: Tuple[int, int] = (0, 0)
    parent_image: Optional[str] = None


def classify_image(path: str) -> ImageType:
    """Filename routing: ``full`` wins over stripe/island."""
    name = path.replace("\\", "/").split("/")[-1].lower()
    if "full" in name:
        return ImageType.FULL
    if "stripe" in name:
        return ImageType.STRIPE
    if "island" in name:
        return ImageType.ISLAND
    return ImageType.UNKNOWN


def bbox_from_metrics(metrics: Dict[str, Any]) -> Optional[BBox]:
    """Best-effort bbox extraction from legacy defect dicts."""
    if "bbox" in metrics and metrics["bbox"] is not None:
        box = metrics["bbox"]
        if len(box) == 4:
            x1, y1, x2, y2 = [int(v) for v in box]
            # Some detectors store x,y,w,h; others store x1,y1,x2,y2.
            if x2 >= x1 and y2 >= y1 and (x2 - x1) > 0 and (y2 - y1) > 0:
                # Could be either. Prefer xywh when x2/y2 look like sizes
                # (width/height typically smaller than a huge absolute x2).
                return (x1, y1, x2, y2)
            return (x1, y1, x1 + abs(x2), y1 + abs(y2))
    if all(k in metrics for k in ("x", "y")):
        x, y = int(metrics["x"]), int(metrics["y"])
        return (x, y, x + 1, y + 1)
    if "location" in metrics and metrics["location"] is not None:
        loc = metrics["location"]
        if len(loc) >= 2:
            x, y = int(loc[0]), int(loc[1])
            return (x, y, x + 1, y + 1)
    return None


def defects_from_legacy(detector: str, raw: List[Dict[str, Any]],
                        origin_xy: Tuple[int, int] = (0, 0)) -> List[Defect]:
    ox, oy = origin_xy
    out: List[Defect] = []
    for item in raw:
        kind = str(item.get("type", detector))
        box = bbox_from_metrics(item)
        box_full = None
        if box is not None and (ox or oy):
            box_full = (box[0] + ox, box[1] + oy, box[2] + ox, box[3] + oy)
        out.append(
            Defect(
                detector=detector,
                kind=kind,
                bbox=box,
                bbox_full=box_full,
                metrics=item,
            )
        )
    return out
