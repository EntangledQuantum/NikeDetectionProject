"""Detector protocol: score defects from a shared ImageContext, no I/O."""

from __future__ import annotations

from typing import List, Protocol, runtime_checkable

from nike_detection.pipeline.types import Defect


@runtime_checkable
class Detector(Protocol):
    key: str

    def required_layers(self) -> frozenset[str]:
        ...

    def detect(self, ctx) -> List[Defect]:
        ...

    def render(self, ctx, defects: List[Defect]):
        ...
