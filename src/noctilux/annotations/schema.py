from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class BoundingBox:
    """COCO-style pixel bbox: x, y, width, height."""

    x: float
    y: float
    width: float
    height: float
    category_id: int
    annotation_id: int | str | None = None
    iscrowd: int | None = None
    area: float | None = None

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Bounding box width and height must be positive.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
            "category_id": self.category_id,
            "annotation_id": self.annotation_id,
            "iscrowd": self.iscrowd,
            "area": self.area,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> BoundingBox:
        return cls(
            x=payload["x"],
            y=payload["y"],
            width=payload["width"],
            height=payload["height"],
            category_id=payload["category_id"],
            annotation_id=payload.get("annotation_id"),
            iscrowd=payload.get("iscrowd"),
            area=payload.get("area"),
        )


@dataclass
class Keypoint:
    x: float
    y: float
    visible: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "x": self.x,
            "y": self.y,
            "visible": self.visible,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Keypoint:
        return cls(
            x=payload["x"],
            y=payload["y"],
            visible=payload["visible"],
        )


@dataclass
class MaskRef:
    path: str | None = None
    segmentation: Any | None = None
    size: list[int] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "segmentation": self.segmentation,
            "size": self.size,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> MaskRef:
        return cls(
            path=payload.get("path"),
            segmentation=payload.get("segmentation"),
            size=payload.get("size"),
        )


@dataclass
class AnnotationRecord:
    image_id: int | str
    image_path: str | None = None
    width: int | None = None
    height: int | None = None
    boxes: list[BoundingBox] = field(default_factory=list)
    masks: list[MaskRef] = field(default_factory=list)
    keypoints: list[Keypoint] = field(default_factory=list)
    raw: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "image_id": self.image_id,
            "image_path": self.image_path,
            "width": self.width,
            "height": self.height,
            "boxes": [box.to_dict() for box in self.boxes],
            "masks": [mask.to_dict() for mask in self.masks],
            "keypoints": [keypoint.to_dict() for keypoint in self.keypoints],
            "raw": self.raw,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> AnnotationRecord:
        return cls(
            image_id=payload["image_id"],
            image_path=payload.get("image_path"),
            width=payload.get("width"),
            height=payload.get("height"),
            boxes=[BoundingBox.from_dict(box) for box in payload.get("boxes", [])],
            masks=[MaskRef.from_dict(mask) for mask in payload.get("masks", [])],
            keypoints=[Keypoint.from_dict(keypoint) for keypoint in payload.get("keypoints", [])],
            raw=payload.get("raw"),
        )
