"""Annotation schema and parser prototypes."""

from noctilux.annotations.parsers import BaseAnnotationParser, CocoAnnotationParser, YoloAnnotationParser
from noctilux.annotations.schema import AnnotationRecord, BoundingBox, Keypoint, MaskRef

__all__ = [
    "AnnotationRecord",
    "BaseAnnotationParser",
    "BoundingBox",
    "CocoAnnotationParser",
    "Keypoint",
    "MaskRef",
    "YoloAnnotationParser",
]
