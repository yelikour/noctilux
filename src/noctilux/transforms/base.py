from __future__ import annotations

from typing import Any

from PIL import Image


class BaseTransform:
    name: str = "base_transform"

    def __init__(self, backend: str = "pillow", **params: Any) -> None:
        self.backend = backend
        self.params = params
        self.validate_params()

    def validate_params(self) -> None:
        """Validate transform parameters."""

    def __call__(self, image: Image.Image, context: dict[str, Any] | None = None) -> Image.Image:
        raise NotImplementedError

    def get_params(self) -> dict[str, Any]:
        return dict(self.params)
