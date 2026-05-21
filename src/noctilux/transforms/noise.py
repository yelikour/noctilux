from __future__ import annotations

import numpy as np
from PIL import Image

from noctilux.registry import register_transform
from noctilux.transforms.base import BaseTransform


@register_transform("gaussian_noise")
class GaussianNoiseTransform(BaseTransform):
    name = "gaussian_noise"

    def validate_params(self) -> None:
        std = self.params.get("std")
        if not isinstance(std, (int, float)) or float(std) < 0:
            raise ValueError("gaussian_noise std must be a number >= 0.")

    def __call__(self, image: Image.Image, context: dict | None = None) -> Image.Image:
        source = np.asarray(image.copy().convert("RGB"), dtype=np.float32)
        generator = (context or {}).get("np_rng")
        if generator is None:
            generator = np.random.default_rng()
        noise = generator.normal(loc=0.0, scale=float(self.params["std"]), size=source.shape)
        output = np.clip(source + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(output, mode="RGB")
