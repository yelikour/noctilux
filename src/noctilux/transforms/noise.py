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


@register_transform("poisson_noise")
class PoissonNoiseTransform(BaseTransform):
    name = "poisson_noise"

    def validate_params(self) -> None:
        self.params.setdefault("scale", 1.0)
        scale = self.params["scale"]
        if not isinstance(scale, (int, float)) or float(scale) <= 0:
            raise ValueError("poisson_noise scale must be a number > 0.")

    def __call__(self, image: Image.Image, context: dict | None = None) -> Image.Image:
        source = np.asarray(image.copy().convert("RGB"), dtype=np.float32)
        generator = (context or {}).get("np_rng")
        if generator is None:
            generator = np.random.default_rng()
        scale = float(self.params["scale"])
        noisy = generator.poisson(source * scale) / scale
        return Image.fromarray(np.clip(noisy, 0, 255).astype(np.uint8), mode="RGB")


@register_transform("salt_pepper_noise")
class SaltPepperNoiseTransform(BaseTransform):
    name = "salt_pepper_noise"

    def validate_params(self) -> None:
        amount = self.params.get("amount")
        salt_vs_pepper = self.params.get("salt_vs_pepper")
        if not isinstance(amount, (int, float)) or not 0 <= float(amount) <= 1:
            raise ValueError("salt_pepper_noise amount must be a number in the range [0, 1].")
        if not isinstance(salt_vs_pepper, (int, float)) or not 0 <= float(salt_vs_pepper) <= 1:
            raise ValueError("salt_pepper_noise salt_vs_pepper must be a number in the range [0, 1].")

    def __call__(self, image: Image.Image, context: dict | None = None) -> Image.Image:
        source = np.asarray(image.copy().convert("RGB"), dtype=np.uint8)
        generator = (context or {}).get("np_rng")
        if generator is None:
            generator = np.random.default_rng()
        output = source.copy()
        total_pixels = source.shape[0] * source.shape[1]
        amount = float(self.params["amount"])
        salt_ratio = float(self.params["salt_vs_pepper"])
        salt_count = int(round(total_pixels * amount * salt_ratio))
        pepper_count = int(round(total_pixels * amount * (1 - salt_ratio)))
        if salt_count:
            indices = generator.choice(total_pixels, size=salt_count, replace=False)
            y_coords, x_coords = np.unravel_index(indices, source.shape[:2])
            output[y_coords, x_coords] = 255
        if pepper_count:
            indices = generator.choice(total_pixels, size=pepper_count, replace=False)
            y_coords, x_coords = np.unravel_index(indices, source.shape[:2])
            output[y_coords, x_coords] = 0
        return Image.fromarray(output, mode="RGB")
