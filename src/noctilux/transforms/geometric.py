from __future__ import annotations

from PIL import Image, ImageOps

from noctilux.registry import register_transform
from noctilux.transforms.base import BaseTransform


@register_transform("horizontal_flip")
class HorizontalFlipTransform(BaseTransform):
    name = "horizontal_flip"

    def __call__(self, image: Image.Image, context: dict | None = None) -> Image.Image:
        return ImageOps.mirror(image.copy())


@register_transform("vertical_flip")
class VerticalFlipTransform(BaseTransform):
    name = "vertical_flip"

    def __call__(self, image: Image.Image, context: dict | None = None) -> Image.Image:
        return ImageOps.flip(image.copy())


@register_transform("rotate")
class RotateTransform(BaseTransform):
    name = "rotate"

    def validate_params(self) -> None:
        self.params.setdefault("expand", False)
        self.params.setdefault("fill_color", 0)
        angle = self.params.get("angle")
        if not isinstance(angle, (int, float)):
            raise ValueError("rotate angle must be a number.")
        if not isinstance(self.params["expand"], bool):
            raise ValueError("rotate expand must be a boolean.")
        fill_color = self.params["fill_color"]
        if isinstance(fill_color, (list, tuple)):
            if len(fill_color) != 3 or not all(isinstance(value, int) for value in fill_color):
                raise ValueError("rotate fill_color must be an int or a 3-item RGB tuple/list of ints.")
            self.params["fill_color"] = tuple(fill_color)
        elif not isinstance(fill_color, int):
            raise ValueError("rotate fill_color must be an int or a 3-item RGB tuple/list of ints.")

    def __call__(self, image: Image.Image, context: dict | None = None) -> Image.Image:
        if self.backend == "opencv":
            return self._call_opencv(image)
        return self._call_pillow(image)

    def _call_pillow(self, image: Image.Image) -> Image.Image:
        source = image.copy().convert("RGB")
        return source.rotate(
            angle=float(self.params["angle"]),
            expand=self.params["expand"],
            resample=Image.Resampling.BICUBIC,
            fillcolor=self.params["fill_color"],
        )

    def _call_opencv(self, image: Image.Image) -> Image.Image:
        from noctilux.backends.opencv_backend import cv2_to_pil, pil_to_cv2, require_opencv

        require_opencv()
        import cv2

        if self.params["expand"]:
            raise NotImplementedError(
                "rotate with expand=True is not supported by the OpenCV backend. "
                "Use backend: pillow instead."
            )

        arr = pil_to_cv2(image)
        h, w = arr.shape[:2]
        center = (w / 2.0, h / 2.0)
        matrix = cv2.getRotationMatrix2D(center, float(self.params["angle"]), 1.0)
        fill_color = self.params["fill_color"]
        if isinstance(fill_color, (list, tuple)):
            bgr_fill = (fill_color[2], fill_color[1], fill_color[0])
        else:
            bgr_fill = (fill_color, fill_color, fill_color)
        rotated = cv2.warpAffine(arr, matrix, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=bgr_fill)
        return cv2_to_pil(rotated)
