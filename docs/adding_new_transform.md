# Adding a New Transform

下面示例新增一个 `invert_rgb` transform。

## 1. 新建模块

文件：`src/noctilux/transforms/custom_example.py`

```python
from PIL import ImageOps

from noctilux.registry import register_transform
from noctilux.transforms.base import BaseTransform


@register_transform("invert_rgb")
class InvertRGBTransform(BaseTransform):
    name = "invert_rgb"

    def validate_params(self) -> None:
        if self.backend != "pillow":
            raise ValueError("invert_rgb only supports the pillow backend in v0.1.0.")

    def __call__(self, image, context=None):
        return ImageOps.invert(image.copy().convert("RGB"))
```

## 2. 在 transforms 包中导入

更新 `src/noctilux/transforms/__init__.py`：

```python
from noctilux.transforms import custom_example
```

## 3. 在 YAML 中使用

```yaml
pipelines:
  - name: invert_preview
    transforms:
      - name: invert_rgb
        params: {}
```

## 4. 为新 transform 写测试

```python
from PIL import Image

from noctilux.registry import build_transform


def test_invert_rgb_returns_pil_image():
    image = Image.new("RGB", (32, 32), color=(10, 20, 30))
    transform = build_transform("invert_rgb", params={})
    output = transform(image)
    assert output.mode == "RGB"
    assert output.size == image.size
```

新增 transform 时不要在 `pipeline.py` 中写 `if/elif` 分发。统一走 registry。

在 v0.2.0 中，新增 transform 仍然需要遵守同样规则：

- 输入输出统一为 `PIL.Image.Image`
- 参数校验放在 `validate_params()`
- 随机行为从 `context["rng"]` 或 `context["np_rng"]` 读取，以支持 pipeline seed 复现
- 注册后可直接被 `noctilux list-transforms` 和 YAML pipeline 使用
