# Noctilux

Noctilux 是一个通用的离线图像批处理与增强工具。它面向训练前的数据准备阶段，使用 YAML 配置定义可复现、可追溯、可扩展的图像处理流水线，并将输出图片与 metadata 一起落盘。

当前稳定版本为 `v0.2.1`。

## 安装

```bash
pip install -e .
```

离线或受限网络环境可尝试：

```bash
pip install -e . --no-build-isolation
```

如果本机缺少构建基础组件，可先安装：

```bash
pip install setuptools wheel
```

## 快速开始

1. 准备图片目录或 manifest CSV。
2. 选择一个示例配置，例如 `configs/examples/basic_resize.yaml`。
3. 检查配置：

```bash
noctilux inspect-config --config configs/examples/basic_resize.yaml
```

4. 运行批处理：

```bash
noctilux run --config configs/examples/basic_resize.yaml
```

如果 `examples/images` 不存在，`run` 会给出清晰错误；测试使用临时目录图片，不依赖仓库内样例图。

## v0.1.x 已支持功能

- YAML 配置驱动批处理
- `folder` / `manifest` 两种输入扫描
- transform registry 动态构建
- pipeline 顺序执行
- `p`、`repeat`、`seed`、随机参数解析
- Pillow 读取与保存，默认 EXIF orientation + RGB
- 输出命名、冲突避让、默认不覆盖
- `manifest.csv`、`transform_log.jsonl`、`failed_images.csv`、`summary.csv`
- CLI：`--help`、`inspect-config`、`list-transforms`、`run`、`make-manifest`
- 基础 transforms：JPEG 压缩、长边缩放、中心裁剪、高斯模糊、高斯噪声、亮度对比度

## v0.2.0 新增 transforms

- Compression：`webp_compression`、`png_resave`、`double_jpeg_compression`
- Resize：`resize_exact`、`resize_short_edge`、`downscale_upscale`
- Crop：`random_crop_ratio`、`random_resized_crop`、`square_crop`
- Geometric：`horizontal_flip`、`vertical_flip`、`rotate`
- Blur：`median_blur`、`motion_blur`
- Noise：`poisson_noise`、`salt_pepper_noise`
- Color：`gamma_correction`、`saturation_hue`、`grayscale`、`sharpen`、`posterize`

## v0.2.1 preset 与预览增强

- 新增 `configs/presets/`：
  - `classification_light.yaml`
  - `compression_robustness.yaml`
  - `resize_crop_suite.yaml`
  - `visual_degradation_light.yaml`
  - `all_basic_v021.yaml`
- 新增单图预览工作流：`scripts/preview_transforms.py`
- `motion_blur` 改为轻量向量化实现，边界使用 `edge padding`

## 输入格式

- `folder`：扫描文件夹中的图片，可选从子目录推断 `label`。
- `manifest`：从 CSV 读取 `image_path`、`label`、`split`、`task` 等列。

详见 [docs/input_formats.md](docs/input_formats.md)。

## 输出格式

默认输出到：

```text
outputs/example_run/
├── images/
├── metadata/
├── logs/
└── previews/
```

metadata 至少包含：

- `manifest.csv`
- `transform_log.jsonl`
- `failed_images.csv`
- `summary.csv`

各文件说明：

- `manifest.csv`：每个输出样本一行，记录原图、输出图、pipeline、尺寸、格式、seed 和成功状态。
- `transform_log.jsonl`：每个输出样本一条 JSON，记录 transform 顺序、是否执行、实际采样参数和输入输出信息。
- `failed_images.csv`：记录失败样本的 `pipeline_name`、`repeat_index`、`seed`、`stage` 和错误信息。
- `summary.csv`：按 pipeline 聚合统计总数、成功数、失败数。

详见 [docs/output_formats.md](docs/output_formats.md)。

## YAML 配置示例

```yaml
project:
  name: noctilux_basic_resize
  seed: 42

input:
  mode: folder
  image_root: examples/images
  infer_label_from_subdir: true
  recursive: true

output:
  root: outputs/example_run
  save_format: jpg

runtime:
  dry_run: false
  num_workers: 1
  skip_broken_images: true

pipelines:
  - name: resize_512
    repeat: 1
    transforms:
      - name: resize_long_edge
        params:
          long_edge: 512
          interpolation: bicubic
```

## CLI 示例

```bash
noctilux --help
noctilux list-transforms
noctilux inspect-config --config configs/examples/basic_resize.yaml
noctilux inspect-config --config configs/examples/full_v020.yaml
noctilux make-manifest --image-root path/to/images --output manifest.csv --infer-label-from-subdir
noctilux run --config configs/examples/basic_resize.yaml
noctilux run --config configs/examples/full_v020.yaml --dry-run
```

更多配置示例：

- `configs/examples/compression_plus.yaml`
- `configs/examples/resize_plus.yaml`
- `configs/examples/crop_plus.yaml`
- `configs/examples/geometric_color.yaml`
- `configs/examples/full_v020.yaml`

Preset 配置：

- `configs/presets/classification_light.yaml`
- `configs/presets/compression_robustness.yaml`
- `configs/presets/resize_crop_suite.yaml`
- `configs/presets/visual_degradation_light.yaml`
- `configs/presets/all_basic_v021.yaml`

## 推荐工作流

1. 生成 manifest：

```bash
noctilux make-manifest --image-root path/to/images --output manifest.csv --infer-label-from-subdir
```

2. 检查配置：

```bash
noctilux inspect-config --config configs/presets/all_basic_v021.yaml
```

3. 预览配置效果：

```bash
python scripts/preview_transforms.py \
  --config configs/examples/full_v020.yaml \
  --image path/to/image.jpg \
  --output outputs/previews/preview_grid.jpg \
  --max-pipelines 8 \
  --seed 42
```

4. 先 dry-run：

```bash
noctilux run --config configs/presets/all_basic_v021.yaml --dry-run
```

5. 正式运行：

```bash
noctilux run --config configs/presets/all_basic_v021.yaml
```

6. 查看 metadata：

- `outputs/.../metadata/manifest.csv`
- `outputs/.../metadata/transform_log.jsonl`
- `outputs/.../metadata/failed_images.csv`
- `outputs/.../metadata/summary.csv`

## 第一个真实运行示例

```bash
mkdir -p /tmp/noctilux_demo/images/class_a
python - <<'PY'
from pathlib import Path
from PIL import Image
root = Path("/tmp/noctilux_demo/images/class_a")
root.mkdir(parents=True, exist_ok=True)
Image.new("RGB", (640, 480), color=(120, 80, 40)).save(root / "sample.jpg")
PY

cat > /tmp/noctilux_demo/config.yaml <<'YAML'
project:
  name: first_run
  seed: 42
input:
  mode: folder
  image_root: /tmp/noctilux_demo/images
  infer_label_from_subdir: true
output:
  root: /tmp/noctilux_demo/output
runtime:
  dry_run: false
  num_workers: 1
  skip_broken_images: true
pipelines:
  - name: resize_512
    transforms:
      - name: resize_long_edge
        params:
          long_edge: 512
          interpolation: bicubic
YAML

noctilux inspect-config --config /tmp/noctilux_demo/config.yaml
noctilux run --config /tmp/noctilux_demo/config.yaml
```

## Preview 脚本示例

```bash
python scripts/preview_transforms.py \
  --config configs/presets/all_basic_v021.yaml \
  --image /tmp/noctilux_demo/images/class_a/sample.jpg \
  --output /tmp/noctilux_demo/preview_grid.jpg \
  --max-pipelines 8 \
  --seed 42
```

## 如何新增 Transform

1. 在 `src/noctilux/transforms/` 下新增模块或类。
2. 继承 `BaseTransform`。
3. 用 `@register_transform("your_transform")` 注册。
4. 在 `src/noctilux/transforms/__init__.py` 导入新模块。
5. 在 YAML 中引用。

完整示例见 [docs/adding_new_transform.md](docs/adding_new_transform.md)。

## 当前版本限制

- 仅实现离线图片批处理，不包含训练逻辑。
- 仅支持 `folder` 与 `manifest` 两种输入模式。
- 当前后端仅实现 Pillow + NumPy 基础变换。
- `num_workers` 配置已保留，但 v0.2.1 仍为串行执行，是未来并行接口占位。
- 当前没有 OpenCV / Albumentations / AugLy 后端。
- detection / segmentation 的 annotation 同步增强尚未实现。
- `motion_blur` 是轻量实现，适合预览和离线批处理基线，不是高性能图像处理内核。

## 常见错误

- 输入目录不存在：`run` 或 `make-manifest` 会报 `Input image_root does not exist: ...`
- 配置路径错误：`inspect-config` / `run` 会报 `Config file does not exist: ...`
- 图片损坏：失败样本会写入 `failed_images.csv`，并在 `transform_log.jsonl` / `manifest.csv` 中标记失败
- `overwrite=false` 文件冲突：不会覆盖已有文件，会自动生成 `__dup1`、`__dup2` 等安全后缀
