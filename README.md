# Noctilux

[![CI](https://github.com/yelikour/noctilux/actions/workflows/ci.yml/badge.svg)](https://github.com/yelikour/noctilux/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-0.2.4-orange.svg)](CHANGELOG.md)

Noctilux 是一个通用的离线图像批处理与增强工具。它面向训练前的数据准备阶段，使用 YAML 配置定义可复现、可追溯、可扩展的图像处理流水线，并将输出图片与 metadata 一起落盘。

说明：当前仓库是 private，GitHub Actions badge 在 private 仓库下可能无法正常显示；仓库切换为 public 后可以正常展示。

## 项目状态

- Current version: `0.2.4`
- Execution: serial in `v0.2.x`
- Backends: Pillow + NumPy only
- Not yet supported:
  - parallel processing
  - OpenCV / Albumentations / AugLy
  - detection / segmentation annotation sync
  - PyPI release

## 安装

常规安装：

```bash
pip install -e .
```

开发安装：

```bash
pip install -e ".[dev]"
```

离线或受限网络环境可尝试：

```bash
pip install -e . --no-build-isolation
```

如果本机缺少构建基础组件，可先安装：

```bash
pip install setuptools wheel
```

## 本地构建

```bash
python -m pip install build
python -m build
```

说明：`build` 是开发/发布辅助工具。当前项目不发布 PyPI；发布流程以 git commit 和 git tag 为主，不自动创建 GitHub Release。

## 已支持能力

- YAML 配置驱动批处理
- `folder` / `manifest` 两种输入扫描
- transform registry 动态构建
- pipeline 顺序执行
- `p`、`repeat`、`seed`、随机参数解析
- Pillow 读取与保存，默认 EXIF orientation + RGB
- 输出命名、冲突避让、默认不覆盖
- `manifest.csv`、`transform_log.jsonl`、`failed_images.csv`、`summary.csv`
- CLI：`inspect-config`、`list-transforms`、`preview`、`run`、`make-manifest`
- 单图预览：推荐 `noctilux preview`，兼容入口保留 `scripts/preview_transforms.py`

## Quickstart

仓库内置了一个最小可运行示例图片：`examples/images/sample.jpg`。

```bash
pip install -e ".[dev]"

noctilux list-transforms

noctilux preview \
  --config configs/examples/full_v020.yaml \
  --image examples/images/sample.jpg \
  --output outputs/previews/sample_preview_grid.jpg \
  --max-pipelines 6 \
  --seed 42

noctilux run \
  --config configs/examples/quickstart_sample.yaml \
  --dry-run
```

如果你想实际生成一张处理后的输出图，可以直接运行：

```bash
noctilux run --config configs/examples/quickstart_sample.yaml
```

说明：

- `examples/images/sample.jpg` 是无隐私、无版权争议的合成样例图。
- `preview` 只生成预览 grid，不写 metadata。
- `run` 会生成输出图片和 metadata。
- `outputs/` 是运行产物，已被 `.gitignore` 忽略，不会提交到 Git。

## Transform 覆盖范围

- Compression：`jpeg_compression`、`webp_compression`、`png_resave`、`double_jpeg_compression`
- Resize：`resize_long_edge`、`resize_exact`、`resize_short_edge`、`downscale_upscale`
- Crop：`center_crop_ratio`、`random_crop_ratio`、`random_resized_crop`、`square_crop`
- Geometric：`horizontal_flip`、`vertical_flip`、`rotate`
- Blur：`gaussian_blur`、`median_blur`、`motion_blur`
- Noise：`gaussian_noise`、`poisson_noise`、`salt_pepper_noise`
- Color：`brightness_contrast`、`gamma_correction`、`saturation_hue`、`grayscale`、`sharpen`、`posterize`

## 配置示例与 Presets

示例配置：

- `configs/examples/basic_resize.yaml`
- `configs/examples/full_v020.yaml`
- `configs/examples/compression_plus.yaml`
- `configs/examples/resize_plus.yaml`
- `configs/examples/crop_plus.yaml`
- `configs/examples/geometric_color.yaml`

Preset 配置：

- `configs/presets/classification_light.yaml`
- `configs/presets/compression_robustness.yaml`
- `configs/presets/resize_crop_suite.yaml`
- `configs/presets/visual_degradation_light.yaml`
- `configs/presets/all_basic_v021.yaml`

## 推荐工作流

`inspect-config -> preview -> dry-run -> run -> inspect metadata`

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
noctilux preview \
  --config configs/examples/full_v020.yaml \
  --image examples/images/sample.jpg \
  --output outputs/previews/preview_grid.jpg \
  --max-pipelines 8 \
  --seed 42
```

兼容入口仍可用，但推荐优先使用 CLI：

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

或使用仓库内置 quickstart 配置直接处理 `sample.jpg`：

```bash
noctilux run --config configs/examples/quickstart_sample.yaml
```

6. 查看 metadata：

- `outputs/.../metadata/manifest.csv`
- `outputs/.../metadata/transform_log.jsonl`
- `outputs/.../metadata/failed_images.csv`
- `outputs/.../metadata/summary.csv`

## 输出与 metadata

默认输出目录：

```text
outputs/example_run/
├── images/
├── metadata/
├── logs/
└── previews/
```

metadata 说明：

- `manifest.csv`：每个输出样本一行，记录原图、输出图、pipeline、尺寸、格式、seed 和成功状态。
- `transform_log.jsonl`：每个输出样本一条 JSON，记录 transform 顺序、是否执行、实际采样参数和输入输出信息。
- `failed_images.csv`：记录失败样本的 `pipeline_name`、`repeat_index`、`seed`、`stage` 和错误信息。
- `summary.csv`：按 pipeline 聚合统计总数、成功数、失败数。

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

## 开发者命令

```bash
pip install -e ".[dev]"
python -m pytest
ruff check src tests scripts
noctilux list-transforms
noctilux preview --help
```

## 文档

- [docs/getting_started.md](docs/getting_started.md)
- [docs/configuration.md](docs/configuration.md)
- [docs/input_formats.md](docs/input_formats.md)
- [docs/output_formats.md](docs/output_formats.md)
- [docs/adding_new_transform.md](docs/adding_new_transform.md)

## 新增 transform

新增 transform 的完整示例见 [docs/adding_new_transform.md](docs/adding_new_transform.md)。

## 当前限制

- `num_workers` 仍然只是未来并行接口占位，`v0.2.x` 仍为串行执行。
- 暂无 OpenCV / Albumentations / AugLy 等后端。
- 尚未实现 detection / segmentation annotation 同步增强。
- `motion_blur` 仍是轻量实现，侧重可复现和低依赖，不是高性能图像处理后端。
- `preview` 只做视觉检查，不会生成 `manifest.csv`、`transform_log.jsonl` 或其他批处理 metadata。
