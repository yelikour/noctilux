# noctilux Project Guide

## 1. Project Overview

noctilux 是一个通用的离线图像批处理、图像增强、图像扰动生成与数据集扩展工具。

本项目的核心目标不是训练模型，而是在模型训练前，对图像数据集进行可配置、可复现、可追踪、可扩展的批量处理，生成新的图像数据及对应的 metadata。

它可以服务于多种计算机视觉任务，包括但不限于：

- 图像分类
- 目标检测
- 图像分割
- OCR / 文档图像处理
- 遥感图像处理
- 医学图像预处理
- AIGC 图像检测
- 图像鲁棒性训练
- 数据集清洗与标准化
- 图像压缩、裁剪、缩放、重采样等后处理模拟

noctilux 应该被设计成一个通用工具，而不是某个单一任务的私有脚本。

---

## 2. Project Positioning

### 2.1 What noctilux Is

noctilux 是：

1. 一个离线图像批处理框架。
2. 一个基于 YAML 配置的图像处理流水线工具。
3. 一个可扩展的 transform registry 系统。
4. 一个可以批量生成增强图像并保存到磁盘的工具。
5. 一个可以记录每张输出图像来源、处理方法、参数和随机种子的 metadata 管理工具。
6. 一个可以兼容 Pillow、OpenCV、Albumentations、AugLy、imagecorruptions 等底层库的统一封装层。
7. 一个适合长期维护、开源发布、多人扩展的通用图像增强项目。

### 2.2 What noctilux Is Not

noctilux 不是：

1. 模型训练框架。
2. 深度学习框架。
3. 在线 data loader augmentation 的替代品。
4. 只服务 AIGC 图像检测的专用工具。
5. 只支持分类任务的数据增强脚本。
6. 只支持单张图片处理的小工具。
7. 只依赖某一个第三方增强库的简单封装。

---

## 3. Core Design Principles

### 3.1 Offline First

noctilux 的核心模式是训练前离线处理：

```text
Raw Images
    ↓
noctilux
    ↓
Processed Images + Manifest Metadata
    ↓
Training / Evaluation / Dataset Release
````

模型训练代码不应该依赖 noctilux 的内部实现。训练项目只需要读取处理后的图片和 manifest 文件。

---

### 3.2 Config First

所有处理流程都应该优先通过 YAML 配置文件定义，而不是写死在 Python 脚本中。

示例：

```yaml
pipelines:
  - name: jpeg_q80
    transforms:
      - name: jpeg_compression
        params:
          quality: 80
```

这样可以做到：

* 处理流程可复现
* 实验配置可版本管理
* 不同用户可以无需修改代码就使用项目
* Codex / Claude Code 可以更安全地自动修改配置与模块

---

### 3.3 Metadata First

每一张输出图像都必须能追溯：

* 原始图像路径
* 输出图像路径
* 使用了哪个 pipeline
* 使用了哪些 transforms
* 每个 transform 的实际参数
* 随机种子
* 原图尺寸
* 输出尺寸
* 图像格式
* label / task 信息
* 处理时间
* 是否处理成功
* 失败原因

没有 metadata 的增强数据是不可控的，也不适合长期实验复现。

---

### 3.4 Non-destructive by Default

默认绝不覆盖原始图像。

所有输出应保存到独立目录：

```text
outputs/
├── images/
├── metadata/
└── logs/
```

如果用户想覆盖已有输出，必须显式设置：

```yaml
output:
  overwrite: true
```

---

### 3.5 Extensible Transform Registry

所有 transform 都必须通过 registry 注册。

主程序不应写大量类似下面的判断：

```python
if name == "jpeg":
    ...
elif name == "blur":
    ...
elif name == "crop":
    ...
```

而应该通过统一 registry 动态构建：

```python
transform = build_transform(config)
```

新增 transform 时，只需要：

1. 新建一个 transform 类。
2. 使用 `@register_transform("transform_name")` 注册。
3. 在 YAML 中调用。

---

### 3.6 Backend Agnostic

noctilux 不应该绑定某一个图像处理库。

同一个 transform 可以由不同 backend 实现：

* Pillow
* OpenCV
* Albumentations
* AugLy
* imagecorruptions
* NumPy
* SciPy
* Kornia, optional future backend

示例：

```yaml
- name: gaussian_blur
  backend: opencv
  params:
    sigma: 1.2
```

或：

```yaml
- name: gaussian_blur
  backend: albumentations
  params:
    blur_limit: 5
```

---

### 3.7 Reproducibility

项目必须支持可复现处理。

要求：

1. 支持全局 seed。
2. 支持每张图像独立 seed。
3. 支持记录实际随机参数。
4. 相同输入、相同配置、相同 seed 下，应生成一致的输出。
5. 随机范围参数必须在执行时采样，并记录采样后的实际值。

示例：

```yaml
params:
  quality: [60, 95]
```

执行时可能采样为：

```json
{
  "quality": 82
}
```

metadata 中必须记录 `82`，而不是只记录 `[60, 95]`。

---

## 4. Recommended Repository Structure

推荐项目结构：

```text
noctilux/
├── README.md
├── PROJECT_GUIDE.md
├── CHANGELOG.md
├── LICENSE
├── pyproject.toml
├── requirements.txt
├── requirements-dev.txt
├── configs/
│   ├── examples/
│   │   ├── basic_resize.yaml
│   │   ├── compression.yaml
│   │   ├── crop_resize.yaml
│   │   ├── blur_noise.yaml
│   │   ├── color_jitter.yaml
│   │   ├── social_like.yaml
│   │   ├── corruption.yaml
│   │   └── full_pipeline.yaml
│   └── presets/
│       ├── classification_train_light.yaml
│       ├── classification_train_heavy.yaml
│       ├── detection_safe_resize.yaml
│       ├── document_ocr_cleaning.yaml
│       ├── robustness_compression.yaml
│       └── aigc_detector_robust_train.yaml
├── docs/
│   ├── getting_started.md
│   ├── configuration.md
│   ├── input_formats.md
│   ├── output_formats.md
│   ├── transform_registry.md
│   ├── adding_new_transform.md
│   ├── metadata_schema.md
│   ├── cli_reference.md
│   └── faq.md
├── src/
│   └── noctilux/
│       ├── __init__.py
│       ├── cli.py
│       ├── config.py
│       ├── scanner.py
│       ├── pipeline.py
│       ├── registry.py
│       ├── saver.py
│       ├── metadata.py
│       ├── logging_utils.py
│       ├── random_utils.py
│       ├── exceptions.py
│       ├── image_io/
│       │   ├── __init__.py
│       │   ├── loader.py
│       │   ├── writer.py
│       │   ├── color.py
│       │   └── exif.py
│       ├── backends/
│       │   ├── __init__.py
│       │   ├── pillow_backend.py
│       │   ├── opencv_backend.py
│       │   ├── albumentations_backend.py
│       │   ├── augly_backend.py
│       │   ├── imagecorruptions_backend.py
│       │   └── custom_backend.py
│       ├── transforms/
│       │   ├── __init__.py
│       │   ├── base.py
│       │   ├── compression.py
│       │   ├── resize.py
│       │   ├── crop.py
│       │   ├── geometric.py
│       │   ├── blur.py
│       │   ├── noise.py
│       │   ├── color.py
│       │   ├── occlusion.py
│       │   ├── frequency.py
│       │   ├── document.py
│       │   ├── platform.py
│       │   └── composition.py
│       ├── tasks/
│       │   ├── __init__.py
│       │   ├── classification.py
│       │   ├── detection.py
│       │   ├── segmentation.py
│       │   └── generic.py
│       └── reports/
│           ├── __init__.py
│           ├── summary.py
│           ├── preview.py
│           └── markdown.py
├── scripts/
│   ├── run_batch.py
│   ├── preview_transforms.py
│   ├── check_dataset.py
│   ├── make_manifest.py
│   └── inspect_config.py
├── tests/
│   ├── test_config.py
│   ├── test_scanner.py
│   ├── test_registry.py
│   ├── test_pipeline.py
│   ├── test_saver.py
│   ├── test_metadata.py
│   └── test_transforms/
│       ├── test_compression.py
│       ├── test_resize.py
│       ├── test_crop.py
│       ├── test_blur.py
│       ├── test_noise.py
│       └── test_color.py
├── examples/
│   ├── sample_images/
│   ├── sample_metadata.csv
│   └── quickstart.sh
└── outputs/
    ├── images/
    ├── metadata/
    ├── previews/
    └── logs/
```

---

## 5. Input Data Design

noctilux 应该支持多种输入形式。

### 5.1 Folder Mode

适合简单分类数据集：

```text
data/input/
├── cat/
│   ├── 0001.jpg
│   └── 0002.jpg
├── dog/
│   ├── 0001.jpg
│   └── 0002.jpg
└── bird/
    ├── 0001.jpg
    └── 0002.jpg
```

配置示例：

```yaml
input:
  mode: folder
  image_root: data/input
  infer_label_from_subdir: true
```

---

### 5.2 Manifest CSV Mode

推荐用于正式实验。

```csv
image_path,label,split,task,source,width,height
images/cat/0001.jpg,cat,train,classification,custom_dataset,1024,768
images/dog/0002.jpg,dog,train,classification,custom_dataset,800,800
```

配置示例：

```yaml
input:
  mode: manifest
  manifest_path: data/metadata.csv
  image_root: data/images
  path_column: image_path
  label_column: label
  split_column: split
```

---

### 5.3 Generic Image List Mode

适合无 label 的普通图片批处理。

```csv
image_path
images/a.jpg
images/b.png
images/c.webp
```

配置示例：

```yaml
input:
  mode: image_list
  manifest_path: data/image_list.csv
  image_root: data/images
```

---

### 5.4 Future Task-aware Input

未来支持目标检测和分割时，需要额外字段。

目标检测：

```csv
image_path,annotation_path,annotation_format,split
images/0001.jpg,labels/0001.xml,voc,train
images/0002.jpg,labels/0002.json,coco,train
```

语义分割：

```csv
image_path,mask_path,split
images/0001.jpg,masks/0001.png,train
```

第一阶段可以不实现 annotation 同步变换，但目录和接口设计应提前保留。

---

## 6. Output Data Design

### 6.1 Output Directory

推荐输出结构：

```text
outputs/
├── images/
│   ├── pipeline_name_1/
│   │   ├── class_a/
│   │   └── class_b/
│   ├── pipeline_name_2/
│   │   ├── class_a/
│   │   └── class_b/
│   └── original/
├── metadata/
│   ├── manifest.csv
│   ├── transform_log.jsonl
│   ├── failed_images.csv
│   └── summary.csv
├── previews/
│   ├── preview_grid.jpg
│   └── transform_examples/
└── logs/
    └── run.log
```

---

### 6.2 Output Naming Rule

默认命名规则：

```text
{stem}__{pipeline_name}__{index}.{ext}
```

示例：

```text
cat_0001__jpeg_q80__000.jpg
cat_0001__crop_resize_512__000.jpg
cat_0001__social_heavy__000.jpg
```

如果同一 pipeline 对一张图 repeat 多次：

```text
cat_0001__random_crop__000.jpg
cat_0001__random_crop__001.jpg
cat_0001__random_crop__002.jpg
```

---

### 6.3 Manifest Schema

`manifest.csv` 必须包含以下基础字段：

```csv
sample_id,original_path,output_path,pipeline_name,repeat_index,input_width,input_height,output_width,output_height,input_format,output_format,success,error,seed,label,split,task
```

可选字段：

```csv
source,dataset,generator,license,annotation_path,mask_path,group_id
```

其中：

* `sample_id`: 原始图像唯一 ID
* `original_path`: 原始图像路径
* `output_path`: 输出图像路径
* `pipeline_name`: 使用的处理流水线
* `repeat_index`: 同一个 pipeline 重复生成的序号
* `success`: 是否处理成功
* `error`: 失败原因
* `seed`: 本次处理使用的随机种子
* `label`: 分类标签，可为空
* `split`: train / val / test / unknown
* `task`: classification / detection / segmentation / generic / unknown

---

### 6.4 Transform Log Schema

`transform_log.jsonl` 用于记录更详细的 transform 信息。

每一行是一个 JSON object：

```json
{
  "sample_id": "000001",
  "original_path": "data/images/cat/0001.jpg",
  "output_path": "outputs/images/jpeg_q80/cat/0001__jpeg_q80__000.jpg",
  "pipeline_name": "jpeg_q80",
  "repeat_index": 0,
  "seed": 42,
  "transforms": [
    {
      "name": "jpeg_compression",
      "backend": "pillow",
      "params": {
        "quality": 80,
        "subsampling": "4:2:0"
      }
    }
  ],
  "input_info": {
    "width": 1024,
    "height": 768,
    "mode": "RGB",
    "format": "JPEG"
  },
  "output_info": {
    "width": 1024,
    "height": 768,
    "mode": "RGB",
    "format": "JPEG"
  },
  "success": true,
  "error": null
}
```

---

## 7. Configuration File Specification

### 7.1 Basic Config

```yaml
project:
  name: noctilux_example
  seed: 42
  description: "Example offline image augmentation run"

input:
  mode: folder
  image_root: data/input
  infer_label_from_subdir: true
  recursive: true
  extensions: [".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"]

output:
  root: outputs/example_run
  image_dir: images
  metadata_dir: metadata
  log_dir: logs
  preview_dir: previews
  preserve_subdirs: true
  keep_original: false
  overwrite: false
  save_format: jpg
  jpg_quality: 95
  png_compression: 3

runtime:
  num_workers: 8
  batch_size: 1
  skip_broken_images: true
  fail_fast: false
  show_progress: true
  dry_run: false

pipelines:
  - name: jpeg_q80
    repeat: 1
    enabled: true
    transforms:
      - name: jpeg_compression
        backend: pillow
        params:
          quality: 80
          subsampling: "4:2:0"
```

---

### 7.2 Random Parameter Config

随机参数可以使用列表或 range 语法。

方式一：列表随机采样。

```yaml
params:
  quality: [60, 70, 80, 90]
```

方式二：范围随机采样。

```yaml
params:
  quality:
    type: randint
    min: 60
    max: 95
```

方式三：浮点范围。

```yaml
params:
  sigma:
    type: uniform
    min: 0.2
    max: 2.0
```

方式四：固定值。

```yaml
params:
  quality: 80
```

要求：

1. pipeline 执行前将随机参数解析成具体值。
2. 具体值必须写入 transform log。
3. 不要只记录随机范围。

---

### 7.3 Pipeline Probability

支持 transform 级别概率：

```yaml
transforms:
  - name: gaussian_blur
    p: 0.3
    params:
      sigma:
        type: uniform
        min: 0.3
        max: 1.5
```

当 transform 未执行时，也应在 log 中记录：

```json
{
  "name": "gaussian_blur",
  "applied": false,
  "p": 0.3
}
```

---

### 7.4 Multiple Pipelines

一个 config 可以定义多个 pipeline。

```yaml
pipelines:
  - name: resize_512
    repeat: 1
    transforms:
      - name: resize_long_edge
        params:
          long_edge: 512

  - name: jpeg_q70
    repeat: 1
    transforms:
      - name: jpeg_compression
        params:
          quality: 70

  - name: crop_resize_jpeg
    repeat: 3
    transforms:
      - name: random_crop_ratio
        params:
          ratio:
            type: uniform
            min: 0.6
            max: 0.9
      - name: resize_long_edge
        params:
          long_edge: 512
      - name: jpeg_compression
        params:
          quality:
            type: randint
            min: 60
            max: 95
```

---

## 8. Core Modules

### 8.1 config.py

负责：

* 读取 YAML
* 校验必填字段
* 设置默认值
* 解析路径
* 解析随机参数定义
* 生成最终运行配置

应提供：

```python
load_config(path: str) -> dict
validate_config(config: dict) -> None
resolve_config(config: dict) -> dict
```

---

### 8.2 scanner.py

负责：

* 扫描输入图片
* 读取 manifest
* 过滤扩展名
* 过滤 split
* 检查路径是否存在
* 生成标准样本列表

标准样本结构：

```python
{
    "sample_id": "000001",
    "image_path": "data/images/a.jpg",
    "label": "cat",
    "split": "train",
    "task": "classification",
    "metadata": {}
}
```

---

### 8.3 registry.py

负责 transform 注册和构建。

要求：

```python
TRANSFORM_REGISTRY = {}

def register_transform(name: str):
    ...

def build_transform(name: str, backend: str, params: dict):
    ...

def list_transforms():
    ...
```

注册示例：

```python
@register_transform("jpeg_compression")
class JPEGCompression(BaseTransform):
    ...
```

---

### 8.4 pipeline.py

负责：

* 根据 config 构建 pipeline
* 按顺序执行 transforms
* 处理 transform 概率
* 采样随机参数
* 记录 transform log
* 捕获异常
* 返回图像和 metadata

核心类：

```python
class AugmentPipeline:
    def __init__(self, name, transforms, repeat=1, seed=None):
        ...

    def apply(self, image, sample):
        ...
```

---

### 8.5 saver.py

负责：

* 构建输出路径
* 保存图片
* 处理命名冲突
* 支持 preserve_subdirs
* 支持输出格式转换
* 支持 jpg_quality / png_compression
* 禁止默认覆盖

---

### 8.6 metadata.py

负责：

* 写 manifest.csv
* 写 transform_log.jsonl
* 写 failed_images.csv
* 写 summary.csv
* 支持追加模式
* 支持运行结束汇总

---

### 8.7 image_io/

负责：

* 安全读取图像
* 修正 EXIF orientation
* 统一 RGB / RGBA / grayscale 处理
* 兼容 jpg、png、webp、bmp、tiff
* 处理损坏图片
* 保留或丢弃 alpha 通道

建议默认行为：

```yaml
image:
  apply_exif_orientation: true
  convert_mode: RGB
  keep_alpha: false
```

---

## 9. Transform System

所有 transform 继承统一基类。

```python
class BaseTransform:
    name: str = "base_transform"

    def __init__(self, **params):
        self.params = params

    def __call__(self, image, context=None):
        raise NotImplementedError

    def get_params(self):
        return self.params
```

### 9.1 Transform Input / Output Contract

第一阶段统一使用 PIL.Image 作为 transform 输入输出。

要求：

```text
Input: PIL.Image.Image
Output: PIL.Image.Image
```

如果底层 backend 使用 OpenCV / NumPy，必须在 transform 内部完成转换。

不要让 pipeline 中同时出现混乱的 image type。

---

## 10. Transform Categories

### 10.1 Compression Transforms

必须优先实现：

* `jpeg_compression`
* `webp_compression`
* `png_resave`
* `double_jpeg_compression`

可选扩展：

* `avif_compression`
* `heic_compression`
* `jpeg_progressive`
* `jpeg_subsampling`
* `jpeg_block_artifact`

---

### 10.2 Resize Transforms

必须优先实现：

* `resize_long_edge`
* `resize_short_edge`
* `resize_exact`
* `downscale_upscale`
* `random_interpolation_resize`

参数示例：

```yaml
- name: resize_long_edge
  params:
    long_edge: 512
    interpolation: bicubic
```

---

### 10.3 Crop Transforms

必须优先实现：

* `center_crop_ratio`
* `random_crop_ratio`
* `random_resized_crop`
* `square_crop`
* `letterbox`

参数示例：

```yaml
- name: center_crop_ratio
  params:
    ratio: 0.75
```

---

### 10.4 Geometric Transforms

可逐步实现：

* `horizontal_flip`
* `vertical_flip`
* `rotate`
* `affine`
* `perspective`
* `pad`
* `shift_scale_rotate`

---

### 10.5 Blur Transforms

必须优先实现：

* `gaussian_blur`
* `motion_blur`
* `median_blur`

可选扩展：

* `defocus_blur`
* `zoom_blur`
* `glass_blur`

---

### 10.6 Noise Transforms

必须优先实现：

* `gaussian_noise`
* `poisson_noise`
* `salt_pepper_noise`

可选扩展：

* `speckle_noise`
* `iso_noise`
* `shot_noise`

---

### 10.7 Color Transforms

必须优先实现：

* `brightness_contrast`
* `gamma_correction`
* `saturation_hue`
* `grayscale`

可选扩展：

* `white_balance_shift`
* `color_temperature`
* `histogram_equalization`
* `clahe`
* `posterize`
* `solarize`
* `channel_shuffle`

---

### 10.8 Occlusion Transforms

可逐步实现：

* `cutout`
* `random_erasing`
* `grid_mask`
* `coarse_dropout`
* `patch_shuffle`
* `random_overlay`

---

### 10.9 Frequency Transforms

后续作为高级扩展：

* `fft_low_pass`
* `fft_high_pass`
* `fft_frequency_dropout`
* `dct_perturbation`
* `wavelet_denoise`
* `frequency_masking`

频域 transform 对 AIGC 检测、图像压缩鲁棒性、纹理鲁棒性任务比较重要，但不建议在 MVP 阶段优先实现。

---

### 10.10 Document / OCR Transforms

面向文档图像和 OCR 场景：

* `scan_shadow`
* `paper_noise`
* `document_blur`
* `perspective_scan`
* `text_fade`
* `background_texture`
* `document_export_like`

---

### 10.11 Platform-like Transforms

用于模拟真实互联网平台后处理：

* `screenshot_like`
* `social_media_light`
* `social_media_heavy`
* `thumbnail_like`
* `messaging_app_compression`
* `document_insert_export`
* `repost_like`

这些 transform 本质上是多个基础 transform 的组合。

示例：

```yaml
- name: social_media_heavy
  params:
    long_edge: 1080
    jpeg_quality: 75
    blur_sigma: 0.3
    sharpen: true
```

---

## 11. Preset Pipelines

项目应该提供一些内置 preset，方便用户直接上手。

### 11.1 Basic Resize

```yaml
name: basic_resize
pipelines:
  - name: resize_512
    transforms:
      - name: resize_long_edge
        params:
          long_edge: 512
```

---

### 11.2 Compression Robustness

```yaml
name: compression_robustness
pipelines:
  - name: jpeg_q95
    transforms:
      - name: jpeg_compression
        params:
          quality: 95

  - name: jpeg_q80
    transforms:
      - name: jpeg_compression
        params:
          quality: 80

  - name: jpeg_q60
    transforms:
      - name: jpeg_compression
        params:
          quality: 60
```

---

### 11.3 Crop Resize

```yaml
name: crop_resize
pipelines:
  - name: crop075_resize512
    transforms:
      - name: center_crop_ratio
        params:
          ratio: 0.75
      - name: resize_long_edge
        params:
          long_edge: 512
```

---

### 11.4 Classification Train Light

```yaml
name: classification_train_light
pipelines:
  - name: cls_light_aug
    repeat: 2
    transforms:
      - name: random_crop_ratio
        p: 0.5
        params:
          ratio:
            type: uniform
            min: 0.8
            max: 1.0
      - name: horizontal_flip
        p: 0.5
      - name: brightness_contrast
        p: 0.3
        params:
          brightness:
            type: uniform
            min: -0.1
            max: 0.1
          contrast:
            type: uniform
            min: -0.1
            max: 0.1
```

---

### 11.5 AIGC Detector Robust Train

这个 preset 可以作为专用场景，但不应影响项目通用定位。

```yaml
name: aigc_detector_robust_train
pipelines:
  - name: aigc_compression_crop_resize
    repeat: 3
    transforms:
      - name: random_crop_ratio
        p: 0.5
        params:
          ratio:
            type: uniform
            min: 0.6
            max: 1.0
      - name: resize_long_edge
        p: 1.0
        params:
          long_edge:
            type: choice
            values: [512, 768, 1024]
      - name: jpeg_compression
        p: 0.8
        params:
          quality:
            type: randint
            min: 50
            max: 95
```

---

## 12. CLI Design

### 12.1 Main Command

推荐 CLI：

```bash
noctilux run --config configs/examples/compression.yaml
```

支持覆盖参数：

```bash
noctilux run \
  --config configs/examples/compression.yaml \
  --input data/input \
  --output outputs/compression_run \
  --num-workers 8 \
  --seed 42
```

---

### 12.2 Preview Command

预览某个配置对单张图片的效果：

```bash
noctilux preview \
  --config configs/examples/full_pipeline.yaml \
  --image examples/sample_images/test.jpg \
  --output outputs/previews
```

---

### 12.3 Inspect Config Command

检查配置是否有效：

```bash
noctilux inspect-config --config configs/examples/full_pipeline.yaml
```

输出：

```text
Config valid.
Pipelines: 3
Transforms: 8
Estimated outputs: 12000
```

---

### 12.4 List Transforms Command

列出当前可用 transform：

```bash
noctilux list-transforms
```

输出：

```text
compression:
  - jpeg_compression
  - webp_compression
  - double_jpeg_compression

resize:
  - resize_long_edge
  - resize_exact
  - downscale_upscale
```

---

### 12.5 Make Manifest Command

从文件夹创建 manifest：

```bash
noctilux make-manifest \
  --image-root data/input \
  --output data/metadata.csv \
  --infer-label-from-subdir
```

---

## 13. Development Stages

### Stage 0: Project Bootstrap

目标：建立标准 Python 项目结构。

完成内容：

1. `pyproject.toml`
2. `src/noctilux/`
3. CLI 入口
4. 基础 README
5. 基础测试框架
6. GitHub Actions 可选

验收标准：

```bash
pip install -e .
noctilux --help
pytest
```

---

### Stage 1: MVP Offline Processor

目标：实现最小可用离线批处理。

必须完成：

1. folder mode 输入
2. manifest mode 输入
3. 读取 YAML 配置
4. transform registry
5. pipeline 顺序执行
6. PIL 图片读取和保存
7. 输出 manifest.csv
8. 输出 transform_log.jsonl
9. 输出 failed_images.csv
10. 支持 seed
11. 支持 dry-run

必须实现 transforms：

* `jpeg_compression`
* `resize_long_edge`
* `center_crop_ratio`
* `gaussian_blur`
* `gaussian_noise`
* `brightness_contrast`

验收命令：

```bash
noctilux run --config configs/examples/basic_resize.yaml
noctilux run --config configs/examples/compression.yaml
noctilux preview --config configs/examples/compression.yaml --image examples/sample_images/test.jpg
pytest
```

---

### Stage 2: Common Augmentation Expansion

目标：覆盖常见图像增强。

新增 transforms：

* `webp_compression`
* `png_resave`
* `double_jpeg_compression`
* `resize_exact`
* `downscale_upscale`
* `random_crop_ratio`
* `random_resized_crop`
* `horizontal_flip`
* `rotate`
* `motion_blur`
* `poisson_noise`
* `salt_pepper_noise`
* `gamma_correction`
* `saturation_hue`
* `grayscale`

新增能力：

1. transform 概率 `p`
2. repeat
3. 随机参数解析
4. 多 pipeline 同时执行
5. summary.csv
6. preview grid

---

### Stage 3: Backend Integration

目标：接入外部库，但保持项目解耦。

优先级：

1. OpenCV backend
2. Albumentations backend
3. imagecorruptions backend
4. AugLy backend

要求：

1. 外部依赖应设计为 optional dependencies。
2. 如果用户未安装某 backend，应给出清晰错误信息。
3. 不允许主流程强依赖所有 backend。

示例安装：

```bash
pip install "noctilux[albumentations]"
pip install "noctilux[augly]"
pip install "noctilux[full]"
```

---

### Stage 4: Task-aware Support

目标：支持更多视觉任务。

分类任务：

* 支持 label。
* 支持按类别保存。
* 支持 class balance 增强策略。

目标检测任务：

* 支持 COCO / VOC annotation。
* 几何变换时同步更新 bbox。
* 非几何变换不修改 bbox。

分割任务：

* 支持 mask_path。
* 几何变换时同步处理 mask。
* mask 不应被颜色扰动、模糊、压缩影响，除非用户明确开启。

第一阶段只需保留接口，不强制实现完整 annotation 同步。

---

### Stage 5: Advanced Robustness and Platform Simulation

目标：加入更真实的后处理链。

新增 transforms：

* `screenshot_like`
* `social_media_light`
* `social_media_heavy`
* `messaging_app_compression`
* `document_export_like`
* `thumbnail_like`
* `fft_low_pass`
* `dct_perturbation`
* `cutout`
* `grid_mask`
* `patch_shuffle`

---

## 14. Testing Requirements

测试必须覆盖：

### 14.1 Config Tests

1. 能读取合法 YAML。
2. 缺少必填字段时报错清晰。
3. 随机参数能正确解析。
4. 路径能正确解析。
5. dry-run 不产生图片输出。

---

### 14.2 Scanner Tests

1. 能扫描 folder mode。
2. 能读取 manifest mode。
3. 能跳过非图片文件。
4. 能递归扫描。
5. 能处理不存在的图片路径。

---

### 14.3 Transform Tests

每个 transform 必须测试：

1. 输入 PIL.Image。
2. 输出 PIL.Image。
3. 输出尺寸合法。
4. 输出 mode 合法。
5. 不会修改原始 image object。
6. 参数非法时给出清晰错误。

---

### 14.4 Pipeline Tests

1. 多 transform 顺序执行。
2. transform 概率生效。
3. repeat 生效。
4. seed 固定时结果一致。
5. transform log 记录完整。

---

### 14.5 Metadata Tests

1. manifest.csv 行数正确。
2. transform_log.jsonl 每行是合法 JSON。
3. failed_images.csv 能记录失败样本。
4. summary.csv 能正确统计 pipeline 输出数量。

---

## 15. Code Style Requirements

### 15.1 General

1. 代码应简单直接。
2. 优先小函数、小类。
3. 不要写大型全局脚本。
4. 不要把所有 transform 写在一个文件里。
5. 不要让 CLI、pipeline、transform、metadata 强耦合。
6. 所有路径使用 `pathlib.Path`。
7. 所有异常应有清晰错误信息。

---

### 15.2 Type Hints

核心函数必须写 type hints。

示例：

```python
from pathlib import Path
from PIL import Image

def load_image(path: Path, mode: str = "RGB") -> Image.Image:
    ...
```

---

### 15.3 Logging

不要大量使用 `print()`。

使用 logging：

```python
import logging

logger = logging.getLogger(__name__)
```

CLI 中可以配置 log level：

```bash
noctilux run --config config.yaml --log-level INFO
```

---

### 15.4 Exceptions

定义项目内部异常：

```python
class noctiluxError(Exception):
    pass

class ConfigError(noctiluxError):
    pass

class TransformError(noctiluxError):
    pass

class ImageLoadError(noctiluxError):
    pass
```

---

## 16. Safety and Data Integrity

### 16.1 Never Overwrite Raw Data

默认禁止覆盖原始图像。

### 16.2 Validate Output Path

所有输出路径必须位于 output root 内部，防止路径穿越问题。

### 16.3 Broken Image Handling

配置项：

```yaml
runtime:
  skip_broken_images: true
  fail_fast: false
```

如果 `skip_broken_images=true`，损坏图像写入 `failed_images.csv`。

### 16.4 Large Dataset Handling

大数据集处理要求：

1. 不要一次性把所有图片读入内存。
2. 逐图处理。
3. 支持多进程。
4. metadata 流式写入。
5. 进度条显示当前处理进度。
6. 支持中断后继续处理，后续可实现 resume。

---

## 17. Optional Dependencies

建议将依赖分层。

基础依赖：

```text
pillow
numpy
pandas
pyyaml
tqdm
```

OpenCV 扩展：

```text
opencv-python
```

开发依赖：

```text
pytest
ruff
mypy
```

可选增强后端：

```text
albumentations
augly
imagecorruptions
scipy
```

`pyproject.toml` 中建议设计：

```toml
[project.optional-dependencies]
opencv = ["opencv-python"]
albumentations = ["albumentations"]
augly = ["augly"]
corruptions = ["imagecorruptions"]
dev = ["pytest", "ruff", "mypy"]
full = [
  "opencv-python",
  "albumentations",
  "augly",
  "imagecorruptions",
  "scipy"
]
```

---

## 18. README Requirements

README 至少包含：

1. 项目简介
2. 安装方式
3. 快速开始
4. 输入数据格式
5. 输出数据格式
6. YAML 配置示例
7. CLI 使用示例
8. 如何新增 transform
9. 常见问题
10. License

快速开始示例：

```bash
git clone https://github.com/your-name/noctilux.git
cd noctilux

pip install -e .

noctilux make-manifest \
  --image-root examples/sample_images \
  --output examples/sample_metadata.csv \
  --infer-label-from-subdir

noctilux run \
  --config configs/examples/basic_resize.yaml
```

---

## 19. Documentation Requirements

`docs/adding_new_transform.md` 应包含完整示例。

示例 transform：

```python
from PIL import Image, ImageEnhance

from noctilux.registry import register_transform
from noctilux.transforms.base import BaseTransform

@register_transform("brightness")
class BrightnessTransform(BaseTransform):
    def __init__(self, factor: float = 1.0):
        self.factor = factor

    def __call__(self, image: Image.Image, context=None) -> Image.Image:
        enhancer = ImageEnhance.Brightness(image)
        return enhancer.enhance(self.factor)
```

YAML 调用：

```yaml
- name: brightness
  params:
    factor: 1.2
```

---

## 20. Roadmap

以下是建议路线，实际版本可能根据维护需要调整。

### v0.1.x（已完成）

目标：MVP 与稳定性。

功能：

* folder input / manifest input
* YAML config
* registry / pipeline
* 6 个基础 transform（JPEG compression、resize、crop、blur、noise、brightness/contrast）
* manifest.csv / transform_log.jsonl / failed_images.csv
* CLI run / preview
* 干跑、输出冲突避让、失败记录

---

### v0.2.x（已完成）

目标：常见增强完整化、CI、quickstart。

新增：

* random parameter、transform probability、repeat、multi-pipeline
* 21 个常见 transform（compression、resize、crop、geometric、blur、noise、color）
* reusable presets
* `noctilux preview` CLI
* GitHub Actions CI（Python 3.10–3.12）
* tracked synthetic sample image + runnable quickstart workflow
* summary.csv

---

### v0.3.x（已完成）

目标：metadata 报告与文档一致性。

* v0.3.0：`noctilux report` CLI（Markdown + 可选 CSV 报告）
* v0.3.1：Python 3.10 report 兼容性修复
* v0.3.2：agent handoff 文档
* v0.3.3：Python 版本声明对齐 CI、路线图修正、public-readiness checklist

---

### v0.4.0（已完成）

目标：optional backend 探索。

设计文档：`docs/backend_design.md`

已完成：

* v0.4.0a / design：backend 架构设计文档（v0.3.7）
* v0.4.0：最小 OpenCV backend 实现（resize_exact、resize_long_edge、gaussian_blur、rotate）

后续计划：

* v0.4.x：扩展 backend 覆盖范围，考虑 Albumentations 等

原则：

* Pillow + NumPy 仍是默认 backend
* OpenCV 为 optional dependency（`noctilux[opencv]`）
* 所有 transform 输入输出仍为 PIL.Image.Image
* 考虑 Albumentations / imagecorruptions / AugLy 等 backend
* 所有外部 backend 为 optional dependencies，主流程不强依赖

---

### v0.5.0（计划中）

目标：并行处理与恢复。

设计文档：`docs/parallel_resume_design.md`

分阶段计划：

* v0.5.0：parallel/resume 设计文档
* v0.5.1：metadata-safe writer 重构（流式写入，替代内存收集）
* v0.5.2：serial resume（`--resume`、`--skip-existing`、`--retry-failed`）
* v0.5.3：process pool prototype（`ProcessPoolExecutor`，worker 返回 result，main process 写 metadata）

原则：

* `num_workers=1` 仍为默认值
* 并行不改变 seed 确定性
* metadata 写入始终单进程
* resume 基于 manifest.csv 和 transform_log.jsonl 判断已完成任务

---

### v0.6.0（已完成）

目标：并行处理加固。

已完成：
* bounded in-flight task submission
* worker future exception handling
* spawn-mode smoke coverage
* save-image failure alignment with skip_broken_images
* parallel remains experimental / hardening-stage, not stable

---

### v0.7.x（已完成）

目标：annotation 同步设计。

设计文档：`docs/annotation_sync_design.md`

分阶段计划：

* v0.7.0：annotation sync 设计文档（无代码改动）
* v0.7.1：annotation schema / parser prototype（AnnotationRecord, BoundingBox, COCO parser）
* v0.7.2：bbox sync for resize / flip
* v0.7.3：crop bbox handling（clipping, elimination, min_area）
* v0.7.4：annotation writer prototype（COCO JSON writer, YOLO TXT writer；未接入 `noctilux run`）
* v0.7.5：annotation writer cleanup（全局唯一 annotation_id，无独立 mask annotation，可选 YOLO bounds 校验；未接入 `noctilux run`）

原则：

* 默认不启用 annotation sync
* image-only pipeline 行为不能改变
* 只有 geometry transform 需要同步 annotation
* photometric transform 不改变 annotation
* 输出 annotation 必须可追溯
* metadata schema 不能破坏现有字段

---

### v0.8.0（已完成）

目标：minimal annotation IO integration。

已完成：

* experimental opt-in `annotations` config
* COCO-like bbox-only input/output wired into `noctilux run`
* bbox sync for `resize_exact`, `resize_long_edge`, `horizontal_flip`, `vertical_flip`
* photometric transforms keep bbox unchanged
* unsupported transform policy: `error` / `ignore`
* output COCO-like annotation JSON writing
* image-only 默认行为和现有 metadata schema 字段保持兼容

仍不支持：

* mask / polygon / keypoint sync
* rotate bbox sync
* crop integration（当前 transform log 未暴露可靠 crop window）
* full COCO feature parity
* YOLO dataset-level integration
* VOC XML integration

---

## 21. Coding Instructions for Codex / Claude Code

当使用 Codex 或 Claude Code 迭代本项目时，应遵守：

1. 先阅读 `PROJECT_GUIDE.md`、`README.md`、`pyproject.toml` 和现有源码。
2. 不要重写整个项目，优先做小步增量修改。
3. 每次新增功能后必须补充测试。
4. 每次新增 transform 后必须：

   * 注册 transform
   * 添加 YAML 示例
   * 添加单元测试
   * 更新 transform list 文档
5. 不要把任务特定逻辑写死在主程序中。
6. 不要让 AIGC、分类、检测、分割等任务逻辑污染通用 pipeline。
7. 不要把所有 transform 塞进一个文件。
8. 所有输出必须可追溯。
9. 所有随机参数必须记录实际采样值。
10. 修改后运行：

    ```bash
    pytest
    noctilux --help
    noctilux inspect-config --config configs/examples/basic_resize.yaml
    ```
11. 如果测试失败，先修测试暴露的问题，不要删除测试。
12. 如果需要新增第三方依赖，必须说明原因，并优先设为 optional dependency。
13. 如果某个功能暂时不实现，应保留清晰 TODO，而不是写假实现。
14. 不要为了“看起来完整”写无法运行的伪代码。
15. 每次提交应聚焦一个明确主题。

---

## 22. Initial Implementation Priority

如果从零开始，请按以下顺序实现：

### Step 1

建立项目结构：

```text
src/noctilux/
tests/
configs/examples/
docs/
```

### Step 2

实现：

* `config.py`
* `scanner.py`
* `registry.py`
* `pipeline.py`
* `saver.py`
* `metadata.py`
* `cli.py`

### Step 3

实现基础 transforms：

* `jpeg_compression`
* `resize_long_edge`
* `center_crop_ratio`
* `gaussian_blur`
* `gaussian_noise`
* `brightness_contrast`

### Step 4

实现示例配置：

* `basic_resize.yaml`
* `compression.yaml`
* `crop_resize.yaml`
* `blur_noise.yaml`

### Step 5

补充测试：

* config
* scanner
* registry
* pipeline
* metadata
* transforms

### Step 6

完善 README 和 quickstart。

---

## 23. Final Goal

noctilux 的最终目标是成为一个：

> 通用、可配置、可复现、可扩展的离线图像处理与增强框架。

它应该既能处理简单任务：

```bash
noctilux run --config resize_512.yaml
```

也能支持复杂任务：

```bash
noctilux run --config robust_training_full_pipeline.yaml
```

最终用户不需要懂内部代码，只需要准备图片和配置文件，就可以批量生成可追溯的增强图像数据集。
