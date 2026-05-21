# Configuration

Noctilux 使用 YAML 驱动离线处理流程。核心顶层字段如下：

- `project`: 项目名、描述、全局 seed。
- `input`: 输入模式和路径配置。
- `output`: 输出根目录、保存格式、覆盖策略。
- `runtime`: 运行时行为，例如 `dry_run`、`num_workers`、`skip_broken_images`。
- `pipelines`: 一个或多个 transform pipeline。

随机参数支持四种形式：

```yaml
quality: 80
quality: [60, 70, 80]
quality:
  type: choice
  values: [60, 70, 80]
quality:
  type: randint
  min: 60
  max: 95
sigma:
  type: uniform
  min: 0.2
  max: 2.0
```

transform 概率：

```yaml
- name: gaussian_blur
  p: 0.3
  params:
    radius:
      type: uniform
      min: 0.2
      max: 1.5
```

repeat 示例：

```yaml
pipelines:
  - name: crop_variants
    repeat: 2
    transforms:
      - name: random_crop_ratio
        params:
          ratio: 0.85
```

综合随机参数示例：

```yaml
pipelines:
  - name: mixed_random
    transforms:
      - name: rotate
        p: 0.5
        params:
          angle:
            type: choice
            values: [-10, -5, 5, 10]
          expand: false
          fill_color: [0, 0, 0]
      - name: jpeg_compression
        params:
          quality:
            type: randint
            min: 70
            max: 92
```

Presets 使用方式：

```bash
noctilux inspect-config --config configs/presets/classification_light.yaml
noctilux run --config configs/presets/classification_light.yaml --dry-run
```

当前 `configs/presets/` 提供的配置：

- `classification_light.yaml`
- `compression_robustness.yaml`
- `resize_crop_suite.yaml`
- `visual_degradation_light.yaml`
- `all_basic_v021.yaml`

Preview 行为：

```bash
noctilux preview \
  --config configs/examples/full_v020.yaml \
  --image path/to/image.jpg \
  --output outputs/previews/preview_grid.jpg \
  --max-pipelines 4 \
  --seed 42
```

- `max_pipelines` 只影响预览时展示多少个 pipeline，不影响批处理执行。
- `preview` 只做单图视觉检查，不会写 `manifest.csv`、`transform_log.jsonl` 或失败统计文件。
