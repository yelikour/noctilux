# Getting Started

1. 安装项目：

```bash
pip install -e .
```

2. 准备少量测试图片：

```bash
mkdir -p /tmp/noctilux_demo/images/class_a
python - <<'PY'
from pathlib import Path
from PIL import Image
root = Path("/tmp/noctilux_demo/images/class_a")
root.mkdir(parents=True, exist_ok=True)
Image.new("RGB", (640, 480), color=(120, 80, 40)).save(root / "sample.jpg")
PY
```

3. 查看可用 transforms：

```bash
noctilux list-transforms
```

4. 从目录生成 manifest：

```bash
noctilux make-manifest --image-root /tmp/noctilux_demo/images --output /tmp/noctilux_demo/manifest.csv --infer-label-from-subdir
```

5. 检查配置：

```bash
noctilux inspect-config --config configs/presets/all_basic_v021.yaml
```

6. 用单图做预览：

```bash
python scripts/preview_transforms.py \
  --config configs/examples/full_v020.yaml \
  --image /tmp/noctilux_demo/images/class_a/sample.jpg \
  --output /tmp/noctilux_demo/preview_grid.jpg \
  --max-pipelines 8 \
  --seed 42
```

7. 先 dry-run：

```bash
noctilux run --config configs/presets/all_basic_v021.yaml --dry-run
```

8. 运行批处理：

```bash
noctilux run --config configs/presets/all_basic_v021.yaml
```

9. 查看输出：

- `outputs/.../images/`
- `outputs/.../metadata/manifest.csv`
- `outputs/.../metadata/transform_log.jsonl`
- `outputs/.../metadata/failed_images.csv`
- `outputs/.../metadata/summary.csv`

metadata 文件用途：

- `manifest.csv`：每个输出样本的索引表，适合后续训练或分析
- `transform_log.jsonl`：每张输出图的详细 transform 执行记录
- `failed_images.csv`：失败样本、阶段、pipeline、seed 和错误信息
- `summary.csv`：按 pipeline 汇总成功/失败数量
