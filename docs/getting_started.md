# Getting Started

1. 安装项目：

```bash
pip install -e .
```

2. 查看可用 transforms：

```bash
noctilux list-transforms
```

3. 检查配置：

```bash
noctilux inspect-config --config configs/examples/basic_resize.yaml
```

4. 运行批处理：

```bash
noctilux run --config configs/examples/basic_resize.yaml
```

5. 查看输出：

- `outputs/.../images/`
- `outputs/.../metadata/manifest.csv`
- `outputs/.../metadata/transform_log.jsonl`
