# Input Formats

## Folder Mode

```yaml
input:
  mode: folder
  image_root: examples/images
  infer_label_from_subdir: true
  recursive: true
```

仓库自带的 `examples/images/sample.jpg` 是最小示例图片，可直接用于 quickstart 和 preview 流程。

注意：

- 如果图片直接放在 `examples/images/` 根目录下，而不是子目录中，`infer_label_from_subdir: true` 不会推断出稳定标签。
- 如果你想让单张样例图直接参与 quickstart，建议使用 `configs/examples/quickstart_sample.yaml` 这类 `infer_label_from_subdir: false` 的配置。

输出样本至少包含：

- `sample_id`
- `image_path`
- `label`
- `split`
- `task`
- `metadata`

## Manifest Mode

```yaml
input:
  mode: manifest
  manifest_path: examples/sample_manifest.csv
  image_root: examples/images
  path_column: image_path
  label_column: label
  split_column: split
```

推荐 CSV 字段：

```csv
sample_id,image_path,label,split,task
000001,class_a/a.jpg,class_a,train,generic
```
