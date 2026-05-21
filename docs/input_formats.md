# Input Formats

## Folder Mode

```yaml
input:
  mode: folder
  image_root: examples/images
  infer_label_from_subdir: true
  recursive: true
```

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
