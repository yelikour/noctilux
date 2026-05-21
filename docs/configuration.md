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
