# Output Formats

默认输出目录：

```text
outputs/example_run/
├── images/
├── metadata/
├── logs/
└── previews/
```

## Preview vs Run

- `noctilux preview`：只生成一张预览 grid 图片，用于快速做视觉检查；不会生成 `manifest.csv`、`transform_log.jsonl`、`failed_images.csv` 或 `summary.csv`。
- `noctilux run`：执行完整离线批处理，生成输出图片和 metadata。

## manifest.csv

至少包含：

```csv
sample_id,original_path,output_path,pipeline_name,repeat_index,input_width,input_height,output_width,output_height,input_format,output_format,success,error,seed,label,split,task
```

## transform_log.jsonl

每行一个 JSON object，记录：

- 原图路径
- 输出路径
- pipeline 名称
- repeat 序号
- seed
- 每个 transform 的实际参数和是否执行

## failed_images.csv

```csv
sample_id,image_path,pipeline_name,repeat_index,seed,stage,error
```

## summary.csv

```csv
pipeline_name,total,success,failed
```

## report.md

`noctilux report` 读取 `manifest.csv`、`summary.csv`、`failed_images.csv` 和 `transform_log.jsonl`，生成轻量 Markdown 汇总。报告包含总数、成功率、pipeline 汇总、输出格式、尺寸统计、失败阶段、错误信息和 transform 使用统计。

## summary_report.csv

如果传入 `--csv-output`，`noctilux report` 会额外写出 `summary_report.csv`，用于脚本化读取关键统计值。
