# Output Formats

默认输出目录：

```text
outputs/example_run/
├── images/
├── metadata/
├── logs/
└── previews/
```

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
sample_id,image_path,error
```

## summary.csv

```csv
pipeline_name,total,success,failed
```
