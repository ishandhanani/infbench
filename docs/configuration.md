# Configuration

## Cluster Config (`srtslurm.yaml`)

Created by `make setup`. Define your cluster defaults and aliases:

```yaml
cluster:
  account: "your-account"
  partition: "batch"
  network_interface: "enP6p9s0np0"
  gpus_per_node: 4
  default_time_limit: "4:00:00"
  default_container: "/path/to/container.sqsh"

# Model aliases (optional)
model_paths:
  deepseek-r1: "/models/DeepSeek-R1"
  llama-70b: "/models/Llama-3-70B"

# Container aliases (optional)
containers:
  latest: "/containers/sglang-v0.5.5.sqsh"
  stable: "/containers/sglang-v0.5.4.sqsh"
```

## Job Config

### Minimal Example

```yaml
name: "my-benchmark"

model:
  path: "deepseek-r1"        # Alias or full path
  container: "latest"
  precision: "fp8"

resources:
  prefill_nodes: 1
  decode_nodes: 4

benchmark:
  type: "sa-bench"
  isl: 1024
  osl: 1024
  concurrencies: [256, 512]
```

### Full Example

```yaml
name: "production-benchmark"

model:
  path: "deepseek-r1"
  container: "latest"
  precision: "fp8"

resources:
  gpu_type: "gb200"
  prefill_nodes: 1
  decode_nodes: 4
  prefill_workers: 1
  decode_workers: 4
  gpus_per_node: 4

slurm:
  account: "your-account"
  partition: "batch"
  time_limit: "04:00:00"

backend:
  prefill_environment:
    TORCH_DISTRIBUTED_DEFAULT_TIMEOUT: "1800"

  sglang_config:
    prefill:
      kv-cache-dtype: "fp8_e4m3"
      mem-fraction-static: 0.95
      tensor-parallel-size: 4
    decode:
      kv-cache-dtype: "fp8_e4m3"
      mem-fraction-static: 0.92
      tensor-parallel-size: 4

benchmark:
  type: "sa-bench"
  isl: 1024
  osl: 1024
  concurrencies: [256, 512, 1024]
  req_rate: "inf"
```

## Config Sections

| Section | Required | Description |
|---------|----------|-------------|
| `name` | Yes | Job identifier |
| `model` | Yes | Model path, container, precision |
| `resources` | Yes | Node/worker allocation |
| `slurm` | No | Override cluster defaults |
| `backend` | No | SGLang flags and environment |
| `benchmark` | Yes | Benchmark type and parameters |

## Model Section

```yaml
model:
  path: "deepseek-r1"      # Alias from srtslurm.yaml or full path
  container: "latest"       # Alias or full path to .sqsh file
  precision: "fp8"          # fp4, fp8, fp16, bf16
```

## Resources Section

### Disaggregated Mode

Separate prefill and decode workers:

```yaml
resources:
  gpu_type: "gb200"         # gb200, gb300, h100
  prefill_nodes: 1
  decode_nodes: 4
  prefill_workers: 1        # Workers per prefill allocation
  decode_workers: 4         # Workers per decode allocation
  gpus_per_node: 4
```

### Aggregated Mode

Combined prefill+decode workers:

```yaml
resources:
  gpu_type: "gb200"
  agg_nodes: 2
  agg_workers: 2
  gpus_per_node: 4
```

## SLURM Section

Override cluster defaults:

```yaml
slurm:
  account: "your-account"
  partition: "batch"
  time_limit: "04:00:00"
```

## Backend Section

SGLang-specific configuration:

```yaml
backend:
  enable_multiple_frontends: true
  num_additional_frontends: 9
  enable_profiling: false

  prefill_environment:
    TORCH_DISTRIBUTED_DEFAULT_TIMEOUT: "1800"
    NCCL_MNNVL_ENABLE: "1"

  decode_environment:
    TORCH_DISTRIBUTED_DEFAULT_TIMEOUT: "1800"

  sglang_config:
    prefill:
      kv-cache-dtype: "fp8_e4m3"
      mem-fraction-static: 0.95
      tensor-parallel-size: 4
      attention-backend: "trtllm_mla"
      disable-radix-cache: true
    decode:
      kv-cache-dtype: "fp8_e4m3"
      mem-fraction-static: 0.92
      tensor-parallel-size: 4
      dp-size: 32
      enable-dp-attention: true
```

### Common SGLang Flags

| Flag | Description |
|------|-------------|
| `tensor-parallel-size` | GPUs per worker for tensor parallelism |
| `dp-size` | Data parallelism degree |
| `mem-fraction-static` | GPU memory fraction for KV cache (0.0-1.0) |
| `kv-cache-dtype` | KV cache precision (fp8_e4m3, fp16, etc.) |
| `attention-backend` | Attention implementation |
| `quantization` | Model quantization (fp8, fp4, etc.) |

## Benchmark Section

```yaml
benchmark:
  type: "sa-bench"          # sa-bench, gpqa, mmlu, manual
  isl: 1024                 # Input sequence length
  osl: 1024                 # Output sequence length
  concurrencies: [256, 512] # Concurrency levels to test
  req_rate: "inf"           # Request rate (inf = max throughput)
```

### Benchmark Types

| Type | Description |
|------|-------------|
| `sa-bench` | Synthetic throughput/latency benchmark |
| `gpqa` | GPQA evaluation benchmark |
| `mmlu` | MMLU evaluation benchmark |
| `manual` | No benchmark, keep workers running |

## Dry-Run

Always validate before submitting:

```bash
srtctl config.yaml --dry-run
```

This generates all artifacts in `dry-runs/` without submitting to SLURM.
