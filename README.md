# srtctl

YAML-based toolkit for distributed LLM inference benchmarks on SLURM clusters using Dynamo + SGLang.

## Quick Start

```bash
# One-time setup (downloads NATS/ETCD, creates srtslurm.yaml)
make setup ARCH=aarch64  # or ARCH=x86_64

# Submit a job
uv run srtctl configs/example.yaml

# Validate without submitting
uv run srtctl configs/example.yaml --dry-run

# Parameter sweep
uv run srtctl configs/example-sweep.yaml --sweep
```

## Example Config

```yaml
name: "my-benchmark"

model:
  path: "deepseek-r1" # Alias from srtslurm.yaml or full path
  container: "latest" # Container alias or full path
  precision: "fp8"

resources:
  gpu_type: "gb200"
  prefill_nodes: 1
  decode_nodes: 4
  prefill_workers: 1
  decode_workers: 4
  gpus_per_node: 4

backend:
  sglang_config:
    prefill:
      kv-cache-dtype: "fp8_e4m3"
      mem-fraction-static: 0.84
    decode:
      kv-cache-dtype: "fp8_e4m3"
      mem-fraction-static: 0.82
      dp-size: 16

benchmark:
  type: "sa-bench"
  isl: 1024
  osl: 1024
  concurrencies: [256, 512]
```

## Features

- **Declarative YAML configs** - No more managing 50+ CLI flags
- **Disaggregated/aggregated modes** - Separate or combined prefill/decode workers
- **Parameter sweeps** - Grid search with `{placeholder}` syntax
- **Dry-run validation** - Preview before submitting
- **Auto-generated SLURM scripts** - SGLang configs and job scripts generated automatically

## Configuration

**Cluster defaults** (`srtslurm.yaml`): Created by `make setup`. Add model/container aliases here.

**Job configs**: See `examples/example.yaml` for full template. Key sections:

- `model` - Path, container, precision
- `resources` - GPU type, node/worker counts
- `backend.sglang_config` - SGLang flags (kebab-case)
- `benchmark` - sa-bench, MMLU, or GPQA settings

## Monitoring

```bash
squeue -u $USER                                    # Check status
tail -f logs/{JOB_ID}_*/log.out                    # Main log
tail -f logs/{JOB_ID}_*/benchmark.out              # Benchmark results
```

Logs saved to `logs/{JOB_ID}_{P}P_{D}D_{TIMESTAMP}/` with configs, scripts, and worker outputs.

## Documentation

- [Installation](docs/installation.md) - Setup and first job
- [Monitoring](docs/monitoring.md) - Understanding logs and debugging
- [Parameter Sweeps](docs/sweeps.md) - Grid search workflows
- [Examples](examples/) - Config templates

## Requirements

- Python 3.10+, `uv` package manager
- SLURM cluster with Pyxis (container support)
- GPU nodes (tested on GB200, H100)
