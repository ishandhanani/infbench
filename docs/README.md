# Introduction

`srtctl` is a command-line tool for running distributed LLM inference benchmarks on SLURM clusters. It replaces complex shell scripts and 50+ CLI flags with clean, declarative YAML configuration files.

## Why srtctl?

Running large language models across multiple GPUs and nodes requires orchestrating many moving parts: SLURM job scripts, container mounts, SGLang configuration, worker coordination, and benchmark execution. Traditionally, this meant maintaining brittle bash scripts with hardcoded parameters.

`srtctl` solves this by:

- **Declarative configuration** - Define your entire job in a single YAML file
- **Validation** - Catch configuration errors before submitting to SLURM
- **Reproducibility** - Every job saves its full configuration for later reference
- **Parameter sweeps** - Run grid searches across configurations with a single command

## Architecture Overview

`srtctl` orchestrates distributed inference using SGLang workers in either **disaggregated** or **aggregated** mode:

**Disaggregated Mode** separates prefill and decode into specialized workers:
- Prefill workers handle the initial prompt processing
- Decode workers handle token generation
- An nginx load balancer distributes requests across frontends

**Aggregated Mode** runs combined prefill+decode on each worker, simpler but potentially less efficient for high-throughput scenarios.

## Core Concepts

### Job Configuration

Every job is defined in a YAML file with these sections:

| Section | Purpose |
|---------|---------|
| `name` | Job identifier used in logs and SLURM |
| `model` | Model path, container, and precision |
| `resources` | Node allocation and worker counts |
| `slurm` | Account, partition, time limits |
| `backend` | SGLang-specific flags and environment |
| `benchmark` | Benchmark type and parameters |

### Cluster Configuration

Cluster-wide defaults live in `srtslurm.yaml`:
- Default SLURM account and partition
- Model path aliases (e.g., `deepseek-r1` -> `/models/DeepSeek-R1`)
- Container aliases (e.g., `latest` -> `/containers/sglang-v0.5.5.sqsh`)

This means job configs can use short aliases instead of full paths.

### What Happens When You Submit

When you run `srtctl config.yaml`, the tool validates your configuration, resolves any aliases from your cluster config, generates a SLURM batch script and SGLang configuration files, then submits to SLURM. Once allocated, workers launch inside containers, discover each other through ETCD and NATS, and begin serving. If you've configured a benchmark, it runs automatically against the serving endpoint and saves results to the log directory.

## Quick Example

```yaml
name: "my-benchmark"

model:
  path: "deepseek-r1"
  container: "latest"
  precision: "fp8"

resources:
  gpu_type: "gb200"
  prefill_nodes: 1
  decode_nodes: 4

benchmark:
  type: "sa-bench"
  isl: 1024
  osl: 1024
  concurrencies: [256, 512, 1024]
```

```bash
# Validate without submitting
srtctl config.yaml --dry-run

# Submit to SLURM
srtctl config.yaml
```

## Next Steps

- [Installation](installation.md) - Set up `srtctl` on your cluster
- [Configuration](configuration.md) - Full configuration reference
- [Monitoring](monitoring.md) - Understanding job logs and status
- [Parameter Sweeps](sweeps.md) - Run grid searches
- [Analysis](analysis.md) - Visualize results with the UI
