# Installation

## Prerequisites

- Access to a SLURM cluster with GPU nodes
- Python 3.10+
- Container runtime (enroot/pyxis) configured on the cluster
- Model weights accessible from compute nodes
- SGLang container image (`.sqsh` format)

## Clone and Install

```bash
git clone https://github.com/your-org/srtctl.git
cd srtctl
pip install -e .
```

## Run Setup

First, check your login node architecture:
```bash
uname -m
```

This returns either `x86_64` (AMD/Intel) or `aarch64` (ARM). The setup downloads architecture-specific binaries for NATS and ETCD.

**Important:** The login node architecture sometimes differs from the compute nodes. For example, you might be on a Grace Hopper cluster with ARM compute nodes, but your login node could be x86_64. Always check with `uname -m` and use the architecture that matches your **compute nodes**.

Then run setup with your architecture:
```bash
make setup ARCH=aarch64  # For ARM systems (e.g., Grace Hopper)
make setup ARCH=x86_64   # For AMD/Intel systems
```

The setup will:
1. Download NATS and ETCD binaries for your architecture
2. Prompt you for cluster settings:
   - SLURM account (default: `restricted`)
   - SLURM partition (default: `batch`)
   - GPUs per node (default: `4`)
   - Time limit (default: `4:00:00`)
3. Create `srtslurm.yaml` with your settings

## Configure srtslurm.yaml

After setup, edit `srtslurm.yaml` to add model paths and containers:

### Adding Model Paths

The `model_paths` section maps short aliases to full filesystem paths:

```yaml
model_paths:
  deepseek-r1: "/mnt/lustre/models/DeepSeek-R1"
  deepseek-r1-fp4: "/mnt/lustre/models/deepseek-r1-0528-fp4-v2"
  llama-70b: "/mnt/lustre/models/Llama-3-70B"
```

Models must be accessible from all compute nodes (typically on a shared filesystem like Lustre or GPFS).

### Adding Containers

The `containers` section maps version aliases to `.sqsh` container images:

```yaml
containers:
  latest: "/mnt/containers/lmsysorg+sglang+v0.5.5.sqsh"
  stable: "/mnt/containers/lmsysorg+sglang+v0.5.4.sqsh"
```

To create a container image from Docker:
```bash
enroot import docker://lmsysorg/sglang:v0.5.5
mv lmsysorg+sglang+v0.5.5.sqsh /mnt/containers/
```

## Create a Job Config

Create `configs/my-job.yaml`:

```yaml
name: "my-benchmark"

model:
  path: "deepseek-r1"      # Uses alias from srtslurm.yaml
  container: "latest"       # Uses alias from srtslurm.yaml
  precision: "fp8"

resources:
  gpu_type: "gb200"
  prefill_nodes: 1
  decode_nodes: 4
  prefill_workers: 1
  decode_workers: 4
  gpus_per_node: 4

slurm:
  time_limit: "02:00:00"

backend:
  prefill_environment:
    TORCH_DISTRIBUTED_DEFAULT_TIMEOUT: "1800"
  decode_environment:
    TORCH_DISTRIBUTED_DEFAULT_TIMEOUT: "1800"
    NCCL_MNNVL_ENABLE: "1"

  sglang_config:
    prefill:
      kv-cache-dtype: "fp8_e4m3"
      mem-fraction-static: 0.84
      tensor-parallel-size: 4
    decode:
      kv-cache-dtype: "fp8_e4m3"
      mem-fraction-static: 0.83
      tensor-parallel-size: 4
      dp-size: 32

benchmark:
  type: "sa-bench"
  isl: 1024
  osl: 1024
  concurrencies: [256, 512]
```

## Validate with Dry Run

Always validate before submitting:

```bash
srtctl configs/my-job.yaml --dry-run
```

This validates your config, resolves aliases, generates all files, and saves them to `dry-runs/` without submitting to SLURM.

## Submit the Job

```bash
srtctl configs/my-job.yaml
```

Output:
```
Submitted batch job 12345
Logs: logs/12345_1P_4D_20251122_143052/
```

## Monitor Your Job

```bash
# Check job status
squeue -u $USER

# Watch logs
tail -f logs/12345_*/log.out

# Check benchmark progress
tail -f logs/12345_*/benchmark.out
```

See [Monitoring](monitoring.md) for detailed log structure.
