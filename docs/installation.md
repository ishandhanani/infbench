# Installation

This guide walks through setting up `srtctl` on a SLURM cluster, configuring model paths and containers, and submitting your first job.

## Prerequisites

- Access to a SLURM cluster with GPU nodes
- Python 3.10+
- Container runtime (enroot/pyxis) configured on the cluster
- Model weights accessible from compute nodes
- SGLang container image (`.sqsh` format)

## Step 1: Clone and Install

```bash
git clone https://github.com/your-org/srtctl.git
cd srtctl

# Option A: pip install (recommended for SLURM clusters)
pip install -e .

# Option B: uv (if available on your cluster)
uv sync
```

**Why pip?** Many SLURM clusters restrict installing binaries like `uv` in user space. Using `pip` ensures compatibility across environments.

## Step 2: Run Setup

```bash
make setup
```

This creates `srtslurm.yaml` in your current directory - your cluster configuration file. If the file already exists, it won't be overwritten.

## Step 3: Configure Cluster Defaults

Edit `srtslurm.yaml` with your cluster-specific settings:

```yaml
cluster:
  # SLURM settings
  account: "your-slurm-account"
  partition: "gpu-batch"
  network_interface: "enP6p9s0np0"  # High-speed network interface
  gpus_per_node: 4
  default_time_limit: "4:00:00"
  default_container: "/path/to/default/container.sqsh"

# Model path aliases - map short names to full paths
model_paths:
  deepseek-r1: "/mnt/models/DeepSeek-R1"
  deepseek-r1-fp4: "/mnt/models/deepseek-r1-0528-fp4-v2"
  llama-70b: "/mnt/models/Llama-3-70B"

# Container aliases - map versions to sqsh files
containers:
  latest: "/mnt/containers/sglang-v0.5.5.sqsh"
  stable: "/mnt/containers/sglang-v0.5.4.sqsh"
  nightly: "/mnt/containers/sglang-nightly.sqsh"
```

### Finding Your Settings

**SLURM Account and Partition:**
```bash
# List available accounts
sacctmgr show associations user=$USER

# List partitions
sinfo -s
```

**Network Interface:**
```bash
# On a compute node, find the high-speed network
ip link show | grep -E "^[0-9]+:"
# Look for interfaces like enP6p9s0np0, ib0, etc.
```

**GPUs per Node:**
```bash
# Check GPU configuration
sinfo -N -l | head -20
# Or on a compute node:
nvidia-smi -L | wc -l
```

## Step 4: Set Up Model Paths

Models must be accessible from all compute nodes. Typically this means:

1. **Shared filesystem** (Lustre, GPFS, NFS):
   ```yaml
   model_paths:
     deepseek-r1: "/mnt/lustre/models/DeepSeek-R1"
   ```

2. **Local node storage** (if models are pre-cached):
   ```yaml
   model_paths:
     deepseek-r1: "/local/models/DeepSeek-R1"
   ```

The model path is mounted into containers at `/model/` automatically.

### Model Path Structure

Your model directory should contain the standard HuggingFace structure:
```
/mnt/models/DeepSeek-R1/
├── config.json
├── tokenizer.json
├── tokenizer_config.json
├── model-00001-of-00XXX.safetensors
├── model-00002-of-00XXX.safetensors
└── ...
```

## Step 5: Set Up Containers

`srtctl` uses Squashfs container images (`.sqsh` files). These are typically created from Docker images:

```bash
# Convert Docker image to sqsh (run once)
enroot import docker://lmsysorg/sglang:v0.5.5
enroot create lmsysorg+sglang+v0.5.5.sqsh
mv lmsysorg+sglang+v0.5.5.sqsh /mnt/containers/
```

Add the container to your aliases:
```yaml
containers:
  latest: "/mnt/containers/lmsysorg+sglang+v0.5.5.sqsh"
```

## Step 6: Create Your First Job Config

Create `configs/my-first-job.yaml`:

```yaml
name: "test-benchmark"

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
  account: "your-account"   # Or omit to use cluster default
  partition: "gpu-batch"    # Or omit to use cluster default
  time_limit: "02:00:00"

benchmark:
  type: "sa-bench"
  isl: 1024
  osl: 1024
  concurrencies: [256, 512]
```

## Step 7: Validate with Dry Run

Always validate before submitting:

```bash
srtctl configs/my-first-job.yaml --dry-run
```

This:
- Validates your YAML configuration
- Resolves all aliases to full paths
- Generates the SLURM script and SGLang config
- Saves everything to `dry-runs/` for inspection
- Does NOT submit to SLURM

Check the generated files:
```bash
ls dry-runs/test-benchmark_*/
# sbatch_script.sh    - The SLURM job script
# config.yaml         - Resolved configuration
# sglang_config.yaml  - SGLang worker configuration
```

## Step 8: Submit the Job

```bash
srtctl configs/my-first-job.yaml
```

Output:
```
Submitted batch job 12345
Logs: logs/12345_1P_4D_20251122_143052/
```

## Step 9: Monitor Your Job

```bash
# Check job status
squeue -u $USER

# Watch logs
tail -f logs/12345_*/log.out

# Check benchmark progress
tail -f logs/12345_*/benchmark.out
```

See [Monitoring](monitoring.md) for detailed log structure documentation.

## Troubleshooting

### Job fails immediately

Check SLURM output:
```bash
cat logs/12345_*/log.err
```

Common issues:
- Invalid account/partition
- Container not found
- Model path not accessible

### Workers fail to start

Check worker logs:
```bash
cat logs/12345_*/*_prefill_w0.err
cat logs/12345_*/*_decode_w0.err
```

Common issues:
- Out of GPU memory (reduce `mem-fraction-static`)
- Network interface mismatch
- NCCL initialization failures

### Benchmark hangs

Check if workers are healthy:
```bash
# From the log output, find the frontend URL
curl http://<frontend-node>:8000/health
```

### Container issues

Verify container exists and is readable:
```bash
ls -la /path/to/container.sqsh
# Should be readable by your user
```

## Next Steps

- [Configuration](configuration.md) - Full reference for all config options
- [Monitoring](monitoring.md) - Understanding logs and debugging
- [Parameter Sweeps](sweeps.md) - Run multiple configurations
