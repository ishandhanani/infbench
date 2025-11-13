# Parameter Sweep System

This system enables automated parameter sweeping for SGLang performance benchmarking by generating cartesian products of configurations and submitting parallel SLURM jobs.

## Overview

The sweep system consists of:

- **`sweep_jobs.py`**: Main orchestrator that reads YAML configs and submits jobs
- **YAML sweep config**: Defines parameter arrays to sweep over
- **Automatic config generation**: Creates individual SGLang YAML configs per job
- **Sweep manifest**: Tracks all submitted jobs and their parameters

## Quick Start

### 1. Create a Sweep Config

Create a YAML file with parameter arrays (see `example_sweep_config.yaml`):

```yaml
slurm:
  prefill_nodes: [6, 12]
  decode_nodes: [12, 24]
  prefill_workers: [3]
  decode_workers: [1]

sglang:
  max-running-requests: [5632, 8000]
  context-length: [2176, 4096]
  mem-fraction-static: [0.82, 0.84]
```

### 2. Run the Sweep

```bash
python slurm_jobs/sweep_jobs.py \
  --sweep-config my_sweep_config.yaml \
  --account $SLURM_ACCOUNT \
  --partition $SLURM_PARTITION \
  --model-dir $MODEL_PATH \
  --config-dir $CONFIG_DIR \
  --container-image $CONTAINER_IMAGE \
  --gpu-type gb200-fp8 \
  --script-variant max-tpt
```

### 3. Dry Run (Optional)

Test your config without submitting jobs:

```bash
python slurm_jobs/sweep_jobs.py \
  --sweep-config my_sweep_config.yaml \
  --dry-run \
  ... (other args)
```

## How It Works

### 1. Cartesian Product Generation

The system generates all possible combinations of parameters:

```
Example: 2 prefill_nodes × 2 max_running_requests × 2 mem_fraction_static
       = 8 total job submissions
```

### 2. SGLang Config Generation

For each combination with `sglang` parameters, a YAML config is created.

**Disaggregated mode** creates nested YAML with `prefill` and `decode` keys:

```yaml
# sglang_config_0.yaml (disaggregated)
prefill:
  max-running-requests: 5632
  context-length: 2176
  mem-fraction-static: 0.82
decode:
  max-running-requests: 5632
  context-length: 2176
  mem-fraction-static: 0.82
```

Prefill nodes use: `--config /configs/sglang_config_0.yaml --config-key prefill`  
Decode nodes use: `--config /configs/sglang_config_0.yaml --config-key decode`

**Aggregated mode** creates flat YAML:

```yaml
# sglang_config_0.yaml (aggregated)
max-running-requests: 5632
context-length: 2176
mem-fraction-static: 0.82
```

This is passed to `dynamo.sglang` via `--config /configs/sglang_config_0.yaml`

### 3. Job Submission

Each combination submits via `submit_job_script.py`:

- SLURM params are passed as CLI args
- SGLang config path is passed via `--sglang-config`

### 4. Tracking

A sweep manifest is created:

```json
{
  "sweep_timestamp": "2025-11-13 12:34:56",
  "total_jobs": 8,
  "jobs": [
    {
      "job_id": "12345",
      "parameters": {
        "slurm": {"prefill_nodes": 6, ...},
        "sglang": {"max-running-requests": 5632, ...}
      }
    },
    ...
  ]
}
```

## Configuration Reference

### SLURM Parameters

Sweep over cluster configuration:

```yaml
slurm:
  # Disaggregated mode
  prefill_nodes: [6, 12]
  decode_nodes: [12, 24]
  prefill_workers: [3, 6]
  decode_workers: [1, 2]

  # OR aggregated mode
  agg_nodes: [12, 24]
  agg_workers: [3, 6]
```

### SGLang Parameters

Any `dynamo.sglang` CLI flag can be swept.

**Important for Disaggregated Mode**:

- The sweep system automatically creates nested YAML with `prefill` and `decode` keys
- Parameters in your sweep config are applied to BOTH prefill and decode workers
- If you need DIFFERENT parameters for prefill vs decode, create a manual config (see `example_manual_disagg_config.yaml`) and use `--sglang-config` directly with `submit_job_script.py` instead of the sweep system

Examples:

```yaml
sglang:
  # Memory settings
  max-running-requests: [5632, 8000, 10000]
  max-total-tokens: [131072, 262144, 524288]
  mem-fraction-static: [0.75, 0.80, 0.82, 0.84]

  # Context/prefill settings
  context-length: [2176, 4096, 8192]
  chunked-prefill-size: [131072, 262144]
  max-prefill-tokens: [32768, 65536]

  # Decode settings
  cuda-graph-bs: [[1, 2, 4, 8, 16], [1, 2, 4, 8, 16, 32]]
  decode-log-interval: [100, 1000]

  # Quantization/precision
  kv-cache-dtype: ["fp8_e4m3", "fp16"]
  quantization: ["fp8", "fp4", "modelopt_fp4"]

  # Backend settings
  attention-backend: ["trtllm_mla", "flashinfer"]
  moe-runner-backend: ["flashinfer_cutlass", "flashinfer_cutedsl"]
```

**Note**: The `--config` flag implementation [merged here](https://github.com/ai-dynamo/dynamo/pull/4272) supports standard YAML with CLI flag names as keys.

## Command-Line Options

### Required Arguments

```bash
--sweep-config PATH          # Path to YAML sweep configuration
--account ACCOUNT            # SLURM account
--partition PARTITION        # SLURM partition
--model-dir PATH            # Model directory path
--config-dir PATH           # Config directory path
--container-image IMAGE     # Container image
--gpu-type TYPE             # GPU type (gb200-fp8, gb200-fp4, etc.)
--script-variant VARIANT    # Script variant (max-tpt, 1p_4d, etc.)
```

### Optional Arguments

```bash
--job-name NAME             # Base job name (default: sweep_job)
--time-limit TIME           # Time limit HH:MM:SS (default: 04:00:00)
--gpus-per-node N           # GPUs per node (default: 8)
--network-interface IFACE   # Network interface (default: eth3)
--log-dir PATH              # Log directory
--output-dir PATH           # Sweep artifacts directory
--use-init-location         # Use init expert locations
--disable-config-dump       # Disable config dump
--use-dynamo-whls           # Use dynamo wheels
--enable-multiple-frontends # Enable multiple frontends
--num-additional-frontends N # Number of additional frontends
--profiler CONFIG           # Profiler configurations
--retries N                 # Number of retries per job
--dry-run                   # Print combinations without submitting
```

## Output

After running a sweep, you'll get:

```
slurm_jobs/sweep_20251113_123456/
├── sglang_config_0.yaml      # Generated configs
├── sglang_config_1.yaml
├── ...
└── sweep_manifest.json        # Job tracking manifest
```

The manifest maps job IDs to their exact parameter combinations for easy analysis.

## Example Workflows

### Throughput Optimization Sweep

Find optimal memory and concurrency settings:

```yaml
slurm:
  prefill_nodes: [6]
  decode_nodes: [12]
  prefill_workers: [3]
  decode_workers: [1]

sglang:
  max-running-requests: [5000, 5632, 6000, 7000, 8000]
  mem-fraction-static: [0.80, 0.82, 0.84, 0.86]
  max-total-tokens: [524288, 786432, 1048576]
```

### Quantization Comparison

Compare different quantization strategies:

```yaml
slurm:
  prefill_nodes: [6]
  decode_nodes: [12]
  prefill_workers: [3]
  decode_workers: [1]

sglang:
  quantization: ["fp8", "modelopt_fp4"]
  kv-cache-dtype: ["fp8_e4m3", "fp8_e5m2"]
  moe-runner-backend: ["flashinfer_cutlass", "flashinfer_cutedsl"]
```

### Scaling Study

Test scaling characteristics:

```yaml
slurm:
  prefill_nodes: [3, 6, 12, 18]
  decode_nodes: [6, 12, 24, 36]
  prefill_workers: [3]
  decode_workers: [1]

sglang:
  max-running-requests: [5632]
```

## Backward Compatibility

All existing workflows continue to work unchanged:

- Direct `submit_job_script.py` calls work as before
- `submit_disagg.sh` wrapper unchanged
- No `--sglang-config` means scripts use hardcoded defaults

The sweep system is purely additive.

## Tips

1. **Start small**: Begin with a 2×2 sweep to verify your config before scaling up
2. **Use dry-run**: Always test with `--dry-run` first
3. **Check quotas**: Large sweeps can submit many jobs - ensure you have quota
4. **Monitor manifest**: Use the manifest to track which jobs correspond to which configs
5. **Organize results**: Set `--log-dir` to keep sweep logs separate from regular runs
