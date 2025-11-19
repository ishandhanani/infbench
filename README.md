# InfBench

Benchmarking toolkit for distributed LLM inference on SLURM clusters with YAML-based configuration and interactive analysis dashboard.

## Quick Start

### 1. Setup (One-Time)

```bash
# Install dependencies and create cluster config
make setup

# For GB200 clusters (aarch64)
make setup ARCH=aarch64
```

This downloads dependencies (nats, etcd, dynamo wheels) and creates `srtslurm.yaml` with your cluster settings.

### 2. Submit a Benchmark Job

Using the new YAML-based submission:

```bash
# Submit from YAML config
uv run infbench configs/gb200_fp4_max_tpt.yaml

# Dry-run mode (validate without submitting)
uv run infbench configs/gb200_fp4_max_tpt.yaml --dry-run
```

Example YAML config:

```yaml
name: "gb200-fp4-max-tpt"

model:
  path: "/path/to/model"
  container: "your-container.sqsh"

resources:
  prefill_nodes: 1
  decode_nodes: 12
  prefill_workers: 1
  decode_workers: 1
  gpus_per_node: 4

backend:
  type: "sglang"
  gpu_type: "gb200-fp4"
  script_variant: "max-tpt"

  sglang_config:
    prefill:
      kv_cache_dtype: "fp8_e4m3"
      mem_fraction_static: 0.84
      # ... SGLang flags
    decode:
      kv_cache_dtype: "fp8_e4m3"
      mem_fraction_static: 0.82
      # ... SGLang flags

benchmark:
  type: "sa-bench"
  isl: 1024
  osl: 1024
  concurrencies: [1024, 2048, 4096]
  req_rate: "inf"

slurm:
  account: "your-account"
  partition: "batch"
  time_limit: "4:00:00"
```

Logs saved to `../infbench/logs/{JOB_ID}_{P}P_{D}D_{TIMESTAMP}/`

### 3. Analyze Results

```bash
uv run streamlit run dashboard/app.py
```

Opens interactive dashboard at http://localhost:8501

## Features

### 📊 Interactive Dashboard

- **Pareto Analysis** - TPS/GPU vs TPS/User tradeoffs
- **Latency Breakdown** - TTFT, TPOT, ITL across concurrency levels
- **Node Metrics** - Runtime metrics from prefill/decode nodes
- **Config Comparison** - Side-by-side configuration diffs
- **Run Comparison** - Performance deltas between runs

### 🚀 YAML-Based Job Submission

- Declarative configuration with validation
- Support for disaggregated (prefill/decode) or aggregated mode
- SGLang config generation (no more 50+ CLI flags!)
- Parameter sweeping for grid searches
- Dry-run mode for validation
- Multiple frontends with nginx load balancing
- Template expansion for environment variables

### ☁️ Cloud Sync (Optional)

Sync benchmark results to S3-compatible storage:

```bash
# Install dependency
pip install boto3

# Configure in srtslurm.yaml
cloud:
  endpoint_url: "https://s3.your-cloud.com"
  bucket: "benchmark-results"
  prefix: "runs/"

# Push results
./push_after_benchmark.sh

# Dashboard auto-pulls missing runs
```

## Configuration

### Cluster Defaults (`srtslurm.yaml`)

Created by `make setup`:

```yaml
cluster:
  account: "your-account"
  partition: "batch"
  network_interface: "enP6p9s0np0"
  gpus_per_node: 4
  default_time_limit: "4:00:00"
  default_container: "/path/to/container.sqsh"

# Model path aliases
model_paths:
  deepseek-r1: "/models/deepseek-r1"
  llama-3-70b: "/models/llama-3-70b"

# Container aliases
containers:
  latest: "/containers/sglang-latest.sqsh"
  stable: "/containers/sglang-stable.sqsh"

cloud:
  endpoint_url: ""  # Optional
  bucket: ""
  prefix: "benchmark-results/"
```

### Job Configuration

Each job is defined in a YAML file. See `configs/gb200_fp4_max_tpt.yaml` for a complete example.

Override cluster defaults in your job config or use CLI flags.

## Repository Structure

```
infbench-yaml-config/
├── src/infbench/        # Python package (submission logic)
│   ├── cli/             # CLI entrypoints (submit.py)
│   ├── backends/        # Backend implementations (SGLang, etc.)
│   └── core/            # Config loading and validation
├── scripts/             # Runtime scripts for SLURM jobs
│   ├── templates/       # Jinja2 templates
│   ├── benchmarks/      # Benchmark scripts (sa-bench, gpqa, mmlu)
│   ├── profiling/       # Profiling utilities
│   ├── utils/           # Shared utilities
│   ├── legacy/          # GPU-specific legacy configs
│   └── worker_setup.py  # Main worker launcher
├── configs/             # Job configuration YAML files
├── dashboard/           # Streamlit UI (modular tabs)
├── srtslurm/           # Core analysis library
├── tests/              # Unit tests
└── srtslurm.yaml       # Cluster config (gitignored)

../infbench/            # Shared with main infbench repo
└── logs/               # Benchmark results
```

## Key Metrics

- **Output TPS/GPU** - Token generation throughput per GPU (efficiency)
- **Output TPS/User** - Tokens per second per concurrent user (responsiveness)
- **TTFT** - Time to first token (perceived latency)
- **TPOT** - Time per output token (streaming speed)
- **ITL** - Inter-token latency (includes queueing)

## Cloud Storage Sync

### Setup

1. Install boto3: `pip install boto3`
2. Add cloud settings to `srtslurm.yaml` (see above)
3. Set credentials:
   ```bash
   export AWS_ACCESS_KEY_ID="your-key"
   export AWS_SECRET_ACCESS_KEY="your-secret"
   ```

### Usage

**Push from cluster:**

```bash
./push_after_benchmark.sh                    # Push all runs
./push_after_benchmark.sh --log-dir /path    # Specify directory
./push_after_benchmark.sh 3667_1P_12D_...   # Push single run
```

**Pull locally:**
Dashboard auto-syncs missing runs on startup. Or manually:

```bash
uv run python scripts/sync_results.py pull-missing
uv run python scripts/sync_results.py list-remote
```

## Development

```bash
make lint        # Run linters
make test        # Run tests
make dashboard   # Launch dashboard
```

## Requirements

- Python 3.10+
- uv (package manager)
- SLURM cluster with Pyxis (for container support)
- GPU nodes (tested on GB200 NVL72, H100)

## Documentation

- [scripts/README.md](scripts/README.md) - Runtime scripts and templates
- [configs/](configs/) - Example job configurations

## Architecture

### Submission Flow

1. Load YAML config → Apply cluster defaults
2. Validate configuration
3. Create backend instance (e.g., SGLangBackend)
4. Generate SGLang config file
5. Render SLURM job script from Jinja template
6. Submit to SLURM with `sbatch`
7. Save artifacts to logs directory

### Backend Protocol

Inspired by ignition's design, backends implement:
- `generate_config_file()` - Generate backend-specific configs
- `render_command()` - Render execution commands
- `generate_slurm_script()` - Create SLURM job scripts from templates

Easy to extend for new backends (vLLM, TensorRT-LLM, etc.).
