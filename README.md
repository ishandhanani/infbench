# srtctl

YAML-based job submission toolkit for distributed LLM inference on SLURM clusters.

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
uv run srtctl configs/gb200_fp4_max_tpt.yaml

# Dry-run mode (validate without submitting)
uv run srtctl configs/gb200_fp4_max_tpt.yaml --dry-run
```

Example YAML config:

```yaml
name: "gb200-fp4-max-tpt"

model:
  path: "/path/to/model"
  container: "your-container.sqsh"
  precision: "fp4"  # fp4, fp8, fp16, bf16

resources:
  gpu_type: "gb200"  # gb200, h100
  prefill_nodes: 1
  decode_nodes: 12
  prefill_workers: 1
  decode_workers: 1
  gpus_per_node: 4

slurm:
  account: "your-account"
  partition: "batch"
  time_limit: "4:00:00"

backend:
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
```

Logs saved to `logs/{JOB_ID}_{P}P_{D}D_{TIMESTAMP}/` (or path specified in `srtslurm.yaml`)

### 3. Parameter Sweeps

Run grid searches over multiple configurations:

```bash
# Submit parameter sweep
uv run srtctl configs/example-sweep.yaml --sweep

# Dry-run sweep (generate all configs without submitting)
uv run srtctl configs/example-sweep.yaml --sweep --dry-run
```

See `configs/example-sweep.yaml` for sweep syntax with template placeholders.

### 4. Monitor Jobs

```bash
# Check job status
squeue -u $USER

# View logs
tail -f logs/{JOB_ID}_*/slurm-*.out

# Cancel job
scancel {JOB_ID}
```

## Features

### 🚀 YAML-Based Job Submission

- **Declarative configuration** - Define jobs in clean YAML files with validation
- **Disaggregated or aggregated modes** - Support for prefill/decode separation or combined workers
- **SGLang config generation** - No more managing 50+ CLI flags manually
- **Parameter sweeps** - Grid search over multiple configurations with Cartesian product
- **Dry-run mode** - Validate configs and preview generated commands without submitting
- **Multiple frontends** - Nginx load balancing across frontend workers
- **Template expansion** - Use `{param}` placeholders for sweep parameters
- **Cluster defaults** - Define account, partition, container aliases in `srtslurm.yaml`

## Configuration

### Cluster Defaults (`srtslurm.yaml`)

Created by `make setup`:

```yaml
cluster:
  account: "your-account"
  partition: "batch"
  network_interface: "enP6p9s0np0"  # Network interface for multi-node communication
  gpus_per_node: 4
  default_time_limit: "4:00:00"
  default_container: "/path/to/container.sqsh"

# Optional: Model path aliases (use short names in job configs)
model_paths:
  deepseek-r1: "/models/deepseek-r1"
  llama-3-70b: "/models/llama-3-70b"

# Optional: Container aliases (use short names in job configs)
containers:
  latest: "/containers/sglang-latest.sqsh"
  stable: "/containers/sglang-stable.sqsh"

# Optional: Override log directory (defaults to ./logs)
srtctl_root: "/path/to/srtctl"  # Logs will go to {srtctl_root}/logs/
```

### Job Configuration

Each job is defined in a YAML file. See `configs/example.yaml` for a minimal template and `configs/gb200_fp8_1p_4d.yaml` for a complete example.

**Key Concepts:**
- **Model paths/containers**: Use aliases from `srtslurm.yaml` or full paths
- **Disaggregated mode**: Separate prefill and decode workers (`prefill_nodes`, `decode_nodes`)
- **Aggregated mode**: Combined workers (`agg_nodes`, `agg_workers`)
- **SGLang config**: Define flags under `backend.sglang_config.prefill` and `backend.sglang_config.decode`
- **Environment variables**: Set per-worker-type under `backend.prefill_environment` and `backend.decode_environment`
- **Benchmarks**: Configure sa-bench, MMLU, GPQA under `benchmark` section

## Repository Structure

```
srtctl/
├── src/srtctl/          # Python package (submission logic)
│   ├── cli/             # CLI entrypoints (submit.py)
│   ├── backends/        # Backend implementations (SGLang)
│   └── core/            # Config loading, validation, schema
├── scripts/             # Runtime scripts executed on SLURM nodes
│   ├── templates/       # Jinja2 SLURM job templates
│   ├── benchmarks/      # Benchmark scripts (sa-bench, mmlu, gpqa)
│   ├── utils/           # Shared utilities
│   └── worker_setup/    # Worker initialization logic
├── configs/             # Example job configuration files
│   ├── example.yaml           # Minimal starter template
│   ├── example-sweep.yaml     # Parameter sweep example
│   └── gb200_fp8_*.yaml       # Production configs
├── tests/               # Unit tests
├── logs/                # Job outputs (created by submissions)
└── srtslurm.yaml        # Cluster config (gitignored, created by setup)
```

## Understanding Job Outputs

After submission, job artifacts are saved to `logs/{JOB_ID}_{CONFIG}_{TIMESTAMP}/`:

```
logs/12345_1P_4D_20250119_120000/
├── slurm-12345.out           # SLURM output log
├── config.yaml               # Resolved config (with defaults applied)
├── sglang_config.yaml        # Generated SGLang flags
├── sbatch_script.sh          # Generated SLURM script
└── jobid.json                # Metadata (job info, resources, benchmark config)
```

**Dry-run outputs** are saved to `dry-runs/{NAME}_{TIMESTAMP}/` with similar structure.

## Development

```bash
make lint        # Run linters (ruff)
make test        # Run tests (pytest)
```

## Requirements

- Python 3.10+
- uv (package manager)
- SLURM cluster with Pyxis (for container support)
- GPU nodes (tested on GB200 NVL72, H100)

## Common Workflows

### Running a quick test
```bash
# Use dry-run to validate config without submitting
uv run srtctl configs/example.yaml --dry-run

# Check generated commands
cat dry-runs/example_*/commands.sh
```

### Tuning hyperparameters
```bash
# Create sweep config with parameters to test
# See configs/example-sweep.yaml for syntax

uv run srtctl configs/my-sweep.yaml --sweep --dry-run  # Preview jobs
uv run srtctl configs/my-sweep.yaml --sweep            # Submit all
```

### Debugging failed jobs
```bash
# Check SLURM logs
tail -f logs/{JOB_ID}_*/slurm-*.out

# Review generated config
cat logs/{JOB_ID}_*/sglang_config.yaml

# Check what commands were run
cat logs/{JOB_ID}_*/sbatch_script.sh
```

## Documentation

- [scripts/README.md](scripts/README.md) - Runtime scripts and templates
- [configs/example.yaml](configs/example.yaml) - Minimal job template
- [configs/example-sweep.yaml](configs/example-sweep.yaml) - Parameter sweep template

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

Backends implement a clean protocol for framework-specific logic:
- `generate_config_file()` - Generate backend-specific configs
- `render_command()` - Render execution commands
- `generate_slurm_script()` - Create SLURM job scripts from templates

Easy to extend for new backends (vLLM, TensorRT-LLM, etc.).
