# InfBench

Benchmarking toolkit for Dynamo and SGLang on SLURM clusters with interactive analysis dashboard.

## Quick Start

### 1. Setup (One-Time)

```bash
# add ARCH when login node is aarch64 and you are trying to run on GB200
make setup ARCH=aarch64
```

This downloads dependencies (nats, etcd, dynamo wheels) and creates `srtslurm.yaml` with your cluster settings and stores them in a folder called `configs`. This is automatically mounted to the container when you submit a job and all pieces are accessible via the `/config` mount.

### 2. Run E2E Benchmark with SemiAnalysis Benchmarking Script

#### Example: Run FP4 Disagg with the `max-tpt.sh` script

In this example we submit a job with 1 prefill worker (taking up 1 node so 4 GPUs) and 1 decode worker (taking up 12 nodes so 48 GPUs) and run the `max-tpt.sh` script (which you can find in `scripts/gb200-fp4/disagg/max-tpt.sh`). After the model is loaded, we will automatically run the benchmarking script.

```bash
cd slurm_runner
python3 submit_job_script.py \
  --model-dir /path/to/model \
  --gpu-type gb200-fp4 \
  --gpus-per-node 4 \
  --prefill-nodes 1 \
  --decode-nodes 12 \
  --prefill-workers 1 \
  --decode-workers 1 \
  --script-variant max-tpt \
  --benchmark "type=sa-bench; isl=1024; osl=1024; concurrencies=1024x2048x4096; req-rate=inf"
```

Logs saved to `logs/{JOB_ID}_{P}P_{D}D_{TIMESTAMP}/`

See [slurm_runner/README.md](slurm_runner/README.md) for detailed options.

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

### 🚀 SLURM Job Submission

- Disaggregated (prefill/decode) or aggregated mode
- Multiple frontends with nginx load balancing (default)
- Automated benchmarking with sa-bench
- Job metadata tracking

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

See **Cloud Storage Sync** section below for details.

## Configuration

All defaults in `srtslurm.yaml` (created by `make setup`):

```yaml
cluster:
  account: "your-account"
  partition: "batch"
  network_interface: "enP6p9s0np0"
  time_limit: "4:00:00"
  container_image: "/path/to/container.sqsh"

cloud:
  endpoint_url: "" # Optional
  bucket: ""
  prefix: "benchmark-results/"
```

Override any setting via CLI flags.

## Repository Structure

```
infbench/
├── dashboard/           # Streamlit UI (modular tabs)
├── srtslurm/           # Core analysis library
├── slurm_runner/       # SLURM job submission scripts
├── logs/               # Benchmark results
├── configs/            # Dynamo dependencies (nats, etcd, wheels)
├── tests/              # Unit tests
└── srtslurm.yaml       # Configuration (gitignored)
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
uv run python slurm_runner/scripts/sync_results.py pull-missing
uv run python slurm_runner/scripts/sync_results.py list-remote
```

## Development

```bash
make lint        # Run linters
make test        # Run tests
make dashboard   # Launch dashboard
```

## Requirements

- Python 3.10+ (stdlib yaml support)
- SLURM cluster with Pyxis (for container support)
- GPU nodes (tested on GB200 NVL72)

For detailed SLURM job submission docs, see [slurm_runner/README.md](slurm_runner/README.md).
