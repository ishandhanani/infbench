# Monitoring Jobs

This guide covers how to monitor running jobs and understand the log structure produced by srtctl.

## Checking Job Status

### SLURM Commands

```bash
# List your running jobs
squeue -u $USER

# Detailed job info
scontrol show job <job_id>

# Cancel a job
scancel <job_id>
```

### Watching Logs

After submission, `srtctl` tells you where logs are stored:
```
Submitted batch job 4459
Logs: logs/4459_4P_1D_20251122_041341/
```

Watch the main log in real-time:
```bash
tail -f logs/4459_4P_1D_20251122_041341/log.out
```

## Log Directory Structure

Each job creates a directory with this naming pattern:
```
logs/{job_id}_{prefill}P_{decode}D_{timestamp}/
```

For example: `logs/4459_4P_1D_20251122_041341/` means:
- Job ID: 4459
- 4 Prefill workers, 1 Decode worker
- Started: Nov 22, 2025 at 04:13:41

### Directory Contents

```
logs/4459_4P_1D_20251122_041341/
├── config.yaml                    # Resolved job configuration
├── sglang_config.yaml             # SGLang worker configuration
├── sbatch_script.sh               # Generated SLURM script
├── 4459.json                      # Job metadata
├── nginx.conf                     # Load balancer configuration
│
├── log.out                        # Main job stdout
├── log.err                        # Main job stderr
├── benchmark.out                  # Benchmark results stdout
├── benchmark.err                  # Benchmark stderr
│
├── {node}_prefill_w{n}.out        # Prefill worker stdout
├── {node}_prefill_w{n}.err        # Prefill worker stderr
├── {node}_decode_w{n}.out         # Decode worker stdout
├── {node}_decode_w{n}.err         # Decode worker stderr
├── {node}_frontend_{n}.out        # Frontend stdout
├── {node}_frontend_{n}.err        # Frontend stderr
├── {node}_nginx.out               # Nginx stdout
├── {node}_nginx.err               # Nginx stderr
│
├── {node}_config.json             # Per-node SGLang config dump
│
├── cached_assets/                 # Cached model assets
└── sa-bench_isl_{isl}_osl_{osl}/  # Benchmark results directory
```

## Understanding Key Log Files

### log.out - Main Job Log

Shows the orchestration process:

```
Node 0: watchtower-aqua-cn01
Node 1: watchtower-aqua-cn02
Node 2: watchtower-aqua-cn03
...
Master IP address (node 1): 10.30.1.49
Nginx node (node 0): watchtower-aqua-cn01
Additional frontend 1 on node 2: watchtower-aqua-cn03 (10.30.1.106)
...
Prefill worker 0 leader: watchtower-aqua-cn01 (10.30.1.163)
Launching prefill worker 0, node 0 (local_rank 0): watchtower-aqua-cn01
...
Decode worker 0 leader: watchtower-aqua-cn05 (10.30.1.153)
Launching decode worker 0, node 4 (local_rank 0): watchtower-aqua-cn05
...
Frontend available at: http://watchtower-aqua-cn01:8000
```

Key information:
- Node assignments
- IP addresses for debugging
- Worker launch commands
- Frontend URL for manual testing

### benchmark.out - Benchmark Results

Shows benchmark progress and results:

```
Polling http://localhost:8000/health every 5 seconds...
Model is not ready, waiting for 4 prefills and 1 decodes to spin up.
Model is ready.

Warming up model with concurrency 128
============ Serving Benchmark Result ============
Successful requests:                     640
Benchmark duration (s):                  93.97
Request throughput (req/s):              6.81
Output token throughput (tok/s):         6278.02
---------------Time to First Token----------------
Mean TTFT (ms):                          1924.07
Median TTFT (ms):                        342.39
P99 TTFT (ms):                           13652.77
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          16.78
Median TPOT (ms):                        15.48
P99 TPOT (ms):                           22.36
==================================================
```

Key metrics:
- **Request throughput** - Requests per second
- **Output token throughput** - Tokens generated per second
- **TTFT** - Time to first token (latency to start generating)
- **TPOT** - Time per output token (generation speed)
- **ITL** - Inter-token latency
- **E2EL** - End-to-end latency

### Worker Logs

Worker logs (`{node}_prefill_w0.err`, `{node}_decode_w0.err`) show SGLang worker initialization and runtime logs. Useful for debugging:

- Model loading progress
- Memory allocation
- CUDA errors
- NCCL communication issues

### config.yaml - Resolved Configuration

The fully resolved configuration with all defaults applied. Useful to verify what actually ran.

## Benchmark Results Directory

Benchmark JSON files are saved in a subdirectory:
```
sa-bench_isl_1024_osl_1024/
├── isl_1024_osl_1024_concurrency_128_req_rate_inf.json
├── isl_1024_osl_1024_concurrency_512_req_rate_inf.json
└── ...
```

Each JSON file contains detailed per-request metrics for analysis.

## Common Monitoring Workflows

### Check if workers are ready

```bash
# Look for "Model is ready" in benchmark output
grep -i "ready" logs/4459_*/benchmark.out

# Or check health endpoint (from log.out, find frontend URL)
curl http://<frontend-node>:8000/health
```

### Debug a failing worker

```bash
# Find which worker failed
grep -l "Error\|Exception\|CUDA" logs/4459_*/*.err

# Check specific worker
cat logs/4459_*/watchtower-aqua-cn05_decode_w0.err | tail -100
```

### Compare benchmark runs

```bash
# Extract throughput from multiple runs
for dir in logs/*/; do
  echo "=== $dir ==="
  grep "Output token throughput" "$dir/benchmark.out"
done
```

## Connecting to Running Jobs

The log.out file includes commands to connect to running nodes:

```bash
# Connect to nginx node
srun --jobid 4459 -w watchtower-aqua-cn01 --overlap --pty bash

# Connect to master node (NATS/ETCD)
srun --jobid 4459 -w watchtower-aqua-cn02 --overlap --pty bash
```

From inside the container, you can:
- Check process status with `ps aux`
- Test endpoints with `curl`
- Inspect GPU memory with `nvidia-smi`

## Cleaning Up

```bash
# Cancel running job
scancel 4459

# Archive old logs
tar -czvf logs_archive.tar.gz logs/

# Remove old logs
rm -rf logs/old_job_*/
```
