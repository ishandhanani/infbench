# Monitoring

## Checking Job Status

```bash
# List your running jobs
squeue -u $USER

# Detailed job info
scontrol show job <job_id>

# Cancel a job
scancel <job_id>
```

## Log Directory

After submission, `srtctl` tells you where logs are stored:
```
Submitted batch job 4459
Logs: logs/4459_4P_1D_20251122_041341/
```

The directory name follows the pattern: `{job_id}_{prefill}P_{decode}D_{timestamp}`

## Log Structure

```
logs/4459_4P_1D_20251122_041341/
│
├── config.yaml                              # Resolved job configuration
├── sglang_config.yaml                       # SGLang worker configuration
├── sbatch_script.sh                         # Generated SLURM script
├── nginx.conf                               # Load balancer configuration
├── 4459.json                                # Job metadata
│
├── log.out                                  # Main orchestration stdout
├── log.err                                  # Main orchestration stderr
├── benchmark.out                            # Benchmark results
├── benchmark.err                            # Benchmark errors
│
├── {node}_prefill_w{n}.out                  # Prefill worker stdout
├── {node}_prefill_w{n}.err                  # Prefill worker stderr (SGLang logs)
├── {node}_decode_w{n}.out                   # Decode worker stdout
├── {node}_decode_w{n}.err                   # Decode worker stderr (SGLang logs)
├── {node}_frontend_{n}.out                  # Frontend stdout
├── {node}_frontend_{n}.err                  # Frontend stderr
├── {node}_nginx.out                         # Nginx stdout
├── {node}_nginx.err                         # Nginx stderr
├── {node}_config.json                       # Per-node SGLang config dump
│
├── cached_assets/                           # Cached model assets
└── sa-bench_isl_1024_osl_1024/              # Benchmark results
    ├── isl_1024_osl_1024_concurrency_128_req_rate_inf.json
    ├── isl_1024_osl_1024_concurrency_512_req_rate_inf.json
    └── ...
```

## Key Files

### log.out

The main orchestration log showing node assignments, worker launches, and the frontend URL:

```
Node 0: watchtower-aqua-cn01
Node 1: watchtower-aqua-cn02
...
Master IP address (node 1): 10.30.1.49
Nginx node (node 0): watchtower-aqua-cn01
...
Prefill worker 0 leader: watchtower-aqua-cn01 (10.30.1.163)
Launching prefill worker 0, node 0 (local_rank 0): watchtower-aqua-cn01
...
Decode worker 0 leader: watchtower-aqua-cn05 (10.30.1.153)
...
Frontend available at: http://watchtower-aqua-cn01:8000
```

### benchmark.out

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

### Worker Logs ({node}_prefill_w0.err, {node}_decode_w0.err)

SGLang worker logs showing model loading, memory allocation, and runtime info. Check these for debugging CUDA errors, OOM issues, or NCCL failures.

### config.yaml

The fully resolved configuration showing exactly what ran, with all aliases expanded and defaults applied.

## Common Commands

```bash
# Watch main log
tail -f logs/4459_*/log.out

# Watch benchmark progress
tail -f logs/4459_*/benchmark.out

# Check for errors across all workers
grep -l "Error\|Exception\|CUDA" logs/4459_*/*.err

# Check specific worker
cat logs/4459_*/*_decode_w0.err | tail -100

# Compare throughput across runs
for dir in logs/*/; do
  echo "=== $dir ==="
  grep "Output token throughput" "$dir/benchmark.out" 2>/dev/null
done
```

## Connecting to Running Jobs

The log.out file includes commands to connect to running nodes:

```bash
# Connect to nginx node
srun --jobid 4459 -w watchtower-aqua-cn01 --overlap --pty bash

# Connect to master node
srun --jobid 4459 -w watchtower-aqua-cn02 --overlap --pty bash
```
