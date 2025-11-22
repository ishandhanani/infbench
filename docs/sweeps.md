# Parameter Sweeps

Parameter sweeps let you run multiple configurations with a single command. This is useful for:

- Finding optimal concurrency levels
- Tuning memory fractions
- Comparing different model configurations
- Benchmarking across input/output sequence lengths

## How Sweeps Work

1. Add `{placeholder}` markers in your YAML config
2. Run with `--sweep placeholder=value1,value2,value3`
3. srtctl generates one job per value (or Cartesian product for multiple placeholders)

## Basic Example

### Step 1: Create a sweep config

```yaml
# configs/concurrency-sweep.yaml
name: "concurrency-sweep"

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
  concurrencies: [{concurrency}]  # Placeholder
```

### Step 2: Preview the sweep

```bash
srtctl configs/concurrency-sweep.yaml --sweep concurrency=128,256,512,1024 --dry-run
```

Output:
```
Sweep will generate 4 jobs:
  - concurrency-sweep_concurrency_128
  - concurrency-sweep_concurrency_256
  - concurrency-sweep_concurrency_512
  - concurrency-sweep_concurrency_1024

Generated files in dry-runs/
```

### Step 3: Submit the sweep

```bash
srtctl configs/concurrency-sweep.yaml --sweep concurrency=128,256,512,1024
```

This submits 4 separate SLURM jobs.

## Multiple Parameters (Cartesian Product)

When you specify multiple placeholders, srtctl generates the Cartesian product:

```yaml
backend:
  sglang_config:
    decode:
      mem-fraction-static: {mem_fraction}

benchmark:
  concurrencies: [{concurrency}]
```

```bash
srtctl config.yaml --sweep mem_fraction=0.85,0.90,0.95 concurrency=256,512
```

This generates **6 jobs** (3 memory fractions x 2 concurrencies):
- mem_fraction=0.85, concurrency=256
- mem_fraction=0.85, concurrency=512
- mem_fraction=0.90, concurrency=256
- mem_fraction=0.90, concurrency=512
- mem_fraction=0.95, concurrency=256
- mem_fraction=0.95, concurrency=512

## Walkthrough: Memory Fraction Tuning

A common use case is finding the optimal `mem-fraction-static` setting.

### Step 1: Create the sweep config

```yaml
# configs/mem-fraction-sweep.yaml
name: "mem-fraction-tuning"

model:
  path: "deepseek-r1"
  container: "latest"
  precision: "fp8"

resources:
  gpu_type: "gb200"
  prefill_nodes: 1
  decode_nodes: 4

backend:
  sglang_config:
    prefill:
      mem-fraction-static: {prefill_mem}
    decode:
      mem-fraction-static: {decode_mem}

benchmark:
  type: "sa-bench"
  isl: 1024
  osl: 1024
  concurrencies: [512]
```

### Step 2: Run conservative sweep first

```bash
srtctl configs/mem-fraction-sweep.yaml \
  --sweep prefill_mem=0.80 decode_mem=0.80,0.82,0.84 \
  --dry-run
```

### Step 3: Submit and monitor

```bash
srtctl configs/mem-fraction-sweep.yaml \
  --sweep prefill_mem=0.80 decode_mem=0.80,0.82,0.84
```

```bash
# Check for OOM errors
grep -r "CUDA out of memory" logs/*/
```

### Step 4: Analyze results

```bash
for dir in logs/*mem-fraction*/; do
  echo "=== $dir ==="
  grep "Output token throughput" "$dir/benchmark.out"
done
```

### Step 5: Narrow down

Based on results, run a finer sweep around the best value:

```bash
srtctl configs/mem-fraction-sweep.yaml \
  --sweep prefill_mem=0.80 decode_mem=0.88,0.89,0.90,0.91,0.92
```

## Placeholder Syntax

Placeholders can appear anywhere in the YAML:

```yaml
# In strings
name: "sweep-{param}"

# In numbers
mem-fraction-static: {mem}

# In lists
concurrencies: [{conc}]

# In nested structures
backend:
  sglang_config:
    decode:
      dp-size: {dp}
```

## Best Practices

### 1. Always preview first

```bash
srtctl config.yaml --sweep param=a,b,c --dry-run
```

### 2. Start small

Begin with 2-3 values to ensure configs work:
```bash
--sweep concurrency=256,512
```

### 3. Avoid combinatorial explosion

Cartesian products grow quickly:
- 3 params x 3 values = 27 jobs
- 4 params x 4 values = 256 jobs

### 4. Use meaningful names

```yaml
name: "decode-mem-sweep-v2"  # Not "test123"
```

## Limitations

### No Paired Sweeps

srtctl generates Cartesian products, not paired values. If you need:
```
(tp=4, nodes=1), (tp=8, nodes=2), (tp=16, nodes=4)
```

Create separate config files instead:
```
configs/tp4-nodes1.yaml
configs/tp8-nodes2.yaml
configs/tp16-nodes4.yaml
```
