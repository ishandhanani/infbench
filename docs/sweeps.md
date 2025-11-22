# Parameter Sweeps

Parameter sweeps let you run multiple configurations with a single command.

## How It Works

1. Add `{placeholder}` markers in your YAML config
2. Run with `--sweep placeholder=value1,value2,value3`
3. `srtctl` submits one job per value

## Simple Walkthrough

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
  concurrencies: [{concurrency}]  # <-- placeholder
```

### Step 2: Preview with dry-run

```bash
srtctl configs/concurrency-sweep.yaml --sweep concurrency=128,256,512 --dry-run
```

This shows you what will be generated without submitting.

### Step 3: Submit

```bash
srtctl configs/concurrency-sweep.yaml --sweep concurrency=128,256,512
```

This submits 3 separate jobs, one for each concurrency value.

## Multiple Parameters

Multiple placeholders create a Cartesian product:

```yaml
backend:
  sglang_config:
    decode:
      mem-fraction-static: {mem}

benchmark:
  concurrencies: [{conc}]
```

```bash
srtctl config.yaml --sweep mem=0.85,0.90 conc=256,512
```

This generates 4 jobs (2 x 2):
- mem=0.85, conc=256
- mem=0.85, conc=512
- mem=0.90, conc=256
- mem=0.90, conc=512

## Where Placeholders Can Go

Placeholders work anywhere in the YAML:

```yaml
name: "sweep-{param}"
mem-fraction-static: {mem}
concurrencies: [{conc}]
dp-size: {dp}
```

## Tips

- Always use `--dry-run` first to verify
- Start with 2-3 values before running large sweeps
- Cartesian products grow fast: 3 params x 4 values = 64 jobs
