# YAML Validation & Auto-Population

## What Gets Auto-Populated?

### 1. Backend Config (`backend` section)

**Current behavior:** If you don't specify `backend:` in your YAML, it gets auto-created with defaults.

```python
# schema.py:225-232
def model_post_init(self, __context: Any) -> None:
    """Auto-populate backend config if not provided."""
    if self.backend is None:
        self.backend = BackendConfig()  # Creates empty backend

    # Auto-populate gpu_type from resources
    if self.backend.gpu_type is None:
        self.backend.gpu_type = f"{self.resources.gpu_type}-{self.model.precision}"
```

**What this means:**
- You can omit `backend:` entirely and it will be created
- `backend.gpu_type` is computed as `"{gpu_type}-{precision}"` (e.g., `"gb200-fp8"`)
- All other backend fields use their defaults (e.g., `enable_multiple_frontends: true`)

**Example:**
```yaml
# Minimal YAML (no backend section)
name: "test"
model:
  path: "/models/model"
  container: "container.sqsh"
  precision: "fp8"
resources:
  gpu_type: "gb200"
  # ...
slurm:
  account: "acc"
  partition: "part"

# After validation, backend is auto-created:
# backend:
#   type: "sglang"
#   gpu_type: "gb200-fp8"  # Auto-computed!
#   enable_multiple_frontends: true
#   num_additional_frontends: 9
#   enable_profiling: false
```

### 2. Benchmark Config

**Default:** If you don't specify `benchmark:`, it defaults to `type: manual`.

```python
# schema.py:220
benchmark: BenchmarkConfig = Field(default_factory=BenchmarkConfig)

# schema.py:148
type: BenchmarkType = Field(BenchmarkType.MANUAL, description="Benchmark type")
```

### 3. Resource Defaults

```python
# schema.py:111
gpus_per_node: int = Field(4, description="Number of GPUs per node")
```

Default is `4` GPUs per node if not specified.

### 4. SLURM Defaults

```python
# schema.py:142
time_limit: str = Field("04:00:00", description="Job time limit (HH:MM:SS)")
```

### 5. Other Defaults

```python
# schema.py:223
enable_config_dump: bool = True

# schema.py:200-201
enable_multiple_frontends: bool = True
num_additional_frontends: int = 9
```

## What's Required vs Optional?

### Required Fields (must be in YAML):

```yaml
name: "..."                    # Job name

model:
  path: "..."                  # Model path or alias
  container: "..."             # Container path or alias
  precision: "fp8"             # fp4, fp8, fp16, bf16

resources:
  gpu_type: "gb200"            # gb200, h100
  # EITHER disaggregated:
  prefill_nodes: 1
  decode_nodes: 4
  prefill_workers: 1
  decode_workers: 4
  # OR aggregated:
  agg_nodes: 4
  agg_workers: 4

slurm:
  account: "..."               # SLURM account
  partition: "..."             # SLURM partition
```

### Optional Fields (have defaults):

```yaml
resources:
  gpus_per_node: 4             # Default: 4

slurm:
  time_limit: "04:00:00"       # Default: 4 hours

backend:                       # Entire section is optional!
  type: "sglang"               # Default: sglang (only option)
  gpu_type: "gb200-fp8"        # Auto-computed from resources
  enable_multiple_frontends: true
  num_additional_frontends: 9
  enable_profiling: false
  prefill_environment: {}      # Optional env vars
  decode_environment: {}
  sglang_config:               # Optional SGLang flags
    prefill: {}
    decode: {}

benchmark:                     # Optional, defaults to manual
  type: "manual"

enable_config_dump: true       # Default: true
```

## Validation That Exists

### 1. Mode Validation (Disagg vs Agg)

```python
# schema.py:123-134
@field_validator("prefill_nodes", "decode_nodes", "agg_nodes")
def validate_mode(cls, v, info):
    """Validate that either disagg or agg mode is specified."""
    data = info.data
    has_disagg = any(k in data for k in ["prefill_nodes", "decode_nodes"])
    has_agg = "agg_nodes" in data

    if has_disagg and has_agg:
        raise ValueError("Cannot specify both disaggregated and aggregated mode")
```

**This prevents:**
```yaml
resources:
  prefill_nodes: 1    # Disaggregated
  agg_nodes: 4        # Aggregated - ERROR!
```

### 2. Profiling Mode Validation (on main branch, not in YAML yet)

On main branch (`submit_job_script.py`), there's validation for:
- Profiling mode can't have multiple workers
- Profiling mode can't run benchmarks

**This is NOT yet in the YAML validation schema.**

## What's Missing? Resource Validation

### Missing Validation #1: TP Size vs Nodes/GPUs

**Problem:** No validation that `tensor-parallel-size` matches available GPUs.

Example invalid config:
```yaml
resources:
  prefill_nodes: 1      # 1 node
  gpus_per_node: 4      # 4 GPUs/node = 4 total GPUs

backend:
  sglang_config:
    prefill:
      tensor-parallel-size: 8   # ERROR: Need 8 GPUs but only have 4!
```

**Should validate:**
- `tensor-parallel-size * workers <= nodes * gpus_per_node`
- Note: We only check TP size, not DP/EP (those don't affect GPU requirements per worker)

### Missing Validation #2: Worker Count vs Nodes

**Problem:** No validation that workers fit on nodes.

Example:
```yaml
resources:
  prefill_nodes: 1       # 1 node
  prefill_workers: 4     # 4 workers - but each worker needs GPUs!
  gpus_per_node: 4       # Only 4 GPUs total

backend:
  sglang_config:
    prefill:
      tensor-parallel-size: 4  # Each worker needs 4 GPUs
```

4 workers × 4 GPUs/worker = 16 GPUs needed, but only have 4 GPUs!

### Missing Validation #3: Profiling Constraints

From main branch, these should be added:
- Profiling mode → single worker only
- Profiling mode → no benchmarks

## Recommendations

### Option 1: Keep Auto-Population (Current Approach)

**Pros:**
- Convenient for users (less typing)
- Sensible defaults
- Follows "convention over configuration"

**Cons:**
- "Magic" behavior not obvious to users
- Harder to debug when defaults are wrong
- User doesn't see full config until validation

**Improve by:**
- Adding resource validation (TP/nodes/workers)
- Adding profiling constraints
- Document what gets auto-populated clearly

### Option 2: Remove Auto-Population (Explicit Config)

**Pros:**
- No surprises - what you write is what you get
- Easier to debug
- Config files are complete and self-documenting

**Cons:**
- More verbose YAML files
- Users need to know all fields

**Would require:**
- Make `backend` required in schema
- Remove `model_post_init` auto-population
- Update all example configs

### Option 3: Hybrid (Recommended)

Keep simple defaults but validate strictly:

1. **Keep auto-population for:**
   - `backend.gpu_type` (auto-computed from resources)
   - `benchmark.type: manual` (safe default)
   - Simple flags like `enable_config_dump`, `enable_multiple_frontends`

2. **Require explicit specification for:**
   - All SGLang flags (no auto-generation)
   - Worker/node counts
   - Environment variables

3. **Add validation for:**
   - TP size matches available GPUs
   - Worker count fits on nodes
   - Profiling mode constraints
   - Any other resource conflicts

## Proposed Validation to Add

```python
@field_validator("resources")
def validate_resources(cls, resources):
    """Validate resource allocation makes sense."""
    # Check TP size in sglang_config matches available GPUs
    # Check worker counts fit on nodes
    # Check total GPU requirements
    pass

@field_validator("backend")
def validate_profiling_mode(cls, backend):
    """Validate profiling mode constraints."""
    if backend.enable_profiling:
        # Single worker only
        # No benchmarks
        pass
```

Would you like me to implement these validations?
