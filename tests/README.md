# Tests

Unit tests for srtctl YAML configuration and command generation.

## Test Structure

### `test_command_generation.py`

Tests SGLang command generation from YAML configs. Verifies that the generated `commands.sh` files contain the expected flags and environment variables.

**Test Cases:**

1. **`test_basic_disaggregated_commands()`**
   - Tests disaggregated mode (1 prefill node + 4 decode nodes)
   - Verifies prefill and decode commands are generated correctly
   - Checks environment variables, SGLang flags, and coordination flags
   - Validates disaggregation-mode flags (prefill/decode)
   - Ensures max-total-tokens only appears in prefill command

2. **`test_basic_aggregated_commands()`**
   - Tests aggregated mode (4 combined nodes)
   - Verifies aggregated command uses `aggregated_environment`
   - Checks that aggregated config section is used
   - Validates no disaggregation-mode flag is present
   - Ensures correct nnodes for aggregated workers

3. **`test_environment_variable_handling()`**
   - Tests configs with no environment variables
   - Verifies commands work correctly without env vars
   - Ensures no spurious env var lines in output

4. **`test_profiling_mode()`**
   - Tests profiling mode configuration
   - Verifies `sglang.launch_server` is used instead of `dynamo.sglang`
   - Checks that disaggregation-mode flag is skipped when profiling

5. **`test_config_from_yaml_file()`**
   - Tests loading from actual `configs/example.yaml`
   - End-to-end validation of config loading and command generation
   - Ensures example configs are valid

## Running Tests

```bash
# Run all tests with pytest
uv run pytest tests/

# Run specific test file
uv run pytest tests/test_command_generation.py

# Run tests directly (without pytest)
uv run python tests/test_command_generation.py

# Verbose output
uv run pytest tests/ -v
```

## What's Tested

### Command Structure
- Environment variables are rendered before python command
- Correct python module (`dynamo.sglang` vs `sglang.launch_server`)
- SGLang flags are properly formatted with `--flag-name value`
- Coordination flags (--nnodes, --dist-init-addr, --node-rank)

### Configuration Modes
- **Disaggregated**: Separate prefill/decode workers
  - `prefill_nodes`, `decode_nodes`, `prefill_workers`, `decode_workers`
  - `prefill_environment` and `decode_environment`
  - `sglang_config.prefill` and `sglang_config.decode`

- **Aggregated**: Combined workers
  - `agg_nodes`, `agg_workers`
  - `aggregated_environment`
  - `sglang_config.aggregated`

### SGLang Flags
- Required flags: `--model-path`, `--tensor-parallel-size`
- Mode-specific flags: `--disaggregation-mode` (disagg only)
- Prefill-only flags: `--max-total-tokens`
- Memory flags: `--mem-fraction-static`, `--kv-cache-dtype`
- Quantization flags: `--quantization`

## Test Philosophy

These tests focus on **command generation correctness** rather than end-to-end job execution:

1. **Config → SGLang Config**: YAML configs are transformed into SGLang config files
2. **Config → Commands**: Commands are rendered with correct flags and env vars
3. **Validation**: Generated commands match expected structure

Tests use minimal configs to isolate specific functionality and avoid dependencies on external services (SLURM, containers, models).

## Adding New Tests

When adding SGLang flags or changing command generation logic:

1. Add test case to `test_command_generation.py`
2. Create minimal config with the new feature
3. Assert expected flags/env vars in generated command
4. Run tests to verify

Example:
```python
def test_new_sglang_flag():
    config = {
        "name": "test",
        # ... minimal config ...
        "backend": {
            "sglang_config": {
                "prefill": {
                    "my-new-flag": "value"
                }
            }
        }
    }

    backend = SGLangBackend(config)
    sglang_config_path = backend.generate_config_file()
    cmd = backend.render_command(mode="prefill", config_path=sglang_config_path)

    assert "--my-new-flag value" in cmd
```
