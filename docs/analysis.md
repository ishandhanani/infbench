# Analysis UI

The analysis UI provides interactive visualization of benchmark results.

## Starting the UI

```bash
# From the srtctl directory
make ui

# Or directly
streamlit run ui/app.py
```

The UI opens in your browser at `http://localhost:8501`.

## Loading Results

Point the UI to your logs directory:

1. Enter the path to your logs directory (e.g., `logs/`)
2. Select specific job directories to analyze
3. Click "Load Results"

The UI reads benchmark JSON files from:
```
logs/{job_id}_*/sa-bench_isl_{isl}_osl_{osl}/*.json
```

## Key Metrics

| Metric | Description |
|--------|-------------|
| **Output Throughput** | Tokens generated per second |
| **Request Throughput** | Requests completed per second |
| **TTFT (P50/P99)** | Time to first token |
| **TPOT (P50/P99)** | Time per output token |
| **ITL (P50/P99)** | Inter-token latency |

## Comparing Runs

1. Use the sidebar to select 2+ runs
2. The comparison view shows metrics side-by-side
3. Hover over charts for detailed values

### Sweep Analysis

For parameter sweeps, the UI can:

- Group runs by sweep parameter
- Plot throughput vs parameter value
- Identify optimal configurations

## Exporting Results

### CSV Export

Export selected metrics to CSV for further analysis.

### Chart Export

Right-click charts to save as PNG or SVG.
