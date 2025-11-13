"""
RunLoader service for loading and parsing benchmark runs

Clean, JSON-only implementation - requires {jobid}.json in each run directory.
"""

import json
import logging
import os
import re

from .models import BenchmarkRun

logger = logging.getLogger(__name__)


class RunLoader:
    """Service for loading benchmark run data from directories.

    Requires {jobid}.json metadata file in each run directory.
    """

    def __init__(self, logs_dir: str):
        """Initialize RunLoader with a logs directory.

        Args:
            logs_dir: Path to directory containing benchmark run subdirectories
        """
        self.logs_dir = logs_dir

    def load_all(self) -> list[BenchmarkRun]:
        """Load all benchmark runs from the logs directory.

        Only loads runs that have {jobid}.json metadata files.

        Returns:
            List of BenchmarkRun objects, sorted by job ID (newest first)
        """
        paths = self._find_run_directories()

        runs = []
        for path in sorted(
            paths, key=lambda p: self._extract_job_id(os.path.basename(p)), reverse=True
        ):
            try:
                run = BenchmarkRun.from_json_file(path)
                if run is not None:
                    # Load benchmark results from profiler output files
                    self._load_benchmark_results(run)

                    # Only include runs with benchmark data
                    if run.profiler.output_tps:
                        runs.append(run)
                    else:
                        logger.debug(f"Skipping run {run.job_id} - no benchmark data")
                else:
                    logger.warning(f"No metadata JSON found for {os.path.basename(path)}")
            except Exception as e:
                logger.error(f"Error loading run from {path}: {e}")
                continue

        return runs

    def load_single(self, run_dir: str) -> BenchmarkRun | None:
        """Load a single benchmark run.

        Args:
            run_dir: Path to run directory (relative to logs_dir or absolute)

        Returns:
            BenchmarkRun object or None if loading failed
        """
        # Handle both relative and absolute paths
        if not os.path.isabs(run_dir):
            run_path = os.path.join(self.logs_dir, run_dir)
        else:
            run_path = run_dir

        if not os.path.exists(run_path):
            logger.error(f"Run directory does not exist: {run_path}")
            return None

        try:
            run = BenchmarkRun.from_json_file(run_path)
            if run is not None:
                self._load_benchmark_results(run)
            return run
        except Exception as e:
            logger.error(f"Error loading run from {run_path}: {e}")
            return None

    def has_metadata_json(self, run_dir: str) -> bool:
        """Check if a run directory has a metadata JSON file.

        Args:
            run_dir: Path to run directory

        Returns:
            True if {jobid}.json exists, False otherwise
        """
        # Handle both relative and absolute paths
        if not os.path.isabs(run_dir):
            run_path = os.path.join(self.logs_dir, run_dir)
        else:
            run_path = run_dir

        dirname = os.path.basename(run_path)
        job_id = dirname.split("_")[0]
        json_path = os.path.join(run_path, f"{job_id}.json")
        return os.path.exists(json_path)

    def _find_run_directories(self) -> list[str]:
        """Find all valid benchmark run directories in logs_dir.

        Returns:
            List of absolute paths to run directories
        """
        paths = []

        if not os.path.exists(self.logs_dir):
            logger.error(f"Logs directory does not exist: {self.logs_dir}")
            return paths

        for entry in os.listdir(self.logs_dir):
            # Skip hidden directories and files
            if entry.startswith("."):
                continue
            # Skip common non-job directories
            if entry in ["utils", "__pycache__", "venv", ".venv"]:
                continue
            # Skip Python files
            if ".py" in entry:
                continue

            full_path = os.path.join(self.logs_dir, entry)
            if not os.path.isdir(full_path):
                continue

            # Only include directories that start with a numeric job ID
            first_part = entry.split("_")[0]
            if not first_part.isdigit():
                continue

            paths.append(full_path)

        return paths

    def _extract_job_id(self, dirname: str) -> int:
        """Extract numeric job ID from directory name for sorting.

        Args:
            dirname: Directory name like "3667_1P_1D_20251110_192145"

        Returns:
            Job ID as integer, or -1 if not parseable
        """
        try:
            return int(dirname.split("_")[0])
        except (ValueError, IndexError):
            return -1

    def _load_benchmark_results(self, run: BenchmarkRun) -> None:
        """Load benchmark results from profiler output files.

        Looks for directories like "sa-bench_isl_1024_osl_1024/" or "vllm_isl_1024_osl_1024/" and parses JSON files.

        Args:
            run: BenchmarkRun object to populate with results
        """
        run_path = run.metadata.path

        # Look for profiler result directories
        # Support both new naming (sa-bench) and old naming (vllm) for backward compatibility
        profiler_type = run.profiler.profiler_type
        if profiler_type == "sa-bench":
            # Try sa-bench first, fall back to vllm for old runs
            patterns = [
                rf"sa-bench_isl_{run.profiler.isl}_osl_{run.profiler.osl}",
                rf"vllm_isl_{run.profiler.isl}_osl_{run.profiler.osl}",
            ]
        else:
            # For other types (sglang, gap, manual), use exact match
            patterns = [rf"{profiler_type}_isl_{run.profiler.isl}_osl_{run.profiler.osl}"]

        for pattern_str in patterns:
            profiler_pattern = re.compile(pattern_str)
            for entry in os.listdir(run_path):
                if profiler_pattern.match(entry):
                    result_dir = os.path.join(run_path, entry)
                    if os.path.isdir(result_dir):
                        results = self._parse_profiler_results(result_dir)
                        run.profiler.add_benchmark_results(results)
                        return  # Found results, stop searching

    def _parse_profiler_results(self, result_dir: str) -> dict:
        """Parse profiler result JSON files.

        Args:
            result_dir: Path to directory containing benchmark result JSON files

        Returns:
            Dict with concurrencies, output_tps, mean_itl_ms, etc.
        """
        result = []

        for file in os.listdir(result_dir):
            if not file.endswith(".json"):
                continue

            filepath = os.path.join(result_dir, file)
            try:
                with open(filepath) as f:
                    content = json.load(f)

                    # Parse based on profiler type (vllm/sglang use same format)
                    res = [
                        content.get("max_concurrency"),
                        content.get("output_throughput"),
                        content.get("mean_itl_ms"),
                        content.get("mean_ttft_ms"),
                        content.get("request_rate"),
                    ]

                    if "mean_tpot_ms" in content:
                        res.append(content["mean_tpot_ms"])

                    result.append(res)
            except Exception as e:
                logger.warning(f"Error parsing {filepath}: {e}")
                continue

        # Organize results
        out = {
            "request_rate": [],
            "concurrencies": [],
            "output_tps": [],
            "mean_itl_ms": [],
            "mean_ttft_ms": [],
            "mean_tpot_ms": [],
        }

        for data in sorted(result, key=lambda x: x[0] if x[0] else 0):
            if len(data) >= 5:
                con, tps, itl, ttft, req_rate = data[0:5]
                out["concurrencies"].append(con)
                out["output_tps"].append(tps)
                out["mean_itl_ms"].append(itl)
                out["mean_ttft_ms"].append(ttft)
                out["request_rate"].append(req_rate)

                if len(data) >= 6:
                    out["mean_tpot_ms"].append(data[5])

        return out

    def get_run_count(self) -> int:
        """Get count of valid benchmark runs in logs directory.

        Returns:
            Number of valid run directories found
        """
        return len(self._find_run_directories())

    def get_runs_with_metadata(self) -> list[str]:
        """Get list of run directories that have metadata JSON files.

        Returns:
            List of run directory names (not full paths)
        """
        runs_with_metadata = []

        for path in self._find_run_directories():
            if self.has_metadata_json(path):
                runs_with_metadata.append(os.path.basename(path))

        return runs_with_metadata

    def get_runs_without_metadata(self) -> list[str]:
        """Get list of run directories that DON'T have metadata JSON files.

        Useful for identifying which runs need metadata file generation.

        Returns:
            List of run directory names (not full paths)
        """
        runs_without_metadata = []

        for path in self._find_run_directories():
            if not self.has_metadata_json(path):
                runs_without_metadata.append(os.path.basename(path))

        return runs_without_metadata

    def to_dataframe(self, runs: list[BenchmarkRun] | None = None):
        """Convert runs to pandas DataFrame for analysis.

        Args:
            runs: List of BenchmarkRun objects. If None, loads all runs.

        Returns:
            pandas DataFrame with one row per concurrency level
        """
        import pandas as pd

        if runs is None:
            runs = self.load_all()

        rows = []

        for run in runs:
            run_id = f"{run.job_id}_{run.metadata.prefill_nodes}P_{run.metadata.decode_nodes}D_{run.metadata.run_date}"
            total_gpus = run.total_gpus

            # Create a row for each concurrency level
            for i in range(len(run.profiler.output_tps)):
                # Calculate derived metrics
                tps = run.profiler.output_tps[i]
                tps_per_gpu = tps / total_gpus if total_gpus > 0 else 0

                # Output TPS/User = 1000 / TPOT(ms)
                tpot = run.profiler.mean_tpot_ms[i] if i < len(run.profiler.mean_tpot_ms) else None
                tps_per_user = 1000 / tpot if tpot and tpot > 0 else 0

                row = {
                    "Run ID": run_id,
                    "Run Date": run.metadata.run_date,
                    "Profiler": run.profiler.profiler_type,
                    "ISL": run.profiler.isl,
                    "OSL": run.profiler.osl,
                    "Prefill TP": run.metadata.gpus_per_node,
                    "Prefill DP": run.metadata.prefill_nodes,
                    "Decode TP": run.metadata.gpus_per_node,
                    "Decode DP": run.metadata.decode_nodes,
                    "Frontends": run.metadata.num_additional_frontends,
                    "Total GPUs": total_gpus,
                    "Request Rate": (
                        run.profiler.request_rate[i]
                        if i < len(run.profiler.request_rate)
                        else "N/A"
                    ),
                    "Concurrency": (
                        run.profiler.concurrency_values[i]
                        if i < len(run.profiler.concurrency_values)
                        else "N/A"
                    ),
                    "Output TPS": tps,
                    "Output TPS/GPU": tps_per_gpu,
                    "Output TPS/User": tps_per_user,
                    "Mean TTFT (ms)": (
                        run.profiler.mean_ttft_ms[i]
                        if i < len(run.profiler.mean_ttft_ms)
                        else "N/A"
                    ),
                    "Mean TPOT (ms)": tpot if tpot else "N/A",
                    "Mean ITL (ms)": (
                        run.profiler.mean_itl_ms[i] if i < len(run.profiler.mean_itl_ms) else "N/A"
                    ),
                }
                rows.append(row)

        return pd.DataFrame(rows)
