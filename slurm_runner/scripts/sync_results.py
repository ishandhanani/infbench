#!/usr/bin/env python3
"""
CLI tool for syncing benchmark results with S3-compatible cloud storage.

Usage:
    python scripts/sync_results.py push <run_dir>        - Push single run
    python scripts/sync_results.py push-all              - Push all local runs
    python scripts/sync_results.py pull <run_id>         - Pull specific run
    python scripts/sync_results.py pull-missing          - Pull all missing runs
    python scripts/sync_results.py list-remote           - List remote runs
    python scripts/sync_results.py test                  - Test cloud connection
"""

import argparse
import logging
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from srtslurm.cloud_sync import create_sync_manager_from_config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def progress_callback(current: int, total: int, filename: str):
    """Progress callback for file operations."""
    percentage = (current / total) * 100
    print(f"\r[{current}/{total}] ({percentage:.1f}%) {filename}", end="", flush=True)
    if current == total:
        print()  # New line when done


def sync_progress_callback(run_name: str, current: int, total: int):
    """Progress callback for sync operations."""
    print(f"[{current}/{total}] Syncing {run_name}...")


def cmd_push(args, sync_manager):
    """Push a single run to cloud storage."""
    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        logger.error(f"Run directory does not exist: {args.run_dir}")
        return 1

    logger.info(f"Pushing {run_dir.name} to cloud storage...")
    success = sync_manager.push_run(str(run_dir), progress_callback)

    if success:
        logger.info(f"✓ Successfully pushed {run_dir.name}")
        return 0
    else:
        logger.error(f"✗ Failed to push {run_dir.name}")
        return 1


def cmd_push_all(args, sync_manager):
    """Push all local runs to cloud storage."""
    # Find all run directories in current directory or specified logs_dir
    logs_dir = Path(args.logs_dir) if args.logs_dir else Path.cwd()

    if not logs_dir.exists():
        logger.error(f"Logs directory does not exist: {logs_dir}")
        return 1

    # Find run directories (format: JOBID_*P_*D_*)
    run_dirs = []
    for entry in logs_dir.iterdir():
        if entry.is_dir() and not entry.name.startswith("."):
            # Check if looks like a run directory
            parts = entry.name.split("_")
            if len(parts) >= 4 and parts[0].isdigit():
                run_dirs.append(entry)

    if not run_dirs:
        logger.warning(f"No run directories found in {logs_dir}")
        return 0

    logger.info(f"Found {len(run_dirs)} run(s) to push")

    # Push each run
    success_count = 0
    for i, run_dir in enumerate(run_dirs, 1):
        logger.info(f"\n[{i}/{len(run_dirs)}] Pushing {run_dir.name}...")

        # Check if already exists in cloud
        if sync_manager.run_exists_in_cloud(run_dir.name):
            logger.info(f"  → Already exists in cloud, skipping")
            success_count += 1
            continue

        success = sync_manager.push_run(str(run_dir), progress_callback)
        if success:
            success_count += 1
            logger.info(f"  ✓ Success")
        else:
            logger.error(f"  ✗ Failed")

    logger.info(f"\nPushed {success_count}/{len(run_dirs)} runs successfully")
    return 0 if success_count == len(run_dirs) else 1


def cmd_pull(args, sync_manager):
    """Pull a specific run from cloud storage."""
    logs_dir = Path(args.logs_dir) if args.logs_dir else Path.cwd()
    logs_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Pulling {args.run_id} from cloud storage...")
    result = sync_manager.pull_run(args.run_id, str(logs_dir), progress_callback)

    if result:
        logger.info(f"✓ Successfully pulled to {result}")
        return 0
    else:
        logger.error(f"✗ Failed to pull {args.run_id}")
        return 1


def cmd_pull_missing(args, sync_manager):
    """Pull all missing runs from cloud storage."""
    logs_dir = Path(args.logs_dir) if args.logs_dir else Path.cwd()
    logs_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Syncing missing runs from cloud storage...")
    count = sync_manager.sync_missing_runs(str(logs_dir), sync_progress_callback)

    if count > 0:
        logger.info(f"✓ Downloaded {count} run(s)")
        return 0
    else:
        logger.info("No missing runs to download")
        return 0


def cmd_list_remote(args, sync_manager):
    """List all runs in cloud storage."""
    logger.info("Fetching list of remote runs...")
    runs = sync_manager.list_remote_runs()

    if not runs:
        logger.info("No runs found in cloud storage")
        return 0

    print(f"\nFound {len(runs)} run(s) in cloud storage:\n")
    for run in runs:
        print(f"  • {run}")

    return 0


def cmd_test(args, sync_manager):
    """Test cloud storage connection."""
    logger.info("Testing cloud storage connection...")
    success = sync_manager.test_connection()

    if success:
        logger.info("✓ Connection successful")
        return 0
    else:
        logger.error("✗ Connection failed")
        return 1


def main():
    parser = argparse.ArgumentParser(
        description="Sync benchmark results with S3-compatible cloud storage"
    )
    parser.add_argument(
        "--config",
        default="cloud_config.toml",
        help="Path to cloud config file (default: cloud_config.toml)",
    )
    parser.add_argument(
        "--logs-dir",
        help="Logs directory (default: current directory)",
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Push command
    push_parser = subparsers.add_parser("push", help="Push a single run to cloud")
    push_parser.add_argument("run_dir", help="Path to run directory")

    # Push all command
    subparsers.add_parser("push-all", help="Push all local runs to cloud")

    # Pull command
    pull_parser = subparsers.add_parser("pull", help="Pull a specific run from cloud")
    pull_parser.add_argument("run_id", help="Run ID (directory name)")

    # Pull missing command
    subparsers.add_parser("pull-missing", help="Pull all missing runs from cloud")

    # List remote command
    subparsers.add_parser("list-remote", help="List all runs in cloud storage")

    # Test command
    subparsers.add_parser("test", help="Test cloud storage connection")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    # Create sync manager from config
    sync_manager = create_sync_manager_from_config(args.config)
    if not sync_manager:
        logger.error(f"Failed to load cloud config from {args.config}")
        logger.error("Create cloud_config.toml from cloud_config.toml.example")
        return 1

    # Execute command
    commands = {
        "push": cmd_push,
        "push-all": cmd_push_all,
        "pull": cmd_pull,
        "pull-missing": cmd_pull_missing,
        "list-remote": cmd_list_remote,
        "test": cmd_test,
    }

    return commands[args.command](args, sync_manager)


if __name__ == "__main__":
    sys.exit(main())

