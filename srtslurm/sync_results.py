#!/usr/bin/env python3
"""
CLI tool for syncing benchmark results with S3-compatible cloud storage.

This script intelligently syncs files - it only uploads/downloads files that don't
already exist on the destination, making it safe to run repeatedly.

Usage Examples:
    # From logs directory
    cd logs
    
    # Test connection and see what's in the bucket
    python -m srtslurm.sync_results test
    python -m srtslurm.sync_results list-remote
    
    # Push all local runs (only uploads missing files)
    python -m srtslurm.sync_results push-all
    
    # Pull missing runs (only downloads missing files)
    python -m srtslurm.sync_results pull-missing
    
    # Push/pull specific run
    python -m srtslurm.sync_results push 3667_1P_1D_20251110_192145
    python -m srtslurm.sync_results pull 3667_1P_1D_20251110_192145
    
    # Delete run from cloud (requires confirmation)
    python -m srtslurm.sync_results delete 3667_1P_1D_20251110_192145
    python -m srtslurm.sync_results delete 3667_1P_1D_20251110_192145 --force

Commands:
    test          - Test cloud connection
    list-remote   - List all runs in bucket
    push          - Push single run (skips existing files)
    push-all      - Push all local runs (skips existing files)
    pull          - Pull specific run (skips existing files)
    pull-missing  - Pull all runs from cloud (skips existing files)
    delete        - Delete specific run from cloud (DESTRUCTIVE!)
"""

import argparse
import logging
import sys
from pathlib import Path

from .cloud_sync import create_sync_manager_from_config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def progress_callback(current: int, total: int, filename: str, status: str = "processing"):
    """Progress callback for file operations.
    
    Note: Uses print() for interactive progress display with carriage returns.
    This is intentional for CLI tools and doesn't work well with logging.
    """
    percentage = (current / total) * 100
    status_icon = "✓" if status == "uploaded" else "⊙" if status == "skipped" else "↓" if status == "downloaded" else "→"
    print(f"\r{status_icon} [{current}/{total}] ({percentage:.1f}%) {filename:<60}", end="", flush=True)
    if current == total:
        print()  # New line when done


def sync_progress_callback(run_name: str, current: int, total: int, status: str = "syncing"):
    """Progress callback for sync operations.
    
    Note: Uses print() for interactive progress display.
    """
    print(f"[{current}/{total}] Syncing {run_name}...")


def cmd_push(args, sync_manager):
    """Push a single run to cloud storage."""
    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        logger.error(f"Run directory does not exist: {args.run_dir}")
        return 1

    logger.info(f"Pushing {run_dir.name} to cloud storage...")
    success, uploaded, skipped = sync_manager.push_run(str(run_dir), progress_callback)

    if success:
        logger.info(f"✓ Successfully pushed {run_dir.name}: {uploaded} uploaded, {skipped} skipped")
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
    total_uploaded = 0
    total_skipped = 0
    
    for i, run_dir in enumerate(run_dirs, 1):
        logger.info(f"\n[{i}/{len(run_dirs)}] Pushing {run_dir.name}...")

        success, uploaded, skipped = sync_manager.push_run(str(run_dir), progress_callback)
        if success:
            success_count += 1
            total_uploaded += uploaded
            total_skipped += skipped
            logger.info(f"  ✓ Success: {uploaded} uploaded, {skipped} skipped")
        else:
            logger.error(f"  ✗ Failed")

    logger.info(f"\nPushed {success_count}/{len(run_dirs)} runs: {total_uploaded} uploaded, {total_skipped} skipped")
    return 0 if success_count == len(run_dirs) else 1


def cmd_pull(args, sync_manager):
    """Pull a specific run from cloud storage."""
    logs_dir = Path(args.logs_dir) if args.logs_dir else Path.cwd()
    logs_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Pulling {args.run_id} from cloud storage...")
    result_path, downloaded, skipped = sync_manager.pull_run(args.run_id, str(logs_dir), progress_callback)

    if result_path:
        logger.info(f"✓ Successfully pulled to {result_path}: {downloaded} downloaded, {skipped} skipped")
        return 0
    else:
        logger.error(f"✗ Failed to pull {args.run_id}")
        return 1


def cmd_pull_missing(args, sync_manager):
    """Pull all missing runs from cloud storage."""
    logs_dir = Path(args.logs_dir) if args.logs_dir else Path.cwd()
    logs_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Syncing missing runs from cloud storage...")
    runs_synced, files_downloaded, files_skipped = sync_manager.sync_missing_runs(
        str(logs_dir), sync_progress_callback
    )

    if files_downloaded > 0:
        logger.info(f"✓ Synced {runs_synced} run(s): {files_downloaded} downloaded, {files_skipped} skipped")
        return 0
    else:
        logger.info(f"All runs up to date ({files_skipped} files already present)")
        return 0


def cmd_list_remote(args, sync_manager):
    """List all runs in cloud storage."""
    logger.info("Fetching list of remote runs...")
    runs = sync_manager.list_remote_runs()

    if not runs:
        logger.info("No runs found in cloud storage")
        return 0

    logger.info(f"Found {len(runs)} run(s) in cloud storage:")
    for run in runs:
        logger.info(f"  • {run}")

    return 0


def cmd_delete(args, sync_manager):
    """Delete a run from cloud storage."""
    run_id = args.run_id
    
    # Check if run exists
    if not sync_manager.run_exists_in_cloud(run_id):
        logger.error(f"Run {run_id} does not exist in cloud storage")
        return 1
    
    # Confirm deletion unless --force is used
    if not args.force:
        logger.warning(f"⚠️  WARNING: This will permanently delete run {run_id} from cloud storage!")
        response = input("Type the run ID to confirm deletion: ")
        if response != run_id:
            logger.info("Deletion cancelled")
            return 0
    
    logger.info(f"Deleting {run_id} from cloud storage...")
    success, deleted = sync_manager.delete_run(run_id, progress_callback)
    
    if success:
        logger.info(f"✓ Successfully deleted {deleted} files")
        return 0
    else:
        logger.error(f"✗ Failed to delete run (deleted {deleted} files before error)")
        return 1


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
        default="srtslurm.yaml",
        help="Path to config file (default: srtslurm.yaml)",
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

    # Delete command
    delete_parser = subparsers.add_parser("delete", help="Delete a run from cloud storage")
    delete_parser.add_argument("run_id", help="Run ID (directory name) to delete")
    delete_parser.add_argument("--force", action="store_true", help="Skip confirmation prompt")

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
        logger.error("Create srtslurm.yaml from srtslurm.yaml.example (or run 'make setup')")
        return 1

    # Execute command
    commands = {
        "push": cmd_push,
        "push-all": cmd_push_all,
        "pull": cmd_pull,
        "pull-missing": cmd_pull_missing,
        "list-remote": cmd_list_remote,
        "delete": cmd_delete,
        "test": cmd_test,
    }

    return commands[args.command](args, sync_manager)


if __name__ == "__main__":
    sys.exit(main())

