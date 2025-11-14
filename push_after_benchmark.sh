#!/bin/bash
#
# Push benchmark results to cloud storage
#
# Usage:
#   ./push_after_benchmark.sh                  # Defaults to logs/ directory
#   ./push_after_benchmark.sh --log-dir <dir>  # Specify logs directory
#   ./push_after_benchmark.sh <run_dir>        # Push single run
#
# Examples:
#   ./push_after_benchmark.sh
#   ./push_after_benchmark.sh --log-dir /mnt/lustre01/users-public/slurm-shared/joblogs
#   ./push_after_benchmark.sh 3667_1P_1D_20251110_192145
#

set -e  # Exit on error

# Find the sync script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
SYNC_SCRIPT="$SCRIPT_DIR/slurm_runner/scripts/sync_results.py"

if [ ! -f "$SYNC_SCRIPT" ]; then
    echo "Error: sync_results.py not found at $SYNC_SCRIPT"
    exit 1
fi

# Check if credentials are set
if [ -z "$AWS_ACCESS_KEY_ID" ] || [ -z "$AWS_SECRET_ACCESS_KEY" ]; then
    echo "Error: AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY must be set"
    echo "Export these environment variables before running this script"
    exit 1
fi

# Parse arguments
if [ $# -eq 0 ]; then
    # Default: push all from logs/ directory
    LOGS_DIR="$SCRIPT_DIR/logs"
    
    if [ ! -d "$LOGS_DIR" ]; then
        echo "Error: Default logs directory '$LOGS_DIR' does not exist"
        echo "Usage: $0 --log-dir <logs_directory>"
        exit 1
    fi
    
    echo "Pushing all runs from $LOGS_DIR to cloud storage (skipping existing)..."
    python3 "$SYNC_SCRIPT" --logs-dir "$LOGS_DIR" push-all

elif [ "$1" = "--log-dir" ]; then
    # Push all runs from specified logs directory
    if [ $# -lt 2 ]; then
        echo "Error: --log-dir requires a directory path"
        exit 1
    fi
    
    LOGS_DIR="$2"
    
    if [ ! -d "$LOGS_DIR" ]; then
        echo "Error: Directory '$LOGS_DIR' does not exist"
        exit 1
    fi
    
    echo "Pushing all runs from $LOGS_DIR to cloud storage (skipping existing)..."
    python3 "$SYNC_SCRIPT" --logs-dir "$LOGS_DIR" push-all
    
else
    # Push single run directory
    RUN_DIR="$1"
    
    if [ ! -d "$RUN_DIR" ]; then
        echo "Error: Directory '$RUN_DIR' does not exist"
        exit 1
    fi
    
    echo "Pushing $RUN_DIR to cloud storage..."
    python3 "$SYNC_SCRIPT" push "$RUN_DIR"
fi

if [ $? -eq 0 ]; then
    echo "✓ Successfully pushed to cloud storage"
    exit 0
else
    echo "✗ Failed to push to cloud storage"
    exit 1
fi

