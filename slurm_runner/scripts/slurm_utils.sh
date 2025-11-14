#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Slurm utility functions for infbench
# Common functions used across SLURM job templates

# Function to get node IP address in SLURM environment with fallback mechanisms
# This function tries multiple methods to discover the IP address, making it
# robust across different cluster configurations (GB200, H100/EOS, etc.)
#
# Usage: get_node_ip "node_name" "slurm_job_id" "network_interface"
#   node_name: hostname of the node to query
#   slurm_job_id: SLURM job ID for srun context
#   network_interface: (optional) specific network interface to use
#
# Returns: IP address on stdout, exits with code 1 on failure
get_node_ip() {
    local node=$1
    local slurm_job_id=$2
    local network_interface=$3

    # Create inline script that tries multiple methods
    local ip_script="
        # Method 1: Use specific interface if provided
        if [ -n \"$network_interface\" ]; then
            ip=\$(ip addr show $network_interface 2>/dev/null | grep 'inet ' | awk '{print \$2}' | cut -d'/' -f1)
            if [ -n \"\$ip\" ]; then
                echo \"\$ip\"
                exit 0
            fi
        fi

        # Method 2: Use hostname -I (gets first non-loopback IP)
        ip=\$(hostname -I 2>/dev/null | awk '{print \$1}')
        if [ -n \"\$ip\" ]; then
            echo \"\$ip\"
            exit 0
        fi

        # Method 3: Use ip route to find default source IP
        ip=\$(ip route get 8.8.8.8 2>/dev/null | awk -F'src ' 'NR==1{split(\$2,a,\" \");print a[1]}')
        if [ -n \"\$ip\" ]; then
            echo \"\$ip\"
            exit 0
        fi

        exit 1
    "

    # Execute the script on target node with single srun command
    local result
    result=$(srun --jobid $slurm_job_id --nodes=1 --ntasks=1 --nodelist=$node bash -c "$ip_script" 2>&1)
    local rc=$?

    if [ $rc -eq 0 ] && [ -n "$result" ]; then
        echo "$result"
        return 0
    else
        echo "Error: Could not retrieve IP address for node $node" >&2
        if [ -n "$network_interface" ]; then
            echo "  Attempted with interface: $network_interface" >&2
        fi
        echo "  Tried fallback methods: hostname -I, ip route" >&2
        return 1
    fi
}

# Function to wait for a service to be ready
# Usage: wait_for_service "service_name" "check_command" [max_retries] [retry_interval]
wait_for_service() {
    local service_name=$1
    local check_cmd=$2
    local max_retries=${3:-15}  # Default 15 retries
    local retry_interval=${4:-10}  # Default 10 seconds

    if [ -z "$service_name" ] || [ -z "$check_cmd" ]; then
        echo "Error: wait_for_service requires service_name and check_cmd parameters" >&2
        return 1
    fi

    echo "Waiting for $service_name to be ready..."

    for i in $(seq 1 $max_retries); do
        if eval "$check_cmd" >/dev/null 2>&1; then
            echo "$service_name is ready!"
            return 0
        fi
        echo "Attempt $i/$max_retries: $service_name not ready yet, waiting ${retry_interval}s..."
        sleep $retry_interval
    done

    echo "Error: $service_name failed to start within timeout (${max_retries} retries, ${retry_interval}s interval)" >&2
    return 1
}

# Function to print timestamped message
# Usage: log_message "message"
log_message() {
    local message="$1"
    local timestamp=$(date +%Y-%m-%d_%H:%M:%S)
    echo "[${timestamp}] $message"
}
