#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Function to print usage
print_usage() {
    echo "Usage: $0 <mode>"
    echo "  mode: prefill or decode"
    echo ""
    echo "Examples:"
    echo "  $0 prefill"
    echo "  $0 decode"
    exit 1
}

# Check if correct number of arguments provided
if [ $# -ne 1 ]; then
    echo "Error: Expected 1 argument, got $#"
    print_usage
fi

# Parse arguments
mode=$1

# Validate mode argument
if [ "$mode" != "prefill" ] && [ "$mode" != "decode" ]; then
    echo "Error: mode must be 'prefill' or 'decode', got '$mode'"
    print_usage
fi

echo "Mode: $mode"

# Determine which Python module to use based on USE_SGLANG_LAUNCH_SERVER
if [[ "${USE_SGLANG_LAUNCH_SERVER,,}" == "true" ]]; then
    PYTHON_MODULE="sglang.launch_server"
    echo "Command: sglang.launch_server (profiling mode)"
else
    PYTHON_MODULE="dynamo.sglang"
    echo "Command: dynamo.sglang"
fi
echo "Command: dynamo"

# Check if required environment variables are set
if [ -z "$HOST_IP_MACHINE" ]; then
    echo "Error: HOST_IP_MACHINE environment variable is not set"
    exit 1
fi

if [ -z "$PORT" ]; then
    echo "Error: PORT environment variable is not set"
    exit 1
fi

if [ -z "$TOTAL_GPUS" ]; then
    echo "Error: TOTAL_GPUS environment variable is not set"
    exit 1
fi

if [ -z "$RANK" ]; then
    echo "Error: RANK environment variable is not set"
    exit 1
fi

if [ -z "$TOTAL_NODES" ]; then
    echo "Error: TOTAL_NODES environment variable is not set"
    exit 1
fi

if [ -z "$USE_INIT_LOCATIONS" ]; then
    echo "Error: USE_INIT_LOCATIONS environment variable is not set"
    exit 1
fi

if [ -z "$USE_DYNAMO_WHLS" ]; then
    echo "Error: USE_DYNAMO_WHLS environment variable is not set"
    exit 1
fi

# Construct command based on mode
if [ "$mode" = "prefill" ]; then
    set -x
    if [[ "${USE_DYNAMO_WHLS,,}" == "true" ]]; then
        python3 -m pip install /configs/ai_dynamo_runtime-0.7.0-cp310-abi3-manylinux_2_28_x86_64.whl
        python3 -m pip install /configs/ai_dynamo-0.7.0-py3-none-any.whl
    fi
    export TORCH_DISTRIBUTED_DEFAULT_TIMEOUT=1800

    command_suffix=""
    if [[ -n "${DUMP_CONFIG_PATH}" ]]; then command_suffix="${command_suffix} --dump-config-to ${DUMP_CONFIG_PATH}"; fi

    # Only add --disaggregation-mode if not in profiling mode
    if [[ "${USE_SGLANG_LAUNCH_SERVER,,}" != "true" ]]; then
        DISAGG_MODE_FLAG="--disaggregation-mode prefill"
    else
        DISAGG_MODE_FLAG=""
    fi

    DYN_SKIP_SGLANG_LOG_FORMATTING=1 \
    PYTHONUNBUFFERED=1 \
    python3 -m $PYTHON_MODULE \
        --model-path /model/ \
        --served-model-name deepseek-ai/DeepSeek-R1 \
        --tp 16 \
        --dp-size 16 \
        --dist-init-addr "$HOST_IP_MACHINE:$PORT" \
        --nnodes "$TOTAL_NODES" \
        --node-rank "$RANK" \
        --enable-dp-attention \
        --trust-remote-code \
        --skip-tokenizer-init \
        $DISAGG_MODE_FLAG \
        --disaggregation-transfer-backend nixl \
        --disaggregation-bootstrap-port 30001 \
        --load-balance-method round_robin \
        --host 0.0.0.0 \
        --mem-fraction-static 0.82 ${command_suffix}

elif [ "$mode" = "decode" ]; then
    set -x
    if [[ "${USE_DYNAMO_WHLS,,}" == "true" ]]; then
        python3 -m pip install /configs/ai_dynamo_runtime-0.7.0-cp310-abi3-manylinux_2_28_x86_64.whl
        python3 -m pip install /configs/ai_dynamo-0.7.0-py3-none-any.whl
    fi
    export TORCH_DISTRIBUTED_DEFAULT_TIMEOUT=1800

    command_suffix=""
    if [[ -n "${DUMP_CONFIG_PATH}" ]]; then command_suffix="${command_suffix} --dump-config-to ${DUMP_CONFIG_PATH}"; fi

    # Only add --disaggregation-mode if not in profiling mode
    if [[ "${USE_SGLANG_LAUNCH_SERVER,,}" != "true" ]]; then
        DISAGG_MODE_FLAG="--disaggregation-mode decode"
    else
        DISAGG_MODE_FLAG=""
    fi

    DYN_SKIP_SGLANG_LOG_FORMATTING=1 \
    PYTHONUNBUFFERED=1 \
    python3 -m $PYTHON_MODULE \
        --model-path /model/ \
        --served-model-name deepseek-ai/DeepSeek-R1 \
        --tp 16 \
        --dp-size 16 \
        --dist-init-addr "$HOST_IP_MACHINE:$PORT" \
        --nnodes "$TOTAL_NODES" \
        --node-rank "$RANK" \
        --enable-dp-attention \
        --trust-remote-code \
        --skip-tokenizer-init \
        $DISAGG_MODE_FLAG \
        --disaggregation-transfer-backend nixl \
        --disaggregation-bootstrap-port 30001 \
        --host 0.0.0.0 \
        --prefill-round-robin-balance \
        --mem-fraction-static 0.82 \
        --cuda-graph-max-bs 8 ${command_suffix}
fi
