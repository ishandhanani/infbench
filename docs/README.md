# Introduction

`srtctl` is a command-line tool for running distributed LLM inference benchmarks on SLURM clusters. It replaces complex shell scripts and 50+ CLI flags with clean, declarative YAML configuration files.

## Why srtctl?

Running large language models across multiple GPUs and nodes requires orchestrating many moving parts: SLURM job scripts, container mounts, SGLang configuration, worker coordination, and benchmark execution. Traditionally, this meant maintaining brittle bash scripts with hardcoded parameters.

`srtctl` solves this by providing:

- **Declarative configuration** - Define your entire job in a single YAML file
- **Validation** - Catch configuration errors before submitting to SLURM
- **Reproducibility** - Every job saves its full configuration for later reference
- **Parameter sweeps** - Run grid searches across configurations with a single command

## Architecture Overview

`srtctl` orchestrates distributed inference using SGLang workers in either **disaggregated** or **aggregated** mode.

**Disaggregated Mode** separates prefill and decode into specialized workers:
- Prefill workers handle the initial prompt processing
- Decode workers handle token generation
- An nginx load balancer distributes requests across frontends

**Aggregated Mode** runs combined prefill+decode on each worker, simpler but potentially less efficient for high-throughput scenarios.

## How It Works

When you run `srtctl config.yaml`, the tool validates your configuration, resolves any aliases from your cluster config, generates a SLURM batch script and SGLang configuration files, then submits to SLURM. Once allocated, workers launch inside containers, discover each other through ETCD and NATS, and begin serving. If you've configured a benchmark, it runs automatically against the serving endpoint and saves results to the log directory.

## Next Steps

- [Installation](installation.md) - Set up `srtctl` and submit your first job
- [Monitoring](monitoring.md) - Understanding job logs and debugging
- [Parameter Sweeps](sweeps.md) - Run grid searches across configurations
