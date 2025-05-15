# Stage 1: Graph Profiler

This document provides an overview of the Graph Profiler implementation for Stage 1 of the activation checkpointing project.

## Overview

The Graph Profiler is a critical component that analyzes PyTorch models to collect detailed memory and performance metrics. It extends PyTorch's FX Interpreter to trace model execution and gather statistics needed for making informed activation checkpointing decisions in later stages.

## Key Components

### 1. GraphProfiler (`graph_prof.py`)

The `GraphProfiler` class is the core of the profiling system, with the following key features:

- **Static Analysis**: Analyzes the graph structure to identify:
  - Forward/backward boundaries
  - Node types (parameters, activations, gradients, optimizer states)
  - Activation liveness (creation and usage ranks)

- **Runtime Profiling**: Collects metrics during model execution:
  - Execution time for each node
  - Peak memory usage per node
  - Active memory before each node execution
  - Memory sizes of activation tensors
  - Parameter and gradient sizes

- **Statistics Aggregation**: Processes raw data to calculate:
  - Median runtime and memory statistics
  - MuTWO metrics (inactive time, recomputation time, recomputation memory)
  - Memory breakdown (parameters, gradients, activations)

- **Reporting**: Provides visualization and data export:
  - CSV export of node and activation statistics
  - Memory curve plots
  - Runtime and memory usage plots

### 2. Batch Memory Analysis (`batch_memory_analysis.py`)

The `batch_memory_analysis.py` script demonstrates the profiler's capabilities by:

- Profiling ResNet-152 with different batch sizes (4, 8, 16, 32, 64)
- Generating visualizations:
  - Peak memory usage vs. batch size
  - Memory vs. execution rank for each batch size
  - Memory breakdown (weights, gradients, activations)
  - Latency vs. batch size comparison

### 3. Unit Tests (`tests/test_profiler.py`)

The test suite verifies the correctness of the profiler implementation:

- `test_activation_liveness`: Ensures activation liveness information is correct
- `test_memory_conservation`: Verifies memory breakdown adds up to peak memory
- `test_csv_schema`: Checks that CSV files have required headers and no NaNs

## Implementation Details

### Node Types

The profiler categorizes nodes into the following types:

- `PARAM`: Model parameters
- `ACT`: Activations (tensors created in forward pass and used in backward pass)
- `GRAD`: Gradients of parameters
- `OPT_STATE`: Optimizer state tensors
- `OTHER`: Other tensors and operations

### Memory Tracking

The profiler tracks several memory metrics:

- **Peak Memory**: Maximum memory allocated during a node's execution
- **Active Memory**: Memory allocated before a node's execution
- **Memory Sizes**: Size of activation tensors

### CSV Output

The profiler generates two CSV files:

1. **Node Statistics** (`profiler_stats_node_stats.csv`):
   - `rank`: Execution rank
   - `node_name`: Node identifier
   - `node_type`: Type of node (parameter, activation, gradient, etc.)
   - `gtype`: Graph type (forward, backward, other)
   - `median_run_time_s`: Median execution time in seconds
   - `median_peak_mem_bytes`: Median peak memory in bytes
   - `median_active_mem_bytes`: Median active memory in bytes
   - `device`: Device where the node is executed

2. **Activation Statistics** (`profiler_stats_activation_stats.csv`):
   - `activation_name`: Activation identifier
   - `creation_rank`: Rank where the activation is created
   - `last_fw_use_rank`: Rank of last use in forward pass
   - `first_bw_use_rank`: Rank of first use in backward pass
   - `last_bw_use_rank`: Rank of last use in backward pass
   - `median_mem_size_bytes`: Median memory size in bytes
   - `inactive_time_s`: Time between last forward use and first backward use
   - `recomp_time_s`: Estimated time to recompute the activation
   - `recomp_memory_bytes`: Memory required for recomputation

## How to Run

### Profiling a Model

To profile a model, use the `GraphProfiler` class:

```python
# Initialize the profiler with a graph module
profiler = GraphProfiler(graph_module)

# Run the profiler
profiler.run(*args)

# Aggregate statistics
profiler.aggregate_stats()

# Save statistics to CSV
profiler.save_stats_to_csv()

# Generate plots
profiler.plot_stats()
```

### Running Batch Memory Analysis

To analyze memory usage across different batch sizes:

```bash
conda run -n ml_env python starter_code/batch_memory_analysis.py
```

This will generate CSV files and plots in the `reports/` directory.

### Running Tests

To run the test suite:

```bash
conda run -n ml_env python tests/test_profiler.py
```

## Validation

To validate the Stage 1 implementation, run:

```bash
./stage1_validate.sh
```

This script will:
1. Run the batch memory analysis
2. Check for errors
3. Verify that all required files are generated

## Next Steps

The profiling data collected in Stage 1 will be used in Stage 2 to make activation checkpointing decisions. The CSV files contain all the necessary information for the activation checkpointing algorithm to determine which activations to checkpoint and which to recompute.