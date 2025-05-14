# Midway Review Checklist

This document outlines the deliverables required for the midway review as specified in the project requirements.

## Phase 1: Graph Profiler Completion

- [x] Implemented a PyTorch-based computation-graph profiler that:
  - [x] Traces every operation in a forward-backward-optimizer iteration
  - [x] Records per-operation runtime and memory usage
  - [x] Tags each tensor as parameter / gradient / activation / optimizer-state / other
  - [x] Performs static first-use / last-use analysis on activations
  - [x] Produces a peak-memory breakdown graph

## Verification Checklist for Stage 1

- [x] **Trace completeness**: Graph includes all forward, backward, and optimizer operations for one iteration
- [x] **Node metadata**: For every operation, `runtime_ms`, `active_mem_B`, and `peak_mem_B` are logged with sensible values
- [x] **Tensor labelling**: 100% of tensors are classified into parameter / gradient / activation / optimizer-state / other
- [x] **Static analysis**: For several random activations, 'first-fw-use' and 'last-bw-use' indices match manual inspection
- [x] **Peak-memory plot** renders and shows the characteristic hump (grows through forward pass, falls through backward pass)
- [x] **Unit test**: Running the profiler on a toy 3-layer MLP produces exactly 3 forward + 3 backward nodes and the expected memory curve
- [ ] **Code quality**: Code passes lint, README reproduces results in < 5 min on reference GPU

## Required Deliverables for Midway Review

- [x] **Profiler source code & unit tests**:
  - [x] `graph_prof.py`: Main profiler implementation
  - [x] `test_profiler_mlp.py`: Unit test with toy 3-layer MLP

- [x] **CSV files with collected metrics**:
  - [x] `profiler_stats_node_stats.csv`: Per-node statistics
  - [x] `profiler_stats_activation_stats.csv`: Per-activation statistics
  - [x] ResNet-152 profiling data
  - [ ] BERT-Base profiling data (Note: BERT profiling encountered errors with `aten._local_scalar_dense.default`)

- [x] **Peak-memory breakdown graphs**:
  - [x] `profiler_plots_node_runtime.png`: Node runtime plot
  - [x] `profiler_plots_node_peak_memory.png`: Node peak memory plot
  - [x] `profiler_plots_activation_memory_size.png`: Activation memory size plot
  - [x] `profiler_plots_activation_inactive_time.png`: Activation inactive time plot
  - [x] `profiler_plots_memory_vs_rank.png`: Memory vs. execution rank plot with FW/BW separators and GPU memory limit

- [ ] **Documentation**:
  - [ ] README with build and run instructions
  - [ ] Slide/demo clip showing profiler output on a mini-batch

## Experimental Analysis

- [x] **Computation and memory profiling statistics**:
  - [x] Per-node runtime and peak memory statistics
  - [x] Per-activation memory size, inactive time, swap time, and recomputation metrics
  - [x] Overall execution time and peak memory breakdown

- [ ] **Peak memory consumption vs. mini-batch size bar graph (without AC)**:
  - [ ] Need to generate this graph by running the profiler with different batch sizes

## Next Steps

- [ ] Complete any remaining items in the midway review checklist
- [ ] Proceed to Phase 2: Activation Checkpointing algorithm implementation
- [ ] Prepare for the final review with all deliverables from Phases 1, 2, and 3