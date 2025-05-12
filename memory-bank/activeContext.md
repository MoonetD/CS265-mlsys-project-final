# Active Context

  This file tracks the project's current status, including recent changes, current goals, and open questions.
  2025-05-12 18:14:34 - Log of updates made.

*

## Current Focus

*   

## Recent Changes

*   

## Open Questions/Issues

*
* 2025-05-12 18:18:08 - Current Focus: Implemented static analysis in `GraphProfiler.__init__`.
* 2025-05-12 18:18:08 - Recent Changes: Added logic for boundary detection, node categorization, and activation liveness to `GraphProfiler.__init__` in `starter code/graph_prof.py`.
* [2025-05-12 18:20:27] - Current Focus: Implementing run-time profiling in `GraphProfiler.run_node` as per `PLAN_stage_1.md`, Section 2.
* [2025-05-12 18:20:27] - Recent Changes: Added logic to `GraphProfiler.run_node` in `starter code/graph_prof.py` for:
    * Timing node execution using `torch.cuda.Event`.
    * Measuring peak memory during node execution using `torch.cuda.max_memory_allocated()` after `torch.cuda.reset_peak_memory_stats()`.
    * Measuring output activation tensor sizes.
    * Simulating swap-in/swap-out times for activations based on pre-defined bandwidth estimates and liveness data.
    * Initialized related storage attributes in `__init__` and clearing in `reset_stats`.
* [2025-05-12 18:22:40] - Recent Changes: Implemented MuTWO metric calculations (`inactive_time`, `recomp_time`, `recomp_memory`, `recompute_ratio`) within the `aggregate_stats` method of `GraphProfiler` in [`starter code/graph_prof.py`](starter code/graph_prof.py:249). This includes dependency tracing for recomputation cost estimation.
* [2025-05-12 18:27:13] - Current Focus: Completed implementation of Statistics Aggregation & Reporting in `GraphProfiler`.
* [2025-05-12 18:27:13] - Recent Changes: Modified `GraphProfiler` in [`starter code/graph_prof.py`](starter code/graph_prof.py) to:
    * Accumulate raw runtime stats (`run_times`, `peak_mem_node`, `memory_sizes`, `swap_times`) as lists in `run_node`.
    * Calculate average statistics from these lists in `aggregate_stats`.
    * Update MuTWO metric calculations in `aggregate_stats` to use averaged values.
    * Implement `print_stats` to display averaged node stats, MuTWO metrics, total time, and estimated peak memory breakdown.
    * Update `reset_stats` to clear all raw and aggregated statistics.
* [2025-05-12 18:29:30] - Recent Changes: Integrated GraphProfiler into example scripts ([`starter code/starter_code.py`](starter code/starter_code.py:72), [`starter code/benchmarks.py`](starter code/benchmarks.py:111)) by updating the `graph_transformation` function to call `aggregate_stats` with `num_runs`.
* [2025-05-12 18:29:30] - Current Focus: Completed Stage 1 tasks. Ready for Stage 2 or next defined task.
## [2025-05-12 18:53:30] DevOps Task: Profiler Enhancement for Stage 1 - Active Context
- **Current Status:** Stage 1 (Graph Profiler) deliverables completed.
- **Generated Artifacts:**
    - `profiler_stats_node_stats.csv`
    - `profiler_stats_activation_stats.csv`
    - `profiler_plots_node_runtime.png`
    - `profiler_plots_node_peak_memory.png`
    - `profiler_plots_activation_memory_size.png`
    - `profiler_plots_activation_inactive_time.png`
    - `profiler_plots_memory_vs_rank.png`
- **Key Observations:**
    - Profiler now outputs detailed static and runtime statistics to CSVs.
    - Profiler generates required plots, including the new "Memory vs. Execution Rank" plot with FW/BW separators and GPU memory limit.
    - CSVs contain most attributes from Paper Tables A & B, sufficient for starting Stage 2. Some advanced attributes for specific AC algorithms (e.g., precise recomputation sources, active memory) might need later addition.
- **Pending:** Review Stage 2 plan and begin implementation.