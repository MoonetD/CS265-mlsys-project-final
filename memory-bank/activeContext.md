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
* [2025-05-13 10:04:22] - Current Focus: Implementing Activation Checkpointing Algorithm (Phase 2).
* [2025-05-13 10:04:22] - Recent Changes: Created initial structure for `ActivationCheckpointingAlgorithm` in `starter code/activation_checkpointing.py`. Includes data loading from profiler CSVs, methods for calculating recompute/swap overheads, a node execution order helper, and initial (simplified) implementations of `_simulate_memory_usage` and `decide_checkpoints` based on Algorithm B from the μ-TWO paper. Added an example usage block.
* [2025-05-13 10:09:59] - Current Focus: Refining `ActivationCheckpointingAlgorithm`.
* [2025-05-13 10:09:59] - Recent Changes: Updated `_simulate_memory_usage` in `starter code/activation_checkpointing.py` to more accurately reflect Algorithm G from the μ-TWO paper. The simulation now tracks peak memory by considering `fixed_overhead_bytes`, memory of live checkpointed activations, and `avg_peak_mem_node` from profiler data. Total execution time calculation sums base node run times and recomputation times for 'RECOMPUTE'd activations.
* [2025-05-13 10:12:49] - Current Focus: Finalizing `ActivationCheckpointingAlgorithm` refinements.
* [2025-05-13 10:12:49] - Recent Changes:
    * Corrected column name references in `_calculate_recompute_overhead`, `_calculate_swap_overhead`, `_simulate_memory_usage`, and `decide_checkpoints` within `starter code/activation_checkpointing.py` to match `profiler_stats_activation_stats.csv` (e.g., `recomp_time` to `recomp_time_s`, `avg_memory_size` to `avg_mem_size_bytes`).
    * Modified `_simulate_memory_usage` to identify activations created by a forward node by matching the node's rank with the activation's `creation_rank`, removing dependency on the non-existent `producing_node_name` column.
* [2025-05-13 10:29:22] - Debug Status Update: Resolved `ValueError` in `starter code/activation_checkpointing.py` by correcting 'act_name' to 'activation_name' to match `profiler_stats_activation_stats.csv`.
* [2025-05-13 10:34:05] - Debug Status Update: Resolved `KeyError: 'avg_run_time'` in `starter code/activation_checkpointing.py` by correcting the column name to 'avg_run_time_s' to match `profiler_stats_node_stats.csv` in the `_simulate_memory_usage` method.
* [2025-05-13 10:45:24] - Current Focus: Validating activation checkpointing recomputation logic under memory pressure.
* [2025-05-13 10:45:24] - Recent Changes:
    * Modified `starter code/activation_checkpointing.py` to test with `memory_budget_gb = 0.05` and `fixed_overhead_gb = 0.1`.
    * Executed the script.
    * **Outcome:**
        * Budget (0.05GB) < Fixed Overhead (0.1GB) correctly triggered recomputations.
        * 309 activations set to RECOMPUTE, 311 to CHECKPOINT.
        * Estimated Peak GPU Memory remained at 0.10 GB throughout, equal to fixed_overhead.
        * Estimated Total Execution Time increased from 0.92s to 1.22s.
        * All evicted activations had a recomputation `ratio: 0.000000`, suggesting their `recomp_time_s` in the CSV might be 0.
    * **Next Steps:** Investigate why peak memory doesn't decrease and why eviction ratios are zero. This might point to issues in `profiler_stats_activation_stats.csv` data or the simulation logic's sensitivity to it.
* [2025-05-13 11:05:00] - Debug Status Update: Resumed debugging `activation_checkpointing.py`.
    * Investigated zero eviction ratio: Confirmed `recomp_time_s` is often 0 in `profiler_stats_activation_stats.csv`.
    * Fixed eviction logic in `decide_checkpoints` by removing incorrect skip for 0-cost activations and adding benefit-based tie-breaking for 0-ratio candidates.
    * Investigated peak memory simulation: Confirmed `_simulate_memory_usage` correctly excludes `RECOMPUTE`d activations from checkpointed memory. Observed peak memory correctly reflects `fixed_overhead_gb` when it's the dominant factor or exceeds budget. No code changes made to this part.
    * Retested with `memory_budget_gb = 0.05`, `fixed_overhead_gb = 0.1`.
        * Output showed improved eviction decisions (benefit tie-breaking visible).
        * Simulated peak memory correctly reported 0.10 GB (the fixed overhead, which was > budget).
        * Script ran to completion, producing a schedule where all 620 activations were set to RECOMPUTE (correct for budget < fixed_overhead).
        * Execution time increased to 1.22s due to recomputations.
    * Encountered NumPy version incompatibility (1.x vs 2.2.5) causing import errors with pandas/pyarrow, though script execution proceeded. User has been informed.
    * Scope conformance for Phase 2 (activation checkpointing algorithm) verified.