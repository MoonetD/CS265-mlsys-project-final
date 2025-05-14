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
* 2025-05-12 18:18:08 - Recent Changes: Added logic for boundary detection, node categorization, and activation liveness to `GraphProfiler.__init__` in `starter_code/graph_prof.py`.
* [2025-05-12 18:20:27] - Current Focus: Implementing run-time profiling in `GraphProfiler.run_node` as per `PLAN_stage_1.md`, Section 2.
* [2025-05-12 18:20:27] - Recent Changes: Added logic to `GraphProfiler.run_node` in `starter_code/graph_prof.py` for:
    * Timing node execution using `torch.cuda.Event`.
    * Measuring peak memory during node execution using `torch.cuda.max_memory_allocated()` after `torch.cuda.reset_peak_memory_stats()`.
    * Measuring output activation tensor sizes.
    * Simulating swap-in/swap-out times for activations based on pre-defined bandwidth estimates and liveness data.
    * Initialized related storage attributes in `__init__` and clearing in `reset_stats`.
* [2025-05-12 18:22:40] - Recent Changes: Implemented MuTWO metric calculations (`inactive_time`, `recomp_time`, `recomp_memory`, `recompute_ratio`) within the `aggregate_stats` method of `GraphProfiler` in [`starter_code/graph_prof.py`](starter_code/graph_prof.py:249). This includes dependency tracing for recomputation cost estimation.
* [2025-05-12 18:27:13] - Current Focus: Completed implementation of Statistics Aggregation & Reporting in `GraphProfiler`.
* [2025-05-12 18:27:13] - Recent Changes: Modified `GraphProfiler` in [`starter_code/graph_prof.py`](starter_code/graph_prof.py) to:
    * Accumulate raw runtime stats (`run_times`, `peak_mem_node`, `memory_sizes`, `swap_times`) as lists in `run_node`.
    * Calculate average statistics from these lists in `aggregate_stats`.
    * Update MuTWO metric calculations in `aggregate_stats` to use averaged values.
    * Implement `print_stats` to display averaged node stats, MuTWO metrics, total time, and estimated peak memory breakdown.
    * Update `reset_stats` to clear all raw and aggregated statistics.
* [2025-05-12 18:29:30] - Recent Changes: Integrated GraphProfiler into example scripts ([`starter_code/starter_code.py`](starter_code/starter_code.py:72), [`starter_code/benchmarks.py`](starter_code/benchmarks.py:111)) by updating the `graph_transformation` function to call `aggregate_stats` with `num_runs`.
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
* [2025-05-13 10:04:22] - Recent Changes: Created initial structure for `ActivationCheckpointingAlgorithm` in `starter_code/activation_checkpointing.py`. Includes data loading from profiler CSVs, methods for calculating recompute/swap overheads, a node execution order helper, and initial (simplified) implementations of `_simulate_memory_usage` and `decide_checkpoints` based on Algorithm B from the μ-TWO paper. Added an example usage block.
* [2025-05-13 10:09:59] - Current Focus: Refining `ActivationCheckpointingAlgorithm`.
* [2025-05-13 10:09:59] - Recent Changes: Updated `_simulate_memory_usage` in `starter_code/activation_checkpointing.py` to more accurately reflect Algorithm G from the μ-TWO paper. The simulation now tracks peak memory by considering `fixed_overhead_bytes`, memory of live checkpointed activations, and `avg_peak_mem_node` from profiler data. Total execution time calculation sums base node run times and recomputation times for 'RECOMPUTE'd activations.
* [2025-05-13 10:12:49] - Current Focus: Finalizing `ActivationCheckpointingAlgorithm` refinements.
* [2025-05-13 10:12:49] - Recent Changes:
    * Corrected column name references in `_calculate_recompute_overhead`, `_calculate_swap_overhead`, `_simulate_memory_usage`, and `decide_checkpoints` within `starter_code/activation_checkpointing.py` to match `profiler_stats_activation_stats.csv` (e.g., `recomp_time` to `recomp_time_s`, `avg_memory_size` to `avg_mem_size_bytes`).
    * Modified `_simulate_memory_usage` to identify activations created by a forward node by matching the node's rank with the activation's `creation_rank`, removing dependency on the non-existent `producing_node_name` column.
* [2025-05-13 10:29:22] - Debug Status Update: Resolved `ValueError` in `starter_code/activation_checkpointing.py` by correcting 'act_name' to 'activation_name' to match `profiler_stats_activation_stats.csv`.
* [2025-05-13 10:34:05] - Debug Status Update: Resolved `KeyError: 'avg_run_time'` in `starter_code/activation_checkpointing.py` by correcting the column name to 'avg_run_time_s' to match `profiler_stats_node_stats.csv` in the `_simulate_memory_usage` method.
* [2025-05-13 10:45:24] - Current Focus: Validating activation checkpointing recomputation logic under memory pressure.
* [2025-05-13 10:45:24] - Recent Changes:
    * Modified `starter_code/activation_checkpointing.py` to test with `memory_budget_gb = 0.05` and `fixed_overhead_gb = 0.1`.
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
* [2025-05-13 22:34:14] - Current Focus: Improving recomputation metrics calculation in `GraphProfiler`.
* [2025-05-13 22:34:14] - Recent Changes: Enhanced the recomputation metrics calculation in `starter_code/graph_prof.py` to avoid zero values:
    * Implemented a more robust dependency tracing approach that considers direct dependencies between creation and last use
    * Added size-based estimation that correlates activation size with computation cost
    * Applied a minimum threshold to ensure all activations have a non-zero recomputation cost
    * Improved the recompute ratio calculation with better handling of edge cases
    * Updated documentation to reflect these changes
* [2025-05-13 23:04:46] - Current Focus: Optimizing activation checkpointing algorithm in `starter_code/activation_checkpointing.py`.
* [2025-05-13 23:04:46] - Recent Changes: Enhanced the activation checkpointing algorithm with:
    * Added detailed progress reporting and debugging information
    * Implemented batch processing for faster convergence (evicting multiple activations per iteration)
    * Added pre-computation of benefit values and ratios to avoid redundant calculations
    * Optimized memory simulation with caching and faster lookups
    * Added command-line argument support for better usability
    * Added timing information to track performance
    * These improvements significantly reduce execution time and provide better visibility into the algorithm's progress
* [2025-05-13 23:10:01] - Current Focus: Implemented unit test for GraphProfiler with a toy 3-layer MLP model.
* [2025-05-13 23:10:01] - Recent Changes: Created `starter_code/test_profiler_mlp.py` that:
    * Sets up a simple 3-layer MLP model in PyTorch
    * Configures the GraphProfiler to trace this model
    * Verifies that it correctly identifies exactly 3 forward computational nodes (addmm operations) and at least 3 backward nodes
    * Checks that the memory curve shows the expected pattern (growing through forward pass, falling through backward pass)
    * Includes visualization of the memory curve and detailed verification logic
    * Provides clear output messages indicating test success/failure
    * Documents how to run the test using "conda run -n ml_env python"
    * These improvements will lead to better eviction decisions in the activation checkpointing algorithm in Stage 2
* [2025-05-13 23:27:33] - Current Focus: Created batch memory analysis script for ResNet-152.
* [2025-05-13 23:27:33] - Recent Changes: Implemented `starter_code/batch_memory_analysis.py` that:
    * Runs ResNet-152 with different batch sizes (4, 8, 16, 32)
    * Collects peak memory consumption using GraphProfiler
    * Generates a bar graph showing batch size vs. peak memory consumption
    * Saves the graph as a PNG file in the reports/ directory
    * Includes proper error handling and reporting
* [2025-05-13 23:36:07] - Current Focus: Successfully executed batch memory analysis script for ResNet-152.
* [2025-05-13 23:36:07] - Recent Changes: Fixed and executed `starter_code/batch_memory_analysis.py`:
    * Resolved issue with return value handling in `graph_transformation` function
    * Used global variable to store peak memory instead of returning it from the function
    * Successfully profiled ResNet-152 with batch sizes 4, 8, 16, 32
    * Generated bar graph showing batch size vs. peak memory consumption
    * Saved graph to `reports/resnet152_batch_memory.png`
    * Observed memory usage scaling from 1.4 GB (batch size 4) to 6.2 GB (batch size 32)
* [2025-05-13 23:46:48] - Current Focus: Enhanced batch memory analysis script to generate CSV files for Stage 2.
* [2025-05-13 23:46:48] - Recent Changes: Modified `starter_code/batch_memory_analysis.py` to:
    * Update the `graph_transformation` function to call `graph_profiler.save_stats_to_csv()` after aggregating stats
    * Use batch-size-specific prefixes for CSV files (e.g., `profiler_stats_bs{batch_size}`) to avoid overwriting
    * Save CSV files in the main directory to be consistent with existing CSV files
    * Update the main function to print information about the generated CSV files
    * These changes enable Stage 2 to use batch-specific profiling data for activation checkpointing analysis
* [2025-05-14 00:03:38] - Current Focus: Enhanced batch memory analysis script with comprehensive visualizations.
* [2025-05-14 00:03:38] - Recent Changes: Modified `starter_code/batch_memory_analysis.py` to:
    * Include batch size 64 in addition to existing sizes (4, 8, 16, 32)
    * Add an 8 GB (8192 MiB) OOM cap line to all visualizations
    * Create three types of visualizations:
        * Enhanced bar graph showing peak memory usage with the OOM cap line
        * Memory vs. execution rank graph for all batch sizes showing FW/BW boundaries and OOM cap
        * Stacked bar chart showing memory breakdown (weights, gradients, feature maps) for different batch sizes
    * Implement helper functions `create_memory_vs_rank_plots()` and `create_memory_breakdown_chart()`
    * Save all visualizations to the reports/ directory with appropriate filenames