# Progress

This file tracks the project's progress using a task list format.
2025-05-12 18:14:41 - Log of updates made.

*

## Completed Tasks

*   

## Current Tasks

*   

## Next Steps

*
* 2025-05-12 18:18:20 - Completed Task: Static Analysis (`GraphProfiler.__init__`) in `starter_code/graph_prof.py` including:
    * Identified Forward/Backward Boundary.
    * Categorized Nodes/Tensors (PARAM, ACT, GRAD, OTHER).
    * Performed Activation Liveness Analysis (creation, last_fw_use, first_bw_use, last_bw_use).
* [2025-05-12 18:20:37] - In Progress: Task 2: Run-time Profiling (`GraphProfiler.run_node`) from `PLAN_stage_1.md`.
    * Implemented Timing: Using `torch.cuda.Event` for `run_time`.
    * Implemented Memory Measurement: Using `torch.cuda.max_memory_allocated` for `peak_mem` and tensor properties for `memory_size`.
    * Implemented Swap Simulation: Logic for `swap_time` based on estimated bandwidth and liveness.
* [2025-05-12 18:22:51] - Completed Task: Task 3: Calculate MuTWO Metrics (`inactive_time`, `recomp_time`, `recomp_memory`, `recompute_ratio`) in `GraphProfiler.aggregate_stats` as per `PLAN_stage_1.md`, Section 3. Implemented in [`starter_code/graph_prof.py`](starter_code/graph_prof.py:249).
* [2025-05-12 18:27:25] - Completed Task: Task 4: Statistics Aggregation & Reporting (`aggregate_stats`, `reset_stats`, `print_stats`) from `PLAN_stage_1.md`, Section 4. Implemented in [`starter_code/graph_prof.py`](starter_code/graph_prof.py).
* [2025-05-12 18:29:45] - Completed Task: Task 5: Integrate `GraphProfiler` into example scripts (`starter_code/starter_code.py`, `starter_code/benchmarks.py`) as per `PLAN_stage_1.md`, Section 5. Updated `graph_transformation` to call `aggregate_stats` correctly.
## [2025-05-12 18:52:56] DevOps Task: Profiler Enhancement for Stage 1
- **Status:** Success
- **Summary:** Modified `starter_code/graph_prof.py` to save detailed node and activation statistics to CSV files (`profiler_stats_node_stats.csv`, `profiler_stats_activation_stats.csv`). Added various plots including node runtime, node peak memory, activation memory size, activation inactive time, and a new "Memory vs. Execution Rank" plot with FW/BW separators and GPU memory limit.
- **Outputs:**
    - `profiler_stats_node_stats.csv`
    - `profiler_stats_activation_stats.csv`
    - `profiler_plots_node_runtime.png`
    - `profiler_plots_node_peak_memory.png`
    - `profiler_plots_activation_memory_size.png`
    - `profiler_plots_activation_inactive_time.png`
    - `profiler_plots_memory_vs_rank.png`
- **Next Steps:** Stage 1 deliverables are complete. Ready to review Stage 2 plan and begin implementation.
## [2025-05-12 19:12:03] DevOps Task: Profiler Model Switching & Refinement
- **Status:** Partially Successful
- **Summary:**
    - Modified `starter_code/graph_prof.py` to use median for aggregated statistics.
    - Modified `starter_code/starter_code.py` to use ResNet-152 and BERT models.
    - ResNet-152 profiling completed successfully, generating CSVs and plots.
    - BERT profiling consistently fails with `aten._local_scalar_dense.default` error despite several attempts to adjust loss calculation and wrapper logic.
- **Outputs (ResNet-152):**
    - `profiler_stats_node_stats.csv` (reflects ResNet-152 run)
    - `profiler_stats_activation_stats.csv` (reflects ResNet-152 run)
    - `profiler_plots_node_runtime.png` (ResNet-152)
    - `profiler_plots_node_peak_memory.png` (ResNet-152)
    - `profiler_plots_activation_memory_size.png` (ResNet-152)
    - `profiler_plots_activation_inactive_time.png` (ResNet-152)
    - `profiler_plots_memory_vs_rank.png` (ResNet-152)
- **Issues:**
    - BERT profiling blocked by `aten._local_scalar_dense.default` error.
- **Next Steps:** Proceed with Stage 2 development focusing on ResNet-152 data. Defer further BERT debugging unless critical.
* [2025-05-13 10:04:32] - Current Task: Phase 2: Activation Checkpointing Algorithm Implementation.
* [2025-05-13 10:04:32] - Progress: Created `starter_code/activation_checkpointing.py` with the `ActivationCheckpointingAlgorithm` class.
    * Implemented initial data loading from profiler CSVs.
    * Added helper methods for recompute/swap overhead.
    * Implemented a simplified `_simulate_memory_usage` function.
    * Implemented a version of `decide_checkpoints` based on μ-TWO paper's Algorithm B.
    * Added an example `if __name__ == "__main__":` block for testing.
* [2025-05-13 10:10:10] - Progress: Refined `_simulate_memory_usage` in `ActivationCheckpointingAlgorithm` ([`starter_code/activation_checkpointing.py`](starter%20code/activation_checkpointing.py)) to better align with μ-TWO's Algorithm G. This includes improved peak memory tracking and execution time calculation.
* [2025-05-13 10:13:01] - Progress: Further refined `ActivationCheckpointingAlgorithm` in [`starter_code/activation_checkpointing.py`](starter%20code/activation_checkpointing.py):
    * Corrected all identified CSV column name mismatches (e.g., `recomp_time` to `recomp_time_s`).
    * Updated `_simulate_memory_usage` to use `creation_rank` for identifying when activations are created, removing the need for a `producing_node_name` column.
* [2025-05-13 11:06:00] - Completed Task: Debug `activation_checkpointing.py` for zero eviction ratio and peak memory simulation.
    * Fixed zero eviction ratio by correcting logic in `decide_checkpoints` to consider 0-cost activations and use benefit for tie-breaking.
    * Verified peak memory simulation in `_simulate_memory_usage` correctly handles `RECOMPUTE`d activations and reflects memory savings.
    * Retested with constrained budget (`memory_budget_gb = 0.05`, `fixed_overhead_gb = 0.1`), confirming fixes and observing expected behavior (all activations recomputed, peak memory at fixed overhead level).
    * Verified Phase 2 scope conformance.
    * Noted NumPy environment incompatibility for future resolution.
* [2025-05-13 23:05:17] - Completed Task: Improved recomputation metrics calculation and optimized activation checkpointing algorithm:
    * Enhanced the recomputation metrics calculation in `starter_code/graph_prof.py` to avoid zero values
    * Implemented a multi-method approach for recomputation time estimation including dependency tracing, size-based estimation, and minimum thresholds
    * Optimized the activation checkpointing algorithm in `starter_code/activation_checkpointing.py` with:
        * Batch processing for faster convergence
        * Pre-computation of benefit values and ratios
        * Detailed progress reporting and debugging information
        * Command-line argument support for better usability
        * Performance optimizations for faster execution
    * These improvements lead to better eviction decisions in the activation checkpointing algorithm and significantly improved performance
* [2025-05-13 22:34:45] - Completed Task: Improved recomputation metrics calculation in `GraphProfiler`:
    * Enhanced the implementation in `starter_code/graph_prof.py` to avoid zero values in `recomp_time_s`
    * Implemented a multi-method approach including dependency tracing, size-based estimation, and minimum thresholds
    * Improved the recompute ratio calculation with better handling of edge cases
* [2025-05-13 23:09:48] - Completed Task: Implemented unit test for GraphProfiler with toy MLP model:
    * Created `starter_code/test_profiler_mlp.py` with a simple 3-layer MLP model
    * Verified that GraphProfiler correctly identifies forward and backward nodes
    * Confirmed that the memory curve shows the expected pattern (growing through forward pass, falling through backward pass)
    * Added visualization of the memory curve and detailed verification logic
    * Implemented clear output messages indicating test success/failure
    * These improvements will lead to better eviction decisions in the activation checkpointing algorithm in Stage 2