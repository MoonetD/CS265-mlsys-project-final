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
* 2025-05-12 18:18:20 - Completed Task: Static Analysis (`GraphProfiler.__init__`) in `starter code/graph_prof.py` including:
    * Identified Forward/Backward Boundary.
    * Categorized Nodes/Tensors (PARAM, ACT, GRAD, OTHER).
    * Performed Activation Liveness Analysis (creation, last_fw_use, first_bw_use, last_bw_use).
* [2025-05-12 18:20:37] - In Progress: Task 2: Run-time Profiling (`GraphProfiler.run_node`) from `PLAN_stage_1.md`.
    * Implemented Timing: Using `torch.cuda.Event` for `run_time`.
    * Implemented Memory Measurement: Using `torch.cuda.max_memory_allocated` for `peak_mem` and tensor properties for `memory_size`.
    * Implemented Swap Simulation: Logic for `swap_time` based on estimated bandwidth and liveness.
* [2025-05-12 18:22:51] - Completed Task: Task 3: Calculate MuTWO Metrics (`inactive_time`, `recomp_time`, `recomp_memory`, `recompute_ratio`) in `GraphProfiler.aggregate_stats` as per `PLAN_stage_1.md`, Section 3. Implemented in [`starter code/graph_prof.py`](starter code/graph_prof.py:249).
* [2025-05-12 18:27:25] - Completed Task: Task 4: Statistics Aggregation & Reporting (`aggregate_stats`, `reset_stats`, `print_stats`) from `PLAN_stage_1.md`, Section 4. Implemented in [`starter code/graph_prof.py`](starter code/graph_prof.py).
* [2025-05-12 18:29:45] - Completed Task: Task 5: Integrate `GraphProfiler` into example scripts (`starter code/starter_code.py`, `starter code/benchmarks.py`) as per `PLAN_stage_1.md`, Section 5. Updated `graph_transformation` to call `aggregate_stats` correctly.
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