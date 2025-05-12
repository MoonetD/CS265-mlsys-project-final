# Decision Log

This file records architectural and implementation decisions using a list format.
2025-05-12 18:14:49 - Log of updates made.

*

## Decision

*

## Rationale 

*

## Implementation Details

*
---
### Decision (Code)
[2025-05-12 18:18:27] - Implemented static graph analysis in `GraphProfiler.__init__`.

**Rationale:**
A multi-pass approach was chosen for clarity and to allow information gathered in earlier passes (e.g., node ranks, boundary locations) to inform subsequent analyses (e.g., `NodeType` determination, liveness). Parameter identification primarily uses `module.named_parameters()` for robustness, with a fallback to optimizer node arguments. Gradient identification relies on the `_fused_adam` optimizer node's arguments as specified in comments.

**Details:**
The implementation involved three main passes over the graph nodes:
1.  **First Pass:** Assign ranks to all nodes, identify `sep` and `sep_backward` boundary nodes, and perform initial identification of parameter and gradient node names. Parameter names are sourced from `self.module.named_parameters()`. Gradient names (and potentially additional parameter names) are sourced from the arguments of the `_fused_adam` node if it's present and its arguments are `prim::ListConstruct` nodes.
2.  **Second Pass:** Determine the `NodeType` (PARAM, GRAD, ACT, OTHER) for each node based on information from the first pass and its usage context (e.g., created in forward, used in backward for ACT).
3.  **Third Pass:** For nodes identified as ACT (activations), calculate their liveness information: creation rank, last forward use rank, first backward use rank, and last backward use rank.

Reference: [`starter code/graph_prof.py`](starter code/graph_prof.py:35) (specifically the `__init__` method)
---
### Decision (Code)
[2025-05-12 18:20:45] - Implemented run-time profiling in `GraphProfiler.run_node`.

**Rationale:**
The implementation follows the specifications in `PLAN_stage_1.md`, Section 2, to collect timing, memory, and simulated swap metrics.
*   **Timing:** `torch.cuda.Event` is used for precise measurement of CUDA kernel execution times. Times are stored in seconds.
*   **Memory Measurement:**
    *   `peak_mem_node`: `torch.cuda.max_memory_allocated()` is used after calling `torch.cuda.reset_peak_memory_stats()` before each node's execution. This captures the peak memory usage on the device *during* that node's execution, assuming a single stream of execution for the profiled model.
    *   `memory_sizes`: For activations, `tensor.element_size() * tensor.nelement()` is used to get the size of the output tensor.
*   **Swap Time Simulation:**
    *   Swap-in/out times are estimated based on tensor sizes (obtained from `memory_sizes`) and pre-defined (currently constant) CPU-GPU and GPU-CPU bandwidth figures (`BYTES_PER_SEC_CPU_TO_GPU`, `BYTES_PER_SEC_GPU_TO_CPU`).
    *   A set `swapped_out_activations` tracks activations notionally moved to CPU memory.
    *   Swap-in is simulated before a node in the backward pass if it needs a swapped-out activation.
    *   Swap-out is simulated after a node in the forward pass if it's the last forward user of an activation.
    *   Cumulative `swap_times` are stored per activation.
*   **Storage:** New dictionaries (`run_times`, `peak_mem_node`, `memory_sizes`, `swap_times`) and a set (`swapped_out_activations`) are added to the `GraphProfiler` instance to store the collected data. These are cleared by `reset_stats`.

**Details:**
The changes were applied to the `__init__`, `run_node`, and `reset_stats` methods of the `GraphProfiler` class in [`starter code/graph_prof.py`](starter code/graph_prof.py:1). The `run_node` method now incorporates logic for event-based timing, peak memory capture, activation output size recording, and simulation of swap-in/out operations based on liveness information from static analysis.
---
### Decision (Code)
[2025-05-12 18:27:33] - Implemented statistics aggregation and reporting in `GraphProfiler`.

**Rationale:**
To support profiling over multiple iterations and provide clearer insights, the following approach was taken:
*   **Aggregation:** Raw runtime statistics (`run_times`, `peak_mem_node`, `memory_sizes`, `swap_times`) are collected as lists within `run_node` across multiple calls. The `aggregate_stats` method now calculates the mean of these lists (using `statistics.mean`) before computing derived metrics. This provides more stable results than single-run measurements. Swap times are aggregated by summing individual swap event times recorded in the list and dividing by the number of runs.
*   **MuTWO Metrics:** Calculations for `inactive_time`, `recomp_time`, `recomp_memory`, and `recompute_ratio` in `aggregate_stats` were updated to use the newly calculated *average* statistics (`avg_run_times`, `avg_memory_sizes`, `avg_swap_times`). The recomputation cost approximation (summing times from creation to last forward use) remains but now uses averaged times.
*   **Reporting (`print_stats`):** A dedicated method was implemented to display the results clearly. It includes:
    *   A table of average per-node run times and peak memory usage.
    *   A table of per-activation MuTWO metrics, including average memory size, inactive time, average swap time, recomputation time, recomputation memory, and the recompute ratio.
    *   Overall total estimated execution time (sum of average node times).
    *   An estimated peak memory breakdown, calculating peak concurrent activation memory based on liveness and average sizes, and including placeholders for parameter, gradient, and optimizer state memory (which require further implementation to track accurately). A helper function `format_bytes` was added for readability.
*   **Reset (`reset_stats`):** The method was updated to clear all raw statistic lists, averaged statistic dictionaries, and MuTWO metric dictionaries to prepare for new profiling runs.

**Details:**
Changes were applied to `__init__`, `run_node`, `aggregate_stats`, `reset_stats`, and `print_stats` methods of the `GraphProfiler` class in [`starter code/graph_prof.py`](starter code/graph_prof.py). Imports for `statistics`, `defaultdict`, and `math` were added. The `aggregate_stats` method now accepts an optional `num_runs` argument for correct swap time averaging.
## [2025-05-12 18:53:14] DevOps Task: Profiler Enhancement for Stage 1 - Decisions
- **Decision:** Modified `GraphProfiler` in `starter_code/graph_prof.py` to output profiling statistics to CSV files.
    - **Reason:** Project requirements ([`Material Markdown/Project Requirement.md:100`](Material%20Markdown/Project%20Requirement.md:100)) mandate computation and memory profiling statistics. CSV is a structured and accessible format.
    - **Files created:** `profiler_stats_node_stats.csv`, `profiler_stats_activation_stats.csv`.
- **Decision:** Added plotting capabilities to `GraphProfiler` using `matplotlib`.
    - **Reason:** Project requirements ([`Material Markdown/Project Requirement.md:111`](Material%20Markdown/Project%20Requirement.md:111)) and user request for visual representation of profiling data.
    - **Plots created:** Node runtime, node peak memory, activation memory size, activation inactive time, and a new "Memory vs. Execution Rank" plot.
- **Decision:** The "Memory vs. Execution Rank" plot includes:
    - X-axis: Node execution rank (topological order).
    - Y-axis: Peak memory per node (MiB).
    - Vertical lines for FW/BW separation (derived from `sep_fw_end_rank` and `sep_bw_start_rank`).
    - Horizontal line for GPU memory limit (configurable, set to 40GiB).
    - **Reason:** User explicitly requested these features for better visualization of memory behavior across the execution graph.
- **Decision:** Used `conda run -n ml_env python ...` for script execution.
    - **Reason:** User feedback indicated potential issues with `conda activate ... && python ...` in a single command. `conda run` provides a more robust way to execute within a specific environment.
- **Decision:** Persisted with `write_to_file` for `starter_code/graph_prof.py` after multiple `insert_content` and `apply_diff` attempts failed to resolve indentation issues.
    - **Reason:** To ensure consistent and correct indentation throughout the class definition after repeated Pylance errors.
## [2025-05-12 19:12:23] DevOps Task: Profiler Model Switching & BERT Debugging - Decisions
- **Decision:** Switched profiler from `DummyModel` to ResNet-152 and BERT.
    - **Reason:** As per project requirements ([`Material Markdown/Project Requirement.md:91-93`](Material%20Markdown/Project%20Requirement.md:91-93)).
- **Decision:** Changed profiler aggregation from `mean` to `median` for runtime statistics.
    - **Reason:** Align with paper specification ([`Material Markdown/Paper.md:157`](Material%20Markdown/Paper.md:157)).
- **Decision:** Added `gtype` (forward/backward/other) to node profiling and CSV output.
    - **Reason:** To meet Table A requirements from [`Material Markdown/Paper.md`](Material%20Markdown/Paper.md) more closely.
- **Decision:** Attempted multiple strategies to resolve `aten._local_scalar_dense.default` error for BERT profiling:
    1.  `BertWrapper.forward` returning `last_hidden_state.mean()`, with generic `train_step` calling `.sum()` on it.
    2.  `BertWrapper.forward` returning full `last_hidden_state`, with `bert_train_step_wrapper` explicitly calculating `loss = last_hidden_state.mean()` and handling backward/optimizer steps.
    3.  `BertWrapper.forward` returning full `last_hidden_state`, with `bert_train_step_wrapper` calculating `loss = last_hidden_state.sum(dim=-1).mean()` and handling backward/optimizer steps.
    - **Outcome:** All attempts failed to resolve the error for BERT. ResNet-152 profiling was successful.
- **Decision:** Proceed with Stage 2 development focusing on ResNet-152 data due to persistent BERT profiling issues.
    - **Reason:** Pragmatic approach to maintain project momentum. BERT debugging for FX tracing is complex and can be deferred.
- **Decision:** Installed `transformers` library in `ml_env`.
    - **Reason:** `ModuleNotFoundError` indicated it was missing.