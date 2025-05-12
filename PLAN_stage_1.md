# Plan - Stage 1: Graph Profiler Enhancements

This stage focuses on enhancing the `starter code/graph_prof.py` ([`starter code/graph_prof.py`](starter code/graph_prof.py:1)) to collect detailed static and run-time information required by the MuTWO activation checkpointing algorithm.

## Tasks

### 1. Static Analysis (`GraphProfiler.__init__`)

*   [ ] **Identify Forward/Backward Boundary:**
    *   Locate the `torch.ops.separator.sep.default` node marking the end of the forward pass.
    *   Locate the `torch.ops.separator.sep_backward.default` node marking the start of the backward pass.
    *   Store references to these boundary nodes or their positions (`rank`).
*   [ ] **Categorize Nodes/Tensors:**
    *   Implement logic to assign `NodeType` ([`starter code/graph_prof.py:17`](starter code/graph_prof.py:17)) (PARAM, ACT, GRAD, OTHER) to relevant nodes or the tensors they produce/represent.
    *   Use optimizer node (`torch.ops.aten._fused_adam.default`) args to identify PARAMs and GRADs.
    *   Identify ACTivations (intermediate tensors created in forward, used in backward, not PARAMs).
    *   Tag other nodes appropriately (e.g., inputs, optimizer states).
*   [ ] **Activation Liveness Analysis:**
    *   For each identified ACTivation tensor:
        *   Find the node representing its creation.
        *   Find the node representing its *last use* within the forward pass region (`last_fw_access`).
        *   Find the node representing its *first use* within the backward pass region (`first_bw_access`).
        *   Find the node representing its *last use* within the backward pass region (`last_bw_access`).
    *   Store these node references or ranks as attributes associated with the activation tensor/node (e.g., in a dictionary mapping tensor name/node to its analysis results).

### 2. Run-time Profiling (`GraphProfiler.run_node`)

*   [ ] **Implement Timing:**
    *   Use `torch.cuda.Event(enable_timing=True)` before and after the `super().run_node(n)` call ([`starter code/graph_prof.py:92`](starter code/graph_prof.py:92)) to record the execution time (`run_time`) for each node `n`.
    *   Store the recorded time associated with the node `n`.
*   [ ] **Implement Memory Measurement:**
    *   Use `torch.cuda.memory_stats()` or `torch.cuda.max_memory_allocated()` / `torch.cuda.memory_allocated()` appropriately around `super().run_node(n)` to measure:
        *   Peak memory during the node's execution (`peak_mem`).
        *   Memory occupied by the node's output tensor(s) (`memory_size` for activations).
        *   Potentially active memory (`active_mem` - may require careful definition/measurement).
    *   Store these memory values associated with the node `n` or its output tensors.
*   [ ] **Simulate Swapping & Measure Swap Time:**
    *   Implement the logic hinted at in `run_node` comments ([`starter code/graph_prof.py:87`](starter code/graph_prof.py:87), [`starter code/graph_prof.py:96`](starter code/graph_prof.py:96)):
        *   *Before* `super().run_node(n)`: If in backward pass and `n` requires an activation `x` that *would have been* swapped out, simulate swapping `x` *in*. Measure the time this *would* take (`swap_time` - potentially profile actual `tensor.to('cuda')` / `tensor.to('cpu')` on tensors of relevant sizes beforehand).
        *   *After* `super().run_node(n)`: If in forward pass and `n` is the `last_fw_access` for an activation `x`, simulate swapping `x` *out*. Measure the time this *would* take (contributes to `swap_time`).
    *   Store the measured `swap_time` associated with each activation tensor.

### 3. Calculate MuTWO Metrics

*   [ ] **Calculate `inactive_time`:** For each activation, calculate the time difference between its `last_fw_access` node's end time and its `first_bw_access` node's start time using the profiled `run_time` data.
*   [ ] **Calculate Recomputation Cost (Initial Estimate):**
    *   For each activation, determine the subgraph required to recompute it (`recomp_graph`) by tracing back dependencies in the forward pass graph.
    *   Estimate the time to recompute (`recomp_time`) by summing the `run_time` of nodes in `recomp_graph`. (Note: Actual recomputation might be faster/slower, this is an initial estimate).
    *   Estimate peak memory during recomputation (`recomp_memory`).
*   [ ] **Calculate `recompute_ratio`:** For each activation, calculate `memory_size / recomp_time`. (Refine `recomp_time` if a more accurate measure is implemented later).

### 4. Statistics Aggregation & Reporting

*   [ ] **Implement `aggregate_stats` ([`starter code/graph_prof.py:102`](starter code/graph_prof.py:102)):**
    *   Store run-time stats (time, memory) per node/tensor for each measurement iteration.
    *   Calculate the average (or median, as per MuTWO) `run_time`, `peak_mem`, etc., over the measurement iterations.
*   [ ] **Implement `reset_stats` ([`starter code/graph_prof.py:111`](starter code/graph_prof.py:111)):**
    *   Clear out statistics collected during warm-up iterations.
*   [ ] **Implement `print_stats` ([`starter code/graph_prof.py:108`](starter code/graph_prof.py:108)):**
    *   Output the collected and calculated statistics in a readable format (e.g., per-node times, activation liveness, inactive times, recompute ratios).
    *   Generate the peak memory breakdown graph required by project deliverables (Phase 1, 4a).

### 5. Integration & Testing

*   [ ] Integrate the profiler with the main training script (`starter_code.py` likely).
*   [ ] Run the profiler on the target models (ResNet-152, BERT) for a few iterations.
*   [ ] Verify the collected statistics seem reasonable.
*   [ ] Debug any issues in graph traversal, boundary detection, or metric calculation.