# Plan - Stage 2: Activation Checkpointing Algorithm Implementation

This stage focuses on implementing the core activation checkpointing algorithm (Scheduler logic from the Π-TWO paper, specifically Algorithm B) which decides which activations to keep in memory and which to recompute based on the profiling data gathered in Stage 1. This will be implemented as a separate function or class that operates on the profiling data.

## Tasks

### 1. Algorithm Input Preparation

*   [ ] **Define Input:** Determine the precise input format for the algorithm. This should include:
    *   The profiling data from Stage 1 (node statistics and activation statistics).
    *   The target GPU memory limit (`memory_budget`).
    *   All necessary profiling attributes (e.g., `recomp_time`, `run_time`, `memory_size`, `first_bw_use_rank`, `last_fw_use_rank`, etc.).
*   [ ] **Data Access:** Ensure the algorithm can easily access all required attributes for any given activation candidate or graph node.

### 2. Implement Core Scheduling Logic (Algorithm B)

*   [ ] **Initialization:**
    *   Initialize `recomps = {}` (set of activations to be recomputed).
    *   Initialize all activations as checkpointed by default.
*   [ ] **Main Loop (`while candidate_set != ∅`):**
    *   [ ] **Select Candidates:**
        *   Find recompute candidate `r_cand`: Activation in `candidate_set` with the maximum `recompute_ratio` (memory saved / recomputation time).
    *   [ ] **Calculate Overheads:**
        *   Implement `RecomputeOverhead(r_cand)` function (returns `r_cand.recomp_time` or a refined calculation).
    *   [ ] **Make Decision:**
        *   Mark `r_cand` for recomputation (e.g., set status to `RECOMPUTE`).
        *   Add `r_cand` to `recomps`.
        *   Set `cand = r_cand`.
    *   [ ] **Update State:**
        *   Remove `cand` from `candidate_set`.
        *   Update attributes (like estimated memory savings) of remaining candidates based on the choice of `cand`.
    *   [ ] **Memory Simulation & Check:**
        *   Simulate the peak memory usage based on the current `recomps` decisions.
        *   Check if memory constraint is met; if so, break the loop.

### 3. Implement Recompute Overhead Calculation

*   [ ] **Function Signature:** `RecomputeOverhead(r_cand)` returning `r_overhead`.
*   [ ] **Calculation:** Return `r_cand.recomp_time` (the estimated time to execute the forward operations needed to recompute `r_cand`).

### 4. Memory Simulation

*   [ ] **Function Signature:** `SimulateMemoryUsage(recomps, node_stats, activation_stats)` returning `peak_memory`.
*   [ ] **Initialization:** Set up data structures to track live activations and memory usage during simulation.
*   [ ] **Execution Simulation:** Iterate through nodes in execution order (forward and backward passes).
*   [ ] **Memory Tracking:**
    *   For each forward node, add its output activations to the live set if they're not marked for recomputation.
    *   For each backward node, remove activations from the live set if they're no longer needed.
    *   Track the peak memory usage throughout the simulation.
*   [ ] **Return Value:** Return the peak memory usage observed during the simulation.

### 5. Algorithm Output

*   [ ] **Define Output:** Determine the output format. This should clearly indicate for each activation whether it should be kept or recomputed. This could be:
    *   A dictionary mapping activation names to their status (CHECKPOINT or RECOMPUTE).
    *   A CSV file with activation decisions for easier analysis and debugging.

### 6. Testing & Validation

*   [ ] Create test cases with simple graphs and known profiling data.
*   [ ] Manually trace the algorithm's decisions for these test cases.
*   [ ] Test the algorithm with the actual profiled data from Stage 1.
*   [ ] Verify the output decisions seem logical based on the input metrics (`recompute_ratio`, `memory_budget`).
*   [ ] Validate that the algorithm correctly reduces memory usage below the specified limit when possible.
*   [ ] Measure the impact on execution time due to recomputation overhead.

### 7. Integration with Stage 3

*   [ ] Ensure the output format is compatible with the Graph Rewriter in Stage 3.
*   [ ] Provide clear documentation on how the recomputation decisions should be interpreted and implemented by the Graph Rewriter.
*   [ ] Consider adding a validation step to verify that the Graph Rewriter correctly implements the recomputation decisions.
*   [ ] Create a unified interface that allows Stage 3 to easily access the decisions made in Stage 2.