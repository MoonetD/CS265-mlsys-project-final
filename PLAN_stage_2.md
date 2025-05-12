# Plan - Stage 2: Activation Checkpointing Algorithm Implementation

This stage focuses on implementing the core MuTWO activation checkpointing algorithm (Scheduler logic from the paper, specifically Algorithm B) which decides which activations to keep, swap, or recompute based on the profiling data gathered in Stage 1. This will likely be implemented as a separate function or class that operates on the profiled `fx.Graph`.

## Tasks

### 1. Algorithm Input Preparation

*   [ ] **Define Input:** Determine the precise input format for the algorithm. This should include:
    *   The profiled `fx.Graph` (or relevant data structure containing nodes and their attributes).
    *   A list/set of all identified activation tensors (`candidate_set`).
    *   The target GPU memory limit (`mem_limit`).
    *   All necessary profiling attributes attached to nodes/tensors (e.g., `inactive_time`, `recompute_ratio`, `swap_time`, `run_time`, `memory_size`, `first_bw_access`, `last_fw_access`, etc.).
*   [ ] **Data Access:** Ensure the algorithm can easily access all required attributes for any given activation candidate or graph node.

### 2. Implement Core Scheduling Logic (MuTWO Algorithm B)

*   [ ] **Initialization:**
    *   Initialize `swaps = {}`, `recomps = {}`.
    *   Initialize `last_prompt` (node where last swap-in was scheduled) appropriately (e.g., end of backward graph).
*   [ ] **Main Loop (`while candidate_set != ∅`):**
    *   [ ] **Select Candidates:**
        *   Find swap candidate `s_cand`: Activation in `candidate_set` with the maximum `inactive_time`.
        *   Find recompute candidate `r_cand`: Activation in `candidate_set` with the maximum `recompute_ratio`.
    *   [ ] **Calculate Overheads:**
        *   Implement `SwapOverhead(s_cand, last_prompt)` function (detailed below, based on MuTWO Section 4.2.2 / Algorithm C).
        *   Implement `RecomputeOverhead(r_cand)` function (likely returns `r_cand.recomp_time` or a refined calculation).
    *   [ ] **Make Decision (`if s_overhead < r_overhead`):**
        *   If swapping is cheaper:
            *   Call `Swap(s_cand, prompt_node)`: Mark `s_cand` for swapping (e.g., set flags/attributes like `to_offload`, `to_prefetch`). Update `last_prompt`.
            *   Add `s_cand` to `swaps`.
            *   Set `cand = s_cand`.
        *   Else (recomputing is cheaper or equal):
            *   Call `Recompute(r_cand)`: Mark `r_cand` for recomputation (e.g., set flag `to_recompute`).
            *   Add `r_cand` to `recomps`.
            *   Set `cand = r_cand`.
    *   [ ] **Update State:**
        *   Remove `cand` from `candidate_set`.
        *   Implement `update_recomps`: Update recomputation counts/costs if choosing `cand` affects other recomputation dependencies.
        *   Implement `update_candidates`: Update attributes (like estimated memory savings) of remaining candidates based on the choice of `cand`.
        *   Implement `update_swap_prompts`: Adjust swap scheduling based on the choice.
    *   [ ] **Memory Simulation & Check:**
        *   Implement `get_mem_consumption()`: Simulate the peak memory usage based on the current `swaps` and `recomps` decisions. This requires simulating the graph execution order and tracking tensor lifetimes considering swaps/recomputes.
        *   Check `if (mem_consumption - mem_limit) < 0`: If memory constraint is met, `break` the loop.

### 3. Implement Swap Overhead Calculation (MuTWO Section 4.2.2 / Algorithm C)

*   [ ] **Function Signature:** `SwapOverhead(swap_cand, last_prompt)` returning `(swap_overhead, prompt_node)`.
*   [ ] **Get Inputs:** `bw_access = swap_cand.first_bw_access`, `swap_time = swap_cand.swap_time`.
*   [ ] **Simulate Overlap (Core Logic):**
    *   Determine the potential overlap window (nodes between `prefetch_prompt` and `bw_access`).
    *   Identify the `peak_memory_interval` where swapping isn't possible.
    *   Attempt to schedule the `swap_time` within the overlap window, avoiding the peak interval, potentially using forward pass nodes (`FW_i`) and backward pass nodes (`BW_j`) from *other* sub-arrays if implementing the full MuTWO multiplexing (or just available compute if simplifying for single model).
    *   **Case 1 (No Overlap / Peak Interval):** Calculate overhead based on `swap_time` and potential conflicts with `last_prompt`.
    *   **Case 2 (Complete Overlap):** Overhead is 0.
    *   **Case 3 (Partial Overlap):** Overhead is the remaining `swap_time` that couldn't be overlapped.
*   [ ] **Return Values:** Return the calculated `swap_overhead` and the determined `prefetch_prompt` node.

### 4. Implement Recompute Overhead Calculation

*   [ ] **Function Signature:** `RecomputeOverhead(r_cand)` returning `r_overhead`.
*   [ ] **Calculation:** Return `r_cand.recomp_time` (the estimated time to execute `r_cand.recomp_graph`). Consider potential refinements if needed (e.g., accounting for recomputing dependencies).

### 5. Algorithm Output

*   [ ] **Define Output:** Determine the output format. This should clearly indicate for each activation whether it should be kept, swapped, or recomputed. This could be:
    *   Updating attributes directly on the input `fx.Graph` nodes (e.g., setting `to_offload`, `to_prefetch`, `to_recompute` boolean flags).
    *   Returning separate lists/dictionaries mapping activations to their fate (Keep, Swap, Recompute).

### 6. Testing & Validation

*   [ ] Create test cases with simple graphs and known profiling data.
*   [ ] Manually trace the algorithm's decisions for these test cases.
*   [ ] Implement unit tests for `SwapOverhead`, `RecomputeOverhead`, and the main scheduling loop.
*   [ ] Test the algorithm with the actual profiled data from Stage 1 (once available).
*   [ ] Verify the output decisions seem logical based on the input metrics (`inactive_time`, `recompute_ratio`, `mem_limit`).