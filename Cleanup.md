# TODO: Codebase Simplification Plan

This plan outlines the steps to simplify the codebase by removing features and logic not strictly required by the project's three-stage scope, focusing on the custom activation recomputation pipeline.

## Overall Goals:
1.  Remove all CPU swapping (offloading) simulation, decision-making, and related metrics.
2.  Ensure Stage 2 (`activation_checkpointing.py`) calculates the `recompute_ratio` as `mem_saved / recompute_time_overhead`.
3.  Simplify the Stage 2 greedy algorithm to focus solely on "keep vs. recompute" decisions.
4.  Ensure Stage 3 evaluation in `ac_comparison.py` primarily tests the custom graph rewriter.

---

## File: `graph_prof.py` (Stage 1 - Graph Profiler)

**Objective:** Remove swap simulation logic and `recompute_ratio` calculation. Ensure it outputs `median_memory_sizes` and `recomp_times` for Stage 2.

* **[ ] Remove Swap-Related Attributes:**
    * In `__init__`:
        * `self.swap_times: Dict[str, List[float]]`
        * `self.swapped_out_activations: Set[str]`
        * `self.median_swap_times: Dict[str, float]`
        * `self.BYTES_PER_SEC_CPU_TO_GPU`
        * `self.BYTES_PER_SEC_GPU_TO_CPU`
    * **Reason:** Swap simulation is out of scope for Stage 1 profiling, which should focus on gathering data for *recomputation* decisions.

* **[ ] Remove Swap Simulation Logic in `run_node`:**
    * Remove section: `1. Swap-in Simulation (Before node execution, during backward pass)`
    * Remove section: `3. Swap-out Simulation (After node execution, during forward pass)`
    * **Reason:** This logic supports the out-of-scope swap simulation.

* **[ ] Modify `aggregate_stats`:**
    * Remove calculation and storage of `self.median_swap_times`.
    * **Remove `self.recompute_ratios` calculation.** Stage 2 is responsible for calculating its decision metric.
        * Original line: `self.recompute_ratios[act_name] = recomp_time / effective_swap_time`
    * Ensure `self.recomp_times` (recomputation time for an activation) and `self.median_memory_sizes` (memory size of an activation) are correctly calculated and made available.
    * **Reason:** Decouple profiler from specific Stage 2 ratio calculations and remove swap dependencies.

* **[ ] Modify `reset_stats`:**
    * Remove clearing of `self.swap_times` and `self.swapped_out_activations`.
    * Remove clearing of `self.median_swap_times`.
    * Remove clearing of `self.recompute_ratios`.
    * **Reason:** Align with removed attributes.

* **[ ] Modify `print_stats`:**
    * Remove printing of `Avg Swap Time (s)` and `Recomp Ratio` from the "[Per-Activation MuTWO Metrics]" table.
    * **Reason:** These metrics are being removed or moved.

* **[ ] Modify `save_stats_to_csv`:**
    * In `activation_csv_filename` generation:
        * Remove `'avg_swap_time_s'` and `'recompute_ratio'` from `fieldnames`.
        * Remove their corresponding data from the `writer.writerow` call.
    * **Reason:** These metrics are being removed or moved.

---

## File: `activation_checkpointing.py` (Stage 2 - Activation-Checkpointing Algorithm)

**Objective:** Remove swap-related decision logic. Implement the `recompute_ratio = memory_saved / recomputation_time_overhead` calculation. Simplify the greedy selection loop.

* **[ ] Remove Swap-Related Helper Functions:**
    * Remove `_calculate_swap_overhead(self, activation_name)` (if it was only used for swap decisions).
    * Remove `_calculate_swap_overhead_v2(self, act_name, last_prompt)`.
    * Remove `_update_swap_prompts(self, swaps, candidate_set)`.
    * **Reason:** Stage 2 focuses on "keep vs. recompute", not swap decisions.

* **[ ] Refactor `decide_checkpoints` Method:**
    * **Calculate `recompute_benefit_ratio` here:** For each candidate activation, calculate a benefit ratio for recomputation. This could be `memory_size / recomp_time_s`.
        * This replaces reliance on the `recompute_ratio` from the CSV.
    * **Simplify Greedy Selection:**
        * The loop should iterate while `current_peak_memory > self.memory_budget_bytes`.
        * In each iteration, select the activation candidate that, if recomputed, provides the best benefit (e.g., highest `recompute_benefit_ratio`).
        * Change its status in `current_schedule` to `'RECOMPUTE'`.
        * Remove it from the `candidate_set`.
    * Remove logic related to `swaps` set, `last_prompt`, and choosing between swap and recompute. The decision is only whether to recompute or keep.
    * Rename `_get_max_recompute_ratio_candidate` to something like `_get_best_recompute_candidate` and ensure it uses the newly calculated `recompute_benefit_ratio`.
    * Remove `_get_max_inactive_time_candidate` if the primary greedy metric is `recompute_benefit_ratio`. The project brief mentions "max idle/recompute ratio" so `inactive_time_s` might still be used in the numerator instead of `memory_size`. Clarify the exact ratio: `median_mem_size_bytes / recomp_time_s` OR `inactive_time_s / recomp_time_s`. Let's assume `median_mem_size_bytes / recomp_time_s` for now as "mem / recompute_time" was in the brief.
    * **Reason:** Align with Stage 2's specific goal of choosing activations to *recompute* to save memory, based on a simplified greedy approach.

* **[ ] Review `_simulate_memory_usage`:**
    * Ensure this accurately simulates memory *without* considering CPU swapping. It should sum memory for parameters, gradients, optimizer states, and activations currently kept in GPU memory.
    * The existing logic for `fw_inter_mem`, `bw_inter_mem`, `fw_active_mem`, `bw_active_mem` seems generally okay for simulating activations kept or recomputed, but double-check it doesn't have implicit swap assumptions.
    * **Reason:** Accurate memory simulation is key to the greedy algorithm's stopping condition.

* **[ ] Ensure `ac_decisions_bs{batch_size}.csv` Output:**
    * The `if __name__ == "__main__":` block currently saves this. Make sure it reflects the simplified decisions ('CHECKPOINT' or 'RECOMPUTE').
    * **Reason:** This output is crucial for Stage 3.

---

## File: `graph_rewriter.py` (Stage 3 - Sub-graph Extractor & Rewriter)

**Objective:** No direct removals, but acknowledge limitations and ensure it works with Stage 2's output.

* **[ ] Verify Compatibility with Simplified Stage 2 Output:**
    * Ensure `extract_recomputation_subgraphs` and `rewrite_graph_with_recomputation` correctly process the decision map from Stage 2 (which will only contain 'CHECKPOINT' or 'RECOMPUTE').
    * **Reason:** Stage 3 consumes Stage 2's output.

* **[ ] No Action Needed for Swap Logic (as it's not implemented):**
    * The project brief mentions Stage 3 handling activations tagged "swap". Since Stage 2 will no longer produce "swap" tags, this functionality in Stage 3 will not be exercised. No code needs to be removed from `graph_rewriter.py` itself on this point as it wasn't implemented.
    * **Reason:** Simplification in Stage 2 makes this part of Stage 3's described scope unused.

* **[ ] Acknowledge Missing CUDA Stream Implementation:**
    * The Stage 3 requirement "Wrap runtime with 3 CUDA streams (compute, H2D, D2H) + events for proper overlap" is not implemented. This is a complex task.
    * **Action:** Document this as a known unimplemented part of the advanced Stage 3 requirements.
    * **Reason:** Transparency about project scope fulfillment.

---

## File: `ac_comparison.py` (Stage 3 - Validation)

**Objective:** Ensure it primarily tests the custom graph rewriter and correctly uses the simplified AC decisions.

* **[ ] Ensure `apply_activation_checkpointing` Prioritizes Custom Rewriter:**
    * The current logic uses `ac_decisions` and `activation_liveness` to attempt the custom graph rewriter path. This is good.
    * The fallback to `torch.utils.checkpoint.checkpoint` should be a secondary path.
    * Make sure `activation_liveness` is correctly passed and used (it's extracted from `ac_algo.activation_stats_df`). The profiler (Stage 1) should be the source of this liveness info, and `ac_algo` should load it from the CSV.
    * **Reason:** The project goal is to build and validate the custom rewriter.

* **[ ] Verify Input to `apply_activation_checkpointing`:**
    * `ac_decisions` will now be simpler (only 'CHECKPOINT'/'RECOMPUTE').
    * `activation_liveness` should be correctly sourced from the profiler's output (via the CSV loaded by `ActivationCheckpointingAlgorithm`).
    * **Reason:** Correct inputs are needed for the rewriter to function.

---

## General Cleanup (Across Files)

* **[ ] Remove Unused Imports:** After removing logic, some imports might become unnecessary.
* **[ ] Update Comments and Docstrings:** Reflect the changes in logic and scope.
* **[ ] Test Thoroughly:** After refactoring, test each stage and the final comparison to ensure correctness.

---