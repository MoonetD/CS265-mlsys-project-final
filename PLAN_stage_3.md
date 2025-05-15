# Plan - Stage 3: Graph Extractor and Rewriter Implementation

This stage focuses on modifying the PyTorch computational graph (`fx.Graph`) based on the decisions made by the activation checkpointing algorithm in Stage 2. Specifically, it involves implementing the recomputation mechanism by extracting relevant forward subgraphs and inserting them into the backward pass using `torch.fx` APIs.

## Tasks

### 1. Input Processing

*   [ ] **Receive Input:** Take the `fx.Graph` (potentially already annotated by the Stage 2 algorithm with `to_recompute` flags) and the original forward pass graph as input.
*   [ ] **Identify Recomputation Targets:** Iterate through the graph or the Stage 2 output to find all activation tensors marked for recomputation (`to_recompute = True`).

### 2. Subgraph Extraction

*   [ ] **Implement Extraction Logic:** For each activation `act` marked for recomputation:
    *   Start from the node that originally computed `act` in the *forward pass graph*.
    *   Perform a backward traversal (following `node.args`) to identify all ancestor nodes within the forward pass that are necessary to compute `act`.
    *   Stop traversal at placeholder nodes or nodes producing activations that were *kept* (not marked for recomputation). These become the inputs (`recomp_srcs`) to the recomputation subgraph.
    *   Collect the sequence of nodes forming the subgraph (`recomp_graph`) needed to compute `act` from `recomp_srcs`.
    *   Store this `recomp_graph` and its `recomp_srcs` (potentially already done in Stage 1, but verify/refine here).

### 3. Graph Rewriting (Backward Pass Modification)

*   [ ] **Identify Insertion Points:** For each activation `act` marked for recomputation, find its `first_bw_access` node in the *backward pass graph*. This is where the recomputation needs to happen *before*.
*   [ ] **Implement Subgraph Insertion:**
    *   Use `graph.inserting_before(first_bw_access)` context manager.
    *   Inside the context, iterate through the nodes of the corresponding `recomp_graph`.
    *   For each node in `recomp_graph`, create a copy in the main backward graph using `new_graph.node_copy(node, lambda n: env[n.name])`.
        *   Maintain an environment `env` mapping original node names in `recomp_graph` to the newly created nodes in the backward pass.
        *   Ensure the `args` and `kwargs` of the copied node correctly reference the new nodes created within the backward pass (using the `env` lookup). The inputs (`recomp_srcs`) need to be mapped to the corresponding nodes available *at the insertion point* in the backward pass (these might be kept activations or outputs of prior recomputations).
    *   The final node created in this insertion process corresponds to the recomputed activation `act`.
*   [ ] **Update Node Usage:**
    *   Use `original_node.replace_all_uses_with(recomputed_node)` to replace all uses of the *original* (now non-existent) activation node within the backward pass with the *newly recomputed* activation node. Note: The original node won't exist if it was discarded during the forward pass; ensure references in the backward pass point to the correct recomputed value.
*   [ ] **Handle Recomputed Activations:** Ensure that nodes corresponding to activations marked for recomputation are effectively removed or bypassed during the initial forward pass execution of the *modified* graph (this might be implicit if they are never used, or require explicit modification depending on implementation).

### 4. Finalization

*   [ ] **Lint Graph:** Call `graph.lint()` to check for inconsistencies after modifications.
*   [ ] **Recompile Module:** Call `graph_module.recompile()` to generate the new `forward` method for the modified `fx.GraphModule`.

### 5. Integration & Testing

*   [ ] **Integrate Rewriter:** Combine the rewriter logic with the output of the Stage 2 algorithm.
*   [ ] **Test with ResNet-152:** Run the full pipeline (Profile -> Decide -> Rewrite) on ResNet-152 as the primary model.
*   [ ] **Verify Correctness:** Check if the output of the rewritten graph matches the output of the original graph (using `torch.allclose`).
*   [ ] **Verify Performance:** Measure iteration latency and peak memory consumption of the rewritten graph. Compare against the baseline (no activation checkpointing) and ensure memory is reduced while keeping latency increase acceptable. Debug any graph errors or performance regressions.