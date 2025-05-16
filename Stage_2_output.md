# Stage 2 Activation Checkpointing Results

This document summarizes the results of the Stage 2 implementation of the activation checkpointing algorithm based on the Π-TWO paper. The algorithm decides which activations to keep in memory and which to recompute during the backward pass to reduce peak memory usage.

## Implementation Overview

The activation checkpointing algorithm was implemented following Algorithm B from the Π-TWO paper, which:

1. Takes profiling data from Stage 1 as input
2. Analyzes activation statistics (memory size, recomputation time, usage patterns)
3. Selects activations for recomputation based on their memory-to-computation ratio
4. Simulates memory usage to ensure decisions meet the memory budget constraints

## Results Summary

The algorithm was tested with two different batch sizes (4 and 64) with a memory budget of 2GB. Here are the key results:

| Metric | Batch Size 4 | Batch Size 64 |
|--------|--------------|---------------|
| Total activations | 620 | 620 |
| Activations marked for RECOMPUTE | 409 (66%) | 410 (66%) |
| Activations marked for RETAIN | 211 (34%) | 210 (34%) |
| Estimated peak GPU memory | 3.26 GB | 23.24 GB |
| Estimated execution time | 1.77 s | 12.28 s |
| Memory budget target | 2.00 GB | 2.00 GB |

## Analysis

### Why Similar Recomputation Decisions Across Batch Sizes

It might seem surprising that both batch sizes result in almost identical numbers of activations being marked for recomputation (409 vs 410 out of 620), despite the significant difference in memory requirements. This is explained by:

1. **Ratio-Based Selection**: The algorithm selects activations based on their recompute benefit ratio (memory saved / recomputation time), not absolute memory size. These ratios remain similar across batch sizes.

2. **Incompressible Memory**: For both batch sizes, the 2GB memory budget is impossible to achieve:
   - Batch size 4: Incompressible memory is 3.26 GB (FW active: 1.35 GB, BW active: 1.41 GB, Fixed overhead: 0.50 GB)
   - Batch size 64: Incompressible memory is 23.24 GB (FW active: 11.36 GB, BW active: 11.39 GB, Fixed overhead: 0.50 GB)

3. **Exhaustive Processing**: Since the memory budget is unachievable in both cases, the algorithm continues until it processes all viable candidates.

### Why Certain Activations Remain RETAINED

Looking at the decisions in `ac_decisions.csv`, we can see that activations remain RETAINED for several reasons:

1. **Size Threshold**: Activations smaller than 100KB are automatically excluded from consideration. For example, many `getitem` operations have sizes of only 256 bytes to 1KB. This threshold is an implementation detail in the code, not a requirement from the Π-TWO paper. The comment "Reduced minimum memory size threshold to 100KB (was 1MB)" suggests this was a design decision that was adjusted during development to balance practical considerations.

2. **Poor Memory-to-Computation Ratio**: Some activations don't provide enough memory savings relative to their recomputation cost.

3. **Diminishing Returns**: The algorithm selects the "best" activations first (highest benefit ratio), leaving those with progressively worse ratios.

### Memory Reduction

- **Batch Size 4**: The algorithm attempted to meet the 2GB memory budget but could only reduce memory to 3.26GB due to incompressible memory components.

- **Batch Size 64**: With the larger batch size, memory requirements increased significantly to 23.24GB. The algorithm still marked a similar percentage of activations for recomputation (~66%), but the absolute memory size of each activation is much larger with batch size 64.

### Computation-Memory Tradeoff

The algorithm demonstrates the fundamental tradeoff between memory usage and computation time:

- By marking ~66% of activations for recomputation, the algorithm significantly reduces memory usage compared to retaining all activations.
- This comes at the cost of increased execution time due to recomputation overhead.

## Output Format

The algorithm produces a CSV file (`ac_decisions.csv`) that contains the following information for each activation:

| Column | Description |
|--------|-------------|
| `activation_name` | Unique identifier for the activation |
| `decision` | Either "RECOMPUTE" or "RETAINED" |
| `median_mem_size_bytes` | Memory size of the activation in bytes |
| `recomp_time_s` | Time required to recompute the activation in seconds |
| `creation_rank` | Topological rank when the activation is created |
| `first_bw_use_rank` | Topological rank of first use in backward pass |

## Preparation for Stage 3

The output of Stage 2 provides all the necessary information for Stage 3 (Graph Rewriter):

1. **Clear Decisions**: Each activation is marked as either "RECOMPUTE" or "RETAINED", providing unambiguous instructions for the Graph Rewriter.

2. **Timing Information**: The `creation_rank` and `first_bw_use_rank` fields help the Graph Rewriter determine where to insert recomputation subgraphs in the backward pass.

3. **Memory and Performance Metrics**: The memory and timing statistics help validate that the Graph Rewriter's implementation achieves the expected memory savings and performance characteristics.

In Stage 3, the Graph Rewriter will:
- Extract subgraphs for each activation marked as "RECOMPUTE"
- Insert these subgraphs at the appropriate points in the backward pass
- Modify the execution strategy to implement these decisions

## Conclusion

The Stage 2 implementation successfully applies the activation checkpointing algorithm from the Π-TWO paper to determine which activations to recompute vs. retain. While the algorithm couldn't meet the 2GB memory budget due to incompressible memory components, it made optimal decisions within the constraints, marking approximately 66% of activations for recomputation.

The implementation is ready for Stage 3, which will modify the execution strategy to implement these decisions by extracting and inserting recomputation subgraphs.