# Path to Completion: Activation Checkpointing Project

This document outlines the remaining tasks needed to complete the CS265 Systems Project on Activation Checkpointing, based on the requirements in `Material Markdown/Project Requirement.md`.

## Current Status

We have implemented all three stages of the project:

1. **Stage 1: Graph Profiler** ✅
   - Created a comprehensive computational graph
   - Collected data on computation time and memory usage
   - Categorized inputs/outputs as parameters, gradients, activations, etc.
   - Conducted static data analysis on activations
   - Generated CSV files with profiling statistics

2. **Stage 2: Activation Checkpointing Algorithm** ✅
   - Implemented the μ-TWO algorithm for deciding which activations to checkpoint/recompute
   - Used metrics like inactive time and recompute ratio
   - Created a memory simulator to validate decisions

3. **Stage 3: Graph Extractor and Rewriter** ✅
   - Created a graph rewriter that can extract subgraphs for recomputation
   - Implemented insertion of recomputation subgraphs in the backward pass
   - Added a fallback mechanism for when graph rewriting fails

## Issues to Address

1. **Algorithm Decisions**
   - The algorithm is currently deciding to checkpoint all activations (0 RECOMPUTE, 620 CHECKPOINT)
   - With a 4GB budget, we would expect some activations to be marked for recomputation
   - Need to investigate why the simulated peak memory (0.50 GB) is well below the budget (4.00 GB)

2. **Graph Rewriter Integration**
   - The graph rewriter is implemented but not being used in practice
   - The system falls back to the simplified approach (applying checkpointing to 50% of bottleneck blocks)
   - Need to debug why tracing the model with `fx.symbolic_trace` might be failing

3. **BERT Model Support**
   - Currently only tested with ResNet-152
   - Need to add support for BERT as specified in the project requirements

## Remaining Tasks

### 1. Debug Activation Checkpointing Algorithm

- [ ] **Investigate Memory Budget Issue**
  - Analyze why the simulated peak memory is only 0.50 GB
  - Check if the fixed overhead (0.5 GB) is being calculated correctly
  - Consider lowering the memory budget further to force recomputation decisions

- [ ] **Verify Activation Statistics**
  - Ensure the CSV files contain accurate information about activations
  - Check if the recomputation time and memory size estimates are reasonable
  - Validate that inactive time calculations are correct

### 2. Improve Graph Rewriter Integration

- [ ] **Debug Model Tracing**
  - Add more detailed error logging to `trace_model_for_ac`
  - Try using PyTorch's `make_fx` instead of `symbolic_trace` for more robust tracing
  - Test with simpler models first to verify the graph rewriter works

- [ ] **Enhance Subgraph Extraction**
  - Improve the logic for identifying subgraph inputs
  - Add better handling of complex node dependencies
  - Implement more robust error handling

### 3. Add BERT Model Support

- [ ] **Implement BERT Profiling**
  - Resolve the `aten._local_scalar_dense.default` error for BERT profiling
  - Create a proper wrapper for BERT that works with the profiler
  - Generate CSV files for BERT similar to ResNet-152

- [ ] **Test Activation Checkpointing with BERT**
  - Apply the algorithm to BERT
  - Measure memory reduction and time overhead
  - Compare results with ResNet-152

### 4. Prepare Final Deliverables

- [ ] **Complete Experimental Analysis Document**
  - [ ] **Computation and Memory Profiling Statistics**
    - Include detailed profiling statistics for both ResNet-152 and BERT
    - Add static analysis of the computational graph
    - Present memory breakdown by type (parameters, gradients, activations)

  - [ ] **Peak Memory Consumption vs. Mini-batch Size**
    - Create bar graphs comparing memory usage with and without AC
    - Include data for multiple batch sizes (4, 8, 16, 32, 64)
    - Add percentage reduction labels

  - [ ] **Iteration Latency vs. Mini-batch Size**
    - Create line graphs comparing execution time with and without AC
    - Include data for multiple batch sizes
    - Add percentage overhead/improvement labels

- [ ] **Code Cleanup and Documentation**
  - Add comprehensive comments to all code files
  - Create a README with instructions for running the code
  - Ensure consistent coding style throughout the project

## Testing and Validation Plan

1. **Unit Testing**
   - Test each component individually
   - Verify that the profiler collects accurate statistics
   - Ensure the algorithm makes reasonable decisions
   - Validate that the graph rewriter correctly transforms the graph

2. **Integration Testing**
   - Test the full pipeline from profiling to graph rewriting
   - Verify that the modified model produces the same results as the original
   - Measure memory reduction and time overhead

3. **Performance Testing**
   - Test with different batch sizes
   - Compare memory usage and execution time
   - Verify that the results match expectations

## Timeline

1. **Week 1: Debugging and Fixes (3 days)**
   - Debug activation checkpointing algorithm
   - Fix graph rewriter integration issues

2. **Week 1-2: BERT Support (4 days)**
   - Implement BERT profiling
   - Test activation checkpointing with BERT

3. **Week 2: Final Deliverables (3 days)**
   - Complete experimental analysis document
   - Clean up code and add documentation

4. **Week 2-3: Testing and Validation (4 days)**
   - Perform unit, integration, and performance testing
   - Make final adjustments based on test results

5. **Week 3: Final Review and Submission (1 day)**
   - Review all deliverables
   - Prepare for code review and demo
   - Submit final project