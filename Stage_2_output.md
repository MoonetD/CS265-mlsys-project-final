# Stage 2: Activation Checkpointing Algorithm

This document explains the implementation of the activation checkpointing algorithm based on the Π-TWO paper. The algorithm decides which activation tensors to retain in memory during the forward pass and which to discard and recompute during the backward pass.

## 1. Introduction to Activation Checkpointing

### 1.1 The Memory Problem in Deep Learning

Training deep neural networks requires significant GPU memory. During the forward pass, intermediate tensors (activations) are generated and stored for later use in the backward pass to compute gradients. These activations can consume up to 70-85% of the total memory during training, as noted in the project requirements.

The key observation is that activations have high "inactive time" - they are created during the forward pass but may not be used until much later in the backward pass. This leads to inefficient memory usage, as these tensors occupy valuable GPU memory while being idle.

### 1.2 The Activation Checkpointing Solution

Activation checkpointing (AC) addresses this memory inefficiency by:

1. **Not storing all activations** in memory during the forward pass
2. **Storing only a subset** of activations (checkpoints)
3. **Recomputing the others** during the backward pass when needed

This approach trades computation time for memory savings, allowing the training of larger models or the use of larger batch sizes on the same hardware.

## 2. Algorithm Overview from Π-TWO

The activation checkpointing algorithm in Π-TWO (as described in the paper) makes decisions about which activations to retain and which to recompute based on:

1. **Inactive time**: The duration an activation remains idle in memory between its last use in the forward pass and its first use in the backward pass
2. **Recompute ratio**: The ratio of memory occupied by a tensor over the time required to recompute it

The algorithm follows these key steps:

1. Start with all activations marked for retention (checkpointing)
2. Iteratively select candidates for recomputation based on their metrics
3. For each candidate, calculate the overhead of swapping vs. recomputing
4. Choose the option with lower overhead
5. Update the schedule and simulate memory consumption
6. Continue until memory consumption is below the specified limit

## 3. Implementation Details

Our implementation in `activation_checkpointing.py` follows the core principles of the Π-TWO algorithm with optimizations for performance and usability.

### 3.1 Core Components

The implementation consists of these main components:

1. **ActivationCheckpointingAlgorithm class**: The main class that implements the algorithm
2. **Input processing**: Loading and processing profiling data from Stage 1
3. **Memory simulation**: Simulating memory usage during training with different checkpointing decisions
4. **Decision algorithm**: Implementing the core decision logic for activation checkpointing
5. **Output generation**: Saving the final decisions to a CSV file

### 3.2 Algorithm Workflow

The algorithm follows this workflow:

1. **Initialization**: Load profiling data from Stage 1 CSV files
2. **Initial state**: Mark all activations as RETAINED (checkpointed)
3. **Memory analysis**: Analyze memory components to determine if the budget is achievable
4. **Candidate selection**: Identify valid activation candidates for recomputation
5. **Main loop**: Iteratively select activations for recomputation until memory budget is met
6. **Memory simulation**: Simulate memory usage after each decision
7. **Output generation**: Save final decisions to a CSV file

## 4. Input and Output Formats

### 4.1 Input: Profiling Data from Stage 1

The algorithm takes two CSV files as input:

1. **Node statistics** (`profiler_stats_bs<X>_node_stats.csv`):
   - Contains metrics for every operation in the computation graph
   - Key columns: `rank`, `node_name`, `node_type`, `gtype`, `median_run_time_s`, `median_peak_mem_bytes`, `median_active_mem_bytes`

2. **Activation statistics** (`profiler_stats_bs<X>_activation_stats.csv`):
   - Contains metrics for each activation tensor
   - Key columns: `activation_name`, `creation_rank`, `last_fw_use_rank`, `first_bw_use_rank`, `last_bw_use_rank`, `median_mem_size_bytes`, `inactive_time_s`, `recomp_time_s`

### 4.2 Output: Activation Checkpointing Decisions

The algorithm outputs a CSV file with the following information:

- `activation_name`: Name of the activation tensor
- `decision`: Either 'RETAINED' (keep in memory) or 'RECOMPUTE' (discard and recompute)
- Additional columns from the input data for reference

## 5. Key Components of the Algorithm

### 5.1 Memory Simulation

The `_simulate_memory_usage` method is a critical component that:

1. Simulates the execution of the model with a given checkpointing schedule
2. Tracks memory usage throughout forward and backward passes
3. Calculates peak memory consumption and total execution time
4. Accounts for both retained and recomputed activations

This simulation allows the algorithm to evaluate different checkpointing decisions without actually running the model.

### 5.2 Candidate Selection Strategy

The algorithm selects candidates for recomputation based on the "recompute benefit ratio" (memory saved per unit of recomputation time):

```python
ratio = mem_size / (recomp_time + 1e-6)
```

Activations with higher ratios are prioritized for recomputation, as they provide the most memory savings relative to their recomputation cost.

### 5.3 Memory Budget Handling

The algorithm handles the memory budget constraint by:

1. Checking if the budget is achievable given the model's incompressible memory components
2. Iteratively selecting activations for recomputation until the budget is met or no more candidates are available
3. Providing the best possible solution even if the budget cannot be fully met

## 6. Performance Characteristics and Trade-offs

### 6.1 Memory-Computation Trade-off

The algorithm explicitly manages the trade-off between memory usage and computation time:

- **Memory savings**: Achieved by marking activations for recomputation
- **Computation overhead**: Incurred by recomputing activations during the backward pass

The algorithm aims to minimize peak memory usage while keeping the computation overhead reasonable.

### 6.2 Algorithm Efficiency

The implementation includes several optimizations for efficiency:

1. **Caching**: Frequently accessed data is cached to avoid redundant calculations
2. **Simulation optimization**: The memory simulation is optimized to run quickly
3. **Early stopping**: The algorithm stops when the memory budget is met
4. **Timeout mechanism**: Prevents excessive runtime for large models

### 6.3 Scalability

The algorithm is designed to scale with model size and complexity:

- Works with models of varying sizes and architectures
- Handles different batch sizes
- Processes large numbers of activations efficiently

## 7. Usage Example

Here's how to use the activation checkpointing algorithm:

```python
# Initialize the algorithm with profiling data and memory budget
ac_algo = ActivationCheckpointingAlgorithm(
    node_stats_path="reports/profiler_stats_bs16_node_stats.csv",
    activation_stats_path="reports/profiler_stats_bs16_activation_stats.csv",
    memory_budget_gb=4.0
)

# Run the algorithm to decide which activations to checkpoint
final_schedule = ac_algo.decide_checkpoints(
    fixed_overhead_gb=0.3,  # Memory for parameters, gradients, optimizer states
    debug=False,
    max_iterations=1000,
    timeout_seconds=300,
    model_info="model_bs16"  # Optional: for naming output files
)

# The final_schedule is a dictionary mapping activation names to 'RETAINED' or 'RECOMPUTE'
```

## 8. Algorithm Performance Metrics

The algorithm provides detailed performance metrics:

1. **Memory reduction**: The amount of memory saved compared to retaining all activations
2. **Execution time overhead**: The additional computation time due to recomputation
3. **Percentage of activations recomputed**: The proportion of activations marked for recomputation
4. **Gap to budget**: How close the final memory usage is to the specified budget

## 9. Relationship to Stage 3

The output of Stage 2 (the activation checkpointing decisions) will be used in Stage 3 to:

1. Extract subgraphs for activations marked for recomputation
2. Replicate these subgraphs in the backward pass
3. Modify the execution strategy to implement the checkpointing decisions

This will complete the implementation of activation checkpointing in the training process.

## 10. Conclusion

The activation checkpointing algorithm implemented in Stage 2 provides an effective solution to the memory constraints in deep neural network training. By selectively recomputing activations during the backward pass, it reduces peak memory consumption at the cost of increased computation time.

This implementation follows the principles outlined in the Π-TWO paper while adding optimizations for performance and usability. The algorithm makes data-driven decisions based on profiling information from Stage 1, resulting in an optimal trade-off between memory usage and computation time.