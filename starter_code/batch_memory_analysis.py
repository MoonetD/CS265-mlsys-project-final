#!/usr/bin/env python
"""
Batch Memory Analysis Script

This script analyzes the peak memory consumption of the ResNet-152 model
with different batch sizes. It generates a bar graph showing the relationship
between batch size and peak memory usage.

To run this script:
    conda run -n ml_env python starter_code/batch_memory_analysis.py
"""

import os
import torch
import torch.nn as nn
import torch.fx as fx
import torchvision.models as models
import matplotlib.pyplot as plt
from functools import wraps
import numpy as np

from graph_prof import GraphProfiler
from graph_tracer import SEPFunction, compile

def train_step(model, optim, batch):
    """Simple training step function that will be traced by the compiler."""
    loss = model(batch).sum()
    loss = SEPFunction.apply(loss)
    loss.backward()
    optim.step()
    optim.zero_grad()

# Global variable to store peak memory between function calls
_peak_memory = 0

def graph_transformation(gm: fx.GraphModule, args: any) -> fx.GraphModule:
    """
    Graph transformation function that profiles the model execution.
    This is passed to the compile function to analyze the traced graph.
    
    Returns:
        The original graph module (required by the compile function).
        Peak memory is stored in the global _peak_memory variable.
    """
    global _peak_memory
    
    # Get batch size from args
    batch_size = args[2].shape[0]
    print(f"\nProfiling batch size: {batch_size}")
    
    # Initialize the GraphProfiler with the graph module
    graph_profiler = GraphProfiler(gm)
    
    # Run the profiler for a few iterations
    warm_up_iters, profile_iters = 1, 3
    with torch.no_grad():
        # Warm-up run
        for _ in range(warm_up_iters):
            graph_profiler.run(*args)
        
        # Reset stats before actual profiling
        graph_profiler.reset_stats()
        
        # Profile runs
        for _ in range(profile_iters):
            graph_profiler.run(*args)
    
    # Aggregate and analyze the results
    graph_profiler.aggregate_stats(num_runs=profile_iters)
    
    # Save stats to CSV with batch-size-specific prefix
    csv_prefix = f"profiler_stats_bs{batch_size}"
    graph_profiler.save_stats_to_csv(filename_prefix=csv_prefix)
    print(f"CSV files saved with prefix: {csv_prefix}")
    
    # Get the peak memory usage
    _peak_memory = max(graph_profiler.avg_peak_mem_node.values()) if graph_profiler.avg_peak_mem_node else 0
    
    # Print peak memory usage
    print(f"Peak memory usage: {_peak_memory / (1024**2):.2f} MiB")
    
    # Return only the graph module as required by the compile function
    return gm

def profile_batch_size(batch_size, device_str='cuda:0'):
    """
    Profile the ResNet-152 model with a specific batch size.
    
    Args:
        batch_size: The batch size to use for profiling.
        device_str: The device to use for profiling.
        
    Returns:
        The peak memory usage in bytes.
    """
    print(f"\n--- Profiling ResNet-152 with batch size {batch_size} ---")
    
    # Create ResNet-152 model
    model = models.resnet152(weights=models.ResNet152_Weights.IMAGENET1K_V1).to(device_str)
    
    # Create a random batch of data
    batch = torch.randn(batch_size, 3, 224, 224).to(device_str)
    
    # Create an optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.001, foreach=True, capturable=True
    )
    
    # Initialize gradients for optimizer step to be traceable
    for param in model.parameters():
        if param.requires_grad:
            param.grad = torch.rand_like(param, device=device_str)
    
    # Perform one step to initialize optimizer states if needed
    optimizer.step()
    optimizer.zero_grad()
    
    # Compile and profile the model
    compiled_fn = compile(train_step, graph_transformation)
    # Run the compiled function - peak memory is stored in the global _peak_memory variable
    compiled_fn(model, optimizer, batch)
    # Get the peak memory from the global variable
    peak_memory = _peak_memory
    
    print(f"--- Completed profiling for batch size {batch_size} ---")
    
    return peak_memory

def ensure_reports_directory():
    """Ensure the reports directory exists."""
    reports_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'reports')
    if not os.path.exists(reports_dir):
        os.makedirs(reports_dir)
        print(f"Created reports directory: {reports_dir}")
    return reports_dir

def main():
    """Main function to run the batch memory analysis."""
    print("=== Batch Memory Analysis for ResNet-152 ===")
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Use CUDA if available
    device_str = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device_str}")
    
    # Define batch sizes to test
    batch_sizes = [4, 8, 16, 32]
    
    # Collect peak memory usage for each batch size
    peak_memories = []
    
    try:
        for batch_size in batch_sizes:
            peak_memory = profile_batch_size(batch_size, device_str)
            peak_memories.append(peak_memory)
    except Exception as e:
        print(f"Error during profiling: {e}")
        # If we have at least one successful run, continue with plotting
        if not peak_memories:
            print("No successful profiling runs. Exiting.")
            return
    
    # Convert peak memory to MiB for better readability
    peak_memories_mib = [mem / (1024**2) for mem in peak_memories]
    
    # Create a bar graph
    plt.figure(figsize=(10, 6))
    bars = plt.bar(range(len(batch_sizes)), peak_memories_mib, color='skyblue')
    
    # Add values on top of bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{peak_memories_mib[i]:.2f} MiB',
                ha='center', va='bottom', rotation=0)
    
    # Add labels and title
    plt.xlabel('Batch Size')
    plt.ylabel('Peak Memory Usage (MiB)')
    plt.title('ResNet-152 Peak Memory Usage vs. Batch Size')
    plt.xticks(range(len(batch_sizes)), batch_sizes)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Ensure reports directory exists
    reports_dir = ensure_reports_directory()
    
    # Save the plot
    plot_path = os.path.join(reports_dir, 'resnet152_batch_memory.png')
    plt.savefig(plot_path)
    print(f"\nBatch memory analysis plot saved to: {plot_path}")
    
    # Close the figure to free memory
    plt.close()
    
    # Print summary
    print("\n=== Batch Memory Analysis Summary ===")
    print(f"{'Batch Size':<10} {'Peak Memory (MiB)':<20} {'CSV Files':<40}")
    print("-" * 70)
    for i, batch_size in enumerate(batch_sizes):
        if i < len(peak_memories_mib):
            csv_prefix = f"profiler_stats_bs{batch_size}"
            csv_files = f"{csv_prefix}_node_stats.csv, {csv_prefix}_activation_stats.csv"
            print(f"{batch_size:<10} {peak_memories_mib[i]:<20.2f} {csv_files:<40}")
    
    print("\nCSV files have been generated in the main directory for each batch size.")
    print("These files can be used in Stage 2 for activation checkpointing analysis.")
    print("\n=== Analysis completed ===")

if __name__ == "__main__":
    main()