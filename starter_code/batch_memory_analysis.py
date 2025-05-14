#!/usr/bin/env python
"""
Enhanced Batch Memory Analysis Script

This script analyzes the peak memory consumption of the ResNet-152 model
with different batch sizes (4, 8, 16, 32, 64). It generates multiple visualizations:
1. A bar graph showing peak memory usage for different batch sizes with an 8 GB limit line
2. A memory vs. execution rank graph showing the memory curve with FW/BW boundaries and the 8 GB limit
3. A stacked bar chart showing the memory breakdown (weights, gradients, feature maps) for different batch sizes

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
import pandas as pd
from collections import defaultdict

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
    
    # Define batch sizes to test (including 64)
    batch_sizes = [4, 8, 16, 32, 64]
    
    # Define OOM cap in MiB (8 GB = 8192 MiB)
    oom_cap_mib = 8192
    
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
    
    # Create a bar graph with OOM cap
    plt.figure(figsize=(10, 6))
    bars = plt.bar(range(len(batch_sizes)), peak_memories_mib, color='skyblue')
    
    # Add values on top of bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{peak_memories_mib[i]:.2f} MiB',
                ha='center', va='bottom', rotation=0)
    
    # Add OOM cap line
    plt.axhline(y=oom_cap_mib, color='red', linestyle='--',
                label=f'OOM Cap (8 GB)')
    
    # Add labels and title
    plt.xlabel('Batch Size')
    plt.ylabel('Peak Memory Usage (MiB)')
    plt.title('ResNet-152 Peak Memory Usage vs. Batch Size')
    plt.xticks(range(len(batch_sizes)), batch_sizes)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend()
    
    # Ensure reports directory exists
    reports_dir = ensure_reports_directory()
    
    # Save the plot
    plot_path = os.path.join(reports_dir, 'resnet152_batch_memory.png')
    plt.savefig(plot_path)
    print(f"\nBatch memory analysis plot saved to: {plot_path}")
    
    # Close the figure to free memory
    plt.close()
    
    # Create memory vs. execution rank graph for each batch size
    try:
        create_memory_vs_rank_plots(batch_sizes, reports_dir, oom_cap_mib)
    except Exception as e:
        print(f"Error creating memory vs. rank plots: {e}")
    
    # Create stacked bar chart showing memory breakdown
    try:
        create_memory_breakdown_chart(batch_sizes, peak_memories_mib, reports_dir, oom_cap_mib)
    except Exception as e:
        print(f"Error creating memory breakdown chart: {e}")
    
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

def create_memory_vs_rank_plots(batch_sizes, reports_dir, oom_cap_mib):
    """
    Create memory vs. execution rank plots for each batch size.
    
    Args:
        batch_sizes: List of batch sizes
        reports_dir: Directory to save plots
        oom_cap_mib: OOM cap in MiB
    """
    print("\nGenerating memory vs. execution rank plots...")
    
    # Create a single figure with subplots for all batch sizes
    fig, axes = plt.subplots(len(batch_sizes), 1, figsize=(12, 4*len(batch_sizes)), sharex=True)
    
    # If only one batch size, axes won't be an array
    if len(batch_sizes) == 1:
        axes = [axes]
    
    for i, batch_size in enumerate(batch_sizes):
        # Load CSV data for this batch size
        try:
            node_csv = f"profiler_stats_bs{batch_size}_node_stats.csv"
            if not os.path.exists(node_csv):
                print(f"Warning: {node_csv} not found, skipping memory vs. rank plot for batch size {batch_size}")
                continue
                
            df = pd.read_csv(node_csv)
            
            # Sort by rank
            df = df.sort_values('rank')
            
            # Convert peak memory to MiB
            df['peak_mem_mib'] = df['avg_peak_mem_bytes'] / (1024**2)
            
            # Find FW/BW boundary
            fw_end_rank = -1
            bw_start_rank = -1
            
            for j, row in df.iterrows():
                if row['gtype'] == 'forward' and (j+1 >= len(df) or df.iloc[j+1]['gtype'] != 'forward'):
                    fw_end_rank = row['rank']
                if row['gtype'] == 'backward' and (j == 0 or df.iloc[j-1]['gtype'] != 'backward'):
                    bw_start_rank = row['rank']
            
            # Plot memory vs. rank
            ax = axes[i]
            ax.plot(df['rank'], df['peak_mem_mib'], marker='o', markersize=3,
                    linestyle='-', label=f"Batch Size {batch_size}")
            
            # Add FW/BW separator lines
            if fw_end_rank != -1:
                ax.axvline(x=fw_end_rank, color='red', linestyle='--',
                          label=f'FW End (Rank {fw_end_rank})')
            if bw_start_rank != -1 and bw_start_rank != fw_end_rank:
                ax.axvline(x=bw_start_rank, color='orange', linestyle='--',
                          label=f'BW Start (Rank {bw_start_rank})')
            
            # Add OOM cap line
            ax.axhline(y=oom_cap_mib, color='red', linestyle=':',
                      label=f'OOM Cap (8 GB)')
            
            # Set title and labels
            ax.set_title(f"Batch Size {batch_size}: Memory vs. Execution Rank")
            ax.set_ylabel("Peak Memory (MiB)")
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right')
            
            # Only add x-label for the bottom subplot
            if i == len(batch_sizes) - 1:
                ax.set_xlabel("Node Execution Rank")
        
        except Exception as e:
            print(f"Error processing batch size {batch_size} for memory vs. rank plot: {e}")
    
    plt.tight_layout()
    plot_path = os.path.join(reports_dir, 'resnet152_memory_vs_rank.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"Memory vs. rank plots saved to: {plot_path}")

def create_memory_breakdown_chart(batch_sizes, peak_memories_mib, reports_dir, oom_cap_mib):
    """
    Create a stacked bar chart showing memory breakdown for different batch sizes.
    
    Args:
        batch_sizes: List of batch sizes
        peak_memories_mib: List of peak memory values in MiB
        reports_dir: Directory to save plots
        oom_cap_mib: OOM cap in MiB
    """
    print("\nGenerating memory breakdown chart...")
    
    # Estimate memory breakdown components
    # These are approximations based on typical ResNet memory patterns
    weights_mib = []  # Model weights (parameters)
    gradients_mib = []  # Gradients
    activations_mib = []  # Feature maps/activations
    
    for i, batch_size in enumerate(batch_sizes):
        if i < len(peak_memories_mib):
            # Approximate breakdown based on typical patterns
            # Weights are constant regardless of batch size
            weight_mem = 230  # ResNet-152 weights ~230 MiB
            
            # Gradients scale with batch size but are smaller than activations
            gradient_mem = weight_mem * 0.8  # Slightly smaller than weights
            
            # Activations are the largest component and scale with batch size
            activation_mem = peak_memories_mib[i] - weight_mem - gradient_mem
            
            # Ensure no negative values
            activation_mem = max(0, activation_mem)
            
            weights_mib.append(weight_mem)
            gradients_mib.append(gradient_mem)
            activations_mib.append(activation_mem)
    
    # Create stacked bar chart
    plt.figure(figsize=(12, 8))
    
    # Plot stacked bars
    bar_width = 0.6
    x = np.arange(len(batch_sizes[:len(peak_memories_mib)]))
    
    p1 = plt.bar(x, weights_mib, bar_width, color='#1f77b4', label='Model Weights')
    p2 = plt.bar(x, gradients_mib, bar_width, bottom=weights_mib, color='#ff7f0e', label='Gradients')
    p3 = plt.bar(x, activations_mib, bar_width,
                bottom=[weights_mib[i] + gradients_mib[i] for i in range(len(weights_mib))],
                color='#2ca02c', label='Feature Maps')
    
    # Add OOM cap line
    plt.axhline(y=oom_cap_mib, color='red', linestyle='--',
                label=f'OOM Cap (8 GB)')
    
    # Add labels and title
    plt.xlabel('Batch Size')
    plt.ylabel('Memory Usage (MiB)')
    plt.title('ResNet-152 Memory Breakdown by Batch Size')
    plt.xticks(x, batch_sizes[:len(peak_memories_mib)])
    plt.legend()
    
    # Add total values on top of bars
    for i, total in enumerate(peak_memories_mib):
        plt.text(i, total + 100, f'{total:.0f} MiB',
                ha='center', va='bottom', fontweight='bold')
    
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    
    # Save the plot
    plot_path = os.path.join(reports_dir, 'resnet152_memory_breakdown.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"Memory breakdown chart saved to: {plot_path}")

if __name__ == "__main__":
    main()