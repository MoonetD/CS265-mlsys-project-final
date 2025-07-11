#!/usr/bin/env python
"""
Enhanced Batch Memory Analysis Script

This script analyzes the peak memory consumption of neural network models
with different batch sizes. It supports:
- ResNet-152: A deep convolutional neural network for image classification
- Transformer: A sequence-to-sequence model based on attention mechanisms

It generates multiple visualizations:
1. A bar graph showing peak memory usage for different batch sizes with an 1.5 GB limit line
2. A memory vs. execution rank graph showing the memory curve with FW/BW boundaries and the 1.5 GB limit
3. A stacked bar chart showing the memory breakdown (weights, gradients, feature maps) for different batch sizes

To run this script:
    conda run -n ml_env python starter_code/batch_memory_analysis.py
    
To specify batch sizes:
    conda run -n ml_env python starter_code/batch_memory_analysis.py --batch-sizes 4 8 16
    
To specify model:
    conda run -n ml_env python starter_code/batch_memory_analysis.py --model resnet
    conda run -n ml_env python starter_code/batch_memory_analysis.py --model transformer
"""

# Standard library imports
import os
import numpy as np
import pandas as pd
import argparse
from collections import defaultdict
from functools import wraps
import math

# PyTorch imports
import torch
import torch.nn as nn
import torch.fx as fx
import torchvision.models as models

# Visualization imports
import matplotlib.pyplot as plt

# Custom profiling utilities
from graph_prof import GraphProfiler
from graph_tracer import SEPFunction, compile

# Transformer model implementation
class TransformerEncoderLayer(nn.Module):
    """
    Transformer Encoder Layer based on the paper "Attention Is All You Need"
    
    This implementation follows the architecture described in the original paper
    with multi-head self-attention and position-wise feed-forward networks.
    """
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        # Multi-head self-attention
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        # Position-wise feed-forward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout for each sub-layer
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        # Activation function
        self.activation = nn.ReLU()
        
    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # Multi-head self-attention with residual connection and layer norm
        src2 = self.norm1(src)
        src2, _ = self.self_attn(src2, src2, src2, attn_mask=src_mask,
                                key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        
        # Position-wise feed-forward network with residual connection and layer norm
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        
        return src

class TransformerModel(nn.Module):
    """
    Transformer model for activation checkpointing experiments
    
    This model implements a Transformer encoder stack with the following specifications:
    - Embedding dimension (d_model): 512
    - Number of encoder layers: 6
    - Number of attention heads: 8
    - Feed-forward dimension: 2048
    - Dropout rate: 0.1
    - Vocabulary size: 30,000
    - Maximum sequence length: 512
    """
    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 dim_feedforward=2048, dropout=0.1, vocab_size=30000, max_seq_len=512):
        super().__init__()
        
        # Model dimensions
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        
        # Token embedding
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional encoding (fixed sinusoidal)
        self.register_buffer("positional_encoding", self._create_positional_encoding(max_seq_len, d_model))
        
        # Encoder layers
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_encoder_layers)
        ])
        
        # Final layer normalization
        self.norm = nn.LayerNorm(d_model)
        
        # Output projection (to vocabulary size)
        self.output_projection = nn.Linear(d_model, vocab_size)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Initialize parameters
        self._init_parameters()
        
    def _create_positional_encoding(self, max_seq_len, d_model):
        """Create fixed sinusoidal positional encodings"""
        position = torch.arange(max_seq_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pos_encoding = torch.zeros(max_seq_len, d_model)
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)
        
        return pos_encoding
    
    def _init_parameters(self):
        """Initialize model parameters"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x):
        """
        Forward pass through the Transformer model
        
        Args:
            x: Input tensor of shape (batch_size, seq_len)
                For profiling purposes, this can be random data
        
        Returns:
            Output tensor of shape (batch_size, seq_len, vocab_size)
        """
        # Get sequence length and batch size
        batch_size, seq_len = x.shape
        
        # If input is just random numbers (for profiling), treat as token indices
        if x.dtype != torch.long:
            # Convert to token indices (integers between 0 and vocab_size-1)
            x = (x * (self.vocab_size - 1)).long().abs()
        
        # Token embedding
        x = self.embedding(x) * math.sqrt(self.d_model)
        
        # Add positional encoding
        x = x + self.positional_encoding[:seq_len].unsqueeze(0)
        
        # Apply dropout
        x = self.dropout(x)
        
        # Pass through encoder layers
        for layer in self.encoder_layers:
            x = layer(x)
        
        # Apply final layer normalization
        x = self.norm(x)
        
        # Project to vocabulary size
        x = self.output_projection(x)
        
        return x

def train_step(model, optim, batch):
    """
    Simple training step function that will be traced by the compiler.
    
    Args:
        model: The neural network model
        optim: The optimizer
        batch: Input data batch
        
    This function performs a complete training step:
    1. Forward pass
    2. Loss calculation
    3. Backward pass
    4. Optimizer step
    5. Gradient zeroing
    """
    # Forward pass through the model
    loss = model(batch).sum()
    # Apply SEPFunction to mark the separation between forward and backward passes
    loss = SEPFunction.apply(loss)
    # Backward pass to compute gradients
    loss.backward()
    # Update model parameters
    optim.step()
    # Reset gradients to zero
    optim.zero_grad()

# Global variables
_peak_memory = 0  # Store peak memory between function calls
_CURRENT_PROFILING_BATCH_SIZE = 0  # Store the current batch size for filename generation
_CURRENT_MODEL_TYPE = "resnet"  # Store the current model type for filename generation

def graph_transformation(gm: fx.GraphModule, args: any) -> fx.GraphModule:
    """
    Graph transformation function that profiles the model execution.
    This is passed to the compile function to analyze the traced graph.
    
    Args:
        gm: The FX GraphModule to be transformed
        args: Arguments passed to the original function
        
    Returns:
        The original graph module (required by the compile function).
        Peak memory is stored in the global _peak_memory variable.
    """
    global _peak_memory
    global _CURRENT_PROFILING_BATCH_SIZE
    global _CURRENT_MODEL_TYPE  # Add global variable for model type
    
    # Extract batch size from trace args for debugging purposes
    batch_size_from_trace_args = args[2].shape[0]
    print(f"\nGraph transformation called. Trace args batch size: {batch_size_from_trace_args}. Actual profiling batch size: {_CURRENT_PROFILING_BATCH_SIZE}")
    
    # Initialize the GraphProfiler with the graph module
    graph_profiler = GraphProfiler(gm)
    
    # Define profiling parameters
    warm_up_iters, profile_iters = 1, 3
    
    with torch.no_grad():
        # Perform warm-up runs to eliminate initialization overhead
        for _ in range(warm_up_iters):
            graph_profiler.run(*args)
        
        # Reset profiler statistics before actual profiling
        graph_profiler.reset_stats()
        
        # Perform actual profiling runs
        for _ in range(profile_iters):
            graph_profiler.run(*args)
    
    # Process and aggregate the collected statistics
    graph_profiler.aggregate_stats(num_runs=profile_iters)
    
    # Create directory for saving reports if it doesn't exist
    reports_dir = ensure_reports_directory()
    
    # Use the global model type variable that was set in profile_batch_size
    csv_prefix = f"profiler_stats_{_CURRENT_MODEL_TYPE}_bs{_CURRENT_PROFILING_BATCH_SIZE}"
    csv_path = os.path.join(reports_dir, csv_prefix)
    
    # Save profiling statistics to CSV files
    graph_profiler.save_stats_to_csv(filename_prefix=csv_path)
    
    # Save all node statistics to a comprehensive CSV file
    graph_profiler.save_all_nodes_to_csv(filename_prefix=csv_path)
    
    # Verify that CSV files were created successfully
    node_stats_path = f"{csv_path}_node_stats.csv"
    activation_stats_path = f"{csv_path}_activation_stats.csv"
    all_node_stats_path = f"{csv_path}_allNode_stats.csv"
    
    expected_files = [node_stats_path, activation_stats_path, all_node_stats_path]
    missing_files = [f for f in expected_files if not os.path.exists(f)]
    
    if not missing_files:
        print(f"CSV files successfully saved to: {node_stats_path}, {activation_stats_path}, and {all_node_stats_path}")
    else:
        # Identify which files are missing
        print(f"WARNING: The following CSV files were not created: {', '.join(missing_files)}")
        # Fail fast if CSV files are missing to prevent downstream errors
        if missing_files:
            raise RuntimeError(f"Critical CSV files are missing: {', '.join(missing_files)}. Stage 1 profiling failed.")
    
    # Extract peak memory usage from profiler results
    _peak_memory = max(graph_profiler.median_peak_mem_node.values()) if graph_profiler.median_peak_mem_node else 0
    
    # Print peak memory usage in MiB for better readability
    print(f"Peak memory usage: {_peak_memory / (1024**2):.2f} MiB")
    
    # Return the original graph module as required by the compile function
    return gm

def profile_batch_size(batch_size, model_type='resnet', device_str='cuda:0'):
    """
    Profile a neural network model with a specific batch size.
    
    Args:
        batch_size: The batch size to use for profiling
        model_type: The type of model to profile ('resnet' or 'transformer')
        device_str: The device to use for profiling (default: 'cuda:0')
        
    Returns:
        The peak memory usage in bytes
    """
    # Set the global variables for CSV filename generation
    global _CURRENT_PROFILING_BATCH_SIZE
    global _CURRENT_MODEL_TYPE
    _CURRENT_PROFILING_BATCH_SIZE = batch_size
    _CURRENT_MODEL_TYPE = model_type.lower()

    print(f"\n--- Profiling {model_type.upper()} with batch size {batch_size} ---")
    
    # Create the model based on the specified type
    if model_type.lower() == 'resnet':
        # Create ResNet-152 model and move it to the specified device
        model = models.resnet152(weights=models.ResNet152_Weights.IMAGENET1K_V1).to(device_str)
        
        # Create a random batch of data with ImageNet dimensions (3x224x224)
        batch = torch.randn(batch_size, 3, 224, 224).to(device_str)
    elif model_type.lower() == 'transformer':
        # Create Transformer model with the specified dimensions
        model = TransformerModel(
            d_model=512,
            nhead=8,
            num_encoder_layers=6,
            dim_feedforward=2048,
            dropout=0.1,
            vocab_size=30000,
            max_seq_len=512
        ).to(device_str)
        
        # Create a random batch of token indices with sequence length 512
        # For profiling purposes, we use a smaller sequence length to fit in memory
        seq_len = min(512, 128)  # Use 128 as a reasonable sequence length for profiling
        batch = torch.randint(0, 30000, (batch_size, seq_len)).to(device_str)
    else:
        raise ValueError(f"Unknown model type: {model_type}. Expected 'resnet' or 'transformer'")
    
    # Create an optimizer with foreach and capturable options for better compatibility with tracing
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.001, foreach=True, capturable=True
    )
    
    # Initialize gradients for all parameters to ensure optimizer step is traceable
    for param in model.parameters():
        if param.requires_grad:
            param.grad = torch.rand_like(param, device=device_str)
    
    # Perform one optimizer step to initialize internal optimizer states
    optimizer.step()
    optimizer.zero_grad()
    
    # Compile the training step function with our profiling transformation
    compiled_fn = compile(train_step, graph_transformation)
    
    # Run the compiled function - peak memory is stored in the global _peak_memory variable
    compiled_fn(model, optimizer, batch)
    
    # Retrieve the peak memory from the global variable
    peak_memory = _peak_memory
    
    print(f"--- Completed profiling for batch size {batch_size} ---")
    
    return peak_memory

def ensure_reports_directory():
    """
    Ensure the reports directory exists.
    
    Returns:
        Path to the reports directory
    """
    # Get the path to the reports directory (one level up from the script location)
    reports_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'reports')
    
    # Create the directory if it doesn't exist
    if not os.path.exists(reports_dir):
        os.makedirs(reports_dir)
        print(f"Created reports directory: {reports_dir}")
        
    return reports_dir

def parse_args():
    """
    Parse command line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description='Batch Memory Analysis for Neural Network Models')
    parser.add_argument('--batch-sizes', type=int, nargs='+', default=[4, 8],
                        help='Batch sizes to profile (default: 4 8)')
    parser.add_argument('--model', type=str, choices=['resnet', 'transformer'], default='resnet',
                        help='Model to profile (default: resnet)')
    return parser.parse_args()

def main():
    """
    Main function to run the batch memory analysis.
    
    This function:
    1. Profiles the selected model with different batch sizes
    2. Generates visualizations of memory usage
    3. Saves results to the reports directory
    """
    # Parse command line arguments
    args = parse_args()
    
    # Get model type from arguments
    model_type = args.model
    
    print(f"=== Batch Memory Analysis for {model_type.upper()} ===")
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Use CUDA if available, otherwise fall back to CPU
    device_str = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device_str}")
    
    # Use batch sizes from command line arguments
    batch_sizes = args.batch_sizes
    print(f"Profiling batch sizes: {batch_sizes}")
    
    # Define out-of-memory cap in MiB (1.5 GB = 1536 MiB)
    oom_cap_mib = 1536
    
    # Lists to store profiling results
    peak_memories = []  # For memory usage
    iter_times = []     # For iteration latency
    
    try:
        # Profile each batch size
        for batch_size in batch_sizes:
            # Run profiling and get peak memory
            peak_memory = profile_batch_size(batch_size, model_type, device_str)
            peak_memories.append(peak_memory)
            
            # Load node stats CSV to calculate iteration time
            try:
                # Include model type in the CSV filename
                node_csv = os.path.join(ensure_reports_directory(), f"profiler_stats_{model_type}_bs{batch_size}_node_stats.csv")
                if os.path.exists(node_csv):
                    # Read CSV and sum up all node execution times
                    df = pd.read_csv(node_csv)
                    iter_time = df['median_run_time_s'].sum()
                    iter_times.append(iter_time)
                    print(f"Iteration time for batch size {batch_size}: {iter_time:.6f} seconds")
                else:
                    print(f"Warning: Could not find {node_csv} to calculate iteration time")
                    iter_times.append(0)
            except Exception as e:
                print(f"Error calculating iteration time for batch size {batch_size}: {e}")
                iter_times.append(0)
                
    except Exception as e:
        print(f"Error during profiling: {e}")
        # Continue with plotting if we have at least one successful run
        if not peak_memories:
            print("No successful profiling runs. Exiting.")
            return
    
    # Convert peak memory from bytes to MiB for better readability
    peak_memories_mib = [mem / (1024**2) for mem in peak_memories]
    
    # Create latency-vs-batch plot if we have valid timing data
    if any(iter_times):
        plt.figure(figsize=(10, 6))
        plt.plot(batch_sizes[:len(iter_times)], iter_times, marker='o', linestyle='-', color='purple')
        
        # Add time values as labels on each data point
        for i, time in enumerate(iter_times):
            if time > 0:  # Only label non-zero times
                plt.text(batch_sizes[i], time + 0.01, f'{time:.3f}s', ha='center', va='bottom')
        
        plt.xlabel('Batch Size')
        plt.ylabel('Iteration Time (seconds)')
        # Set the title to include the model type
        title_model = "ResNet-152" if model_type == "resnet" else "Transformer"
        plt.title(f'{title_model} Iteration Time vs. Batch Size')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Save the latency plot
        reports_dir = ensure_reports_directory()
        plot_path = os.path.join(reports_dir, f'{model_type}_latency_comparison.png')
        plt.savefig(plot_path)
        plt.close()
        print(f"\nLatency comparison plot saved to: {plot_path}")
    
    # Create a bar graph showing peak memory usage with OOM cap
    plt.figure(figsize=(10, 6))
    bars = plt.bar(range(len(batch_sizes)), peak_memories_mib, color='skyblue')
    
    # Set the title to include the model type
    title_model = "ResNet-152" if model_type == "resnet" else "Transformer"
    
    # Add memory values as labels on top of each bar
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{peak_memories_mib[i]:.2f} MiB',
                ha='center', va='bottom', rotation=0)
    
    # Add horizontal line showing the OOM cap (1.5 GB)
    plt.axhline(y=oom_cap_mib, color='red', linestyle='--',
                label=f'OOM Cap (1.5 GB)')
    
    # Add labels, title, and formatting
    plt.xlabel('Batch Size')
    plt.ylabel('Peak Memory Usage (MiB)')
    plt.title(f'{title_model} Peak Memory Usage vs. Batch Size')
    plt.xticks(range(len(batch_sizes)), batch_sizes)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend()
    
    # Ensure reports directory exists
    reports_dir = ensure_reports_directory()
    
    # Save the memory usage bar chart
    plot_path = os.path.join(reports_dir, f'{model_type}_batch_memory.png')
    plt.savefig(plot_path)
    print(f"\nBatch memory analysis plot saved to: {plot_path}")
    
    # Close the figure to free memory
    plt.close()
    
    # Create memory vs. execution rank graph for each batch size
    try:
        create_memory_vs_rank_plots(batch_sizes, reports_dir, oom_cap_mib, model_type)
    except Exception as e:
        print(f"Error creating memory vs. rank plots: {e}")
    
    # Create stacked bar chart showing memory breakdown
    try:
        create_memory_breakdown_chart(batch_sizes, peak_memories_mib, reports_dir, oom_cap_mib, model_type)
    except Exception as e:
        print(f"Error creating memory breakdown chart: {e}")
    
    # Print summary table of results
    print("\n=== Batch Memory Analysis Summary ===")
    print(f"{'Batch Size':<10} {'Peak Memory (MiB)':<20} {'CSV Files':<65}")
    print("-" * 95)
    for i, batch_size in enumerate(batch_sizes):
        if i < len(peak_memories_mib):
            # Check if CSV files exist for this batch size
            csv_prefix = f"profiler_stats_{model_type}_bs{batch_size}"
            node_stats_path = os.path.join(reports_dir, f"{csv_prefix}_node_stats.csv")
            activation_stats_path = os.path.join(reports_dir, f"{csv_prefix}_activation_stats.csv")
            all_node_stats_path = os.path.join(reports_dir, f"{csv_prefix}_allNode_stats.csv")
            
            # Use checkmarks/X to indicate file presence
            node_status = "✓" if os.path.exists(node_stats_path) else "✗"
            activation_status = "✓" if os.path.exists(activation_stats_path) else "✗"
            all_node_status = "✓" if os.path.exists(all_node_stats_path) else "✗"
            
            csv_files = f"Node: {node_status}, Activation: {activation_status}, AllNode: {all_node_status}"
            print(f"{batch_size:<10} {peak_memories_mib[i]:<20.2f} {csv_files:<65}")
    
    print("\nCSV files have been generated in the reports directory for each batch size.")
    print("These files can be used in Stage 2 for activation checkpointing analysis.")
    print("\n=== Analysis completed ===")

def create_memory_vs_rank_plots(batch_sizes, reports_dir, oom_cap_mib, model_type):
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
            # Check if node stats CSV exists
            node_csv = os.path.join(reports_dir, f"profiler_stats_{model_type}_bs{batch_size}_node_stats.csv")
            if not os.path.exists(node_csv):
                print(f"Warning: {node_csv} not found, skipping memory vs. rank plot for batch size {batch_size}")
                continue
                
            # Load and process the CSV data
            df = pd.read_csv(node_csv)
            
            # Sort nodes by execution rank
            df = df.sort_values('rank')
            
            # Convert peak memory from bytes to MiB
            df['peak_mem_mib'] = df['median_peak_mem_bytes'] / (1024**2)
            
            # Find the boundary between forward and backward passes
            fw_end_rank = -1
            bw_start_rank = -1
            
            # Scan through nodes to identify the FW/BW boundary
            for j, row in df.iterrows():
                # Last forward node is the one where the next node is not forward
                if row['gtype'] == 'forward' and (j+1 >= len(df) or df.iloc[j+1]['gtype'] != 'forward'):
                    fw_end_rank = row['rank']
                # First backward node is the one where the previous node is not backward
                if row['gtype'] == 'backward' and (j == 0 or df.iloc[j-1]['gtype'] != 'backward'):
                    bw_start_rank = row['rank']
            
            # Plot memory vs. rank curve
            ax = axes[i]
            ax.plot(df['rank'], df['peak_mem_mib'], marker='o', markersize=3,
                    linestyle='-', label=f"Batch Size {batch_size}")
            
            # Add vertical lines to mark forward/backward boundaries
            if fw_end_rank != -1:
                ax.axvline(x=fw_end_rank, color='red', linestyle='--',
                          label=f'FW End (Rank {fw_end_rank})')
            if bw_start_rank != -1 and bw_start_rank != fw_end_rank:
                ax.axvline(x=bw_start_rank, color='orange', linestyle='--',
                          label=f'BW Start (Rank {bw_start_rank})')
            
            # Add horizontal line for OOM cap
            ax.axhline(y=oom_cap_mib, color='red', linestyle=':',
                      label=f'OOM Cap (1.5 GB)')
            
            # Set title and labels for this subplot
            ax.set_title(f"Batch Size {batch_size}: Memory vs. Execution Rank")
            ax.set_ylabel("Peak Memory (MiB)")
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right')
            
            # Only add x-label for the bottom subplot
            if i == len(batch_sizes) - 1:
                ax.set_xlabel("Node Execution Rank")
        
        except Exception as e:
            print(f"Error processing batch size {batch_size} for memory vs. rank plot: {e}")
    
    # Adjust layout and save the figure
    plt.tight_layout()
    plot_path = os.path.join(reports_dir, f'{model_type}_memory_vs_rank.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"Memory vs. rank plots saved to: {plot_path}")
def create_memory_breakdown_chart(batch_sizes, peak_memories_mib, reports_dir, oom_cap_mib, model_type):
    """
    Create a stacked bar chart showing memory breakdown for different batch sizes.
    
    Args:
        batch_sizes: List of batch sizes
        peak_memories_mib: List of peak memory values in MiB
        reports_dir: Directory to save plots
        oom_cap_mib: OOM cap in MiB
    """
    print("\nGenerating memory breakdown chart...")
    
    # Lists to store memory components for each batch size
    weights_mib = []  # Model weights (parameters)
    gradients_mib = []  # Gradients
    activations_mib = []  # Feature maps/activations
    
    # Calculate memory breakdown for each batch size
    for i, batch_size in enumerate(batch_sizes):
        if i < len(peak_memories_mib):
            # Approximate breakdown based on typical patterns and model type
            if model_type == 'resnet':
                # ResNet-152 weights ~230 MiB
                weight_mem = 230
            else:
                # Transformer model weights (smaller than ResNet)
                # Rough estimate: embedding + attention + feedforward layers
                # 512 * 30000 + 6 * (512^2 * 8 + 512 * 2048 * 2) ~ 60 MiB
                weight_mem = 60
            
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
    
    # Plot each memory component as a separate layer
    p1 = plt.bar(x, weights_mib, bar_width, color='#1f77b4', label='Model Weights')
    p2 = plt.bar(x, gradients_mib, bar_width, bottom=weights_mib, color='#ff7f0e', label='Gradients')
    p3 = plt.bar(x, activations_mib, bar_width,
                bottom=[weights_mib[i] + gradients_mib[i] for i in range(len(weights_mib))],
                color='#2ca02c', label='Feature Maps')
    
    # Add horizontal line for OOM cap
    plt.axhline(y=oom_cap_mib, color='red', linestyle='--',
                label=f'OOM Cap (1.5 GB)')
    
    # Add labels and title
    plt.xlabel('Batch Size')
    plt.ylabel('Memory Usage (MiB)')
    title_model = "ResNet-152" if model_type == "resnet" else "Transformer"
    plt.title(f'{title_model} Memory Breakdown by Batch Size')
    plt.xticks(x, batch_sizes[:len(peak_memories_mib)])
    plt.legend()
    
    # Add total memory values on top of each bar
    for i, total in enumerate(peak_memories_mib):
        plt.text(i, total + 100, f'{total:.0f} MiB',
                ha='center', va='bottom', fontweight='bold')
    
    # Add grid for better readability
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    
    # Save the plot
    plot_path = os.path.join(reports_dir, f'{model_type}_memory_breakdown.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"Memory breakdown chart saved to: {plot_path}")

if __name__ == "__main__":
    main()

    