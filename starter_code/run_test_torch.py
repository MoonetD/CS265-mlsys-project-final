"""
Test script for activation checkpointing using PyTorch's built-in checkpointing mechanism.

This script supports both ResNet-152 and Transformer models, and can generate
comparison graphs for different batch sizes with and without activation checkpointing.
"""

import os
import sys
import torch
import torch.nn as nn
import torchvision.models as models
import pandas as pd
import numpy as np
import logging
import time
import argparse
import matplotlib.pyplot as plt
import math
from typing import Dict, List, Tuple, Set, Any, Optional

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import our activation checkpointing implementation
from activation_checkpointing_torch import apply_activation_checkpointing

# Import the TransformerModel from batch_memory_analysis.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from batch_memory_analysis import TransformerModel

def load_data(ac_decisions_path: str, activation_stats_path: str, node_stats_path: str) -> Tuple[Dict[str, str], Dict[str, Dict[str, int]], Dict[str, Dict[str, Any]]]:
    """
    Load data from CSV files.
    
    Args:
        ac_decisions_path: Path to the activation checkpointing decisions CSV
        activation_stats_path: Path to the activation statistics CSV
        node_stats_path: Path to the node statistics CSV
        
    Returns:
        Tuple of (activation decisions, activation liveness, node statistics)
    """
    logger = logging.getLogger(__name__)
    
    # Load activation checkpointing decisions
    ac_decisions_df = pd.read_csv(ac_decisions_path)
    ac_decisions = dict(zip(ac_decisions_df['activation_name'], ac_decisions_df['decision']))
    logger.info(f"Loaded {len(ac_decisions)} activation checkpointing decisions")
    
    # Load activation liveness information
    activation_stats_df = pd.read_csv(activation_stats_path)
    activation_liveness = {}
    for _, row in activation_stats_df.iterrows():
        activation_name = row['activation_name']
        activation_liveness[activation_name] = {
            'creation_rank': int(row['creation_rank']) if 'creation_rank' in row else -1,
            'last_fw_use_rank': int(row['last_fw_use_rank']) if 'last_fw_use_rank' in row else -1,
            'first_bw_use_rank': int(row['first_bw_use_rank']) if 'first_bw_use_rank' in row else -1,
            'median_mem_size_bytes': int(row['median_mem_size_bytes']) if 'median_mem_size_bytes' in row else 0,
            'recomp_time_s': float(row['recomp_time_s']) if 'recomp_time_s' in row else 0.0
        }
    logger.info(f"Loaded {len(activation_liveness)} activation liveness records")
    
    # Load node statistics
    node_stats_df = pd.read_csv(node_stats_path)
    node_stats = {}
    for _, row in node_stats_df.iterrows():
        node_name = row['node_name']
        node_stats[node_name] = {
            'rank': int(row['rank']) if 'rank' in row else -1,
            'runtime_s': float(row['runtime_s']) if 'runtime_s' in row else 0.0,
            'peak_memory_bytes': int(row['peak_memory_bytes']) if 'peak_memory_bytes' in row else 0
        }
    logger.info(f"Loaded {len(node_stats)} node statistics")
    
    return ac_decisions, activation_liveness, node_stats

def measure_memory_and_time(model: nn.Module, input_tensor: torch.Tensor, include_backward: bool = True, 
                           num_iterations: int = 5) -> Tuple[float, float]:
    """
    Measure the peak memory usage and average execution time of a model.
    
    Args:
        model: The model to measure
        input_tensor: The input tensor to the model
        include_backward: Whether to include the backward pass in the measurement
        num_iterations: Number of iterations to average over
        
    Returns:
        Tuple of (peak memory usage in MB, average execution time in ms)
    """
    # Warm up
    with torch.no_grad():
        model(input_tensor)
    
    # Clear GPU cache
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    # Measure memory and time
    start_time = time.time()
    
    for _ in range(num_iterations):
        # Forward pass
        output = model(input_tensor)
        
        # Backward pass
        if include_backward:
            output.sum().backward()
    
    end_time = time.time()
    
    # Calculate metrics
    peak_memory = torch.cuda.max_memory_allocated() / (1024 * 1024)  # Convert to MB
    avg_time = (end_time - start_time) * 1000 / num_iterations  # Convert to ms
    
    return peak_memory, avg_time

def get_memory_breakdown(model: nn.Module, input_tensor: torch.Tensor) -> Dict[str, float]:
    """
    Get a breakdown of memory usage by category.
    
    Args:
        model: The model to measure
        input_tensor: The input tensor to the model
        
    Returns:
        Dict mapping memory categories to memory usage in MB
    """
    # Parameters
    param_size = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
    
    # Gradients (same size as parameters for most optimizers)
    grad_size = param_size
    
    # Optimizer state (typically 2x parameters for Adam)
    optimizer_size = 2 * param_size
    
    # Measure activations
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    # Forward pass
    output = model(input_tensor)
    
    # Calculate activations
    activations_size = (torch.cuda.max_memory_allocated() / (1024 * 1024)) - param_size
    
    # Total memory (with some fragmentation)
    total_size = param_size + grad_size + optimizer_size + activations_size
    
    return {
        'parameters': param_size,
        'gradients': grad_size,
        'optimizer_state': optimizer_size,
        'activations': activations_size,
        'total': total_size
    }

def create_model(model_type: str, batch_size: int) -> Tuple[nn.Module, torch.Tensor]:
    """
    Create a model and input tensor based on the model type.
    
    Args:
        model_type: Type of model to create ('resnet' or 'transformer')
        batch_size: Batch size for the input tensor
        
    Returns:
        Tuple of (model, input_tensor)
    """
    if model_type == 'resnet':
        # Create ResNet-152 model
        logger.info(f"Creating ResNet-152 model with batch size {batch_size}")
        model = models.resnet152(weights=None)
        model = model.cuda()
        model.eval()  # Set to evaluation mode
        
        # Create input tensor for ResNet
        input_tensor = torch.randn(batch_size, 3, 224, 224, device='cuda')
    
    elif model_type == 'transformer':
        # Create Transformer model
        logger.info(f"Creating Transformer model with batch size {batch_size}")
        model = TransformerModel(
            d_model=512,
            nhead=8,
            num_encoder_layers=6,
            dim_feedforward=2048,
            dropout=0.1,
            vocab_size=30000,
            max_seq_len=512
        ).cuda()
        model.eval()  # Set to evaluation mode
        
        # Create input tensor for Transformer
        # Use a smaller sequence length for testing
        seq_len = 128
        input_tensor = torch.randint(0, 30000, (batch_size, seq_len), device='cuda')
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model, input_tensor

def get_csv_paths(model_type: str, batch_size: int) -> Tuple[str, str, str]:
    """
    Get paths to CSV files for the specified model type and batch size.
    
    Args:
        model_type: Type of model ('resnet' or 'transformer')
        batch_size: Batch size
        
    Returns:
        Tuple of (ac_decisions_path, activation_stats_path, node_stats_path)
    """
    base_path = "../reports"
    
    ac_decisions_path = f"{base_path}/ac_decisions_{model_type}_bs{batch_size}.csv"
    activation_stats_path = f"{base_path}/profiler_stats_{model_type}_bs{batch_size}_activation_stats.csv"
    node_stats_path = f"{base_path}/profiler_stats_{model_type}_bs{batch_size}_node_stats.csv"
    
    return ac_decisions_path, activation_stats_path, node_stats_path

def plot_memory_comparison(model_type: str, batch_sizes: List[int], baseline_memories: List[float],
                          ac_memories: List[float], save_path: str):
    """
    Plot memory comparison between with and without activation checkpointing.
    
    Args:
        model_type: Type of model ('resnet' or 'transformer')
        batch_sizes: List of batch sizes
        baseline_memories: List of baseline memory values (without AC)
        ac_memories: List of memory values with AC
        save_path: Path to save the plot
    """
    plt.figure(figsize=(12, 8))
    
    x = np.arange(len(batch_sizes))
    width = 0.35
    
    # Convert MB to GB for clearer display
    baseline_gb = [mem / 1024 for mem in baseline_memories]
    ac_gb = [mem / 1024 for mem in ac_memories]
    
    # Calculate percentage reduction
    percent_reduction = [(baseline_memories[i] - ac_memories[i]) / baseline_memories[i] * 100
                         for i in range(len(batch_sizes))]
    percent_reduction_labels = [f"{p:.1f}%" for p in percent_reduction]
    
    bars1 = plt.bar(x - width/2, baseline_gb, width, label='Without AC', color='#1F77B4')
    bars2 = plt.bar(x + width/2, ac_gb, width, label='With AC', color='#FF7F0E')
    
    # Add a red line at 1.5 GB memory budget
    memory_budget_gb = 1.5
    plt.axhline(y=memory_budget_gb, color='red', linestyle='-', linewidth=2, label='Memory Budget (1.5 GB)')
    
    plt.xlabel('Batch Size', fontweight='bold')
    plt.ylabel('Peak Memory (GB)', fontweight='bold')
    title_model = "ResNet-152" if model_type == "resnet" else "Transformer"
    plt.title(f'Peak Memory Usage vs. Batch Size - {title_model}', fontweight='bold', fontsize=18)
    plt.xticks(x, batch_sizes, fontweight='bold')
    plt.legend(fontsize=12)
    
    # Add value labels on top of the bars
    for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
        plt.text(bar1.get_x() + bar1.get_width()/2, bar1.get_height() + 0.05,
                f"{baseline_gb[i]:.2f}", ha='center', va='bottom', fontweight='bold')
        plt.text(bar2.get_x() + bar2.get_width()/2, bar2.get_height() + 0.05,
                f"{ac_gb[i]:.2f}", ha='center', va='bottom', fontweight='bold')
        
        # Add percentage labels
        plt.text(x[i], min(bar1.get_height(), bar2.get_height())/2,
                percent_reduction_labels[i], ha='center', va='center', fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.3'))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved memory comparison plot to {save_path}")

def plot_time_comparison(model_type: str, batch_sizes: List[int], baseline_times: List[float],
                        ac_times: List[float], save_path: str):
    """
    Plot execution time comparison between with and without activation checkpointing.
    
    Args:
        model_type: Type of model ('resnet' or 'transformer')
        batch_sizes: List of batch sizes
        baseline_times: List of baseline time values (without AC)
        ac_times: List of time values with AC
        save_path: Path to save the plot
    """
    plt.figure(figsize=(12, 8))
    
    x = np.arange(len(batch_sizes))
    width = 0.35
    
    # Convert ms to s for clearer display
    baseline_s = [time / 1000 for time in baseline_times]
    ac_s = [time / 1000 for time in ac_times]
    
    # Calculate percentage increase
    percent_increase = [(ac_times[i] - baseline_times[i]) / baseline_times[i] * 100
                        for i in range(len(batch_sizes))]
    percent_increase_labels = [f"+{p:.1f}%" for p in percent_increase]
    
    bars1 = plt.bar(x - width/2, baseline_s, width, label='Without AC', color='#1F77B4')
    bars2 = plt.bar(x + width/2, ac_s, width, label='With AC', color='#FF7F0E')
    
    plt.xlabel('Batch Size', fontweight='bold')
    plt.ylabel('Execution Time (s)', fontweight='bold')
    title_model = "ResNet-152" if model_type == "resnet" else "Transformer"
    plt.title(f'Execution Time vs. Batch Size - {title_model}', fontweight='bold', fontsize=18)
    plt.xticks(x, batch_sizes, fontweight='bold')
    plt.legend(fontsize=12)
    
    # Add value labels on top of the bars
    for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
        plt.text(bar1.get_x() + bar1.get_width()/2, bar1.get_height() + 0.01,
                f"{baseline_s[i]:.3f}s", ha='center', va='bottom', fontweight='bold')
        plt.text(bar2.get_x() + bar2.get_width()/2, bar2.get_height() + 0.01,
                f"{ac_s[i]:.3f}s", ha='center', va='bottom', fontweight='bold')
        
        # Add percentage labels
        plt.text(x[i], max(bar1.get_height(), bar2.get_height())/2,
                percent_increase_labels[i], ha='center', va='center', fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.3'))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved time comparison plot to {save_path}")

def run_test_for_batch_size(model_type: str, batch_size: int) -> Tuple[float, float, float, float]:
    """
    Run activation checkpointing test for a specific model type and batch size.
    
    Args:
        model_type: Type of model ('resnet' or 'transformer')
        batch_size: Batch size to test
        
    Returns:
        Tuple of (baseline_memory, ac_memory, baseline_time, ac_time)
    """
    logger.info(f"Running test for {model_type} with batch size {batch_size}")
    
    # Get CSV paths
    ac_decisions_path, activation_stats_path, node_stats_path = get_csv_paths(model_type, batch_size)
    
    # Check if files exist
    for path in [ac_decisions_path, activation_stats_path, node_stats_path]:
        if not os.path.exists(path):
            logger.error(f"File not found: {path}")
            return 0, 0, 0, 0
    
    # Load data
    ac_decisions, activation_liveness, node_stats = load_data(
        ac_decisions_path, activation_stats_path, node_stats_path
    )
    
    # Create model and input tensor
    model, input_tensor = create_model(model_type, batch_size)
    
    # Measure baseline memory and time
    logger.info("Measuring baseline memory and time")
    baseline_memory, baseline_time = measure_memory_and_time(model, input_tensor)
    logger.info(f"Baseline peak memory: {baseline_memory:.2f} MB")
    logger.info(f"Baseline average time: {baseline_time:.2f} ms")
    
    # Apply activation checkpointing
    logger.info("Applying activation checkpointing")
    checkpointed_model = apply_activation_checkpointing(model, ac_decisions, activation_liveness)
    
    # Measure memory and time with activation checkpointing
    logger.info("Measuring memory and time with activation checkpointing")
    ac_memory, ac_time = measure_memory_and_time(checkpointed_model, input_tensor)
    logger.info(f"Activation checkpointing peak memory: {ac_memory:.2f} MB")
    logger.info(f"Activation checkpointing average time: {ac_time:.2f} ms")
    
    # Calculate memory reduction and time overhead
    memory_reduction = (baseline_memory - ac_memory) / baseline_memory * 100
    time_overhead = (ac_time - baseline_time) / baseline_time * 100
    
    logger.info(f"Memory reduction: {memory_reduction:.2f}%")
    logger.info(f"Time overhead: {time_overhead:.2f}%")
    
    return baseline_memory, ac_memory, baseline_time, ac_time

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run activation checkpointing tests')
    parser.add_argument('--model', type=str, choices=['resnet', 'transformer', 'both'], default='both',
                        help='Model type to test (default: both)')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Specific batch size to test (default: test all batch sizes)')
    return parser.parse_args()

def main():
    """
    Main function to run the test.
    """
    args = parse_args()
    
    logger.info("Running activation checkpointing test using PyTorch's built-in mechanism")
    
    # Define batch sizes for each model type
    batch_sizes = {
        'resnet': [4, 8, 16, 32, 64],
        'transformer': [2, 4, 8, 16, 32, 64, 128, 256]
    }
    
    # Create final_images directory if it doesn't exist
    os.makedirs("../final_images", exist_ok=True)
    
    # Determine which models to test
    models_to_test = []
    if args.model == 'both':
        models_to_test = ['resnet', 'transformer']
    else:
        models_to_test = [args.model]
    
    # Run tests for each model type
    for model_type in models_to_test:
        # Determine which batch sizes to test
        if args.batch_size is not None:
            batch_sizes_to_test = [args.batch_size]
        else:
            batch_sizes_to_test = batch_sizes[model_type]
        
        # Lists to store results
        baseline_memories = []
        ac_memories = []
        baseline_times = []
        ac_times = []
        valid_batch_sizes = []
        
        # Run tests for each batch size
        for bs in batch_sizes_to_test:
            try:
                baseline_memory, ac_memory, baseline_time, ac_time = run_test_for_batch_size(model_type, bs)
                
                # Only include valid results
                if baseline_memory > 0 and ac_memory > 0:
                    baseline_memories.append(baseline_memory)
                    ac_memories.append(ac_memory)
                    baseline_times.append(baseline_time)
                    ac_times.append(ac_time)
                    valid_batch_sizes.append(bs)
            except Exception as e:
                logger.error(f"Error testing {model_type} with batch size {bs}: {e}")
        
        # Generate plots if we have valid results
        if valid_batch_sizes:
            # Plot memory comparison
            memory_plot_path = f"../final_images/{model_type}_memory_comparison.png"
            plot_memory_comparison(model_type, valid_batch_sizes, baseline_memories, ac_memories, memory_plot_path)
            
            # Plot time comparison
            time_plot_path = f"../final_images/{model_type}_time_comparison.png"
            plot_time_comparison(model_type, valid_batch_sizes, baseline_times, ac_times, time_plot_path)
            
            # Print summary table
            print(f"\n=== {model_type.upper()} ACTIVATION CHECKPOINTING SUMMARY ===")
            print(f"{'Batch Size':<10} {'Baseline Memory (MB)':<20} {'AC Memory (MB)':<20} {'Memory Reduction (%)':<20} {'Baseline Time (ms)':<20} {'AC Time (ms)':<20} {'Time Overhead (%)':<20}")
            print("-" * 120)
            
            for i, bs in enumerate(valid_batch_sizes):
                memory_reduction = (baseline_memories[i] - ac_memories[i]) / baseline_memories[i] * 100
                time_overhead = (ac_times[i] - baseline_times[i]) / baseline_times[i] * 100
                
                print(f"{bs:<10} {baseline_memories[i]:<20.2f} {ac_memories[i]:<20.2f} {memory_reduction:<20.2f} {baseline_times[i]:<20.2f} {ac_times[i]:<20.2f} {time_overhead:<20.2f}")
            
            print("=" * 120)

if __name__ == "__main__":
    main()