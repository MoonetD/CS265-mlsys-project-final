#!/usr/bin/env python
"""
Activation Checkpointing Comparison Script

This script compares the performance of ResNet-152 with and without activation checkpointing:
1. Runs ResNet-152 with different batch sizes (4, 8, 16, 32)
2. For each batch size, measures:
   - Peak memory usage without AC
   - Peak memory usage with AC (using our algorithm's decisions)
   - Iteration latency without AC
   - Iteration latency with AC
3. Generates comparison charts:
   - Bar chart: Peak memory vs. batch size (with and without AC)
   - Line chart: Iteration latency vs. batch size (with and without AC)
4. Validates that AC preserves model correctness by comparing loss and gradients

To run this script:
    conda run -n ml_env python starter_code/ac_comparison.py
"""

import os
import time
import torch
import torch.nn as nn
import torch.fx as fx
import torchvision.models as models
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from functools import wraps
from collections import defaultdict
import argparse
import copy

from graph_prof import GraphProfiler
from graph_tracer import SEPFunction, compile
from activation_checkpointing import ActivationCheckpointingAlgorithm
from graph_rewriter import (
    extract_recomputation_subgraphs,
    rewrite_graph_with_recomputation,
    trace_model_for_ac,
    apply_rewritten_graph
)

def ensure_reports_directory():
    """Ensure the reports directory exists."""
    # TODO: Implement directory creation
    # - Create reports directory if it doesn't exist
    # - Return the path to the reports directory
    pass

def checkpoint_wrapper(function):
    """
    A wrapper that applies torch.utils.checkpoint to a function.
    This is used to implement activation checkpointing.
    
    The key mechanism is that during the forward pass, activations are NOT saved.
    Instead, only the inputs are saved, and the activations are recomputed during
    the backward pass when they are needed for gradient computation.
    
    This trades increased computation time for reduced memory usage.
    
    Args:
        function: The function to wrap with checkpointing
        
    Returns:
        A wrapped function that uses checkpointing
    """
    # TODO: Implement checkpoint wrapper
    # - Create a wrapped function that applies torch.utils.checkpoint
    # - During forward pass, run the function with no_grad to avoid storing activations
    # - During backward pass, recompute the activations
    # - Use PyTorch's checkpoint mechanism with a custom recompute function
    pass

def measure_memory_and_time(model, input_batch, num_runs=3, debug=False):
    """
    Measure the peak memory usage and average execution time of a model.
    
    This function performs multiple steps to ensure accurate measurements:
    1. Multiple warm-up runs to stabilize GPU caches and JIT compilation
    2. Explicit CUDA synchronization to ensure all operations complete
    3. Multiple measurement runs to get stable averages
    4. Proper memory tracking with reset between runs
    5. Includes backward pass to properly measure activation checkpointing effects
    
    Args:
        model: The model to measure
        input_batch: The input batch
        num_runs: Number of runs to average over
        debug: Whether to print debug information
        
    Returns:
        peak_memory: Peak memory usage in bytes
        avg_time: Average execution time in seconds
    """
    # TODO: Implement memory and time measurement
    # - Perform warm-up runs to stabilize GPU caches
    # - Clear cache and reset memory stats
    # - Measure time and memory over multiple runs
    # - Include backward pass to properly measure activation checkpointing
    # - Return peak memory and average time
    pass

def apply_activation_checkpointing(model, ac_decisions=None, percentage=0.5, activation_liveness=None, debug=False):
    """
    Apply activation checkpointing to a model.
    
    Args:
        model: The model to apply checkpointing to
        ac_decisions: Dictionary mapping activation names to 'RETAINED' or 'RECOMPUTE'
        percentage: Percentage of bottleneck blocks to apply checkpointing to if ac_decisions is None
        activation_liveness: Dictionary with activation liveness information
        debug: Whether to print debug information
        
    Returns:
        model_with_ac: The model with activation checkpointing applied
    """
    # TODO: Implement activation checkpointing application
    # - Create a deep copy of the model
    # - Try to use the graph rewriter approach if we have AC decisions and activation liveness info
    #   - Trace the model to get an FX graph
    #   - Extract subgraphs for recomputation
    #   - Rewrite the graph with recomputation
    #   - Apply the rewritten graph to the model
    # - Fall back to bottleneck checkpointing approach if graph rewriter fails
    #   - Apply checkpointing to bottleneck blocks
    # - Return the model with activation checkpointing applied
    pass

def validate_correctness(model, model_with_ac, input_batch, rtol=1e-2, atol=1e-2, debug=False):
    """
    Validate that activation checkpointing preserves model correctness.
    
    Args:
        model: The original model
        model_with_ac: The model with activation checkpointing
        input_batch: The input batch
        rtol: Relative tolerance for comparing outputs
        atol: Absolute tolerance for comparing outputs
        debug: Whether to print debug information
        
    Returns:
        is_valid: Whether the model is correct within tolerance
        rel_diff: Relative difference between outputs
    """
    # TODO: Implement correctness validation
    # - Set both models to eval mode
    # - Perform forward pass without AC
    # - Perform forward pass with AC
    # - Compare outputs using torch.allclose
    # - Calculate relative difference
    # - Return validation result and relative difference
    pass

def profile_batch_size(batch_size, device_str='cuda:0', memory_budget_gb=None, debug=False, timeout=120):
    """
    Profile ResNet-152 with a specific batch size, with and without activation checkpointing.
    
    Args:
        batch_size: The batch size to use for profiling
        device_str: The device to use for profiling
        memory_budget_gb: Memory budget for activation checkpointing in GB
        debug: Whether to print debug information
        
    Returns:
        results: Dictionary containing profiling results
    """
    # TODO: Implement batch size profiling
    # - Create ResNet-152 model
    # - Create a random batch of data
    # - Measure memory and time without activation checkpointing
    # - Set memory budget based on peak memory without AC
    # - Find and load appropriate CSV files with profiling data
    # - Run the activation checkpointing algorithm
    # - Apply activation checkpointing to the model
    # - Measure memory and time with activation checkpointing
    # - Validate correctness
    # - Calculate memory reduction and time overhead
    # - Return results
    pass

def create_comparison_charts(results, reports_dir, debug=False):
    """
    Create comparison charts for peak memory and iteration latency.
    
    Args:
        results: List of dictionaries containing profiling results
        reports_dir: Directory to save the charts
        debug: Whether to print debug information
    """
    # TODO: Implement chart creation
    # - Extract data from results
    # - Create bar chart for peak memory vs. batch size
    # - Add value labels and reduction percentages
    # - Create line chart for iteration latency vs. batch size
    # - Add value labels and overhead percentages
    # - Save charts to reports directory
    pass

def main():
    """Main function to run the activation checkpointing comparison."""
    # TODO: Implement main function
    # - Parse command line arguments
    # - Set random seed for reproducibility
    # - Use CUDA if available
    # - Ensure reports directory exists
    # - Profile each batch size
    # - Create comparison charts
    # - Print summary table
    pass

if __name__ == "__main__":
    main()