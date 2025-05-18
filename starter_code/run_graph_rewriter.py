"""
Test script for the graph rewriter implementation with ResNet-152.

This script loads the activation checkpointing decisions and profiling data,
applies the activation checkpointing to a ResNet-152 model, and compares
the memory usage and performance with and without activation checkpointing.
"""

import os
import time
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models
import logging
from graph_rewriter import (
    extract_recomputation_subgraphs,
    rewrite_graph_with_recomputation,
    apply_rewritten_graph,
    trace_model_for_ac
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"logs/graph_rewriter_test_{time.strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_data(ac_decisions_path, activation_stats_path, node_stats_path):
    """
    Load the activation checkpointing decisions and profiling data.
    
    Args:
        ac_decisions_path: Path to the activation checkpointing decisions CSV
        activation_stats_path: Path to the activation statistics CSV
        node_stats_path: Path to the node statistics CSV
        
    Returns:
        Tuple of (ac_decisions, activation_liveness, node_stats)
    """
    logger.info(f"Loading data from {ac_decisions_path}, {activation_stats_path}, {node_stats_path}")
    
    # Load activation checkpointing decisions
    ac_decisions_df = pd.read_csv(ac_decisions_path)
    ac_decisions = {}
    for _, row in ac_decisions_df.iterrows():
        ac_decisions[row['activation_name']] = row['decision']
    
    # Load activation statistics
    activation_stats_df = pd.read_csv(activation_stats_path)
    activation_liveness = {}
    for _, row in activation_stats_df.iterrows():
        activation_liveness[row['activation_name']] = {
            'creation_rank': row['creation_rank'],
            'last_fw_use_rank': row['last_fw_use_rank'],
            'first_bw_use_rank': row['first_bw_use_rank'],
            'last_bw_use_rank': row['last_bw_use_rank'],
            'median_mem_size_bytes': row['median_mem_size_bytes'],
            'recomp_time_s': row['recomp_time_s'],
            'recomp_memory_bytes': row['recomp_memory_bytes']
        }
    
    # Load node statistics
    node_stats_df = pd.read_csv(node_stats_path)
    node_stats = {}
    for _, row in node_stats_df.iterrows():
        node_stats[row['node_name']] = dict(row)
    
    logger.info(f"Loaded {len(ac_decisions)} activation checkpointing decisions")
    logger.info(f"Loaded {len(activation_liveness)} activation liveness records")
    logger.info(f"Loaded {len(node_stats)} node statistics")
    
    return ac_decisions, activation_liveness, node_stats

def measure_memory_and_time(model, input_tensor, num_iterations=10, include_backward=True):
    """
    Measure the peak memory usage and execution time of a model.
    
    Args:
        model: The model to measure
        input_tensor: The input tensor to use
        num_iterations: Number of iterations to run
        include_backward: Whether to include backward pass in measurement
        
    Returns:
        Tuple of (peak_memory, avg_time)
    """
    # Create optimizer to match graph_prof.py behavior
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Warm-up
    for _ in range(3):
        optimizer.zero_grad()
        output = model(input_tensor)
        if include_backward:
            # Use sum() to create a scalar for backward
            loss = output.sum()
            loss.backward()
            optimizer.step()
    
    # Reset peak memory stats
    torch.cuda.reset_peak_memory_stats()
    
    # Measure time and memory
    start_time = time.time()
    for _ in range(num_iterations):
        optimizer.zero_grad()
        output = model(input_tensor)
        if include_backward:
            # Use sum() to create a scalar for backward
            loss = output.sum()
            loss.backward()
            optimizer.step()
    end_time = time.time()
    
    # Get peak memory
    peak_memory = torch.cuda.max_memory_allocated()
    avg_time = (end_time - start_time) / num_iterations
    
    return peak_memory, avg_time

def analyze_memory_breakdown(model, peak_memory):
    """
    Analyze the memory breakdown of a model.
    
    Args:
        model: The model to analyze
        peak_memory: The peak memory usage in bytes
        
    Returns:
        Dictionary with memory breakdown
    """
    # Calculate parameter memory
    param_memory = sum(p.element_size() * p.nelement() for p in model.parameters())
    
    # Estimate gradient memory (similar size to parameters)
    grad_memory = param_memory
    
    # Estimate optimizer state memory (2x parameters for Adam)
    optimizer_memory = 2 * param_memory
    
    # Estimate activation memory (remainder, with some adjustment for fragmentation)
    fragmentation_factor = 1.1  # 10% fragmentation estimate
    activation_memory = (peak_memory - param_memory - grad_memory - optimizer_memory) / fragmentation_factor
    
    # Ensure activation memory is not negative
    activation_memory = max(0, activation_memory)
    
    # Print breakdown
    logger.info("\n[Memory Breakdown]")
    logger.info(f"Parameters:      {param_memory / (1024**2):.2f} MB")
    logger.info(f"Gradients:       {grad_memory / (1024**2):.2f} MB")
    logger.info(f"Optimizer state: {optimizer_memory / (1024**2):.2f} MB")
    logger.info(f"Activations:     {activation_memory / (1024**2):.2f} MB")
    logger.info(f"Total (with fragmentation): {peak_memory / (1024**2):.2f} MB")
    
    return {
        "parameters": param_memory,
        "gradients": grad_memory,
        "optimizer_state": optimizer_memory,
        "activations": activation_memory,
        "total": peak_memory
    }

def main():
    # Paths to input files
    ac_decisions_path = "../reports/ac_decisions_resnet_bs32.csv"
    activation_stats_path = "../reports/profiler_stats_resnet_bs32_activation_stats.csv"
    node_stats_path = "../reports/profiler_stats_resnet_bs32_node_stats.csv"
    
    # Check if files exist
    for path in [ac_decisions_path, activation_stats_path, node_stats_path]:
        if not os.path.exists(path):
            logger.error(f"File not found: {path}")
            return
    
    # Load data
    ac_decisions, activation_liveness, node_stats = load_data(
        ac_decisions_path, activation_stats_path, node_stats_path
    )
    
    # Create ResNet-152 model
    logger.info("Creating ResNet-152 model")
    model = models.resnet152(pretrained=False)
    model = model.cuda()
    model.eval()  # Set to evaluation mode
    
    # Create example input
    batch_size = 32  # Using batch size 4 as specified
    input_tensor = torch.randn(batch_size, 3, 224, 224, device='cuda')
    
    # Measure baseline memory and time
    logger.info("Measuring baseline memory and time")
    baseline_memory, baseline_time = measure_memory_and_time(model, input_tensor, include_backward=True)
    logger.info(f"Baseline peak memory: {baseline_memory / (1024**2):.2f} MB")
    logger.info(f"Baseline average time: {baseline_time * 1000:.2f} ms")
    
    # Analyze memory breakdown
    baseline_memory_breakdown = analyze_memory_breakdown(model, baseline_memory)
    
    # Trace the model
    logger.info("Tracing the model")
    try:
        traced_model = trace_model_for_ac(model, input_tensor, activation_liveness)
        logger.info("Model tracing successful")
    except Exception as e:
        logger.error(f"Error tracing model: {e}")
        return
    
    # Extract recomputation subgraphs
    logger.info("Extracting recomputation subgraphs")
    try:
        subgraphs = extract_recomputation_subgraphs(
            traced_model.graph, ac_decisions, activation_liveness
        )
        logger.info(f"Extracted {len(subgraphs)} recomputation subgraphs")
    except Exception as e:
        logger.error(f"Error extracting subgraphs: {e}")
        return
    
    # Rewrite the graph with recomputation
    logger.info("Rewriting graph with recomputation")
    try:
        rewritten_graph = rewrite_graph_with_recomputation(
            traced_model.graph, subgraphs, activation_liveness
        )
        logger.info("Graph rewriting successful")
    except Exception as e:
        logger.error(f"Error rewriting graph: {e}")
        return
    
    # Apply the rewritten graph to the model
    logger.info("Applying rewritten graph to model")
    try:
        rewritten_model = apply_rewritten_graph(
            model, traced_model.graph, rewritten_graph
        )
        logger.info("Applied rewritten graph to model")
    except Exception as e:
        logger.error(f"Error applying rewritten graph: {e}")
        return
    
    # Measure memory and time with activation checkpointing
    logger.info("Measuring memory and time with activation checkpointing")
    ac_memory, ac_time = measure_memory_and_time(rewritten_model, input_tensor, include_backward=True)
    logger.info(f"Activation checkpointing peak memory: {ac_memory / (1024**2):.2f} MB")
    logger.info(f"Activation checkpointing average time: {ac_time * 1000:.2f} ms")
    
    # Analyze memory breakdown with activation checkpointing
    ac_memory_breakdown = analyze_memory_breakdown(rewritten_model, ac_memory)
    
    # Calculate and report improvements
    memory_reduction = (baseline_memory - ac_memory) / baseline_memory * 100
    time_overhead = (ac_time - baseline_time) / baseline_time * 100
    
    logger.info(f"Memory reduction: {memory_reduction:.2f}%")
    logger.info(f"Time overhead: {time_overhead:.2f}%")
    
    print("\n=== Activation Checkpointing Results ===")
    print(f"Baseline peak memory: {baseline_memory / (1024**2):.2f} MB")
    print(f"AC peak memory: {ac_memory / (1024**2):.2f} MB")
    print(f"Memory reduction: {memory_reduction:.2f}%")
    print(f"Baseline average time: {baseline_time * 1000:.2f} ms")
    print(f"AC average time: {ac_time * 1000:.2f} ms")
    print(f"Time overhead: {time_overhead:.2f}%")
    print("========================================")
    
    print("\n=== Memory Breakdown (Baseline) ===")
    print(f"Parameters:      {baseline_memory_breakdown['parameters'] / (1024**2):.2f} MB")
    print(f"Gradients:       {baseline_memory_breakdown['gradients'] / (1024**2):.2f} MB")
    print(f"Optimizer state: {baseline_memory_breakdown['optimizer_state'] / (1024**2):.2f} MB")
    print(f"Activations:     {baseline_memory_breakdown['activations'] / (1024**2):.2f} MB")
    print(f"Total:           {baseline_memory_breakdown['total'] / (1024**2):.2f} MB")
    print("========================================")
    
    print("\n=== Memory Breakdown (AC) ===")
    print(f"Parameters:      {ac_memory_breakdown['parameters'] / (1024**2):.2f} MB")
    print(f"Gradients:       {ac_memory_breakdown['gradients'] / (1024**2):.2f} MB")
    print(f"Optimizer state: {ac_memory_breakdown['optimizer_state'] / (1024**2):.2f} MB")
    print(f"Activations:     {ac_memory_breakdown['activations'] / (1024**2):.2f} MB")
    print(f"Total:           {ac_memory_breakdown['total'] / (1024**2):.2f} MB")
    print("========================================")

if __name__ == "__main__":
    main()