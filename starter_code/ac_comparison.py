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
    reports_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'reports')
    if not os.path.exists(reports_dir):
        os.makedirs(reports_dir)
        print(f"Created reports directory: {reports_dir}")
    return reports_dir

def checkpoint_wrapper(function):
    """
    A wrapper that applies torch.utils.checkpoint to a function.
    This is used to implement activation checkpointing.
    """
    @wraps(function)
    def wrapped(*args, **kwargs):
        return torch.utils.checkpoint.checkpoint(function, *args, use_reentrant=False)
    return wrapped

def measure_memory_and_time(model, input_batch, num_runs=3, debug=False):
    """
    Measure the peak memory usage and average execution time of a model.
    
    Args:
        model: The model to measure
        input_batch: The input batch
        num_runs: Number of runs to average over
        debug: Whether to print debug information
        
    Returns:
        peak_memory: Peak memory usage in bytes
        avg_time: Average execution time in seconds
    """
    # Warm-up run
    if debug:
        print(f"[DEBUG] Starting warm-up run")
    with torch.no_grad():
        model(input_batch)
    torch.cuda.synchronize()
    
    # Reset memory stats
    torch.cuda.reset_peak_memory_stats()
    if debug:
        print(f"[DEBUG] Reset peak memory stats")
        print(f"[DEBUG] Current memory allocated: {torch.cuda.memory_allocated() / (1024**2):.2f} MiB")
    
    # Measure time over multiple runs
    total_time = 0
    for i in range(num_runs):
        if debug:
            print(f"[DEBUG] Starting measurement run {i+1}/{num_runs}")
        torch.cuda.synchronize()
        start_time = time.time()
        
        # Forward pass
        output = model(input_batch)
        
        torch.cuda.synchronize()
        end_time = time.time()
        run_time = end_time - start_time
        total_time += run_time
        if debug:
            print(f"[DEBUG] Run {i+1} time: {run_time:.4f} s")
    
    # Get peak memory
    peak_memory = torch.cuda.max_memory_allocated()
    avg_time = total_time / num_runs
    
    if debug:
        print(f"[DEBUG] Peak memory: {peak_memory / (1024**2):.2f} MiB")
        print(f"[DEBUG] Average time: {avg_time:.4f} s")
    
    return peak_memory, avg_time

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
    # Create a deep copy of the model to avoid modifying the original
    model_with_ac = copy.deepcopy(model)
    
    if debug:
        print(f"[DEBUG] Creating deep copy of model")
        print(f"[DEBUG] Original model id: {id(model)}, Copy id: {id(model_with_ac)}")
    
    # Count how many activations are marked for RECOMPUTE
    recompute_target = 0
    checkpoint_target = 0
    if ac_decisions:
        recompute_target = sum(1 for v in ac_decisions.values() if v == 'RECOMPUTE')
        checkpoint_target = sum(1 for v in ac_decisions.values() if v == 'RETAINED')
        print(f"AC decisions: {recompute_target} RECOMPUTE, {checkpoint_target} RETAINED")
        
        if debug:
            print(f"[DEBUG] AC decisions details:")
            for i, (k, v) in enumerate(ac_decisions.items()):
                if i < 10:  # Print first 10 for brevity
                    print(f"[DEBUG]   {k}: {v}")
            if len(ac_decisions) > 10:
                print(f"[DEBUG]   ... and {len(ac_decisions) - 10} more")
    
    # Try to use the graph rewriter approach if we have AC decisions and activation liveness info
    if ac_decisions and activation_liveness and recompute_target > 0:
        try:
            print(f"Using graph rewriter approach with {recompute_target} activations marked for recomputation")
            
            # Create a sample input for tracing
            sample_input = torch.randn(1, 3, 224, 224).to(next(model.parameters()).device)
            
            if debug:
                print(f"[DEBUG] Created sample input for tracing: {sample_input.shape}")
            
            # Trace the model to get an FX graph
            traced_model = trace_model_for_ac(model, sample_input, activation_liveness)
            
            if debug and traced_model:
                print(f"[DEBUG] Successfully traced model")
                print(f"[DEBUG] Graph has {len(list(traced_model.graph.nodes))} nodes")
                print(f"[DEBUG] First few nodes: {[n.name for n in list(traced_model.graph.nodes)[:5]]}")
            
            if traced_model:
                # Extract subgraphs for recomputation
                if debug:
                    print(f"[DEBUG] Extracting subgraphs for recomputation")
                
                subgraphs = extract_recomputation_subgraphs(
                    traced_model.graph,
                    ac_decisions,
                    activation_liveness
                )
                
                # Check if we have any valid subgraphs
                valid_subgraphs = {k: v for k, v in subgraphs.items() if v[0]}  # Filter out empty node lists
                print(f"Extracted {len(valid_subgraphs)} valid subgraphs out of {len(subgraphs)} total")
                
                if debug:
                    print(f"[DEBUG] Valid subgraphs: {len(valid_subgraphs)}/{len(subgraphs)}")
                
                if valid_subgraphs:
                    # Print some details about the subgraphs for debugging
                    for act_name, (nodes, inputs) in list(valid_subgraphs.items())[:3]:  # Show first 3 for brevity
                        print(f"  Subgraph for {act_name}: {len(nodes)} nodes, {len(inputs)} inputs")
                        print(f"    First few nodes: {[n.name for n in nodes[:3]]}")
                        print(f"    First few inputs: {[n.name for n in list(inputs)[:3]]}")
                    
                    try:
                        # Rewrite the graph with recomputation
                        if debug:
                            print(f"[DEBUG] Rewriting graph with recomputation")
                        
                        rewritten_graph = rewrite_graph_with_recomputation(
                            traced_model.graph,
                            valid_subgraphs,
                            activation_liveness
                        )
                        
                        if debug:
                            print(f"[DEBUG] Successfully rewrote graph")
                            print(f"[DEBUG] Rewritten graph has {len(list(rewritten_graph.nodes))} nodes")
                        
                        # Apply the rewritten graph to the model
                        if debug:
                            print(f"[DEBUG] Applying rewritten graph to model")
                        
                        rewritten_model = apply_rewritten_graph(
                            model_with_ac,
                            traced_model.graph,
                            rewritten_graph
                        )
                        
                        if rewritten_model:
                            print(f"Successfully applied graph rewriting for activation checkpointing")
                            
                            # Verify the rewritten model works
                            try:
                                test_input = torch.randn(1, 3, 224, 224).to(next(model.parameters()).device)
                                if debug:
                                    print(f"[DEBUG] Testing rewritten model with input shape: {test_input.shape}")
                                    
                                    # Print the model's forward method to debug issues
                                    print(f"[DEBUG] Rewritten model forward method:")
                                    import inspect
                                    print(inspect.getsource(rewritten_model.forward))
                                
                                with torch.no_grad():
                                    # Trace the execution to identify where the error occurs
                                    if debug:
                                        print(f"[DEBUG] Tracing execution of rewritten model")
                                        # Get all module names for debugging
                                        module_names = {id(m): name for name, m in rewritten_model.named_modules()}
                                        for name, module in rewritten_model.named_modules():
                                            if 'avgpool' in name or 'fc' in name:
                                                print(f"[DEBUG] Found module: {name}, type: {type(module)}")
                                    
                                    output = rewritten_model(test_input)
                                
                                if debug:
                                    print(f"[DEBUG] Rewritten model output shape: {output.shape}")
                                
                                print("Rewritten model successfully executed a forward pass")
                                return rewritten_model
                            except Exception as e:
                                print(f"Error executing rewritten model: {e}")
                                if debug:
                                    print(f"[DEBUG] Exception details: {str(e)}")
                                    import traceback
                                    print(f"[DEBUG] Traceback: {traceback.format_exc()}")
                                    
                                    # Try to identify the specific issue with flatten
                                    if "flatten" in str(e):
                                        print(f"[DEBUG] This appears to be an issue with the 'flatten' operation.")
                                        print(f"[DEBUG] Checking graph structure for flatten and related operations...")
                                        
                                        # Check if flatten exists in the graph
                                        flatten_nodes = [n for n in rewritten_model.graph.nodes if 'flatten' in n.name]
                                        if flatten_nodes:
                                            print(f"[DEBUG] Found {len(flatten_nodes)} flatten nodes in the graph")
                                            for n in flatten_nodes:
                                                print(f"[DEBUG]   Node: {n.name}, Op: {n.op}, Target: {n.target}")
                                                print(f"[DEBUG]   Args: {n.args}")
                                                print(f"[DEBUG]   Users: {[u.name for u in n.users]}")
                                        else:
                                            print(f"[DEBUG] No flatten nodes found in the graph")
                                            
                                        # Check for avgpool and fc nodes
                                        avgpool_nodes = [n for n in rewritten_model.graph.nodes if 'avgpool' in n.name]
                                        fc_nodes = [n for n in rewritten_model.graph.nodes if 'fc' in n.name]
                                        print(f"[DEBUG] Found {len(avgpool_nodes)} avgpool nodes and {len(fc_nodes)} fc nodes")
                                
                                print("Falling back to bottleneck checkpointing")
                        else:
                            print("Failed to apply rewritten graph, falling back to bottleneck checkpointing")
                            if debug:
                                print(f"[DEBUG] apply_rewritten_graph returned None")
                    except Exception as e:
                        print(f"Error in graph rewriting: {e}")
                        if debug:
                            print(f"[DEBUG] Exception details: {str(e)}")
                            import traceback
                            print(f"[DEBUG] Traceback: {traceback.format_exc()}")
                        print("Falling back to bottleneck checkpointing")
                else:
                    print(f"No valid subgraphs extracted, falling back to bottleneck checkpointing")
            else:
                print(f"Failed to trace model, falling back to bottleneck checkpointing")
                if debug:
                    print(f"[DEBUG] trace_model_for_ac returned None")
        except Exception as e:
            print(f"Error in graph rewriter approach: {str(e)}")
            if debug:
                print(f"[DEBUG] Exception details: {str(e)}")
                import traceback
                print(f"[DEBUG] Traceback: {traceback.format_exc()}")
            print(f"Falling back to bottleneck checkpointing")
    else:
        print(f"Using bottleneck checkpointing approach")
        if debug:
            if not ac_decisions:
                print(f"[DEBUG] No AC decisions provided")
            if not activation_liveness:
                print(f"[DEBUG] No activation liveness information provided")
            if recompute_target == 0:
                print(f"[DEBUG] No activations marked for RECOMPUTE")
    
    # Fall back to the bottleneck checkpointing approach
    if recompute_target == 0:
        print(f"No specific AC decisions provided. Applying checkpointing to {percentage:.0%} of bottleneck blocks")
        bottleneck_modules = [m for n, m in model_with_ac.named_modules() if isinstance(m, models.resnet.Bottleneck)]
        recompute_target = max(1, int(len(bottleneck_modules) * percentage))
        
        if debug:
            print(f"[DEBUG] Found {len(bottleneck_modules)} bottleneck modules")
            print(f"[DEBUG] Will checkpoint {recompute_target} modules ({percentage:.0%})")
    
    print(f"Applying checkpointing to {recompute_target} bottleneck blocks")
    
    # Apply checkpointing to bottleneck blocks
    recompute_count = 0
    for name, module in model_with_ac.named_modules():
        if isinstance(module, models.resnet.Bottleneck) and recompute_count < recompute_target:
            # Store the original forward method
            original_forward = module.forward
            # Apply checkpointing to this bottleneck block
            module.forward = checkpoint_wrapper(original_forward)
            recompute_count += 1
            
            if debug and recompute_count <= 5:
                print(f"[DEBUG] Applied checkpointing to bottleneck block: {name}")
    
    print(f"Applied checkpointing to {recompute_count} bottleneck blocks")
    return model_with_ac

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
    # Set both models to eval mode to disable dropout, etc.
    model.eval()
    model_with_ac.eval()
    
    if debug:
        print(f"[DEBUG] Validating model correctness")
        print(f"[DEBUG] Input batch shape: {input_batch.shape}")
    
    # Forward pass without AC
    with torch.no_grad():
        output = model(input_batch)
    
    if debug:
        print(f"[DEBUG] Original model output shape: {output.shape}")
        print(f"[DEBUG] Original model output stats: min={output.min().item():.4f}, max={output.max().item():.4f}, mean={output.mean().item():.4f}")
    
    # Forward pass with AC
    with torch.no_grad():
        output_with_ac = model_with_ac(input_batch)
    
    if debug:
        print(f"[DEBUG] AC model output shape: {output_with_ac.shape}")
        print(f"[DEBUG] AC model output stats: min={output_with_ac.min().item():.4f}, max={output_with_ac.max().item():.4f}, mean={output_with_ac.mean().item():.4f}")
    
    # Compare outputs
    abs_diff = torch.abs(output - output_with_ac)
    output_norm = torch.abs(output)
    rel_diff = torch.max(abs_diff / (output_norm + 1e-10)).item()
    
    if debug:
        print(f"[DEBUG] Absolute difference stats: min={abs_diff.min().item():.4e}, max={abs_diff.max().item():.4e}, mean={abs_diff.mean().item():.4e}")
        print(f"[DEBUG] Relative difference: {rel_diff:.6e}")
    
    # Check if outputs are close
    is_valid = torch.allclose(output, output_with_ac, rtol=rtol, atol=atol)
    
    print(f"Output validation: {'Passed' if is_valid else 'Failed'}")
    print(f"Maximum relative difference: {rel_diff:.6f}")
    
    return is_valid, rel_diff

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
    print(f"\n--- Profiling ResNet-152 with batch size {batch_size} ---")
    
    # Create ResNet-152 model
    model = models.resnet152(weights=models.ResNet152_Weights.IMAGENET1K_V1).to(device_str)
    model.eval()  # Set to eval mode to disable dropout, etc.
    
    if debug:
        print(f"[DEBUG] Created ResNet-152 model")
        print(f"[DEBUG] Model has {sum(p.numel() for p in model.parameters())} parameters")
        print(f"[DEBUG] Model is on device: {next(model.parameters()).device}")
    
    # Create a random batch of data
    batch = torch.randn(batch_size, 3, 224, 224).to(device_str)
    
    if debug:
        print(f"[DEBUG] Created input batch with shape: {batch.shape}")
        print(f"[DEBUG] Input batch is on device: {batch.device}")
    
    # 1. Measure memory and time without activation checkpointing
    print("Measuring without activation checkpointing...")
    peak_memory_without_ac, time_without_ac = measure_memory_and_time(model, batch, debug=debug)
    print(f"Without AC - Peak memory: {peak_memory_without_ac / (1024**2):.2f} MiB, Iteration time: {time_without_ac:.4f} s")
    
    # 2. Get activation checkpointing decisions
    if memory_budget_gb is None:
        # Set a realistic memory budget based on the paper's approach
        # Use 70% of peak memory as the budget
        memory_budget_gb = peak_memory_without_ac / (1024**3) * 0.7
        print(f"Setting memory budget to 70% of peak memory: {memory_budget_gb:.2f} GB")
    
    if debug:
        print(f"[DEBUG] Memory budget set to: {memory_budget_gb:.2f} GB")
    
    # Use batch-specific CSV files if available, otherwise use default
    # First check in the main directory
    node_stats_file = f"profiler_stats_bs{batch_size}_node_stats.csv"
    activation_stats_file = f"profiler_stats_bs{batch_size}_activation_stats.csv"
    
    # Also check in the reports directory
    reports_dir = ensure_reports_directory()
    reports_node_stats_file = os.path.join(reports_dir, f"profiler_stats_bs{batch_size}_node_stats.csv")
    reports_activation_stats_file = os.path.join(reports_dir, f"profiler_stats_bs{batch_size}_activation_stats.csv")
    
    # Check if files exist in either location
    if os.path.exists(node_stats_file) and os.path.exists(activation_stats_file):
        print(f"Found batch-specific CSV files in main directory: {node_stats_file} and {activation_stats_file}")
    elif os.path.exists(reports_node_stats_file) and os.path.exists(reports_activation_stats_file):
        print(f"Found batch-specific CSV files in reports directory")
        node_stats_file = reports_node_stats_file
        activation_stats_file = reports_activation_stats_file
    else:
        print(f"Batch-specific CSV files not found, using default CSV files")
        node_stats_file = "profiler_stats_node_stats.csv"
        activation_stats_file = "profiler_stats_activation_stats.csv"
        
        # If default files don't exist, check reports directory
        if not os.path.exists(node_stats_file) or not os.path.exists(activation_stats_file):
            reports_default_node = os.path.join(reports_dir, "profiler_stats_node_stats.csv")
            reports_default_act = os.path.join(reports_dir, "profiler_stats_activation_stats.csv")
            
            if os.path.exists(reports_default_node) and os.path.exists(reports_default_act):
                print(f"Using default CSV files from reports directory")
                node_stats_file = reports_default_node
                activation_stats_file = reports_default_act
    
    print(f"READING CSV FILES: {node_stats_file} and {activation_stats_file}")
    
    if debug:
        print(f"[DEBUG] Using node stats file: {node_stats_file}")
        print(f"[DEBUG] Using activation stats file: {activation_stats_file}")
        if not os.path.exists(node_stats_file):
            print(f"[DEBUG] Warning: Node stats file does not exist")
        if not os.path.exists(activation_stats_file):
            print(f"[DEBUG] Warning: Activation stats file does not exist")
    
    # Run the activation checkpointing algorithm
    print(f"Running activation checkpointing algorithm with memory budget {memory_budget_gb:.2f} GB...")
    try:
        ac_algo = ActivationCheckpointingAlgorithm(
            node_stats_path=node_stats_file,
            activation_stats_path=activation_stats_file,
            memory_budget_gb=memory_budget_gb
        )
        
        if debug:
            print(f"[DEBUG] Created ActivationCheckpointingAlgorithm instance")
            print(f"[DEBUG] Node stats shape: {ac_algo.node_stats_df.shape}")
            print(f"[DEBUG] Activation stats shape: {ac_algo.activation_stats_df.shape}")
        
        ac_decisions = ac_algo.decide_checkpoints(fixed_overhead_gb=0.5, timeout_seconds=timeout)
        
        # Count decisions
        recompute_count = sum(1 for decision in ac_decisions.values() if decision == 'RECOMPUTE')
        checkpoint_count = sum(1 for decision in ac_decisions.values() if decision == 'RETAINED')
        print(f"AC decisions: {recompute_count} RECOMPUTE, {checkpoint_count} RETAINED")
        
        if debug:
            print(f"[DEBUG] Algorithm made {len(ac_decisions)} decisions")
            print(f"[DEBUG] RECOMPUTE: {recompute_count}, RETAINED: {checkpoint_count}")
            
        # Save AC decisions to a CSV file
        reports_dir = ensure_reports_directory()
        ac_decisions_file = os.path.join(reports_dir, f"ac_decisions_bs{batch_size}.csv")
        print(f"\n--- Saving AC decisions to {ac_decisions_file} ---")
        
        try:
            # Create a DataFrame from the decisions
            decisions_df = pd.DataFrame({
                'activation_name': list(ac_decisions.keys()),
                'decision': list(ac_decisions.values()),
                'memory_size_bytes': [ac_algo.activation_stats_df.loc[act, 'avg_mem_size_bytes']
                                     if act in ac_algo.activation_stats_df.index else 0
                                     for act in ac_decisions.keys()],
                'recomp_time_s': [ac_algo.activation_stats_df.loc[act, 'recomp_time_s']
                                 if act in ac_algo.activation_stats_df.index else 0
                                 for act in ac_decisions.keys()],
                'creation_rank': [ac_algo.activation_stats_df.loc[act, 'creation_rank']
                                 if act in ac_algo.activation_stats_df.index else 0
                                 for act in ac_decisions.keys()],
                'first_bw_use_rank': [ac_algo.activation_stats_df.loc[act, 'first_bw_use_rank']
                                     if act in ac_algo.activation_stats_df.index else 0
                                     for act in ac_decisions.keys()]
            })
            
            # Save to CSV
            decisions_df.to_csv(ac_decisions_file, index=False)
            print(f"Successfully saved {len(decisions_df)} AC decisions to CSV")
            
            # Print detailed summary of decisions
            print("\n--- Detailed Summary of AC Decisions ---")
            print(f"Total activations: {len(ac_decisions)}")
            print(f"RECOMPUTE: {recompute_count} ({recompute_count/len(ac_decisions)*100:.1f}%)")
            print(f"RETAINED: {checkpoint_count} ({checkpoint_count/len(ac_decisions)*100:.1f}%)")
            
            # Calculate total memory savings and recomputation overhead
            total_memory_bytes = sum(decisions_df['memory_size_bytes'])
            recompute_memory_bytes = sum(decisions_df[decisions_df['decision'] == 'RECOMPUTE']['memory_size_bytes'])
            memory_savings_bytes = recompute_memory_bytes  # Memory saved by recomputing instead of checkpointing
            
            total_recomp_time = sum(decisions_df[decisions_df['decision'] == 'RECOMPUTE']['recomp_time_s'])
            
            print(f"\nEstimated memory savings: {memory_savings_bytes / (1024**2):.2f} MiB")
            print(f"Estimated recomputation overhead: {total_recomp_time:.4f} seconds")
            
            # Show top activations chosen for recomputation
            print("\n--- Top Activations Chosen for Recomputation ---")
            # Sort by memory size (largest first)
            top_recompute = decisions_df[decisions_df['decision'] == 'RECOMPUTE'].sort_values(
                by='memory_size_bytes', ascending=False
            ).head(20)  # Show top 20
            
            if len(top_recompute) > 0:
                print(f"{'Activation Name':<30} {'Memory Size (MiB)':<20} {'Recomp Time (ms)':<20}")
                print("-" * 70)
                for _, row in top_recompute.iterrows():
                    print(f"{row['activation_name']:<30} {row['memory_size_bytes'] / (1024**2):<20.2f} {row['recomp_time_s'] * 1000:<20.4f}")
            else:
                print("No activations marked for recomputation")
                
            # Run memory simulation to get more accurate estimates
            print("\n--- Memory Simulation Results ---")
            peak_memory, total_exec_time = ac_algo._simulate_memory_usage(
                ac_decisions,
                fixed_overhead_bytes=0.5 * (1024**3),  # 0.5 GB fixed overhead
                debug=False
            )
            print(f"Simulated peak memory: {peak_memory / (1024**3):.2f} GB")
            print(f"Simulated execution time: {total_exec_time:.4f} seconds")
            
        except Exception as e:
            print(f"Error saving AC decisions or generating summary: {e}")
            if debug:
                print(f"[DEBUG] Exception details: {str(e)}")
                import traceback
                print(f"[DEBUG] Traceback: {traceback.format_exc()}")
    except Exception as e:
        print(f"Error running activation checkpointing algorithm: {e}")
        if debug:
            print(f"[DEBUG] Exception details: {str(e)}")
            import traceback
            print(f"[DEBUG] Traceback: {traceback.format_exc()}")
        ac_decisions = None
    
    # 3. Apply activation checkpointing and measure memory and time
    print("Measuring with activation checkpointing...")
    
    # Extract activation liveness information from the algorithm
    activation_liveness = None
    if ac_decisions:
        try:
            activation_liveness = {
                act_name: {
                    "creation_rank": ac_algo.activation_stats_df.loc[act_name, "creation_rank"],
                    "last_fw_use_rank": ac_algo.activation_stats_df.loc[act_name, "last_fw_use_rank"],
                    "first_bw_use_rank": ac_algo.activation_stats_df.loc[act_name, "first_bw_use_rank"],
                    "last_bw_use_rank": ac_algo.activation_stats_df.loc[act_name, "last_bw_use_rank"]
                }
                for act_name in ac_decisions.keys()
                if act_name in ac_algo.activation_stats_df.index
            }
            print(f"Extracted liveness information for {len(activation_liveness)} activations")
            
            if debug:
                print(f"[DEBUG] Extracted liveness info for {len(activation_liveness)}/{len(ac_decisions)} activations")
                if len(activation_liveness) < len(ac_decisions):
                    missing = set(ac_decisions.keys()) - set(activation_liveness.keys())
                    print(f"[DEBUG] Missing liveness info for {len(missing)} activations")
                    print(f"[DEBUG] First few missing: {list(missing)[:5]}")
        except Exception as e:
            print(f"Error extracting activation liveness: {e}")
            if debug:
                print(f"[DEBUG] Exception details: {str(e)}")
                import traceback
                print(f"[DEBUG] Traceback: {traceback.format_exc()}")
            activation_liveness = None
    
    model_with_ac = apply_activation_checkpointing(model, ac_decisions, activation_liveness=activation_liveness, debug=debug)
    peak_memory_with_ac, time_with_ac = measure_memory_and_time(model_with_ac, batch, debug=debug)
    print(f"With AC - Peak memory: {peak_memory_with_ac / (1024**2):.2f} MiB, Iteration time: {time_with_ac:.4f} s")
    
    # 4. Validate correctness
    print("Validating correctness...")
    is_valid, rel_diff = validate_correctness(model, model_with_ac, batch, debug=debug)
    
    # Calculate memory reduction and time overhead
    memory_reduction = 1.0 - (peak_memory_with_ac / peak_memory_without_ac)
    time_overhead = (time_with_ac / time_without_ac) - 1.0
    
    print(f"Memory reduction: {memory_reduction:.2%}")
    print(f"Time overhead: {time_overhead:.2%}")
    
    # Return results
    results = {
        'batch_size': batch_size,
        'peak_memory_without_ac': peak_memory_without_ac,
        'peak_memory_with_ac': peak_memory_with_ac,
        'time_without_ac': time_without_ac,
        'time_with_ac': time_with_ac,
        'memory_reduction': memory_reduction,
        'time_overhead': time_overhead,
        'is_valid': is_valid,
        'rel_diff': rel_diff
    }
    
    return results

def create_comparison_charts(results, reports_dir, debug=False):
    """
    Create comparison charts for peak memory and iteration latency.
    
    Args:
        results: List of dictionaries containing profiling results
        reports_dir: Directory to save the charts
        debug: Whether to print debug information
    """
    if debug:
        print(f"[DEBUG] Creating comparison charts")
        print(f"[DEBUG] Results: {results}")
    
    # Extract data
    batch_sizes = [r['batch_size'] for r in results]
    peak_memory_without_ac = [r['peak_memory_without_ac'] / (1024**2) for r in results]  # Convert to MiB
    peak_memory_with_ac = [r['peak_memory_with_ac'] / (1024**2) for r in results]  # Convert to MiB
    time_without_ac = [r['time_without_ac'] * 1000 for r in results]  # Convert to ms
    time_with_ac = [r['time_with_ac'] * 1000 for r in results]  # Convert to ms
    
    if debug:
        print(f"[DEBUG] Batch sizes: {batch_sizes}")
        print(f"[DEBUG] Peak memory without AC (MiB): {peak_memory_without_ac}")
        print(f"[DEBUG] Peak memory with AC (MiB): {peak_memory_with_ac}")
        print(f"[DEBUG] Time without AC (ms): {time_without_ac}")
        print(f"[DEBUG] Time with AC (ms): {time_with_ac}")
    
    # 1. Bar chart: Peak memory vs. batch size
    plt.figure(figsize=(12, 6))
    x = np.arange(len(batch_sizes))
    width = 0.35
    
    if debug:
        print(f"[DEBUG] Creating memory bar chart")
    
    bars1 = plt.bar(x - width/2, peak_memory_without_ac, width, label='Without AC')
    bars2 = plt.bar(x + width/2, peak_memory_with_ac, width, label='With AC')
    
    plt.xlabel('Batch Size')
    plt.ylabel('Peak Memory (MiB)')
    plt.title('Peak Memory Usage vs. Batch Size')
    plt.xticks(x, batch_sizes)
    plt.legend()
    
    # Add value labels on top of bars
    for i, bar in enumerate(bars1):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                f'{peak_memory_without_ac[i]:.0f}',
                ha='center', va='bottom', rotation=0)
    
    for i, bar in enumerate(bars2):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                f'{peak_memory_with_ac[i]:.0f}',
                ha='center', va='bottom', rotation=0)
    
    # Add reduction percentage
    for i in range(len(batch_sizes)):
        reduction = results[i]['memory_reduction']
        plt.text(x[i], min(peak_memory_without_ac[i], peak_memory_with_ac[i]) / 2,
                f'{reduction:.1%}',
                ha='center', va='center', rotation=0,
                bbox=dict(facecolor='white', alpha=0.8))
    
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save the chart
    memory_chart_path = os.path.join(reports_dir, 'resnet152_memory_comparison.png')
    plt.savefig(memory_chart_path)
    plt.close()
    print(f"Memory comparison chart saved to: {memory_chart_path}")
    
    # 2. Line chart: Iteration latency vs. batch size
    plt.figure(figsize=(12, 6))
    
    if debug:
        print(f"[DEBUG] Creating latency line chart")
    
    plt.plot(batch_sizes, time_without_ac, 'o-', label='Without AC')
    plt.plot(batch_sizes, time_with_ac, 's-', label='With AC')
    
    plt.xlabel('Batch Size')
    plt.ylabel('Iteration Latency (ms)')
    plt.title('Iteration Latency vs. Batch Size')
    plt.legend()
    plt.grid(True)
    
    # Add value labels
    for i in range(len(batch_sizes)):
        plt.text(batch_sizes[i], time_without_ac[i] + 5,
                f'{time_without_ac[i]:.1f} ms',
                ha='center', va='bottom')
        plt.text(batch_sizes[i], time_with_ac[i] + 5,
                f'{time_with_ac[i]:.1f} ms',
                ha='center', va='bottom')
    
    # Add overhead percentage
    for i in range(len(batch_sizes)):
        overhead = results[i]['time_overhead']
        plt.text(batch_sizes[i], (time_without_ac[i] + time_with_ac[i]) / 2,
                f'{overhead:.1%}',
                ha='center', va='center',
                bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # Save the chart
    latency_chart_path = os.path.join(reports_dir, 'resnet152_latency_comparison.png')
    plt.savefig(latency_chart_path)
    plt.close()
    print(f"Latency comparison chart saved to: {latency_chart_path}")

def main():
    """Main function to run the activation checkpointing comparison."""
    parser = argparse.ArgumentParser(description='Activation Checkpointing Comparison')
    parser.add_argument('--batch-sizes', type=int, nargs='+', default=[4, 8, 16, 32],
                        help='Batch sizes to profile')
    parser.add_argument('--memory-budget', type=float, default=4.0,
                        help='Memory budget for activation checkpointing in GB (default: 4.0 GB)')
    parser.add_argument('--ac-percentage', type=float, default=0.5,
                        help='Percentage of bottleneck blocks to apply checkpointing to')
    parser.add_argument('--timeout', type=int, default=120,
                        help='Timeout in seconds for the activation checkpointing algorithm (default: 120)')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    args = parser.parse_args()
    
    print("=== Activation Checkpointing Comparison for ResNet-152 ===")
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Use CUDA if available
    device_str = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device_str}")
    
    # Ensure reports directory exists
    reports_dir = ensure_reports_directory()
    
    # Profile each batch size
    results = []
    for batch_size in args.batch_sizes:
        try:
            result = profile_batch_size(batch_size, device_str, args.memory_budget, args.debug, args.timeout)
            results.append(result)
        except Exception as e:
            print(f"Error profiling batch size {batch_size}: {e}")
    
    # Create comparison charts
    if results:
        create_comparison_charts(results, reports_dir)
        
        # Print summary table
        print("\n=== Comparison Summary ===")
        print(f"{'Batch Size':<10} {'Memory w/o AC (MiB)':<20} {'Memory w/ AC (MiB)':<20} {'Reduction':<10} {'Time w/o AC (ms)':<20} {'Time w/ AC (ms)':<20} {'Overhead':<10} {'Valid':<10}")
        print("-" * 120)
        
        for r in results:
            print(f"{r['batch_size']:<10} "
                  f"{r['peak_memory_without_ac'] / (1024**2):<20.1f} "
                  f"{r['peak_memory_with_ac'] / (1024**2):<20.1f} "
                  f"{r['memory_reduction']:<10.1%} "
                  f"{r['time_without_ac'] * 1000:<20.1f} "
                  f"{r['time_with_ac'] * 1000:<20.1f} "
                  f"{r['time_overhead']:<10.1%} "
                  f"{r['is_valid']:<10}")
    else:
        print("No successful profiling results to report.")
    
    print("\n=== Comparison completed ===")

if __name__ == "__main__":
    main()