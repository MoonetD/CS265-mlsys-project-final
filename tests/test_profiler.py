#!/usr/bin/env python
"""
Unit tests for GraphProfiler.

This file contains tests to verify the correctness of the GraphProfiler implementation.
It includes tests for activation liveness, memory conservation, and CSV schema.

To run these tests:
    conda run -n ml_env python tests/test_profiler.py
"""

import os
import sys
import unittest
import pandas as pd
import torch
import torch.nn as nn

# Add parent directory to path so we can import from starter_code
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from starter_code.graph_prof import GraphProfiler, NodeType
from starter_code.graph_tracer import SEPFunction, compile

class SimpleMLP(nn.Module):
    """A simple 3-layer MLP model for testing the GraphProfiler."""
    
    def __init__(self, input_dim=64, hidden_dim=128, output_dim=10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x

def train_step(model, optim, batch):
    """Simple training step function that will be traced by the compiler."""
    loss = model(batch).sum()
    loss = SEPFunction.apply(loss)
    loss.backward()
    optim.step()
    optim.zero_grad()

# Global variable to store the profiler instance for tests
_graph_profiler = None

def graph_transformation(gm, args):
    """Graph transformation function that profiles the model execution."""
    global _graph_profiler
    
    # Initialize the GraphProfiler with the graph module
    _graph_profiler = GraphProfiler(gm)
    
    # Run the profiler for a few iterations
    warm_up_iters, profile_iters = 1, 3
    with torch.no_grad():
        # Warm-up run
        for _ in range(warm_up_iters):
            _graph_profiler.run(*args)
        
        # Reset stats before actual profiling
        _graph_profiler.reset_stats()
        
        # Profile runs
        for _ in range(profile_iters):
            _graph_profiler.run(*args)
    
    # Aggregate and analyze the results
    _graph_profiler.aggregate_stats(num_runs=profile_iters)
    
    # Save statistics to CSV files for testing
    _graph_profiler.save_stats_to_csv(filename_prefix="test_profiler_stats")
    
    return gm

class TestProfiler(unittest.TestCase):
    """Test cases for GraphProfiler."""
    
    @classmethod
    def setUpClass(cls):
        """Set up the test environment once for all tests."""
        print("Setting up test environment...")
        
        # Set random seed for reproducibility
        torch.manual_seed(42)
        
        # Use CUDA if available
        device_str = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device_str}")
        
        # Create a simple MLP model
        input_dim, hidden_dim, output_dim = 64, 128, 10
        model = SimpleMLP(input_dim, hidden_dim, output_dim).to(device_str)
        
        # Create a random batch of data
        batch_size = 32
        batch = torch.randn(batch_size, input_dim).to(device_str)
        
        # Create an optimizer
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        
        # Initialize gradients for optimizer step to be traceable
        for param in model.parameters():
            if param.requires_grad:
                param.grad = torch.rand_like(param, device=device_str)
        
        # Perform one step to initialize optimizer states if needed
        optimizer.step()
        optimizer.zero_grad()
        
        # Compile and profile the model
        print("Compiling and profiling the model...")
        compiled_fn = compile(train_step, graph_transformation)
        compiled_fn(model, optimizer, batch)
        
        # Ensure the profiler was created
        if _graph_profiler is None:
            raise RuntimeError("GraphProfiler was not created during setup")
        
        print("Test environment setup complete.")
    
    def test_activation_liveness(self):
        """Test that activation liveness information is correct."""
        print("\nRunning test_activation_liveness...")
        
        # Check that the profiler exists
        self.assertIsNotNone(_graph_profiler, "GraphProfiler not initialized")
        
        # Check that we have activation liveness information
        self.assertGreater(len(_graph_profiler.activation_liveness), 0, 
                          "No activation liveness information found")
        
        # Check that for every activation, first_bw_use_rank > last_fw_use_rank
        errors = []
        for act_name, liveness in _graph_profiler.activation_liveness.items():
            first_bw_use_rank = liveness["first_bw_use_rank"]
            last_fw_use_rank = liveness["last_fw_use_rank"]
            
            # Skip if either rank is -1 (not used in that phase)
            if first_bw_use_rank == -1 or last_fw_use_rank == -1:
                continue
            
            if first_bw_use_rank <= last_fw_use_rank:
                errors.append(f"Activation {act_name}: first_bw_use_rank ({first_bw_use_rank}) <= last_fw_use_rank ({last_fw_use_rank})")
        
        # Assert that there are no errors
        self.assertEqual(len(errors), 0, f"Activation liveness errors found:\n" + "\n".join(errors))
        
        print("test_activation_liveness passed!")
    
    def test_memory_conservation(self):
        """Test that memory breakdown adds up to peak memory."""
        print("\nRunning test_memory_conservation...")
        
        # Check that the profiler exists
        self.assertIsNotNone(_graph_profiler, "GraphProfiler not initialized")
        
        # Get the peak memory from the profiler
        max_node_peak = max(_graph_profiler.median_peak_mem_node.values()) if _graph_profiler.median_peak_mem_node else 0
        
        # Calculate the sum of parameters, gradients, and peak activations
        total_param_mem = sum(_graph_profiler.param_sizes.values())
        total_grad_mem = sum(_graph_profiler.grad_sizes.values())
        
        # Calculate peak activation memory based on liveness and avg sizes
        peak_activation_mem = 0
        if _graph_profiler.activation_liveness and _graph_profiler.median_memory_sizes:
            # Use the same logic as in GraphProfiler.print_stats
            from collections import defaultdict
            live_activations_at_rank = defaultdict(set)
            
            # Determine live intervals
            for act_name, liveness in _graph_profiler.activation_liveness.items():
                create_rank = liveness['creation_rank']
                last_use_rank = max(liveness['last_fw_use_rank'], liveness['last_bw_use_rank'])
                if last_use_rank == -1:  # Handle cases where only created, not used
                    last_use_rank = create_rank
                
                # Ensure ranks are valid before adding to dict
                if create_rank >= 0 and last_use_rank >= create_rank:
                    for rank in range(create_rank, last_use_rank + 1):
                        live_activations_at_rank[rank].add(act_name)
            
            # Find peak memory sum across ranks
            if live_activations_at_rank:  # Check if any activations were live
                max_rank = max(live_activations_at_rank.keys()) if live_activations_at_rank else -1
                for rank in range(max_rank + 1):
                    current_concurrent_mem = 0
                    for act_name in live_activations_at_rank[rank]:
                        current_concurrent_mem += _graph_profiler.median_memory_sizes.get(act_name, 0)
                    peak_activation_mem = max(peak_activation_mem, current_concurrent_mem)
        
        # Sum of all components
        total_mem = total_param_mem + total_grad_mem + peak_activation_mem
        
        # Allow for some margin of error (5%)
        margin = 0.05
        lower_bound = total_mem * (1 - margin)
        upper_bound = total_mem * (1 + margin)
        
        print(f"Max node peak memory: {max_node_peak / (1024**2):.2f} MiB")
        print(f"Sum of components: {total_mem / (1024**2):.2f} MiB")
        print(f"  - Parameters: {total_param_mem / (1024**2):.2f} MiB")
        print(f"  - Gradients: {total_grad_mem / (1024**2):.2f} MiB")
        print(f"  - Peak Activations: {peak_activation_mem / (1024**2):.2f} MiB")
        
        # Assert that the sum is within the margin of error
        self.assertTrue(lower_bound <= max_node_peak <= upper_bound,
                       f"Memory conservation error: max_node_peak ({max_node_peak}) not within {margin*100}% of total_mem ({total_mem})")
        
        print("test_memory_conservation passed!")
    
    def test_csv_schema(self):
        """Test that the CSV files have the required headers and no NaNs."""
        print("\nRunning test_csv_schema...")
        
        # Check that the CSV files exist
        node_csv = "test_profiler_stats_node_stats.csv"
        activation_csv = "test_profiler_stats_activation_stats.csv"
        
        self.assertTrue(os.path.exists(node_csv), f"Node stats CSV file not found: {node_csv}")
        self.assertTrue(os.path.exists(activation_csv), f"Activation stats CSV file not found: {activation_csv}")
        
        # Check node stats CSV
        node_df = pd.read_csv(node_csv)
        
        # Check required headers
        required_node_headers = ['rank', 'node_name', 'node_type', 'gtype', 'median_run_time_s', 
                                'median_peak_mem_bytes', 'median_active_mem_bytes', 'device']
        for header in required_node_headers:
            self.assertIn(header, node_df.columns, f"Required header '{header}' not found in node stats CSV")
        
        # Check for NaNs
        for header in required_node_headers:
            self.assertFalse(node_df[header].isna().any(), f"NaN values found in '{header}' column of node stats CSV")
        
        # Check activation stats CSV
        act_df = pd.read_csv(activation_csv)
        
        # Check required headers
        required_act_headers = ['activation_name', 'creation_rank', 'last_fw_use_rank', 'first_bw_use_rank', 
                               'last_bw_use_rank', 'median_mem_size_bytes', 'inactive_time_s', 'recomp_time_s']
        for header in required_act_headers:
            self.assertIn(header, act_df.columns, f"Required header '{header}' not found in activation stats CSV")
        
        # Check for NaNs
        for header in required_act_headers:
            self.assertFalse(act_df[header].isna().any(), f"NaN values found in '{header}' column of activation stats CSV")
        
        print("test_csv_schema passed!")

if __name__ == "__main__":
    unittest.main()