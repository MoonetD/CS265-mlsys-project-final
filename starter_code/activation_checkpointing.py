"""
Activation Checkpointing Algorithm Implementation (Stage 2)

This module implements the core activation checkpointing algorithm (Scheduler logic from the Π-TWO paper, 
specifically Algorithm B) which decides which activations to keep in memory and which to recompute 
based on the profiling data gathered in Stage 1.

Tasks to implement:
1. Algorithm Input Preparation
2. Core Scheduling Logic (Algorithm B)
3. Recompute Overhead Calculation
4. Memory Simulation
5. Algorithm Output
"""

import pandas as pd
import numpy as np
import time
import os
import logging
import datetime
from pathlib import Path

# Configure logging with datetime-formatted log file
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
log_filename = f"activation_checkpointing_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
log_path = log_dir / log_filename

# Set up file handler with datetime log file
file_handler = logging.FileHandler(log_path)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

# Set up console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(file_handler)
logger.addHandler(console_handler)

class ActivationCheckpointingAlgorithm:
    def __init__(self, node_stats_path, activation_stats_path, memory_budget_gb):
        """
        Initializes the Activation Checkpointing Algorithm.

        Args:
            node_stats_path (str): Path to the CSV file containing node statistics.
            activation_stats_path (str): Path to the CSV file containing activation statistics.
            memory_budget_gb (float): GPU memory budget in Gigabytes.
        """
        try:
            self.node_stats_df = pd.read_csv(node_stats_path)
            self.activation_stats_df = pd.read_csv(activation_stats_path)
            logger.info(f"Loaded node stats with {len(self.node_stats_df)} rows")
            logger.info(f"Loaded activation stats with {len(self.activation_stats_df)} rows")
        except FileNotFoundError as e:
            logger.error(f"Error: One or both profiler statistics files not found: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading CSV files: {e}")
            raise

        # Convert memory budget from GB to bytes
        self.memory_budget_bytes = memory_budget_gb * (1024**3)
        logger.info(f"Memory budget set to {memory_budget_gb} GB ({self.memory_budget_bytes} bytes)")
        
        # Initialize schedule dictionary to store decisions (act_name -> 'RETAINED' or 'RECOMPUTE')
        self.schedule = {}
        
        # Ensure activation_name is the index for activation_stats_df
        if 'activation_name' in self.activation_stats_df.columns:
            self.activation_stats_df.set_index('activation_name', inplace=True, drop=False)
        else:
            raise ValueError("activation_stats_df must contain an 'activation_name' column.")
        
        # Ensure node_name is the index for node_stats_df
        if 'node_name' in self.node_stats_df.columns:
            self.node_stats_df.set_index('node_name', inplace=True, drop=False)
        else:
            raise ValueError("node_stats_df must contain a 'node_name' column.")
        
        # Validate required columns in activation_stats_df
        required_act_cols = ['creation_rank', 'last_fw_use_rank', 'first_bw_use_rank',
                            'last_bw_use_rank', 'median_mem_size_bytes', 'recomp_time_s']
        missing_cols = [col for col in required_act_cols if col not in self.activation_stats_df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in activation_stats_df: {missing_cols}")
        
        # Validate required columns in node_stats_df
        required_node_cols = ['rank', 'gtype', 'median_run_time_s', 'median_active_mem_bytes']
        missing_cols = [col for col in required_node_cols if col not in self.node_stats_df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in node_stats_df: {missing_cols}")

    def _calculate_recompute_overhead(self, activation_name):
        """
        Calculates the recomputation time overhead for a given activation.
        
        Args:
            activation_name (str): Name of the activation to calculate overhead for.
            
        Returns:
            tuple: (recompute_time, activation_memory_size)
        """
        if activation_name not in self.activation_stats_df.index:
            logger.warning(f"Activation {activation_name} not found in stats for recompute overhead.")
            return 0, 0  # Time, Memory
        
        # Get recomputation time from activation stats
        recomp_time = self.activation_stats_df.loc[activation_name, 'recomp_time_s']
        
        # Get memory size of the activation
        act_memory = self.activation_stats_df.loc[activation_name, 'median_mem_size_bytes']
        
        return recomp_time if pd.notna(recomp_time) else 0, act_memory if pd.notna(act_memory) else 0

    def _get_node_execution_order(self):
        """
        Returns a list of node names in their execution order (rank).
        Includes both forward and backward pass nodes.
        
        Returns:
            list: Node names in execution order.
        """
        # Sort nodes by their rank
        sorted_nodes = self.node_stats_df.dropna(subset=['rank']).sort_values(by='rank')
        return sorted_nodes['node_name'].tolist()

    def _get_activation_details(self, activation_name):
        """
        Helper to get all details for an activation.
        
        Args:
            activation_name (str): Name of the activation.
            
        Returns:
            pd.Series: Activation details or None if not found.
        """
        if activation_name not in self.activation_stats_df.index:
            return None
        return self.activation_stats_df.loc[activation_name]

    def _get_node_details(self, node_name):
        """
        Helper to get all details for a node.
        
        Args:
            node_name (str): Name of the node.
            
        Returns:
            pd.Series: Node details or None if not found.
        """
        if node_name not in self.node_stats_df.index:
            return None
        return self.node_stats_df.loc[node_name]

    def _simulate_memory_usage(self, current_schedule, fixed_overhead_bytes=0, debug=False):
        """
        Simulates peak memory usage and total execution time based on a given schedule.
        
        Args:
            current_schedule (dict): Activation name to decision ('RETAINED', 'RECOMPUTE').
            fixed_overhead_bytes (float): Estimated memory for parameters, gradients, optimizer.
            debug (bool): Whether to print detailed debug information.
            
        Returns:
            float: Estimated peak GPU memory in bytes.
            float: Total execution time in seconds.
        """
        # Start timing the simulation
        sim_start_time = time.time()
        
        # Initialize timing dictionary for detailed breakdown
        sim_timing = {
            'execution_time_calc': 0,
            'recompute_overhead_calc': 0,
            'memory_mapping': 0,
            'node_processing': 0,
            'total': 0
        }
        
        if debug:
            logger.info(f"Starting memory simulation with {len(current_schedule)} activations...")
            logger.info(f"Fixed overhead: {fixed_overhead_bytes / (1024**3):.3f} GB")
            
        # --- Calculate Total Execution Time ---
        # Start with sum of all base node run times
        exec_time_start = time.time()
        if debug:
            logger.info("Calculating total execution time...")
            
        total_execution_time = self.node_stats_df['median_run_time_s'].sum()
        
        # Add recomputation times for activations scheduled for RECOMPUTE
        recompute_count = 0
        recompute_time_total = 0.0
        
        for act_name, decision in current_schedule.items():
            if decision == 'RECOMPUTE':
                act_details = self._get_activation_details(act_name)
                # Ensure act_details is a Series and 'recomp_time_s' exists
                if act_details is not None and isinstance(act_details, pd.Series) and \
                   'recomp_time_s' in act_details.index and pd.notna(act_details['recomp_time_s']):
                    recomp_time = act_details['recomp_time_s']
                    total_execution_time += recomp_time
                    recompute_time_total += recomp_time
                    recompute_count += 1
        
        sim_timing['execution_time_calc'] = time.time() - exec_time_start

        # if debug:
        #     logger.info(f"Base execution time: {total_execution_time - recompute_time_total:.4f}s")
        #     logger.info(f"Added recomputation time for {recompute_count} activations: {recompute_time_total:.4f}s")
        #     logger.info(f"Total execution time: {total_execution_time:.4f}s")

        # # --- Simulate Peak Memory Usage ---
        # if debug:
        #     logger.info("Simulating peak memory usage...")
            
        # --- Simulate Peak Memory Usage ---
        memory_mapping_start = time.time()
        
        # Initialize memory tracking variables
        fw_inter_mem = 0  # Memory for intermediate tensors in forward pass
        bw_inter_mem = 0  # Memory for intermediate tensors in backward pass
        fw_active_mem = 0  # Active memory during forward pass
        bw_active_mem = 0  # Active memory during backward pass
        peak_mem = fixed_overhead_bytes  # Track peak memory, start with fixed overhead
        
        # We'll track forward and backward passes separately
        fw_peak_mem = fixed_overhead_bytes
        bw_peak_mem = fixed_overhead_bytes
        
        # We'll also track retained activations that need to be kept for backward pass
        retained_activations_mem = 0
        
        sim_timing['memory_mapping'] = time.time() - memory_mapping_start
        
        if debug:
            logger.info(f"Fixed overhead: {fixed_overhead_bytes / (1024**3):.3f} GB")
            logger.info(f"Starting peak memory: {peak_mem / (1024**3):.3f} GB")
        
        # Get execution order
        execution_order = self._get_node_execution_order()
        if not execution_order:
            if debug:
                logger.warning("Could not determine execution order.")
            return float('inf'), total_execution_time
        
        # Pre-fetch all node details to avoid repeated lookups
        node_details_cache = {}
        for node_name in execution_order:
            node_details_cache[node_name] = self._get_node_details(node_name)
        
        # Create a mapping from rank to activations created/used at that rank
        activations_by_creation_rank = {}
        activations_by_first_bw_use_rank = {}
        activations_by_last_fw_use_rank = {}
        activations_by_last_bw_use_rank = {}  # Add mapping for last backward use
        
        for act_idx, act_details_series in self.activation_stats_df.iterrows():
            act_name = act_details_series['activation_name']
            
            # Skip if not in schedule
            if act_name not in current_schedule:
                continue
                
            # Map by creation rank
            if pd.notna(act_details_series['creation_rank']):
                rank = int(act_details_series['creation_rank'])
                if rank not in activations_by_creation_rank:
                    activations_by_creation_rank[rank] = []
                activations_by_creation_rank[rank].append(act_details_series)
            
            # Map by first backward use rank
            if pd.notna(act_details_series['first_bw_use_rank']):
                rank = int(act_details_series['first_bw_use_rank'])
                if rank not in activations_by_first_bw_use_rank:
                    activations_by_first_bw_use_rank[rank] = []
                activations_by_first_bw_use_rank[rank].append(act_details_series)
            
            # Map by last backward use rank
            if pd.notna(act_details_series['last_bw_use_rank']):
                rank = int(act_details_series['last_bw_use_rank'])
                if rank not in activations_by_last_bw_use_rank:
                    activations_by_last_bw_use_rank[rank] = []
                activations_by_last_bw_use_rank[rank].append(act_details_series)
            
            # Map by last forward use rank
            if pd.notna(act_details_series['last_fw_use_rank']) and act_details_series['last_fw_use_rank'] != -1:
                rank = int(act_details_series['last_fw_use_rank'])
                if rank not in activations_by_last_fw_use_rank:
                    activations_by_last_fw_use_rank[rank] = []
                activations_by_last_fw_use_rank[rank].append(act_details_series)
        
        # Process nodes in execution order
        node_processing_start = time.time()
        total_nodes = len(execution_order)
        if debug:
            logger.info(f"Processing {total_nodes} nodes in execution order...")
        
        # First, simulate forward pass
        if debug:
            logger.info("Simulating forward pass...")
        
        # Find the boundary between forward and backward passes
        forward_nodes = []
        backward_nodes = []
        for node_name in execution_order:
            node_details = node_details_cache[node_name]
            if node_details is None:
                continue
            
            if node_details['gtype'] == 'forward':
                forward_nodes.append(node_name)
            elif node_details['gtype'] == 'backward':
                backward_nodes.append(node_name)
        
        # Process forward pass nodes
        for i, node_name in enumerate(forward_nodes):
            if debug and i % max(1, len(forward_nodes) // 10) == 0:
                logger.info(f"  Processing forward node {i+1}/{len(forward_nodes)} ({(i+1)/len(forward_nodes)*100:.1f}%)...")
            
            node_details = node_details_cache[node_name]
            if node_details is None:
                continue
            
            node_rank = node_details['rank']
            
            # Update active memory for forward pass - use a scaling factor to avoid double-counting
            # Since active memory might include some intermediate tensors we're already tracking
            if 'median_active_mem_bytes' in node_details and pd.notna(node_details['median_active_mem_bytes']):
                # Use 70% of reported active memory to avoid double-counting with intermediate tensors
                fw_active_mem = node_details['median_active_mem_bytes'] * 1
            
            # Add memory for newly created tensors
            if node_rank in activations_by_creation_rank:
                for act_details in activations_by_creation_rank[node_rank]:
                    act_name = act_details['activation_name']
                    if current_schedule.get(act_name) == 'RETAINED':
                        # Only add memory for activations we're keeping (not recomputing)
                        if 'median_mem_size_bytes' in act_details and pd.notna(act_details['median_mem_size_bytes']):
                            mem_size = act_details['median_mem_size_bytes']
                            fw_inter_mem += mem_size
                            
                            # Also track retained activations for backward pass
                            if pd.notna(act_details['first_bw_use_rank']):
                                retained_activations_mem += mem_size
                                
                            if debug and mem_size > 1024*1024:  # Only log significant activations (>1MB)
                                logger.info(f"  Added {mem_size/(1024**2):.2f} MB for retained activation {act_name}")
            
            # Remove memory for tensors that are no longer needed in forward pass
            if node_rank in activations_by_last_fw_use_rank:
                for act_details in activations_by_last_fw_use_rank[node_rank]:
                    act_name = act_details['activation_name']
                    if current_schedule.get(act_name) == 'RETAINED':
                        if 'median_mem_size_bytes' in act_details and pd.notna(act_details['median_mem_size_bytes']):
                            fw_inter_mem -= act_details['median_mem_size_bytes']
            
            # Calculate current forward pass memory
            current_fw_mem = fw_active_mem + fw_inter_mem + fixed_overhead_bytes
            
            # Update peak forward memory
            if current_fw_mem > fw_peak_mem:
                fw_peak_mem = current_fw_mem
                if debug:
                    logger.info(f"  New forward peak memory: {fw_peak_mem/(1024**3):.3f} GB at node {node_name} (rank {node_rank})")
                    logger.info(f"    Breakdown: FW active={fw_active_mem/(1024**3):.3f} GB, FW inter={fw_inter_mem/(1024**3):.3f} GB")
                    logger.info(f"    Fixed overhead={fixed_overhead_bytes/(1024**3):.3f} GB")
        
        # Now, simulate backward pass
        if debug:
            logger.info("Simulating backward pass...")
            logger.info(f"Retained activations memory: {retained_activations_mem / (1024**3):.3f} GB")
        
        # Reset intermediate memory for backward pass
        fw_inter_mem = 0
        bw_inter_mem = 0  # Don't start with all retained activations at once
        
        # Process backward pass nodes
        for i, node_name in enumerate(backward_nodes):
            if debug and i % max(1, len(backward_nodes) // 10) == 0:
                logger.info(f"  Processing backward node {i+1}/{len(backward_nodes)} ({(i+1)/len(backward_nodes)*100:.1f}%)...")
            
            node_details = node_details_cache[node_name]
            if node_details is None:
                continue
            
            node_rank = node_details['rank']
            
            # Update active memory for backward pass - use a scaling factor to avoid double-counting
            if 'median_active_mem_bytes' in node_details and pd.notna(node_details['median_active_mem_bytes']):
                # Use 70% of reported active memory to avoid double-counting with intermediate tensors
                bw_active_mem = node_details['median_active_mem_bytes'] * 1
            
            # Add memory for activations needed at this rank (both retained and recomputed)
            if node_rank in activations_by_first_bw_use_rank:
                for act_details in activations_by_first_bw_use_rank[node_rank]:
                    act_name = act_details['activation_name']
                    decision = current_schedule.get(act_name)
                    if 'median_mem_size_bytes' in act_details and pd.notna(act_details['median_mem_size_bytes']):
                        mem_size = act_details['median_mem_size_bytes']
                        
                        if decision == 'RECOMPUTE':
                            # Add memory for recomputed activation
                            bw_inter_mem += mem_size
                            if debug and mem_size > 1024*1024:
                                logger.info(f"  Added {mem_size/(1024**2):.2f} MB for recomputed activation {act_name}")
                        elif decision == 'RETAINED':
                            # Add memory for retained activation only when it's first used in backward pass
                            bw_inter_mem += mem_size
                            if debug and mem_size > 1024*1024:
                                logger.info(f"  Added {mem_size/(1024**2):.2f} MB for retained activation {act_name}")
            
            # Remove memory for tensors that are no longer needed in backward pass
            if node_rank in activations_by_last_bw_use_rank:
                for act_details in activations_by_last_bw_use_rank[node_rank]:
                    act_name = act_details['activation_name']
                    if 'median_mem_size_bytes' in act_details and pd.notna(act_details['median_mem_size_bytes']):
                        # Remove the memory for this activation after its last use
                        mem_size = act_details['median_mem_size_bytes']
                        bw_inter_mem -= mem_size
                        if debug and mem_size > 1024*1024:  # Only log significant activations (>1MB)
                            logger.info(f"  Removed {mem_size/(1024**2):.2f} MB for activation {act_name} (no longer needed)")
            
            # Calculate current backward pass memory
            current_bw_mem = bw_active_mem + bw_inter_mem + fixed_overhead_bytes
            
            # Update peak backward memory
            if current_bw_mem > bw_peak_mem:
                bw_peak_mem = current_bw_mem
                if debug:
                    logger.info(f"  New backward peak memory: {bw_peak_mem/(1024**3):.3f} GB at node {node_name} (rank {node_rank})")
                    logger.info(f"    Breakdown: BW active={bw_active_mem/(1024**3):.3f} GB, BW inter={bw_inter_mem/(1024**3):.3f} GB")
                    logger.info(f"    Fixed overhead={fixed_overhead_bytes/(1024**3):.3f} GB")
        
        # Overall peak memory is the maximum of forward and backward peaks
        peak_mem = max(fw_peak_mem, bw_peak_mem)
        
        sim_timing['node_processing'] = time.time() - node_processing_start
        
        # Calculate total simulation time
        sim_timing['total'] = time.time() - sim_start_time
        
        if debug:
            logger.info(f"Memory simulation complete.")
            logger.info(f"Forward pass peak memory: {fw_peak_mem / (1024**3):.3f} GB")
            logger.info(f"Backward pass peak memory: {bw_peak_mem / (1024**3):.3f} GB")
            logger.info(f"Overall peak memory: {peak_mem / (1024**3):.3f} GB")
            logger.info(f"Final execution time: {total_execution_time:.4f}s")
            
            # Calculate memory savings
            memory_savings = sum(
                act_details['median_mem_size_bytes']
                for act_name, decision in current_schedule.items()
                if decision == 'RECOMPUTE'
                for act_details in [self._get_activation_details(act_name)]
                if act_details is not None and 'median_mem_size_bytes' in act_details
            )
            logger.info(f"Memory savings from activation checkpointing: {memory_savings / (1024**3):.3f} GB")
            logger.info(f"Computation overhead from recomputation: {recompute_time_total:.4f}s")
            
            # Log timing information for the simulation
            logger.info(f"Simulation timing breakdown:")
            logger.info(f"  Total simulation time: {sim_timing['total']:.4f}s")
            logger.info(f"  Execution time calculation: {sim_timing['execution_time_calc']:.4f}s ({sim_timing['execution_time_calc']/sim_timing['total']*100:.1f}%)")
            logger.info(f"  Memory mapping: {sim_timing['memory_mapping']:.4f}s ({sim_timing['memory_mapping']/sim_timing['total']*100:.1f}%)")
            logger.info(f"  Node processing: {sim_timing['node_processing']:.4f}s ({sim_timing['node_processing']/sim_timing['total']*100:.1f}%)")
        
        return peak_mem, total_execution_time

    def _save_schedule_to_csv(self, schedule):
        """
        Save the activation checkpointing schedule to a CSV file.
        
        Args:
            schedule (dict): The schedule mapping activation names to 'RETAINED' or 'RECOMPUTE'.
        """
        try:
            # Create a DataFrame from the schedule
            schedule_df = pd.DataFrame({
                'activation_name': list(schedule.keys()),
                'decision': list(schedule.values())
            })
            
            # Add additional information from activation_stats_df if available
            for col in ['median_mem_size_bytes', 'recomp_time_s', 'creation_rank', 'first_bw_use_rank']:
                if col in self.activation_stats_df.columns:
                    schedule_df[col] = [
                        self.activation_stats_df.loc[act, col] if act in self.activation_stats_df.index else None
                        for act in schedule_df['activation_name']
                    ]
            
            # Ensure reports directory exists
            reports_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'reports')
            os.makedirs(reports_dir, exist_ok=True)
            
            # Save to CSV
            output_path = os.path.join(reports_dir, 'ac_decisions.csv')
            schedule_df.to_csv(output_path, index=False)
            logger.info(f"Saved activation checkpointing decisions to {output_path}")
        except Exception as e:
            logger.error(f"Error saving schedule to CSV: {e}")

    def decide_checkpoints(self, fixed_overhead_gb=0.5, debug=False, batch_size=50, max_iterations=1000, timeout_seconds=120):
        """
        Decides which activations to checkpoint and which to recompute.
        Implements Algorithm B from the Π-TWO paper.

        Args:
            fixed_overhead_gb (float): Estimated fixed memory overhead for parameters,
                                      gradients, optimizer states in GB.
            debug (bool): Whether to print detailed debug information.
            batch_size (int): Number of activations to process at once for faster convergence.
            max_iterations (int): Maximum number of iterations to prevent infinite loops.
            timeout_seconds (int): Maximum time in seconds to run the algorithm before timing out.
            
        Returns:
            dict: A schedule mapping activation names to 'RETAINED' or 'RECOMPUTE'.
        """
        # Start overall timing
        overall_start_time = time.time()
        
        logger.info(f"Starting checkpoint decision algorithm...")
        logger.info(f"Fixed overhead: {fixed_overhead_gb} GB, Memory budget: {self.memory_budget_bytes / (1024**3):.2f} GB")
        
        fixed_overhead_bytes = fixed_overhead_gb * (1024**3)

        # Initialize timing dictionary to track execution time of different parts
        timing_stats = {
            'initialization': 0,
            'initial_memory_simulation': 0,
            'memory_component_analysis': 0,
            'candidate_filtering': 0,
            'main_loop': 0,
            'main_loop_iterations': [],
            'memory_simulations': 0,
            'candidate_selection': 0,
            'final_memory_simulation': 0,
            'total': 0
        }
        
        # Initial state: checkpoint all activations (mark all as RETAINED)
        init_start = time.time()
        logger.info("Initializing schedule with all activations checkpointed...")
        current_schedule = {
            act_name: 'RETAINED'
            for act_name in self.activation_stats_df['activation_name']
            if pd.notna(self.activation_stats_df.loc[act_name, 'median_mem_size_bytes']) and
               self.activation_stats_df.loc[act_name, 'median_mem_size_bytes'] > 0
        }
        timing_stats['initialization'] = time.time() - init_start
        
        logger.info(f"Initial schedule has {len(current_schedule)} activations marked for RETAINED")
        
        # Run initial memory simulation to get baseline
        sim_start = time.time()
        initial_peak_memory, initial_exec_time = self._simulate_memory_usage(
            current_schedule,
            fixed_overhead_bytes,
            debug=debug
        )
        timing_stats['initial_memory_simulation'] = time.time() - sim_start
        
        logger.info(f"Initial peak memory: {initial_peak_memory / (1024**3):.2f} GB")
        logger.info(f"Initial execution time: {initial_exec_time:.2f}s")
        
        # Analyze memory components to determine if budget is achievable
        mem_analysis_start = time.time()
        fw_active_mem, bw_active_mem, fw_inter_mem, bw_inter_mem = self._analyze_memory_components(
            current_schedule, fixed_overhead_bytes, debug=debug
        )
        timing_stats['memory_component_analysis'] = time.time() - mem_analysis_start
        
        # Calculate incompressible memory (memory that can't be reduced through activation checkpointing)
        # Apply a safety factor to account for potential underestimation
        safety_factor = 1.0  # No safety factor - use exact incompressible memory
        # Take the maximum of forward and backward active memory, not their sum
        # This is because forward and backward passes don't occur simultaneously
        incompressible_memory = max(fw_active_mem, bw_active_mem) + fixed_overhead_bytes
        logger.info(f"Incompressible memory: {incompressible_memory / (1024**3):.2f} GB (Max of FW active: {fw_active_mem / (1024**3):.2f} GB and BW active: {bw_active_mem / (1024**3):.2f} GB, plus Fixed overhead: {fixed_overhead_bytes / (1024**3):.2f} GB)")
        logger.info(f"Applied safety factor of {safety_factor:.2f} to incompressible memory estimate")
        
        # Check if budget is achievable
        if incompressible_memory > self.memory_budget_bytes:
            logger.warning(f"Memory budget of {self.memory_budget_bytes / (1024**3):.2f} GB is not achievable!")
            logger.warning(f"Estimated minimum possible memory is {incompressible_memory / (1024**3):.2f} GB due to incompressible components")
            logger.warning(f"Will try to get as close as possible to the budget")
            
            # Set a more realistic target if budget is unachievable
            # Use the incompressible memory as the target - we can't go below this
            effective_budget = incompressible_memory
            logger.warning(f"Setting effective target budget to incompressible memory: {effective_budget / (1024**3):.2f} GB")
        else:
            # Set a slightly lower effective budget to ensure we meet the actual budget
            effective_budget = self.memory_budget_bytes * 0.98  # 2% below requested budget
            logger.info(f"Setting effective target budget to {effective_budget / (1024**3):.2f} GB (2% below requested)")
        
        # Filter out activations with no valid size or recompute stats
        filter_start = time.time()
        logger.info("Filtering valid activations...")
        valid_activations_df = self.activation_stats_df.dropna(subset=['median_mem_size_bytes', 'recomp_time_s', 'creation_rank', 'last_fw_use_rank'])
        # No minimum memory size threshold - consider all activations regardless of size
        timing_stats['candidate_filtering'] = time.time() - filter_start
        logger.info(f"Found {len(valid_activations_df)} valid activations for consideration")

        # Create a set of candidate activations
        candidate_set = set(valid_activations_df.index)
        
        # Initialize tracking set for recomputation
        recomps = set()
        
        # Track best schedule so far
        best_schedule = current_schedule.copy()
        best_peak_memory = initial_peak_memory
        best_exec_time = initial_exec_time
        
        # Main loop of Algorithm B
        main_loop_start = time.time()
        iteration = 0
        start_time = time.time()
        while iteration < max_iterations and candidate_set:
            # Start timing for this iteration
            iter_start_time = time.time()
            
            # Check for timeout
            current_time = time.time()
            elapsed_time = current_time - start_time
            if elapsed_time > timeout_seconds:
                logger.warning(f"Timeout reached after {elapsed_time:.1f} seconds. Stopping algorithm.")
                break
                
            iteration += 1
            logger.info(f"\nIteration {iteration}/{max_iterations} (Elapsed time: {elapsed_time:.1f}s)")
            
            # Run memory simulation to check current state
            sim_start = time.time()
            current_peak_memory, current_exec_time = self._simulate_memory_usage(
                current_schedule,
                fixed_overhead_bytes,
                debug=(iteration == 1 and debug)
            )
            sim_time = time.time() - sim_start
            timing_stats['memory_simulations'] += sim_time
            
            logger.info(f"Simulated peak memory: {current_peak_memory / (1024**3):.2f} GB. Budget: {self.memory_budget_bytes / (1024**3):.2f} GB. Exec time: {current_exec_time:.2f}s")
            logger.info(f"Current checkpoint count: {sum(1 for d in current_schedule.values() if d == 'RETAINED')}")
            logger.info(f"Current recompute count: {sum(1 for d in current_schedule.values() if d == 'RECOMPUTE')}")

            # Update best schedule if this one is better
            if current_peak_memory < best_peak_memory:
                best_schedule = current_schedule.copy()
                best_peak_memory = current_peak_memory
                best_exec_time = current_exec_time
                logger.info(f"New best schedule found with peak memory: {best_peak_memory / (1024**3):.2f} GB")

            # Check if we've met the budget
            # Use effective_budget for comparison to ensure we meet the actual budget
            # Check if we've met the original budget (not the effective budget)
            if current_peak_memory <= self.memory_budget_bytes:
                logger.info(f"Memory budget of {self.memory_budget_bytes / (1024**3):.2f} GB met.")
                logger.info(f"Actual memory usage: {current_peak_memory / (1024**3):.2f} GB")
                break # Original budget met
                
            # We'll always try to get as close as possible to the original budget
            # Never exit early even if we've met the effective budget
            # Only exit if we've met the original budget or run out of candidates
            
            # Enhanced candidate selection strategy
            # Try to select multiple candidates at once if we're far from the budget
            # Use the actual requested budget for calculating the gap, not the effective budget
            # This ensures we're always trying to get as close as possible to the user's requested budget
            selection_start = time.time()
            memory_gap = current_peak_memory - self.memory_budget_bytes
            
            # If we're far from the budget, process multiple candidates at once
            batch_candidates = []
            batch_memory_saved = 0
            
            # Process candidates in batches if we're more than 10% over budget
            if memory_gap > 0.1 * (1024**3) and len(candidate_set) > 5:
                # Get top candidates by memory/time ratio
                candidates_with_ratios = []
                for act_name in list(candidate_set)[:min(20, len(candidate_set))]:
                    act_details = self._get_activation_details(act_name)
                    if act_details is None or 'median_mem_size_bytes' not in act_details or 'recomp_time_s' not in act_details:
                        continue
                    
                    mem_size = act_details.get('median_mem_size_bytes', 0)
                    recomp_time = act_details.get('recomp_time_s', 0)
                    
                    if pd.isna(mem_size) or pd.isna(recomp_time) or recomp_time <= 0:
                        continue
                        
                    ratio = mem_size / (recomp_time + 1e-6)
                    candidates_with_ratios.append((act_name, ratio, mem_size, recomp_time))
                
                # Sort by ratio (highest first)
                candidates_with_ratios.sort(key=lambda x: x[1], reverse=True)
                
                # Take top candidates up to 50% of memory gap
                for act_name, ratio, mem_size, recomp_time in candidates_with_ratios:
                    if batch_memory_saved < memory_gap * 0.5 and len(batch_candidates) < 5:
                        batch_candidates.append((act_name, mem_size, recomp_time))
                        batch_memory_saved += mem_size
                        logger.info(f"Batch selecting: {act_name}, ratio: {ratio:.2f}, mem_size: {mem_size/(1024*1024):.2f} MB")
            
            # If batch processing, apply all candidates at once
            if batch_candidates:
                logger.info(f"Batch processing {len(batch_candidates)} candidates to save {batch_memory_saved/(1024*1024):.2f} MB")
                for act_name, mem_size, recomp_time in batch_candidates:
                    logger.info(f"Choosing to RECOMPUTE {act_name} (memory saved: {mem_size/(1024*1024):.2f} MB, recompute overhead: {recomp_time:.6f}s)")
                    current_schedule[act_name] = 'RECOMPUTE'
                    recomps.add(act_name)
                    candidate_set.remove(act_name)
            else:
                # Fall back to single candidate selection
                r_cand = self._get_max_recompute_ratio_candidate(candidate_set)
                if r_cand is None:
                    logger.warning(f"No valid recompute candidate found")
                    break
                
                # Get details for debugging
                r_details = self._get_activation_details(r_cand)
                
                # Calculate memory savings and recompute overhead
                mem_size = r_details.get('median_mem_size_bytes', 0) / (1024 * 1024)  # Convert to MB
                recomp_time = r_details.get('recomp_time_s', 0)
                recompute_benefit_ratio = r_details.get('median_mem_size_bytes', 0) / (recomp_time + 1e-10)
                
                logger.info(f"Considering activation: {r_cand}, recompute_benefit_ratio: {recompute_benefit_ratio:.2f}, mem_size: {mem_size:.2f} MB, recomp_time: {recomp_time:.6f}s")
                
                # Always choose to recompute to save memory
                logger.info(f"Choosing to RECOMPUTE {r_cand} (memory saved: {mem_size:.2f} MB, recompute overhead: {recomp_time:.6f}s)")
                current_schedule[r_cand] = 'RECOMPUTE'
                recomps.add(r_cand)
                
                # Remove chosen candidate from set
                candidate_set.remove(r_cand)
                
            # Record timing for candidate selection in this iteration
            selection_time = time.time() - selection_start
            timing_stats['candidate_selection'] += selection_time
            
            # Record total time for this iteration
            iter_time = time.time() - iter_start_time
            timing_stats['main_loop_iterations'].append({
                'iteration': iteration,
                'time': iter_time,
                'memory_simulation': sim_time,
                'candidate_selection': selection_time
            })
            
            # If we've run out of candidates but still over budget
            if not candidate_set and current_peak_memory > self.memory_budget_bytes:
                logger.warning("No candidates left to process, but still over requested budget.")
                
                # Calculate how close we got to the budget
                memory_gap = current_peak_memory - self.memory_budget_bytes
                logger.warning(f"Gap to requested budget: {memory_gap / (1024**3):.2f} GB")
                
                # Check if we're close to the incompressible memory
                gap_to_incompressible = current_peak_memory - incompressible_memory
                logger.warning(f"Gap to incompressible memory: {gap_to_incompressible / (1024**3):.2f} GB")
                
                # Always try more aggressive recomputation to get as close as possible to the budget
                logger.warning("Attempting more aggressive recomputation to get closer to the requested budget...")
                    
                # Find all remaining RETAINED activations for potential recomputation
                remaining_activations = []
                for act_name in self.activation_stats_df['activation_name']:
                    if act_name not in current_schedule:
                        continue
                        
                    if current_schedule[act_name] == 'RETAINED':
                        act_details = self._get_activation_details(act_name)
                        if act_details is not None and 'median_mem_size_bytes' in act_details:
                            mem_size = act_details['median_mem_size_bytes']
                            recomp_time = act_details.get('recomp_time_s', 0)
                            if pd.notna(mem_size) and mem_size > 0:
                                # Include even activations with very small recomputation time
                                # Use a minimum recomputation time to avoid division by zero
                                effective_recomp_time = max(recomp_time, 1e-6) if pd.notna(recomp_time) else 1e-6
                                ratio = mem_size / effective_recomp_time
                                remaining_activations.append((act_name, mem_size, recomp_time, ratio))
                    
                # Sort by ratio (best memory/time tradeoff first)
                remaining_activations.sort(key=lambda x: x[3], reverse=True)
                
                # Mark additional activations for recomputation
                additional_marked = 0
                additional_memory_saved = 0
                for act_name, mem_size, recomp_time, ratio in remaining_activations[:min(100, len(remaining_activations))]:
                    current_schedule[act_name] = 'RECOMPUTE'
                    additional_marked += 1
                    additional_memory_saved += mem_size
                    logger.info(f"Aggressively marking {act_name} for RECOMPUTE (size: {mem_size/(1024*1024):.2f} MB, ratio: {ratio:.2f})")
                    
                    # Check if we've saved enough memory
                    if additional_memory_saved > memory_gap * 1.1:  # Add 10% margin
                        break
                    
                    logger.info(f"Marked {additional_marked} additional activations for recomputation")
                    logger.info(f"Additional memory saved: {additional_memory_saved / (1024**3):.2f} GB")
                    
                    # Run one more memory simulation
                    final_sim_start = time.time()
                    final_peak_memory, final_exec_time = self._simulate_memory_usage(
                        current_schedule,
                        fixed_overhead_bytes,
                        debug=debug
                    )
                    timing_stats['final_memory_simulation'] = time.time() - final_sim_start
                    
                    logger.info(f"After aggressive recomputation: peak memory = {final_peak_memory / (1024**3):.2f} GB")
                    if final_peak_memory <= self.memory_budget_bytes:
                        logger.info("Memory budget met after aggressive recomputation!")
                    else:
                        logger.warning(f"Still over budget by {(final_peak_memory - self.memory_budget_bytes) / (1024**3):.2f} GB")
                        
                        if gap_to_incompressible < 0.1 * (1024**3):  # Within 100MB of incompressible
                            logger.warning("Current memory usage is very close to the incompressible minimum.")
                            logger.warning("Further reduction is likely not possible through activation checkpointing.")
                # Run one more memory simulation
                final_sim_start = time.time()
                final_peak_memory, final_exec_time = self._simulate_memory_usage(
                    current_schedule,
                    fixed_overhead_bytes,
                    debug=debug
                )
                timing_stats['final_memory_simulation'] = time.time() - final_sim_start
                
                logger.info(f"After aggressive recomputation: peak memory = {final_peak_memory / (1024**3):.2f} GB")
                if final_peak_memory <= self.memory_budget_bytes:
                    logger.info("Memory budget met after aggressive recomputation!")
                else:
                    logger.warning(f"Still over budget by {(final_peak_memory - self.memory_budget_bytes) / (1024**3):.2f} GB")
                    
                    if gap_to_incompressible < 0.1 * (1024**3):  # Within 100MB of incompressible
                        logger.warning("Current memory usage is very close to the incompressible minimum.")
                        logger.warning("Further reduction is likely not possible through activation checkpointing.")
                break

        # End timing for main loop
        timing_stats['main_loop'] = time.time() - main_loop_start
        
        # Check if we timed out
        elapsed_time = time.time() - start_time
        if elapsed_time > timeout_seconds:
            logger.warning(f"Algorithm timed out after {elapsed_time:.1f} seconds.")
            logger.warning(f"Returning best schedule found so far with {sum(1 for d in best_schedule.values() if d == 'RETAINED')} checkpoints and {sum(1 for d in best_schedule.values() if d == 'RECOMPUTE')} recomputes.")
            current_schedule = best_schedule
        
        # Final memory simulation
        final_sim_start = time.time()
        final_peak_memory, final_exec_time = self._simulate_memory_usage(
            current_schedule,
            fixed_overhead_bytes,
            debug=debug
        )
        timing_stats['final_memory_simulation'] = time.time() - final_sim_start
        
        # Report final results
        logger.info(f"\nFinal Results:")
        logger.info(f"Initial peak memory: {initial_peak_memory / (1024**3):.2f} GB")
        logger.info(f"Final peak memory: {final_peak_memory / (1024**3):.2f} GB")
        logger.info(f"Memory reduction: {(initial_peak_memory - final_peak_memory) / (1024**3):.2f} GB ({(initial_peak_memory - final_peak_memory) / initial_peak_memory * 100:.1f}%)")
        logger.info(f"Memory budget: {self.memory_budget_bytes / (1024**3):.2f} GB")
        logger.info(f"Gap to budget: {(final_peak_memory - self.memory_budget_bytes) / (1024**3):.2f} GB")
        logger.info(f"Incompressible memory: {incompressible_memory / (1024**3):.2f} GB")
        logger.info(f"Initial execution time: {initial_exec_time:.2f}s")
        logger.info(f"Final execution time: {final_exec_time:.2f}s")
        logger.info(f"Execution time overhead: {(final_exec_time - initial_exec_time):.2f}s ({(final_exec_time - initial_exec_time) / initial_exec_time * 100:.1f}%)")
        
        self.schedule = current_schedule
        
        # Save the schedule to a CSV file for easier analysis
        self._save_schedule_to_csv(current_schedule)
        
        # Calculate total execution time and update timing stats
        timing_stats['total'] = time.time() - overall_start_time
        
        # Log timing statistics
        logger.info("\nTiming Statistics:")
        logger.info(f"Total execution time: {timing_stats['total']:.2f}s")
        logger.info(f"Initialization: {timing_stats['initialization']:.2f}s ({timing_stats['initialization']/timing_stats['total']*100:.1f}%)")
        logger.info(f"Initial memory simulation: {timing_stats['initial_memory_simulation']:.2f}s ({timing_stats['initial_memory_simulation']/timing_stats['total']*100:.1f}%)")
        logger.info(f"Memory component analysis: {timing_stats['memory_component_analysis']:.2f}s ({timing_stats['memory_component_analysis']/timing_stats['total']*100:.1f}%)")
        logger.info(f"Candidate filtering: {timing_stats['candidate_filtering']:.2f}s ({timing_stats['candidate_filtering']/timing_stats['total']*100:.1f}%)")
        logger.info(f"Main loop: {timing_stats['main_loop']:.2f}s ({timing_stats['main_loop']/timing_stats['total']*100:.1f}%)")
        logger.info(f"  - Memory simulations: {timing_stats['memory_simulations']:.2f}s ({timing_stats['memory_simulations']/timing_stats['total']*100:.1f}%)")
        logger.info(f"  - Candidate selection: {timing_stats['candidate_selection']:.2f}s ({timing_stats['candidate_selection']/timing_stats['total']*100:.1f}%)")
        logger.info(f"Final memory simulation: {timing_stats['final_memory_simulation']:.2f}s ({timing_stats['final_memory_simulation']/timing_stats['total']*100:.1f}%)")
        
        # Find the slowest iterations
        if timing_stats['main_loop_iterations']:
            sorted_iterations = sorted(timing_stats['main_loop_iterations'], key=lambda x: x['time'], reverse=True)
            logger.info("\nTop 5 slowest iterations:")
            for i, iter_data in enumerate(sorted_iterations[:5]):
                logger.info(f"  {i+1}. Iteration {iter_data['iteration']}: {iter_data['time']:.2f}s (Simulation: {iter_data['memory_simulation']:.2f}s, Selection: {iter_data['candidate_selection']:.2f}s)")
        
        return self.schedule
        
    def _analyze_memory_components(self, current_schedule, fixed_overhead_bytes, debug=False):
        """
        Analyze memory components to determine incompressible memory.
        
        Args:
            current_schedule (dict): Activation name to decision ('RETAINED', 'RECOMPUTE').
            fixed_overhead_bytes (float): Estimated memory for parameters, gradients, optimizer.
            debug (bool): Whether to print detailed debug information.
            
        Returns:
            tuple: (fw_active_mem, bw_active_mem, fw_inter_mem, bw_inter_mem)
        """
        # Start timing the analysis
        analysis_start_time = time.time()
        
        # Initialize timing dictionary for detailed breakdown
        analysis_timing = {
            'initialization': 0,
            'execution_order': 0,
            'node_details_cache': 0,
            'active_memory_analysis': 0,
            'intermediate_memory_calculation': 0,
            'total': 0
        }
        
        init_start = time.time()
        # Initialize memory tracking variables
        fw_active_mem = 0
        bw_active_mem = 0
        fw_inter_mem = 0
        bw_inter_mem = 0
        analysis_timing['initialization'] = time.time() - init_start
        
        # Get execution order
        exec_order_start = time.time()
        execution_order = self._get_node_execution_order()
        analysis_timing['execution_order'] = time.time() - exec_order_start
        
        if not execution_order:
            logger.warning("Could not determine execution order.")
            return fw_active_mem, bw_active_mem, fw_inter_mem, bw_inter_mem
        
        # Pre-fetch all node details to avoid repeated lookups
        cache_start = time.time()
        node_details_cache = {}
        for node_name in execution_order:
            node_details_cache[node_name] = self._get_node_details(node_name)
        analysis_timing['node_details_cache'] = time.time() - cache_start
        
        # Find maximum active memory for forward and backward passes
        active_mem_start = time.time()
        for node_name in execution_order:
            node_details = node_details_cache[node_name]
            if node_details is None:
                continue
            
            node_gtype = node_details['gtype']
            
            if node_gtype == 'forward':
                if 'median_active_mem_bytes' in node_details and pd.notna(node_details['median_active_mem_bytes']):
                    fw_active_mem = max(fw_active_mem, node_details['median_active_mem_bytes'])
            elif node_gtype == 'backward':
                if 'median_active_mem_bytes' in node_details and pd.notna(node_details['median_active_mem_bytes']):
                    bw_active_mem = max(bw_active_mem, node_details['median_active_mem_bytes'])
        analysis_timing['active_memory_analysis'] = time.time() - active_mem_start
        
        # Calculate intermediate memory
        inter_mem_start = time.time()
        for act_name, decision in current_schedule.items():
            act_details = self._get_activation_details(act_name)
            if act_details is None or 'median_mem_size_bytes' not in act_details:
                continue
                
            mem_size = act_details['median_mem_size_bytes']
            if pd.isna(mem_size):
                continue
                
            if decision == 'RETAINED':
                bw_inter_mem += mem_size
        analysis_timing['intermediate_memory_calculation'] = time.time() - inter_mem_start
        
        # Calculate total analysis time
        analysis_timing['total'] = time.time() - analysis_start_time
        
        if debug:
            logger.info(f"Memory components analysis:")
            logger.info(f"  FW active memory: {fw_active_mem / (1024**3):.2f} GB")
            logger.info(f"  BW active memory: {bw_active_mem / (1024**3):.2f} GB")
            logger.info(f"  FW intermediate memory: {fw_inter_mem / (1024**3):.2f} GB")
            logger.info(f"  BW intermediate memory: {bw_inter_mem / (1024**3):.2f} GB")
            logger.info(f"  Fixed overhead: {fixed_overhead_bytes / (1024**3):.2f} GB")
            logger.info(f"  Total incompressible memory: {(fw_active_mem + bw_active_mem + fixed_overhead_bytes) / (1024**3):.2f} GB")
            
            # Log timing information for the analysis
            logger.info(f"Memory analysis timing breakdown:")
            logger.info(f"  Total analysis time: {analysis_timing['total']:.4f}s")
            logger.info(f"  Initialization: {analysis_timing['initialization']:.4f}s ({analysis_timing['initialization']/analysis_timing['total']*100:.1f}%)")
            logger.info(f"  Execution order: {analysis_timing['execution_order']:.4f}s ({analysis_timing['execution_order']/analysis_timing['total']*100:.1f}%)")
            logger.info(f"  Node details cache: {analysis_timing['node_details_cache']:.4f}s ({analysis_timing['node_details_cache']/analysis_timing['total']*100:.1f}%)")
            logger.info(f"  Active memory analysis: {analysis_timing['active_memory_analysis']:.4f}s ({analysis_timing['active_memory_analysis']/analysis_timing['total']*100:.1f}%)")
            logger.info(f"  Intermediate memory calculation: {analysis_timing['intermediate_memory_calculation']:.4f}s ({analysis_timing['intermediate_memory_calculation']/analysis_timing['total']*100:.1f}%)")
        
        return fw_active_mem, bw_active_mem, fw_inter_mem, bw_inter_mem

    def _get_max_recompute_ratio_candidate(self, candidate_set):
        """
        Select the candidate with maximum recompute benefit ratio (memory_size / recompute_time).
        
        Args:
            candidate_set (set): Set of candidate activation names.
            
        Returns:
            str: Name of the candidate with maximum recompute benefit ratio.
        """
        max_ratio = -1
        max_candidate = None
        
        for act_name in candidate_set:
            act_details = self._get_activation_details(act_name)
            if act_details is None:
                continue
                
            if 'median_mem_size_bytes' not in act_details or 'recomp_time_s' not in act_details:
                continue
                
            mem_size = act_details['median_mem_size_bytes']
            recomp_time = act_details['recomp_time_s']
            
            if pd.isna(mem_size) or pd.isna(recomp_time) or recomp_time <= 0:
                continue
                
            # Calculate recompute benefit ratio (memory saved / recomputation time)
            # No discrimination against small bytes - consider all activations
            ratio = mem_size / (recomp_time + 1e-6)
            
            if ratio > max_ratio:
                max_ratio = ratio
                max_candidate = act_name
                
        return max_candidate

    def _update_recomps(self, cand, recomps):
        """
        Update recomputation counts and dependencies.
        
        Args:
            cand (str): The candidate that was chosen.
            recomps (set): Set of activations marked for recomputation.
            
        Returns:
            int: Number of times the candidate will be recomputed.
        """
        # Get activation details
        act_details = self._get_activation_details(cand)
        if act_details is None:
            return 1
            
        # In a more complex implementation, we would track dependencies between activations
        # and calculate how many times this activation needs to be recomputed based on
        # its dependencies with other activations marked for recomputation
        
        # For now, we assume each activation is recomputed once
        # This could be extended to handle more complex dependency tracking
        return 1

    def _update_candidates(self, cand, recomp_cnt, candidate_set):
        """
        Update remaining candidates based on the chosen candidate.
        
        Args:
            cand (str): The candidate that was chosen.
            recomp_cnt (int): Number of times the candidate will be recomputed.
            candidate_set (set): Set of remaining candidates.
        """
        # In a more complex implementation, we would update the recomputation times
        # and memory savings of remaining candidates based on the chosen candidate
        
        # For example, if two activations share computation, recomputing one might
        # reduce the recomputation cost of the other
        
        # This is a placeholder for future improvements
        # For now, we don't update other candidates
        pass


if __name__ == "__main__":
    # Start overall timing
    script_start_time = time.time()
    
    # Example usage
    node_stats_file = "reports/profiler_stats_bs4_node_stats.csv"
    activation_stats_file = "reports/profiler_stats_bs4_activation_stats.csv"
    memory_budget = 4.0  # GB
    
    # Log startup information
    logger.info("=" * 80)
    logger.info("Activation Checkpointing Algorithm Starting")
    logger.info(f"Log file: {log_path}")

    # Parse command line arguments
    import argparse
    
    parser = argparse.ArgumentParser(description='Activation Checkpointing Algorithm')
    parser.add_argument('--node-stats', type=str, default=node_stats_file,
                        help='Path to node statistics CSV file')
    parser.add_argument('--activation-stats', type=str, default=activation_stats_file,
                        help='Path to activation statistics CSV file')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    parser.add_argument('--batch-size', type=int, default=50, help='Number of activations to evict per iteration')
    parser.add_argument('--max-iterations', type=int, default=1000, help='Maximum number of iterations')
    parser.add_argument('--memory-budget', type=float, default=memory_budget, help='Memory budget in GB')
    parser.add_argument('--fixed-overhead', type=float, default=0.3, help='Fixed overhead in GB')
    args = parser.parse_args()

    try:
        # Initialize the algorithm
        init_start_time = time.time()
        ac_algo = ActivationCheckpointingAlgorithm(
            node_stats_path=args.node_stats,
            activation_stats_path=args.activation_stats,
            memory_budget_gb=args.memory_budget
        )
        init_time = time.time() - init_start_time
        logger.info(f"Algorithm initialization time: {init_time:.2f}s")
        
        # Run the algorithm with increased timeout for more thorough search
        final_schedule = ac_algo.decide_checkpoints(
            fixed_overhead_gb=args.fixed_overhead,
            debug=args.debug,
            batch_size=args.batch_size,
            max_iterations=args.max_iterations,
            timeout_seconds=120  # Increased timeout to 2 minutes
        )
        
        # Print summary
        recomputed_count = sum(1 for decision in final_schedule.values() if decision == 'RECOMPUTE')
        checkpointed_count = sum(1 for decision in final_schedule.values() if decision == 'RETAINED')
        
        print(f"\nSummary:")
        print(f"Total activations considered: {len(final_schedule)}")
        print(f"Number of activations to RECOMPUTE: {recomputed_count}")
        print(f"Number of activations to RETAINED: {checkpointed_count}")
        
        # Run final simulation
        final_peak_mem, final_exec_time = ac_algo._simulate_memory_usage(
            final_schedule,
            fixed_overhead_bytes=args.fixed_overhead * (1024**3),
            debug=args.debug
        )
        
        print(f"Estimated Peak GPU Memory with schedule: {final_peak_mem / (1024**3):.2f} GB")
        print(f"Estimated Total Execution Time with schedule: {final_exec_time:.2f} s")
        
        # Calculate and print total script execution time
        script_execution_time = time.time() - script_start_time
        print(f"\nTotal script execution time: {script_execution_time:.2f} seconds")
        
        # Print timing breakdown
        print("\nTiming Breakdown:")
        print(f"  Initialization: {init_time:.2f}s ({init_time/script_execution_time*100:.1f}%)")
        algorithm_time = script_execution_time - init_time
        print(f"  Algorithm execution: {algorithm_time:.2f}s ({algorithm_time/script_execution_time*100:.1f}%)")
        
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()