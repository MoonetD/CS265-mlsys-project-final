import pandas as pd
import numpy as np
import time

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
        except FileNotFoundError as e:
            print(f"Error: One or both profiler statistics files not found: {e}")
            raise
        except Exception as e:
            print(f"Error loading CSV files: {e}")
            raise

        self.memory_budget_bytes = memory_budget_gb * (1024**3)
        self.schedule = {} # To store decisions: act_name -> 'RETAINED' or 'RECOMPUTE'

        # Preprocess data if necessary
        # Ensure 'activation_name' is suitable as a dictionary key and is the index
        if 'activation_name' in self.activation_stats_df.columns:
            self.activation_stats_df.set_index('activation_name', inplace=True, drop=False)
        else:
            raise ValueError("activation_stats_df must contain an 'activation_name' column.")

        if 'node_name' in self.node_stats_df.columns:
            self.node_stats_df.set_index('node_name', inplace=True, drop=False)
        else:
            raise ValueError("node_stats_df must contain a 'node_name' column.")


    def _calculate_recompute_overhead(self, activation_name):
        """
        Calculates the recomputation time overhead for a given activation.
        Corresponds to parts of Algorithm D, E, F from the paper.
        """
        if activation_name not in self.activation_stats_df.index:
            # print(f"Warning: Activation {activation_name} not found in stats for recompute overhead.")
            return 0, 0 # Time, Memory
        
        recomp_time = self.activation_stats_df.loc[activation_name, 'recomp_time_s']
        # recomp_memory = self.activation_stats_df.loc[activation_name, 'recomp_memory_bytes'] # Memory during recomputation
        act_memory = self.activation_stats_df.loc[activation_name, 'median_mem_size_bytes'] # Memory of the activation itself
        return recomp_time if pd.notna(recomp_time) else 0, act_memory if pd.notna(act_memory) else 0


    def _get_node_execution_order(self):
        """
        Returns a list of node names in their execution order (rank).
        Includes both forward and backward pass nodes.
        """
        # Assuming 'rank' column exists and defines unique execution order
        # NaN ranks should be handled or filtered if they exist
        sorted_nodes = self.node_stats_df.dropna(subset=['rank']).sort_values(by='rank')
        return sorted_nodes['node_name'].tolist()

    def _get_activation_details(self, activation_name):
        """Helper to get all details for an activation."""
        if activation_name not in self.activation_stats_df.index:
            return None
        return self.activation_stats_df.loc[activation_name]

    def _get_node_details(self, node_name):
        """Helper to get all details for a node."""
        if node_name not in self.node_stats_df.index:
            return None
        return self.node_stats_df.loc[node_name]

    def _simulate_memory_usage(self, current_schedule, fixed_overhead_bytes=0, debug=False):
        """
        Simulates peak memory usage and total execution time based on a given schedule.
        Implements Algorithm G from the μ-TWO paper.
        
        This is the core of activation checkpointing memory simulation. It tracks:
        1. Which activations are kept in memory vs. recomputed
        2. When activations are created and when they're last used
        3. The memory impact of discarding activations during forward pass
        4. The computational cost of recomputing activations during backward pass
        
        The key insight is that by discarding some activations during forward pass
        and recomputing them during backward pass, we can reduce peak memory usage
        at the cost of increased computation time.

        Args:
            current_schedule (dict): Activation name to decision ('RETAINED', 'RECOMPUTE').
            fixed_overhead_bytes (float): Estimated memory for parameters, gradients, optimizer.
            debug (bool): Whether to print detailed debug information.

        Returns:
            float: Estimated peak GPU memory in bytes.
            float: Total execution time in seconds.
        """
        if debug:
            print(f"Starting memory simulation with {len(current_schedule)} activations...")
            print(f"Fixed overhead: {fixed_overhead_bytes / (1024**3):.3f} GB")
            
        # --- Calculate Total Execution Time ---
        # Start with sum of all base node run times
        if debug:
            print("Calculating total execution time...")
            
        total_execution_time = self.node_stats_df['median_run_time_s'].sum()
        
        # Add recomputation times for activations scheduled for RECOMPUTE
        recompute_count = 0
        recompute_time_total = 0.0
        
        for act_name, decision in current_schedule.items():
            if decision == 'RECOMPUTE':
                act_details = self._get_activation_details(act_name)
                # Ensure act_details is a Series and 'recomp_time' exists
                if act_details is not None and isinstance(act_details, pd.Series) and \
                   'recomp_time_s' in act_details.index and pd.notna(act_details['recomp_time_s']):
                    recomp_time = act_details['recomp_time_s']
                    total_execution_time += recomp_time
                    recompute_time_total += recomp_time
                    recompute_count += 1

        if debug:
            print(f"Base execution time: {total_execution_time - recompute_time_total:.4f}s")
            print(f"Added recomputation time for {recompute_count} activations: {recompute_time_total:.4f}s")
            print(f"Total execution time: {total_execution_time:.4f}s")

        # --- Simulate Peak Memory Usage (Algorithm G) ---
        if debug:
            print("Simulating peak memory usage...")
            
        # Initialize memory tracking variables
        fw_inter_mem = 0  # Memory for intermediate tensors in forward pass
        bw_inter_mem = 0  # Memory for intermediate tensors in backward pass
        fw_active_mem = 0  # Active memory during forward pass
        bw_active_mem = 0  # Active memory during backward pass
        peak_mem = fixed_overhead_bytes  # Track peak memory, start with fixed overhead
        
        # For activation checkpointing, we only keep RETAINED activations in memory
        # RECOMPUTE activations are discarded during forward pass and recomputed during backward pass
        
        # Calculate initial bw_inter_mem (all checkpointed activations)
        # This is a key difference from regular execution - we only keep RETAINED activations
        for act_name, decision in current_schedule.items():
            if decision == 'RETAINED':  # Only count activations we're keeping in memory
                act_details = self._get_activation_details(act_name)
                if act_details is not None and 'median_mem_size_bytes' in act_details:
                    bw_inter_mem += act_details['median_mem_size_bytes']
        
        if debug:
            print(f"Initial intermediate memory: {bw_inter_mem / (1024**3):.3f} GB")
            print(f"Fixed overhead: {fixed_overhead_bytes / (1024**3):.3f} GB")
            print(f"Starting peak memory: {peak_mem / (1024**3):.3f} GB")
        
        # Get execution order
        execution_order = self._get_node_execution_order()
        if not execution_order:
            if debug:
                print("Warning: Could not determine execution order.")
            return float('inf'), total_execution_time
        
        # Pre-fetch all node details to avoid repeated lookups
        node_details_cache = {}
        for node_name in execution_order:
            node_details_cache[node_name] = self._get_node_details(node_name)
        
        # Create a mapping from rank to activations created/used at that rank
        activations_by_creation_rank = {}
        activations_by_first_bw_use_rank = {}
        activations_by_last_fw_use_rank = {}
        
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
            
            # Map by last forward use rank
            if pd.notna(act_details_series['last_fw_use_rank']):
                rank = int(act_details_series['last_fw_use_rank'])
                if rank not in activations_by_last_fw_use_rank:
                    activations_by_last_fw_use_rank[rank] = []
                activations_by_last_fw_use_rank[rank].append(act_details_series)
        
        # Process nodes in execution order
        total_nodes = len(execution_order)
        if debug:
            print(f"Processing {total_nodes} nodes in execution order...")
        
        for i, node_name in enumerate(execution_order):
            if debug and i % max(1, total_nodes // 10) == 0:
                print(f"  Processing node {i+1}/{total_nodes} ({(i+1)/total_nodes*100:.1f}%)...")
            
            node_details = node_details_cache[node_name]
            if node_details is None:
                continue
            
            node_rank = node_details['rank']
            node_gtype = node_details['gtype']
            
            # Update active memory based on node type
            if node_gtype == 'bw':
                if 'avg_active_mem' in node_details and pd.notna(node_details['avg_active_mem']):
                    bw_active_mem = node_details['avg_active_mem']
                
                # Add memory for prefetched tensors
                if node_rank in activations_by_first_bw_use_rank:
                    for act_details in activations_by_first_bw_use_rank[node_rank]:
                        act_name = act_details['activation_name']
                        if current_schedule.get(act_name) == 'RETAINED':
                            if 'median_mem_size_bytes' in act_details and pd.notna(act_details['median_mem_size_bytes']):
                                bw_inter_mem += act_details['median_mem_size_bytes']
                
                # Add memory for recomputed tensors
                # This is where we account for the memory cost of recomputation
                # When we reach a node that needs a recomputed activation,
                # we add its memory size to the intermediate memory
                if node_rank in activations_by_first_bw_use_rank:
                    for act_details in activations_by_first_bw_use_rank[node_rank]:
                        act_name = act_details['activation_name']
                        if current_schedule.get(act_name) == 'RECOMPUTE':
                            if 'median_mem_size_bytes' in act_details and pd.notna(act_details['median_mem_size_bytes']):
                                # Add the memory for this recomputed activation
                                mem_size = act_details['median_mem_size_bytes']
                                bw_inter_mem += mem_size
                                if debug and mem_size > 1024*1024:  # Only log significant activations (>1MB)
                                    print(f"  Added {mem_size/(1024**2):.2f} MB for recomputed activation {act_name}")
            
            elif node_gtype == 'fw':
                if 'avg_active_mem' in node_details and pd.notna(node_details['avg_active_mem']):
                    fw_active_mem = node_details['avg_active_mem']
                
                # Add memory for newly created tensors
                # In the forward pass, we only keep RETAINED activations in memory
                # RECOMPUTE activations are discarded immediately
                if node_rank in activations_by_creation_rank:
                    for act_details in activations_by_creation_rank[node_rank]:
                        act_name = act_details['activation_name']
                        if current_schedule.get(act_name) == 'RETAINED':
                            # Only add memory for activations we're keeping (not recomputing)
                            if 'median_mem_size_bytes' in act_details and pd.notna(act_details['median_mem_size_bytes']):
                                mem_size = act_details['median_mem_size_bytes']
                                fw_inter_mem += mem_size
                                if debug and mem_size > 1024*1024:  # Only log significant activations (>1MB)
                                    print(f"  Added {mem_size/(1024**2):.2f} MB for retained activation {act_name}")
                        else:
                            # For RECOMPUTE activations, we don't keep them in memory
                            # This is the key memory savings of activation checkpointing
                            if debug and 'median_mem_size_bytes' in act_details and pd.notna(act_details['median_mem_size_bytes']):
                                mem_size = act_details['median_mem_size_bytes']
                                if mem_size > 1024*1024:  # Only log significant activations (>1MB)
                                    print(f"  Discarded {mem_size/(1024**2):.2f} MB for activation {act_name} (will recompute)")
                
                # Remove memory for tensors that are no longer needed
                if node_rank in activations_by_last_fw_use_rank:
                    for act_details in activations_by_last_fw_use_rank[node_rank]:
                        act_name = act_details['activation_name']
                        if current_schedule.get(act_name) == 'RETAINED':
                            if 'median_mem_size_bytes' in act_details and pd.notna(act_details['median_mem_size_bytes']):
                                fw_inter_mem -= act_details['median_mem_size_bytes']
            
            # Calculate current memory consumption
            current_mem = fw_active_mem + bw_active_mem + fw_inter_mem + bw_inter_mem + fixed_overhead_bytes
            
            # Update peak memory
            if current_mem > peak_mem:
                peak_mem = current_mem
                if debug:
                    print(f"  New peak memory: {peak_mem/(1024**3):.3f} GB at node {node_name} (rank {node_rank})")
                    print(f"    Breakdown: FW active={fw_active_mem/(1024**3):.3f} GB, BW active={bw_active_mem/(1024**3):.3f} GB")
                    print(f"    FW inter={fw_inter_mem/(1024**3):.3f} GB, BW inter={bw_inter_mem/(1024**3):.3f} GB")
                    print(f"    Fixed overhead={fixed_overhead_bytes/(1024**3):.3f} GB")
            
            # Also consider the peak memory during this specific operation
            if 'median_peak_mem_node' in node_details and pd.notna(node_details['median_peak_mem_node']):
                node_peak = node_details['median_peak_mem_node'] + fixed_overhead_bytes
                if node_peak > peak_mem:
                    peak_mem = node_peak
                    if debug:
                        print(f"  New peak memory from node operation: {peak_mem/(1024**3):.3f} GB at node {node_name}")
        
        if debug:
            print(f"Memory simulation complete.")
            print(f"Peak memory: {peak_mem / (1024**3):.3f} GB")
            print(f"Final execution time: {total_execution_time:.4f}s")
            print(f"Memory savings from activation checkpointing: {sum(act_details['median_mem_size_bytes'] for act_name, decision in current_schedule.items() if decision == 'RECOMPUTE' for act_details in [self._get_activation_details(act_name)] if act_details is not None and 'median_mem_size_bytes' in act_details) / (1024**3):.3f} GB")
            print(f"Computation overhead from recomputation: {recompute_time_total:.4f}s")
        
        return peak_mem, total_execution_time


    def decide_checkpoints(self, fixed_overhead_gb=2.0, debug=False, batch_size=50, max_iterations=1000, timeout_seconds=60):
        """
        Decides which activations to checkpoint and which to recompute.
        Implements Algorithm B from the μ-TWO paper.

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
        print(f"Starting checkpoint decision algorithm...")
        print(f"Fixed overhead: {fixed_overhead_gb} GB, Memory budget: {self.memory_budget_bytes / (1024**3):.2f} GB")
        
        fixed_overhead_bytes = fixed_overhead_gb * (1024**3)

        # Initial state: checkpoint all activations
        print("Initializing schedule with all activations checkpointed...")
        current_schedule = {
            act_name: 'RETAINED'
            for act_name in self.activation_stats_df['activation_name'] # Iterate using the correct column name
            if pd.notna(self.activation_stats_df.loc[act_name, 'median_mem_size_bytes']) and self.activation_stats_df.loc[act_name, 'median_mem_size_bytes'] > 0
        }
        
        print(f"Initial schedule has {len(current_schedule)} activations marked for RETAINED")
        
        # Filter out activations with no valid size or recompute stats
        print("Filtering valid activations...")
        valid_activations_df = self.activation_stats_df.dropna(subset=['median_mem_size_bytes', 'recomp_time_s', 'creation_rank', 'last_fw_use_rank'])
        valid_activations_df = valid_activations_df[valid_activations_df['median_mem_size_bytes'] > 0]
        print(f"Found {len(valid_activations_df)} valid activations for consideration")

        # Pre-compute benefit values for all activations to avoid repeated calculations
        print("Pre-computing benefit values...")
        
        # Create a set of candidate activations
        candidate_set = set(valid_activations_df.index)
        
        # Initialize tracking set for recomputation
        recomps = set()
        
        # Main loop of Algorithm B
        iteration = 0
        start_time = time.time()
        while iteration < max_iterations and candidate_set:
            # Check for timeout
            current_time = time.time()
            elapsed_time = current_time - start_time
            if elapsed_time > timeout_seconds:
                print(f"Timeout reached after {elapsed_time:.1f} seconds. Stopping algorithm.")
                break
                
            iteration += 1
            print(f"\nIteration {iteration}/{max_iterations} (Elapsed time: {elapsed_time:.1f}s)")
            
            # Run memory simulation to check current state
            current_peak_memory, current_exec_time = self._simulate_memory_usage(
                current_schedule,
                fixed_overhead_bytes,
                debug=(iteration == 1 and debug)
            )
            
            print(f"Simulated peak memory: {current_peak_memory / (1024**3):.2f} GB. Budget: {self.memory_budget_bytes / (1024**3):.2f} GB. Exec time: {current_exec_time:.2f}s")
            print(f"Current checkpoint count: {sum(1 for d in current_schedule.values() if d == 'RETAINED')}")
            print(f"Current recompute count: {sum(1 for d in current_schedule.values() if d == 'RECOMPUTE')}")

            # Check if we've met the budget
            if current_peak_memory <= self.memory_budget_bytes:
                print("Memory budget met.")
                break # Budget met
            
            # Process a batch of candidates at a time for efficiency
            candidates_to_process = list(candidate_set)[:batch_size]
            
            print(f"Processing {len(candidates_to_process)} candidates in this batch")
            
            for act_name in candidates_to_process:
                # For activation checkpointing, we only consider recomputation
                # Select candidate with maximum memory savings potential (memory size / recompute time)
                r_cand = self._get_max_recompute_ratio_candidate(candidate_set)
                if r_cand is None:
                    print(f"DEBUG: No valid recompute candidate found")
                    continue
                
                # Get details for debugging
                r_details = self._get_activation_details(r_cand)
                
                # Calculate memory savings and recompute overhead
                mem_size = r_details.get('median_mem_size_bytes', 0) / (1024 * 1024)  # Convert to MB
                recomp_time = r_details.get('recomp_time_s', 0)
                recompute_benefit_ratio = r_details.get('median_mem_size_bytes', 0) / (recomp_time + 1e-10)
                
                print(f"DEBUG: Considering activation: {r_cand}, recompute_benefit_ratio: {recompute_benefit_ratio:.2f}, mem_size: {mem_size:.2f} MB, recomp_time: {recomp_time:.6f}s")
                
                # Always choose to recompute to save memory
                # This is the core of activation checkpointing - we discard activations during forward pass
                # and recompute them during backward pass when needed
                print(f"Choosing to RECOMPUTE {r_cand} (memory saved: {mem_size:.2f} MB, recompute overhead: {recomp_time:.6f}s)")
                current_schedule[r_cand] = 'RECOMPUTE'
                recomps.add(r_cand)
                cand = r_cand
                
                # Remove chosen candidate from set
                candidate_set.remove(cand)
                
                # Update recomputation counts and dependencies
                recomp_cnt = self._update_recomps(cand, recomps)
                
                # Update remaining candidates based on this decision
                self._update_candidates(cand, recomp_cnt, candidate_set)
                
                # Check if memory budget is met after this decision
                current_peak_memory, _ = self._simulate_memory_usage(current_schedule, fixed_overhead_bytes)
                if current_peak_memory <= self.memory_budget_bytes:
                    print("Memory budget met after processing candidate.")
                    break
            
            # If we've run out of candidates but still over budget
            if not candidate_set and current_peak_memory > self.memory_budget_bytes:
                print("Warning: No candidates left to process, but still over budget.")
                break

        # Check if we timed out
        elapsed_time = time.time() - start_time
        if elapsed_time > timeout_seconds:
            print(f"Warning: Algorithm timed out after {elapsed_time:.1f} seconds.")
            print(f"Returning best schedule found so far with {sum(1 for d in current_schedule.values() if d == 'RETAINED')} checkpoints and {sum(1 for d in current_schedule.values() if d == 'RECOMPUTE')} recomputes.")
            print(f"This may not be optimal. Consider increasing the timeout or using a higher memory budget.")
        
        self.schedule = current_schedule
        return self.schedule
        
    def _get_max_inactive_time_candidate(self, candidate_set):
        """
        Select the candidate with maximum inactive time.
        """
        max_inactive_time = -1
        max_candidate = None
        
        for act_name in candidate_set:
            act_details = self._get_activation_details(act_name)
            if act_details is None or 'inactive_time_s' not in act_details:
                continue
                
            inactive_time = act_details['inactive_time_s']
            if pd.notna(inactive_time) and inactive_time > max_inactive_time:
                max_inactive_time = inactive_time
                max_candidate = act_name
                
        return max_candidate
        
    def _get_max_recompute_ratio_candidate(self, candidate_set):
        """
        Select the candidate with maximum recompute benefit ratio (memory_size / recompute_time).
        Only consider candidates with significant memory size to make recomputation worthwhile.
        
        This is a key function for activation checkpointing. It identifies which activation
        will give us the best memory savings relative to its recomputation cost.
        
        The recompute_benefit_ratio (memory_size / recompute_time) represents:
        - Higher values = more memory saved per unit of recomputation time
        - Lower values = less memory saved per unit of recomputation time
        
        We want to prioritize activations with high ratios to maximize memory savings
        while minimizing the computational overhead of recomputation.
        """
        max_ratio = -1
        max_candidate = None
        
        print(f"DEBUG: _get_max_recompute_ratio_candidate - Checking {len(candidate_set)} candidates")
        
        # Count candidates with valid recompute ratios
        valid_candidates = 0
        significant_candidates = 0
        
        # Minimum memory size to consider for recomputation (1 MB)
        MIN_MEMORY_SIZE_BYTES = 1 * 1024 * 1024
        
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
                
            valid_candidates += 1
            
            # Skip candidates with very small memory size
            if mem_size < MIN_MEMORY_SIZE_BYTES:
                continue
                
            significant_candidates += 1
            ratio = mem_size / recomp_time
            
            if ratio > max_ratio:
                max_ratio = ratio
                max_candidate = act_name
        
        print(f"DEBUG: _get_max_recompute_ratio_candidate - Found {valid_candidates} valid candidates, {significant_candidates} with significant memory size")
        if max_candidate:
            act_details = self._get_activation_details(max_candidate)
            mem_size = act_details['median_mem_size_bytes'] / (1024 * 1024)  # Convert to MB
            print(f"DEBUG: _get_max_recompute_ratio_candidate - Best candidate: {max_candidate} with ratio: {max_ratio:.2f}, mem_size: {mem_size:.2f} MB")
        else:
            print(f"DEBUG: _get_max_recompute_ratio_candidate - No valid candidate found with significant memory size")
                
        return max_candidate
        
        
        
    def _update_recomps(self, cand, recomps):
        """
        Update recomputation counts and dependencies.
        
        Args:
            cand: The candidate that was chosen
            recomps: Set of activations marked for recomputation
            
        Returns:
            recomp_cnt: Number of times the candidate will be recomputed
        """
        # This is a simplification - in a full implementation, we would need to
        # track dependencies between activations and update recomputation counts
        return 1
        
    def _update_candidates(self, cand, recomp_cnt, candidate_set):
        """
        Update remaining candidates based on the chosen candidate.
        
        Args:
            cand: The candidate that was chosen
            recomp_cnt: Number of times the candidate will be recomputed
            candidate_set: Set of remaining candidates
        """
        # This is a simplification - in a full implementation, we would need to
        # update the recomputation sources and times of remaining candidates
        pass
        

if __name__ == "__main__":
    # Example Usage:
    # These paths should point to your actual CSV files.
    # Ensure these files exist in the specified locations.
    # The relative paths assume this script is run from c:/Users/ydeng/Documents/GitHub/CS265-mlsys-project-final
    node_stats_file = "profiler_stats_node_stats.csv"
    activation_stats_file = "profiler_stats_activation_stats.csv"
    
    # GPU memory budget (e.g., 16 GB for a T4, 24GB for 3090, 40GB for A100)
    # The paper mentions experiments with 12GB, 16GB.
    # ResNet-152 on ImageNet typically needs ~30GB without checkpointing for BS=64.
    # Let's try a budget that forces some recomputation.
    memory_budget = 0.05  # GB

    # Parse command line arguments
    import argparse
    import time
    
    parser = argparse.ArgumentParser(description='Activation Checkpointing Algorithm')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    parser.add_argument('--batch-size', type=int, default=50, help='Number of activations to evict per iteration')
    parser.add_argument('--max-iterations', type=int, default=1000, help='Maximum number of iterations')
    parser.add_argument('--memory-budget', type=float, default=memory_budget, help='Memory budget in GB')
    parser.add_argument('--fixed-overhead', type=float, default=0.1, help='Fixed overhead in GB')
    args = parser.parse_args()

    print(f"Starting Activation Checkpointing Algorithm...")
    print(f"Node Stats: {node_stats_file}")
    print(f"Activation Stats: {activation_stats_file}")
    print(f"Memory Budget: {args.memory_budget} GB")
    print(f"Fixed Overhead: {args.fixed_overhead} GB")
    print(f"Batch Size: {args.batch_size}")
    print(f"Max Iterations: {args.max_iterations}")
    print(f"Debug Mode: {'Enabled' if args.debug else 'Disabled'}")

    try:
        print("Loading data...")
        start_time = time.time()
        
        ac_algo = ActivationCheckpointingAlgorithm(
            node_stats_path=node_stats_file,
            activation_stats_path=activation_stats_file,
            memory_budget_gb=args.memory_budget
        )
        
        load_time = time.time() - start_time
        print(f"Data loaded in {load_time:.2f} seconds")

        # Run the algorithm with the specified parameters
        print("\nRunning checkpoint decision algorithm...")
        start_time = time.time()
        
        final_schedule = ac_algo.decide_checkpoints(
            fixed_overhead_gb=args.fixed_overhead,
            debug=args.debug,
            batch_size=args.batch_size,
            max_iterations=args.max_iterations
        )
        
        algorithm_time = time.time() - start_time
        print(f"\nAlgorithm completed in {algorithm_time:.2f} seconds")
        
        print("\nFinal Activation Checkpointing Schedule:")
        recomputed_count = 0
        checkpointed_count = 0
        for act_name, decision in final_schedule.items():
            if decision == 'RECOMPUTE':
                recomputed_count += 1
            else:
                checkpointed_count +=1
            # print(f"Activation: {act_name}, Decision: {decision}")

        print(f"\nSummary:")
        print(f"Total activations considered: {len(final_schedule)}")
        print(f"Number of activations to RECOMPUTE: {recomputed_count}")
        print(f"Number of activations to RETAINED: {checkpointed_count}")

        # Run final simulation with debug output
        print("\nRunning final memory simulation...")
        final_peak_mem, final_exec_time = ac_algo._simulate_memory_usage(
            final_schedule,
            fixed_overhead_bytes=args.fixed_overhead * (1024**3),
            debug=args.debug
        )
        
        print(f"Estimated Peak GPU Memory with schedule: {final_peak_mem / (1024**3):.2f} GB")
        print(f"Estimated Total Execution Time with schedule: {final_exec_time:.2f} s")
        print(f"Total processing time: {time.time() - start_time + load_time:.2f} seconds")

    except FileNotFoundError:
        print("Execution failed: Ensure profiler CSV files are in the correct path.")
    except ValueError as ve:
        print(f"Execution failed due to data error: {ve}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()