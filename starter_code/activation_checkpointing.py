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
        self.schedule = {} # To store decisions: act_name -> 'CHECKPOINT' or 'RECOMPUTE'

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
        act_memory = self.activation_stats_df.loc[activation_name, 'avg_mem_size_bytes'] # Memory of the activation itself
        return recomp_time if pd.notna(recomp_time) else 0, act_memory if pd.notna(act_memory) else 0

    def _calculate_swap_overhead(self, activation_name):
        """
        Calculates the swap time overhead for a given activation.
        Corresponds to Algorithm C from the paper.
        """
        if activation_name not in self.activation_stats_df.index:
            # print(f"Warning: Activation {activation_name} not found in stats for swap overhead.")
            return 0, 0 # Time, Memory (memory here refers to the activation's own size)
        
        swap_time = self.activation_stats_df.loc[activation_name, 'avg_swap_time_s']
        act_memory = self.activation_stats_df.loc[activation_name, 'avg_mem_size_bytes']
        return swap_time if pd.notna(swap_time) else 0, act_memory if pd.notna(act_memory) else 0

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

        Args:
            current_schedule (dict): Activation name to decision ('CHECKPOINT', 'RECOMPUTE').
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
            
        total_execution_time = self.node_stats_df['avg_run_time_s'].sum()
        
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
        
        # Calculate initial bw_inter_mem (all checkpointed activations)
        for act_name, decision in current_schedule.items():
            if decision == 'CHECKPOINT':
                act_details = self._get_activation_details(act_name)
                if act_details is not None and 'avg_mem_size_bytes' in act_details:
                    bw_inter_mem += act_details['avg_mem_size_bytes']
        
        if debug:
            print(f"Initial intermediate memory: {bw_inter_mem / (1024**3):.3f} GB")
        
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
                        if current_schedule.get(act_name) == 'CHECKPOINT':
                            if 'avg_mem_size_bytes' in act_details and pd.notna(act_details['avg_mem_size_bytes']):
                                bw_inter_mem += act_details['avg_mem_size_bytes']
                
                # Add memory for recomputed tensors
                if node_rank in activations_by_first_bw_use_rank:
                    for act_details in activations_by_first_bw_use_rank[node_rank]:
                        act_name = act_details['activation_name']
                        if current_schedule.get(act_name) == 'RECOMPUTE':
                            if 'avg_mem_size_bytes' in act_details and pd.notna(act_details['avg_mem_size_bytes']):
                                bw_inter_mem += act_details['avg_mem_size_bytes']
            
            elif node_gtype == 'fw':
                if 'avg_active_mem' in node_details and pd.notna(node_details['avg_active_mem']):
                    fw_active_mem = node_details['avg_active_mem']
                
                # Add memory for newly created tensors
                if node_rank in activations_by_creation_rank:
                    for act_details in activations_by_creation_rank[node_rank]:
                        act_name = act_details['activation_name']
                        if current_schedule.get(act_name) == 'CHECKPOINT':
                            if 'avg_mem_size_bytes' in act_details and pd.notna(act_details['avg_mem_size_bytes']):
                                fw_inter_mem += act_details['avg_mem_size_bytes']
                
                # Remove memory for tensors that are no longer needed
                if node_rank in activations_by_last_fw_use_rank:
                    for act_details in activations_by_last_fw_use_rank[node_rank]:
                        act_name = act_details['activation_name']
                        if current_schedule.get(act_name) == 'CHECKPOINT':
                            if 'avg_mem_size_bytes' in act_details and pd.notna(act_details['avg_mem_size_bytes']):
                                fw_inter_mem -= act_details['avg_mem_size_bytes']
            
            # Calculate current memory consumption
            current_mem = fw_active_mem + bw_active_mem + fw_inter_mem + bw_inter_mem
            
            # Update peak memory
            peak_mem = max(peak_mem, current_mem)
            
            # Also consider the peak memory during this specific operation
            if 'avg_peak_mem_node' in node_details and pd.notna(node_details['avg_peak_mem_node']):
                peak_mem = max(peak_mem, node_details['avg_peak_mem_node'])
        
        if debug:
            print(f"Memory simulation complete.")
            print(f"Peak memory: {peak_mem / (1024**3):.3f} GB")
            print(f"Final execution time: {total_execution_time:.4f}s")
        
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
            dict: A schedule mapping activation names to 'CHECKPOINT' or 'RECOMPUTE'.
        """
        print(f"Starting checkpoint decision algorithm...")
        print(f"Fixed overhead: {fixed_overhead_gb} GB, Memory budget: {self.memory_budget_bytes / (1024**3):.2f} GB")
        
        fixed_overhead_bytes = fixed_overhead_gb * (1024**3)

        # Initial state: checkpoint all activations
        print("Initializing schedule with all activations checkpointed...")
        current_schedule = {
            act_name: 'CHECKPOINT'
            for act_name in self.activation_stats_df['activation_name'] # Iterate using the correct column name
            if pd.notna(self.activation_stats_df.loc[act_name, 'avg_mem_size_bytes']) and self.activation_stats_df.loc[act_name, 'avg_mem_size_bytes'] > 0
        }
        
        print(f"Initial schedule has {len(current_schedule)} activations marked for CHECKPOINT")
        
        # Filter out activations with no valid size or recompute stats
        print("Filtering valid activations...")
        valid_activations_df = self.activation_stats_df.dropna(subset=['avg_mem_size_bytes', 'recomp_time_s', 'creation_rank', 'last_fw_use_rank'])
        valid_activations_df = valid_activations_df[valid_activations_df['avg_mem_size_bytes'] > 0]
        print(f"Found {len(valid_activations_df)} valid activations for consideration")

        # Pre-compute benefit values for all activations to avoid repeated calculations
        print("Pre-computing benefit values...")
        
        # Create a set of candidate activations
        candidate_set = set(valid_activations_df.index)
        
        # Initialize tracking sets
        swaps = set()
        recomps = set()
        
        # Initialize last_prompt to the last node in the backward graph
        last_prompt = None
        last_rank = -1
        for _, node_details in self.node_stats_df.iterrows():
            if node_details['gtype'] == 'bw' and node_details['rank'] > last_rank:
                last_rank = node_details['rank']
                last_prompt = node_details['node_name']
        
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
            print(f"Current checkpoint count: {sum(1 for d in current_schedule.values() if d == 'CHECKPOINT')}")
            print(f"Current recompute count: {sum(1 for d in current_schedule.values() if d == 'RECOMPUTE')}")

            # Check if we've met the budget
            if current_peak_memory <= self.memory_budget_bytes:
                print("Memory budget met.")
                break # Budget met
            
            # Process a batch of candidates at a time for efficiency
            candidates_to_process = list(candidate_set)[:batch_size]
            
            for act_name in candidates_to_process:
                # Select swap candidate with maximum inactive time
                s_cand = self._get_max_inactive_time_candidate(candidate_set)
                if s_cand is None:
                    continue
                    
                # Calculate swap overhead
                s_overhead, prompt_node = self._calculate_swap_overhead_v2(s_cand, last_prompt)
                
                # Select recompute candidate with maximum recompute ratio
                r_cand = self._get_max_recompute_ratio_candidate(candidate_set)
                if r_cand is None:
                    continue
                    
                # Calculate recompute overhead
                r_overhead = self._calculate_recompute_overhead_v2(r_cand)
                
                # Make decision based on overhead comparison
                if s_overhead < r_overhead:
                    # Choose to swap
                    print(f"Choosing to CHECKPOINT {s_cand} (swap overhead: {s_overhead:.6f}s, recompute overhead: {r_overhead:.6f}s)")
                    current_schedule[s_cand] = 'CHECKPOINT'
                    swaps.add(s_cand)
                    last_prompt = prompt_node
                    cand = s_cand
                else:
                    # Choose to recompute
                    print(f"Choosing to RECOMPUTE {r_cand} (swap overhead: {s_overhead:.6f}s, recompute overhead: {r_overhead:.6f}s)")
                    current_schedule[r_cand] = 'RECOMPUTE'
                    recomps.add(r_cand)
                    cand = r_cand
                
                # Remove chosen candidate from set
                candidate_set.remove(cand)
                
                # Update recomputation counts and dependencies
                recomp_cnt = self._update_recomps(cand, recomps)
                
                # Update remaining candidates based on this decision
                self._update_candidates(cand, recomp_cnt, candidate_set)
                
                # Update swap prompts if needed
                self._update_swap_prompts(swaps, candidate_set)
                
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
            print(f"Returning best schedule found so far with {sum(1 for d in current_schedule.values() if d == 'CHECKPOINT')} checkpoints and {sum(1 for d in current_schedule.values() if d == 'RECOMPUTE')} recomputes.")
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
        Select the candidate with maximum recompute ratio (memory_size / recompute_time).
        """
        max_ratio = -1
        max_candidate = None
        
        for act_name in candidate_set:
            act_details = self._get_activation_details(act_name)
            if act_details is None:
                continue
                
            if 'avg_mem_size_bytes' not in act_details or 'recomp_time_s' not in act_details:
                continue
                
            mem_size = act_details['avg_mem_size_bytes']
            recomp_time = act_details['recomp_time_s']
            
            if pd.isna(mem_size) or pd.isna(recomp_time) or recomp_time <= 0:
                continue
                
            ratio = mem_size / recomp_time
            if ratio > max_ratio:
                max_ratio = ratio
                max_candidate = act_name
                
        return max_candidate
        
    def _calculate_swap_overhead_v2(self, act_name, last_prompt):
        """
        Calculate the swap overhead for a given activation.
        Implements Algorithm C from the μ-TWO paper.
        
        Args:
            act_name: Name of the activation to calculate swap overhead for
            last_prompt: Last node used as a prefetch prompt
            
        Returns:
            swap_overhead: The overhead of swapping this activation
            prompt_node: The node that should be used as the prefetch prompt
        """
        act_details = self._get_activation_details(act_name)
        if act_details is None:
            return float('inf'), last_prompt
            
        # Get first backward access node and swap time
        first_bw_use = act_details['first_bw_use_rank']
        if pd.isna(first_bw_use):
            return float('inf'), last_prompt
            
        bw_access = None
        for _, node_details in self.node_stats_df.iterrows():
            if node_details['rank'] == first_bw_use:
                bw_access = node_details['node_name']
                break
                
        if bw_access is None:
            return float('inf'), last_prompt
            
        swap_time = act_details['avg_swap_time_s']
        if pd.isna(swap_time):
            return float('inf'), last_prompt
            
        # Check if we're in peak memory interval
        # This is a simplification - in a full implementation, we would need to
        # determine the peak memory interval more accurately
        reached_peak = False
        
        # Case 1: No overlap possible (peak interval already reached)
        if reached_peak:
            # Case 1(a): No conflict with existing swap
            if first_bw_use < self._get_node_details(last_prompt)['rank']:
                return swap_time, bw_access
            else:
                # Case 1(b): Conflicts with existing swap
                # This is a simplification - in a full implementation, we would need to
                # calculate the remaining time of the existing swap
                return swap_time * 1.5, bw_access
        
        # Case 2: Complete overlap possible
        # This is a simplification - in a full implementation, we would need to
        # calculate the overlap more accurately
        
        # For now, assume we can overlap 50% of the swap time
        remaining_swap_time = swap_time * 0.5
        
        # Case 3: Partial overlap
        # Return the remaining swap time as the overhead
        return max(0, remaining_swap_time), bw_access
        
    def _calculate_recompute_overhead_v2(self, act_name):
        """
        Calculate the recomputation overhead for a given activation.
        
        Args:
            act_name: Name of the activation to calculate recompute overhead for
            
        Returns:
            recompute_overhead: The overhead of recomputing this activation
        """
        act_details = self._get_activation_details(act_name)
        if act_details is None:
            return float('inf')
            
        recomp_time = act_details['recomp_time_s']
        if pd.isna(recomp_time):
            return float('inf')
            
        return recomp_time
        
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
        
    def _update_swap_prompts(self, swaps, candidate_set):
        """
        Update swap prompts for remaining candidates.
        
        Args:
            swaps: Set of activations marked for swapping
            candidate_set: Set of remaining candidates
        """
        # This is a simplification - in a full implementation, we would need to
        # update the prefetch prompts for remaining candidates
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
        print(f"Number of activations to CHECKPOINT: {checkpointed_count}")

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