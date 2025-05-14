import pandas as pd
import numpy as np

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
        Inspired by Algorithm G from the u-TWO paper.

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


        # --- Simulate Peak Memory Usage (Algorithm G inspired) ---
        if debug:
            print("Simulating peak memory usage...")
            
        # M_fixed: memory for parameters, gradients, optimizer states. Provided by fixed_overhead_bytes.
        # M_active: memory for currently live checkpointed activations.
        # M_op_peak: peak memory during a specific operation, taken from profiler (node_details['avg_peak_mem_node']).
        #            This value inherently includes M_fixed, M_active (at that op's time), op's own I/O, and transients.

        peak_memory_observed_bytes = fixed_overhead_bytes  # Initialize with fixed part
        
        # This tracks sum of M_fixed + M_active (live checkpointed activations)
        current_checkpointed_plus_fixed_mem_bytes = fixed_overhead_bytes
        
        live_checkpointed_activations = {}  # act_name -> pd.Series (details of live checkpointed activations)

        # Get execution order once and cache it
        execution_order = self._get_node_execution_order()
        if not execution_order:
            # Return inf for memory if order can't be determined, but use calculated time
            if debug:
                print("Warning: Could not determine execution order.")
            return float('inf'), total_execution_time

        # Pre-fetch all node details to avoid repeated lookups
        if debug:
            print("Pre-fetching node details...")
        node_details_cache = {}
        for node_name in execution_order:
            node_details_cache[node_name] = self._get_node_details(node_name)
            
        # Create a mapping from rank to activations created at that rank for faster lookup
        if debug:
            print("Creating rank-to-activations mapping...")
        activations_by_creation_rank = {}
        for act_idx, act_details_series in self.activation_stats_df.iterrows():
            if pd.notna(act_details_series['creation_rank']):
                rank = int(act_details_series['creation_rank'])
                if rank not in activations_by_creation_rank:
                    activations_by_creation_rank[rank] = []
                activations_by_creation_rank[rank].append(act_details_series)

        # Process nodes in execution order
        total_nodes = len(execution_order)
        if debug:
            print(f"Processing {total_nodes} nodes in execution order...")
            
        for i, node_name in enumerate(execution_order):
            # Print progress every 10% of nodes
            if debug and i % max(1, total_nodes // 10) == 0:
                print(f"  Processing node {i+1}/{total_nodes} ({(i+1)/total_nodes*100:.1f}%)...")
                
            node_details = node_details_cache[node_name]
            if node_details is None:
                continue
            
            node_rank = node_details['rank']
            node_gtype = node_details['gtype']

            # 1. Update peak_memory_observed_bytes with the peak during this specific operation.
            #    node_details['avg_peak_mem_node'] is the total memory allocated when this node ran.
            if 'avg_peak_mem_node' in node_details.index and pd.notna(node_details['avg_peak_mem_node']):
                peak_memory_observed_bytes = max(peak_memory_observed_bytes, node_details['avg_peak_mem_node'])

            # 2. Update current_checkpointed_plus_fixed_mem_bytes based on liveness changes of checkpointed activations.
            #    A. Add newly created checkpointed activations to the live set.
            if node_gtype == 'fw':
                # Use the pre-computed mapping to find activations created at this rank
                if node_rank in activations_by_creation_rank:
                    for act_details_series in activations_by_creation_rank[node_rank]:
                        act_name = act_details_series['activation_name']
                        act_mem_size = act_details_series['avg_mem_size_bytes']

                        if pd.isna(act_mem_size) or act_mem_size <= 0:
                            continue # Skip activations with no or invalid memory size
                        
                        if current_schedule.get(act_name) == 'CHECKPOINT' and act_name not in live_checkpointed_activations:
                            live_checkpointed_activations[act_name] = act_details_series # Store the full Series
                            current_checkpointed_plus_fixed_mem_bytes += act_mem_size
            
            #    B. Remove checkpointed activations from the live set that are no longer needed.
            #       An activation is no longer needed if the current node_rank is at or after its effective last use rank.
            activations_to_remove = []
            for act_name_live, act_details_live_series in live_checkpointed_activations.items():
                # Determine effective last use rank for the live activation
                last_fw_use = act_details_live_series['last_fw_use_rank']
                last_bw_use = act_details_live_series['last_bw_use_rank']
                
                effective_last_use_rank = last_fw_use
                if pd.notna(last_bw_use) and \
                   (pd.isna(effective_last_use_rank) or last_bw_use > effective_last_use_rank):
                    effective_last_use_rank = last_bw_use
                
                if pd.notna(effective_last_use_rank) and node_rank >= effective_last_use_rank:
                    activations_to_remove.append(act_name_live)
            
            for act_name_to_remove in activations_to_remove:
                if act_name_to_remove in live_checkpointed_activations:
                    mem_size_to_remove = live_checkpointed_activations[act_name_to_remove]['avg_mem_size_bytes']
                    if pd.notna(mem_size_to_remove):
                         current_checkpointed_plus_fixed_mem_bytes -= mem_size_to_remove
                    del live_checkpointed_activations[act_name_to_remove]

            # 3. Update peak_memory_observed_bytes with the memory state *between* operations.
            #    This is the sum of fixed overhead and all currently live checkpointed activations.
            peak_memory_observed_bytes = max(peak_memory_observed_bytes, current_checkpointed_plus_fixed_mem_bytes)
            
        if debug:
            print(f"Memory simulation complete.")
            print(f"Peak memory: {peak_memory_observed_bytes / (1024**3):.3f} GB")
            print(f"Final execution time: {total_execution_time:.4f}s")
            
        return peak_memory_observed_bytes, total_execution_time


    def decide_checkpoints(self, fixed_overhead_gb=2.0, debug=False, batch_size=50, max_iterations=1000):
        """
        Decides which activations to checkpoint and which to recompute.
        Implements a simplified version of Algorithm B from the u-TWO paper.

        Args:
            fixed_overhead_gb (float): Estimated fixed memory overhead for parameters,
                                       gradients, optimizer states in GB. This is a simplification.
                                       The paper's Algorithm G would determine this more dynamically.
            debug (bool): Whether to print detailed debug information.
            batch_size (int): Number of activations to evict at once for faster convergence.
            max_iterations (int): Maximum number of iterations to prevent infinite loops.
            
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
        benefit_cache = {}
        ratio_cache = {}
        
        for act_name in valid_activations_df.index:
            act_details = self._get_activation_details(act_name)
            if act_details is None:
                continue
                
            # Calculate benefit components
            creation_rank = act_details['creation_rank']
            last_fw_use = act_details['last_fw_use_rank']
            last_bw_use = act_details['last_bw_use_rank']
            
            # Determine effective last use rank
            effective_last_use_rank = last_fw_use
            if pd.notna(last_bw_use) and last_bw_use > effective_last_use_rank:
                effective_last_use_rank = last_bw_use
                
            if pd.isna(creation_rank) or pd.isna(effective_last_use_rank):
                continue
                
            live_duration_ranks = effective_last_use_rank - creation_rank
            if live_duration_ranks <= 0:
                live_duration_ranks = 1
                
            memory_saved = act_details['avg_mem_size_bytes']
            if memory_saved <= 0:
                continue
                
            benefit = live_duration_ranks * memory_saved
            benefit_cache[act_name] = benefit
            
            # Also pre-compute ratio
            recomp_time, _ = self._calculate_recompute_overhead(act_name)
            if benefit == 0:
                ratio = float('inf')
            else:
                ratio = recomp_time / benefit
                
            ratio_cache[act_name] = ratio

        iteration = 0
        while iteration < max_iterations:
            iteration += 1
            print(f"\nIteration {iteration}/{max_iterations}")
            
            # Run memory simulation with debug info on first iteration
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

            # Check if we've run out of checkpointed activations
            if not any(decision == 'CHECKPOINT' for decision in current_schedule.values()):
                print("Warning: No checkpointed activations left to evict, but still over budget. Check fixed_overhead or budget.")
                break
                
            # Find candidates to evict (move from CHECKPOINT to RECOMPUTE)
            print(f"Finding up to {batch_size} candidates to evict...")
            candidates = []

            for act_name, decision in current_schedule.items():
                if decision == 'CHECKPOINT':
                    if act_name not in valid_activations_df.index:
                        continue
                        
                    # Use cached values if available
                    if act_name in ratio_cache and act_name in benefit_cache:
                        ratio = ratio_cache[act_name]
                        benefit = benefit_cache[act_name]
                        candidates.append((act_name, ratio, benefit))
                        
            # Sort candidates by ratio (ascending) and then by benefit (descending) for tie-breaking
            candidates.sort(key=lambda x: (x[1], -x[2]))
            
            # Take the top batch_size candidates
            batch_candidates = candidates[:batch_size]
            
            if not batch_candidates:
                print("Could not find any suitable candidates to evict. Algorithm might be stuck or all options exhausted.")
                break
                
            # Evict all candidates in the batch
            for act_name, ratio, benefit in batch_candidates:
                print(f"Evicting {act_name} (ratio: {ratio:.6f}, benefit: {benefit:.2f}) to RECOMPUTE.")
                current_schedule[act_name] = 'RECOMPUTE'
                
            # If we're only evicting one activation per iteration, print more details
            if len(batch_candidates) == 1:
                act_name, ratio, benefit = batch_candidates[0]
                act_details = self._get_activation_details(act_name)
                if act_details is not None:
                    mem_size = act_details['avg_mem_size_bytes'] / (1024**2)  # Convert to MB
                    print(f"  Memory saved: {mem_size:.2f} MB")
                    if 'recomp_time_s' in act_details:
                        print(f"  Recomputation time: {act_details['recomp_time_s']:.6f}s")

        self.schedule = current_schedule
        return self.schedule

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