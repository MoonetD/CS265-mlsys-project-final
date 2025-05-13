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

    def _simulate_memory_usage(self, current_schedule, fixed_overhead_bytes=0):
        """
        Simulates peak memory usage and total execution time based on a given schedule.
        Inspired by Algorithm G from the u-TWO paper.

        Args:
            current_schedule (dict): Activation name to decision ('CHECKPOINT', 'RECOMPUTE').
            fixed_overhead_bytes (float): Estimated memory for parameters, gradients, optimizer.

        Returns:
            float: Estimated peak GPU memory in bytes.
            float: Total execution time in seconds.
        """
        # --- Calculate Total Execution Time ---
        # Start with sum of all base node run times
        total_execution_time = self.node_stats_df['avg_run_time_s'].sum()
        # Add recomputation times for activations scheduled for RECOMPUTE
        for act_name, decision in current_schedule.items():
            if decision == 'RECOMPUTE':
                act_details = self._get_activation_details(act_name)
                # Ensure act_details is a Series and 'recomp_time' exists
                if act_details is not None and isinstance(act_details, pd.Series) and \
                   'recomp_time_s' in act_details.index and pd.notna(act_details['recomp_time_s']):
                    total_execution_time += act_details['recomp_time_s']
                elif act_details is None:
                    # This might happen if an activation in current_schedule is not in activation_stats_df
                    # Or if it was filtered out earlier. Should be handled by how current_schedule is built.
                    pass


        # --- Simulate Peak Memory Usage (Algorithm G inspired) ---
        # M_fixed: memory for parameters, gradients, optimizer states. Provided by fixed_overhead_bytes.
        # M_active: memory for currently live checkpointed activations.
        # M_op_peak: peak memory during a specific operation, taken from profiler (node_details['avg_peak_mem_node']).
        #            This value inherently includes M_fixed, M_active (at that op's time), op's own I/O, and transients.

        peak_memory_observed_bytes = fixed_overhead_bytes  # Initialize with fixed part
        
        # This tracks sum of M_fixed + M_active (live checkpointed activations)
        current_checkpointed_plus_fixed_mem_bytes = fixed_overhead_bytes
        
        live_checkpointed_activations = {}  # act_name -> pd.Series (details of live checkpointed activations)

        execution_order = self._get_node_execution_order()
        if not execution_order:
            # Return inf for memory if order can't be determined, but use calculated time
            return float('inf'), total_execution_time

        for node_name in execution_order:
            node_details = self._get_node_details(node_name) # This returns a pd.Series
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
                # Iterate through all activations to find those created by this FW node (matching creation_rank)
                for act_idx, act_details_series in self.activation_stats_df.iterrows():
                    if pd.notna(act_details_series['creation_rank']) and act_details_series['creation_rank'] == node_rank:
                        act_name = act_details_series['activation_name'] # Get the name from the correct column
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
            
        return peak_memory_observed_bytes, total_execution_time


    def decide_checkpoints(self, fixed_overhead_gb=2.0):
        """
        Decides which activations to checkpoint and which to recompute.
        Implements a simplified version of Algorithm B from the u-TWO paper.

        Args:
            fixed_overhead_gb (float): Estimated fixed memory overhead for parameters,
                                       gradients, optimizer states in GB. This is a simplification.
                                       The paper's Algorithm G would determine this more dynamically.
        Returns:
            dict: A schedule mapping activation names to 'CHECKPOINT' or 'RECOMPUTE'.
        """
        fixed_overhead_bytes = fixed_overhead_gb * (1024**3)

        # Initial state: checkpoint all activations
        current_schedule = {
            act_name: 'CHECKPOINT'
            for act_name in self.activation_stats_df['activation_name'] # Iterate using the correct column name
            if pd.notna(self.activation_stats_df.loc[act_name, 'avg_mem_size_bytes']) and self.activation_stats_df.loc[act_name, 'avg_mem_size_bytes'] > 0
        }
        
        # Filter out activations with no valid size or recompute stats
        valid_activations_df = self.activation_stats_df.dropna(subset=['avg_mem_size_bytes', 'recomp_time_s', 'creation_rank', 'last_fw_use_rank'])
        valid_activations_df = valid_activations_df[valid_activations_df['avg_mem_size_bytes'] > 0]


        while True:
            current_peak_memory, current_exec_time = self._simulate_memory_usage(current_schedule, fixed_overhead_bytes)
            
            print(f"Simulated peak memory: {current_peak_memory / (1024**3):.2f} GB. Budget: {self.memory_budget_bytes / (1024**3):.2f} GB. Exec time: {current_exec_time:.2f}s")

            if current_peak_memory <= self.memory_budget_bytes:
                print("Memory budget met.")
                break # Budget met

            # Find candidate to evict (move from CHECKPOINT to RECOMPUTE)
            # Algorithm B, lines 6-13: minimize overhead(a) / benefit(a)
            best_candidate = None
            min_ratio = float('inf')
            # Initialize max_benefit_at_min_ratio for tie-breaking
            max_benefit_at_min_ratio = -1.0

            if not any(decision == 'CHECKPOINT' for decision in current_schedule.values()):
                print("Warning: No checkpointed activations left to evict, but still over budget. Check fixed_overhead or budget.")
                break


            for act_name, decision in current_schedule.items():
                if decision == 'CHECKPOINT':
                    act_details = self._get_activation_details(act_name)
                    if act_details is None or act_name not in valid_activations_df.index:
                        continue

                    # Overhead: recomputation time
                    recomp_time, _ = self._calculate_recompute_overhead(act_name)
                    # Removed the problematic skip:
                    # if recomp_time == 0 and act_details['avg_mem_size_bytes'] > 0:
                    #     continue

                    # Benefit: memory saved * "live time"
                    # "live time" can be approximated by the span of ranks it's active.
                    # Paper uses max_live_time(a) or avg_live_time(a).
                    # Let's use (last_fw_use_rank - creation_rank) as a proxy for now.
                    # A more accurate live time would consider backward pass usage too.
                    # Effective live interval: from creation_rank to last_bw_use_rank (if exists, else last_fw_use_rank)
                    
                    creation_rank = act_details['creation_rank']
                    last_fw_use = act_details['last_fw_use_rank']
                    # first_bw_use = act_details['first_bw_use_rank'] # Not directly used in this benefit calc
                    last_bw_use = act_details['last_bw_use_rank']

                    # Determine the effective end of life for the activation
                    effective_last_use_rank = last_fw_use
                    if pd.notna(last_bw_use) and last_bw_use > effective_last_use_rank :
                         effective_last_use_rank = last_bw_use
                    
                    if pd.isna(creation_rank) or pd.isna(effective_last_use_rank):
                        # print(f"Skipping {act_name}: missing rank info for benefit calculation.")
                        continue
                    
                    live_duration_ranks = effective_last_use_rank - creation_rank
                    if live_duration_ranks <= 0: # Should not happen for valid activations
                        live_duration_ranks = 1 # Avoid division by zero, give it some weight

                    memory_saved = act_details['avg_mem_size_bytes']
                    if memory_saved <= 0:
                        continue # No benefit if it takes no memory

                    benefit = live_duration_ranks * memory_saved 
                                        
                    if benefit == 0: # Avoid division by zero if benefit is unexpectedly zero
                        ratio = float('inf')
                    else:
                        ratio = recomp_time / benefit
                    
                    # print(f"Candidate {act_name}: recomp_time={recomp_time:.4f}, mem_saved={memory_saved/1024**2:.2f}MB, live_ranks={live_duration_ranks}, benefit={benefit:.2f}, ratio={ratio:.6f}")

                    current_act_benefit = benefit

                    if ratio < min_ratio:
                        min_ratio = ratio
                        max_benefit_at_min_ratio = current_act_benefit
                        best_candidate = act_name
                    elif ratio == min_ratio:
                        # Tie-breaking: if ratios are equal (especially if 0),
                        # prefer the one with greater benefit.
                        if current_act_benefit > max_benefit_at_min_ratio:
                            max_benefit_at_min_ratio = current_act_benefit
                            best_candidate = act_name
            
            if best_candidate:
                print(f"Evicting {best_candidate} (ratio: {min_ratio:.6f}, benefit: {max_benefit_at_min_ratio:.2f}) to RECOMPUTE.")
                current_schedule[best_candidate] = 'RECOMPUTE'
            else:
                print("Could not find a suitable candidate to evict. Algorithm might be stuck or all options exhausted.")
                break # No candidate found, or all are already RECOMPUTE

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

    print(f"Starting Activation Checkpointing Algorithm...")
    print(f"Node Stats: {node_stats_file}")
    print(f"Activation Stats: {activation_stats_file}")
    print(f"Memory Budget: {memory_budget} GB")

    try:
        ac_algo = ActivationCheckpointingAlgorithm(
            node_stats_path=node_stats_file,
            activation_stats_path=activation_stats_file,
            memory_budget_gb=memory_budget
        )

        # You might want to pass a more realistic fixed_overhead_gb based on your model
        # E.g., ResNet-152 parameters are ~232MB. Grads ~232MB. Optimizer states can be 1x or 2x of params.
        # So, roughly 0.7GB to 1GB could be a starting point for fixed overhead.
        # The paper's Algorithm G is more precise.
        # Let's use 1 GB for parameters, gradients, and optimizer states as a rough estimate.
        final_schedule = ac_algo.decide_checkpoints(fixed_overhead_gb=0.1)
        
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

        final_peak_mem, final_exec_time = ac_algo._simulate_memory_usage(final_schedule, fixed_overhead_bytes=0.1 * (1024**3))
        print(f"Estimated Peak GPU Memory with schedule: {final_peak_mem / (1024**3):.2f} GB")
        print(f"Estimated Total Execution Time with schedule: {final_exec_time:.2f} s")

    except FileNotFoundError:
        print("Execution failed: Ensure profiler CSV files are in the correct path.")
    except ValueError as ve:
        print(f"Execution failed due to data error: {ve}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()