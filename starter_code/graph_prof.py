import statistics
from collections import defaultdict
from enum import Enum
from typing import Dict, Any, Set, List # Added Set, List
import torch
import torch.fx as fx
import csv # Added for CSV output
import matplotlib.pyplot as plt # Added for plotting
# from typing import Dict, Any # Duplicate, removed


class OP(str, Enum):
    CALL_FUNCTION = "call_function"
    CALL_MODULE = "call_module"
    CALL_METHOD = "call_method"
    GET_ATTR = "get_attr"
    OUTPUT = "output"
    PLACEHOLDER = "placeholder"


class NodeType(str, Enum): # Changed back to str Enum for consistency if needed elsewhere
    """
    NodeType is a enum that records the type of the tensors in the graph.
    """
    PARAM = "parameter"
    ACT = "activation"
    GRAD = "gradient"
    OTHER = "other"


# This is an example graph_profiler that extends the fx.Interpreter class, it
# will perform graph execution by running the graph node by node.


class GraphProfiler(fx.Interpreter):
    def __init__(self, module: fx.GraphModule, garbage_collect_values: bool = True):
        super().__init__(module, garbage_collect_values)

        # Static analysis attributes
        self.sep_fw_end_rank: int = -1
        self.sep_bw_start_rank: int = -1
        self.node_types: Dict[str, NodeType] = {}
        self.node_gtypes: Dict[str, str] = {} # To store 'forward', 'backward', or 'optimizer/other'
        self.activation_liveness: Dict[str, Dict[str, int]] = {}

        self.param_node_names: Set[str] = set()
        self.grad_node_names: Set[str] = set()
        # TODO: Add calculation of param_sizes and grad_sizes if needed for peak memory breakdown
        # self.param_sizes: Dict[str, int] = {}
        # self.grad_sizes: Dict[str, int] = {}

        self.node_ranks: Dict[fx.Node, int] = {}
        self.ranked_nodes: List[fx.Node] = []

        # Runtime profiling attributes (Raw data collected per run, stored as lists)
        self.run_times: Dict[str, List[float]] = defaultdict(list)
        self.peak_mem_node: Dict[str, List[int]] = defaultdict(list)
        self.memory_sizes: Dict[str, List[int]] = defaultdict(list) # Activation output sizes per run
        self.swap_times: Dict[str, List[float]] = defaultdict(list) # Individual swap event times per activation
        self.swapped_out_activations: Set[str] = set() # Tracks names of activations currently in CPU (state during a run)

        # Averaged runtime stats (Calculated after aggregation)
        self.avg_run_times: Dict[str, float] = {}
        self.avg_peak_mem_node: Dict[str, float] = {}
        self.avg_memory_sizes: Dict[str, float] = {}
        self.avg_swap_times: Dict[str, float] = {} # Average *total* swap time per activation per run

        # MuTWO specific metrics (Calculated after aggregation using averaged stats)
        self.inactive_times: Dict[str, float] = {}
        self.recomp_times: Dict[str, float] = {}
        self.recomp_memory: Dict[str, int] = {} # Uses avg_memory_sizes
        self.recompute_ratios: Dict[str, float] = {}

        # Constants for swap speed simulation (example values, e.g., 10 GB/s)
        # These should ideally be profiled on the target hardware.
        self.BYTES_PER_SEC_CPU_TO_GPU = 10 * (1024**3)
        self.BYTES_PER_SEC_GPU_TO_CPU = 10 * (1024**3)
        self.GPU_MEMORY_LIMIT_MIB = 40 * 1024 # Example: 40 GiB in MiB

        # --- First Pass: Rank nodes, find boundaries, identify initial Params/Grads ---
        _fused_adam_node: fx.Node | None = None
        for rank, node in enumerate(self.module.graph.nodes):
            self.node_ranks[node] = rank
            self.ranked_nodes.append(node)

            if node.op == OP.CALL_FUNCTION:
                if node.target == torch.ops.separator.sep.default:
                    self.sep_fw_end_rank = rank
                elif node.target == torch.ops.separator.sep_backward.default:
                    self.sep_bw_start_rank = rank
                elif node.target == torch.ops.aten._fused_adam.default:
                    _fused_adam_node = node

        # Identify parameter names from module parameters
        for param_name_dot, _ in self.module.named_parameters():
            fx_node_name = param_name_dot.replace('.', '_') # FX naming convention
            self.param_node_names.add(fx_node_name)

        # Identify gradient names from _fused_adam_node (if found and args are ListConstruct)
        if _fused_adam_node and len(_fused_adam_node.args) > 1:
            # Per comments, arg 0 is params list, arg 1 is grads list
            # This assumes these args are nodes that are 'prim::ListConstruct'
            param_list_provider_node = _fused_adam_node.args[0]
            grad_list_provider_node = _fused_adam_node.args[1]

            if isinstance(param_list_provider_node, fx.Node) and \
               param_list_provider_node.op == OP.CALL_FUNCTION and \
               str(param_list_provider_node.target) == 'prim::ListConstruct':
                for p_node_in_list in param_list_provider_node.args:
                    if isinstance(p_node_in_list, fx.Node):
                        self.param_node_names.add(p_node_in_list.name)

            if isinstance(grad_list_provider_node, fx.Node) and \
               grad_list_provider_node.op == OP.CALL_FUNCTION and \
               str(grad_list_provider_node.target) == 'prim::ListConstruct':
                for g_node_in_list in grad_list_provider_node.args:
                    if isinstance(g_node_in_list, fx.Node):
                        self.grad_node_names.add(g_node_in_list.name)

        # --- Second Pass: Determine NodeType, gtype and identify Activations ---
        for node in self.ranked_nodes:
            rank = self.node_ranks[node]
            node_name = node.name

            # Determine gtype
            if self.sep_fw_end_rank != -1 and rank <= self.sep_fw_end_rank:
                self.node_gtypes[node_name] = "forward"
            elif self.sep_bw_start_rank != -1 and rank >= self.sep_bw_start_rank:
                # Nodes after sep_bw_start_rank are backward or optimizer steps
                # Further refinement could be done if optimizer steps are explicitly marked
                self.node_gtypes[node_name] = "backward"
            elif self.sep_fw_end_rank == -1 and self.sep_bw_start_rank != -1 and rank < self.sep_bw_start_rank:
                # If no explicit FW end, but BW start exists, assume FW before BW
                self.node_gtypes[node_name] = "forward"
            elif self.sep_fw_end_rank != -1 and self.sep_bw_start_rank == -1 and rank > self.sep_fw_end_rank:
                 # If no explicit BW start, but FW end exists, assume optimizer/other after FW
                self.node_gtypes[node_name] = "optimizer/other"
            elif self.sep_fw_end_rank == -1 and self.sep_bw_start_rank == -1:
                # No separators, assume all 'forward' for simplicity or 'unknown'
                # This case might need specific handling based on graph structure if it occurs
                self.node_gtypes[node_name] = "unknown"
            else: # Between fw_end and bw_start, or other complex cases
                self.node_gtypes[node_name] = "intermediate/other"


            if node_name in self.param_node_names:
                self.node_types[node_name] = NodeType.PARAM
            elif node_name in self.grad_node_names:
                self.node_types[node_name] = NodeType.GRAD
            else:
                is_activation = False
                if node.op not in [OP.PLACEHOLDER, OP.OUTPUT]: # Activations are not placeholders or outputs
                    # Created in forward pass (or at its boundary)
                    created_in_fwd = (self.sep_fw_end_rank == -1 or rank <= self.sep_fw_end_rank)

                    used_in_bwd = False
                    if self.sep_bw_start_rank != -1: # Backward pass exists
                        for user_node in node.users:
                            if self.node_ranks[user_node] >= self.sep_bw_start_rank:
                                used_in_bwd = True
                                break

                    if created_in_fwd and used_in_bwd:
                        is_activation = True

                if is_activation:
                    self.node_types[node_name] = NodeType.ACT
                else:
                    self.node_types[node_name] = NodeType.OTHER

        # --- Third Pass: Activation Liveness ---
        for node in self.ranked_nodes:
            node_name = node.name
            if self.node_types.get(node_name) == NodeType.ACT:
                creation_rank = self.node_ranks[node]
                last_fw_use_rank = -1
                first_bw_use_rank = float('inf')
                last_bw_use_rank = -1

                for user_node in node.users: # node.users is a Dict[fx.Node, None]
                    user_rank = self.node_ranks[user_node]

                    # Check forward pass usage (up to and including sep_fw_end_rank)
                    if self.sep_fw_end_rank != -1 and user_rank <= self.sep_fw_end_rank:
                        last_fw_use_rank = max(last_fw_use_rank, user_rank)
                    elif self.sep_fw_end_rank == -1: # No sep_fw, all non-BW considered FW
                         if self.sep_bw_start_rank == -1 or user_rank < self.sep_bw_start_rank:
                            last_fw_use_rank = max(last_fw_use_rank, user_rank)


                    # Check backward pass usage (from sep_bw_start_rank onwards)
                    if self.sep_bw_start_rank != -1 and user_rank >= self.sep_bw_start_rank:
                        first_bw_use_rank = min(first_bw_use_rank, user_rank)
                        last_bw_use_rank = max(last_bw_use_rank, user_rank)

                if first_bw_use_rank == float('inf'): # Should not happen for ACT type by definition
                    first_bw_use_rank = -1

                self.activation_liveness[node_name] = {
                    "creation_rank": creation_rank,
                    "last_fw_use_rank": last_fw_use_rank,
                    "first_bw_use_rank": first_bw_use_rank,
                    "last_bw_use_rank": last_bw_use_rank,
                }

    def run(
        self,
        *args,
        initial_env: Dict[fx.Node, Any] | None = None,
        enable_io_processing: bool = True
    ) -> Any:
        return super().run(
            *args, initial_env=initial_env, enable_io_processing=enable_io_processing
        )

    def run_node(self, n: fx.Node) -> Any:
        node_name = n.name
        current_rank = self.node_ranks[n]

        # 1. Swap-in Simulation (Before node execution, during backward pass)
        if self.sep_bw_start_rank != -1 and current_rank >= self.sep_bw_start_rank:
            for input_node in n.all_input_nodes:
                input_node_name = input_node.name
                if self.node_types.get(input_node_name) == NodeType.ACT and \
                   input_node_name in self.swapped_out_activations:

                    # This activation was swapped out and is needed now. Simulate swap-in.
                    # Use the *last recorded* memory size for this activation for simulation
                    # Check if memory_sizes has data for this node before accessing [-1]
                    if input_node_name in self.memory_sizes and self.memory_sizes[input_node_name]:
                        tensor_size_bytes = self.memory_sizes[input_node_name][-1] # Use last known size
                        if tensor_size_bytes > 0: # Only simulate if size is known and positive
                            swap_in_time_sec = tensor_size_bytes / self.BYTES_PER_SEC_CPU_TO_GPU
                            self.swap_times[input_node_name].append(swap_in_time_sec) # Append event time
                            # print(f"Simulating SWAP-IN for {input_node_name} ({tensor_size_bytes} B): {swap_in_time_sec:.6f} s")

                    self.swapped_out_activations.remove(input_node_name) # Mark as back in GPU

        # 2. Timing and Memory Measurement (Around node execution)
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        # Reset peak memory stats for the current device to measure peak for this node
        # Note: This measures peak for the *device*. If other operations run concurrently
        # on the same device but outside this profiler's control, they might affect this.
        # However, for typical single-stream model execution, this should be representative.
        torch.cuda.reset_peak_memory_stats()

        start_event.record()
        result = super().run_node(n)
        end_event.record()

        torch.cuda.synchronize() # Wait for all kernels to complete for accurate timing & memory

        # Store run time
        run_time_ms = start_event.elapsed_time(end_event)
        self.run_times[node_name].append(run_time_ms / 1000.0) # Append time in seconds

        # Store peak memory allocated during this node's execution
        self.peak_mem_node[node_name].append(torch.cuda.max_memory_allocated())

        # Store output tensor memory size if it's an activation
        mem_size = 0
        if self.node_types.get(node_name) == NodeType.ACT and isinstance(result, torch.Tensor):
            if result.device.type == 'cuda':
                 mem_size = result.element_size() * result.nelement()
            elif result.is_quantized:
                 mem_size = result.untyped_storage().size()
        # Append memory size even if 0 (e.g., non-tensor activation or CPU tensor)
        # This ensures memory_sizes list length matches run count for averaging.
        if self.node_types.get(node_name) == NodeType.ACT:
             self.memory_sizes[node_name].append(mem_size)


        # 3. Swap-out Simulation (After node execution, during forward pass)
        if (self.sep_fw_end_rank != -1 and current_rank <= self.sep_fw_end_rank) or \
           (self.sep_fw_end_rank == -1 and (self.sep_bw_start_rank == -1 or current_rank < self.sep_bw_start_rank)): # In forward pass
            if self.node_types.get(node_name) == NodeType.ACT:
                liveness_info = self.activation_liveness.get(node_name)
                # Use the *last recorded* memory size for this activation for simulation
                if liveness_info and current_rank == liveness_info["last_fw_use_rank"] and \
                   node_name in self.memory_sizes and self.memory_sizes[node_name]:

                    tensor_size_bytes = self.memory_sizes[node_name][-1] # Use last known size
                    if tensor_size_bytes > 0: # Only simulate if size is known and positive
                        swap_out_time_sec = tensor_size_bytes / self.BYTES_PER_SEC_GPU_TO_CPU
                        self.swap_times[node_name].append(swap_out_time_sec) # Append event time
                        # print(f"Simulating SWAP-OUT for {node_name} ({tensor_size_bytes} B): {swap_out_time_sec:.6f} s")
                        self.swapped_out_activations.add(node_name) # Mark as swapped out
        return result

    def aggregate_stats(self, num_runs: int = 1) -> None:
        """
        Calculates average statistics from raw data collected over potentially
        multiple runs and then computes MuTWO metrics.
        Args:
            num_runs (int): The number of profiling runs performed, used for averaging swap times.
                            Defaults to 1 if only one run was done.
        """
        if num_runs <= 0:
            print("Warning: num_runs must be positive for aggregation. Defaulting to 1.")
            num_runs = 1

        # 1. Calculate Averages from Raw Data
        self.avg_run_times.clear()
        self.avg_peak_mem_node.clear()
        self.avg_memory_sizes.clear()
        self.avg_swap_times.clear()

        for name, times in self.run_times.items():
            try:
                self.avg_run_times[name] = statistics.median(times) if times else 0.0
            except statistics.StatisticsError:
                self.avg_run_times[name] = 0.0 # Handle case with no data

        for name, peaks in self.peak_mem_node.items():
            try:
                self.avg_peak_mem_node[name] = statistics.median(peaks) if peaks else 0.0
            except statistics.StatisticsError:
                self.avg_peak_mem_node[name] = 0.0

        for name, sizes in self.memory_sizes.items():
            try:
                self.avg_memory_sizes[name] = statistics.median(sizes) if sizes else 0.0
            except statistics.StatisticsError:
                self.avg_memory_sizes[name] = 0.0

        # Average total swap time per activation per run
        for name, event_times in self.swap_times.items():
            if event_times:
                total_swap_time_all_runs = sum(event_times)
                self.avg_swap_times[name] = total_swap_time_all_runs / num_runs
            else:
                 self.avg_swap_times[name] = 0.0


        # 2. Calculate MuTWO Metrics using Averaged Stats
        self.inactive_times.clear()
        self.recomp_times.clear()
        self.recomp_memory.clear()
        self.recompute_ratios.clear()

        name_to_node = {node.name: node for node in self.ranked_nodes}

        for act_name in self.activation_liveness.keys():
            liveness = self.activation_liveness[act_name]
            last_fw_rank = liveness["last_fw_use_rank"]
            first_bw_rank = liveness["first_bw_use_rank"]

            # 2a. Calculate inactive_time (using avg_run_times)
            inactive_time = 0.0
            if first_bw_rank > last_fw_rank and first_bw_rank != -1 and last_fw_rank != -1:
                for i in range(last_fw_rank + 1, first_bw_rank):
                    node = self.ranked_nodes[i]
                    inactive_time += self.avg_run_times.get(node.name, 0.0)
            self.inactive_times[act_name] = inactive_time

            # 2b. Calculate recomputation time (recomp_time) (using avg_run_times)
            # Approximation: sum avg times from creation to last_fw_use. Needs refinement.
            # TODO: Implement accurate dependency tracing for recomputation cost.
            # The current implementation sums times from creation to last forward use,
            # which is an overestimate.
            recomp_time = 0.0
            creation_rank = liveness["creation_rank"]
            if last_fw_rank != -1:
                 for i in range(creation_rank, last_fw_rank + 1):
                     node = self.ranked_nodes[i]
                     recomp_time += self.avg_run_times.get(node.name, 0.0)
            self.recomp_times[act_name] = recomp_time

            # 2c. Calculate recomputation memory (recomp_memory) (using avg_memory_sizes)
            # This is the average memory size of the activation itself.
            self.recomp_memory[act_name] = int(self.avg_memory_sizes.get(act_name, 0))

            # 2d. Calculate recompute ratio (using avg_swap_times)
            avg_swap_time = self.avg_swap_times.get(act_name, 0.0)
            if avg_swap_time > 1e-12: # Avoid division by zero or near-zero
                self.recompute_ratios[act_name] = recomp_time / avg_swap_time
            else:
                # If swap time is negligible/zero, recompute is only worthwhile if free
                self.recompute_ratios[act_name] = float('inf') if recomp_time > 1e-12 else 0.0

    def reset_stats(self) -> None:
        """Clears all collected runtime statistics and calculated metrics."""
        # Raw data lists
        self.run_times.clear()
        self.peak_mem_node.clear()
        self.memory_sizes.clear()
        self.swap_times.clear()
        self.swapped_out_activations.clear() # Reset state for next run
        # Averaged data dictionaries
        self.avg_run_times.clear()
        self.avg_peak_mem_node.clear()
        self.avg_memory_sizes.clear()
        self.avg_swap_times.clear()
        # MuTWO metrics dictionaries
        self.inactive_times.clear()
        self.recomp_times.clear()
        self.recomp_memory.clear()
        self.recompute_ratios.clear()
        # Note: Global CUDA peak memory is not reset here, only per-node in run_node.
        # Call torch.cuda.reset_peak_memory_stats() externally if needed before a full run.

    def print_stats(self) -> None:
        """Prints the aggregated statistics and metrics based on averaged values."""

        import math # Lazy import for format_bytes
        from collections import defaultdict # For peak memory calculation

        def format_bytes(size_bytes):
            if size_bytes < 0: size_bytes = 0 # Handle potential negative averages if lists were empty
            if size_bytes == 0: return "0 B"
            size_name = ("B", "KiB", "MiB", "GiB", "TiB", "PiB", "EiB", "ZiB", "YiB")
            i = int(math.floor(math.log(size_bytes, 1024))) if size_bytes > 0 else 0
            p = math.pow(1024, i)
            s = round(size_bytes / p, 2)
            return f"{s} {size_name[i]}"

        print("\n--- Aggregated Statistics ---")
        name_to_node = {node.name: node for node in self.ranked_nodes} # Ensure name_to_node is available

        # --- Per-Node Average Stats ---
        print("\n[Per-Node Average Statistics]")
        if not self.avg_run_times:
             print("No node statistics collected.")
        else:
            print(f"{'Node Name':<40} {'Avg Run Time (s)':<20} {'Avg Peak Memory':<20}")
            print("-" * 80)
            # Sort by rank for logical order
            sorted_node_names = sorted(self.avg_run_times.keys(), key=lambda name: self.node_ranks.get(name_to_node.get(name), float('inf')))

            for node_name in sorted_node_names:
                avg_time = self.avg_run_times.get(node_name, 0.0)
                avg_mem = self.avg_peak_mem_node.get(node_name, 0.0)
                print(f"{node_name:<40} {avg_time:<20.6f} {format_bytes(int(avg_mem)):<20}")

        # --- Per-Activation Metrics ---
        print("\n[Per-Activation MuTWO Metrics]")
        if not self.activation_liveness:
             print("No activations found or metrics calculated.")
        else:
            print(f"{'Activation Name':<30} {'Avg Mem Size':<15} {'Inactive Time (s)':<20} {'Avg Swap Time (s)':<20} {'Recomp Time (s)':<20} {'Recomp Memory':<15} {'Recomp Ratio':<15}")
            print("-" * 135)
            sorted_act_names = sorted(self.activation_liveness.keys(), key=lambda name: self.activation_liveness[name]['creation_rank'])

            for act_name in sorted_act_names:
                avg_mem = self.avg_memory_sizes.get(act_name, 0.0)
                inactive_t = self.inactive_times.get(act_name, 0.0)
                avg_swap_t = self.avg_swap_times.get(act_name, 0.0)
                recomp_t = self.recomp_times.get(act_name, 0.0)
                recomp_mem = self.recomp_memory.get(act_name, 0) # Already int from aggregation
                recomp_ratio = self.recompute_ratios.get(act_name, float('inf'))

                ratio_str = f"{recomp_ratio:.4f}" if recomp_ratio != float('inf') else "inf"

                print(f"{act_name:<30} {format_bytes(int(avg_mem)):<15} {inactive_t:<20.6f} {avg_swap_t:<20.6f} {recomp_t:<20.6f} {format_bytes(recomp_mem):<15} {ratio_str:<15}")

        # --- Overall Stats ---
        print("\n[Overall Statistics]")
        total_time = sum(self.avg_run_times.values())
        print(f"Total Estimated Execution Time (Sum of Avg Node Times): {total_time:.6f} s")

        # --- Peak Memory Breakdown ---
        print("\n[Peak Memory Breakdown (Estimate)]")

        # TODO: Calculate these accurately if possible by storing param/grad sizes in __init__
        total_param_mem = 0 # sum(self.param_sizes.values())
        total_grad_mem = 0 # sum(self.grad_sizes.values()) # Or estimate based on params
        total_optimizer_mem = 0 # Needs specific optimizer info

        # Calculate peak activation memory based on liveness and avg sizes
        peak_activation_mem = 0
        if self.activation_liveness and self.avg_memory_sizes:
            max_concurrent_mem = 0
            live_activations_at_rank: Dict[int, Set[str]] = defaultdict(set)

            # Determine live intervals
            for act_name, liveness in self.activation_liveness.items():
                create_rank = liveness['creation_rank']
                # Activation is live from creation until last use (either fw or bw)
                last_use_rank = max(liveness['last_fw_use_rank'], liveness['last_bw_use_rank'])
                if last_use_rank == -1: # Handle cases where only created, not used?
                    last_use_rank = create_rank

                # Ensure ranks are valid before adding to dict
                if create_rank >= 0 and last_use_rank >= create_rank:
                    for rank in range(create_rank, last_use_rank + 1):
                        live_activations_at_rank[rank].add(act_name)

            # Find peak memory sum across ranks
            if live_activations_at_rank: # Check if any activations were live
                max_rank = max(live_activations_at_rank.keys()) if live_activations_at_rank else -1
                for rank in range(max_rank + 1):
                    current_concurrent_mem = 0
                    for act_name in live_activations_at_rank[rank]:
                        current_concurrent_mem += self.avg_memory_sizes.get(act_name, 0)
                    max_concurrent_mem = max(max_concurrent_mem, current_concurrent_mem)

            peak_activation_mem = int(max_concurrent_mem)
        else:
             print("  (Skipping peak activation calculation: No liveness/size data)")


        print(f"  - Parameters:          {format_bytes(total_param_mem)} (Requires explicit calculation)")
        print(f"  - Peak Activations:    {format_bytes(peak_activation_mem)}")
        print(f"  - Gradients:           {format_bytes(total_grad_mem)} (Requires explicit calculation or estimation)")
        print(f"  - Optimizer States:    {format_bytes(total_optimizer_mem)} (Requires optimizer analysis)")

        estimated_total_peak = total_param_mem + peak_activation_mem + total_grad_mem + total_optimizer_mem
        print(f"  -----------------------------")
        print(f"  Estimated Peak (Sum):  {format_bytes(estimated_total_peak)}")

        max_node_peak = max(self.avg_peak_mem_node.values()) if self.avg_peak_mem_node else 0
        print(f"  Max Avg Per-Node Peak: {format_bytes(int(max_node_peak))} (Indicates peak during a single op)")

        print("\n--- End Statistics ---")

    def save_stats_to_csv(self, filename_prefix: str = "profiler_stats") -> None:
        """Saves the aggregated node and activation statistics to CSV files."""
        if not self.avg_run_times:
            print("Warning: No aggregated stats found to save to CSV.")
            return

        name_to_node = {node.name: node for node in self.ranked_nodes}

        # --- Save Node Statistics ---
        node_csv_filename = f"{filename_prefix}_node_stats.csv"
        try:
            with open(node_csv_filename, 'w', newline='') as csvfile:
                fieldnames = ['rank', 'node_name', 'node_type', 'gtype', 'avg_run_time_s', 'avg_peak_mem_bytes'] # Added gtype
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                writer.writeheader()
                sorted_node_names = sorted(self.avg_run_times.keys(), key=lambda name: self.node_ranks.get(name_to_node.get(name), float('inf')))

                for node_name in sorted_node_names:
                    node = name_to_node.get(node_name)
                    if not node: continue

                    rank = self.node_ranks.get(node, -1)
                    node_type = self.node_types.get(node_name, NodeType.OTHER)
                    gtype = self.node_gtypes.get(node_name, "unknown") # Get gtype
                    avg_time = self.avg_run_times.get(node_name, 0.0)
                    avg_mem = self.avg_peak_mem_node.get(node_name, 0.0)

                    writer.writerow({
                        'rank': rank,
                        'node_name': node_name,
                        'node_type': node_type.value,
                        'gtype': gtype, # Added gtype
                        'avg_run_time_s': avg_time,
                        'avg_peak_mem_bytes': int(avg_mem)
                    })
            print(f"Node statistics saved to {node_csv_filename}")
        except IOError as e:
            print(f"Error saving node statistics to {node_csv_filename}: {e}")


        # --- Save Activation Statistics ---
        activation_csv_filename = f"{filename_prefix}_activation_stats.csv"
        if not self.activation_liveness:
            print("No activation statistics to save.")
            return

        try:
            with open(activation_csv_filename, 'w', newline='') as csvfile:
                fieldnames = [
                    'activation_name', 'creation_rank', 'last_fw_use_rank', 'first_bw_use_rank', 'last_bw_use_rank',
                    'avg_mem_size_bytes', 'inactive_time_s', 'avg_swap_time_s', 'recomp_time_s',
                    'recomp_memory_bytes', 'recompute_ratio'
                ]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()

                sorted_act_names = sorted(self.activation_liveness.keys(), key=lambda name: self.activation_liveness[name]['creation_rank'])

                for act_name in sorted_act_names:
                    liveness = self.activation_liveness[act_name]
                    avg_mem = self.avg_memory_sizes.get(act_name, 0.0)
                    inactive_t = self.inactive_times.get(act_name, 0.0)
                    avg_swap_t = self.avg_swap_times.get(act_name, 0.0)
                    recomp_t = self.recomp_times.get(act_name, 0.0)
                    recomp_mem = self.recomp_memory.get(act_name, 0)
                    recomp_ratio = self.recompute_ratios.get(act_name, float('inf'))

                    writer.writerow({
                        'activation_name': act_name,
                        'creation_rank': liveness['creation_rank'],
                        'last_fw_use_rank': liveness['last_fw_use_rank'],
                        'first_bw_use_rank': liveness['first_bw_use_rank'],
                        'last_bw_use_rank': liveness['last_bw_use_rank'],
                        'avg_mem_size_bytes': int(avg_mem),
                        'inactive_time_s': inactive_t,
                        'avg_swap_time_s': avg_swap_t,
                        'recomp_time_s': recomp_t,
                        'recomp_memory_bytes': recomp_mem,
                        'recompute_ratio': recomp_ratio if recomp_ratio != float('inf') else 'inf'
                    })
            print(f"Activation statistics saved to {activation_csv_filename}")
        except IOError as e:
            print(f"Error saving activation statistics to {activation_csv_filename}: {e}")

    def plot_stats(self, filename_prefix: str = "profiler_plots", top_n: int = 20) -> None:
        """Generates and saves plots for key profiling statistics."""
        if not self.avg_run_times:
            print("Warning: No aggregated stats found to generate plots.")
            return

        name_to_node = {node.name: node for node in self.ranked_nodes}

        # --- Plot 1: Top N Nodes by Average Run Time ---
        try:
            # Sort nodes by average run time, descending
            sorted_nodes_by_time = sorted(self.avg_run_times.items(), key=lambda item: item[1], reverse=True)
            top_nodes_time = sorted_nodes_by_time[:top_n]
            node_names_time = [item[0] for item in top_nodes_time]
            run_times = [item[1] for item in top_nodes_time]

            if node_names_time: # Check if there's data to plot
                plt.figure(figsize=(12, max(6, len(node_names_time) * 0.4))) # Adjust height based on number of bars
                plt.barh(node_names_time, run_times, color='skyblue')
                plt.xlabel("Average Run Time (seconds)")
                plt.ylabel("Node Name")
                plt.title(f"Top {len(node_names_time)} Nodes by Average Run Time")
                plt.gca().invert_yaxis() # Display top node at the top
                plt.tight_layout()
                plot_filename_time = f"{filename_prefix}_node_runtime.png"
                plt.savefig(plot_filename_time)
                plt.close() # Close the figure to free memory
                print(f"Node runtime plot saved to {plot_filename_time}")
            else:
                 print("No node runtime data to plot.")

        except Exception as e:
            print(f"Error generating node runtime plot: {e}")

        # --- Plot 2: Top N Nodes by Average Peak Memory ---
        try:
            # Sort nodes by average peak memory, descending
            sorted_nodes_by_mem = sorted(self.avg_peak_mem_node.items(), key=lambda item: item[1], reverse=True)
            top_nodes_mem = sorted_nodes_by_mem[:top_n]
            node_names_mem = [item[0] for item in top_nodes_mem]
            peak_mems = [item[1] / (1024**2) for item in top_nodes_mem] # Convert to MiB

            if node_names_mem: # Check if there's data to plot
                plt.figure(figsize=(12, max(6, len(node_names_mem) * 0.4)))
                plt.barh(node_names_mem, peak_mems, color='lightcoral')
                plt.xlabel("Average Peak Memory per Node (MiB)")
                plt.ylabel("Node Name")
                plt.title(f"Top {len(node_names_mem)} Nodes by Average Peak Memory")
                plt.gca().invert_yaxis()
                plt.tight_layout()
                plot_filename_mem = f"{filename_prefix}_node_peak_memory.png"
                plt.savefig(plot_filename_mem)
                plt.close()
                print(f"Node peak memory plot saved to {plot_filename_mem}")
            else:
                 print("No node peak memory data to plot.")

        except Exception as e:
            print(f"Error generating node peak memory plot: {e}")


        # --- Plot 3: Top N Activations by Average Memory Size ---
        if not self.activation_liveness:
            print("No activation statistics to plot.")
            return
        try:
            # Sort activations by average memory size, descending
            sorted_acts_by_mem = sorted(self.avg_memory_sizes.items(), key=lambda item: item[1], reverse=True)
            # Filter only activations
            act_mems = [(name, size) for name, size in sorted_acts_by_mem if self.node_types.get(name) == NodeType.ACT]
            top_acts_mem = act_mems[:top_n]

            act_names_mem = [item[0] for item in top_acts_mem]
            act_sizes_mib = [item[1] / (1024**2) for item in top_acts_mem] # Convert to MiB

            if act_names_mem: # Check if there's data to plot
                plt.figure(figsize=(12, max(6, len(act_names_mem) * 0.4)))
                plt.barh(act_names_mem, act_sizes_mib, color='lightgreen')
                plt.xlabel("Average Activation Memory Size (MiB)")
                plt.ylabel("Activation Name")
                plt.title(f"Top {len(act_names_mem)} Activations by Average Memory Size")
                plt.gca().invert_yaxis()
                plt.tight_layout()
                plot_filename_act_mem = f"{filename_prefix}_activation_memory_size.png"
                plt.savefig(plot_filename_act_mem)
                plt.close()
                print(f"Activation memory size plot saved to {plot_filename_act_mem}")
            else:
                 print("No activation memory data to plot.")

        except Exception as e:
            print(f"Error generating activation memory size plot: {e}")

        # --- Plot 4: Top N Activations by Inactive Time ---
        try:
            # Sort activations by inactive time, descending
            sorted_acts_by_inactive_time = sorted(self.inactive_times.items(), key=lambda item: item[1], reverse=True)
            top_acts_inactive = sorted_acts_by_inactive_time[:top_n]

            act_names_inactive = [item[0] for item in top_acts_inactive]
            inactive_times = [item[1] for item in top_acts_inactive]

            if act_names_inactive: # Check if there's data to plot
                plt.figure(figsize=(12, max(6, len(act_names_inactive) * 0.4)))
                plt.barh(act_names_inactive, inactive_times, color='gold')
                plt.xlabel("Inactive Time (seconds)")
                plt.ylabel("Activation Name")
                plt.title(f"Top {len(act_names_inactive)} Activations by Inactive Time")
                plt.gca().invert_yaxis()
                plt.tight_layout()
                plot_filename_act_inactive = f"{filename_prefix}_activation_inactive_time.png"
                plt.savefig(plot_filename_act_inactive)
                plt.close()
                print(f"Activation inactive time plot saved to {plot_filename_act_inactive}")
            else:
                 print("No activation inactive time data to plot.")

        except Exception as e:
            print(f"Error generating activation inactive time plot: {e}")

# --- Plot 5: Memory vs. Execution Rank ---
        try:
            if self.ranked_nodes and self.avg_peak_mem_node:
                ranks = [self.node_ranks[node] for node in self.ranked_nodes]
                # Get peak memory for each node by its rank, convert to MiB
                # Ensure we only try to get memory for nodes that have entries in avg_peak_mem_node
                peak_mems_mib = []
                valid_ranks_for_plot = []
                for node in self.ranked_nodes:
                    if node.name in self.avg_peak_mem_node:
                        peak_mems_mib.append(self.avg_peak_mem_node[node.name] / (1024**2))
                        valid_ranks_for_plot.append(self.node_ranks[node])

                if not valid_ranks_for_plot: # Check if any valid data points exist
                    print("No valid peak memory data to plot for Memory vs. Rank.")
                else:
                    plt.figure(figsize=(15, 7))
                    plt.plot(valid_ranks_for_plot, peak_mems_mib, marker='o', linestyle='-', label="Peak Memory per Node")
                    
                    # Add FW/BW separator lines if they exist
                    if self.sep_fw_end_rank != -1:
                        plt.axvline(x=self.sep_fw_end_rank, color='red', linestyle='--', label=f'FW/BW Sep (Rank {self.sep_fw_end_rank})')
                    if self.sep_bw_start_rank != -1 and self.sep_bw_start_rank != self.sep_fw_end_rank : # Avoid double line if they are same
                         plt.axvline(x=self.sep_bw_start_rank, color='orange', linestyle='--', label=f'BW Start Sep (Rank {self.sep_bw_start_rank})')


                    # Add GPU Memory Limit line
                    if hasattr(self, 'GPU_MEMORY_LIMIT_MIB'):
                        plt.axhline(y=self.GPU_MEMORY_LIMIT_MIB, color='green', linestyle=':', label=f'GPU Limit ({self.GPU_MEMORY_LIMIT_MIB} MiB)')

                    plt.xlabel("Node Execution Rank (Topological Order)")
                    plt.ylabel("Peak Memory (MiB)")
                    plt.title("Peak Memory vs. Node Execution Rank")
                    plt.legend()
                    plt.grid(True)
                    plt.tight_layout()
                    plot_filename_mem_rank = f"{filename_prefix}_memory_vs_rank.png"
                    plt.savefig(plot_filename_mem_rank)
                    plt.close()
                    print(f"Memory vs. Rank plot saved to {plot_filename_mem_rank}")
            else:
                print("Not enough data to generate Memory vs. Rank plot (ranked_nodes or avg_peak_mem_node is empty).")
        except Exception as e:
            print(f"Error generating Memory vs. Rank plot: {e}")
