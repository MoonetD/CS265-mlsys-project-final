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
    """Enum representing the different operation types in an FX graph."""
    CALL_FUNCTION = "call_function"  # Function calls like torch.add
    CALL_MODULE = "call_module"      # Module calls like self.linear
    CALL_METHOD = "call_method"      # Method calls like tensor.view
    GET_ATTR = "get_attr"            # Attribute access like self.weight
    OUTPUT = "output"                # Graph output nodes
    PLACEHOLDER = "placeholder"      # Input placeholders


class NodeType(str, Enum): # Changed back to str Enum for consistency if needed elsewhere
    """
    NodeType is a enum that records the type of the tensors in the graph.
    """
    PARAM = "parameter"        # Model parameters (weights, biases)
    ACT = "activation"         # Activations (tensors created in forward, used in backward)
    GRAD = "gradient"          # Gradients of parameters
    OPT_STATE = "optimizer_state"  # Added for optimizer state tensors (momentum, etc.)
    OTHER = "other"            # Other tensors that don't fit the categories above


class GraphProfiler(fx.Interpreter):
    """
    Profiler for PyTorch FX graphs that collects detailed execution statistics.
    
    This profiler extends the FX Interpreter to track memory usage, execution time,
    and tensor liveness information during model execution. It's designed to gather
    data needed for activation checkpointing decisions.
    """
    def __init__(self, module: fx.GraphModule, garbage_collect_values: bool = True):
        """
        Initialize the GraphProfiler with a traced module.
        
        Args:
            module: The FX GraphModule to profile
            garbage_collect_values: Whether to garbage collect values after they're no longer needed
        """
        super().__init__(module, garbage_collect_values)

        # Static analysis attributes
        self.sep_fw_end_rank: int = -1      # Rank of the separator node marking the end of forward pass
        self.sep_bw_start_rank: int = -1    # Rank of the separator node marking the start of backward pass
        self.node_types: Dict[str, NodeType] = {}  # Maps node names to their tensor type classification
        self.node_gtypes: Dict[str, str] = {} # Maps node names to graph section ("forward", "backward", etc.)
        self.activation_liveness: Dict[str, Dict[str, int]] = {}  # Tracks when activations are created and used
        # Remove the mapping between reported and original names - we'll use original names consistently

        self.param_node_names: Set[str] = set()  # Set of node names that represent parameters
        self.grad_node_names: Set[str] = set()   # Set of node names that represent gradients
        # Store parameter and gradient sizes
        self.param_sizes: Dict[str, int] = {}    # Maps parameter names to their memory sizes
        self.grad_sizes: Dict[str, int] = {}     # Maps gradient names to their memory sizes
        # Store active memory before each node execution
        self.active_mem_node: Dict[str, List[int]] = defaultdict(list)  # Active memory before each node execution
        self.median_active_mem_node: Dict[str, float] = {}  # Median active memory across runs

        self.node_ranks: Dict[fx.Node, int] = {}  # Maps nodes to their topological rank
        self.ranked_nodes: List[fx.Node] = []     # List of nodes in topological order

        # Runtime profiling attributes (Raw data collected per run, stored as lists)
        self.run_times: Dict[str, List[float]] = defaultdict(list)  # Execution time for each node
        self.peak_mem_node: Dict[str, List[int]] = defaultdict(list)  # Peak memory during node execution
        self.memory_sizes: Dict[str, List[int]] = defaultdict(list)  # Activation output sizes per run
        
        # Averaged runtime stats (Calculated after aggregation)
        self.median_run_times: Dict[str, float] = {}      # Median execution time across runs
        self.median_peak_mem_node: Dict[str, float] = {}  # Median peak memory across runs
        self.median_memory_sizes: Dict[str, float] = {}   # Median memory sizes across runs

        # MuTWO specific metrics (Calculated after aggregation using averaged stats)
        self.inactive_times: Dict[str, float] = {}    # Time between last forward use and first backward use
        self.recomp_times: Dict[str, float] = {}      # Estimated time to recompute an activation
        self.recomp_memory: Dict[str, int] = {}       # Estimated memory needed to recompute an activation

        self.GPU_MEMORY_LIMIT_MIB = 4 * 1024  # Fixed 4 GiB memory limit for activation checkpointing

        # --- First Pass: Rank nodes, find boundaries, identify initial Params/Grads ---
        _fused_adam_node: fx.Node | None = None  # Track the fused Adam optimizer node if present
        for rank, node in enumerate(self.module.graph.nodes):
            # Assign topological ranks to each node
            self.node_ranks[node] = rank
            self.ranked_nodes.append(node)

            if node.op == OP.CALL_FUNCTION:
                # Identify special separator nodes and optimizer nodes
                if node.target == torch.ops.separator.sep.default:
                    # This marks the end of forward pass
                    self.sep_fw_end_rank = rank
                elif node.target == torch.ops.separator.sep_backward.default:
                    # This marks the start of backward pass
                    self.sep_bw_start_rank = rank
                elif node.target == torch.ops.aten._fused_adam.default:
                    # Track the Adam optimizer node for parameter/gradient identification
                    _fused_adam_node = node

        # Identify parameter names from module parameters and compute sizes
        for param_name_dot, param in self.module.named_parameters():
            # Convert PyTorch parameter names to FX node naming convention
            fx_node_name = param_name_dot.replace('.', '_')  # FX naming convention
            self.param_node_names.add(fx_node_name)
            # Store parameter size in bytes
            self.param_sizes[fx_node_name] = param.element_size() * param.nelement()

        # Identify gradient names from _fused_adam_node (if found and args are ListConstruct)
        if _fused_adam_node and len(_fused_adam_node.args) > 1:
            # Per comments, arg 0 is params list, arg 1 is grads list
            # This assumes these args are nodes that are 'prim::ListConstruct'
            param_list_provider_node = _fused_adam_node.args[0]
            grad_list_provider_node = _fused_adam_node.args[1]

            # Extract parameter nodes from the parameter list
            if isinstance(param_list_provider_node, fx.Node) and \
               param_list_provider_node.op == OP.CALL_FUNCTION and \
               str(param_list_provider_node.target) == 'prim::ListConstruct':
                for p_node_in_list in param_list_provider_node.args:
                    if isinstance(p_node_in_list, fx.Node):
                        self.param_node_names.add(p_node_in_list.name)

            # Extract gradient nodes from the gradient list
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

            # Determine graph section type (forward, backward, etc.)
            if self.sep_fw_end_rank != -1 and rank <= self.sep_fw_end_rank:
                # Nodes before or at the forward separator
                self.node_gtypes[node_name] = "forward"
            elif self.sep_bw_start_rank != -1 and rank >= self.sep_bw_start_rank:
                # Nodes after or at the backward separator
                self.node_gtypes[node_name] = "backward"
            elif self.sep_fw_end_rank == -1 and self.sep_bw_start_rank != -1 and rank < self.sep_bw_start_rank:
                # If no explicit FW end, but BW start exists, assume FW before BW
                self.node_gtypes[node_name] = "forward"
            elif self.sep_fw_end_rank != -1 and self.sep_bw_start_rank == -1 and rank > self.sep_fw_end_rank:
                 # If no explicit BW start, but FW end exists, assume optimizer/other after FW
                self.node_gtypes[node_name] = "optimizer/other"
            elif self.sep_fw_end_rank == -1 and self.sep_bw_start_rank == -1:
                # No separators, assume all 'forward' for simplicity or 'unknown'
                self.node_gtypes[node_name] = "unknown"
            else: # Between fw_end and bw_start, or other complex cases
                self.node_gtypes[node_name] = "intermediate/other"

            # Enhanced parameter identification
            is_param = False
            if node_name in self.param_node_names:
                is_param = True
            elif node.op == OP.GET_ATTR:
                # Parameters are often accessed via get_attr
                is_param = True
            elif node_name.lower().endswith(('weight', 'bias', 'param', 'parameter')):
                # Common naming patterns for parameters
                is_param = True
            # Check for convolution, linear, or batch norm operations which use parameters
            elif node.op == OP.CALL_FUNCTION and hasattr(node, 'target'):
                target_str = str(node.target).lower()
                if any(op in target_str for op in ['conv', 'linear', 'batch_norm', 'bn']):
                    # Check if this node has args that could be parameters
                    for arg in node.args:
                        if isinstance(arg, fx.Node) and arg.op == OP.GET_ATTR:
                            # Mark the argument as a parameter
                            self.param_node_names.add(arg.name)
                            self.node_types[arg.name] = NodeType.PARAM
            
            # Enhanced gradient identification
            is_grad = False
            if node_name in self.grad_node_names:
                is_grad = True
            elif node_name.lower().find('grad') >= 0 or node_name.lower().find('derivative') >= 0:
                # Common naming patterns for gradients
                is_grad = True
            elif self.node_gtypes[node_name] == "backward":
                # Check if this is a backward operation that produces gradients
                if node.op == OP.CALL_FUNCTION and hasattr(node, 'target'):
                    target_str = str(node.target).lower()
                    if "_backward" in target_str or "gradient" in target_str or "grad" in target_str:
                        is_grad = True
            
            # Enhanced optimizer state identification
            is_opt_state = False
            if node.op == OP.CALL_FUNCTION and hasattr(node, 'target'):
                target_str = str(node.target).lower()
                # Expanded list of optimizer-related function names
                opt_keywords = [
                    "aten._fused_", "adam", "sgd", "adagrad", "rmsprop", "momentum",
                    "optimizer", "step", "update", "learning_rate", "lr", "decay"
                ]
                if any(keyword in target_str for keyword in opt_keywords):
                    is_opt_state = True
                    
                # Special handling for optimizer operations
                if "adam" in target_str or "sgd" in target_str or "optim" in target_str:
                    # Mark this node as optimizer state
                    is_opt_state = True
                    
                    # Check the arguments to this optimizer operation
                    for i, arg in enumerate(node.args):
                        if isinstance(arg, fx.Node):
                            # First argument is often parameters
                            if i == 0 and not arg.name in self.param_node_names:
                                self.param_node_names.add(arg.name)
                                self.node_types[arg.name] = NodeType.PARAM
                            # Second argument is often gradients
                            elif i == 1 and not arg.name in self.grad_node_names:
                                self.grad_node_names.add(arg.name)
                                self.node_types[arg.name] = NodeType.GRAD
                            # Other arguments might be optimizer states
                            elif i > 1:
                                self.node_types[arg.name] = NodeType.OPT_STATE
                                
            # Check for optimizer state naming patterns
            elif any(keyword in node_name.lower() for keyword in ["momentum", "velocity", "adam_", "optimizer_state"]):
                is_opt_state = True
                
            # Check if node is in optimizer section and not a parameter or gradient
            elif self.node_gtypes[node_name] == "optimizer/other" and not is_param and not is_grad:
                is_opt_state = True

            # Classify nodes by type with enhanced logic
            if is_param:
                self.node_types[node_name] = NodeType.PARAM
            elif is_grad:
                self.node_types[node_name] = NodeType.GRAD
            elif is_opt_state:
                self.node_types[node_name] = NodeType.OPT_STATE
            else:
                is_activation = False
                if node.op not in [OP.PLACEHOLDER, OP.OUTPUT]: # Activations are not placeholders or outputs
                    # Created in forward pass (or at its boundary)
                    created_in_fwd = (self.sep_fw_end_rank == -1 or rank <= self.sep_fw_end_rank)

                    # Check if used in backward pass
                    used_in_bwd = False
                    if self.sep_bw_start_rank != -1: # Backward pass exists
                        for user_node in node.users:
                            if self.node_ranks[user_node] >= self.sep_bw_start_rank:
                                used_in_bwd = True
                                break

                    # An activation is created in forward and used in backward
                    if created_in_fwd and used_in_bwd:
                        is_activation = True

                # Set the node type based on our classification
                if is_activation:
                    self.node_types[node_name] = NodeType.ACT
                else:
                    self.node_types[node_name] = NodeType.OTHER

        # --- Third Pass: Activation Liveness Analysis ---
        for node in self.ranked_nodes:
            node_name = node.name # Always use the original FX node name
            if self.node_types.get(node_name) == NodeType.ACT:
                # For each activation, track when it's created and used
                creation_rank = self.node_ranks[node]
                last_fw_use_rank = -1  # Default if not used in forward
                first_bw_use_rank = float('inf')  # Will be replaced with actual rank
                last_bw_use_rank = -1  # Default if not used in backward

                # Analyze each node that uses this activation
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

                # If activation is never used in backward, reset the infinity value
                if first_bw_use_rank == float('inf'): # Should not happen for ACT type by definition
                    first_bw_use_rank = -1

                # Store liveness information for this activation
                self.activation_liveness[node_name] = { # Use original node name as key
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
        """
        Run the graph with the given inputs.
        
        Args:
            *args: Input arguments to the model
            initial_env: Initial environment mapping nodes to values
            enable_io_processing: Whether to enable input/output processing
            
        Returns:
            The output of the model
        """
        return super().run(
            *args, initial_env=initial_env, enable_io_processing=enable_io_processing
        )

    def run_node(self, n: fx.Node) -> Any:
        """
        Execute a single node and collect profiling information.
        
        This method wraps the parent's run_node method with timing and memory
        tracking to collect performance metrics for each node.
        
        Args:
            n: The node to execute
            
        Returns:
            The result of executing the node
        """
        node_name = n.name
        current_rank = self.node_ranks[n]

        # Track active memory before node execution
        pre_mem = torch.cuda.memory_allocated()
        self.active_mem_node[node_name].append(pre_mem)

        # Timing and Memory Measurement (Around node execution)
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        # Reset peak memory stats for the current device to measure peak for this node
        # Note: This measures peak for the *device*. If other operations run concurrently
        # on the same device but outside this profiler's control, they might affect this.
        # However, for typical single-stream model execution, this should be representative.
        torch.cuda.reset_peak_memory_stats()

        # Record start time, execute node, record end time
        start_event.record()
        result = super().run_node(n)
        end_event.record()

        # Wait for all CUDA operations to complete for accurate timing & memory measurements
        torch.cuda.synchronize()

        # Store run time in seconds (convert from milliseconds)
        # Convert elapsed time from milliseconds to seconds and store it
        run_time_ms = start_event.elapsed_time(end_event)
        self.run_times[node_name].append(run_time_ms / 1000.0) # Append time in seconds

        # Store the maximum memory allocated during this node's execution
        # This captures the peak memory usage that occurred while running this node
        self.peak_mem_node[node_name].append(torch.cuda.max_memory_allocated())

        # Calculate and store the memory size for all tensor types
        mem_size = 0
        node_type = self.node_types.get(node_name)
        
        # Process the result to calculate memory size
        if isinstance(result, torch.Tensor) and result.device.type == 'cuda':
            # Direct tensor result
            mem_size = result.element_size() * result.nelement()
            
            # Additional type detection based on result properties
            if node_type == NodeType.OTHER:
                # Try to identify parameters and gradients based on tensor properties
                if hasattr(result, 'requires_grad') and result.requires_grad:
                    # Parameters typically require gradients
                    self.node_types[node_name] = NodeType.PARAM
                    self.param_node_names.add(node_name)
                    node_type = NodeType.PARAM
                elif hasattr(result, 'grad_fn') and result.grad_fn is not None:
                    # Tensors with grad_fn are typically activations or gradients
                    if self.node_gtypes.get(node_name) == "backward":
                        self.node_types[node_name] = NodeType.GRAD
                        self.grad_node_names.add(node_name)
                        node_type = NodeType.GRAD
        else:
            # For complex outputs (like tuples), flatten the structure to process all tensors
            for t in torch.utils._pytree.tree_flatten(result)[0]:
                # Only count CUDA tensors (skip CPU tensors and non-tensor objects)
                if isinstance(t, torch.Tensor) and t.device.type == 'cuda':
                    # Calculate tensor size in bytes (element size × number of elements)
                    mem_size += t.element_size() * t.nelement()
                    
                    # Try to identify parameters and gradients in complex outputs
                    if node_type == NodeType.OTHER:
                        if hasattr(t, 'requires_grad') and t.requires_grad:
                            self.node_types[node_name] = NodeType.PARAM
                            self.param_node_names.add(node_name)
                            node_type = NodeType.PARAM
                        elif hasattr(t, 'grad_fn') and t.grad_fn is not None:
                            if self.node_gtypes.get(node_name) == "backward":
                                self.node_types[node_name] = NodeType.GRAD
                                self.grad_node_names.add(node_name)
                                node_type = NodeType.GRAD
        
        # Record memory size for all node types
        if mem_size > 0:
            self.memory_sizes[node_name].append(mem_size)
            
            # Store sizes in type-specific dictionaries for easier analysis
            if node_type == NodeType.PARAM:
                self.param_sizes[node_name] = mem_size
            elif node_type == NodeType.GRAD:
                self.grad_sizes[node_name] = mem_size
        
        # Enhanced gradient tracking during backward pass
        if self.node_gtypes.get(node_name) == "backward":
            # Check if this node produces gradients
            if node_type == NodeType.GRAD:
                # Store gradient size directly
                if mem_size > 0:
                    self.grad_sizes[node_name] = mem_size
            
            # Also check for parameters with gradients
            for param_name in self.param_node_names:
                # Check if parameter exists and has a gradient
                if param_name in self.env and isinstance(self.env[param_name], torch.Tensor) and self.env[param_name].grad is not None:
                    # Calculate and store the gradient size in bytes
                    grad_size = self.env[param_name].grad.element_size() * self.env[param_name].grad.nelement()
                    self.grad_sizes[param_name] = grad_size
                    
                    # Also add this gradient to the grad_node_names set for future reference
                    grad_node_name = f"{param_name}_grad"
                    self.grad_node_names.add(grad_node_name)
                    self.node_types[grad_node_name] = NodeType.GRAD
        
        # Check for optimizer operations
        if n.op == OP.CALL_FUNCTION and hasattr(n, 'target'):
            target_str = str(n.target).lower()
            if "adam" in target_str or "sgd" in target_str or "optim" in target_str or "step" in target_str:
                # This is an optimizer operation
                self.node_types[node_name] = NodeType.OPT_STATE
                
                # Check the environment for optimizer state tensors
                for env_name, env_val in self.env.items():
                    if isinstance(env_val, torch.Tensor) and env_val.device.type == 'cuda':
                        # Skip parameters and gradients we've already identified
                        if env_name in self.param_node_names or env_name in self.grad_node_names:
                            continue
                            
                        # This might be an optimizer state tensor
                        if env_name not in self.node_types or self.node_types[env_name] == NodeType.OTHER:
                            self.node_types[env_name] = NodeType.OPT_STATE

        return result

    def aggregate_stats(self, num_runs: int = 1) -> None:
        """
        Calculates average statistics from raw data collected over potentially
        multiple runs and then computes MuTWO metrics.
        
        The implementation includes an improved recomputation metrics calculation
        that ensures non-zero values for better activation checkpointing decisions.
        This uses a multi-method approach:
        1. Dependency tracing: Estimates dependencies between creation and last use
        2. Size-based estimation: Correlates activation size with computation cost
        3. Minimum threshold: Ensures all activations have a non-zero recomputation cost
        
        Args:
            num_runs (int): The number of profiling runs performed, used for averaging statistics.
                            Defaults to 1 if only one run was done.
        """
        if num_runs <= 0:
            print("Warning: num_runs must be positive for aggregation. Defaulting to 1.")
            num_runs = 1

        # 1. Calculate Averages from Raw Data
        # Clear previous aggregated statistics before recalculating
        self.median_run_times.clear()
        self.median_peak_mem_node.clear()
        self.median_memory_sizes.clear()
        self.median_active_mem_node.clear()

        # Calculate median run times for each node
        for name, times in self.run_times.items():
            try:
                # Use median instead of mean to reduce impact of outliers
                self.median_run_times[name] = statistics.median(times) if times else 0.0
            except statistics.StatisticsError:
                # Handle edge case where statistics.median fails (e.g., empty list)
                self.median_run_times[name] = 0.0 

        # Calculate median peak memory for each node
        for name, peaks in self.peak_mem_node.items():
            try:
                self.median_peak_mem_node[name] = statistics.median(peaks) if peaks else 0.0
            except statistics.StatisticsError:
                self.median_peak_mem_node[name] = 0.0

        # Calculate median memory sizes for activation tensors
        for name, sizes in self.memory_sizes.items():
            try:
                self.median_memory_sizes[name] = statistics.median(sizes) if sizes else 0.0
            except statistics.StatisticsError:
                self.median_memory_sizes[name] = 0.0
                
        # Calculate median active memory before each node's execution
        for name, active_mems in self.active_mem_node.items():
            try:
                self.median_active_mem_node[name] = statistics.median(active_mems) if active_mems else 0.0
            except statistics.StatisticsError:
                self.median_active_mem_node[name] = 0.0


        # 2. Calculate MuTWO Metrics using Averaged Stats
        # Clear previous metrics before recalculating
        self.inactive_times.clear()
        self.recomp_times.clear()
        self.recomp_memory.clear()

        # Create a mapping from node names to node objects for easier lookup
        name_to_node = {node.name: node for node in self.ranked_nodes}

        # Process each activation to calculate its metrics
        for act_name in self.activation_liveness.keys():
            # Get liveness information for this activation
            liveness = self.activation_liveness[act_name]
            last_fw_rank = liveness["last_fw_use_rank"]
            first_bw_rank = liveness["first_bw_use_rank"]

            # 2a. Calculate inactive_time - time between last forward use and first backward use
            inactive_time = 0.0
            if first_bw_rank > last_fw_rank and first_bw_rank != -1 and last_fw_rank != -1:
                # Sum up execution times of all nodes between last forward use and first backward use
                for i in range(last_fw_rank + 1, first_bw_rank):
                    node = self.ranked_nodes[i]
                    inactive_time += self.median_run_times.get(node.name, 0.0)
            self.inactive_times[act_name] = inactive_time

            # 2b. Calculate recomputation time (recomp_time) using a more robust approach
            # This estimates how long it would take to recompute this activation if checkpointed
            recomp_time = 0.0
            creation_rank = liveness["creation_rank"]
            creation_node = self.ranked_nodes[creation_rank]
            
            # Method 1: Trace direct dependencies from creation node to producing this activation
            if last_fw_rank != -1:
                # Start with the creation node's time
                recomp_time += self.median_run_times.get(creation_node.name, 0.0)
                
                # Add a portion of times for nodes between creation and last forward use
                # Using a dependency factor since not all nodes in this range directly contribute
                dependency_factor = 0.5  # Assume ~50% of nodes are dependencies on average
                for i in range(creation_rank + 1, last_fw_rank + 1):
                    node = self.ranked_nodes[i]
                    node_time = self.median_run_times.get(node.name, 0.0)
                    # Only count nodes with non-zero time to avoid skipping important operations
                    if node_time > 0:
                        recomp_time += node_time * dependency_factor
            
            # Method 2: Ensure minimum recomputation cost based on activation size
            # Larger activations typically require more computation to produce
            avg_mem_size = self.median_memory_sizes.get(act_name, 0.0)
            if avg_mem_size > 0:
                # Estimate minimum recomputation time based on activation size
                # Assume at least 1 microsecond per KB of activation data as a heuristic
                min_recomp_time = (avg_mem_size / 1024) * 1e-6
                recomp_time = max(recomp_time, min_recomp_time)
            
            # Method 3: Apply a minimum threshold to ensure no zero values
            # This prevents division-by-zero errors in optimization algorithms
            MIN_RECOMP_TIME = 1e-6  # 1 microsecond minimum
            recomp_time = max(recomp_time, MIN_RECOMP_TIME)
            
            # Store the final recomputation time estimate
            self.recomp_times[act_name] = recomp_time

            # 2c. Calculate recomputation memory (recomp_memory)
            # This is simply the memory size of the activation itself
            # It represents the memory that would be saved by not storing this activation
            self.recomp_memory[act_name] = int(self.median_memory_sizes.get(act_name, 0))

            # No recompute ratio calculation needed

    def reset_stats(self) -> None:
        """Clears all collected runtime statistics and calculated metrics."""
        # Raw data lists
        self.run_times.clear()          # Clear raw execution times for each node
        self.peak_mem_node.clear()      # Clear raw peak memory usage for each node
        self.memory_sizes.clear()       # Clear raw memory sizes for activations
        self.active_mem_node.clear()    # Clear raw active memory for each node
        
        # Averaged data dictionaries
        self.median_run_times.clear()           # Clear median execution times
        self.median_peak_mem_node.clear()       # Clear median peak memory usage
        self.median_memory_sizes.clear()        # Clear median memory sizes
        self.median_active_mem_node.clear()     # Clear median active memory
        
        # MuTWO metrics dictionaries
        self.inactive_times.clear()     # Clear inactive time metrics for activations
        self.recomp_times.clear()       # Clear recomputation time metrics
        self.recomp_memory.clear()      # Clear recomputation memory metrics
        # Note: Global CUDA peak memory is not reset here, only per-node in run_node.
        # Call torch.cuda.reset_peak_memory_stats() externally if needed before a full run.

    def print_stats(self) -> None:
        """Prints the aggregated statistics and metrics based on averaged values."""

        import math # Lazy import for format_bytes
        from collections import defaultdict # For peak memory calculation

        # Helper function to format byte values into human-readable form
        def format_bytes(size_bytes):
            if size_bytes < 0: size_bytes = 0 # Handle potential negative averages if lists were empty
            if size_bytes == 0: return "0 B"
            size_name = ("B", "KiB", "MiB", "GiB", "TiB", "PiB", "EiB", "ZiB", "YiB")
            i = int(math.floor(math.log(size_bytes, 1024))) if size_bytes > 0 else 0
            p = math.pow(1024, i)
            s = round(size_bytes / p, 2)
            return f"{s} {size_name[i]}"

        print("\n--- Aggregated Statistics ---")
        name_to_node = {node.name: node for node in self.ranked_nodes} # Create lookup dict for node names

        # --- Per-Node Average Stats Section ---
        print("\n[Per-Node Average Statistics]")
        if not self.median_run_times:
             print("No node statistics collected.")
        else:
            # Print header for node statistics table
            print(f"{'Node Name':<40} {'Avg Run Time (s)':<20} {'Avg Peak Memory':<20}")
            print("-" * 80)
            
            # Sort nodes by their execution rank for logical display order
            sorted_node_names = sorted(self.median_run_times.keys(), 
                                      key=lambda name: self.node_ranks.get(name_to_node.get(name), float('inf')))

            # Print each node's statistics
            for node_name in sorted_node_names:
                avg_time = self.median_run_times.get(node_name, 0.0)
                avg_mem = self.median_peak_mem_node.get(node_name, 0.0)
                print(f"{node_name:<40} {avg_time:<20.6f} {format_bytes(int(avg_mem)):<20}")

        # --- Per-Activation Metrics Section ---
        print("\n[Per-Activation MuTWO Metrics]")
        if not self.activation_liveness:
             print("No activations found or metrics calculated.")
        else:
            # Print header for activation metrics table
            print(f"{'Activation Name':<30} {'Avg Mem Size':<15} {'Inactive Time (s)':<20} {'Recomp Time (s)':<20} {'Recomp Memory':<15}")
            print("-" * 100)
            
            # Sort activations by their creation rank for logical display order
            sorted_act_names = sorted(self.activation_liveness.keys(), 
                                     key=lambda name: self.activation_liveness[name]['creation_rank'])

            # Print each activation's metrics
            for act_name in sorted_act_names:
                avg_mem = self.median_memory_sizes.get(act_name, 0.0)
                inactive_t = self.inactive_times.get(act_name, 0.0)
                recomp_t = self.recomp_times.get(act_name, 0.0)
                recomp_mem = self.recomp_memory.get(act_name, 0) # Already int from aggregation

                print(f"{act_name:<30} {format_bytes(int(avg_mem)):<15} {inactive_t:<20.6f} {recomp_t:<20.6f} {format_bytes(recomp_mem):<15}")

        # --- Overall Stats Section ---
        print("\n[Overall Statistics]")
        # Calculate total execution time by summing all node times
        total_time = sum(self.median_run_times.values())
        print(f"Total Estimated Execution Time (Sum of Avg Node Times): {total_time:.6f} s")

        # --- Peak Memory Breakdown Section ---
        print("\n[Peak Memory Breakdown (Estimate)]")

        # Calculate memory usage by different components
        total_param_mem = sum(self.param_sizes.values())  # Sum all parameter memory
        total_grad_mem = sum(self.grad_sizes.values())    # Sum all gradient memory
        total_optimizer_mem = 0 # Placeholder for optimizer memory (needs specific optimizer info)

        # Calculate peak activation memory based on liveness analysis
        peak_activation_mem = 0
        if self.activation_liveness and self.median_memory_sizes:
            max_concurrent_mem = 0
            # Track which activations are live at each rank
            live_activations_at_rank: Dict[int, Set[str]] = defaultdict(set)

            # Determine live intervals for each activation
            for act_name, liveness in self.activation_liveness.items():
                create_rank = liveness['creation_rank']
                # Activation is live from creation until last use (either fw or bw)
                last_use_rank = max(liveness['last_fw_use_rank'], liveness['last_bw_use_rank'])
                if last_use_rank == -1: # Handle cases where only created, not used
                    last_use_rank = create_rank

                # Ensure ranks are valid before adding to dict
                if create_rank >= 0 and last_use_rank >= create_rank:
                    # Mark this activation as live for all ranks in its lifetime
                    for rank in range(create_rank, last_use_rank + 1):
                        live_activations_at_rank[rank].add(act_name)

            # Find peak memory sum across all ranks
            if live_activations_at_rank: # Check if any activations were live
                max_rank = max(live_activations_at_rank.keys()) if live_activations_at_rank else -1
                # For each rank, calculate total memory of all live activations
                for rank in range(max_rank + 1):
                    current_concurrent_mem = 0
                    for act_name in live_activations_at_rank[rank]:
                        current_concurrent_mem += self.median_memory_sizes.get(act_name, 0)
                    # Update peak if current rank has more memory usage
                    max_concurrent_mem = max(max_concurrent_mem, current_concurrent_mem)

            peak_activation_mem = int(max_concurrent_mem)
        else:
             print("  (Skipping peak activation calculation: No liveness/size data)")

        # Print memory breakdown by component
        print(f"  - Parameters:          {format_bytes(total_param_mem)} (Requires explicit calculation)")
        print(f"  - Peak Activations:    {format_bytes(peak_activation_mem)}")
        print(f"  - Gradients:           {format_bytes(total_grad_mem)} (Requires explicit calculation or estimation)")
        print(f"  - Optimizer States:    {format_bytes(total_optimizer_mem)} (Requires optimizer analysis)")

        # Calculate and print total estimated peak memory
        estimated_total_peak = total_param_mem + peak_activation_mem + total_grad_mem + total_optimizer_mem
        print(f"  -----------------------------")
        print(f"  Estimated Peak (Sum):  {format_bytes(estimated_total_peak)}")

        # Print max per-node peak memory for comparison
        max_node_peak = max(self.median_peak_mem_node.values()) if self.median_peak_mem_node else 0
        print(f"  Max Avg Per-Node Peak: {format_bytes(int(max_node_peak))} (Indicates peak during a single op)")

        print("\n--- End Statistics ---")

    def save_stats_to_csv(self, filename_prefix: str = "profiler_stats") -> None:
        """Saves the aggregated node and activation statistics to CSV files."""
        if not self.median_run_times:
            print("Warning: No aggregated stats found to save to CSV.")
            return

        # Create lookup dictionary for node names
        name_to_node = {node.name: node for node in self.ranked_nodes}

        # --- Save Node Statistics to CSV ---
        node_csv_filename = f"{filename_prefix}_node_stats.csv"
        try:
            with open(node_csv_filename, 'w', newline='') as csvfile:
                # Define CSV columns
                fieldnames = ['rank', 'node_name', 'node_type', 'gtype', 'median_run_time_s', 
                             'median_peak_mem_bytes', 'median_active_mem_bytes', 'device']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                
                # Sort nodes by execution rank
                sorted_node_names = sorted(self.median_run_times.keys(), 
                                          key=lambda name: self.node_ranks.get(name_to_node.get(name), float('inf')))

                # Write each node's data to CSV
                for node_name in sorted_node_names:
                    node = name_to_node.get(node_name)
                    if not node: continue  # Skip if node not found

                    # Gather node information
                    rank = self.node_ranks.get(node, -1)
                    node_type = self.node_types.get(node_name, NodeType.OTHER)
                    gtype = self.node_gtypes.get(node_name, "unknown")
                    avg_time = self.median_run_times.get(node_name, 0.0)
                    avg_mem = self.median_peak_mem_node.get(node_name, 0.0)

                    # Get device information for the node
                    device = "unknown"
                    if node in self.env and isinstance(self.env[node], torch.Tensor):
                        device = str(self.env[node].device)
                    
                    # Write row to CSV
                    writer.writerow({
                        'rank': rank,
                        'node_name': node_name,
                        'node_type': node_type.value,
                        'gtype': gtype,
                        'median_run_time_s': avg_time,
                        'median_peak_mem_bytes': int(avg_mem),
                        'median_active_mem_bytes': int(self.median_active_mem_node.get(node_name, 0)),
                        'device': device
                    })
            print(f"Node statistics saved to {node_csv_filename}")
        except IOError as e:
            print(f"Error saving node statistics to {node_csv_filename}: {e}")

        # --- Save Activation Statistics to CSV ---
        activation_csv_filename = f"{filename_prefix}_activation_stats.csv"
        if not self.activation_liveness:
            print("No activation statistics to save.")
            return

        try:
            with open(activation_csv_filename, 'w', newline='') as csvfile:
                # Define CSV columns for activations
                fieldnames = [
                    'activation_name', 'node_type', 'creation_rank', 'last_fw_use_rank', 'first_bw_use_rank', 'last_bw_use_rank',
                    'median_mem_size_bytes', 'inactive_time_s', 'recomp_time_s', 'recomp_memory_bytes'
                ]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()

                # Sort activations by creation rank
                sorted_act_names = sorted(self.activation_liveness.keys(), 
                                         key=lambda name: self.activation_liveness[name]['creation_rank'])

                # Write each activation's data to CSV
                for act_name in sorted_act_names:
                    liveness = self.activation_liveness[act_name]
                    avg_mem = self.median_memory_sizes.get(act_name, 0.0)
                    inactive_t = self.inactive_times.get(act_name, 0.0)
                    recomp_t = self.recomp_times.get(act_name, 0.0)
                    recomp_mem = self.recomp_memory.get(act_name, 0)

                    # Write row to CSV
                    writer.writerow({
                        'activation_name': act_name,  # Always use the original FX node name
                        'node_type': self.node_types.get(act_name, NodeType.OTHER).value,  # Include node type
                        'creation_rank': liveness['creation_rank'],
                        'last_fw_use_rank': liveness['last_fw_use_rank'],
                        'first_bw_use_rank': liveness['first_bw_use_rank'],
                        'last_bw_use_rank': liveness['last_bw_use_rank'],
                        'median_mem_size_bytes': int(avg_mem),
                        'inactive_time_s': inactive_t,
                        'recomp_time_s': recomp_t,
                        'recomp_memory_bytes': recomp_mem
                    })
            print(f"Activation statistics saved to {activation_csv_filename}")
        except IOError as e:
            print(f"Error saving activation statistics to {activation_csv_filename}: {e}")

    def plot_stats(self, filename_prefix: str = "profiler_plots", top_n: int = 20) -> None:
        """Generates and saves plots for key profiling statistics."""
        if not self.median_run_times:
            print("Warning: No aggregated stats found to generate plots.")
            return

        name_to_node = {node.name: node for node in self.ranked_nodes}

        # --- Plot 1: Top N Nodes by Average Run Time ---
        try:
            # Sort nodes by average run time, descending
            sorted_nodes_by_time = sorted(self.median_run_times.items(), key=lambda item: item[1], reverse=True)
            top_nodes_time = sorted_nodes_by_time[:top_n]  # Take only top N nodes
            node_names_time = [item[0] for item in top_nodes_time]
            run_times = [item[1] for item in top_nodes_time]

            if node_names_time:  # Check if there's data to plot
                # Create figure with dynamic height based on number of bars
                plt.figure(figsize=(12, max(6, len(node_names_time) * 0.4)))
                plt.barh(node_names_time, run_times, color='skyblue')
                plt.xlabel("Average Run Time (seconds)")
                plt.ylabel("Node Name")
                plt.title(f"Top {len(node_names_time)} Nodes by Average Run Time")
                plt.gca().invert_yaxis()  # Display top node at the top
                plt.tight_layout()
                plot_filename_time = f"{filename_prefix}_node_runtime.png"
                plt.savefig(plot_filename_time)
                plt.close()  # Close the figure to free memory
                print(f"Node runtime plot saved to {plot_filename_time}")
            else:
                 print("No node runtime data to plot.")

        except Exception as e:
            print(f"Error generating node runtime plot: {e}")

        # --- Plot 2: Top N Nodes by Average Peak Memory ---
        try:
            # Sort nodes by average peak memory, descending
            sorted_nodes_by_mem = sorted(self.median_peak_mem_node.items(), key=lambda item: item[1], reverse=True)
            top_nodes_mem = sorted_nodes_by_mem[:top_n]  # Take only top N nodes
            node_names_mem = [item[0] for item in top_nodes_mem]
            peak_mems = [item[1] / (1024**2) for item in top_nodes_mem]  # Convert bytes to MiB

            if node_names_mem:  # Check if there's data to plot
                # Create figure with dynamic height based on number of bars
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
            sorted_acts_by_mem = sorted(self.median_memory_sizes.items(), key=lambda item: item[1], reverse=True)
            # Filter only activations (exclude other node types)
            act_mems = [(name, size) for name, size in sorted_acts_by_mem if self.node_types.get(name) == NodeType.ACT]
            top_acts_mem = act_mems[:top_n]  # Take only top N activations

            act_names_mem = [item[0] for item in top_acts_mem]
            act_sizes_mib = [item[1] / (1024**2) for item in top_acts_mem]  # Convert bytes to MiB

            if act_names_mem:  # Check if there's data to plot
                # Create figure with dynamic height based on number of bars
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
            top_acts_inactive = sorted_acts_by_inactive_time[:top_n]  # Take only top N activations

            act_names_inactive = [item[0] for item in top_acts_inactive]
            inactive_times = [item[1] for item in top_acts_inactive]

            if act_names_inactive:  # Check if there's data to plot
                # Create figure with dynamic height based on number of bars
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
            if self.ranked_nodes and self.median_peak_mem_node:
                # Get ranks for all nodes
                ranks = [self.node_ranks[node] for node in self.ranked_nodes]
                
                # Collect valid data points for the plot
                peak_mems_mib = []
                valid_ranks_for_plot = []
                for node in self.ranked_nodes:
                    if node.name in self.median_peak_mem_node:
                        # Convert bytes to MiB for better readability
                        peak_mems_mib.append(self.median_peak_mem_node[node.name] / (1024**2))
                        valid_ranks_for_plot.append(self.node_ranks[node])

                if not valid_ranks_for_plot:  # Check if any valid data points exist
                    print("No valid peak memory data to plot for Memory vs. Rank.")
                else:
                    # Create the plot
                    plt.figure(figsize=(15, 7))
                    plt.plot(valid_ranks_for_plot, peak_mems_mib, marker='o', linestyle='-', label="Peak Memory per Node")
                    
                    # Add vertical lines to mark forward/backward pass boundaries
                    if self.sep_fw_end_rank != -1:
                        plt.axvline(x=self.sep_fw_end_rank, color='red', linestyle='--', 
                                   label=f'FW/BW Sep (Rank {self.sep_fw_end_rank})')
                    if self.sep_bw_start_rank != -1 and self.sep_bw_start_rank != self.sep_fw_end_rank:
                         # Avoid double line if they are the same rank
                         plt.axvline(x=self.sep_bw_start_rank, color='orange', linestyle='--', 
                                    label=f'BW Start Sep (Rank {self.sep_bw_start_rank})')

                    # Add horizontal line for GPU memory limit if available
                    if hasattr(self, 'GPU_MEMORY_LIMIT_MIB'):
                        plt.axhline(y=self.GPU_MEMORY_LIMIT_MIB, color='green', linestyle=':', 
                                   label=f'GPU Limit ({self.GPU_MEMORY_LIMIT_MIB} MiB)')

                    # Add labels and formatting
                    plt.xlabel("Node Execution Rank (Topological Order)")
                    plt.ylabel("Peak Memory (MiB)")
                    plt.title("Peak Memory vs. Node Execution Rank")
                    plt.legend()
                    plt.grid(True)
                    plt.tight_layout()
                    
                    # Save the plot
                    plot_filename_mem_rank = f"{filename_prefix}_memory_vs_rank.png"
                    plt.savefig(plot_filename_mem_rank)
                    plt.close()
                    print(f"Memory vs. Rank plot saved to {plot_filename_mem_rank}")
            else:
                print("Not enough data to generate Memory vs. Rank plot (ranked_nodes or median_peak_mem_node is empty).")
        except Exception as e:
            print(f"Error generating Memory vs. Rank plot: {e}")

    def save_all_nodes_to_csv(self, filename_prefix: str = "profiler_stats") -> None:
        """
        Saves all node statistics to a single CSV file for comprehensive analysis.
        This includes all nodes regardless of type (parameters, activations, gradients, etc.)
        """
        if not self.median_run_times:
            print("Warning: No aggregated stats found to save to CSV.")
            return

        # Create lookup dictionary for node names
        name_to_node = {node.name: node for node in self.ranked_nodes}

        # --- Save All Node Statistics to CSV ---
        all_nodes_csv_filename = f"{filename_prefix}_allNode_stats.csv"
        try:
            with open(all_nodes_csv_filename, 'w', newline='') as csvfile:
                # Define CSV columns - include all relevant node information
                fieldnames = [
                    'rank', 'node_name', 'op_type', 'node_type', 'gtype', 
                    'median_run_time_s', 'median_peak_mem_bytes', 'median_active_mem_bytes', 
                    'creation_rank', 'last_fw_use_rank', 'first_bw_use_rank', 'last_bw_use_rank',
                    'memory_size_bytes', 'inactive_time_s', 'recomp_time_s', 'recomp_memory_bytes',
                    'device'
                ]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                
                # Sort nodes by execution rank
                sorted_node_names = sorted(
                    [node.name for node in self.ranked_nodes], 
                    key=lambda name: self.node_ranks.get(name_to_node.get(name), float('inf'))
                )

                # Write each node's data to CSV
                for node_name in sorted_node_names:
                    node = name_to_node.get(node_name)
                    if not node: continue  # Skip if node not found

                    # Gather basic node information
                    rank = self.node_ranks.get(node, -1)
                    node_type = self.node_types.get(node_name, NodeType.OTHER)
                    gtype = self.node_gtypes.get(node_name, "unknown")
                    
                    # Get node operation type
                    op_type = node.op if hasattr(node, 'op') else "unknown"
                    
                    # Get device information for the node
                    device = "unknown"
                    if node in self.env and isinstance(self.env[node], torch.Tensor):
                        device = str(self.env[node].device)
                    
                    # Get timing and memory information
                    avg_time = self.median_run_times.get(node_name, 0.0)
                    avg_peak_mem = self.median_peak_mem_node.get(node_name, 0.0)
                    avg_active_mem = self.median_active_mem_node.get(node_name, 0.0)
                    
                    # Get activation-specific information if this is an activation
                    liveness = self.activation_liveness.get(node_name, {})
                    creation_rank = liveness.get('creation_rank', -1) if liveness else -1
                    last_fw_use_rank = liveness.get('last_fw_use_rank', -1) if liveness else -1
                    first_bw_use_rank = liveness.get('first_bw_use_rank', -1) if liveness else -1
                    last_bw_use_rank = liveness.get('last_bw_use_rank', -1) if liveness else -1
                    
                    # Get memory size for this node (especially for activations)
                    memory_size = self.median_memory_sizes.get(node_name, 0.0)
                    
                    # Get recomputation metrics if available
                    inactive_time = self.inactive_times.get(node_name, 0.0)
                    recomp_time = self.recomp_times.get(node_name, 0.0)
                    recomp_memory = self.recomp_memory.get(node_name, 0)
                    
                    # Write row to CSV with all available information
                    writer.writerow({
                        'rank': rank,
                        'node_name': node_name,
                        'op_type': op_type,
                        'node_type': node_type.value,
                        'gtype': gtype,
                        'median_run_time_s': avg_time,
                        'median_peak_mem_bytes': int(avg_peak_mem),
                        'median_active_mem_bytes': int(avg_active_mem),
                        'creation_rank': creation_rank,
                        'last_fw_use_rank': last_fw_use_rank,
                        'first_bw_use_rank': first_bw_use_rank,
                        'last_bw_use_rank': last_bw_use_rank,
                        'memory_size_bytes': int(memory_size),
                        'inactive_time_s': inactive_time,
                        'recomp_time_s': recomp_time,
                        'recomp_memory_bytes': recomp_memory,
                        'device': device
                    })
            print(f"All node statistics saved to {all_nodes_csv_filename}")
        except IOError as e:
            print(f"Error saving all node statistics to {all_nodes_csv_filename}: {e}")
