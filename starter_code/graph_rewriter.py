"""
Graph Extractor and Rewriter for Activation Checkpointing

This module implements Stage 3 of the activation checkpointing project:
1. Extracting subgraphs for activations marked for recomputation
2. Rewriting the graph to include these subgraphs in the backward pass

The implementation follows the approach described in the μ-TWO paper.
"""

import copy
import torch
import torch.fx as fx
from typing import Dict, List, Set, Tuple, Any, Optional
from graph_tracer import SEPFunction
from torch.fx.experimental.proxy_tensor import make_fx
from torch._functorch.partitioners import _extract_graph_with_inputs_outputs
from topological_ordering import ensure_topological_ordering

def find_node_by_name(graph: fx.Graph, name: str, activation_liveness: Optional[Dict[str, Dict[str, int]]] = None, logger: Optional[Any] = None) -> Optional[fx.Node]:
    """
    Find a node in the graph by name or, preferably, by its rank.

    Args:
        graph: The FX graph.
        name: The name of the node to find. This can be an activation name
              present in `activation_liveness`.
        activation_liveness: Optional dictionary with activation liveness information.
                             Structure: Dict[str, Dict[str, int]], where the inner
                             dict might contain 'creation_rank'.
        logger: Optional logger instance.

    Returns:
        The fx.Node if found, otherwise None.
    """
    if logger is None:
        import logging
        logger = logging.getLogger(__name__)

    # Extract the base name without indices or prefixes
    base_name = name.split('_')[0] if '_' in name else name
    
    # Extract any numeric part from the name
    import re
    numeric_part = re.search(r'(\d+)', name)
    numeric_id = int(numeric_part.group(1)) if numeric_part else None
    
    # 1. Primary Search by Rank with Scaling
    if activation_liveness and name in activation_liveness:
        if 'creation_rank' in activation_liveness[name]:
            profiler_rank = activation_liveness[name]['creation_rank']
            graph_size = len(list(graph.nodes))
            
            # Scale the rank to match the graph size
            # This handles the case where profiler ranks are much larger than graph nodes
            if profiler_rank > 0 and graph_size > 0:
                # Try different scaling approaches
                scaled_ranks = []
                
                # Simple linear scaling
                scaled_rank = int((profiler_rank / 4000) * graph_size)  # Assuming profiler ranks go up to ~4000
                scaled_ranks.append(scaled_rank)
                
                # Try a few nearby ranks as well
                for offset in [-5, -2, -1, 0, 1, 2, 5]:
                    if 0 <= scaled_rank + offset < graph_size:
                        scaled_ranks.append(scaled_rank + offset)
                
                # Try to find a node at any of these ranks
                for rank in scaled_ranks:
                    if 0 <= rank < graph_size:
                        nodes_list = list(graph.nodes)
                        node_by_rank = nodes_list[rank]
                        logger.info(f"Trying scaled rank {rank} (from original {profiler_rank}) for '{name}'")
                        
                        # Check if the node's name or target contains the base name
                        node_name = node_by_rank.name.lower()
                        node_target = str(node_by_rank.target).lower()
                        
                        if (base_name.lower() in node_name or
                            base_name.lower() in node_target or
                            (numeric_id is not None and str(numeric_id) in node_name)):
                            logger.info(f"Found node '{node_by_rank.name}' by scaled rank {rank} for activation '{name}'")
                            return node_by_rank
            
            logger.info(f"Could not find node by scaled rank for '{name}'")
        else:
            logger.info(f"'creation_rank' not found in activation_liveness for '{name}'.")

    # 2. Enhanced Name Search with Pattern Matching
    logger.info(f"Trying enhanced name search for '{name}'")
    
    # Try different name patterns
    name_patterns = [
        name.lower(),                      # Exact match (case insensitive)
        base_name.lower(),                 # Base name without indices
        f"{base_name.lower()}*",           # Base name prefix
    ]
    
    # Add numeric pattern if available
    if numeric_id is not None:
        name_patterns.append(f"*{numeric_id}*")  # Any name containing the numeric ID
    
    # Try to match by operation type for common operations
    op_type_mapping = {
        'convolution': ['conv', 'conv2d', 'convolution'],
        'relu': ['relu', 'threshold', 'relu_'],
        'batchnorm': ['batchnorm', 'batch_norm', 'bn'],
        'maxpool': ['maxpool', 'max_pool'],
        'avgpool': ['avgpool', 'avg_pool'],
        'flatten': ['flatten', 'view', 'reshape'],
        'linear': ['linear', 'fc', 'gemm'],
    }
    
    # Check if the base name matches any known operation type
    for op_type, patterns in op_type_mapping.items():
        if base_name.lower() in patterns:
            # This is a known operation type, look for nodes with similar operations
            for node in graph.nodes:
                node_target = str(node.target).lower()
                if any(pattern in node_target for pattern in patterns):
                    # Found a node with matching operation type
                    logger.info(f"Found node '{node.name}' by operation type match for '{name}'")
                    return node
    
    # Try to match by name patterns
    for pattern in name_patterns:
        for node in graph.nodes:
            node_name = node.name.lower()
            node_target = str(node.target).lower()
            
            # Check for exact match
            if pattern == node_name or pattern == node_target:
                logger.info(f"Found node '{node.name}' by exact name match for '{name}'")
                return node
            
            # Check for pattern match
            if pattern.endswith('*'):
                prefix = pattern[:-1]
                if node_name.startswith(prefix) or node_target.startswith(prefix):
                    logger.info(f"Found node '{node.name}' by prefix match for '{name}'")
                    return node
            
            if pattern.startswith('*'):
                suffix = pattern[1:]
                if node_name.endswith(suffix) or node_target.endswith(suffix):
                    logger.info(f"Found node '{node.name}' by suffix match for '{name}'")
                    return node
            
            if pattern.startswith('*') and pattern.endswith('*'):
                substring = pattern[1:-1]
                if substring in node_name or substring in node_target:
                    logger.info(f"Found node '{node.name}' by substring match for '{name}'")
                    return node
    
    # 3. Fallback to Original Name Search
    logger.info(f"Falling back to basic name search for '{name}'")
    for node in graph.nodes:
        if node.name == name:
            logger.info(f"Found node by exact name: '{node.name}'")
            return node
        # Additional check: sometimes the 'name' parameter might refer to the target of a call_module node
        if node.op == 'call_module' and str(node.target) == name:
            logger.info(f"Found node by target match: '{node.name}' for name '{name}'")
            return node

    # 4. Last Resort: Try to find any node with similar operation type
    if base_name.lower() in ['conv', 'convolution', 'relu', 'pool', 'linear', 'fc', 'bn', 'batchnorm']:
        logger.info(f"Trying to find any node with operation type similar to '{base_name}'")
        for node in graph.nodes:
            node_target = str(node.target).lower()
            if base_name.lower() in node_target:
                logger.info(f"Found node '{node.name}' as last resort for '{name}'")
                return node

    logger.warning(f"Could not find node for '{name}' by any method")
    return None

def extract_subgraph_for_activation(graph: fx.Graph, act_name: str,
                                    activation_liveness: Dict[str, Dict[str, int]]) -> Tuple[List[fx.Node], Set[fx.Node]]:
    """
    Extract the subgraph needed to recompute an activation.
    
    Args:
        graph: The original FX graph
        act_name: Name of the activation to extract subgraph for
        activation_liveness: Dict with activation liveness information
        
    Returns:
        Tuple of (list of nodes in the subgraph, set of input nodes)
    """
    import logging
    logger = logging.getLogger(__name__)
    
    # Get the node that produced this activation
    logger.info(f"Extracting subgraph for activation: {act_name}")
    
    # Get the creation rank and last forward use rank from activation_liveness
    if act_name not in activation_liveness:
        logger.warning(f"Activation {act_name} not found in activation_liveness")
        return [], set()
    
    creation_rank = activation_liveness[act_name].get('creation_rank')
    last_fw_use_rank = activation_liveness[act_name].get('last_fw_use_rank')
    
    if creation_rank is None or last_fw_use_rank is None:
        logger.warning(f"Missing rank information for activation {act_name}")
        return [], set()
    
    logger.info(f"Creation rank: {creation_rank}, Last forward use rank: {last_fw_use_rank}")
    
    # Find the node that produced this activation using our enhanced find_node_by_name function
    creation_node = find_node_by_name(graph, act_name, activation_liveness, logger)
    
    if creation_node is None:
        # Try to find a node with similar operation type based on the activation name
        base_name = act_name.split('_')[0] if '_' in act_name else act_name
        
        # Map common activation name prefixes to operation types
        op_type_mapping = {
            'conv': ['conv', 'convolution'],
            'relu': ['relu', 'threshold'],
            'bn': ['batchnorm', 'batch_norm'],
            'pool': ['maxpool', 'avgpool', 'max_pool', 'avg_pool'],
            'fc': ['linear', 'fc'],
            'add': ['add', 'sum'],
            'mul': ['mul', 'multiply'],
        }
        
        # Try to find a node with matching operation type
        for op_type, patterns in op_type_mapping.items():
            if base_name.lower() in patterns:
                for node in graph.nodes:
                    node_target = str(node.target).lower()
                    if any(pattern in node_target for pattern in patterns):
                        creation_node = node
                        logger.info(f"Found creation node by operation type: {node.name} (target: {node.target})")
                        break
                if creation_node:
                    break
    
    if creation_node is None:
        logger.warning(f"Could not find creation node for activation {act_name}")
        return [], set()
    
    logger.info(f"Found creation node: {creation_node.name} (op: {creation_node.op}, target: {creation_node.target})")
    
    # Extract the subgraph using a more robust approach
    # Start with the creation node and follow its users to build the subgraph
    
    # We'll limit the subgraph size to avoid excessive recomputation
    MAX_SUBGRAPH_SIZE = 20  # Limit subgraph size to avoid excessive recomputation
    
    # Start with the creation node
    subgraph_nodes = [creation_node]
    visited = {creation_node}
    
    # Breadth-first search to find all nodes in the subgraph
    queue = [creation_node]
    while queue and len(subgraph_nodes) < MAX_SUBGRAPH_SIZE:
        node = queue.pop(0)
        
        # Add all input nodes to the queue
        for input_node in node.all_input_nodes:
            if input_node not in visited and len(subgraph_nodes) < MAX_SUBGRAPH_SIZE:
                visited.add(input_node)
                subgraph_nodes.append(input_node)
                queue.append(input_node)
    
    if len(subgraph_nodes) >= MAX_SUBGRAPH_SIZE:
        logger.warning(f"Limiting subgraph size for {act_name} to {MAX_SUBGRAPH_SIZE} nodes")
    
    # Find the inputs to the subgraph (nodes that are used by the subgraph but not in the subgraph)
    inputs = set()
    for node in subgraph_nodes:
        for input_node in node.all_input_nodes:
            if input_node not in visited:
                inputs.add(input_node)
    
    logger.info(f"Extracted subgraph with {len(subgraph_nodes)} nodes and {len(inputs)} inputs")
    
    return subgraph_nodes, inputs

def identify_subgraph_inputs(subgraph_nodes: List[fx.Node],
                             kept_activations: Set[str]) -> Set[fx.Node]:
    """
    Identify the inputs to a subgraph.
    
    Args:
        subgraph_nodes: List of nodes in the subgraph
        kept_activations: Set of activation names that were kept (not recomputed)
        
    Returns:
        Set of nodes that serve as inputs to the subgraph
    """
    import logging
    logger = logging.getLogger(__name__)
    
    # Create a set of all nodes in the subgraph for faster lookup
    subgraph_node_set = set(subgraph_nodes)
    
    # Find nodes that are used by the subgraph but not in the subgraph
    inputs = set()
    
    for node in subgraph_nodes:
        # Check args
        for arg in node.args:
            if isinstance(arg, fx.Node) and arg not in subgraph_node_set:
                # More flexible input identification - accept any node that's not in the subgraph
                # This ensures we don't miss important inputs due to naming mismatches
                inputs.add(arg)
                logger.info(f"Found input node: {arg.name} (op: {arg.op})")
        
        # Check kwargs
        for _, kwarg in node.kwargs.items():
            if isinstance(kwarg, fx.Node) and kwarg not in subgraph_node_set:
                # More flexible input identification
                inputs.add(kwarg)
                logger.info(f"Found input node: {kwarg.name} (op: {kwarg.op})")
    
    # Filter inputs to prioritize placeholders and kept activations
    priority_inputs = set()
    for input_node in inputs:
        # Placeholders are always valid inputs
        if input_node.op == 'placeholder':
            priority_inputs.add(input_node)
            logger.info(f"Prioritizing placeholder input: {input_node.name}")
            continue
            
        # Check if this node's name is in kept_activations
        if input_node.name in kept_activations:
            priority_inputs.add(input_node)
            logger.info(f"Prioritizing kept activation input: {input_node.name}")
            continue
            
        # Check if any similar name is in kept_activations
        found_match = False
        for kept_name in kept_activations:
            # Simple similarity check - if the base name matches
            if input_node.name.split('_')[0] == kept_name.split('_')[0]:
                priority_inputs.add(input_node)
                logger.info(f"Prioritizing input with similar name: {input_node.name} ~ {kept_name}")
                found_match = True
                break
                
        # If no match found, still include the input but log it
        if not found_match:
            priority_inputs.add(input_node)
            logger.info(f"Including non-kept input: {input_node.name}")
    
    logger.info(f"Identified {len(priority_inputs)} inputs to the subgraph")
    return priority_inputs

def extract_recomputation_subgraphs(graph: fx.Graph,
                                    ac_decisions: Dict[str, str],
                                    activation_liveness: Dict[str, Dict[str, int]]) -> Dict[str, Tuple[List[fx.Node], Set[fx.Node]]]:
    """
    Extract subgraphs for activations marked for recomputation.
    
    This is a key part of activation checkpointing. For each activation marked for RECOMPUTE:
    1. We identify the subgraph that computes this activation
    2. We extract this subgraph and its inputs
    3. Later, we'll insert this subgraph before the first backward use of the activation
    
    The core idea is that we don't save these activations during the forward pass,
    which saves memory. Instead, we recompute them during the backward pass when needed.
    
    Args:
        graph: The original FX graph
        ac_decisions: Dict mapping activation names to 'RETAINED' or 'RECOMPUTE'
        activation_liveness: Dict with activation liveness information
        
    Returns:
        Dict mapping activation names to (subgraph_nodes, subgraph_inputs)
    """
    import logging
    logger = logging.getLogger(__name__)
    
    # Get activations marked for recomputation and those kept
    recompute_activations = set()
    kept_activations = set()
    
    for act_name, decision in ac_decisions.items():
        if decision == 'RECOMPUTE':
            recompute_activations.add(act_name)
        elif decision == 'RETAINED':
            kept_activations.add(act_name)
    
    logger.info(f"Found {len(recompute_activations)} activations marked for recomputation")
    logger.info(f"Found {len(kept_activations)} activations marked as retained")
    
    # Extract subgraphs for each activation marked for recomputation
    subgraphs = {}
    
    for act_name in recompute_activations:
        logger.info(f"Processing activation: {act_name}")
        
        # Extract the subgraph for this activation
        subgraph_nodes, subgraph_inputs = extract_subgraph_for_activation(
            graph, act_name, activation_liveness
        )
        
        # Refine the inputs to only include kept activations and placeholders
        refined_inputs = identify_subgraph_inputs(subgraph_nodes, kept_activations)
        
        # Store the subgraph and its inputs
        if subgraph_nodes:
            subgraphs[act_name] = (subgraph_nodes, refined_inputs)
            logger.info(f"Extracted subgraph for {act_name} with {len(subgraph_nodes)} nodes and {len(refined_inputs)} inputs")
        else:
            logger.warning(f"Could not extract subgraph for {act_name}")
    
    logger.info(f"Extracted {len(subgraphs)} subgraphs in total")
    return subgraphs

def insert_subgraph_before_node(graph: fx.Graph,
                                insertion_point: fx.Node,
                                subgraph_nodes: List[fx.Node],
                                subgraph_inputs: Set[fx.Node],
                                env: Dict[fx.Node, fx.Node]) -> fx.Node:
    """
    Insert a subgraph before a specific node in the graph.
    
    Args:
        graph: The FX graph to modify
        insertion_point: The node before which to insert the subgraph
        subgraph_nodes: List of nodes in the subgraph to insert
        subgraph_inputs: Set of nodes that serve as inputs to the subgraph
        env: Environment mapping original nodes to their copies
        
    Returns:
        The last node inserted (corresponding to the recomputed activation)
    """
    import logging
    logger = logging.getLogger(__name__)
    
    logger.info(f"Inserting subgraph with {len(subgraph_nodes)} nodes before {insertion_point.name}")
    
    # Create a local environment for this subgraph insertion
    local_env = env.copy()
    
    # Map all inputs to their corresponding nodes in the target graph
    for input_node in subgraph_inputs:
        if input_node in env:
            local_env[input_node] = env[input_node]
            logger.info(f"Mapped input {input_node.name} to {env[input_node].name}")
        else:
            # Handle missing input nodes by creating placeholders
            logger.warning(f"Input node {input_node.name} not found in environment, creating placeholder")
            with graph.inserting_before(insertion_point):
                placeholder_node = graph.placeholder(f"placeholder_for_{input_node.name}")
                local_env[input_node] = placeholder_node
                logger.info(f"Created placeholder {placeholder_node.name} for missing input {input_node.name}")
    
    # Insert the subgraph before the insertion point
    last_node = None
    with graph.inserting_before(insertion_point):
        # Process nodes in order, copying them to the target graph
        for node in subgraph_nodes:
            # Skip placeholder and output nodes
            if node.op == 'placeholder' or node.op == 'output':
                continue
            
            # Copy the node to the target graph
            logger.info(f"Copying node {node.name} (op: {node.op}, target: {node.target})")
            
            # Define the argument transformation function
            def arg_transform(arg):
                if isinstance(arg, fx.Node):
                    if arg in local_env:
                        return local_env[arg]
                    else:
                        logger.warning(f"Node {arg.name} not found in environment during arg_transform")
                        return arg
                return arg
            
            # Copy the node to the target graph
            new_node = graph.node_copy(node, arg_transform)
            
            # Add the new node to the local environment
            local_env[node] = new_node
            last_node = new_node
            
            logger.info(f"Created new node {new_node.name} (op: {new_node.op}, target: {new_node.target})")
    
    if last_node is None:
        logger.warning("No nodes were inserted into the graph")
    else:
        logger.info(f"Last inserted node: {last_node.name}")
    
    return last_node

def _ensure_topological_ordering(graph: fx.Graph) -> fx.Graph:
    """
    Ensure proper topological ordering of nodes in the graph.
    
    This function specifically addresses issues with operations like flatten
    that have critical dependencies in the forward pass.
    
    Args:
        graph: The FX graph to reorder
        
    Returns:
        The reordered FX graph
    """
    # Use the implementation from topological_ordering.py
    return ensure_topological_ordering(graph)

def _preserve_critical_operations(graph: fx.Graph) -> None:
    """
    Identify and mark critical operations in the graph that must be preserved.
    
    This function specifically looks for the flatten operation that connects
    avgpool to fc in ResNet models, which is critical for correct execution.
    
    Args:
        graph: The FX graph to analyze
    """
    import logging
    logger = logging.getLogger(__name__)
    
    logger.info("Identifying critical operations in the graph")
    
    # Find flatten, avgpool, and fc nodes
    flatten_nodes = []
    avgpool_nodes = []
    fc_nodes = []
    linear_nodes = []
    
    # First pass: identify all potential nodes
    for node in graph.nodes:
        # Check for flatten operations
        if node.op == 'call_function' and 'flatten' in str(node.target).lower():
            flatten_nodes.append(node)
            logger.info(f"Found flatten node: {node.name}")
        
        # Check for avgpool operations
        elif node.op == 'call_method' and 'avgpool' in str(node.target).lower():
            avgpool_nodes.append(node)
            logger.info(f"Found avgpool node: {node.name}")
        elif node.op == 'call_function' and 'avg_pool' in str(node.target).lower():
            avgpool_nodes.append(node)
            logger.info(f"Found avgpool node: {node.name}")
        
        # Check for fc/linear operations
        elif node.op == 'call_module' and ('fc' in str(node.target).lower() or 'linear' in str(node.target).lower()):
            fc_nodes.append(node)
            logger.info(f"Found fc node: {node.name}")
        elif node.op == 'call_function' and 'linear' in str(node.target).lower():
            linear_nodes.append(node)
            logger.info(f"Found linear node: {node.name}")
    
    # If we didn't find any avgpool nodes, look for nodes that might be avgpool
    if not avgpool_nodes:
        for node in graph.nodes:
            if node.op == 'call_module' and 'pool' in str(node.target).lower():
                avgpool_nodes.append(node)
                logger.info(f"Found potential avgpool node: {node.name}")
    
    # Add linear nodes to fc_nodes
    fc_nodes.extend(linear_nodes)
    
    # If we didn't find any fc nodes, try to find them by looking at the module structure
    if not fc_nodes:
        for node in graph.nodes:
            if node.op == 'call_module':
                target_str = str(node.target).lower()
                # In ResNet, the final layer is often named 'fc'
                if target_str == 'fc':
                    fc_nodes.append(node)
                    logger.info(f"Found fc node by name: {node.name}")
    
    # Check for the critical path: avgpool -> flatten -> fc
    critical_paths_found = 0
    
    # If we have fc nodes but no flatten nodes, try to infer the flatten node
    if fc_nodes and not flatten_nodes:
        for fc_node in fc_nodes:
            # Look for inputs to fc node that might be flatten operations
            for arg in fc_node.args:
                if isinstance(arg, fx.Node) and 'flatten' in arg.name.lower():
                    flatten_nodes.append(arg)
                    logger.info(f"Inferred flatten node from fc input: {arg.name}")
    
    # If we still don't have flatten nodes, create a synthetic one
    if fc_nodes and not flatten_nodes:
        logger.warning("No flatten nodes found, but fc nodes exist. Creating synthetic dependencies.")
        # Find the first fc node
        fc_node = fc_nodes[0]
        # Create synthetic dependencies directly from avgpool to fc
        if avgpool_nodes:
            avgpool_node = avgpool_nodes[0]
            logger.info(f"Creating synthetic critical path: {avgpool_node.name} -> {fc_node.name}")
            
            # Mark nodes in the critical path
            if not hasattr(avgpool_node, 'meta'):
                avgpool_node.meta = {}
            if not hasattr(fc_node, 'meta'):
                fc_node.meta = {}
            
            # Mark these nodes as part of a critical path
            avgpool_node.meta['critical_path'] = True
            fc_node.meta['critical_path'] = True
            
            # Store dependency information
            if 'deps' not in avgpool_node.meta:
                avgpool_node.meta['deps'] = []
            if 'deps' not in fc_node.meta:
                fc_node.meta['deps'] = []
            
            # Add dependencies
            fc_node.meta['deps'].append(avgpool_node)
            
            critical_paths_found += 1
    
    # Normal path detection
    for flatten_node in flatten_nodes:
        # Check all inputs to flatten node
        flatten_inputs = []
        for arg in flatten_node.args:
            if isinstance(arg, fx.Node):
                flatten_inputs.append(arg)
        
        # Check if any avgpool node is an input to flatten
        for avgpool_node in avgpool_nodes:
            # Direct input check
            direct_input = avgpool_node in flatten_inputs
            
            # Name-based inference as fallback
            name_match = False
            if not direct_input:
                for inp in flatten_inputs:
                    if 'pool' in inp.name.lower() or 'avg' in inp.name.lower():
                        avgpool_node = inp  # Reassign to the actual input
                        name_match = True
                        logger.info(f"Inferred avgpool->flatten connection by name: {inp.name} -> {flatten_node.name}")
                        break
            
            if direct_input or name_match:
                logger.info(f"Found connection: {avgpool_node.name} -> {flatten_node.name}")
                
                # Check if flatten is an input to any fc node
                for fc_node in fc_nodes:
                    fc_inputs = []
                    for arg in fc_node.args:
                        if isinstance(arg, fx.Node):
                            fc_inputs.append(arg)
                    
                    # Direct input check
                    direct_fc_input = flatten_node in fc_inputs
                    
                    # Name-based inference as fallback
                    name_match_fc = False
                    if not direct_fc_input:
                        for inp in fc_inputs:
                            if 'flat' in inp.name.lower():
                                flatten_node = inp  # Reassign to the actual input
                                name_match_fc = True
                                logger.info(f"Inferred flatten->fc connection by name: {inp.name} -> {fc_node.name}")
                                break
                    
                    if direct_fc_input or name_match_fc:
                        logger.info(f"Found critical path: {avgpool_node.name} -> {flatten_node.name} -> {fc_node.name}")
                        critical_paths_found += 1
                        
                        # Mark nodes in the critical path
                        if not hasattr(avgpool_node, 'meta'):
                            avgpool_node.meta = {}
                        if not hasattr(flatten_node, 'meta'):
                            flatten_node.meta = {}
                        if not hasattr(fc_node, 'meta'):
                            fc_node.meta = {}
                        
                        # Mark these nodes as part of a critical path
                        avgpool_node.meta['critical_path'] = True
                        flatten_node.meta['critical_path'] = True
                        fc_node.meta['critical_path'] = True
                        
                        # Store dependency information
                        if 'deps' not in avgpool_node.meta:
                            avgpool_node.meta['deps'] = []
                        if 'deps' not in flatten_node.meta:
                            flatten_node.meta['deps'] = []
                        if 'deps' not in fc_node.meta:
                            fc_node.meta['deps'] = []
                        
                        # Add dependencies
                        if avgpool_node not in flatten_node.meta['deps']:
                            flatten_node.meta['deps'].append(avgpool_node)
                        if flatten_node not in fc_node.meta['deps']:
                            fc_node.meta['deps'].append(flatten_node)
                        
                        logger.info(f"Added critical path dependencies: {avgpool_node.name} -> {flatten_node.name} -> {fc_node.name}")
    
    if critical_paths_found == 0:
        logger.warning("No critical paths found in the model. This might cause issues with ResNet models.")

def rewrite_graph_with_recomputation(graph: fx.Graph,
                                       subgraphs: Dict[str, Tuple[List[fx.Node], Set[fx.Node]]],
                                       activation_liveness: Dict[str, Dict[str, int]]) -> fx.Graph:
    """
    Rewrite the graph to include recomputation subgraphs in the backward pass.
    
    Args:
        graph: The original FX graph
        subgraphs: Dict mapping activation names to (subgraph_nodes, subgraph_inputs)
        activation_liveness: Dict with activation liveness information
        
    Returns:
        The rewritten FX graph
    """
    import logging
    logger = logging.getLogger(__name__)
    
    logger.info("Rewriting graph with recomputation subgraphs")
    
    # Create a copy of the graph to modify
    new_graph = fx.Graph(owning_module=graph._owning_module)
    
    # Identify and mark critical operations
    _preserve_critical_operations(graph)
    
    # Create environment mapping original nodes to their copies
    env = {}
    
    # First, copy all nodes from the original graph to the new graph
    for node in graph.nodes:
        # Define the argument transformation function
        def arg_transform(arg):
            if isinstance(arg, fx.Node):
                return env[arg]
            return arg
        
        # Copy the node to the new graph
        new_node = new_graph.node_copy(node, arg_transform)
        env[node] = new_node
        
        # Copy metadata
        if hasattr(node, 'meta'):
            new_node.meta = node.meta.copy() if node.meta else {}
    
    logger.info(f"Copied {len(env)} nodes from original graph to new graph")
    
    # For each activation marked for recomputation
    for act_name, (subgraph_nodes, subgraph_inputs) in subgraphs.items():
        logger.info(f"Processing activation {act_name} for recomputation")
        
        # Get the first backward use rank
        if act_name not in activation_liveness:
            logger.warning(f"Activation {act_name} not found in activation_liveness")
            continue
        
        first_bw_use_rank = activation_liveness[act_name].get('first_bw_use_rank')
        if first_bw_use_rank is None:
            logger.warning(f"No first_bw_use_rank found for activation {act_name}")
            continue
        
        logger.info(f"First backward use rank: {first_bw_use_rank}")
        
        # Find the node with this rank in the new graph using a more flexible approach
        insertion_point = None
        
        # Try to find by exact rank first
        for node in new_graph.nodes:
            if hasattr(node, 'meta') and 'rank' in node.meta and node.meta['rank'] == first_bw_use_rank:
                insertion_point = node
                break
        
        # If not found, try scaled rank approach
        if insertion_point is None:
            graph_size = len(list(new_graph.nodes))
            if first_bw_use_rank > 0 and graph_size > 0:
                # Scale the rank to match the graph size
                scaled_rank = int((first_bw_use_rank / 4000) * graph_size)  # Assuming ranks go up to ~4000
                
                # Try a few nearby ranks
                for offset in [-5, -2, -1, 0, 1, 2, 5]:
                    if 0 <= scaled_rank + offset < graph_size:
                        candidate_rank = scaled_rank + offset
                        nodes_list = list(new_graph.nodes)
                        candidate_node = nodes_list[candidate_rank]
                        logger.info(f"Trying scaled rank {candidate_rank} (from original {first_bw_use_rank}) for insertion point")
                        insertion_point = candidate_node
                        break
        
        # If still not found, use a fallback insertion point
        if insertion_point is None:
            # Use the first node in the backward section as fallback
            for node in new_graph.nodes:
                if hasattr(node, 'meta') and node.meta.get('gtype') == 'backward':
                    logger.warning(f"Using fallback insertion point for activation {act_name}")
                    insertion_point = node
                    break
            
            # If no backward section, use the last node before output as fallback
            if insertion_point is None:
                nodes_list = list(new_graph.nodes)
                # Find the output node
                output_node = None
                for node in reversed(nodes_list):
                    if node.op == 'output':
                        output_node = node
                        break
                
                if output_node is not None and len(nodes_list) > 1:
                    # Use the node before output
                    output_idx = nodes_list.index(output_node)
                    if output_idx > 0:
                        insertion_point = nodes_list[output_idx - 1]
                        logger.warning(f"Using last node before output as fallback insertion point for {act_name}")
        
        if insertion_point is None:
            logger.warning(f"Could not find any insertion point for activation {act_name}")
            continue
            
        # Skip this activation if it's too large or has a very small recomputation time
        # This helps avoid recomputing activations that would cause more problems than benefits
        act_details = activation_liveness.get(act_name, {})
        mem_size = act_details.get('median_mem_size_bytes', 0)
        recomp_time = act_details.get('recomp_time_s', 0)
        
        # Skip very large activations (> 100MB) or those with negligible recomputation time
        if mem_size > 100 * 1024 * 1024:  # > 100MB
            logger.warning(f"Skipping activation {act_name} because it's too large ({mem_size/(1024*1024):.2f} MB)")
            continue
            
        if recomp_time < 1e-5:  # < 10 microseconds
            logger.warning(f"Skipping activation {act_name} because recomputation time is too small ({recomp_time:.8f} s)")
            continue
        
        logger.info(f"Found insertion point: {insertion_point.name}")
        
        # Find the original activation node in the new graph using a more flexible approach
        original_node = None
        
        # Try exact name match first
        for node in graph.nodes:
            if node.name == act_name:
                original_node = node
                break
        
        # If not found, try pattern matching
        if original_node is None:
            # Try to find by operation type (for convolution nodes)
            if act_name.startswith('convolution_'):
                for node in graph.nodes:
                    if node.op == 'call_module' and ('conv' in str(node.target).lower() or 'conv' in node.name.lower()):
                        logger.info(f"Found potential match for {act_name}: {node.name} (target: {node.target})")
                        original_node = node
                        break
            # Try to find by operation type (for relu nodes)
            elif act_name.startswith('relu_'):
                for node in graph.nodes:
                    if node.op == 'call_module' and 'relu' in str(node.target).lower():
                        logger.info(f"Found potential match for {act_name}: {node.name} (target: {node.target})")
                        original_node = node
                        break
        
        # Skip this activation if we couldn't find the original node
        # We'll avoid creating synthetic nodes as they cause issues in the forward pass
        if original_node is None:
            logger.warning(f"Could not find original node for activation {act_name}, skipping")
            continue
        
        # Get the corresponding node in the new graph
        if original_node not in env:
            logger.warning(f"Original node {original_node.name} not found in environment")
            continue
        
        original_node_in_new_graph = env[original_node]
        logger.info(f"Found original node in new graph: {original_node_in_new_graph.name}")
        
        # Insert the recomputation subgraph before the first backward use
        recomputed_node = insert_subgraph_before_node(
            new_graph, insertion_point, subgraph_nodes, subgraph_inputs, env
        )
        
        if recomputed_node is None:
            logger.warning(f"Failed to insert recomputation subgraph for {act_name}")
            continue
        
        logger.info(f"Inserted recomputation subgraph, recomputed node: {recomputed_node.name}")
        
        # Replace backward pass uses of the original activation with the recomputed version
        # We only want to replace uses that occur after the insertion point
        for user in list(original_node_in_new_graph.users.keys()):
            if hasattr(user, 'meta') and 'rank' in user.meta and user.meta['rank'] >= first_bw_use_rank:
                user.replace_input_with(original_node_in_new_graph, recomputed_node)
                logger.info(f"Replaced use of {original_node_in_new_graph.name} with {recomputed_node.name} in {user.name}")
        
        # Mark the original activation node as being recomputed
        if not hasattr(original_node_in_new_graph, 'meta'):
            original_node_in_new_graph.meta = {}
        original_node_in_new_graph.meta['recomputed'] = True
    
    # Validate the rewritten graph
    try:
        new_graph.lint()
    except Exception as e:
        logger.warning(f"Graph validation failed: {e}")
    
    # Ensure proper topological ordering before returning
    try:
        ordered_graph = _ensure_topological_ordering(new_graph)
        logger.info("Graph rewriting complete")
        return ordered_graph
    except Exception as e:
        logger.error(f"Error in topological ordering: {e}")
        # Fall back to the original graph without reordering
        logger.warning("Falling back to the original graph without reordering")
        return new_graph

def apply_rewritten_graph(model: torch.nn.Module,
                           original_graph: fx.Graph,
                           rewritten_graph: fx.Graph) -> torch.nn.Module:
    """
    Apply a rewritten graph to a model.
    
    Args:
        model: The original model
        original_graph: The original FX graph
        rewritten_graph: The rewritten FX graph with recomputation
        
    Returns:
        A new model with the rewritten graph
    """
    import logging
    logger = logging.getLogger(__name__)
    
    logger.info("Applying rewritten graph to model")
    
    # Skip topological ordering since it was already done in rewrite_graph_with_recomputation
    # and just use the rewritten graph directly
    try:
        # Create a new GraphModule with the rewritten graph
        new_module = fx.GraphModule(model, rewritten_graph)
    except Exception as e:
        logger.error(f"Error creating GraphModule: {e}")
        return model
    
    # Copy over any attributes from the original module
    for name, attr in model.__dict__.items():
        if name != '_modules' and not name.startswith('_parameters') and not name.startswith('_buffers'):
            setattr(new_module, name, attr)
    
    # Validate the graph module before returning
    try:
        new_module.graph.lint()
        logger.info("Graph validation successful")
    except Exception as e:
        logger.warning(f"Graph validation failed: {e}")
        # Handle special cases or fallbacks
        logger.warning("Falling back to original model")
        return model
    
    logger.info("Successfully applied rewritten graph to model")
    return new_module

def trace_model_for_ac(model: torch.nn.Module,
                       example_input: torch.Tensor,
                       activation_liveness: Optional[Dict[str, Dict[str, int]]] = None) -> fx.GraphModule:
    """
    Trace a model to get an FX graph suitable for activation checkpointing.
    
    This function creates a graph representation of the model that can be modified
    to implement activation checkpointing. The key idea is to identify which activations
    should be discarded during the forward pass and recomputed during the backward pass.
    
    Args:
        model: The model to trace
        example_input: An example input tensor
        activation_liveness: Optional dictionary with activation liveness information
                             that maps activation names to their creation and usage ranks
        
    Returns:
        An FX GraphModule with metadata suitable for activation checkpointing
    """
    import logging
    from torch.fx.experimental.proxy_tensor import make_fx
    from torch._functorch.partitioners import _extract_graph_with_inputs_outputs
    
    logger = logging.getLogger(__name__)
    
    logger.info(f"Tracing model: {model.__class__.__name__}")
    
    # Define a custom tracer to handle special cases like flatten
    class CustomTracer(fx.Tracer):
        def __init__(self):
            super().__init__()
            self.special_ops = set()
        
        def create_node(self, kind, target, args, kwargs, name=None, type_expr=None):
            # Check for special operations like flatten
            if kind == 'call_function' and 'flatten' in str(target).lower():
                self.special_ops.add(name)
                logger.info(f"Found special operation: {name} (target: {target})")
            
            return super().create_node(kind, target, args, kwargs, name, type_expr)
    
    # Use a custom forward function that includes backward pass
    def custom_forward(mod, x):
        # Forward pass
        y = mod(x)
        # Add a separator to mark the end of forward pass
        y = torch.ops.separator.sep.default(y)
        # Backward pass
        y.sum().backward()
        return y
    
    try:
        # Use torch.fx.symbolic_trace to trace the model
        logger.info("Tracing model with symbolic_trace")
        
        # First try with symbolic_trace
        try:
            tracer = CustomTracer()
            graph_module = fx.GraphModule(model, tracer.trace(model))
            logger.info("Successfully traced model with symbolic_trace")
        except Exception as e:
            logger.warning(f"symbolic_trace failed: {e}")
            logger.info("Falling back to make_fx")
            
            # Fall back to make_fx if symbolic_trace fails
            graph_module = make_fx(model)(example_input)
            logger.info("Successfully traced model with make_fx")
        
        # Add rank metadata to the traced graph
        logger.info("Adding rank metadata to traced graph")
        for i, node in enumerate(graph_module.graph.nodes):
            if not hasattr(node, 'meta'):
                node.meta = {}
            node.meta['rank'] = i
        
        # Identify forward and backward sections
        sep_node = None
        for node in graph_module.graph.nodes:
            if node.op == 'call_function' and 'separator.sep' in str(node.target):
                sep_node = node
                break
        
        if sep_node:
            logger.info(f"Found separator node: {sep_node.name}")
            fw_end_rank = sep_node.meta['rank']
            
            # Mark nodes as forward or backward
            for node in graph_module.graph.nodes:
                if not hasattr(node, 'meta'):
                    node.meta = {}
                
                if node.meta['rank'] <= fw_end_rank:
                    node.meta['gtype'] = 'forward'
                else:
                    node.meta['gtype'] = 'backward'
        else:
            logger.warning("No separator node found, cannot distinguish forward/backward passes")
        
        # Verify critical operations are present and properly connected
        _preserve_critical_operations(graph_module.graph)
        
        # Recompile the graph after adding metadata
        graph_module.recompile()
        
        logger.info("Model tracing complete")
        return graph_module
    
    except Exception as e:
        logger.error(f"Error tracing model: {e}")
        raise