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

def find_node_by_name(graph: fx.Graph, name: str, activation_liveness: Optional[Dict[str, Dict[str, int]]] = None) -> Optional[fx.Node]:
    """
    Find a node in the graph by name, with multiple fallback strategies.
    
    Args:
        graph: The FX graph
        name: The name of the node to find
        activation_liveness: Optional dictionary with activation liveness information
        
    Returns:
        The node if found, None otherwise
    """
    # Strategy 1: Exact match
    for node in graph.nodes:
        if node.name == name:
            return node
    
    # Strategy 2: Match by rank if activation_liveness is provided
    if activation_liveness and name in activation_liveness:
        creation_rank = activation_liveness[name].get('creation_rank')
        if creation_rank is not None:
            for node in graph.nodes:
                if hasattr(node, 'meta') and node.meta.get('rank') == creation_rank:
                    print(f"Found node {node.name} by rank {creation_rank} for activation {name}")
                    return node
    
    # Strategy 3: Partial matches
    for node in graph.nodes:
        # Check if the node name contains the target name or vice versa
        if name in node.name or node.name in name:
            print(f"Found partial match: node.name={node.name}, target={name}")
            return node
            
    # Strategy 4: Try without suffix (e.g., "relu_1" -> "relu")
    if '_' in name:
        base_name = name.split('_')[0]
        for node in graph.nodes:
            if node.name == base_name or base_name in node.name:
                print(f"Found base name match: node.name={node.name}, base_name={base_name}")
                return node
    
    # Strategy 5: Match by operation type
    if '.' in name:
        op_type = name.split('.')[0]
        for node in graph.nodes:
            if op_type in node.name:
                print(f"Found operation type match: node.name={node.name}, op_type={op_type}")
                return node
    
    print(f"Warning: Could not find node for activation {name} in the graph")
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
    if act_name not in activation_liveness:
        print(f"Warning: Activation {act_name} not found in activation_liveness")
        return [], set()
    
    # Get the node that produced this activation
    act_node = find_node_by_name(graph, act_name, activation_liveness)
    if not act_node:
        print(f"Warning: Node {act_name} not found in graph")
        return [], set()
    
    # Get the creation rank and last forward use rank
    if "creation_rank" not in activation_liveness[act_name] or "last_fw_use_rank" not in activation_liveness[act_name]:
        print(f"Warning: Missing rank information for activation {act_name}")
        return [], set()
        
    creation_rank = activation_liveness[act_name]["creation_rank"]
    last_fw_use_rank = activation_liveness[act_name]["last_fw_use_rank"]
    
    # Handle case where last_fw_use_rank is -1 (no forward use)
    if last_fw_use_rank == -1:
        print(f"Warning: Activation {act_name} has no forward use (last_fw_use_rank = -1)")
        last_fw_use_rank = creation_rank
    
    # Find all nodes between creation and last forward use
    subgraph_nodes = []
    for node in graph.nodes:
        # Get the node's rank (if available in meta)
        node_rank = getattr(node, "meta", {}).get("rank", -1)
        if isinstance(node_rank, int) and creation_rank <= node_rank <= last_fw_use_rank:
            subgraph_nodes.append(node)
    
    if not subgraph_nodes:
        print(f"Warning: No nodes found in subgraph for activation {act_name}")
        return [], set()
    
    # Sort nodes by rank to ensure correct execution order
    subgraph_nodes.sort(key=lambda n: getattr(n, "meta", {}).get("rank", 0))
    
    # Identify inputs to the subgraph
    subgraph_node_set = set(subgraph_nodes)
    inputs = set()
    
    # Find all nodes that are used by the subgraph but not in the subgraph
    for node in subgraph_nodes:
        for input_node in node.all_input_nodes:
            if input_node not in subgraph_node_set:
                # This is an input to the subgraph
                inputs.add(input_node)
    
    print(f"Extracted subgraph for {act_name} with {len(subgraph_nodes)} nodes and {len(inputs)} inputs")
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
    subgraph_node_set = set(subgraph_nodes)
    inputs = set()
    
    # Find nodes that are used by the subgraph but not in the subgraph
    for node in subgraph_nodes:
        for input_node in node.all_input_nodes:
            if input_node not in subgraph_node_set:
                # This is an input to the subgraph
                if input_node.name in kept_activations or input_node.op in ['placeholder', 'get_attr']:
                    inputs.add(input_node)
    
    return inputs

def extract_recomputation_subgraphs(graph: fx.Graph,
                                   ac_decisions: Dict[str, str],
                                   activation_liveness: Dict[str, Dict[str, int]]) -> Dict[str, Tuple[List[fx.Node], Set[fx.Node]]]:
    """
    Extract subgraphs for activations marked for recomputation.
    
    Args:
        graph: The original FX graph
        ac_decisions: Dict mapping activation names to 'CHECKPOINT' or 'RECOMPUTE'
        activation_liveness: Dict with activation liveness information
        
    Returns:
        Dict mapping activation names to (subgraph_nodes, subgraph_inputs)
    """
    recomp_subgraphs = {}
    
    # Get activations marked for recomputation and those kept
    recomp_activations = [act for act, decision in ac_decisions.items()
                         if decision == 'RECOMPUTE']
    kept_activations = set(act for act, decision in ac_decisions.items()
                          if decision == 'CHECKPOINT')
    
    print(f"Processing {len(recomp_activations)} activations marked for RECOMPUTE")
    print(f"Found {len(kept_activations)} activations marked for CHECKPOINT")
    
    # Process all recomputation activations
    for act_name in recomp_activations:
        # Extract the subgraph for this activation
        subgraph_nodes, subgraph_inputs = extract_subgraph_for_activation(graph, act_name, activation_liveness)
        
        if not subgraph_nodes:
            print(f"Warning: Could not extract subgraph for activation {act_name}")
            continue
        
        # Store the subgraph and its inputs
        recomp_subgraphs[act_name] = (subgraph_nodes, subgraph_inputs)
    
    # Print summary
    valid_subgraphs = sum(1 for nodes, _ in recomp_subgraphs.values() if nodes)
    print(f"Successfully extracted {valid_subgraphs} subgraphs out of {len(recomp_activations)} activations marked for recomputation")
    
    return recomp_subgraphs

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
    if not subgraph_nodes:
        print("Warning: No subgraph nodes to insert")
        return None
        
    # Create a local environment for this subgraph insertion
    local_env = {}
    
    # First, map all inputs to their corresponding nodes in the target graph
    for input_node in subgraph_inputs:
        if input_node in env:
            local_env[input_node] = env[input_node]
        else:
            print(f"Warning: Input node {input_node.name} not found in environment")
            # Try to find a node with the same name in the target graph
            for node in graph.nodes:
                if node.name == input_node.name:
                    local_env[input_node] = node
                    break
    
    # Now insert the subgraph
    try:
        with graph.inserting_before(insertion_point):
            # Process nodes in order
            for node in subgraph_nodes:
                # Skip nodes that are inputs (they already exist)
                if node in subgraph_inputs:
                    continue
                    
                # Copy the node, mapping its inputs through the environment
                try:
                    # Use the local environment first, then fall back to the global environment
                    copied_node = graph.node_copy(node, lambda n: local_env.get(n, env.get(n, n)))
                    local_env[node] = copied_node
                    env[node] = copied_node  # Update the global environment too
                except Exception as e:
                    print(f"Error copying node {node.name}: {e}")
                    continue
            
            # The last node in the subgraph corresponds to the recomputed activation
            if subgraph_nodes[-1] in local_env:
                return local_env[subgraph_nodes[-1]]
            else:
                print(f"Warning: Last node {subgraph_nodes[-1].name} not found in local environment")
                return None
    except Exception as e:
        print(f"Error inserting subgraph: {e}")
        return None

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
    print(f"Rewriting graph with {len(subgraphs)} recomputation subgraphs")
    
    # Create a copy of the graph to modify
    new_graph = copy.deepcopy(graph)
    
    # Environment mapping original nodes to their copies in the new graph
    env = {}
    for original_node, new_node in zip(graph.nodes, new_graph.nodes):
        env[original_node] = new_node
    
    # Track successful insertions
    successful_insertions = 0
    
    # For each activation marked for recomputation
    for act_name, (subgraph_nodes, subgraph_inputs) in subgraphs.items():
        if not subgraph_nodes:
            print(f"Skipping {act_name}: No subgraph nodes")
            continue
            
        # Get the first backward use rank for this activation
        if act_name not in activation_liveness or "first_bw_use_rank" not in activation_liveness[act_name]:
            print(f"Skipping {act_name}: Missing first_bw_use_rank")
            continue
            
        first_bw_use_rank = activation_liveness[act_name]["first_bw_use_rank"]
        
        # Find the node with this rank in the new graph
        first_bw_use_node = None
        for node in new_graph.nodes:
            node_rank = getattr(node, "meta", {}).get("rank", -1)
            if isinstance(node_rank, int) and node_rank == first_bw_use_rank:
                first_bw_use_node = node
                break
                
        if not first_bw_use_node:
            print(f"Skipping {act_name}: Could not find first backward use node with rank {first_bw_use_rank}")
            continue
            
        print(f"Inserting recomputation subgraph for {act_name} before node {first_bw_use_node.name} (rank {first_bw_use_rank})")
        
        try:
            # Insert the recomputation subgraph before the first backward use
            recomputed_node = insert_subgraph_before_node(
                new_graph, first_bw_use_node, subgraph_nodes, subgraph_inputs, env)
                
            if recomputed_node:
                # Find the original activation node in the new graph
                original_act_node = find_node_by_name(new_graph, act_name, activation_liveness)
                
                if original_act_node:
                    # Replace uses of the original activation with the recomputed one
                    original_act_node.replace_all_uses_with(recomputed_node)
                    print(f"Successfully replaced uses of {act_name} with recomputed node {recomputed_node.name}")
                    successful_insertions += 1
                else:
                    print(f"Warning: Could not find original activation node {act_name} in new graph")
            else:
                print(f"Warning: Failed to insert recomputation subgraph for {act_name}")
        except Exception as e:
            print(f"Error inserting recomputation subgraph for {act_name}: {e}")
    
    print(f"Successfully inserted {successful_insertions} recomputation subgraphs")
    
    # Validate the rewritten graph
    try:
        new_graph.lint()
        print("Graph validation passed")
    except Exception as e:
        print(f"Warning: Graph validation failed: {e}")
        
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
    # Create a new GraphModule with the rewritten graph
    new_gm = fx.GraphModule(model, rewritten_graph)
    
    # Copy over any attributes from the original module
    for name, attr in model.__dict__.items():
        if name != '_modules':
            setattr(new_gm, name, copy.deepcopy(attr))
    
    return new_gm

def trace_model_for_ac(model: torch.nn.Module, 
                      example_input: torch.Tensor) -> fx.GraphModule:
    """
    Trace a model to get an FX graph suitable for activation checkpointing.
    
    Args:
        model: The model to trace
        example_input: An example input tensor
        
    Returns:
        An FX GraphModule
    """
    # Use torch.fx.symbolic_trace to trace the model
    try:
        gm = fx.symbolic_trace(model)
        return gm
    except Exception as e:
        print(f"Error tracing model: {e}")
        # Fall back to a simpler approach if symbolic_trace fails
        return None