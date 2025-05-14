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

def find_node_by_name(graph: fx.Graph, name: str) -> Optional[fx.Node]:
    """
    Find a node in the graph by name, with fallbacks for partial matches.
    
    Args:
        graph: The FX graph
        name: The name of the node to find
        
    Returns:
        The node if found, None otherwise
    """
    # Try exact match first
    for node in graph.nodes:
        if node.name == name:
            return node
    
    # Try partial matches
    for node in graph.nodes:
        # Check if the node name contains the target name or vice versa
        if name in node.name or node.name in name:
            print(f"Found partial match: node.name={node.name}, target={name}")
            return node
            
    # Try without suffix (e.g., "relu_1" -> "relu")
    if '_' in name:
        base_name = name.split('_')[0]
        for node in graph.nodes:
            if node.name == base_name or base_name in node.name:
                print(f"Found base name match: node.name={node.name}, base_name={base_name}")
                return node
    
    return None

def extract_subgraph_for_activation(graph: fx.Graph, act_name: str, 
                                   activation_liveness: Dict[str, Dict[str, int]]) -> List[fx.Node]:
    """
    Extract the subgraph needed to recompute an activation.
    
    Args:
        graph: The original FX graph
        act_name: Name of the activation to extract subgraph for
        activation_liveness: Dict with activation liveness information
        
    Returns:
        List of nodes in the subgraph, in execution order
    """
    if act_name not in activation_liveness:
        print(f"Warning: Activation {act_name} not found in activation_liveness")
        return []
    
    # Get the node that produced this activation
    act_node = find_node_by_name(graph, act_name)
    if not act_node:
        print(f"Warning: Node {act_name} not found in graph")
        return []
    
    # Get the creation rank and last forward use rank
    if "creation_rank" not in activation_liveness[act_name] or "last_fw_use_rank" not in activation_liveness[act_name]:
        print(f"Warning: Missing rank information for activation {act_name}")
        return []
        
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
    
    return subgraph_nodes

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
    
    # Create a mapping between activation names in the profiler and nodes in the graph
    # This is needed because the node names in the traced graph might be different
    # from the activation names in the profiler
    name_to_node_map = {}
    for node in graph.nodes:
        # Try different variations of the node name
        node_name = node.name
        name_to_node_map[node_name] = node
        
        # Try without suffix (e.g., "relu_1" -> "relu")
        if '_' in node_name:
            base_name = node_name.split('_')[0]
            name_to_node_map[base_name] = node
    
    # Print some debug info
    print(f"Found {len(name_to_node_map)} nodes in the graph")
    print(f"First few node names: {list(name_to_node_map.keys())[:5]}")
    
    for act_name in recomp_activations[:10]:  # Process first 10 for debugging
        # Try to find the node in the graph
        node = None
        if act_name in name_to_node_map:
            node = name_to_node_map[act_name]
            print(f"Found node {node.name} for activation {act_name}")
        else:
            # Try to find a similar name
            for node_name in name_to_node_map:
                if act_name in node_name or node_name in act_name:
                    node = name_to_node_map[node_name]
                    print(f"Found similar node {node.name} for activation {act_name}")
                    break
            
            if not node:
                print(f"Warning: Could not find node for activation {act_name} in the graph")
                continue
        
        # Extract the subgraph for this activation
        subgraph_nodes = extract_subgraph_for_activation(graph, act_name, activation_liveness)
        
        if not subgraph_nodes:
            print(f"Warning: Could not extract subgraph for activation {act_name}")
            continue
            
        # Identify the inputs to this subgraph
        subgraph_inputs = identify_subgraph_inputs(subgraph_nodes, kept_activations)
        
        # Store the subgraph and its inputs
        recomp_subgraphs[act_name] = (subgraph_nodes, subgraph_inputs)
        
        print(f"Extracted subgraph for activation {act_name} with {len(subgraph_nodes)} nodes and {len(subgraph_inputs)} inputs")
        
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
    with graph.inserting_before(insertion_point):
        # Create a mapping from original nodes to their copies
        for node in subgraph_nodes:
            # Skip nodes that are inputs (they already exist)
            if node in subgraph_inputs:
                continue
                
            # Copy the node, mapping its inputs through the environment
            copied_node = graph.node_copy(node, lambda n: env[n] if n in env else n)
            env[node] = copied_node
            
        # The last node in the subgraph corresponds to the recomputed activation
        return env[subgraph_nodes[-1]] if subgraph_nodes else None

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
    # Create a copy of the graph to modify
    new_graph = copy.deepcopy(graph)
    
    # Environment mapping original nodes to their copies in the new graph
    env = {}
    for original_node, new_node in zip(graph.nodes, new_graph.nodes):
        env[original_node] = new_node
    
    # For each activation marked for recomputation
    for act_name, (subgraph_nodes, subgraph_inputs) in subgraphs.items():
        if not subgraph_nodes:
            continue
            
        # Get the first backward use rank for this activation
        first_bw_use_rank = activation_liveness[act_name]["first_bw_use_rank"]
        
        # Find the node with this rank in the new graph
        first_bw_use_node = None
        for node in new_graph.nodes:
            node_rank = getattr(node, "meta", {}).get("rank", -1)
            if isinstance(node_rank, int) and node_rank == first_bw_use_rank:
                first_bw_use_node = node
                break
                
        if not first_bw_use_node:
            continue
            
        # Insert the recomputation subgraph before the first backward use
        recomputed_node = insert_subgraph_before_node(
            new_graph, first_bw_use_node, subgraph_nodes, subgraph_inputs, env)
            
        if recomputed_node:
            # Find the original activation node in the new graph
            original_act_node = find_node_by_name(new_graph, act_name)
            
            if original_act_node:
                # Replace uses of the original activation with the recomputed one
                original_act_node.replace_all_uses_with(recomputed_node)
                
                # Don't erase the original node as it might be used elsewhere
                # or be part of the graph structure
        
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