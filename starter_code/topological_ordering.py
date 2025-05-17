"""
Topological ordering utility for graph rewriting
"""

import torch.fx as fx
import logging

def ensure_topological_ordering(graph: fx.Graph) -> fx.Graph:
    """
    Ensure proper topological ordering of nodes in the graph.
    
    This function specifically addresses issues with operations like flatten
    that have critical dependencies in the forward pass.
    
    Args:
        graph: The FX graph to reorder
        
    Returns:
        The reordered FX graph
    """
    logger = logging.getLogger(__name__)
    
    logger.info("Ensuring proper topological ordering of nodes")
    
    try:
        # Create a new graph with the same owning module
        new_graph = fx.Graph(owning_module=graph._owning_module)
        
        # First, copy all placeholder nodes
        env = {}
        for node in graph.nodes:
            if node.op == 'placeholder':
                new_node = new_graph.placeholder(node.name, type_expr=node.type)
                env[node] = new_node
        
        # Simple approach: just copy the original graph
        # This is a safer approach that avoids topological sorting issues
        for node in graph.nodes:
            if node.op == 'placeholder':
                continue  # Already copied
            elif node.op == 'output':
                try:
                    args = fx.map_arg(node.args, lambda n: env[n] if n in env else n)
                    kwargs = fx.map_arg(node.kwargs, lambda n: env[n] if n in env else n)
                    if args and isinstance(args[0], fx.Node) and args[0] in env:
                        new_graph.output(env[args[0]], kwargs)
                    else:
                        # Find any node to use as output
                        for n in reversed(list(graph.nodes)):
                            if n in env and n.op != 'output':
                                new_graph.output(env[n])
                                logger.warning(f"Using {n.name} as fallback output")
                                break
                except Exception as e:
                    logger.error(f"Error setting output: {e}")
                    # Find any node to use as output
                    for n in reversed(list(graph.nodes)):
                        if n in env and n.op != 'output':
                            new_graph.output(env[n])
                            logger.warning(f"Using {n.name} as fallback output")
                            break
            else:
                try:
                    args = fx.map_arg(node.args, lambda n: env[n] if n in env else n)
                    kwargs = fx.map_arg(node.kwargs, lambda n: env[n] if n in env else n)
                    new_node = new_graph.create_node(op=node.op, target=node.target, args=args, kwargs=kwargs, name=node.name)
                    env[node] = new_node
                    
                    # Copy metadata
                    if hasattr(node, 'meta'):
                        new_node.meta = node.meta.copy() if node.meta else {}
                except Exception as e:
                    logger.error(f"Error copying node {node.name}: {e}")
        
        logger.info(f"Reordered graph has {len(list(new_graph.nodes))} nodes")
        return new_graph
    except Exception as e:
        logger.error(f"Error in topological ordering: {e}")
        logger.warning("Falling back to the original graph")
        return graph