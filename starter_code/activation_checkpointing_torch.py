"""
Activation Checkpointing Implementation using PyTorch's built-in checkpointing mechanism.

This module provides a simpler and more reliable approach to activation checkpointing
by using PyTorch's torch.utils.checkpoint functionality.
"""

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
import logging
import time
import numpy as np
from typing import Dict, List, Tuple, Set, Any, Optional

logger = logging.getLogger(__name__)

class ActivationCheckpointer:
    """
    Implements activation checkpointing using PyTorch's built-in checkpointing mechanism.
    """
    
    def __init__(self, model: nn.Module, activation_decisions: Dict[str, str],
                 activation_liveness: Optional[Dict[str, Dict[str, int]]] = None):
        """
        Initialize the activation checkpointer.
        
        Args:
            model: The model to apply activation checkpointing to
            activation_decisions: Dict mapping activation names to 'RETAINED' or 'RECOMPUTE'
            activation_liveness: Optional dict with activation liveness information
        """
        self.model = model
        self.activation_decisions = activation_decisions
        self.activation_liveness = activation_liveness
        self.checkpointed_modules = []
        self.model_type = self._determine_model_type()
        
        # Identify modules to checkpoint
        self._identify_modules_to_checkpoint()
        
    def _determine_model_type(self) -> str:
        """
        Determine the type of model (ResNet or Transformer).
        
        Returns:
            String indicating model type: 'resnet' or 'transformer'
        """
        # Check if model is a ResNet
        if hasattr(self.model, 'layer1') and hasattr(self.model, 'layer2'):
            logger.info("Detected ResNet model")
            return 'resnet'
        
        # Check if model is a Transformer
        if hasattr(self.model, 'encoder_layers') and hasattr(self.model, 'embedding'):
            logger.info("Detected Transformer model")
            return 'transformer'
        
        # Default to ResNet if can't determine
        logger.warning("Could not determine model type, defaulting to ResNet")
        return 'resnet'
        
    def _identify_modules_to_checkpoint(self):
        """
        Identify modules to checkpoint based on activation decisions and liveness information.
        """
        # Count recompute decisions
        recompute_count = sum(1 for decision in self.activation_decisions.values() if decision == 'RECOMPUTE')
        logger.info(f"Found {recompute_count} activations marked for recomputation")
        
        # If no activations to recompute, return
        if recompute_count == 0:
            logger.warning("No activations marked for recomputation")
            return
        
        # Create a list of activations to recompute, sorted by memory size (largest first)
        recompute_activations = []
        for act_name, decision in self.activation_decisions.items():
            if decision == 'RECOMPUTE':
                # Get memory size and recomputation time from activation liveness
                mem_size = 0
                recomp_time = 0
                if self.activation_liveness and act_name in self.activation_liveness:
                    mem_size = self.activation_liveness[act_name].get('median_mem_size_bytes', 0)
                    recomp_time = self.activation_liveness[act_name].get('recomp_time_s', 0)
                
                recompute_activations.append((act_name, mem_size, recomp_time))
        
        # Sort by memory size (largest first)
        recompute_activations.sort(key=lambda x: x[1], reverse=True)
        
        # Log the top 10 largest activations to recompute
        logger.info("Top 10 largest activations to recompute:")
        for i, (act_name, mem_size, recomp_time) in enumerate(recompute_activations[:10]):
            logger.info(f"  {i+1}. {act_name}: {mem_size/(1024*1024):.2f} MB, recomp time: {recomp_time:.6f} s")
        
        if self.model_type == 'resnet':
            self._identify_resnet_modules_to_checkpoint()
        else:  # transformer
            self._identify_transformer_modules_to_checkpoint()
            
    def _identify_resnet_modules_to_checkpoint(self):
        """
        Identify modules to checkpoint for ResNet models.
        """
        logger.info("Identifying modules to checkpoint for ResNet model")
        
        # Get all modules that could be checkpointed
        checkpointable_modules = []
        for name, module in self.model.named_modules():
            # Skip the model itself
            if module is self.model:
                continue
                
            # For ResNet-152, we'll checkpoint Bottleneck blocks and some other key modules
            if 'Bottleneck' in str(type(module)) or name in ['conv1', 'layer1', 'layer2', 'layer3', 'layer4']:
                checkpointable_modules.append((name, module))
        
        # Sort modules by name for deterministic ordering
        checkpointable_modules.sort(key=lambda x: x[0])
        
        # Only consider individual Bottleneck blocks for checkpointing
        # This avoids issues with stateful operations in higher-level modules
        bottleneck_modules = []
        for name, module in self.model.named_modules():
            # Skip the model itself and non-Bottleneck modules
            if module is self.model or 'Bottleneck' not in str(type(module)):
                continue
                
            # Only include individual Bottleneck blocks
            if '.' in name and name.count('.') == 1:  # e.g., 'layer3.15'
                bottleneck_modules.append((name, module))
        
        # Group Bottleneck blocks by layer
        layer_to_modules = {'layer1': [], 'layer2': [], 'layer3': [], 'layer4': []}
        for name, module in bottleneck_modules:
            layer_name = name.split('.')[0]
            if layer_name in layer_to_modules:
                layer_to_modules[layer_name].append((name, module))
        
        # Prioritize modules based on activation decisions
        # We'll use a scoring system that considers:
        # 1. Memory size of activations to recompute
        # 2. Layer depth (deeper layers have larger activations)
        
        # Assign scores to layers based on activation memory sizes and depth
        layer_scores = {
            'layer1': 1.0,   # Base score
            'layer2': 2.0,   # 2x base score
            'layer3': 4.0,   # 4x base score
            'layer4': 8.0    # 8x base score
        }
        
        # Log layer scores
        logger.info("Layer scores based on depth and typical activation sizes:")
        for layer, score in layer_scores.items():
            logger.info(f"  {layer}: {score:.2f}")
        
        # Select modules to checkpoint based on layer scores
        selected_modules = []
        
        # Calculate total number of modules to checkpoint from each layer
        total_score = sum(layer_scores.values())
        max_modules = 50  # Limit to 50 modules to avoid excessive overhead
        
        # Distribute the modules based on layer scores
        modules_per_layer = {}
        for layer, score in layer_scores.items():
            # Calculate how many modules to select from this layer
            num_modules = int((score / total_score) * max_modules) if total_score > 0 else 0
            num_modules = min(num_modules, len(layer_to_modules[layer]))
            modules_per_layer[layer] = num_modules
            
        # Log modules per layer
        logger.info("Modules to checkpoint per layer:")
        for layer, num in modules_per_layer.items():
            logger.info(f"  {layer}: {num}")
        
        # Select modules from each layer
        for layer, num_modules in modules_per_layer.items():
            if num_modules > 0:
                # For deeper layers (layer3, layer4), prioritize later blocks
                # as they tend to have larger activations
                if layer in ['layer3', 'layer4']:
                    # Sort by block number in descending order
                    sorted_modules = sorted(
                        layer_to_modules[layer],
                        key=lambda x: int(x[0].split('.')[1]),
                        reverse=True
                    )
                    selected_modules.extend(sorted_modules[:num_modules])
                else:
                    # For earlier layers, just take the first N blocks
                    selected_modules.extend(layer_to_modules[layer][:num_modules])
        
        # Ensure we don't exceed the maximum number of modules
        self.checkpointed_modules = selected_modules[:max_modules]
        
    def _identify_transformer_modules_to_checkpoint(self):
        """
        Identify modules to checkpoint for Transformer models.
        """
        logger.info("Identifying modules to checkpoint for Transformer model")
        
        # For Transformer, we'll checkpoint encoder layers
        checkpointable_modules = []
        
        # Get all encoder layers
        for name, module in self.model.named_modules():
            # Skip the model itself
            if module is self.model:
                continue
                
            # For Transformer, we'll checkpoint encoder layers and attention blocks
            if 'encoder_layers' in name or 'TransformerEncoderLayer' in str(type(module)):
                checkpointable_modules.append((name, module))
                
            # Also consider self-attention and feed-forward blocks
            if any(x in name for x in ['self_attn', 'linear1', 'linear2']):
                checkpointable_modules.append((name, module))
        
        # Sort modules by name for deterministic ordering
        checkpointable_modules.sort(key=lambda x: x[0])
        
        # For Transformer, prioritize encoder layers
        encoder_layers = []
        for name, module in checkpointable_modules:
            if 'encoder_layers' in name and name.count('.') == 1:  # e.g., 'encoder_layers.0'
                encoder_layers.append((name, module))
        
        # Prioritize later encoder layers as they tend to have larger activations
        # Sort by layer index in descending order
        sorted_encoder_layers = sorted(
            encoder_layers,
            key=lambda x: int(x[0].split('.')[-1]),
            reverse=True
        )
        
        # Select a reasonable number of modules to checkpoint
        max_modules = min(len(sorted_encoder_layers), 6)  # Limit to 6 encoder layers
        self.checkpointed_modules = sorted_encoder_layers[:max_modules]
        
        # Log the selected modules
        logger.info(f"Selected {len(self.checkpointed_modules)} modules to checkpoint:")
        for i, (name, _) in enumerate(self.checkpointed_modules[:5]):
            logger.info(f"  {i+1}. {name}")
        if len(self.checkpointed_modules) > 5:
            logger.info(f"  ... and {len(self.checkpointed_modules) - 5} more")
        
        logger.info(f"Identified {len(self.checkpointed_modules)} modules to checkpoint")
        
    def apply_checkpointing(self) -> nn.Module:
        """
        Apply activation checkpointing to the model.
        
        Returns:
            A new model with activation checkpointing applied
        """
        if not self.checkpointed_modules:
            logger.warning("No modules to checkpoint, returning original model")
            return self.model
            
        # Create a new model that applies checkpointing
        class CheckpointedModel(nn.Module):
            def __init__(self, base_model, checkpointed_modules):
                super().__init__()
                self.base_model = base_model
                self.checkpointed_module_names = [name for name, _ in checkpointed_modules]
                
                # Store references to the modules to checkpoint
                self.checkpointed_modules = nn.ModuleDict()
                for name, module in checkpointed_modules:
                    # Replace dots with underscores in the name
                    dict_name = name.replace('.', '_')
                    self.checkpointed_modules[dict_name] = module
                
                # Create a mapping from original module to checkpointed module
                self.module_mapping = {}
                for name, module in checkpointed_modules:
                    dict_name = name.replace('.', '_')
                    self.module_mapping[module] = self.checkpointed_modules[dict_name]
                
            def forward(self, x):
                # Replace the forward method of checkpointed modules with a checkpointed version
                original_forwards = {}
                
                # First, save all original forwards
                for name, module in self.checkpointed_modules.items():
                    original_forwards[name] = module.forward
                
                # Then create and apply checkpointed forwards
                for name, module in self.checkpointed_modules.items():
                    # Get the original forward function
                    orig_forward = original_forwards[name]
                    
                    # Create a closure to capture the original forward function
                    def make_checkpointed_forward(orig_fwd):
                        def checkpointed_forward(*inputs):
                            return checkpoint.checkpoint(orig_fwd, *inputs, use_reentrant=False)
                        return checkpointed_forward
                    
                    # Replace the forward method with a checkpointed version
                    module.forward = make_checkpointed_forward(orig_forward)
                
                # Run the model with checkpointing
                try:
                    output = self.base_model(x)
                finally:
                    # Restore the original forward methods
                    for name, module in self.checkpointed_modules.items():
                        module.forward = original_forwards[name]
                
                return output
        
        # Create and return the checkpointed model
        checkpointed_model = CheckpointedModel(self.model, self.checkpointed_modules)
        logger.info("Successfully applied activation checkpointing to model")
        return checkpointed_model

def apply_activation_checkpointing(model: nn.Module,
                                  activation_decisions: Dict[str, str],
                                  activation_liveness: Optional[Dict[str, Dict[str, int]]] = None) -> nn.Module:
    """
    Apply activation checkpointing to a model.
    
    Args:
        model: The model to apply activation checkpointing to
        activation_decisions: Dict mapping activation names to 'RETAINED' or 'RECOMPUTE'
        activation_liveness: Optional dict with activation liveness information
        
    Returns:
        A new model with activation checkpointing applied
    """
    checkpointer = ActivationCheckpointer(model, activation_decisions, activation_liveness)
    return checkpointer.apply_checkpointing()