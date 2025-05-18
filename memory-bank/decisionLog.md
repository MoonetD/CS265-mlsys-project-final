# Decision Log

This file records architectural and implementation decisions using a list format.
2025-05-12 18:14:49 - Log of updates made.

* [2025-05-16 21:00:53] - Optimized pandas operations in `activation_checkpointing.py` by replacing them with native Python data structures.

## Decision

*

## Rationale 

*

## Implementation Details

*
---
### Decision (Code)
[2025-05-17 10:55:42] - Modified activation_checkpointing_torch.py to support both ResNet and Transformer models

**Rationale:**
The original implementation only supported ResNet models, but we needed to extend it to support Transformer models as well for the final deliverables. This required detecting the model type and applying different checkpointing strategies based on the architecture.

**Details:**
- Added a `_determine_model_type` method to detect whether the model is a ResNet or Transformer
- Refactored the `_identify_modules_to_checkpoint` method to handle both model types
- Created separate methods for ResNet (`_identify_resnet_modules_to_checkpoint`) and Transformer (`_identify_transformer_modules_to_checkpoint`)
- For Transformer models, prioritized checkpointing encoder layers, which contain the largest activations

---
### Decision (Code)
[2025-05-17 10:55:42] - Enhanced run_test_torch.py to generate comparison graphs for different batch sizes

**Rationale:**
The original implementation only tested a single batch size (32) for ResNet. We needed to extend it to test multiple batch sizes for both ResNet and Transformer models and generate comparison graphs as required by the project deliverables.

**Details:**
- Added command-line arguments to specify model type and batch size
- Created functions to generate memory and time comparison plots
- Implemented batch size configurations from batch_process_ac.py (ResNet: [4, 8, 16, 32, 64], Transformer: [2, 4, 8, 16, 32, 64, 128, 256])
- Added support for saving plots to the final_images directory