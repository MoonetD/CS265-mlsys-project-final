"""
Simple script to run the graph rewriter test.
"""

import os
import sys
import torch

# Check if CUDA is available
if not torch.cuda.is_available():
    print("CUDA is not available. This test requires a GPU.")
    sys.exit(1)

# Create logs directory if it doesn't exist
os.makedirs("logs", exist_ok=True)

# Run the graph rewriter test
print("Running graph rewriter test...")
from run_graph_rewriter import main
main()