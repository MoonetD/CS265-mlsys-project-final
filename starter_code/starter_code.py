import logging
import os
from functools import wraps
from typing import Any

import torch
import torch.fx as fx
import torch.multiprocessing as mp
import torch.nn as nn
import torchvision.models as models # Added for ResNet
from transformers import BertModel, BertConfig # Added for BERT

from graph_prof import GraphProfiler
from graph_tracer import SEPFunction, compile

# This is the dummy model that is for use in starter_code. But we will
# experiment with Resnet and Transformer model.


class DummyModel(nn.Module):
    def __init__(self, layers: int, dim: int):
        super().__init__()
        modules = []
        for _ in range(layers):
            modules.extend([nn.Linear(dim, dim), nn.ReLU()])
        self.mod = nn.Sequential(*modules)

    def forward(self, x):
        return self.mod(x)


# We wrap the loss with a separator function to call a
# dummy function 'SEPFunction', which is the separator function, that will call
# an identity operator at the end of the forward pass. This identity operator
# will get recorded in the computational graph and will inform you where the
# backward pass ends.


# This is the train_step function that takes in a model, optimizer and an input
# mini batch and calls the forward pass, loss function and the optimizer step. A
# computational graph corresponding to a train_step will be captured by the
# compiler.


def train_step(
    model: torch.nn.Module, optim: torch.optim.Optimizer, batch: torch.Tensor
):
    loss =  model(batch).sum()
    loss = SEPFunction.apply(loss)
    loss.backward()
    optim.step()
    optim.zero_grad()


# Below is a user defined function that accepts a graph module and arguments of
# used to run the graph. You can essentially do any operation, graph
# modification, profiling etc. inside this function. Subsequent to modifications
# or graph analysis, the function expects you to return the modified graph back.
# In the given example, we just print the graph, and then initilize the graph
# profiler. The graph profiler extends the class fx.Interpreter, that allows you
# to run the graph node by node, more explanation in graph_prof.py.


def graph_transformation(gm: fx.GraphModule, args: Any) -> fx.GraphModule:
    print(gm.graph)
    graph_profiler = GraphProfiler(gm)
    # User request: 1 warm-up, 3 profile iterations. Profiler already uses median.
    warm_up_iters, profile_iters = 1, 3
    with torch.no_grad():
        for _ in range(warm_up_iters):
            graph_profiler.run(*args)
        graph_profiler.reset_stats()
        for _ in range(profile_iters):
            graph_profiler.run(*args)
    graph_profiler.aggregate_stats(num_runs=profile_iters)
    graph_profiler.print_stats()
    graph_profiler.save_stats_to_csv() # Save stats to CSV
    graph_profiler.plot_stats() # Generate and save plots
    return gm


# We first initialize the model, pass it to the wrapper model, then create a
# random input mini-batch and initilize the optimizer. We then call the compile
# function that takes in two arguments, a train_step function and a
# graph_transformation function. The train_step function is the one that will be
# traced by the compiler and a computational graph for the same will be created.
# This computational graph is then passed to the graph_transformation function
# to do any graph profiling, modifications and optimizations. This modified
# graph is stored and will be returned as the compiled function. In essence we
# do the following inside the compile function:

# def compile (train_step, graph_transformation):
#     @wraps(train_step)
#     def inner(*args, **kwargs):
#         if not_compiled:
#             original_graph, input_args = graph_tracer(train_step)
#             modified_graph = graph_transformation(original_graph, input_args)
#         output = modified_graph(*args, **kwargs)
#         return output
#     return inner


def experiment():
    logging.getLogger().setLevel(logging.DEBUG)
    torch.manual_seed(20)
    device_str = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device_str}")

    # Common profiling settings
    warm_up_iters, profile_iters = 1, 3 # As per user request & paper
    
    # --- ResNet-152 Experiment ---
    print("\n--- Profiling ResNet-152 ---")
    try:
        resnet_model = models.resnet152(weights=models.ResNet152_Weights.IMAGENET1K_V1).to(device_str)
        # Try a larger batch size for ResNet to increase memory usage
        # Typical input: (N, C, H, W)
        resnet_batch_size = 16 # Adjust as needed, start with a moderate size
        resnet_batch = torch.randn(resnet_batch_size, 3, 224, 224).to(device_str)
        resnet_optim = torch.optim.Adam(
            resnet_model.parameters(), lr=0.001, foreach=True, capturable=True
        )
        # Initialize gradients for optimizer step to be traceable
        for param in resnet_model.parameters():
            if param.requires_grad:
                param.grad = torch.rand_like(param, device=device_str)
        resnet_optim.step() # Perform one step to initialize optimizer states if needed by graph
        resnet_optim.zero_grad()

        # Wrap train_step for ResNet
        def resnet_train_step_wrapper(model, optim, batch):
            return train_step(model, optim, batch)

        compiled_resnet_fn = compile(resnet_train_step_wrapper, graph_transformation)
        print(f"Profiling ResNet-152 with batch size: {resnet_batch_size}")
        compiled_resnet_fn(resnet_model, resnet_optim, resnet_batch)
        print("--- ResNet-152 Profiling Complete ---")
    except Exception as e:
        print(f"Error during ResNet-152 profiling: {e}")

    # --- BERT Experiment ---
    print("\n--- Profiling BERT ---")
    try:
        bert_config = BertConfig(hidden_size=768, num_hidden_layers=12, num_attention_heads=12) # Base BERT config
        bert_model = BertModel(bert_config).to(device_str)
        # Try a larger batch size for BERT
        bert_batch_size = 8 # Adjust as needed
        bert_seq_length = 128
        bert_batch_ids = torch.randint(0, bert_config.vocab_size, (bert_batch_size, bert_seq_length)).to(device_str)
        bert_batch_mask = torch.ones(bert_batch_size, bert_seq_length, dtype=torch.long).to(device_str)
        # BERT forward pass typically takes input_ids and attention_mask
        # For train_step, we need a single tensor input for model(batch).
        # We'll pack them and unpack in a wrapper, or simplify train_step for BERT.
        # For now, let's adapt train_step slightly or use a model wrapper for BERT.
        # Simpler: modify train_step to accept a tuple if model is BERT, or use a wrapper.
        # Let's use a wrapper for the model's forward pass for simplicity with current train_step
        
        class BertWrapper(nn.Module):
            def __init__(self, bert):
                super().__init__()
                self.bert = bert
            def forward(self, inputs): # inputs is a tuple (input_ids, attention_mask)
                input_ids, attention_mask = inputs
                return self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

        bert_model_wrapped = BertWrapper(bert_model).to(device_str)
        bert_batch_tuple = (bert_batch_ids, bert_batch_mask)

        bert_optim = torch.optim.Adam(
            bert_model_wrapped.parameters(), lr=0.001, foreach=True, capturable=True
        )
        for param in bert_model_wrapped.parameters():
            if param.requires_grad:
                param.grad = torch.rand_like(param, device=device_str)
        bert_optim.step()
        bert_optim.zero_grad()
        
        # Wrap train_step for BERT
        def bert_train_step_wrapper(model_wrapped, optim_actual, batch_input_tuple):
            # model_wrapped is bert_model_wrapped. Its forward() returns the full last_hidden_state.
            # optim_actual is bert_optim
            # batch_input_tuple is (input_ids, attention_mask)
            
            last_hidden_state = model_wrapped(batch_input_tuple) # Shape: (batch_size, seq_length, hidden_size)
            # Alternative reduction to scalar: sum over hidden_size, then mean over batch and seq_length
            loss = last_hidden_state.sum(dim=-1).mean()

            # Apply separator, backward pass, and optimizer step directly
            loss = SEPFunction.apply(loss)
            loss.backward()
            optim_actual.step()
            optim_actual.zero_grad()
            
            # The compiled function from `compile` might expect to return the output of the traced function.
            # For profiling, the primary goal is the execution of the graph with side effects.
            # Returning loss is consistent with typical training steps.
            return loss

        compiled_bert_fn = compile(bert_train_step_wrapper, graph_transformation)
        print(f"Profiling BERT with batch size: {bert_batch_size}, sequence length: {bert_seq_length}")
        compiled_bert_fn(bert_model_wrapped, bert_optim, bert_batch_tuple)
        print("--- BERT Profiling Complete ---")
    except Exception as e:
        print(f"Error during BERT profiling: {e}")


if __name__ == "__main__":
    experiment()
