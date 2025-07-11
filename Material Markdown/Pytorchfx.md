torch.fx
Overview
FX is a toolkit for developers to use to transform nn.Module instances. FX consists of three main components: a symbolic tracer, an intermediate representation, and Python code generation. A demonstration of these components in action:

import torch


# Simple module for demonstration
class MyModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.param = torch.nn.Parameter(torch.rand(3, 4))
        self.linear = torch.nn.Linear(4, 5)

    def forward(self, x):
        return self.linear(x + self.param).clamp(min=0.0, max=1.0)


module = MyModule()

from torch.fx import symbolic_trace

# Symbolic tracing frontend - captures the semantics of the module
symbolic_traced: torch.fx.GraphModule = symbolic_trace(module)

# High-level intermediate representation (IR) - Graph representation
print(symbolic_traced.graph)
"""
graph():
    %x : [num_users=1] = placeholder[target=x]
    %param : [num_users=1] = get_attr[target=param]
    %add : [num_users=1] = call_function[target=operator.add](args = (%x, %param), kwargs = {})
    %linear : [num_users=1] = call_module[target=linear](args = (%add,), kwargs = {})
    %clamp : [num_users=1] = call_method[target=clamp](args = (%linear,), kwargs = {min: 0.0, max: 1.0})
    return clamp
"""

# Code generation - valid Python code
print(symbolic_traced.code)
"""
def forward(self, x):
    param = self.param
    add = x + param;  x = param = None
    linear = self.linear(add);  add = None
    clamp = linear.clamp(min = 0.0, max = 1.0);  linear = None
    return clamp
"""
The symbolic tracer performs “symbolic execution” of the Python code. It feeds fake values, called Proxies, through the code. Operations on theses Proxies are recorded. More information about symbolic tracing can be found in the symbolic_trace() and Tracer documentation.

The intermediate representation is the container for the operations that were recorded during symbolic tracing. It consists of a list of Nodes that represent function inputs, callsites (to functions, methods, or torch.nn.Module instances), and return values. More information about the IR can be found in the documentation for Graph. The IR is the format on which transformations are applied.

Python code generation is what makes FX a Python-to-Python (or Module-to-Module) transformation toolkit. For each Graph IR, we can create valid Python code matching the Graph’s semantics. This functionality is wrapped up in GraphModule, which is a torch.nn.Module instance that holds a Graph as well as a forward method generated from the Graph.

Taken together, this pipeline of components (symbolic tracing -> intermediate representation -> transforms -> Python code generation) constitutes the Python-to-Python transformation pipeline of FX. In addition, these components can be used separately. For example, symbolic tracing can be used in isolation to capture a form of the code for analysis (and not transformation) purposes. Code generation can be used for programmatically generating models, for example from a config file. There are many uses for FX!

Several example transformations can be found at the examples repository.

Writing Transformations
What is an FX transform? Essentially, it’s a function that looks like this.

import torch
import torch.fx

def transform(m: nn.Module,
              tracer_class : type = torch.fx.Tracer) -> torch.nn.Module:
    # Step 1: Acquire a Graph representing the code in `m`

    # NOTE: torch.fx.symbolic_trace is a wrapper around a call to
    # fx.Tracer.trace and constructing a GraphModule. We'll
    # split that out in our transform to allow the caller to
    # customize tracing behavior.
    graph : torch.fx.Graph = tracer_class().trace(m)

    # Step 2: Modify this Graph or create a new one
    graph = ...

    # Step 3: Construct a Module to return
    return torch.fx.GraphModule(m, graph)
Your transform will take in a torch.nn.Module, acquire a Graph from it, do some modifications, and return a new torch.nn.Module. You should think of the torch.nn.Module that your FX transform returns as identical to a regular torch.nn.Module – you can pass it to another FX transform, you can pass it to TorchScript, or you can run it. Ensuring that the inputs and outputs of your FX transform are a torch.nn.Module will allow for composability.

Note

It is also possible to modify an existing GraphModule instead of creating a new one, like so:

import torch
import torch.fx

def transform(m : nn.Module) -> nn.Module:
    gm : torch.fx.GraphModule = torch.fx.symbolic_trace(m)

    # Modify gm.graph
    # <...>

    # Recompile the forward() method of `gm` from its Graph
    gm.recompile()

    return gm
Note that you MUST call GraphModule.recompile() to bring the generated forward() method on the GraphModule in sync with the modified Graph.

Given that you’ve passed in a torch.nn.Module that has been traced into a Graph, there are now two primary approaches you can take to building a new Graph.

A Quick Primer on Graphs
Full treatment of the semantics of graphs can be found in the Graph documentation, but we are going to cover the basics here. A Graph is a data structure that represents a method on a GraphModule. The information that this requires is:

What are the inputs to the method?

What are the operations that run inside the method?

What is the output (i.e. return) value from the method?

All three of these concepts are represented with Node instances. Let’s see what we mean by that with a short example:

import torch
import torch.fx

class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.param = torch.nn.Parameter(torch.rand(3, 4))
        self.linear = torch.nn.Linear(4, 5)

    def forward(self, x):
        return torch.topk(torch.sum(
            self.linear(x + self.linear.weight).relu(), dim=-1), 3)

m = MyModule()
gm = torch.fx.symbolic_trace(m)

gm.graph.print_tabular()
Here we define a module MyModule for demonstration purposes, instantiate it, symbolically trace it, then call the Graph.print_tabular() method to print out a table showing the nodes of this Graph:

opcode

name

target

args

kwargs

placeholder

x

x

()

{}

get_attr

linear_weight

linear.weight

()

{}

call_function

add_1

<built-in function add>

(x, linear_weight)

{}

call_module

linear_1

linear

(add_1,)

{}

call_method

relu_1

relu

(linear_1,)

{}

call_function

sum_1

<built-in method sum …>

(relu_1,)

{‘dim’: -1}

call_function

topk_1

<built-in method topk …>

(sum_1, 3)

{}

output

output

output

(topk_1,)

{}

We can use this information to answer the questions we posed above.

What are the inputs to the method? In FX, method inputs are specified via special placeholder nodes. In this case, we have a single placeholder node with a target of x, meaning we have a single (non-self) argument named x.

What are the operations within the method? The get_attr, call_function, call_module, and call_method nodes represent the operations in the method. A full treatment of the semantics of all of these can be found in the Node documentation.

What is the return value of the method? The return value in a Graph is specified by a special output node.

Given that we now know the basics of how code is represented in FX, we can now explore how we would edit a Graph.

Graph Manipulation
Direct Graph Manipulation
One approach to building this new Graph is to directly manipulate your old one. To aid in this, we can simply take the Graph we obtain from symbolic tracing and modify it. For example, let’s say we desire to replace torch.add() calls with torch.mul() calls.

import torch
import torch.fx

# Sample module
class M(torch.nn.Module):
    def forward(self, x, y):
        return torch.add(x, y)

def transform(m: torch.nn.Module,
              tracer_class : type = fx.Tracer) -> torch.nn.Module:
    graph : fx.Graph = tracer_class().trace(m)
    # FX represents its Graph as an ordered list of
    # nodes, so we can iterate through them.
    for node in graph.nodes:
        # Checks if we're calling a function (i.e:
        # torch.add)
        if node.op == 'call_function':
            # The target attribute is the function
            # that call_function calls.
            if node.target == torch.add:
                node.target = torch.mul

    graph.lint() # Does some checks to make sure the
                 # Graph is well-formed.

    return fx.GraphModule(m, graph)
We can also do more involved Graph rewrites, such as deleting or appending nodes. To aid in these transformations, FX has utility functions for transforming the graph that can be found in the Graph documentation. An example of using these APIs to append a torch.relu() call can be found below.

# Specifies the insertion point. Any nodes added to the
# Graph within this scope will be inserted after `node`
with traced.graph.inserting_after(node):
    # Insert a new `call_function` node calling `torch.relu`
    new_node = traced.graph.call_function(
        torch.relu, args=(node,))

    # We want all places that used the value of `node` to
    # now use that value after the `relu` call we've added.
    # We use the `replace_all_uses_with` API to do this.
    node.replace_all_uses_with(new_node)
For simple transformations that only consist of substitutions, you can also make use of the subgraph rewriter.

Subgraph Rewriting With replace_pattern()
FX also provides another level of automation on top of direct graph manipulation. The replace_pattern() API is essentially a “find/replace” tool for editing Graphs. It allows you to specify a pattern and replacement function and it will trace through those functions, find instances of the group of operations in the pattern graph, and replace those instances with copies of the replacement graph. This can help to greatly automate tedious graph manipulation code, which can get unwieldy as the transformations get more complex.

Graph Manipulation Examples
Replace one op

Conv/Batch Norm fusion

replace_pattern: Basic usage

Quantization

Invert Transformation

Proxy/Retracing
Another way of manipulating Graphs is by reusing the Proxy machinery used in symbolic tracing. For example, let’s imagine that we wanted to write a transformation that decomposed PyTorch functions into smaller operations. It would transform every F.relu(x) call into (x > 0) * x. One possibility would be to perform the requisite graph rewriting to insert the comparison and multiplication after the F.relu, and then clean up the original F.relu. However, we can automate this process by using Proxy objects to automatically record operations into the Graph.

To use this method, we write the operations that we want inserted as regular PyTorch code and invoke that code with Proxy objects as arguments. These Proxy objects will capture the operations that are performed on them and append them to the Graph.

# Note that this decomposition rule can be read as regular Python
def relu_decomposition(x):
    return (x > 0) * x

decomposition_rules = {}
decomposition_rules[F.relu] = relu_decomposition

def decompose(model: torch.nn.Module,
              tracer_class : type = fx.Tracer) -> torch.nn.Module:
    """
    Decompose `model` into smaller constituent operations.
    Currently,this only supports decomposing ReLU into its
    mathematical definition: (x > 0) * x
    """
    graph : fx.Graph = tracer_class().trace(model)
    new_graph = fx.Graph()
    env = {}
    tracer = torch.fx.proxy.GraphAppendingTracer(new_graph)
    for node in graph.nodes:
        if node.op == 'call_function' and node.target in decomposition_rules:
            # By wrapping the arguments with proxies,
            # we can dispatch to the appropriate
            # decomposition rule and implicitly add it
            # to the Graph by symbolically tracing it.
            proxy_args = [
                fx.Proxy(env[x.name], tracer) if isinstance(x, fx.Node) else x for x in node.args]
            output_proxy = decomposition_rules[node.target](*proxy_args)

            # Operations on `Proxy` always yield new `Proxy`s, and the
            # return value of our decomposition rule is no exception.
            # We need to extract the underlying `Node` from the `Proxy`
            # to use it in subsequent iterations of this transform.
            new_node = output_proxy.node
            env[node.name] = new_node
        else:
            # Default case: we don't have a decomposition rule for this
            # node, so just copy the node over into the new graph.
            new_node = new_graph.node_copy(node, lambda x: env[x.name])
            env[node.name] = new_node
    return fx.GraphModule(model, new_graph)
In addition to avoiding explicit graph manipulation, using Proxys also allows you to specify your rewrite rules as native Python code. For transformations that require a large amount of rewrite rules (such as vmap or grad), this can often improve readability and maintainability of the rules. Note that while calling Proxy we also passed a tracer pointing to the underlying variable graph. This is done so if in case the operations in graph are n-ary (e.g. add is a binary operator) the call to Proxy does not create multiple instances of a graph tracer which can lead to unexpected runtime errors. We recommend this method of using Proxy especially when the underlying operators can not be safely assumed to be unary.

A worked example of using Proxys for Graph manipulation can be found here.

The Interpreter Pattern
A useful code organizational pattern in FX is to loop over all the Nodes in a Graph and execute them. This can be used for several things including runtime analysis of values flowing through the graph or transformation of the code via retracing with Proxys. For example, suppose we want to run a GraphModule and record the torch.Tensor shape and dtype properties on the nodes as we see them at runtime. That might look like:

import torch
import torch.fx
from torch.fx.node import Node

from typing import Dict

class ShapeProp:
    """
    Shape propagation. This class takes a `GraphModule`.
    Then, its `propagate` method executes the `GraphModule`
    node-by-node with the given arguments. As each operation
    executes, the ShapeProp class stores away the shape and
    element type for the output values of each operation on
    the `shape` and `dtype` attributes of the operation's
    `Node`.
    """
    def __init__(self, mod):
        self.mod = mod
        self.graph = mod.graph
        self.modules = dict(self.mod.named_modules())

    def propagate(self, *args):
        args_iter = iter(args)
        env : Dict[str, Node] = {}

        def load_arg(a):
            return torch.fx.graph.map_arg(a, lambda n: env[n.name])

        def fetch_attr(target : str):
            target_atoms = target.split('.')
            attr_itr = self.mod
            for i, atom in enumerate(target_atoms):
                if not hasattr(attr_itr, atom):
                    raise RuntimeError(f"Node referenced nonexistent target {'.'.join(target_atoms[:i])}")
                attr_itr = getattr(attr_itr, atom)
            return attr_itr

        for node in self.graph.nodes:
            if node.op == 'placeholder':
                result = next(args_iter)
            elif node.op == 'get_attr':
                result = fetch_attr(node.target)
            elif node.op == 'call_function':
                result = node.target(*load_arg(node.args), **load_arg(node.kwargs))
            elif node.op == 'call_method':
                self_obj, *args = load_arg(node.args)
                kwargs = load_arg(node.kwargs)
                result = getattr(self_obj, node.target)(*args, **kwargs)
            elif node.op == 'call_module':
                result = self.modules[node.target](*load_arg(node.args), **load_arg(node.kwargs))

            # This is the only code specific to shape propagation.
            # you can delete this `if` branch and this becomes
            # a generic GraphModule interpreter.
            if isinstance(result, torch.Tensor):
                node.shape = result.shape
                node.dtype = result.dtype

            env[node.name] = result

        return load_arg(self.graph.result)
As you can see, a full interpreter for FX is not that complicated but it can be very useful. To ease using this pattern, we provide the Interpreter class, which encompasses the above logic in a way that certain aspects of the interpreter’s execution can be overridden via method overrides.

In addition to executing operations, we can also generate a new Graph by feeding Proxy values through an interpreter. Similarly, we provide the Transformer class to encompass this pattern. Transformer behaves similarly to Interpreter, but instead of calling the run method to get a concrete output value from the Module, you would call the Transformer.transform() method to return a new GraphModule which was subject to any transformation rules you installed as overridden methods.

Examples of the Interpreter Pattern
Shape Propagation

Performance Profiler

Debugging
Introduction
Often in the course of authoring transformations, our code will not be quite right. In this case, we may need to do some debugging. The key is to work backwards: first, check the results of invoking the generated module to prove or disprove correctness. Then, inspect and debug the generated code. Then, debug the process of transformations that led to the generated code.

If you’re not familiar with debuggers, please see the auxiliary section Available Debuggers.

Common Pitfalls in Transform Authoring
Nondeterministic set iteration order. In Python, the set datatype is unordered. Using set to contain collections of objects like Nodes, for example, can cause unexpected nondeterminism. An example is iterating over a set of Nodes to insert them into a Graph. Because the set data type is unordered, the ordering of the operations in the output program will be nondeterministic and can change across program invocations. The recommended alternative is to use a dict data type, which is insertion ordered as of Python 3.7 (and as of cPython 3.6). A dict can be used equivalently to a set by storing values to be deduplicated in the keys of the dict.

Checking Correctness of Modules
Because the output of most deep learning modules consists of floating point torch.Tensor instances, checking for equivalence between the results of two torch.nn.Module is not as straightforward as doing a simple equality check. To motivate this, let’s use an example:

import torch
import torch.fx
import torchvision.models as models

def transform(m : torch.nn.Module) -> torch.nn.Module:
    gm = torch.fx.symbolic_trace(m)

    # Imagine we're doing some transforms here
    # <...>

    gm.recompile()

    return gm

resnet18 = models.resnet18()
transformed_resnet18 = transform(resnet18)

input_image = torch.randn(5, 3, 224, 224)

assert resnet18(input_image) == transformed_resnet18(input_image)
"""
RuntimeError: Boolean value of Tensor with more than one value is ambiguous
"""
Here, we’ve tried to check equality of the values of two deep learning models with the == equality operator. However, this is not well- defined both due to the issue of that operator returning a tensor and not a bool, but also because comparison of floating point values should use a margin of error (or epsilon) to account for the non-commutativity of floating point operations (see here for more details). We can use torch.allclose() instead, which will give us an approximate comparison taking into account a relative and absolute tolerance threshold:

assert torch.allclose(resnet18(input_image), transformed_resnet18(input_image))
This is the first tool in our toolbox to check if transformed modules are behaving as we expect compared to a reference implementation.

Debugging the Generated Code
Because FX generates the forward() function on GraphModules, using traditional debugging techniques like print statements or pdb is not as straightforward. Luckily, we have several techniques we can use for debugging the generated code.

Use pdb
Invoke pdb to step into the running program. Although the code that represents the Graph is not in any source file, we can still step into it manually using pdb when the forward pass is invoked.

import torch
import torch.fx
import torchvision.models as models

def my_pass(inp: torch.nn.Module, tracer_class : type = fx.Tracer) -> torch.nn.Module:
    graph = tracer_class().trace(inp)
    # Transformation logic here
    # <...>

    # Return new Module
    return fx.GraphModule(inp, graph)

my_module = models.resnet18()
my_module_transformed = my_pass(my_module)

input_value = torch.randn(5, 3, 224, 224)

# When this line is executed at runtime, we will be dropped into an
# interactive `pdb` prompt. We can use the `step` or `s` command to
# step into the execution of the next line
import pdb; pdb.set_trace()

my_module_transformed(input_value)
Print the Generated Code
If you’d like to run the same code multiple times, then it can be a bit tedious to step to the right code with pdb. In that case, one approach is to simply copy-paste the generated forward pass into your code and examine it from there.

# Assume that `traced` is a GraphModule that has undergone some
# number of transforms

# Copy this code for later
print(traced)
# Print the code generated from symbolic tracing. This outputs:
"""
def forward(self, y):
    x = self.x
    add_1 = x + y;  x = y = None
    return add_1
"""

# Subclass the original Module
class SubclassM(M):
    def __init__(self):
        super().__init__()

    # Paste the generated `forward` function (the one we printed and
    # copied above) here
    def forward(self, y):
        x = self.x
        add_1 = x + y;  x = y = None
        return add_1

# Create an instance of the original, untraced Module. Then, create an
# instance of the Module with the copied `forward` function. We can
# now compare the output of both the original and the traced version.
pre_trace = M()
post_trace = SubclassM()
Use the to_folder Function From GraphModule
GraphModule.to_folder() is a method in GraphModule that allows you to dump out the generated FX code to a folder. Although copying the forward pass into the code often suffices as in Print the Generated Code, it may be easier to examine modules and parameters using to_folder.

m = symbolic_trace(M())
m.to_folder("foo", "Bar")
from foo import Bar
y = Bar()
After running the above example, we can then look at the code within foo/module.py and modify it as desired (e.g. adding print statements or using pdb) to debug the generated code.

Debugging the Transformation
Now that we’ve identified that a transformation is creating incorrect code, it’s time to debug the transformation itself. First, we’ll check the Limitations of Symbolic Tracing section in the documentation. Once we verify that tracing is working as expected, the goal becomes figuring out what went wrong during our GraphModule transformation. There may be a quick answer in Writing Transformations, but, if not, there are several ways to examine our traced module:

# Sample Module
class M(torch.nn.Module):
    def forward(self, x, y):
        return x + y

# Create an instance of `M`
m = M()

# Symbolically trace an instance of `M` (returns a GraphModule). In
# this example, we'll only be discussing how to inspect a
# GraphModule, so we aren't showing any sample transforms for the
# sake of brevity.
traced = symbolic_trace(m)

# Print the code produced by tracing the module.
print(traced)
# The generated `forward` function is:
"""
def forward(self, x, y):
    add = x + y;  x = y = None
    return add
"""

# Print the internal Graph.
print(traced.graph)
# This print-out returns:
"""
graph():
    %x : [num_users=1] = placeholder[target=x]
    %y : [num_users=1] = placeholder[target=y]
    %add : [num_users=1] = call_function[target=operator.add](args = (%x, %y), kwargs = {})
    return add
"""

# Print a tabular representation of the internal Graph.
traced.graph.print_tabular()
# This gives us:
"""
opcode         name    target                   args    kwargs
-------------  ------  -----------------------  ------  --------
placeholder    x       x                        ()      {}
placeholder    y       y                        ()      {}
call_function  add     <built-in function add>  (x, y)  {}
output         output  output                   (add,)  {}
"""
Using the utility functions above, we can compare our traced Module before and after we’ve applied our transformations. Sometimes, a simple visual comparison is enough to trace down a bug. If it’s still not clear what’s going wrong, a debugger like pdb can be a good next step.

Going off of the example above, consider the following code:

# Sample user-defined function
def transform_graph(module: torch.nn.Module, tracer_class : type = fx.Tracer) -> torch.nn.Module:
    # Get the Graph from our traced Module
    g = tracer_class().trace(module)

    """
    Transformations on `g` go here
    """

    return fx.GraphModule(module, g)

# Transform the Graph
transformed = transform_graph(traced)

# Print the new code after our transforms. Check to see if it was
# what we expected
print(transformed)
Using the above example, let’s say that the call to print(traced) showed us that there was an error in our transforms. We want to find what goes wrong using a debugger. We start a pdb session. We can see what’s happening during the transform by breaking on transform_graph(traced), then pressing s to “step into” the call to transform_graph(traced).

We may also have good luck by editing the print_tabular method to print different attributes of the Nodes in the Graph. (For example, we might want to see the Node’s input_nodes and users.)

Available Debuggers
The most common Python debugger is pdb. You can start your program in “debug mode” with pdb by typing python -m pdb FILENAME.py into the command line, where FILENAME is the name of the file you want to debug. After that, you can use the pdb debugger commands to move through your running program stepwise. It’s common to set a breakpoint (b LINE-NUMBER) when you start pdb, then call c to run the program until that point. This prevents you from having to step through each line of execution (using s or n) to get to the part of the code you want to examine. Alternatively, you can write import pdb; pdb.set_trace() before the line you want to break at. If you add pdb.set_trace(), your program will automatically start in debug mode when you run it. (In other words, you can just type python FILENAME.py into the command line instead of python -m pdb FILENAME.py.) Once you’re running your file in debug mode, you can step through the code and examine your program’s internal state using certain commands. There are many excellent tutorials on pdb online, including RealPython’s “Python Debugging With Pdb”.

IDEs like PyCharm or VSCode usually have a debugger built in. In your IDE, you can choose to either a) use pdb by pulling up a terminal window in your IDE (e.g. View → Terminal in VSCode), or b) use the built-in debugger (usually a graphical wrapper around pdb).

Limitations of Symbolic Tracing
FX uses a system of symbolic tracing (a.k.a symbolic execution) to capture the semantics of programs in a transformable/analyzable form. The system is tracing in that it executes the program (really a torch.nn.Module or function) to record operations. It is symbolic in that the data flowing through the program during this execution is not real data, but rather symbols (Proxy in FX parlance).

Although symbolic tracing works for most neural net code, it has some limitations.

Dynamic Control Flow
The main limitation of symbolic tracing is it does not currently support dynamic control flow. That is, loops or if statements where the condition may depend on the input values of the program.

For example, let’s examine the following program:

def func_to_trace(x):
    if x.sum() > 0:
        return torch.relu(x)
    else:
        return torch.neg(x)

traced = torch.fx.symbolic_trace(func_to_trace)
"""
  <...>
  File "dyn.py", line 6, in func_to_trace
    if x.sum() > 0:
  File "pytorch/torch/fx/proxy.py", line 155, in __bool__
    return self.tracer.to_bool(self)
  File "pytorch/torch/fx/proxy.py", line 85, in to_bool
    raise TraceError('symbolically traced variables cannot be used as inputs to control flow')
torch.fx.proxy.TraceError: symbolically traced variables cannot be used as inputs to control flow
"""
The condition to the if statement relies on the value of x.sum(), which relies on the value of x, a function input. Since x can change (i.e. if you pass a new input tensor to the traced function), this is dynamic control flow. The traceback walks back up through your code to show you where this situation happens.

Static Control Flow
On the other hand, so-called static control flow is supported. Static control flow is loops or if statements whose value cannot change across invocations. Typically, in PyTorch programs, this control flow arises for code making decisions about a model’s architecture based on hyper-parameters. As a concrete example:

import torch
import torch.fx

class MyModule(torch.nn.Module):
    def __init__(self, do_activation : bool = False):
        super().__init__()
        self.do_activation = do_activation
        self.linear = torch.nn.Linear(512, 512)

    def forward(self, x):
        x = self.linear(x)
        # This if-statement is so-called static control flow.
        # Its condition does not depend on any input values
        if self.do_activation:
            x = torch.relu(x)
        return x

without_activation = MyModule(do_activation=False)
with_activation = MyModule(do_activation=True)

traced_without_activation = torch.fx.symbolic_trace(without_activation)
print(traced_without_activation.code)
"""
def forward(self, x):
    linear_1 = self.linear(x);  x = None
    return linear_1
"""

traced_with_activation = torch.fx.symbolic_trace(with_activation)
print(traced_with_activation.code)
"""
import torch
def forward(self, x):
    linear_1 = self.linear(x);  x = None
    relu_1 = torch.relu(linear_1);  linear_1 = None
    return relu_1
"""
The if-statement if self.do_activation does not depend on any function inputs, thus it is static. do_activation can be considered to be a hyper-parameter, and the traces of different instances of MyModule with different values for that parameter have different code. This is a valid pattern that is supported by symbolic tracing.

Many instances of dynamic control flow are semantically static control flow. These instances can be made to support symbolic tracing by removing the data dependencies on input values, for example by moving values to Module attributes or by binding concrete values to arguments during symbolic tracing:

def f(x, flag):
    if flag: return x
    else: return x*2

fx.symbolic_trace(f) # Fails!

fx.symbolic_trace(f, concrete_args={'flag': True})
In the case of truly dynamic control flow, the sections of the program that contain this code can be traced as calls to the Method (see Customizing Tracing with the Tracer class) or function (see wrap()) rather than tracing through them.

Non-torch Functions
FX uses __torch_function__ as the mechanism by which it intercepts calls (see the technical overview for more information about this). Some functions, such as builtin Python functions or those in the math module, are not covered by __torch_function__, but we would still like to capture them in symbolic tracing. For example:

import torch
import torch.fx
from math import sqrt

def normalize(x):
    """
    Normalize `x` by the size of the batch dimension
    """
    return x / sqrt(len(x))

# It's valid Python code
normalize(torch.rand(3, 4))

traced = torch.fx.symbolic_trace(normalize)
"""
  <...>
  File "sqrt.py", line 9, in normalize
    return x / sqrt(len(x))
  File "pytorch/torch/fx/proxy.py", line 161, in __len__
    raise RuntimeError("'len' is not supported in symbolic tracing by default. If you want "
RuntimeError: 'len' is not supported in symbolic tracing by default. If you want this call to be recorded, please call torch.fx.wrap('len') at module scope
"""
The error tells us that the built-in function len is not supported. We can make it so that functions like this are recorded in the trace as direct calls using the wrap() API:

torch.fx.wrap('len')
torch.fx.wrap('sqrt')

traced = torch.fx.symbolic_trace(normalize)

print(traced.code)
"""
import math
def forward(self, x):
    len_1 = len(x)
    sqrt_1 = math.sqrt(len_1);  len_1 = None
    truediv = x / sqrt_1;  x = sqrt_1 = None
    return truediv
"""
Customizing Tracing with the Tracer class
The Tracer class is the class that underlies the implementation of symbolic_trace. The behavior of tracing can be customized by subclassing Tracer, like so:

class MyCustomTracer(torch.fx.Tracer):
    # Inside here you can override various methods
    # to customize tracing. See the `Tracer` API
    # reference
    pass


# Let's use this custom tracer to trace through this module
class MyModule(torch.nn.Module):
    def forward(self, x):
        return torch.relu(x) + torch.ones(3, 4)

mod = MyModule()

traced_graph = MyCustomTracer().trace(mod)
# trace() returns a Graph. Let's wrap it up in a
# GraphModule to make it runnable
traced = torch.fx.GraphModule(mod, traced_graph)
Leaf Modules
Leaf Modules are the modules that appear as calls in the symbolic trace rather than being traced through. The default set of leaf modules is the set of standard torch.nn module instances. For example:

class MySpecialSubmodule(torch.nn.Module):
    def forward(self, x):
        return torch.neg(x)

class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 4)
        self.submod = MySpecialSubmodule()

    def forward(self, x):
        return self.submod(self.linear(x))

traced = torch.fx.symbolic_trace(MyModule())
print(traced.code)
# `linear` is preserved as a call, yet `submod` is traced though.
# This is because the default set of "Leaf Modules" includes all
# standard `torch.nn` modules.
"""
import torch
def forward(self, x):
    linear_1 = self.linear(x);  x = None
    neg_1 = torch.neg(linear_1);  linear_1 = None
    return neg_1
"""
The set of leaf modules can be customized by overriding Tracer.is_leaf_module().

Miscellanea
Tensor constructors (e.g. torch.zeros, torch.ones, torch.rand, torch.randn, torch.sparse_coo_tensor) are currently not traceable.

The deterministic constructors (zeros, ones) can be used and the value they produce will be embedded in the trace as a constant. This is only problematic if the arguments to these constructors refers to dynamic input sizes. In this case, ones_like or zeros_like may be a viable substitute.

Nondeterministic constructors (rand, randn) will have a single random value embedded in the trace. This is likely not the intended behavior. One workaround is to wrap torch.randn in a torch.fx.wrap function and call that instead.

@torch.fx.wrap
def torch_randn(x, shape):
    return torch.randn(shape)

def f(x):
    return x + torch_randn(x, 5)
fx.symbolic_trace(f)
This behavior may be fixed in a future release.

Type annotations

Python 3-style type annotations (e.g. func(x : torch.Tensor, y : int) -> torch.Tensor) are supported and will be preserved by symbolic tracing.

Python 2-style comment type annotations # type: (torch.Tensor, int) -> torch.Tensor are not currently supported.

Annotations on local names within a function are not currently supported.

Gotcha around training flag and submodules

When using functionals like torch.nn.functional.dropout, it will be common for the training argument to be passed in as self.training. During FX tracing, this will likely be baked in as a constant value.

import torch
import torch.fx

class DropoutRepro(torch.nn.Module):
  def forward(self, x):
    return torch.nn.functional.dropout(x, training=self.training)


traced = torch.fx.symbolic_trace(DropoutRepro())
print(traced.code)
"""
def forward(self, x):
  dropout = torch.nn.functional.dropout(x, p = 0.5, training = True, inplace = False);  x = None
  return dropout
"""

traced.eval()

x = torch.randn(5, 3)
torch.testing.assert_close(traced(x), x)
"""
AssertionError: Tensor-likes are not close!

Mismatched elements: 15 / 15 (100.0%)
Greatest absolute difference: 1.6207983493804932 at index (0, 2) (up to 1e-05 allowed)
Greatest relative difference: 1.0 at index (0, 0) (up to 0.0001 allowed)
"""
However, when the standard nn.Dropout() submodule is used, the training flag is encapsulated and–because of the preservation of the nn.Module object model–can be changed.

class DropoutRepro2(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.drop = torch.nn.Dropout()

  def forward(self, x):
    return self.drop(x)

traced = torch.fx.symbolic_trace(DropoutRepro2())
print(traced.code)
"""
def forward(self, x):
  drop = self.drop(x);  x = None
  return drop
"""

traced.eval()

x = torch.randn(5, 3)
torch.testing.assert_close(traced(x), x)
Because of this difference, consider marking modules that interact with the training flag dynamically as leaf modules.

API Reference
torch.fx.symbolic_trace(root, concrete_args=None)[source][source]
Symbolic tracing API

Given an nn.Module or function instance root, this function will return a GraphModule constructed by recording operations seen while tracing through root.

concrete_args allows you to partially specialize your function, whether it’s to remove control flow or data structures.

For example:

def f(a, b):
    if b == True:
        return a
    else:
        return a * 2
FX can typically not trace through this due to the presence of control flow. However, we can use concrete_args to specialize on the value of b to trace through this:

f = fx.symbolic_trace(f, concrete_args={"b": False})
assert f(3, False) == 6
Note that although you can still pass in different values of b, they will be ignored.

We can also use concrete_args to eliminate data-structure handling from our function. This will use pytrees to flatten your input. To avoid overspecializing, pass in fx.PH for values that shouldn’t be specialized. For example:

def f(x):
    out = 0
    for v in x.values():
        out += v
    return out


f = fx.symbolic_trace(f, concrete_args={"x": {"a": fx.PH, "b": fx.PH, "c": fx.PH}})
assert f({"a": 1, "b": 2, "c": 4}) == 7
Parameters
root (Union[torch.nn.Module, Callable]) – Module or function to be traced and converted into a Graph representation.

concrete_args (Optional[Dict[str, any]]) – Inputs to be partially specialized

Returns
a Module created from the recorded operations from root.

Return type
GraphModule

Note

Backwards-compatibility for this API is guaranteed.

torch.fx.wrap(fn_or_name)[source][source]
This function can be called at module-level scope to register fn_or_name as a “leaf function”. A “leaf function” will be preserved as a CallFunction node in the FX trace instead of being traced through:

# foo/bar/baz.py
def my_custom_function(x, y):
    return x * x + y * y


torch.fx.wrap("my_custom_function")


def fn_to_be_traced(x, y):
    # When symbolic tracing, the below call to my_custom_function will be inserted into
    # the graph rather than tracing it.
    return my_custom_function(x, y)
This function can also equivalently be used as a decorator:

# foo/bar/baz.py
@torch.fx.wrap
def my_custom_function(x, y):
    return x * x + y * y
A wrapped function can be thought of a “leaf function”, analogous to the concept of “leaf modules”, that is, they are functions that are left as calls in the FX trace rather than traced through.

Parameters
fn_or_name (Union[str, Callable]) – The function or name of the global function to insert into the graph when it’s called

Note

Backwards-compatibility for this API is guaranteed.

classtorch.fx.GraphModule(*args, **kwargs)[source][source]
GraphModule is an nn.Module generated from an fx.Graph. Graphmodule has a graph attribute, as well as code and forward attributes generated from that graph.

Warning

When graph is reassigned, code and forward will be automatically regenerated. However, if you edit the contents of the graph without reassigning the graph attribute itself, you must call recompile() to update the generated code.

Note

Backwards-compatibility for this API is guaranteed.

__init__(root, graph, class_name='GraphModule')[source][source]
Construct a GraphModule.

Parameters
root (Union[torch.nn.Module, Dict[str, Any]) – root can either be an nn.Module instance or a Dict mapping strings to any attribute type. In the case that root is a Module, any references to Module-based objects (via qualified name) in the Graph’s Nodes’ target field will be copied over from the respective place within root’s Module hierarchy into the GraphModule’s module hierarchy. In the case that root is a dict, the qualified name found in a Node’s target will be looked up directly in the dict’s keys. The object mapped to by the Dict will be copied over into the appropriate place within the GraphModule’s module hierarchy.

graph (Graph) – graph contains the nodes this GraphModule should use for code generation

class_name (str) – name denotes the name of this GraphModule for debugging purposes. If it’s unset, all error messages will report as originating from GraphModule. It may be helpful to set this to root’s original name or a name that makes sense within the context of your transform.

Note

Backwards-compatibility for this API is guaranteed.

add_submodule(target, m)[source][source]
Adds the given submodule to self.

This installs empty Modules where none exist yet if they are subpaths of target.

Parameters
target (str) – The fully-qualified string name of the new submodule (See example in nn.Module.get_submodule for how to specify a fully-qualified string.)

m (Module) – The submodule itself; the actual object we want to install in the current Module

Returns
Whether or not the submodule could be inserted. For
this method to return True, each object in the chain denoted by target must either a) not exist yet, or b) reference an nn.Module (not a parameter or other attribute)

Return type
bool

Note

Backwards-compatibility for this API is guaranteed.

property code: str
Return the Python code generated from the Graph underlying this GraphModule.

delete_all_unused_submodules()[source][source]
Deletes all unused submodules from self.

A Module is considered “used” if any one of the following is true: 1. It has children that are used 2. Its forward is called directly via a call_module node 3. It has a non-Module attribute that is used from a get_attr node

This method can be called to clean up an nn.Module without manually calling delete_submodule on each unused submodule.

Note

Backwards-compatibility for this API is guaranteed.

delete_submodule(target)[source][source]
Deletes the given submodule from self.

The module will not be deleted if target is not a valid target.

Parameters
target (str) – The fully-qualified string name of the new submodule (See example in nn.Module.get_submodule for how to specify a fully-qualified string.)

Returns
Whether or not the target string referenced a
submodule we want to delete. A return value of False means that the target was not a valid reference to a submodule.

Return type
bool

Note

Backwards-compatibility for this API is guaranteed.

property graph: Graph
Return the Graph underlying this GraphModule

print_readable(print_output=True, include_stride=False, include_device=False, colored=False)[source][source]
Return the Python code generated for current GraphModule and its children GraphModules

Warning

This API is experimental and is NOT backward-compatible.

recompile()[source][source]
Recompile this GraphModule from its graph attribute. This should be called after editing the contained graph, otherwise the generated code of this GraphModule will be out of date.

Note

Backwards-compatibility for this API is guaranteed.

Return type
PythonCode

to_folder(folder, module_name='FxModule')[source][source]
Dumps out module to folder with module_name so that it can be
imported with from <folder> import <module_name>

Args:

folder (Union[str, os.PathLike]): The folder to write the code out to

module_name (str): Top-level name to use for the Module while
writing out the code

Warning

This API is experimental and is NOT backward-compatible.

classtorch.fx.Graph(owning_module=None, tracer_cls=None, tracer_extras=None)[source][source]
Graph is the main data structure used in the FX Intermediate Representation. It consists of a series of Node s, each representing callsites (or other syntactic constructs). The list of Node s, taken together, constitute a valid Python function.

For example, the following code

import torch
import torch.fx


class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.param = torch.nn.Parameter(torch.rand(3, 4))
        self.linear = torch.nn.Linear(4, 5)

    def forward(self, x):
        return torch.topk(
            torch.sum(self.linear(x + self.linear.weight).relu(), dim=-1), 3
        )


m = MyModule()
gm = torch.fx.symbolic_trace(m)
Will produce the following Graph:

print(gm.graph)
graph(x):
    %linear_weight : [num_users=1] = self.linear.weight
    %add_1 : [num_users=1] = call_function[target=operator.add](args = (%x, %linear_weight), kwargs = {})
    %linear_1 : [num_users=1] = call_module[target=linear](args = (%add_1,), kwargs = {})
    %relu_1 : [num_users=1] = call_method[target=relu](args = (%linear_1,), kwargs = {})
    %sum_1 : [num_users=1] = call_function[target=torch.sum](args = (%relu_1,), kwargs = {dim: -1})
    %topk_1 : [num_users=1] = call_function[target=torch.topk](args = (%sum_1, 3), kwargs = {})
    return topk_1
For the semantics of operations represented in the Graph, please see Node.

Note

Backwards-compatibility for this API is guaranteed.

__init__(owning_module=None, tracer_cls=None, tracer_extras=None)[source][source]
Construct an empty Graph.

Note

Backwards-compatibility for this API is guaranteed.

call_function(the_function, args=None, kwargs=None, type_expr=None)[source][source]
Insert a call_function Node into the Graph. A call_function node represents a call to a Python callable, specified by the_function.

Parameters
the_function (Callable[..., Any]) – The function to be called. Can be any PyTorch operator, Python function, or member of the builtins or operator namespaces.

args (Optional[Tuple[Argument, ...]]) – The positional arguments to be passed to the called function.

kwargs (Optional[Dict[str, Argument]]) – The keyword arguments to be passed to the called function

type_expr (Optional[Any]) – an optional type annotation representing the Python type the output of this node will have.

Returns
The newly created and inserted call_function node.

Return type
Node

Note

The same insertion point and type expression rules apply for this method as Graph.create_node().

Note

Backwards-compatibility for this API is guaranteed.

call_method(method_name, args=None, kwargs=None, type_expr=None)[source][source]
Insert a call_method Node into the Graph. A call_method node represents a call to a given method on the 0th element of args.

Parameters
method_name (str) – The name of the method to apply to the self argument. For example, if args[0] is a Node representing a Tensor, then to call relu() on that Tensor, pass relu to method_name.

args (Optional[Tuple[Argument, ...]]) – The positional arguments to be passed to the called method. Note that this should include a self argument.

kwargs (Optional[Dict[str, Argument]]) – The keyword arguments to be passed to the called method

type_expr (Optional[Any]) – an optional type annotation representing the Python type the output of this node will have.

Returns
The newly created and inserted call_method node.

Return type
Node

Note

The same insertion point and type expression rules apply for this method as Graph.create_node().

Note

Backwards-compatibility for this API is guaranteed.

call_module(module_name, args=None, kwargs=None, type_expr=None)[source][source]
Insert a call_module Node into the Graph. A call_module node represents a call to the forward() function of a Module in the Module hierarchy.

Parameters
module_name (str) – The qualified name of the Module in the Module hierarchy to be called. For example, if the traced Module has a submodule named foo, which has a submodule named bar, the qualified name foo.bar should be passed as module_name to call that module.

args (Optional[Tuple[Argument, ...]]) – The positional arguments to be passed to the called method. Note that this should not include a self argument.

kwargs (Optional[Dict[str, Argument]]) – The keyword arguments to be passed to the called method

type_expr (Optional[Any]) – an optional type annotation representing the Python type the output of this node will have.

Returns
The newly-created and inserted call_module node.

Return type
Node

Note

The same insertion point and type expression rules apply for this method as Graph.create_node().

Note

Backwards-compatibility for this API is guaranteed.

create_node(op, target, args=None, kwargs=None, name=None, type_expr=None)[source][source]
Create a Node and add it to the Graph at the current insert-point. Note that the current insert-point can be set via Graph.inserting_before() and Graph.inserting_after().

Parameters
op (str) – the opcode for this Node. One of ‘call_function’, ‘call_method’, ‘get_attr’, ‘call_module’, ‘placeholder’, or ‘output’. The semantics of these opcodes are described in the Graph docstring.

args (Optional[Tuple[Argument, ...]]) – is a tuple of arguments to this node.

kwargs (Optional[Dict[str, Argument]]) – the kwargs of this Node

name (Optional[str]) – an optional string name for the Node. This will influence the name of the value assigned to in the Python generated code.

type_expr (Optional[Any]) – an optional type annotation representing the Python type the output of this node will have.

Returns
The newly-created and inserted node.

Return type
Node

Note

Backwards-compatibility for this API is guaranteed.

eliminate_dead_code(is_impure_node=None)[source][source]
Remove all dead code from the graph, based on each node’s number of users, and whether the nodes have any side effects. The graph must be topologically sorted before calling.

Parameters
is_impure_node (Optional[Callable[[Node], bool]]) – A function that returns

None (whether a node is impure. If this is) –

to (then the default behavior is) –

Node.is_impure. (use) –

Returns
Whether the graph was changed as a result of the pass.

Return type
bool

Example:

Before dead code is eliminated, a from a = x + 1 below has no users and thus can be eliminated from the graph without having an effect.

def forward(self, x):
    a = x + 1
    return x + self.attr_1
After dead code is eliminated, a = x + 1 has been removed, and the rest of forward remains.

def forward(self, x):
    return x + self.attr_1
Warning

Dead code elimination has some heuristics to avoid removing side-effectful nodes (see Node.is_impure) but in general coverage is very bad, so you should assume that this method is not sound to call unless you know that your FX graph consists entirely of functional operations or you supply your own custom function for detecting side-effectful nodes.

Note

Backwards-compatibility for this API is guaranteed.

erase_node(to_erase)[source][source]
Erases a Node from the Graph. Throws an exception if there are still users of that node in the Graph.

Parameters
to_erase (Node) – The Node to erase from the Graph.

Note

Backwards-compatibility for this API is guaranteed.

find_nodes(*, op, target=None, sort=True)[source][source]
Allows for fast query of nodes

Parameters
op (str) – the name of the operation

target (Optional[Target]) – the target of the node. For call_function, the target is required. For other ops, the target is optional.

sort (bool) – whether to return nodes in the order they appear on on the graph.

Returns
Iteratable of nodes with the requested op and target.

Warning

This API is experimental and is NOT backward-compatible.

get_attr(qualified_name, type_expr=None)[source][source]
Insert a get_attr node into the Graph. A get_attr Node represents the fetch of an attribute from the Module hierarchy.

Parameters
qualified_name (str) – the fully-qualified name of the attribute to be retrieved. For example, if the traced Module has a submodule named foo, which has a submodule named bar, which has an attribute named baz, the qualified name foo.bar.baz should be passed as qualified_name.

type_expr (Optional[Any]) – an optional type annotation representing the Python type the output of this node will have.

Returns
The newly-created and inserted get_attr node.

Return type
Node

Note

The same insertion point and type expression rules apply for this method as Graph.create_node.

Note

Backwards-compatibility for this API is guaranteed.

graph_copy(g, val_map, return_output_node=False)[source][source]
Copy all nodes from a given graph into self.

Parameters
g (Graph) – The source graph from which to copy Nodes.

val_map (Dict[Node, Node]) – a dictionary that will be populated with a mapping from nodes in g to nodes in self. Note that val_map can be passed in with values in it already to override copying of certain values.

Returns
The value in self that is now equivalent to the output value in g, if g had an output node. None otherwise.

Return type
Optional[Union[tuple[‘Argument’, …], Sequence[Argument], Mapping[str, Argument], slice, range, Node, str, int, float, bool, complex, dtype, Tensor, device, memory_format, layout, OpOverload, SymInt, SymBool, SymFloat]]

Note

Backwards-compatibility for this API is guaranteed.

inserting_after(n=None)[source][source]
Set the point at which create_node and companion methods will insert into the graph.
When used within a ‘with’ statement, this will temporary set the insert point and then restore it when the with statement exits:

with g.inserting_after(n):
    ...  # inserting after node n
...  # insert point restored to what it was previously
g.inserting_after(n)  #  set the insert point permanently
Args:

n (Optional[Node]): The node before which to insert. If None this will insert after
the beginning of the entire graph.

Returns:
A resource manager that will restore the insert point on __exit__.

Note

Backwards-compatibility for this API is guaranteed.

inserting_before(n=None)[source][source]
Set the point at which create_node and companion methods will insert into the graph.
When used within a ‘with’ statement, this will temporary set the insert point and then restore it when the with statement exits:

with g.inserting_before(n):
    ...  # inserting before node n
...  # insert point restored to what it was previously
g.inserting_before(n)  #  set the insert point permanently
Args:

n (Optional[Node]): The node before which to insert. If None this will insert before
the beginning of the entire graph.

Returns:
A resource manager that will restore the insert point on __exit__.

Note

Backwards-compatibility for this API is guaranteed.

lint()[source][source]
Runs various checks on this Graph to make sure it is well-formed. In particular: - Checks Nodes have correct ownership (owned by this graph) - Checks Nodes appear in topological order - If this Graph has an owning GraphModule, checks that targets exist in that GraphModule

Note

Backwards-compatibility for this API is guaranteed.

node_copy(node, arg_transform=<function Graph.<lambda>>)[source][source]
Copy a node from one graph into another. arg_transform needs to transform arguments from the graph of node to the graph of self. Example:

# Copying all the nodes in `g` into `new_graph`
g: torch.fx.Graph = ...
new_graph = torch.fx.graph()
value_remap = {}
for node in g.nodes:
    value_remap[node] = new_graph.node_copy(node, lambda n: value_remap[n])
Parameters
node (Node) – The node to copy into self.

arg_transform (Callable[[Node], Argument]) – A function that transforms Node arguments in node’s args and kwargs into the equivalent argument in self. In the simplest case, this should retrieve a value out of a table mapping Nodes in the original graph to self.

Return type
Node

Note

Backwards-compatibility for this API is guaranteed.

property nodes: _node_list
Get the list of Nodes that constitute this Graph.

Note that this Node list representation is a doubly-linked list. Mutations during iteration (e.g. delete a Node, add a Node) are safe.

Returns
A doubly-linked list of Nodes. Note that reversed can be called on this list to switch iteration order.

on_generate_code(make_transformer)[source][source]
Register a transformer function when python code is generated

Args:
make_transformer (Callable[[Optional[TransformCodeFunc]], TransformCodeFunc]):
a function that returns a code transformer to be registered. This function is called by on_generate_code to obtain the code transformer.

This function is also given as its input the currently registered code transformer (or None if nothing is registered), in case it is not desirable to overwrite it. This is useful to chain code transformers together.

Returns:
a context manager that when used in a with statement, to automatically restore the previously registered code transformer.

Example:

gm: fx.GraphModule = ...


# This is a code transformer we want to register. This code
# transformer prepends a pdb import and trace statement at the very
# beginning of the generated torch.fx code to allow for manual
# debugging with the PDB library.
def insert_pdb(body):
    return ["import pdb; pdb.set_trace()\n", *body]


# Registers `insert_pdb`, and overwrites the current registered
# code transformer (given by `_` to the lambda):
gm.graph.on_generate_code(lambda _: insert_pdb)

# Or alternatively, registers a code transformer which first
# runs `body` through existing registered transformer, then
# through `insert_pdb`:
gm.graph.on_generate_code(
    lambda current_trans: (
        lambda body: insert_pdb(current_trans(body) if current_trans else body)
    )
)

gm.recompile()
gm(*inputs)  # drops into pdb
This function can also be used as a context manager, with the benefit to automatically restores the previously registered code transformer:

# ... continue from previous example

with gm.graph.on_generate_code(lambda _: insert_pdb):
    # do more stuff with `gm`...
    gm.recompile()
    gm(*inputs)  # drops into pdb

# now previous code transformer is restored (but `gm`'s code with pdb
# remains - that means you can run `gm` with pdb here too, until you
# run next `recompile()`).
Warning

This API is experimental and is NOT backward-compatible.

output(result, type_expr=None)[source][source]
Insert an output Node into the Graph. An output node represents a return statement in Python code. result is the value that should be returned.

Parameters
result (Argument) – The value to be returned.

type_expr (Optional[Any]) – an optional type annotation representing the Python type the output of this node will have.

Note

The same insertion point and type expression rules apply for this method as Graph.create_node.

Note

Backwards-compatibility for this API is guaranteed.

output_node()[source][source]
Warning

This API is experimental and is NOT backward-compatible.

Return type
Node

placeholder(name, type_expr=None, default_value)[source][source]
Insert a placeholder node into the Graph. A placeholder represents a function input.

Parameters
name (str) – A name for the input value. This corresponds to the name of the positional argument to the function this Graph represents.

type_expr (Optional[Any]) – an optional type annotation representing the Python type the output of this node will have. This is needed in some cases for proper code generation (e.g. when the function is used subsequently in TorchScript compilation).

default_value (Any) – The default value this function argument should take on. NOTE: to allow for None as a default value, inspect.Signature.empty should be passed as this argument to specify that the parameter does _not_ have a default value.

Return type
Node

Note

The same insertion point and type expression rules apply for this method as Graph.create_node.

Note

Backwards-compatibility for this API is guaranteed.

print_tabular()[source][source]
Prints the intermediate representation of the graph in tabular format. Note that this API requires the tabulate module to be installed.

Note

Backwards-compatibility for this API is guaranteed.

process_inputs(*args)[source][source]
Processes args so that they can be passed to the FX graph.

Warning

This API is experimental and is NOT backward-compatible.

process_outputs(out)[source][source]
Warning

This API is experimental and is NOT backward-compatible.

python_code(root_module, *, verbose=False, include_stride=False, include_device=False, colored=False)[source][source]
Turn this Graph into valid Python code.

Parameters
root_module (str) – The name of the root module on which to look-up qualified name targets. This is usually ‘self’.

Returns
src: the Python source code representing the object globals: a dictionary of global names in src -> the objects that they reference.

Return type
A PythonCode object, consisting of two fields

Note

Backwards-compatibility for this API is guaranteed.

set_codegen(codegen)[source][source]
Warning

This API is experimental and is NOT backward-compatible.

classtorch.fx.Node(graph, name, op, target, args, kwargs, return_type=None)[source][source]
Node is the data structure that represents individual operations within a Graph. For the most part, Nodes represent callsites to various entities, such as operators, methods, and Modules (some exceptions include nodes that specify function inputs and outputs). Each Node has a function specified by its op property. The Node semantics for each value of op are as follows:

placeholder represents a function input. The name attribute specifies the name this value will take on. target is similarly the name of the argument. args holds either: 1) nothing, or 2) a single argument denoting the default parameter of the function input. kwargs is don’t-care. Placeholders correspond to the function parameters (e.g. x) in the graph printout.

get_attr retrieves a parameter from the module hierarchy. name is similarly the name the result of the fetch is assigned to. target is the fully-qualified name of the parameter’s position in the module hierarchy. args and kwargs are don’t-care

call_function applies a free function to some values. name is similarly the name of the value to assign to. target is the function to be applied. args and kwargs represent the arguments to the function, following the Python calling convention

call_module applies a module in the module hierarchy’s forward() method to given arguments. name is as previous. target is the fully-qualified name of the module in the module hierarchy to call. args and kwargs represent the arguments to invoke the module on, excluding the self argument.

call_method calls a method on a value. name is as similar. target is the string name of the method to apply to the self argument. args and kwargs represent the arguments to invoke the module on, including the self argument

output contains the output of the traced function in its args[0] attribute. This corresponds to the “return” statement in the Graph printout.

Note

Backwards-compatibility for this API is guaranteed.

property all_input_nodes: list['Node']
Return all Nodes that are inputs to this Node. This is equivalent to iterating over args and kwargs and only collecting the values that are Nodes.

Returns
List of Nodes that appear in the args and kwargs of this Node, in that order.

append(x)[source][source]
Insert x after this node in the list of nodes in the graph. Equivalent to self.next.prepend(x)

Parameters
x (Node) – The node to put after this node. Must be a member of the same graph.

Note

Backwards-compatibility for this API is guaranteed.

property args: tuple[Union[tuple['Argument', ...], collections.abc.Sequence['Argument'], collections.abc.Mapping[str, 'Argument'], slice, range, torch.fx.node.Node, str, int, float, bool, complex, torch.dtype, torch.Tensor, torch.device, torch.memory_format, torch.layout, torch._ops.OpOverload, torch.SymInt, torch.SymBool, torch.SymFloat, NoneType], ...]
The tuple of arguments to this Node. The interpretation of arguments depends on the node’s opcode. See the Node docstring for more information.

Assignment to this property is allowed. All accounting of uses and users is updated automatically on assignment.

format_node(placeholder_names=None, maybe_return_typename=None)[source][source]
Return a descriptive string representation of self.

This method can be used with no arguments as a debugging utility.

This function is also used internally in the __str__ method of Graph. Together, the strings in placeholder_names and maybe_return_typename make up the signature of the autogenerated forward function in this Graph’s surrounding GraphModule. placeholder_names and maybe_return_typename should not be used otherwise.

Parameters
placeholder_names (Optional[list[str]]) – A list that will store formatted strings representing the placeholders in the generated forward function. Internal use only.

maybe_return_typename (Optional[list[str]]) – A single-element list that will store a formatted string representing the output of the generated forward function. Internal use only.

Returns
If 1) we’re using format_node as an internal helper
in the __str__ method of Graph, and 2) self is a placeholder Node, return None. Otherwise, return a descriptive string representation of the current Node.

Return type
str

Note

Backwards-compatibility for this API is guaranteed.

insert_arg(idx, arg)[source][source]
Insert an positional argument to the argument list with given index.

Parameters
idx (int) – The index of the element in self.args to be inserted before.

arg (Argument) – The new argument value to insert into args

Note

Backwards-compatibility for this API is guaranteed.

is_impure()[source][source]
Returns whether this op is impure, i.e. if its op is a placeholder or output, or if a call_function or call_module which is impure.

Returns
If the op is impure or not.

Return type
bool

Warning

This API is experimental and is NOT backward-compatible.

property kwargs: dict[str, Union[tuple['Argument', ...], collections.abc.Sequence['Argument'], collections.abc.Mapping[str, 'Argument'], slice, range, torch.fx.node.Node, str, int, float, bool, complex, torch.dtype, torch.Tensor, torch.device, torch.memory_format, torch.layout, torch._ops.OpOverload, torch.SymInt, torch.SymBool, torch.SymFloat, NoneType]]
The dict of keyword arguments to this Node. The interpretation of arguments depends on the node’s opcode. See the Node docstring for more information.

Assignment to this property is allowed. All accounting of uses and users is updated automatically on assignment.

property next: Node
Returns the next Node in the linked list of Nodes.

Returns
The next Node in the linked list of Nodes.

normalized_arguments(root, arg_types=None, kwarg_types=None, normalize_to_only_use_kwargs=False)[source][source]
Returns normalized arguments to Python targets. This means that args/kwargs will be matched up to the module/functional’s signature and return exclusively kwargs in positional order if normalize_to_only_use_kwargs is true. Also populates default values. Does not support positional-only parameters or varargs parameters.

Supports module calls.

May require arg_types and kwarg_types in order to disambiguate overloads.

Parameters
root (torch.nn.Module) – Module upon which to resolve module targets.

arg_types (Optional[Tuple[Any]]) – Tuple of arg types for the args

kwarg_types (Optional[Dict[str, Any]]) – Dict of arg types for the kwargs

normalize_to_only_use_kwargs (bool) – Whether to normalize to only use kwargs.

Returns
Returns NamedTuple ArgsKwargsPair, or None if not successful.

Return type
Optional[ArgsKwargsPair]

Warning

This API is experimental and is NOT backward-compatible.

prepend(x)[source][source]
Insert x before this node in the list of nodes in the graph. Example:

Before: p -> self
        bx -> x -> ax
After:  p -> x -> self
        bx -> ax
Parameters
x (Node) – The node to put before this node. Must be a member of the same graph.

Note

Backwards-compatibility for this API is guaranteed.

property prev: Node
Returns the previous Node in the linked list of Nodes.

Returns
The previous Node in the linked list of Nodes.

replace_all_uses_with(replace_with, delete_user_cb=<function Node.<lambda>>, *, propagate_meta=False)[source][source]
Replace all uses of self in the Graph with the Node replace_with.

Parameters
replace_with (Node) – The node to replace all uses of self with.

delete_user_cb (Callable) – Callback that is called to determine whether a given user of the self node should be removed.

propagate_meta (bool) – Whether or not to copy all properties on the .meta field of the original node onto the replacement node. For safety, this is only valid to do if the replacement node doesn’t already have an existing .meta field.

Returns
The list of Nodes on which this change was made.

Return type
list[‘Node’]

Note

Backwards-compatibility for this API is guaranteed.

replace_input_with(old_input, new_input)[source][source]
Loop through input nodes of self, and replace all instances of old_input with new_input.

Parameters
old_input (Node) – The old input node to be replaced.

new_input (Node) – The new input node to replace old_input.

Note

Backwards-compatibility for this API is guaranteed.

property stack_trace: Optional[str]
Return the Python stack trace that was recorded during tracing, if any. When traced with fx.Tracer, this property is usually populated by Tracer.create_proxy. To record stack traces during tracing for debug purposes, set record_stack_traces = True on the Tracer instance. When traced with dynamo, this property will be populated by default by OutputGraph.create_proxy.

stack_trace would have the innermost frame at the end of the string.

update_arg(idx, arg)[source][source]
Update an existing positional argument to contain the new value arg. After calling, self.args[idx] == arg.

Parameters
idx (int) – The index into self.args of the element to update

arg (Argument) – The new argument value to write into args

Note

Backwards-compatibility for this API is guaranteed.

update_kwarg(key, arg)[source][source]
Update an existing keyword argument to contain the new value arg. After calling, self.kwargs[key] == arg.

Parameters
key (str) – The key in self.kwargs of the element to update

arg (Argument) – The new argument value to write into kwargs

Note

Backwards-compatibility for this API is guaranteed.

classtorch.fx.Tracer(autowrap_modules=(math,), autowrap_functions=())[source][source]
Tracer is the class that implements the symbolic tracing functionality of torch.fx.symbolic_trace. A call to symbolic_trace(m) is equivalent to Tracer().trace(m).

Tracer can be subclassed to override various behaviors of the tracing process. The different behaviors that can be overridden are described in the docstrings of the methods on this class.

Note

Backwards-compatibility for this API is guaranteed.

call_module(m, forward, args, kwargs)[source][source]
Method that specifies the behavior of this Tracer when it encounters a call to an nn.Module instance.

By default, the behavior is to check if the called module is a leaf module via is_leaf_module. If it is, emit a call_module node referring to m in the Graph. Otherwise, call the Module normally, tracing through the operations in its forward function.

This method can be overridden to–for example–create nested traced GraphModules, or any other behavior you would want while tracing across Module boundaries.

Parameters
m (Module) – The module for which a call is being emitted

forward (Callable) – The forward() method of the Module to be invoked

args (Tuple) – args of the module callsite

kwargs (Dict) – kwargs of the module callsite

Returns
The return value from the Module call. In the case that a call_module node was emitted, this is a Proxy value. Otherwise, it is whatever value was returned from the Module invocation.

Return type
Any

Note

Backwards-compatibility for this API is guaranteed.

create_arg(a)[source][source]
A method to specify the behavior of tracing when preparing values to be used as arguments to nodes in the Graph.

By default, the behavior includes:

Iterate through collection types (e.g. tuple, list, dict) and recursively call create_args on the elements.

Given a Proxy object, return a reference to the underlying IR Node

Given a non-Proxy Tensor object, emit IR for various cases:

For a Parameter, emit a get_attr node referring to that Parameter

For a non-Parameter Tensor, store the Tensor away in a special attribute referring to that attribute.

This method can be overridden to support more types.

Parameters
a (Any) – The value to be emitted as an Argument in the Graph.

Returns
The value a converted into the appropriate Argument

Return type
Argument

Note

Backwards-compatibility for this API is guaranteed.

create_args_for_root(root_fn, is_module, concrete_args=None)[source][source]
Create placeholder nodes corresponding to the signature of the root Module. This method introspects root’s signature and emits those nodes accordingly, also supporting *args and **kwargs.

Warning

This API is experimental and is NOT backward-compatible.

create_node(kind, target, args, kwargs, name=None, type_expr=None)[source]
Inserts a graph node given target, args, kwargs, and name.

This method can be overridden to do extra checking, validation, or modification of values used in node creation. For example, one might want to disallow in-place operations from being recorded.

Note

Backwards-compatibility for this API is guaranteed.

Return type
Node

create_proxy(kind, target, args, kwargs, name=None, type_expr=None, proxy_factory_fn=None)[source]
Create a Node from the given arguments, then return the Node wrapped in a Proxy object.

If kind = ‘placeholder’, then we’re creating a Node that represents the parameter of a function. If we need to encode a default parameter, we use the args tuple. args is otherwise empty for placeholder Nodes.

Note

Backwards-compatibility for this API is guaranteed.

get_fresh_qualname(prefix)[source][source]
Gets a fresh name for a prefix and returns it. This function ensures that it will not clash with an existing attribute on the graph.

Note

Backwards-compatibility for this API is guaranteed.

Return type
str

getattr(attr, attr_val, parameter_proxy_cache)[source][source]
Method that specifies the behavior of this Tracer when we call getattr on a call to an nn.Module instance.

By default, the behavior is to return a proxy value for the attribute. It also stores the proxy value in the parameter_proxy_cache, so that future calls will reuse the proxy rather than creating a new one.

This method can be overridden to –for example– not return proxies when querying parameters.

Parameters
attr (str) – The name of the attribute being queried

attr_val (Any) – The value of the attribute

parameter_proxy_cache (Dict[str, Any]) – A cache of attr names to proxies

Returns
The return value from the getattr call.

Warning

This API is experimental and is NOT backward-compatible.

is_leaf_module(m, module_qualified_name)[source][source]
A method to specify whether a given nn.Module is a “leaf” module.

Leaf modules are the atomic units that appear in the IR, referenced by call_module calls. By default, Modules in the PyTorch standard library namespace (torch.nn) are leaf modules. All other modules are traced through and their constituent ops are recorded, unless specified otherwise via this parameter.

Parameters
m (Module) – The module being queried about

module_qualified_name (str) – The path to root of this module. For example, if you have a module hierarchy where submodule foo contains submodule bar, which contains submodule baz, that module will appear with the qualified name foo.bar.baz here.

Return type
bool

Note

Backwards-compatibility for this API is guaranteed.

iter(obj)[source]
Called when a proxy object is being iterated over, such as
when used in control flow. Normally we don’t know what to do because we don’t know the value of the proxy, but a custom tracer can attach more information to the graph node using create_node and can choose to return an iterator.

Note

Backwards-compatibility for this API is guaranteed.

Return type
Iterator

keys(obj)[source]
Called when a proxy object is has the keys() method called.
This is what happens when ** is called on a proxy. This should return an iterator it ** is suppose to work in your custom tracer.

Note

Backwards-compatibility for this API is guaranteed.

Return type
Any

path_of_module(mod)[source][source]
Helper method to find the qualified name of mod in the Module hierarchy of root. For example, if root has a submodule named foo, which has a submodule named bar, passing bar into this function will return the string “foo.bar”.

Parameters
mod (str) – The Module to retrieve the qualified name for.

Return type
str

Note

Backwards-compatibility for this API is guaranteed.

proxy(node)[source]
Note

Backwards-compatibility for this API is guaranteed.

Return type
Proxy

to_bool(obj)[source]
Called when a proxy object is being converted to a boolean, such as
when used in control flow. Normally we don’t know what to do because we don’t know the value of the proxy, but a custom tracer can attach more information to the graph node using create_node and can choose to return a value.

Note

Backwards-compatibility for this API is guaranteed.

Return type
bool

trace(root, concrete_args=None)[source][source]
Trace root and return the corresponding FX Graph representation. root can either be an nn.Module instance or a Python callable.

Note that after this call, self.root may be different from the root passed in here. For example, when a free function is passed to trace(), we will create an nn.Module instance to use as the root and add embedded constants to.

Parameters
root (Union[Module, Callable]) – Either a Module or a function to be traced through. Backwards-compatibility for this parameter is guaranteed.

concrete_args (Optional[Dict[str, any]]) – Concrete arguments that should not be treated as Proxies. This parameter is experimental and its backwards-compatibility is NOT guaranteed.

Returns
A Graph representing the semantics of the passed-in root.

Return type
Graph

Note

Backwards-compatibility for this API is guaranteed.

classtorch.fx.Proxy(node, tracer=None)[source][source]
Proxy objects are Node wrappers that flow through the program during symbolic tracing and record all the operations (torch function calls, method calls, operators) that they touch into the growing FX Graph.

If you’re doing graph transforms, you can wrap your own Proxy method around a raw Node so that you can use the overloaded operators to add additional things to a Graph.

Proxy objects cannot be iterated. In other words, the symbolic tracer will throw an error if a Proxy is used in a loop or as an *args/**kwargs function argument.

There are two main ways around this: 1. Factor out the untraceable logic into a top-level function and use fx.wrap on it. 2. If the control flow is static (i.e. the loop trip count is based on some hyperparameter), the code can be kept in its original position and refactored into something like:

for i in range(self.some_hyperparameter):
    indexed_item = proxied_value[i]
For a more detailed description into the Proxy internals, check out the “Proxy” section in torch/fx/README.md

Note

Backwards-compatibility for this API is guaranteed.

classtorch.fx.Interpreter(module, garbage_collect_values=True, graph=None)[source][source]
An Interpreter executes an FX graph Node-by-Node. This pattern can be useful for many things, including writing code transformations as well as analysis passes.

Methods in the Interpreter class can be overridden to customize the behavior of execution. The map of overrideable methods in terms of call hierarchy:

run()
    +-- run_node
        +-- placeholder()
        +-- get_attr()
        +-- call_function()
        +-- call_method()
        +-- call_module()
        +-- output()
Example

Suppose we want to swap all instances of torch.neg with torch.sigmoid and vice versa (including their Tensor method equivalents). We could subclass Interpreter like so:

class NegSigmSwapInterpreter(Interpreter):
    def call_function(self, target: Target, args: Tuple, kwargs: Dict) -> Any:
        if target == torch.sigmoid:
            return torch.neg(*args, **kwargs)
        return super().call_function(target, args, kwargs)

    def call_method(self, target: Target, args: Tuple, kwargs: Dict) -> Any:
        if target == "neg":
            call_self, *args_tail = args
            return call_self.sigmoid(*args_tail, **kwargs)
        return super().call_method(target, args, kwargs)


def fn(x):
    return torch.sigmoid(x).neg()


gm = torch.fx.symbolic_trace(fn)
input = torch.randn(3, 4)
result = NegSigmSwapInterpreter(gm).run(input)
torch.testing.assert_close(result, torch.neg(input).sigmoid())
Parameters
module (torch.nn.Module) – The module to be executed

garbage_collect_values (bool) – Whether to delete values after their last use within the Module’s execution. This ensures optimal memory usage during execution. This can be disabled to, for example, examine all of the intermediate values in the execution by looking at the Interpreter.env attribute.

graph (Optional[Graph]) – If passed, the interpreter will execute this graph instead of module.graph, using the provided module argument to satisfy any requests for state.

Note

Backwards-compatibility for this API is guaranteed.

boxed_run(args_list)[source][source]
Run module via interpretation and return the result. This uses the “boxed” calling convention, where you pass a list of arguments, which will be cleared by the interpreter. This ensures that input tensors are promptly deallocated.

Note

Backwards-compatibility for this API is guaranteed.

call_function(target, args, kwargs)[source][source]
Execute a call_function node and return the result.

Parameters
target (Target) – The call target for this node. See Node for details on semantics

args (Tuple) – Tuple of positional args for this invocation

kwargs (Dict) – Dict of keyword arguments for this invocation

Return type
Any

Return
Any: The value returned by the function invocation

Note

Backwards-compatibility for this API is guaranteed.

call_method(target, args, kwargs)[source][source]
Execute a call_method node and return the result.

Parameters
target (Target) – The call target for this node. See Node for details on semantics

args (Tuple) – Tuple of positional args for this invocation

kwargs (Dict) – Dict of keyword arguments for this invocation

Return type
Any

Return
Any: The value returned by the method invocation

Note

Backwards-compatibility for this API is guaranteed.

call_module(target, args, kwargs)[source][source]
Execute a call_module node and return the result.

Parameters
target (Target) – The call target for this node. See Node for details on semantics

args (Tuple) – Tuple of positional args for this invocation

kwargs (Dict) – Dict of keyword arguments for this invocation

Return type
Any

Return
Any: The value returned by the module invocation

Note

Backwards-compatibility for this API is guaranteed.

fetch_args_kwargs_from_env(n)[source][source]
Fetch the concrete values of args and kwargs of node n from the current execution environment.

Parameters
n (Node) – The node for which args and kwargs should be fetched.

Returns
args and kwargs with concrete values for n.

Return type
Tuple[Tuple, Dict]

Note

Backwards-compatibility for this API is guaranteed.

fetch_attr(target)[source][source]
Fetch an attribute from the Module hierarchy of self.module.

Parameters
target (str) – The fully-qualified name of the attribute to fetch

Returns
The value of the attribute.

Return type
Any

Note

Backwards-compatibility for this API is guaranteed.

get_attr(target, args, kwargs)[source][source]
Execute a get_attr node. Will retrieve an attribute value from the Module hierarchy of self.module.

Parameters
target (Target) – The call target for this node. See Node for details on semantics

args (Tuple) – Tuple of positional args for this invocation

kwargs (Dict) – Dict of keyword arguments for this invocation

Returns
The value of the attribute that was retrieved

Return type
Any

Note

Backwards-compatibility for this API is guaranteed.

map_nodes_to_values(args, n)[source][source]
Recursively descend through args and look up the concrete value for each Node in the current execution environment.

Parameters
args (Argument) – Data structure within which to look up concrete values

n (Node) – Node to which args belongs. This is only used for error reporting.

Return type
Optional[Union[tuple[‘Argument’, …], Sequence[Argument], Mapping[str, Argument], slice, range, Node, str, int, float, bool, complex, dtype, Tensor, device, memory_format, layout, OpOverload, SymInt, SymBool, SymFloat]]

Note

Backwards-compatibility for this API is guaranteed.

output(target, args, kwargs)[source][source]
Execute an output node. This really just retrieves the value referenced by the output node and returns it.

Parameters
target (Target) – The call target for this node. See Node for details on semantics

args (Tuple) – Tuple of positional args for this invocation

kwargs (Dict) – Dict of keyword arguments for this invocation

Returns
The return value referenced by the output node

Return type
Any

Note

Backwards-compatibility for this API is guaranteed.

placeholder(target, args, kwargs)[source][source]
Execute a placeholder node. Note that this is stateful: Interpreter maintains an internal iterator over arguments passed to run and this method returns next() on that iterator.

Parameters
target (Target) – The call target for this node. See Node for details on semantics

args (Tuple) – Tuple of positional args for this invocation

kwargs (Dict) – Dict of keyword arguments for this invocation

Returns
The argument value that was retrieved.

Return type
Any

Note

Backwards-compatibility for this API is guaranteed.

run(*args, initial_env=None, enable_io_processing=True)[source][source]
Run module via interpretation and return the result.

Parameters
*args – The arguments to the Module to run, in positional order

initial_env (Optional[Dict[Node, Any]]) – An optional starting environment for execution. This is a dict mapping Node to any value. This can be used, for example, to pre-populate results for certain Nodes so as to do only partial evaluation within the interpreter.

enable_io_processing (bool) – If true, we process the inputs and outputs with graph’s process_inputs and process_outputs function first before using them.

Returns
The value returned from executing the Module

Return type
Any

Note

Backwards-compatibility for this API is guaranteed.

run_node(n)[source][source]
Run a specific node n and return the result. Calls into placeholder, get_attr, call_function, call_method, call_module, or output depending on node.op

Parameters
n (Node) – The Node to execute

Returns
The result of executing n

Return type
Any

Note

Backwards-compatibility for this API is guaranteed.

classtorch.fx.Transformer(module)[source][source]
Transformer is a special type of interpreter that produces a new Module. It exposes a transform() method that returns the transformed Module. Transformer does not require arguments to run, as Interpreter does. Transformer works entirely symbolically.

Example

Suppose we want to swap all instances of torch.neg with torch.sigmoid and vice versa (including their Tensor method equivalents). We could subclass Transformer like so:

class NegSigmSwapXformer(Transformer):
    def call_function(
        self, target: "Target", args: Tuple[Argument, ...], kwargs: Dict[str, Any]
    ) -> Any:
        if target == torch.sigmoid:
            return torch.neg(*args, **kwargs)
        return super().call_function(target, args, kwargs)

    def call_method(
        self, target: "Target", args: Tuple[Argument, ...], kwargs: Dict[str, Any]
    ) -> Any:
        if target == "neg":
            call_self, *args_tail = args
            return call_self.sigmoid(*args_tail, **kwargs)
        return super().call_method(target, args, kwargs)


def fn(x):
    return torch.sigmoid(x).neg()


gm = torch.fx.symbolic_trace(fn)

transformed: torch.nn.Module = NegSigmSwapXformer(gm).transform()
input = torch.randn(3, 4)
torch.testing.assert_close(transformed(input), torch.neg(input).sigmoid())
Parameters
module (GraphModule) – The Module to be transformed.

Note

Backwards-compatibility for this API is guaranteed.

call_function(target, args, kwargs)[source][source]
Note

Backwards-compatibility for this API is guaranteed.

Return type
Any

call_module(target, args, kwargs)[source][source]
Note

Backwards-compatibility for this API is guaranteed.

Return type
Any

get_attr(target, args, kwargs)[source][source]
Execute a get_attr node. In Transformer, this is overridden to insert a new get_attr node into the output graph.

Parameters
target (Target) – The call target for this node. See Node for details on semantics

args (Tuple) – Tuple of positional args for this invocation

kwargs (Dict) – Dict of keyword arguments for this invocation

Return type
Proxy

Note

Backwards-compatibility for this API is guaranteed.

placeholder(target, args, kwargs)[source][source]
Execute a placeholder node. In Transformer, this is overridden to insert a new placeholder into the output graph.

Parameters
target (Target) – The call target for this node. See Node for details on semantics

args (Tuple) – Tuple of positional args for this invocation

kwargs (Dict) – Dict of keyword arguments for this invocation

Return type
Proxy

Note

Backwards-compatibility for this API is guaranteed.

transform()[source][source]
Transform self.module and return the transformed GraphModule.

Note

Backwards-compatibility for this API is guaranteed.

Return type
GraphModule

torch.fx.replace_pattern(gm, pattern, replacement)[source][source]
Matches all possible non-overlapping sets of operators and their data dependencies (pattern) in the Graph of a GraphModule (gm), then replaces each of these matched subgraphs with another subgraph (replacement).

Parameters
gm (GraphModule) – The GraphModule that wraps the Graph to operate on

pattern (Union[Callable, GraphModule]) – The subgraph to match in gm for replacement

replacement (Union[Callable, GraphModule]) – The subgraph to replace pattern with

Returns
A list of Match objects representing the places in the original graph that pattern was matched to. The list is empty if there are no matches. Match is defined as:

class Match(NamedTuple):
    # Node from which the match was found
    anchor: Node
    # Maps nodes in the pattern subgraph to nodes in the larger graph
    nodes_map: Dict[Node, Node]
Return type
List[Match]

Examples:

import torch
from torch.fx import symbolic_trace, subgraph_rewriter


class M(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x, w1, w2):
        m1 = torch.cat([w1, w2]).sum()
        m2 = torch.cat([w1, w2]).sum()
        return x + torch.max(m1) + torch.max(m2)


def pattern(w1, w2):
    return torch.cat([w1, w2])


def replacement(w1, w2):
    return torch.stack([w1, w2])


traced_module = symbolic_trace(M())

subgraph_rewriter.replace_pattern(traced_module, pattern, replacement)
The above code will first match pattern in the forward method of traced_module. Pattern-matching is done based on use-def relationships, not node names. For example, if you had p = torch.cat([a, b]) in pattern, you could match m = torch.cat([a, b]) in the original forward function, despite the variable names being different (p vs m).

The return statement in pattern is matched based on its value only; it may or may not match to the return statement in the larger graph. In other words, the pattern doesn’t have to extend to the end of the larger graph.

When the pattern is matched, it will be removed from the larger function and replaced by replacement. If there are multiple matches for pattern in the larger function, each non-overlapping match will be replaced. In the case of a match overlap, the first found match in the set of overlapping matches will be replaced. (“First” here being defined as the first in a topological ordering of the Nodes’ use-def relationships. In most cases, the first Node is the parameter that appears directly after self, while the last Node is whatever the function returns.)

One important thing to note is that the parameters of the pattern Callable must be used in the Callable itself, and the parameters of the replacement Callable must match the pattern. The first rule is why, in the above code block, the forward function has parameters x, w1, w2, but the pattern function only has parameters w1, w2. pattern doesn’t use x, so it shouldn’t specify x as a parameter. As an example of the second rule, consider replacing

def pattern(x, y):
    return torch.neg(x) + torch.relu(y)
with

def replacement(x, y):
    return torch.relu(x)
In this case, replacement needs the same number of parameters as pattern (both x and y), even though the parameter y isn’t used in replacement.

After calling subgraph_rewriter.replace_pattern, the generated Python code looks like this:

def forward(self, x, w1, w2):
    stack_1 = torch.stack([w1, w2])
    sum_1 = stack_1.sum()
    stack_2 = torch.stack([w1, w2])
    sum_2 = stack_2.sum()
    max_1 = torch.max(sum_1)
    add_1 = x + max_1
    max_2 = torch.max(sum_2)
    add_2 = add_1 + max_2
    return add_2
Note

Backwards-compatibility for this API is guaranteed.