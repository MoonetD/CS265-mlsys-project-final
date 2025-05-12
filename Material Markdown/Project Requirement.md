Okay, here is the content of the PDF file converted to Markdown format, including descriptions of the graphs mentioned.

# CS265 Systems Project

**Spring 2024** [cite: 1]
**CS265: Big Data Systems** [cite: 1]
**DASlab** [cite: 1]
**HARVARD School of Engineering and Applied Sciences** [cite: 1]

## Activation Checkpointing: Trading off compute for memory in deep neural network training

### Introduction

The systems project for CS265 is designed to provide hands-on experience on the state-of-the-art systems for deep learning[cite: 1]. It includes understanding the system architecture of modern deep learning frameworks, analyzing the compute memory trade-offs involved in training deep learning models and implementing an algorithm that navigates this trade-off[cite: 2]. Systems projects will be done individually, each student is required to work on their own[cite: 3]. This is a focused project that should not necessarily result in many lines of code (like the CS165 project), but will exercise your understanding of modern deep learning systems[cite: 4].

The goal of this project is to implement an activation checkpointing algorithm in PyTorch[cite: 5]. The project is structured into three stages: the first stage involves creating a profiler to gather performance metrics during the training process, the second stage involves implementing an algorithm that determines which activations to checkpoint based on the profiler's statistics, and the final stage requires modifying the execution strategy to implement the decisions made by the algorithm[cite: 6]. Students that finish the systems project quickly and want to work on research will be able to do so by exploring open research topics directly on top of their project[cite: 7].

### Background

A typical neural network is a function that consists of multiple operations and is defined by a set of learnable parameters, known as weights ($W_1$, $W_3$)[cite: 8]. The example below illustrates a simple neural network composed of two linear operators followed by sigmoid activation functions and a mean squared error loss function[cite: 9]. During a single training iteration, the neural network is provided with an input sample (X) and a target value/label (Y)[cite: 10]. The forward pass processes the input by passing it through various functions to generate a prediction[cite: 11]. The difference between the prediction and the true value/label is referred to as the loss (L)[cite: 12]. The output of one function serves as the input for another and is called an activation/intermediate tensor/feature map ($Z_1$, $Z_2$, $Z_3$, and $Z_4$)[cite: 13]. These activations are stored in GPU memory during the forward pass and are used by the backward pass to calculate the weight gradients ($W_1$, $W_3$)[cite: 14].

*(Page 1 End)* [cite: 15]

---

*(Page 2 Start)* [cite: 16]

The size of these activations is determined by the size of the input and weights, while the number of activations is determined by the depth of the network (or the number of operations in the network)[cite: 16]. The weights and their corresponding gradients typically reside in GPU memory throughout the lifetime of the iteration[cite: 17]. The activations are stored when they are first generated (in forward pass) and are freed after their last use (in backward pass)[cite: 18].

**[Graph Descriptions Start]**

The document contains several diagrams on page 2 illustrating the concepts:

1.  **Forward Pass Box:** [cite: 19]
    * Input X
    * $Z_{1}=W_{1}X$
    * $Z_{2}=\sigma(Z_{1})$
    * $Z_{3}=W_{3}Z_{2}$
    * $Z_{4}=\sigma(Z_{3})$
    * $L=\frac{1}{2}(Z_{4}-Y)^{2}$
    * True Label Y

2.  **Update Rule Box:** [cite: 19]
    * $W_{i}=W_{i}-\alpha\nabla W_{i}$

3.  **Function Composition / Chain Rule Box:** [cite: 19]
    * $y=f(x)$
    * $z=g(y)$
    * $z=g(f(x))$
    * $\frac{\delta z}{\delta x}=\frac{\delta g(y)}{\delta y}\frac{\delta y}{\delta x}$
    * $\frac{\delta z}{\delta x}=\frac{\delta g(f(x))}{\delta f(x)}\frac{\delta f(x)}{\delta x}$

4.  **Backward Pass Box:** [cite: 19]
    * $\nabla W_{1}=\nabla Z_{1}X$
    * $\nabla Z_{1}=\nabla Z_{2}Z_{2}(1-Z_{2})$
    * $\nabla Z_{2}=\nabla Z_{3}W_{3}$  *(Note: Original PDF seems to have $\nabla Z_{1}$ here, corrected to $\nabla Z_{2}$ based on typical backpropagation)*
    * $\nabla W_{3}=\nabla Z_{3}Z_{2}$
    * $\nabla Z_{3}=\nabla Z_{4}Z_{4}(1-Z_{4})$
    * $\nabla Z_{4}=(Z_{4}-Y)$

5.  **Forward Computational Graph:** [cite: 19]
    * A linear flow diagram showing: X -> (W1 applied) -> Z1 -> Z2 -> (W3 applied) -> Z3 -> Z4 -> L

6.  **Backward Computational Graph:** [cite: 19]
    * A more complex diagram showing dependencies for gradient calculations. It shows how gradients (VZ1, VZ2, VZ3, VZ4 - likely representing $\nabla Z_1$ etc.) are computed using activations (Z2, Z4) and weights (W1, W3) flowing back from the Loss L. It highlights which activations (Z2, Z4) are needed for computing gradients related to weights (W1, W3).

**[Graph Descriptions End]**

### Motivation

Training neural networks is a process that demands significant computational resources, both in terms of memory and processing power[cite: 19]. Recent trends indicate a consistent and exponential increase in the size of neural networks and the datasets used for their training[cite: 20]. However, the memory capacity of GPUs, which are the most commonly used accelerators for training, has only seen linear growth over the past decade[cite: 21]. This discrepancy has resulted in accelerator memory becoming a scarce and valuable resource[cite: 22]. The peak memory required during training is directly proportional to the size of the model and the mini-batch size[cite: 23]. Consequently, the limited memory capacity of GPUs imposes restrictions on the size of the model and/or the mini-batch size that can be used during training[cite: 24].

### Problem

In our project, we focus on models whose parameters, gradients, and optimizer states, can be accommodated on a single GPU[cite: 25]. To gain insight into the source of the substantial memory footprint during training, we conduct an analysis of peak memory consumption[cite: 26]. The primary contributor to peak memory usage, accounting for approximately 70-85%, are the activations[cite: 27]. Additionally, it's worth noting that the sequence in which the activations are generated and consumed is reversed, which leads to them lying idle in the memory for a long duration[cite: 28].

*(Page 2 End)* [cite: 29]

---

*(Page 3 Start)* [cite: 30]

### Solution Overview

To reduce the peak memory consumption, one of the most popular techniques recently has been "Activation Checkpointing"[cite: 30]. Activation checkpointing (AC) addresses the issue by not storing all the activations in the memory during the forward pass[cite: 31]. Instead, it stores only a subset of them and recomputes the others during the backward pass as needed[cite: 32]. This approach trades off computation time for memory, as some computations need to be performed twice, but it can significantly reduce the memory requirements, enabling the training of larger models or the use of larger mini-batch sizes[cite: 33].

### Project

In our project we will implement the algorithm for activation checkpointing in Π-TWO[cite: 34].

1.  **Inputs:** We will experiment with two open source models (one vision and one LLM) [cite: 35]
    * a. Resnet-152 [cite: 35]
    * b. Bert [cite: 36]
2.  **AI Framework:** We will be using PyTorch as our framework for implementation and experimentation[cite: 36].
3.  **Components to be built:** [cite: 36]
    * a. Computation graph profiler[cite: 37],
    * b. Activation checkpointing (AC) algorithm[cite: 37],
    * c. Subgraph extractor and rewriter [cite: 37]
4.  **Deliverables** (Code Review and Demo) and a document with experimental analysis consisting of: [cite: 37]
    * a. Computation and memory profiling statistics and static analysis[cite: 38],
    * b. Peak memory consumption vs mini-batch size bar graph (w and w/o AC)[cite: 38],
    * c. Iteration latency vs mini-batch size performance graph (w and w/o AC) [cite: 39]

### Phase 1: Graph Profiler (3 weeks - 35%)

In the initial phase, our primary task is to construct a comprehensive computational graph[cite: 39]. This graph will encapsulate all operations from the forward, backward, and optimizer steps within a single iteration[cite: 40]. The nodes within this graph symbolize individual operations, while the edges represent the dependencies between input and output data[cite: 41]. The profiler's job is: [cite: 42]

1.  Collecting data on the computation time and memory usage of each operator when the graph operations are executed in topological order[cite: 42].
2.  Categorizing the inputs and outputs of each operation as a parameter, gradient, activation, optimizer state, or other types[cite: 43].
3.  Conducting static data analysis on activations by documenting the first and last use of each activation during the forward and backward passes[cite: 44].
4.  Generating a peak memory breakdown graph using the collected statistics[cite: 45].

*(Page 3 End)* [cite: 45]

---

*(Page 4 Start)* [cite: 46]

### Phase 2: Activation Checkpointing algorithm (2 weeks - 20%)

Using the inputs from the profiler, implement the activation checkpointing algorithm in Π-TWO that decides the subset of the activations to be retained and the subset of the activations to be recomputed[cite: 46].

### Phase 3: Graph Extractor and Rewriter (3 weeks - 45%)

Each activation that is chosen to be discarded in the forward pass needs to be recomputed in the backward pass when it is needed for gradient computation[cite: 47]. To implement this we will extract the subgraph that computes the activation in the forward pass and then replicate it[cite: 48]. This replicated sub-graph will be inserted into the backward pass just before it is required for gradient computation[cite: 49].

### Suggested Timeline

* **Week 1:** Familiarize yourself with training a simple model in PyTorch[cite: 50].
* **Week 2-4:** Development of graph profiler[cite: 50].
* **Week 5-6:** Implementing the recomputation algorithm from Π-TWO[cite: 51].
* **Week 7-9:** Implementing the subgraph extractor and rewriter[cite: 51].

### Midway Check-in: [cite: 52]

1.  Phase 1 completed[cite: 52].
2.  Document with experimental analysis consisting of deliverables 4(a) and 4(b) [w/o AC][cite: 52].

*(Page 4 End)* [cite: 52]