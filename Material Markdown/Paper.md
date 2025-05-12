Okay, I can help you convert the text content of the PDF into Markdown. Please note that converting complex layouts, figures, and graphs directly from PDF to Markdown can be challenging, and some manual adjustments might be needed for perfect formatting.

Here is the Markdown conversion of the PDF content:

```markdown
--- PAGE 1 ---

# [cite: 1] μ-TWO: 3× FASTER MULTI-MODEL TRAINING WITH ORCHESTRATION AND MEMORY OPTIMIZATION

Sanket Purandare¹, Abdul Wasay², Animesh Jain³, Stratos Idreos¹

## ABSTRACT

In this paper, we identify that modern GPUs - the key platform for developing neural networks - are being severely underutilized, with ~50% utilization, that further drops as GPUs get faster. [cite: 2] We show that state-of-the-art training techniques that employ operator fusion and larger mini-batch size to improve GPU utilization are limited by memory and do not scale with the size and number of models. [cite: 3] Additionally, we show that using state-of-the art data swapping techniques (between GPU and host memory) to address GPU memory limitations lead to massive computation stalls as network sizes grow. [cite: 4] We introduce μ-two, a novel compiler that maximizes GPU utilization. [cite: 5] At the core of μ-two is an approach that leverages selective data swapping from GPU to host memory only when absolutely necessary, and maximally overlaps data movement with independent computation operations such that GPUs never have to wait for data. [cite: 6] By collecting accurate run-time statistics and data dependencies, μ-two automatically fuses operators across different models, and precisely schedules data movement and computation operations to enable concurrent training of multiple models with minimum stall time. [cite: 7] We show how to generate μ-two schedules for diverse neural network and GPU architectures and integrate μ-two into the PyTorch framework. [cite: 8] Our experiments show that μ-two can achieve up to a 3x speed-up across a range of network architectures and hardware, spanning vision, natural language processing, and recommendation applications.

## [cite: 9] 1 INTRODUCTION

**Deep learning: Ubiquitous but expensive.** Widespread deep learning workflows have enabled groundbreaking results for many applications, including but not limited to image recognition (Szegedy et al., 2017), recommendation systems (Naumov et al., 2019), natural language translation (Devlin et al., 2019), and video games (Berner et al., 2019). [cite: 10] However, training neural networks is expensive and has adverse environmental impact (Zhu et al., 2018; Strubell et al., 2019). [cite: 11] For instance, training BERT - a natural language model (with 200 million parameters) - takes 79 hours on 64 high-end GPUs resulting in an expense of approximately 12,000 USD and a carbon footprint of 1438 lbs (Devlin et al., 2019). [cite: 12] To put this in perspective, BERT's training phase (including architecture search) emits as much carbon as six typical US cars would over their lifetimes.

[cite: 13] ¹School of Engineering and Applied Sciences, Harvard University, Cambridge, USA
²Intel Corporation, Santa Clara, California, USA
³Meta Platforms Inc., Menlo Park, California, USA.
[cite: 14] Correspondence to: Sanket Purandare <sanketpurandare@g.harvard.edu>.

*Proceedings of the $6^{th}$ MLSys Conference, Miami Beach, FL, USA, 2023. Copyright 2023 by the author(s).*

**[cite: 15] Deep learning workflows train multiple models.** Various stages of deep learning workflows involve training more than one models which effectively multiplies the cost of training. [cite: 16] For example, during the model design stage, neural architecture search and hyper-parameter tuning require training of several models to come up with a near-optimal hyperparameter set (e.g., learning rate, momentum, and regularization) and architecture (e.g., number and types of layers) (Bergstra et al., 2011; Bergstra & Bengio, 2012; Elsken et al., 2019). [cite: 17] Similarly, during the training phase, ensemble learning trains multiple models to improve the accuracy (Wasay et al., 2020; Ganaie et al., 2021). [cite: 18] Such repetitive training tasks are prevalent, making up 70.2% of hardware resource consumption in a single-GPU setting (46.2%) and multi-GPU setting (24%) combined (Wang et al., 2021).

**[cite: 19] Low compute utilization.** Modern deep learning hardware (e.g. GPUs, TPUs and accelerators) has high compute power and data parallelism. [cite: 20] However, existing neural network operators cannot fully utilize modern hardware (Coleman et al., 2017; Zhu et al., 2018; Narayanan et al., 2018; Liu et al., 2020; Wang et al., 2021). [cite: 21] This low utilization is due to the prevalence of small memory-bound kernels and the inherent complexity of writing code that fully utilizes this compute-rich hardware (Wang et al., 2021).

--- PAGE 2 ---

## [cite: 22] μ-TWO: 3× Faster Multi-Model Training with Orchestration and Memory Optimization

State-of-the-art research proposes two strategies to tackle the problem of low compute utilization:

1) Increasing the minibatch size within a single neural network helps increase data parallelism and maximizes the number of floating point operations per second, thereby improving utilization on powerful accelerators (Rhu et al., 2016; Jain et al., 2018; Peng et al., 2020; Wahib et al., 2020). [cite: 23]
2) Concurrently training multiple models on a single accelerator improves compute utilization by taking advantage of structural similarity across the various models. [cite: 24] This concurrent training is achieved by fusing identical operators across the different models into a single operator (known as horizontal fusion) (Narayanan et al., 2018; Liu et al., 2020; Wang et al., 2021). [cite: 25]

**Large memory footprint limits scaling.** While these strategies can improve compute utilization, the size and number of models that we can concurrently train on a single GPU are drastically limited by the large (and increasing) memory requirements of the training process as well as the limited memory capacity of modern GPUs. [cite: 26]

**Feature maps lead to memory over-subscription.** For widely used models, the source of these large memory footprints are the feature maps (Rhu et al., 2016; Jain et al., 2018; Peng et al., 2020). [cite: 27] For instance, feature maps occupy 83% of the memory when training VGG-16, whereas it is 97% for Inception (Jain et al., 2018). [cite: 28] A key observation is that feature maps have high inactive time in memory, which we define as the time between when they are produced in the forward pass and utilized in the backward pass. [cite: 29] This is the key issue that leads to inefficient use of the limited GPU memory. [cite: 30] We refer to this problem as memory over-subscription.

**Swap and/or recompute tensors.** Two techniques have been proposed to address memory over-subscription. [cite: 31]
(1) **Tensor recomputation:** a subset of feature maps are discarded after their use in the forward pass and recomputed when needed during the backward pass (Chen et al., 2016; Jain et al., 2020). [cite: 32]
(2) **Tensor swapping:** a subset of feature maps are offloaded to the larger host memory (i.e. CPU DRAM) during the forward pass and are fetched back into the GPU memory during the backward pass (Rhu et al., 2016; Peng et al., 2020; Wahib et al., 2020; Ren et al., 2021). [cite: 33]

**Tensor swapping and recomputation do not scale.** As the size and number of models grow, directly applying the cutting-edge tensor swapping and recomputation techniques [cite: 34] to concurrent multi-model training, leads to significant slowdowns. This is due to non-trivial overheads from tensor fetching delays during the backward pass and performance overheads from superfluous tensor recomputation. [cite: 35] We show in our experiments that this slowdown can be as high as 50%. [cite: 36]

*[Figure 1: Multi-model training performance (latency), is a function of the trade-off space defined by compute utilization, peak memory consumption and degree of independence between operations.]*

**Problem.** [cite: 37] This paper tackles the problem of scaling concurrent multi-model training as models grow and peak memory requirement surpasses the available GPU memory capacity. [cite: 38]

**Challenge.** To efficiently scale concurrent training, we need to answer several questions: How many operations should we fuse across models to saturate compute? [cite: 39] How many, and which intermediate tensors should be swapped or recomputed (if any)? When to discard/offload them and when to recompute/prefetch them back? [cite: 40] How to maximally overlap the data movement with useful compute to minimize stalls? Many of these questions are NP-Hard problems.

**Solution - μ-two.** [cite: 41] We present μ-two, a novel compiler that maximizes GPU utilization to efficiently scale concurrent training of multiple models. [cite: 42] We show that μ-two achieves up to 3x faster end-to-end training latency than state-of-the-art approaches for models with memory requirements up to $6\times$ the GPU memory size. [cite: 43] The core insight in μ-two's design is that the training performance (latency), of any given set of models, is a function of the trade-off space determined by compute utilization, peak memory consumption, and the degree of independence between operations, as illustrated in Figure 1. To achieve scalable model training, we need to efficiently navigate this trade-off, for a given set of models and hardware, instead of having a fixed strategy. [cite: 44] μ-two's compilation strategy automatically adapts to the properties of input models and the performance characteristics of the target GPU, enabling it to land in the sweet spot of the trade-off curve. [cite: 45] To accomplish this, for each training session, μ-two performs static data dependency analysis and lightweight, yet accurate, run-time profiling. [cite: 46] It then uses this information to (1) determine the number of operators to fuse (to saturate compute), (2) select the tensors to be swapped and/or recomputed (to reduce peak memory consumption), and (3) generate a tailored training schedule that maximally overlaps data movement with independent compute operations (to eliminate stalling). [cite: 47]

**Our Contributions are as follows:**

1.  We show that existing multi-model training and memory optimization strategies do not scale with the size and number of models, resulting in a 50% slowdown (§2 & §5).

--- PAGE 3 ---

## [cite: 48] μ-TWO: 3 Faster Multi-Model Training with Orchestration and Memory Optimization

2.  We derive the design of μ-two, a compiler based on the central idea that the training performance (latency) of a given set of models is a function of the trade-off space determined by compute utilization, peak memory consumption, and the degree of independence between operations (§3). [cite: 49]
3.  Given a set of input models and a target GPU, we explain how μ-two collects and uses static and run-time information to generate a tailored training schedule that maximally overlaps swaps with independent compute operations (§4). [cite: 50]
4.  We discuss how to integrate μ-two in the open source framework PyTorch (Paszke et al., 2019). [cite: 51] We show how each component of μ-two can be built with latest compiler tools and, can be smoothly plugged into the existing PyTorch execution stack (§B). [cite: 52]
5.  We conduct a thorough experimental analysis on a diverse set of hardware and several state-of-the-art model architectures spanning vision, natural language processing and recommendation systems. [cite: 53] Our results show that compared to the state-of-the-art approaches (HFTA), μ-two enables concurrent training of $3-5\times$ more models with a memory footprint of up to 6x the GPU memory size and delivers a 3x speedup (§5).

## [cite: 54] 2 BACKGROUND AND MOTIVATION

**Neural network training.** Training happens across several epochs. [cite: 55] During every epoch, the neural network processes the data in subsets called mini-batches. [cite: 56] For every round of mini-batch training, the computation is divided into two phases: (i) forward pass and (ii) backward pass. [cite: 57] Figure 2a shows a computational graph corresponding to a network with 5 parameterized layers.

i) **Forward pass.** [cite: 58] During the forward pass, the mini-batch is passed sequentially through every layer of the network to produce a set of neural network outputs. [cite: 59] As shown in Figure 2a, to produce the output tensor $Z_2$, we just need the input tensor $Z_1$, the weight tensor $W_2$, and enough memory to store the output tensor $Z_2$.
ii) **Backward pass.** [cite: 60] In the second phase, i.e., the backward pass, we compute the weight gradients. [cite: 61] Backward pass is, in principle, application of the derivative chain rule. [cite: 62] Like the forward pass, the backward pass is also processed sequentially but in reverse. [cite: 63] As shown in Figure 2b, For computing the weight gradient $\nabla W_2$ we require $\nabla Z_2$ and $Z_1$. The tensors, that are produced during the forward pass and are required for calculating the weight gradients in the backward pass, are called as feature maps or intermediate tensors, while the corresponding tensors produced during the backward pass for computing the weight gradients are called as gradient maps. [cite: 64]

**Inefficient memory usage.** The major training frameworks, including Tensorflow (Abadi et al., 2016), PyTorch (Paszke et al., 2019), and MXNet (Chen et al., 2015), suffer from inefficient memory utilization because they store all the intermediate tensors, gradient maps, weights, and weight gradients throughout the processing of a minibatch. [cite: 65]

**Out-of-memory strategies.** For many widely used models, the intermediate tensors produced during forward pass and consumed during backward pass are the major consumers of memory(Rhu et al., 2016; Jain et al., 2018; Peng et al., 2020). [cite: 66] Out-of-memory approaches address this problem of fixed and inefficient memory allocation by freeing up memory occupied by intermediate tensors that are not immediately needed. [cite: 67] These approaches are motivated by the fact that there is a huge temporal gap between the last use of a tensor during the forward pass and its first use in the backward pass. [cite: 68] Figure 3a shows this idle time for every feature map when training BERT (a state-of-the-art language model).

**Strategy 1: Tensor swapping.** [cite: 69] The tensor swapping strategies selectively offload tensors during the forward pass from the smaller GPU memory to the larger host memory. [cite: 70] During the backward pass, the offloaded tensors are prefetched before their use to minimize the overall execution time. [cite: 71] In scenarios with stringent memory constraints, we may need to offload several tenors to the host memory. [cite: 72] In such cases, the backward pass may incur several stall cycles waiting for the required tensors to be fetched. [cite: 73] Figure 2d shows what happens when computing $\nabla W_2$: the required tensor $Z_1$ needs to be swapped in from host memory. [cite: 74]

**Strategy 2: Tensor recomputation.** Tensor Recomputation strategies trade-off compute for memory. [cite: 75] These approaches discard a selected number of tensors after their last use in the forward pass. [cite: 76] During the backward pass, when these tensors are required for gradient computation, they are recomputed. [cite: 77] Tensor recomputation involves the repetition of several forward pass operations to recompute the desired tensors which adds compute overhead in favor of memory savings. [cite: 78] Figure 2c shows what happens when computing $\nabla W_2$: the required tensor $Z_1$ needs to be recomputed. [cite: 79] Unlike swapping, tensor recomputation does not add any stalling cycles in the execution path. [cite: 80] Although tensor recomputation keeps the GPU busy at all times, the computation is superfluous.

**Horizontal fusion.** [cite: 81] When concurrently training multiple neural networks on a single GPU to better utilize hardware, recent research proposes deeply fusing the neural networks together. [cite: 82] In this approach, known as horizontal fusion (Wang et al., 2021), operators corresponding to every layer in the set of concurrently trained neural networks are fused together. [cite: 83] These fused neural networks can then be trained simultaneously using a single GPU. [cite: 84] For instance, individual convolutional operators across models are mapped

--- PAGE 4 ---

*[Figure 2: (a) Computation graph of a simple neural network with five layers showing the various intermediate tensors produced and required during the forward and the backward pass. [cite: 85] (b), (c), and (d) show the computation of $\nabla W_2$ when no out-of-memory, tensor recomputation, and tensor swapping strategies are employed respectively.]* [cite: 86]

*[Figure 3: Memory Footprint Characteristics. (a) When training BERT (with batch size 32), feature maps stay idle in memory for a long period of time. [cite: 87] (b) Peak memory increases during the forward pass and decreases during the backward pass (BERT with batch size 32). [cite: 88] (c) We exceed the memory limit of the Nvidia A100 GPU when horizontally fusing four models (batch size: 8 and 16).] [cite: 89, 92, 101]

## [cite: 85] μ-TWO: 3× Faster Multi-Model Training with Orchestration and Memory Optimization

to grouped convolutions, while matrix multiplications are mapped to batch matrix multiplications. [cite: 90] While horizontal fusion can improve compute utilization, naively fusing neural networks easily lead to a scenario, where the fused network does not fit in memory. [cite: 91] As evidence of this, we show the memory requirement of training state-of-the-art models in Figure 3c. [cite: 92] We observe that fusing just four models leads to a scenario where we exceed the memory limit of the Nvidia A100 GPU across all these models. [cite: 93]

## [cite: 94] 3 μ-TWO: INSIGHTS AND OVERVIEW

We discuss the core insights that drive μ-two's design and provide an overview of our compiler through an example. The next section discusses the core algorithm in detail.

### [cite: 94] 3.1 μ-two Insights and design space

**Insight I1:** Swapping makes only the backward pass IO-sensitive. [cite: 95] Any tensor swapping algorithm changes the IO sensitivity of the backward pass. [cite: 96] During the forward pass, the tensors to be offloaded can be sent to the host memory asynchronously without blocking the computation process. [cite: 97] However, fetching the swapped tensor back to the GPU during the backward pass lies in the critical path. [cite: 98] If the required tensor is not fetched before the corresponding gradient calculation, then training stalls. [cite: 99]

**Insight I2:** The forward and backward passes of two concurrently trained models are independent. [cite: 100] When training a single model, the forward and backward passes depend on one another since the feature maps produced during a forward pass are utilized in the next backward pass for computing weight gradients (and the subsequent forward pass uses these gradient updates and so on). [cite: 101, 102] However, when concurrently training multiple models, there is no such data dependency across different models. [cite: 103]

**Insight I3:** Peak memory consumption monotonically increases during the forward pass and decreases during the backward pass. [cite: 104] During the forward pass, feature maps are created. During the backward pass, these accumulated feature maps are used to calculate the gradients and are released as soon as they are used. [cite: 105] Figure 3b shows this phenomenon when training BERT.

**Design implications.** (I1) motivates that swapping should be scheduled conservatively, and in order to minimize stall overheads, one needs to achieve as much overlap with compute as possible. [cite: 106] (I2) suggests that when concurrently training models, we can use compute operations from some models to overlap swapping operations from other models. [cite: 107] Since modern interconnects (PCIe and NVLinks) allow full duplex data transfers, the forward pass and backward pass data transfers across models are resource-independent. [cite: 108] (I3) suggests that, we should only multiplex backward pass operations with forward pass operations (from other models) due to their contrasting memory consumption patterns. [cite: 109] (I1) also supports this observation since the IO sensitivities of the forward pass and backward pass are complementary.

--- PAGE 5 ---

## [cite: 110] μ-TWO: 3× Faster Multi-Model Training with Orchestration and Memory Optimization

**Design trade-off space.** [cite: 111] Recent work has shown that horizontal operator fusion across multiple models is necessary to achieve maximum compute utilization. [cite: 112] Combined with the design implications stated above, this lands us in a design trade-off space where different approaches exist with respect to compute utilization, independence between operations, and memory consumption. [cite: 113] There are two extremes.

a) **Complete fusion.** On one hand, if we fuse all operations horizontally, we get a single and monolithic forward and backward pass, with individual operations from each of the participating models being inseparable. [cite: 114] This means that: (i) we can achieve maximum compute utilization, but (ii) with minimal opportunities to multiplex operations, and (iii) with high peak memory consumption, resulting in more swap/recompute operations. [cite: 115]
b) **No fusion.** On the other end of this spectrum, not applying any horizontal fusion gives us a separate forward and backward pass for each model. [cite: 116] This means that we get (i) severe under-utilization of compute, but (ii) we get maximum independence across multiple operations from different models for overlapping with swaps, and (iii) the peak memory consumption of non-fused operations is low, resulting in minimal swap/recompute operations. [cite: 117]

**Design choice for μ-TWO: Sub-array fusion.** At the core of μ-two's approach is the identification of the optimal point within the trade-off space by utilizing the degree of fusion as a parameter to strike a balance between competing objectives. [cite: 118] This is achieved by partitioning the target model array into multiple sub-arrays and horizontally fusing operations across the models within each sub-array. [cite: 119] This sub-array fusion balances desirable properties of the two extremes: μ-two can (i) sufficiently utilize compute, (ii) multiplex operations from the forward pass of one sub-array with backward pass operations of another sub-array (vice versa), providing opportunities to overlap any necessary swaps and (iii) also reduce peak memory consumption. [cite: 120]

### [cite: 120] 3.2 μ-two system overview and example

We present the system architecture of μ-TWO (Figure 4) through a running example.

**Input.** [cite: 121] The input to the μ-two system is the set of models to train, i.e., their exact specification: architecture, loss function, mini-batch size, etc. Every model can have different hyperparameters, such as momentum, learning rate, and initialization but the architecture should be the same across all models. [cite: 122] We illustrate μ-two's behavior through a running example, where an array of eight models $[M_1...M_8]$ is given as input. [cite: 123]

**1. Model sub-array constructor** The first step is to enumerate all possible sub-array partitions of the input array of models. [cite: 124] For our example of eight models, this results in the following four partitions:

(a) 2 sub-arrays of 4 models
(b) 2 sub-arrays of 3 models, 2 sub-arrays of 1 model
(c) 4 sub-arrays of 2 models each
(d) 8 sub-arrays of 1 model each

The input number of models can be odd or even. [cite: 125] The number of sub-arrays created will always be even, since sub-arrays are processed in pairs. [cite: 126] For example, we show processing of partition (a) in Figure 4: given the model array $[M_1...M_8]$ it creates two sub-arrays, $A_1$ consisting of models $[M_1...M_4]$ and $A_2$ consisting of $[M_5...M_8]$. [cite: 127]

**2. Horizontal fuser** For each possible partition of the input model array, the operators across models within each sub-array are horizontally fused. [cite: 128] For example, Figure 4 shows how, for partition (a), the models within $A_1$ and $A_2$ are horizontally fused to create two horizontally fused training arrays $FA_1$ and $FA_2$. [cite: 129]

**3. Graph tracer.** For each fused sub-array, the Graph tracer derives the forward and backward computational graph consisting of nodes as operations and edges describing the data flow dependencies. [cite: 130] As shown in Figure 4, for fused sub-array $FA_1$, the Graph tracer produces the forward pass graph $FW_1$ and backward pass graph $BW_1$, and similarly for $FA_2$ it traces $FW_2$ and $BW_2$. [cite: 131]

**4. Profiler.** Then for each partition, the Profiler runs 3 iterations (after 1 warm-up) to collect performance profiling statistics so we can compare those partitions. [cite: 132] There is an extensive set of data collected a detailed description is provided in Section 4.1. [cite: 133] Profiling data includes:

(i) Static analysis statistics such as the uses of feature maps in the forward and backward pass. [cite: 134]
(ii) Run-time statistics such as the run-time of each operation, and swap-time of each feature map. [cite: 135]
(iii) Memory usage statistics such as the active and peak memory consumption during execution of each node, and size of feature maps. [cite: 136]

**5. Scheduler.** The collected profiling information is used to make scheduling and memory optimization decisions by the Scheduler. [cite: 137] The scheduler considers all possible partitions (e.g., (a) through (d) in our example) and simulates the expected training time for each partition when making the best possible memory and computation utilization decisions for each one. [cite: 138] If a partition does not satisfy the memory limit, it is discarded. [cite: 139] At the end, the scheduler picks the partition with the shortest expected training time. [cite: 140] The Scheduler considers two sub-arrays at a time (closest iteration time). [cite: 141] For example, for partition (a) in our running example (2 sub-arrays of 4 models), the Scheduler first operates on the backward graph $BW_1$ of fused model sub-array $FA_1$ and the forward graph $FW_2$ of fused model sub-

--- PAGE 6 ---

*[Figure 4: μ-two System architecture: the end-to-end operations flow for 2 sub-arrays of 4 models.]* [cite: 142]

## μ-TWO: 3× Faster Multi-Model Training with Orchestration and Memory Optimization

array $FA_2$, since their operations are pair-wise independent. It then operates on $[BW_2, FW_1]$. The Scheduler utilizes the following components to analyze sub-arrays.

(i) **Swap/recompute calculator.** It makes a greedy decision regarding whether a feature map needs to be swapped or recomputed and calculates the cost for each case. [cite: 143]
(ii) **Multiplexer.** It multiplexes operations from the forward graph to maximize the overlap of compute with swapping operations in the backward graph. [cite: 144]
(iii) **Memory simulator.** It validates the decisions made by the Swap/recompute calculator and Multiplexer by ensuring that they do not violate GPU memory constraints. [cite: 145]

We describe these components in detail in Section 4.2.

**6. Graph rewriter.** [cite: 146] The Graph rewriter processes the graph pairs corresponding to the partition selected by the Scheduler. [cite: 147] In our running example, it is partition (a). As illustrated in Figure 4 the Graph rewriter takes in graphs $[BW_1, FW_2]$ to produce a merged graph $G_1$ that reflects the decisions made by the Scheduler. [cite: 148] The merged graph includes hints at various nodes to enforce decisions like: when a tensor should be swapped in/out, when a tensor should be discarded etc. It also extracts sub-graphs for regenerating the tensors selected for recomputation and inserts them at appropriate locations. [cite: 149] It then repeats the same process for $[BW_2, FW_1]$ to produce $G_2$.

**7. Schedule interpreter.** [cite: 150] The Schedule interpreter enforces the decisions made by the Scheduler and utilizes the hints provided by the graph re-writer to drive the execution of the merged graphs, e.g., $G_1$ and $G_2$ in our running example. [cite: 151]

## [cite: 151] 4 THE μ-TWO ALGORITHM

In this section, we describe the algorithms behind the core components of Profiler and Scheduler. [cite: 152] Appendix A provides detailed algorithms for all μ-two components.

### [cite: 152] 4.1 Profiling algorithm

The Profiler executes the computational graph to collect metrics and populates them as node attributes, as listed in Tables A and B in the Appendix. [cite: 153] Figure 5a depicts the two-phase flowchart for profiling.

(i) **Static data flow analysis** gathers metrics that can be inferred from the computational graph without running it. [cite: 154] These metrics capture information about the order in which tensors are accessed. [cite: 155] For instance, `last_fw_uses` denotes the set of all tensors that had their last forward use at this node, while `first_bw_uses` denotes the set of tensors that had their first backward use at this node. [cite: 156]
(ii) **Run-time analysis** is conducted over three iterations, with one warm-up iteration. For all run-time data collected, the median statistic is used. [cite: 157] The profiling process involves three stages. In stage (1), before executing an operation during the backward pass, all intermediate tensors required for the operation and offloaded to host memory are swapped-in back to GPU memory. [cite: 158] In stage (2), the operation is executed, and its end-to-end run time and memory usage are collected. [cite: 159] In stage (3), after executing an operation in the forward pass, all intermediate tensors are swapped-out to host memory after their last use. [cite: 160] Stages (1) and (3) enable profiling of computational graphs of models that exceed the GPU memory limit with the minimum assumption that inputs, outputs, and operation workspaces must fit on GPU memory in isolation. [cite: 161] For the intermediate tensors we collect additional run-time information required for choosing the memory optimization strategy, such as swap time, in stages (1) and (3). [cite: 162] Subsequent to profiling we populate the attribute `inactive time`, which denotes the time elapsed between the last forward [cite: 163] and first backward access of the intermediate tensor.

### [cite: 163] 4.2 Scheduling algorithm

The Scheduler takes as input a backward graph $BW_j$ and a forward graph $FW_i$ corresponding to fused model sub-arrays $FA_j$ and $FA_i$ and their profiling information with the objective of minimizing GPU idle time and superfluous compute under the given GPU memory constraints. [cite: 164]

#### [cite: 164] 4.2.1 Scheduling policy

The scheduling algorithm determines whether to recompute or swap intermediate tensors for the forward and backward pass graph $[FW_i, BW_j]$. It uses two metrics: (1) the time elapsed between last use in forward pass and first

--- PAGE 7 ---

*[Figure 5: Flow diagrams representing the profiler, scheduling policy and swap simulation for overhead calculation algorithms.]* [cite: 165, 167, 170, 166, 168, 169, 171, 172]

## μ-TWO: 3× Faster Multi-Model Training with Orchestration and Memory Optimization

use in backward pass (inactive time) and, (2) the ratio of memory occupied by a tensor over the time required to recompute it (recompute ratio). [cite: 173, 174] These metrics capture an approximation of the overhead due to swapping or recomputing tensors. [cite: 175] Tensors are selected for swapping or recomputation in decreasing order of their inactive times and recompute ratios using a greedy approach. [cite: 176] See Figure 5b for the detailed steps.

The policy iterates over a set of candidate tensors and selects the best swap and recompute candidates (`s_cand` and `r_cand`, respectively) using the metrics defined above. [cite: 177] Each memory optimization strategy has an associated overhead. For swapping, if the candidate tensor is not fetched before it is required in the backward pass, the processing stalls and the overhead is equal to the stall time. [cite: 178] The Scheduler first attempts to schedule the candidate with zero overhead by overlapping it with compute, and only encounters a stall if it is unavoidable. [cite: 179] For recomputation, the overhead is the minimum time required to recompute the tensor. [cite: 180] We calculate the swap overhead (`s_overhead`) for `s_cand` and recompute overhead (`r_overhead`) for `r_cand`. [cite: 181] The Multiplexer schedules the candidate with a lower overhead. It is then removed from the candidate set, and the side-effects of its selection are accounted for, by updating the candidates already chosen for swap or recompute, as well as the remaining candidates. [cite: 182] The memory simulator then simulates the new schedule to calculate the peak memory consumption. [cite: 183] If it is less than the specified limit, the process exits; otherwise, the steps are repeated. [cite: 184]

#### [cite: 184] 4.2.2 Swap simulation for overhead calculation

To calculate the swap overhead we simulate the swap using the current state of the schedule and attempt multiplexing nodes from forward and backward graph to maximally overlap the swap. [cite: 185] A step-by-step flowchart for calculating the swap overhead is shown in Figure 5c, while the timeline snapshots of the simulation algorithm are shown in Figure 6 and explained using an example. [cite: 186]

The swap overhead calculation involves two terms: (1) the set of consecutive operations where peak memory consumption exceeds the GPU memory limit (`peak_memory_interval`) and, (2) the node in the backward graph (`prefetch_prompt`) that begins swapping-in the intermediate tensor before its first `bw_access`. [cite: 187] Swapping is not possible in the peak memory interval as there is no memory available for the tensor being swapped in. Any delay in this process causes a stall, which is measured as the swapping overhead. [cite: 188] The algorithm takes in the `swap_cand`, its swap time, and a flag indicating whether the peak memory interval has been reached (`reached_peak`). [cite: 189] Based on when we enter the peak memory interval, the calculation of the swap overhead can be classified into three cases:

1.  **No overlap.** [cite: 190] When the peak interval is already reached before scheduling the swap, we cannot overlap this swap with compute and the swap overhead is calculated based on whether this swap: (a) does not conflict with existing swap or (b) conflicts with an existing swap. [cite: 191] In case 1(a), the swap overhead is the total swap time of the candidate. [cite: 192] In case 1(b), the swap overhead is the remaining time of the the conflicting swap (`remaining_time`), plus the `swap_time` of this candidate. [cite: 193]
2.  **Complete overlap.** When peak interval is not reached after scheduling swap, the swap can be completely overlapped by using: (a) forward pass operations only or (b) mix of forward and/or backward pass operations. [cite: 194] For both the cases 2(a) and (b), we are able to completely overlap the swap with useful compute resulting in zero overhead. [cite: 196]
3.  **Partial overlap.** When peak interval is reached while scheduling swap, the swap is partially overlapped by using a mix of forward and/or backward pass operations. [cite: 197] In case 3, any remaining time that we are not able to overlap with compute is the swapping overhead. [cite: 198]

--- PAGE 8 ---

*[Figure 6: Timeline snapshots of swap simulations for calculating the swap overhead of intermediate tensors.]* [cite: 195]

## μ-TWO: 3× Faster Multi-Model Training with Orchestration and Memory Optimization

**Example.** [cite: 199] Figure 6a shows the backward computational graph $BW_1$ for fused sub-array 1, while Figure 6b, shows the forward computational graph $FW_2$ for fused sub-array 2. Figure 6c shows the topologically sorted compute operation timeline of $BW_1$ followed by $FW_2$. [cite: 200] Based on the inactive times, the intermediate tensors to be scheduled for swapping for $BW_1$ are ordered as $Z_1^1, Z_2^1, Z_3^1, Z_4^1$ with their first backward accesses being $\nabla W_2^1, \nabla W_3^1, \nabla W_4^1, \nabla W_5^1$ respectively. [cite: 201] First we try scheduling swap-in of $Z_1^1$ whose `first_bw_access` is at $\nabla W_2^1$. [cite: 202] We then try to overlap it with forward operations $Z_1^2, Z_2^2$ and are able to do so with zero overhead, resulting in Case 2(a) as shown in Figure 6d. [cite: 203] Next, we try scheduling the swap-in of $Z_2^1$. We are only able to use one forward operation $Z_1^2$ due to memory constraints. [cite: 204] Hence, we also make use of the backward operation $\nabla Z_3^1$ to overlap it. [cite: 205] Since we moved $Z_1^2$ ahead in our timeline, we need to adjust the previous overlap of $Z_1^1$ using the subsequent forward operations $Z_2^2, Z_3^2$. We are successful in doing so with zero overhead resulting in Case 2(b) as shown in Figure 6e. [cite: 206] Next we try scheduling the swap-in of $Z_3^1$. We cannot use any more forward operations due to memory constraints. [cite: 207] Hence, we try to make use of backward operations. We see that we could use only one backward operation $\nabla Z_4^1$ since we reach the peak memory interval (indicated by red marker). [cite: 208] We are only able to partially overlap this swap-in resulting in swap overhead equal to its remaining time (stall time) resulting in Case 3, as shown in Figure 6f. [cite: 209] At this point we can also switch to recompute $Z_3^1$ if its recompute overhead is lower than the stall time. [cite: 210] Finally, we try to schedule the swap-in of $Z_4^1$. Since its `first_bw_access` ($\nabla W_5^1$) happens during peak-interval, we cannot overlap it with compute operations. [cite: 213] Hence, we encounter a stall equal to its swap time. [cite: 214] This results in Case 1(a) as shown in Figure 6g. Similar to the previous case, we can decide to recompute $Z_4^1$ instead, if its recompute overhead is lower. [cite: 215]

## [cite: 215] 5 EXPERIMENTAL ANALYSIS

We show how μ-two improves training time up to 3x across several state-of-the-art and diverse models.

**Neural networks.** [cite: 216] We evaluate μ-two on six state-of-the-art neural network models from computer vision, natural language processing, and recommendation systems (Table 1). [cite: 217] These models cover a diverse array of use cases, architectures, and model/batch sizes.

*[Table 1: We evaluate μ-two on state-of-the-art models covering a large space of use cases, arch. [cite: 211] features, and model/batch sizes.]* [cite: 212]

| Application              | Model Name        | Functionality                                      | Architectural Features                           | Params | Batch Sizes |
|--------------------------|-------------------|----------------------------------------------------|--------------------------------------------------|--------|-------------|
| Vision                   | Vision Transformer| Image Classification, Image Segmentation, Action Recognition | Positional image embeddings, transformers         | 60M    | 8, 16       |
|                          | Mobilenet v3 large|                                                    | Depthwise separable convolutions               | 5.4M   | 64, 128     |
|                          | Resnet 101        |                                                    | Convolutions, Skip Connections                   | 44.5M  | 48, 64      |
| Natural Language Processing | Bert              | Predict Next Sentence                              | Transformer Encoders                             | 100M   | 16, 24      |
|                          | GPT2              | Predict Next Token                                 | Transformer Decoders                             | 124M   | 8, 16       |
| Recommender Systems      | NVIDIA DLRM       | Item Recommendation                                | Encoders, Decoders, sparse embeddings            | 40M    | 512, 1024   |

**Experimental setup.** [cite: 218] We report results on setups with two different classes of Nvidia GPUs: A100 with 40 GBs of memory and V100 with 32 GBs of memory (more details in Table C in the Appendix). [cite: 219] All experiments are run on the machine with the latest A100 GPU unless stated otherwise.

--- PAGE 9 ---

*[Figure 7: μ-two saves between 5 to 40 hours in end-to-end training time when concurrently training several BERT and ViT models.]* [cite: 220]

## [cite: 220] μ-TWO: 3× Faster Multi-Model Training with Orchestration and Memory Optimization

**[cite: 221] Baselines.** We compare against two baselines:

1.  **HFTA-NoMemOpt.** HFTA is the state of the art and has shown 3-11x speed-ups over all other concurrent training techniques (Wang et al., 2021). [cite: 222] It offers no memory optimization and requires the allocation of peak memory during every training iteration. [cite: 223] To provide a fair comparison, when the model array we want to train doesn't fit in memory, we break them down into subsets that do and sequentially train each subset. [cite: 224]
2.  **HFTA-Capuchin.** We construct this baseline by applying a state-of-the-art memory optimization algorithm Capuchin (Peng et al., 2020) to HFTA. [cite: 225] Capuchin is a hybrid strategy for memory optimization, developed in the context of single model training, that uses both tensor swapping and tensor recomputation strategies to train models having peak memory consumption more than the GPU memory capacity. [cite: 226] The purpose of this baseline is to evaluate the performance of μ-two relative to naively applying state-of-the-art memory optimization strategy to concurrently training models. [cite: 227]

**Metrics.** We report: (i) the improvement in end-to-end training time (i.e.. decrease in GPU hours) and (ii) improvement in one round of mini-batch training (i.e., normalized speedup). [cite: 228] We do not report accuracy or show convergence curves since the the training algorithm remains unchanged. [cite: 229]

### [cite: 229] 5.1 μ-two saves upto 40 out of 60 GPU hours in end-to-end training time

First, we show the impact of μ-two on end-to-end training time of two large scale state-of-the-art models - Vision Transformer and Bert. [cite: 230] We vary the number of concurrently trained models and train each set of models for 10k iterations. [cite: 231] As shown in Figure 7, μ-two saves between 5 to 40 hours in training time when compared with state-of-the-art HFTA baseline. [cite: 232] This result indicates that concurrently training models on a single GPU can significantly reduce the time (and the dollar cost) of training deep learning models. [cite: 233] Bert and Vision Transformer both require more than 10k iterations to converge, we cap-off our training at 10k, since it is sufficient to show the absolute benefit. thereby enabling model design and training in low-resource environments.

### [cite: 233] 5.2 μ-two achieves 3x speedup in iteration latency

Next, we show how μ-two speeds up training iteration latency. [cite: 234] In addition to the HFTA baseline, we compare against HFTA-Capuchin a variant of the μ-two system that fuses all models together to create a single execution graph (instead of the sub-array fusion employed by μ-two). [cite: 235] Comparing against HFTA-Capuchin help us evaluate the additional benefit provided by the sub-array fusion technique introduced in μ-two. [cite: 236] To measure the end to end latency, we perform 100 warm-up iterations to stabilize training and then measure 10 iterations using the PyTorch profiler and record the median. [cite: 237] We report the normalized speed up (over the HFTA baseline) in end-to-end iteration latency achieved by all three systems across all six models in Figure 8. μ-two consistently outperforms both HFTA-NoMemOpt and HFTA-Capuchin. [cite: 238] This speedup scales as we increase the number of models. Overall, μ-two results in upto 3x speedup compared to state-of-the-art HFTA. [cite: 239] When comparing μ-two and HFTA-Capuchin, we observe that μ-two consistently outperforms HFTA-Capuchin indicating that the sub-array fusion technique employed by μ-two is able to keep the memory optimization overhead to a minimum. [cite: 240] In case of HFTA-Capuchin, we observe that as we increase the number of models the memory optimization overhead (stemming from complete fusion of models) takes precedence and performance takes a hit. [cite: 241] This performance dip is particulary pronounced for compute-intensive vision models (i.e., Mobilenet, Resnet and NV DLRM). [cite: 242] Overall, μ-two speeds up per iteration training latency establishing a new baseline for concurrent training of multiple models on a single GPU. [cite: 243]

**Diverse hardware.** We repeat our experiments on a different set of hardware with V100 GPU (from Table C). [cite: 244] We scale the batch size appropriately to account for the smaller GPU memory size of this setup. [cite: 245] Figure 8g and Figure 8h show results for Bert and DLRM respectively. The trend continues to hold μ-two provides highest speed-up. [cite: 246]

### [cite: 246] 5.3 Performance breakdown

To understand in detail how and why μ-two achieves significant speed-up, we break down its performance using the following metrics. [cite: 247] We do the case study for two models, Bert 8a and Vision Transformer 8c, since they present an interesting behavior; [cite: 248] they both show good speed-up on the lower range of models with HFTA-Capuchin but then show a performance dip later on. [cite: 249] μ-two, on the other hand, shows consistent speed-up in both cases.

(a) **Iteration latency breakdown.** [cite: 250] We report how much time μ-two spends in useful compute and recompute oper-

--- PAGE 10 ---

*[Figure 8: μ-two results in up to 3x improvement in latency for one round of mini-batch training (forward and backward pass) when compared to state-of-the-art approaches across a diverse array of models. [cite: 251, 252] This improvement scales with the number of models.]* [cite: 253]

## μ-TWO: 3× Faster Multi-Model Training with Orchestration and Memory Optimization

ations compared to HFTA-NoMemOpt. [cite: 254] And we see that by applying horizontal fusion to larger set of models, μ-two is able to extract more compute utilization than HFTA-NoMemOpt. [cite: 255] Although, μ-two does some recompute due to fusing larger arrays of models, it does not outweigh its benefit.

(b) **Recomputation ratio.** [cite: 256] (lower is better) We measure the ratio of the time spent in recompute with respect to the total compute time and then compare μ-two with HFTA-Capuchin's recompute ratio. [cite: 257] Clearly, μ-two spends less than half of the relative time in recomputation than HFTA-Capuchin. [cite: 258] Since recompute is redundant, μ-two performs better by minimizing it.

(c) **Swap overlap ratio.** [cite: 259] (higher is better) We measure the ratio of the time μ-two and HFTA-Capuchin are able to overlap their swaps behind total compute time. [cite: 260] μ-two is able to overlap its swaps with useful compute, 2-3x more than HFTA-Capuchin. [cite: 261] This shows that μ-two's multiplexer does better job at using operations from multiple model sub-batches to overlap swaps. [cite: 262] This reduces the need for recomputing as well.

(d) **Peak memory consumption ratio.** [cite: 263] (lower is better) We measure the amount of peak memory that exceeds the GPU memory limit, before optimizing it. [cite: 264] Our goal is to show the ramifications of horizontally fusing all the operators together vs fusing them in sub-batches for HFTA-Capuchin and µμ-two respectively. [cite: 265] μ-two's sub-batching approach reduces the peak memory consumption, that further allows in reducing the number of swap and recompute operations. [cite: 266]

### [cite: 266] 5.4 μ-two profiling and scheduling overhead

Since the profiling is done only for 4 iterations (3 measured + 1 warm-up) per choice of model array partition, it is lightweight and takes on the order of seconds to complete. [cite: 267] Typically models take around thousands of iterations to converge so this overhead is negligible. [cite: 268] Choosing which tensors to swap or recompute has shown to be an NP-Hard problem (Jain et al., 2020; Peng et al., 2020; Wahib et al., 2020). [cite: 269] Hence, we greedily decide to swap or recompute a tensor in every iteration. [cite: 270] Overall our algorithm is quadratic in terms of the number of intermediate tensors. [cite: 271] The scheduler makes all its choices by simulations based on profiling data. [cite: 272] Therefore, it has has zero run-time overhead and typically completes scheduling within a minute.

## [cite: 272] 6 RELATED WORK

**Multi-model training.** [cite: 273] Model and data sharing techniques help to improve hardware utilization. For example, Hive-Mind avoids redundant data transfer costs, performs kernel fusions, and concatenates inputs for models with shared weights to increase the computational intensity of the kernels (Narayanan et al., 2018). [cite: 274] ModelPacking rewrites multiple neural networks as a single concatenated network (Liu et al., 2020). [cite: 275] Similarly, Horizontally Fused Training Array (HFTA) maps kernels across models into a single highly optimized kernel (Wang et al., 2021). [cite: 276] However, all such approaches require the models to fit onto the accelerator's memory. [cite: 277] On the other hand, μ-two supports out-of-memory execution, enabling larger batch sizes, larger model sizes and larger number of models being trained on the same accelerator. [cite: 278] A comparison of multi-

--- PAGE 11 ---

*[Figure 9: Performance breakdown of the BERT and ViT model shows that: (i) μ-two results in lower compute latency, (ii) μ-two provides a lower recompute ratio (i.e., fewer tensors need to be recomputed), (iii) $\mu-fWO$ is able to better overlap swap with useful compute, and (iv) overall results in less peak memory consumption.]* [cite: 279, 280, 281]

## μ-TWO: 3× Faster Multi-Model Training with Orchestration and Memory Optimization

model training techniques is shown in Table D. [cite: 282]

**Out-of-memory approaches.** We classify out-of-memory approaches into three categories. [cite: 283]
1.  **Tensor rematerialization:** Gradient-checkpointing employs recomputation of feature maps discarded at specific checkpoints during the forward pass thereby enabling training neural nets at sub-linear memory cost at the expense of an extra forward pass computation (Chen et al., 2016; Jain et al., 2020). [cite: 284]
2.  **Tensor swapping:** vDNN offloads tensors to the larger host memory during the forward pass and fetches them before they are required in the backward pass. [cite: 285]
3.  **Hybrid strategies:** Recent work combines these two techniques. Capuchin dynamically decides which tensors to discard or offload based on their idle time in GPU memory and their recompute vs. transfer time ratio (Peng et al., 2020). [cite: 286] KARMA proposes to interleave recompute with tensor prefetching to utilize idle GPU cycles when an offloaded tensor is being fetched back into memory (Wahib et al., 2020). [cite: 287] Zero-Infinity and vPipe apply the memory swapping techniques not only to the activations but also to the model parameters to allow scaling of large networks across multiple GPUs in context of model-parallel training. [cite: 288] (Rajbhandari et al., 2021; Zhao et al., 2022).

Although hybrid strategies can overlap some stalls incurred during swapping by recompute, the recomputation of these layers is still redundant computation. [cite: 289] μ-two avoids this problem. It overlaps the stalls incurred during fetching of offloaded layers in the backward pass of some models, with the forward pass operations of others, thereby performing useful compute during the stalling period and employing recompute only if there is no sufficient useful compute to overlap the memory transfers. [cite: 290] Hence, it eliminates the stalling of pure tensor offloading approaches and provides a superior compute-memory trade-off for hybrid approaches resulting in improved throughput. [cite: 291] A comparison of out-of-memory approaches with μ-two is shown in Table E. Appendix D contains additional discussion on related topics. [cite: 292]

## [cite: 292] 7 CONCLUSION

In this paper, we tackle the problem of slow neural network training due to compute underutilization and inefficient memory usage. [cite: 293] These problems become ever more critical as networks become more complex (larger) and as applications need to consider numerous networks simultaneously (e.g., in Auto-ML). [cite: 294] We introduce a compiler μ-two, that is designed to efficiently navigate the performance trade-off of compute utilization, memory consumption, and number of independent operations. [cite: 295] Augmented with lightweight profiling and static analysis, μ-two saturates compute through fusion, efficiently utilizes memory via swap/recompute, and maximally overlaps data movement with independent compute operations. [cite: 296] μ-two generates tailored training schedules for any given set of models and target GPUs. [cite: 297] Compared to the state-of-the-art approaches, μ-two enables concurrent training of $3-5\times$ more models with a memory footprint of up to 6x the GPU memory size and delivers a 3x speedup.

--- PAGE 12 ---

## [cite: 298] REFERENCES

### μ-TWO: 3× Faster Multi-Model Training with Orchestration and Memory Optimization

Nvidia. nvidia multi-instance gpu, 2020. URL https://www.nvidia.com/en-us/technologies/multi-instance-gpu/.

[cite: 299] Nvidia. multi-process service, 2020. URL https://docs.nvidia.com/deploy/mps/.

Aot autograd: Ahead of time tracing pytorch autograd engine, 2021. URL https://pytorch.org/functorch/stable/notebooks/aot_autograd_optimizations.html. [cite: 300]
Nvidia cuda profiling tools interface (cupti) - cuda toolkit, 2021. URL https://developer.nvidia.com/cupti.

Pytorch cuda memory statistics, 2022a. URL https://pytorch.org/docs/stable/generated/torch.cuda.memory_stats.html. [cite: 301]
Cuda semantics pytorch, 2022b. URL https://pytorch.org/docs/stable/notes/cuda.html.

Fake tensor, 2022. URL https://pytorch.org/torchdistx/latest/fake_tensor.html. [cite: 302]
Abadi, M., Barham, P., Chen, J., Chen, Z., Davis, A., Dean, J., Devin, M., Ghemawat, S., Irving, G., Isard, M., Kudlur, M., Levenberg, J., Monga, R., Moore, S., Murray, D. G., Steiner, B., Tucker, P., Vasudevan, V., Warden, P., Wicke, M., Yu, Y., and Zheng, X. Tensorflow: A system for large-scale machine learning. [cite: 303] In 12th USENIX Symposium on Operating Systems Design and Implementation (OSDI 16), pp. 265-283, 2016. URL https://www.usenix.org/system/files/conference/osdil6/osdil6-abadi.pdf. [cite: 304]
Bergstra, J. and Bengio, Y. Random search for hyper-parameter optimization. Journal of Machine Learning Research, 13(10):281-305, 2012. URL http://jmlr.org/papers/v13/bergstral2a.html. [cite: 305]
Bergstra, J., Bardenet, R., Bengio, Y., and Kégl, B. Algorithms for hyper-parameter optimization. [cite: 306] In NIPS, 2011.

Berner, C., Brockman, G., Chan, B., Cheung, V., Debiak, P., Dennison, C., Farhi, D., Fischer, Q., Hashme, S., Hesse, C., et al. [cite: 307] Dota 2 with large scale deep reinforcement learning. arXiv preprint arXiv: 1912.06680, 2019.

Chen, T., Li, M., Li, Y., Lin, M., Wang, N., Wang, M., Xiao, T., Xu, B., Zhang, C., and Zhang, Z. Mxnet: A flexible and efficient machine learning library for heterogeneous distributed systems, 2015.

Chen, T., Xu, B., Zhang, C., and Guestrin, C. Training deep nets with sublinear memory cost, 2016.

Coleman, C. A., Narayanan, D., Kang, D., Zhao, T., Zhang, J., Nardi, L., Bailis, P., Olukotun, K., Ré, C., and Zaharia, M. Dawnbench: An end-to-end deep learning benchmark and competition. [cite: 308] In Advances in Neural Information Processing Systems, 2017.

Devlin, J., Chang, M.-W., Lee, K., and Toutanova, K. Bert: Pre-training of deep bidirectional transformers for language understanding, 2019.

Elsken, T., Metzen, J. H., and Hutter, F. Neural architecture search: A survey. [cite: 309] ArXiv, abs/1808.05377, 2019.

Ganaie, M. A., Hu, M., Tanveer, M., and Suganthan, P. N. Ensemble deep learning: A review. [cite: 310] CORR, abs/2104.02395, 2021. URL https://arxiv.org/abs/2104.02395.

He, H. and Zou, R. functorch: Jax-like composable function transforms for pytorch. [cite: 311] https://github.com/pytorch/functorch, 2021.

Jain, A., Phanishayee, A., Mars, J., Tang, L., and Pekhimenko, G. Gist: Efficient data encoding for deep neural network training. [cite: 312] In 2018 ACM/IEEE 45th Annual International Symposium on Computer Architecture (ISCA), pp. 776-789, 2018. doi: 10.1109/ISCA.2018.00070. [cite: 313]
Jain, P., Jain, A., Nrusimha, A., Gholami, A., Abbeel, P., Gonzalez, J., Keutzer, K., and Stoica, I. Checkmate: Breaking the memory wall with optimal tensor rematerialization. [cite: 314] In Dhillon, I., Papailiopoulos, D., and Sze, V. (eds.), Proceedings of Machine Learning and Systems, volume 2, pp. 497-511, 2020. URL https://proceedings. [cite: 315] mlsys.org/paper/2020/file/084b6fbb10729ed4da8c3d3f5a3ae7c9-Paper.pdf.

Kingma, D. P. and Ba, J. Adam: A method for stochastic optimization, 2017.

Kwon, W., Yu, G.-I., Jeong, E., and Chun, B.-G. [cite: 316] Nimble: Lightweight and parallel gpu task scheduling for deep learning, 2020.

Liu, R., Krishnan, S., Elmore, A. J., and Franklin, M. J. Understanding and optimizing packed neural network training for hyper-parameter tuning, 2020.

Lukiyanov, M., Hua, G., Chauhan, G., and Dankel, G. Pytorch profiler, 2021. URL https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html.


--- PAGE 15 ---

## [cite: 345] μ-TWO: 3× Faster Multi-Model Training with Orchestration and Memory Optimization

# Α μ-TWO ALGORITHMS

## A.1 Profiling Attributes and Algorithm

*[Table A: Profiling and Scheduling attributes for nodes]* [cite: 346]

| Attribute      | Profiling Attributes Definition                                                               |
|----------------|-----------------------------------------------------------------------------------------------|
| rank           | the position of the node in the topological sort of the graph                                 |
| gtype          | type of graph this node belongs to [forward/backward]                                         |
| run\_time      | the run-time of the node in milliseconds                                                      |
| peak\_mem      | the peak memory usage in bytes                                                                |
| active\_mem    | the active memory usage in bytes (minimum required memory)                                    |
|                | **Scheduling Attributes Definition** |
| to\_offload    | list of nodes to be offloaded to host memory after this node is executed                        |
| to\_delete     | list of nodes to be deleted after this node is executed                                       |
| to\_prefetch   | list of nodes to be prefetched from the host memory before this node is executed              |
| to\_recompute  | list of nodes to be recomputed before this node is executed                                   |

*[Table B: Profiling, Swapping and Recomputation attributes for intermediate nodes (feature map tensors)]* [cite: 347, 353]

| Attribute           | Definition                                                                                   |
|---------------------|----------------------------------------------------------------------------------------------|
|                     | **Profiling Attributes** |
| inactive\_time      | the time duration elapsed between last forward access and first backward access              |
| swap\_time          | time required to swap the tensor to/fro host memory and device memory                        |
| memory\_size        | the size of the tensor in bytes                                                              |
| last\_fw\_access    | reference to the node that serves the last access of this tensor in the forward pass           |
| first\_bw\_access   | reference to the node that serves the first access to this tensor in the backward pass         |
| last\_bw\_access    | reference to the node that serves the last access of this tensor in the backward pass          |
|                     | **Attributes for swapping** |
| prefetch\_prompt    | the node that serves as the prefetch prompt if this tensor is swapped                        |
| active\_fw\_interval| the first and last nodes in the forward pass during which the tensor resides in memory       |
| active\_bw\_interval| the first and last nodes in the backward pass during which the tensor resides in memory        |
|                     | **Attributes for recomputation** |
| recomp\_srcs        | intermediate tensors that serve as sources if the tensor needs to be recomputed              |
| recomp\_graph       | the extracted sub-graph that needs to be executed to regenerate this tensor                  |
| recomp\_cnt         | the number of times this tensor needs to be recomputed during its lifetime                   |
| recomp\_time        | the time required to recompute this tensor from its current sources                          |
| total\_recomp\_time | the total time spent in recomputation of this tensor in its lifetime                         |
| recomp\_memory      | the peak memory required during a single recomputation of this tensor                        |
| recompute\_ratio    | memory\_size/total\_recomp\_time                                                             |

The steps for run-time profiling are shown in Algorithm A. For each node in the graph we collect memory consumption of the operation (line 11) and the end-to-end time required for it to complete by executing the operations in the graph one by one (lines 8-10). [cite: 348] Subsequent to the execution of an operation in the forward pass, we swap-out all the intermediate tensors, after their last use, to the CPU memory (lines 12-15). [cite: 349] Prior to the execution of an operation during backward pass, we swap-in all intermediate tensors, required for this operation, offloaded to the CPU memory back to the GPU memory (lines 4-7). [cite: 350] This allows us to profile computational graphs of models that exceed the GPU memory limit with the bare minimum assumption that the inputs, outputs and workspace of every operation must fit on the GPU memory in isolation. [cite: 351]

**[cite: 351] Algorithm A Run-time Profiler**

```
1: Input: graph
2: # Perfrom static data-flow analysis
3: for node in graph.nodes do
4:   for t in node.first_backward_uses do [cite: 352]
5:     swap_in(t)
6:     # Measure swap-in time here
7:   end for
8:   # Start run-time measurement
9:   Execute node [cite: 354]
10:  # End run-time measurement
11:  # Measure memory consumption here
12:  for t in node.last_forward_uses do
13:    swap_out(t)
14:    # Measure swap-out time here
15:  end for
16: end for
```

## A.2 Scheduling Policy Algorithm

The scheduling policy is outlined in Algorithm B, that we now explain in detail. [cite: 355] We first initialize `last_prompt` (line 2), that is the node in the backward graph at which the last swap-in was scheduled. [cite: 356] It is initialized to be the last node in the backward graph. [cite: 357] We choose the `swap_candidate` to be the intermediate tensor with largest `incative_time` (line 6) and calculate the swap overhead for this candidate (line 7). [cite: 358] The calculation for swap overhead is explained in Section 4.2.2. We then choose our recomputation candidate that has the maximum `recompute_ratio` and then calculate the recompute overhead for this candidate (lines 8-9), explained in detail in Section A.4. [cite: 359] We then make the decision to either swap `s_cand` or recompute `r_cand` (lines 10-18). [cite: 360] If a candidate is chosen to be swapped, then we set the `to_prefetch` attribute of its `prefetch_prompt`, and the `to_offload` attribute of its `last_fw_acesss` node to be this candidate (lines 11-12). [cite: 361] If the candidate is chosen to be

--- PAGE 16 ---

*[Algorithm B Scheduling Policy & Algorithm C Swap Overhead Calculation Tables]* [cite: 362]

```
Algorithm B Scheduling Policy

1: Input: candidate_set, mem_limit
2: init(last_prompt)
3: swaps = {}
4: recomps = {}
5: while candidate_set != ∅ do
6:   s_cand = max_idle_candidate(candidate_set)
7:   s_overhead, prompt_node = SwapOverhead(s_cand, last_prompt)
8:   r_cand = max_recomp_candidate(candidate_set)
9:   r_overhead = RecomputeOverhead(r_cand)
10:  if s_overhead < r_overhead then
11:    last_prompt = Swap(s_cand, prompt_node)
12:    swaps.add(s_cand)
13:    cand = s_cand
14:  else
15:    Recompute(r_cand)
16:    recomps.add(r_cand)
17:    cand = r_cand
18:  end if
19:  candidates.remove(cand)
20:  recomp_cnt = update_recomps(cand, recomps)
21:  update_candidates(cand, recomp_cnt, candidates)
22:  update_swap_prompts(swaps, candidates)
23:  mem_consumption = get_mem_consumption()
24:  if (mem_consumption - mem_limit) < 0 then
25:    break
26:  end if
27: end while
```

```
Algorithm C Swap Overhead Calculation

1: Input: swap_cand, last_prompt, reached_peak
2: bw_access = swap_cand.first_bw_access
3: swap_time = swap_cand.swap_time
4: r_time = get_recomp_time(bw_access)
5: swap_time -= r_time
6: if reached_peak then
7:   # Case 1(a): Swap happens during peak interval
8:   if bw_access.rank < last_prompt.rank then
9:     swap_overhead = swap_time
10:    return swap_overhead, bw_access
11:  else
12:    # Case 1(b): Swap happens during other swap
13:    rem_time = get_remaining_time(bw_access)
14:    swap_overhead = swap_time + rem_time
15:    return swap_overhead, bw_access
16:  end if
17: end if
18: # Cases 2, 3: Add forward graph nodes to overlap swap
19: fw_node = first_fw_node
20: while swap_time > 0 do
21:  add_forward_node(bw_access, fw_node)
22:  adjust_graph(bw_access, fw_node)
23:  mem_safe = check_mem_safety()
24:  if mem_safe then
25:    swap_time -= fw_node.run_time
26:    fw_node = fw_node.next
27:  else
28:    break
29:  end if
30: end while
31: # Cases 2, 3: Use backward graph nodes to overlap swap
32: if bw_access.rank < last_prompt.rank then
33:   prefetch_prompt = bw_access.prev
34: else
35:   prefetch_prompt = last_prompt
36: end if
37: while swap_time > 0 and not_reached_peak(prefetch_prompt) do
38:   r_time = get_recomp_time(prefetch_prompt)
39:   swap_time -= (prefetch_prompt.runtime + r_time)
40:   prefetch_prompt = prefetch_prompt.prev
41: end while
42: swap_overhead = swap_time
43: return swap_overhead, prefetch_prompt
```

recomputed then, we simply add it to the recomputation set. we process all of the recomputations together while graph rewriting. We account for the side-effects of this candidate on other candidates already chosen for swap or recompute (lines 20-23) and explain them in detail in Sections A.4 and A.5. Finally we obtain the peak memory consumption from the memory simulator and if it is lower than the memory limit we exit (lines 23-26).

### A.3 Swap Overhead Calculation

The detailed steps for swap overhead calculation are outlined in Algorithm C. The input to the algorithm is the candidate to be swapped (`swap_cand`), the last node in

## [cite: 363] μ-TWO: 3× Faster Multi-Model Training with Orchestration and Memory Optimization

the backward graph that was used as a prefetch prompt (`last_prompt`) and a flag that indicates that we have already reached the peak memory interval in swapping (`reached_peak`). [cite: 364] The peak memory interval is the set of consecutive operations in which the peak memory consumption exceeds GPU memory limit, we cannot perform any swaps in this interval since there is no memory left to allocate for the tensor being swapped in. Based on when the prefetch prompt enters the peak memory interval into the calculation of the swap overhead can be classified into the following cases:

1.  When the peak interval is already reached before scheduling the swap, we cannot overlap this swap with compute and the swap overhead is calculated based on whether

--- PAGE 17 ---

## [cite: 365] μ-TWO: 3 Faster Multi-Model Training with Orchestration and Memory Optimization

    (a) does not conflict with existing swap
    (b) conflicts with an existing swap

2.  When peak interval is not reached after scheduling swap, the swap can be completely overlapped by using:
    (a) forward pass operations only
    (b) mix of forward and backward pass operations or backward pass operations only

3.  When peak interval is reached while scheduling swap, the swap is partially overlapped by using a mix of forward and backward pass operations or backward pass operations only

We first obtain the node on the backward pass before which the swap should complete (`bw_access`) and the time required to swap the [cite: 366] candidate (lines 2-3). For Case 1(a), the swap overhead is the actual swap time (lines 7-10). [cite: 367] For case 1(b), the swap overhead is the actual swap time plus the remaining swap time of an existing in-flight swap that is already scheduled (lines 11-16). [cite: 368] If the peak memory interval is not reached, we first try to add nodes from the forward graph one by one before `bw_acess` to overlap the swap. [cite: 369] After adding a node from the forward graph we check if this actually reduces the peak memory consumption (using Memory Simulator, Section A.6), since adding a forward graph node comes at the cost of increased memory consumption. [cite: 370] If that is the case, only then we proceed and reduce the swap time by the forward node's computation time. [cite: 371] Case 2(a) happens if the swap time reaches zero (lines 19-31). [cite: 372] In case the swap time has not reached zero and we cannot use any more forward graph nodes to overlap, we try to see if we can use the backward graph nodes prior to `bw_acess` to overlap this swap. [cite: 373] We reduce the swap time by a backward node's computation time as we iterate in a reverse fashion through the backward pass graph. [cite: 374] Case 2(b) happens if the remaining swap time reaches zero, else we are left with some swap time that cannot be overlapped and we incur a swap overhead resulting in case 3 (lines 33-43). [cite: 375] The swap overhead calculation also takes into account any recomputation time that can be used to overlap the swaps (lines 4-5, 39-40), we explain how we do this in Section A.5. [cite: 376]

### [cite: 376] A.4 Recompute Overhead Calculation

We largely adapt the recomputation algorithm from (Peng et al., 2020) which is based on the Spark's RDD lineage (Zaharia et al., 2012). [cite: 377] A candidate might be recomputed either once when it is required or while recomputing some other candidate that requires it during its own recomputation. [cite: 378] Hence, the recomputation overhead for a candidate is calculated as the time required to recompute the candidate (`recomp_time`) multiplied by the number of times it will be recomputed (`recomp_cnt`) in its lifetime. [cite: 379] It is tracked as `total_recomp_time` (Algorithm D) and used to compute the recompute ratio for choosing candidates to recompute as explained Section 4.2.1.

*[Algorithm D Recomputation Overhead Table]* [cite: 380]

```
Algorithm D Recomputation Overhead

input recomp_cand
output r_overhead
1: return recomp_cand.total_recomp_time
```

[cite: 381] For each recomputation candidate we maintain a set of `recomp_srcs`, which denote the ancestor nodes of the candidates using which we can recompute the candidate. [cite: 382] When a candidate is chosen for recomputation, it may affect the candidates (1) that are already chosen for recomputation and/or (2) candidates that maybe be chosen for recomputation in future. [cite: 383] In Algorithm E (Case 1), we first iterate through the existing set of recomputations, if the chosen candidate (`cand`) is one of the recomputation sources (`rp.srcs`) of an existing recomputation (`rp`), then we remove it from `rp.srcs` and add the recomputation sources of the candidate (`cand.srcs`) to `rp.srcs` (lines 4-6). [cite: 384] We also count the number of times this candidate will be recomputed in its lifetime (line 7). [cite: 385] In Algorithm F (Case 2), `t` is the candidate chosen for recomputation. [cite: 386] We iterate through the list of future candidates (`cand`) and check if either (a) `t` exists in recomputation sources (`cand.srcs`) of the `cand` or (b) `cand` exists in recomputation sources (`t.srcs`) of `t`. [cite: 387] In Case 2(a), we remove `t` from `cand.srcs` and add `t.srcs` to `cand.srcs` (lines 6-7). [cite: 388] We then add the recomputation time of `t` to `cand`'s recomputation time (line 8). [cite: 389] We then calculate the number of times the candidate may be recomputed for the already chosen recomputations and accumulate that in its potential total recomputation time (lines 10-14). [cite: 390] For Case 2 (b), the potential total recomputation time is calculated as the number of times `t` is recomputed (`recomp_cnt`) multiplied by the recomputation time of `cand` (lines 17-19). [cite: 391] Finally we update the recomputation ratio of all the remaining candidates.

**[cite: 392] Algorithm E Updating existing recomputations.**

```
input recomps, cand
output recomp_cnt
1: recomp_cnt = 1
2: for rp in recomps do
3:   if cand in rp.recomp_sources then
4:     rp.recomp_srcs.remove(cand)
5:     rp.recomp_srcs.add(cand.recomp_srcs)
6:     rp.recomp_time += cand.recomp_time
7:     recomp_cnt += 1
8:   end if
9: end for
10: return recomp_cnt
```

--- PAGE 18 ---

*[Algorithm F Updating remaining candidates. Table]* [cite: 393]

```
Algorithm F Updating remaining candidates.

input t, recomp_count, candidates
1: for cand in candidates do
2:   if t in cand.recomp_srcs then
3:     if cand.first_bw_access in t.active_bw_interval then
4:       continue
5:     else
6:       cand.recomp_srcs.remove(t)
7:       cand.recomp_srcs.add(t.recomp_srcs)
8:       cand.recomp_time += t.recomp_time
9:       cand.total_recomp_time = cand.recomp_time
10:      for rp in recomps do
11:        if cand in rp.recomp_srcs then
12:          cand.total_recomp_time += cand.recomp_time
13:        end if
14:      end for
15:    end if
16:  end if
17:  if cand in t.recomp_srcs then
18:    cand.total_recomp_time = recomp_cnt * cand.recomp_time
19:  end if
20:  cand.updateRecomputeRatio()
21: end for
```

## [cite: 394] μ-TWO: 3× Faster Multi-Model Training with Orchestration and Memory Optimization

before this recomputation takes place. [cite: 395] Secondly, the recomputation time of this candidate can be used for overlapping already scheduled swaps if it lies in their prefetch interval or any future swaps while scheduling. [cite: 396] Both these effects are accounted for by Algorithm B (line 22) and Algorithm C (lines 4-5, 39-40).

## A.6 Memory Simulator

*[Algorithm G Memory Consumption Simulator Table]* [cite: 397]

```
Algorithm G Memory Consumption Simulator

1: Input: graph, static_mem
2: fw_inter_mem = 0
3: bw_inter_mem = graph.inter_mem
4: fw_active_mem = 0
5: bw_active_mem = 0
6: peak_mem = 0
7: for node in graph.nodes do
8:   if node.gtype == backward then
9:     bw_active_mem = node.active_mem
10:    for pnode in node.to_prefetch do
11:      bw_inter_mem += pnode.memory_size
12:    end for
13:    for rnode in node.to_recompute do
14:      bw_inter_mem += rnode.memory_size
15:    end for
16:    for tnode in node.first_backward_uses do
17:      bw_inter_mem -= tnode.memory_size
18:    end for
19:  end if
20:  if node.gtype == forward then
21:    fw_active_mem = node.active_mem
22:  end if
23:  current_mem = fw_active_mem + bw_active_mem + bw_inter_mem + fw_inter_mem - static_mem
24:  peak_mem = max(peak_mem, current_mem)
25:  if node.gtype == forward then
26:    for dnode in node.to_delete do
27:      fw_inter_mem -= dnode.memory_size
28:    end for
29:    for onode in node.to_offload do
30:      fw_inter_mem -= onode.memory_size
31:    end for
32:    for tnode in node.last_forward_uses do
33:      fw_inter_mem += tnode.memory_size
34:    end for
35:  end if
36: end for
37: return peak_mem
```

### [cite: 398] A.5 Effects of swap and recompute on one another

Let's define two intervals to understand the effects of swapping on recompute. [cite: 399] For an intermediate tensor we define `active_bw_interval` as the set of operations that occur between its first use (`first_bw_access`) and last use (`last_bw_access`) in the backward pass. [cite: 400] For a swapped tensor we define the `prefetch interval` as the set of operations that occur between it's prefetch start (`prefetch_prompt`) and `first_bw_access`. [cite: 401] We first discuss the effect of swapping on recomputation. If a tensor is chosen to be swapped, then it may affect any remaining candidates that might be chosen for recomputation in future. [cite: 402] If the swapped tensor is one of the recomputation sources of these candidates and their `first_bw_access` does not lie in the swapped tensor's `active_bw_interval`, then it cannot be used as a recomputation source and the candidate's recomputation sources need to be updated. [cite: 403] This is accounted for in Algorithm F (lines 3-16).

Now we explain the effect of recomputation on swapping. [cite: 404] If a candidate is chosen for recomputation then it needs its sources to be available in memory. [cite: 405] Firstly, if one of the remaining candidates is a source for this recomputation and is chosen for swapping in future then it must be made available

The memory simulation algorithm takes in the current state of schedule for a backward graph $BW_j$ and a forward graph $FW_i$ corresponding to the fused sub-arrays $FA_j$ and $FA_i$.

--- PAGE 19 ---

## [cite: 406] μ-TWO: 3× Faster Multi-Model Training with Orchestration and Memory Optimization

respectively. [cite: 407] It maintains five variables to simulate the memory consumption at any current step:

(a) `static_mem`: the memory occupied by weights and weight gradients. [cite: 408]
(b) `fw_inter_mem`: the memory occupied by the intermediate tensors, between their `last_fw_access` and end of forward pass, in $FW_i$.
(c) `fw_active_mem`: the active memory consumption during the forward pass excluding `fw_inter_mem`. [cite: 409]
(d) `bw_inter_mem`: the memory occupied by the intermediate tensors, between the beginning of backward pass to their prefetch prompts or `first_bw_access`, in $BW_j$.
(e) `bw_active_mem`: the active memory consumption during the backward pass excluding the `bw_inter_mem`. [cite: 410]

When an intermediate tensor is prefetched or recomputed its memory is added to the `bw_inter_mem` (lines 10-15). [cite: 411] When we encounter the `first_bw_acess` of an intermediate tensor, we subtract its memory from `bw_inter_mem` since it is accounted for in `bw_active_mem` (lines 16-18). [cite: 412] We then measure the current memory consumption as `current_mem = (b) + (c) + (d) + (e) - (a)` (line 23). [cite: 413] We subtract `static_mem` since it is already accounted for twice in (b) and (c). [cite: 414] Then we update the peak memory consumption (`peak_mem`) (line 24). [cite: 415] We subtract the memory of an intermediate tensors from `fw_inter_mem` that are deleted or swapped out after their `last_fw_acess` and add the ones that are retained (lines 26-34). [cite: 416] Finally, we return `peak_mem`.

# B IMPLEMENTATION DETAILS

## B.1 μ-two Implementation

### B.1.1 Horizontal fuser

The horizontal operator fusion is implemented in μ-two using PyTorch's vmap library (He & Zou, 2021). [cite: 417] It takes an array of models to be fused together, with the requirement that they must have identical architecture, and outputs a single fused model with horizontally fused operators. [cite: 418] The models can have different loss functions, weight initialization, batch size, learning rate etc.

### B.1.2 Graph tracer

After obtaining a fused sub-array of models we proceed to graph tracing. [cite: 419] We use PyTorch FX to represent computational graphs (Reed et al., 2022). FX provides tools for graph representation, modification, transformation and execution. [cite: 420] FX graphs are obtained by using a PyTorch library AOT Autograd (aot, 2021). [cite: 421] AOT Autograd records the forward and backward operations performed by the fused model using a sample mini-batch. [cite: 422] To enable tracing graphs of models having a memory footprint larger than the GPU memory capacity, we make use of PyTorch Fake Tensor Mode (fak, 2022). [cite: 423] Fake Tensors are initialized on a meta device, they contain no actual data and only have meta data information like data type, size, memory layout, stride etc. We use AOT Autograd under the Fake Tensor Mode to obtain the FX Graphs. [cite: 424]

### [cite: 424] B.1.3 Optimizers

In deep learning training, optimizers are used to update the model weights with weight gradients, following the backward pass. [cite: 425] Currently, neither the vmap library allows horizontally fusing the optimizer calls nor does AOT Autograd allow tracing of optimizer calls. [cite: 426] To circumnavigate this problem we provide a custom implementation of the SGD Optimizer using the point-wise multiply and add functions (Ruder, 2017). [cite: 427] We then batch these function calls using vmap and attach them to the weight gradients in the backward pass graph. [cite: 428] We do not implement other advanced optimizers like Adam in our work, as implementation of any optimizer is sufficient to establish a proof-of-concept of our scheduling mechanism (Kingma & Ba, 2017). [cite: 429]

### [cite: 429] B.1.4 Profiler

We extend PyTorch FX Interpreter to implement the μ-two profiler (Reed et al., 2022). [cite: 430] The FX Interpreter allows node by node execution of the FX Graphs. [cite: 431] We override the `run_node` method of the FX Interpreter and wrap the run call using the PyTorch Profiler context manager (Lukiyanov et al., 2021). [cite: 432] The PyTorch Profiler is a GPU profiling engine, built using Nvidia CUPTI APIs, and is able to capture GPU kernel events with high fidelity (nvi, 2021). [cite: 433] We extract the latency of the CUDA kernel calls made by each node to calculate its run-time to eliminate the host-side overhead of CUDA kernel launches. [cite: 434] For calculating the memory consumption we use the CUDA Memstats tool subsequent to each `run_node` call (cud, 2022a). [cite: 435] Finally, to measure the swap-out and swap-in times of the intermediate tensors we use CUDA Events to measure the Device-to-Host (D2H) and Host-to-Device (H2D) memcpy calls (cud, 2022b). [cite: 436] To optimize the memory allocation we use pinned memory buffers on the host side. [cite: 437] To get stable profiling measurements we warm-up the CUDA caching allocator using a warm-up run before actual profiling. [cite: 438]

### [cite: 438] B.1.5 Schedule interpreter

The Schedule interpreter creates three CUDA Streams to represent compute operation queue, CPU-GPU and GPU-CPU swapping operation queues (cud, 2022b). [cite: 439] The CUDA Streams provide a guarantee that all operations enqueued in a stream will be processed sequentially. However, it pro-

--- PAGE 20 ---

## [cite: 440] μ-TWO: 3× Faster Multi-Model Training with Orchestration and Memory Optimization

vides no guarantees for operations across streams. [cite: 441] To add ordering for operations across streams we uses CUDA events to create synchronization markers. [cite: 442] The CUDA Events API provides us with record and wait calls for each event. [cite: 443] A CUDA event can be recorded in one stream and can be waited upon in another stream. [cite: 444] This allows us to create operation ordering across different streams and controlled asynchronous processing of compute and swapping operations. [cite: 445] Like the profiler, the μ-two Schedule interpreter also extends the FX Interpreter. It identifies and enqueues nodes in appropriate streams. [cite: 446] Upon encountering a prefetch prompt it enqueues the prefetch operations in the CPU-GPU stream, upon encountering `last_fw_access` of swapped tensors it enqueues them in the GPU-CPU stream and all the compute and recompute operations are enqueued in the execution stream. [cite: 447] Finally to overcome the host-side kernel launch and memory allocation overhead we record the operations of the Schedule interpreter into CUDA Graphs and then just replay them (Nguyen et al., 2021). [cite: 448]

Algorithm H presents the detailed execution methodology. The input to the Schedule interpreter is the merged graph produced by the graph re-writer with scheduling hints. [cite: 449] It executes the graph node-by-node and does the following: It first checks if the node to be executed is the prefetch prompt for a tensor to be swapped-in, if yes, it adds a prefetch begin event in the execution stream. [cite: 450] It then waits for the event in the CPU-GPU stream and adds the swap-in operation for the tensor in the CPU-GPU stream. [cite: 451] It then adds a prefetch end event (lines 3-8). [cite: 452] Subsequently, it adds all the recomputation nodes in the execution stream, if any (lines 9-11). [cite: 453] Prior to the execution of the node, it waits for any prefetch end events for its inputs and then enqueues the operation in execution stream lines(13-16). [cite: 454] It then deletes any tensors that are to be recomputed (lines 18-20). [cite: 455] Finally, if there are any tensors to be swapped-out, and have their `last_fw_uses` at this node, then their swap-out operations are enqueued in the GPU-CPU stream (lines 21-25) by using appropriate events and waits. [cite: 456]

## [cite: 456] B.2 Baseline Implementation

### B.2.1 HFTA

We uses PyTorch's vmap library to implement HFTA-NoMemOpt (He & Zou, 2021). [cite: 457] If the want to train 8 models and let's say only 4 models can be concurrently trained and fused by HFTA at a time due to peak memory consumption exceeding the GPU memory capacity, we run it twice to reflect the total running time. [cite: 458] We note that, to the best of our knowledge no prior work applies memory optimization techniques to concurrently training models on a single GPU and hence it reflects the state-of-the-art baseline for evaluating relative performance. [cite: 459] Although HFTA provides an open source implementation, it requires manual changes

*[Algorithm H Execution Engine Table]* [cite: 460]

```
Algorithm H Execution Engine

1: for node in graph.nodes do
2:   if node.gtype == backward then
3:     for pnode in node.to_prefetch do
4:       Execution Stream: pnode.prefetch_begin.record()
5:       CPU-GPU Stream: wait(pnode.prefetch_begin)
6:       prefetch(pnode.cpu_ref)
7:       pnode.prefetch_end.record()
8:     end for
9:     for rnode in node.to_recompute do
10:      Execution Stream: execute(rnode.recomp_graph)
11:    end for
12:  end if
13:  for inp in node.input_nodes do
14:    wait(inp.prefetch_end)
15:  end for
16:  Execution Stream: execute(node)
17:  if node.gtype == forward then
18:    for dnode in node.to_delete do
19:      Execution Stream: delete(dnode)
20:    end for
21:    for onode in node.to_offload do
22:      Execution Stream: onode.offload_begin.record()
23:      GPU-CPU Stream: wait(onode.offload_begin)
24:      offload(onode)
25:    end for
26:  end if
27: end for
```

to the model source code, and manually converting all the operators in the base model to their horizontally fused version. [cite: 461] First, the HFTA operators are not exhaustive and do not cover all implementations. [cite: 462] The examples provided in HFTA represent only a sub-set of our workload models. [cite: 463] The vmap library presents a fully automated way of horizontally fusing operators across models and hence we use that to reflect HFTA performance. [cite: 464] We observe that vmap sometimes introduces additional transpose and cloning operations which may cause it to be slower than the original HFTA implementation. [cite: 465] However, firstly we use the same library for implementing horizontal fusion for μ-two (Appendix B.1.1). [cite: 466] Secondly, our scheduling algorithm is independent of the fusion strategy. [cite: 467] Assuming that original HFTA implementation is faster, that will cause the compute latency to be lower and will provide fewer opportunities to overlap swaps. [cite: 468] At the same time lower compute latency implies lower recompu-

--- PAGE 21 ---

## [cite: 470] μ-TWO: 3× Faster Multi-Model Training with Orchestration and Memory Optimization

tation cost, hence our scheduling policy will automatically choose more tensors to recompute to balance this. [cite: 471] Thirdly, another significant difference between HFTA and vmap is that HFTA allows horizontal fusion of optimizers while vmap does not. [cite: 472] We explain how we circumnavigate this limitation in Appendix B.1.3. Finally, HFTA open-source implementation is not fully composable with other PyTorch components. [cite: 473]

### [cite: 473] B.2.2 HFTA-Capuchin

Capuchin does not provide an open-source implementation, and hence, we thoroughly implement Capuchin and then extend their algorithm to work with HFTA and call it HFTA-Capuchin. [cite: 474]

# C ADDITIONAL EXPERIMENTAL DETAILS

## C.1 Experimental Setup Details

We conduct our experiments on the latest powerful NVIDIA GPUs. [cite: 475] The first machine is the top-end AWS pd24-xlarge instance, having A-100 GPU with 40 GB of high bandwidth memory. [cite: 476] It is connected to the host machine via full duplex PCI-e gen4 interconnect offering upto 32GB/s bidirectional transfer speed. [cite: 477] The host memory size is 1152 GBs, shared across 8 GPUs. [cite: 478] We only make use of 1/8th host memory respecting the proportion per GPU. [cite: 479] Our second machine is the Dell Claudron Server, featuring Tesla V-100 GPU with 32 GB high bandwidth memory. [cite: 480] It uses the same interconnect to the host machine as above. [cite: 481] The host memory size is 384 GBs divided across 4 GPUs and we make use of 1/4th the host memory. [cite: 482] Note that A100 has more compute capability and a larger GPU memory capacity than V100. [cite: 483]

*[Table C: We experiment with two diverse hardware setups.]* [cite: 484]

| Instance                  | Nvidia GPU Version | GPU Mem (GB) | Tensor Cores | CPU-GPU Link             | CPUs | Host Mem (GB) |
|---------------------------|--------------------|--------------|--------------|--------------------------|------|---------------|
| AWS p4d24-large           | A-100              | 40           | Yes          | PCI-e Gen 4 x16 (32GB/s) | 16   | 1152          |
| Dell Claudron DSS 8440    | Tesla V-100        | 32           | Yes          | PCI-e Gen 4 x16 (32GB/s) | 16   | 384           |

## [cite: 485] D ADDITIONAL RELATED WORK

**Other Scheduling Approaches:** Nimble scheduler optimizes for executing the computation graph of a single model in parallel by partitioning the independent paths in the graph across different GPU streams (Kwon et al., 2020), while Hivemind runtime does the same for a multi-model execution graph (Narayanan et al., 2018). [cite: 486] The goal of Nimble and Hivemind differ from μ-two since they use concurrent kernel execution to improve compute utilization for small kernels whereas μ-two uses multiple streams for overlapping data

*[Table D: μ-two outperforms all multi-model training techniques.]* [cite: 487]

| Feature                 |                        | HFTA | μ-two |
|-------------------------|------------------------|------|-------|
| Out-of-memory support   |                        | No   | Yes   |
| Parameters essential for hardware utilization | large minibatch size   | No   | Yes   |
|                         | Large model size       | No   | Yes   |
| Hardware utilization    | Large number of models | Yes  | Yes   |
|                         | High Memory utilization| No   | Yes   |
|                         | High Compute utilization | Yes  | Yes   |

*[Table E: μ-two achieves low overhead amongst all out-of-memory approaches.]* [cite: 488, 489]

| Technique   | Out of Memory Strategy | Compute Overhead | Stalling |
|-------------|------------------------|------------------|----------|
| VDNN        | Swapping               | None             | High     |
| Checkmate   | Recomputation          | High             | None     |
| Capuchin    | Hybrid                 | High             | Low      |
| μ-two       | Hybrid                 | Low              | None     |

transfers with compute for Out-of-memory approaches for large kernels (which already have high compute utilization due to fusion). [cite: 490] Further, Nimble optimizes the Kernel launch overhead by pre-allocation of memory, μ-two does the same using the CUDAGraph API in Pytorch (Nguyen et al., 2021). [cite: 491] All other schedulers assume that the model/s fit on device memory hence only schedule the compute and not the data transfers. [cite: 492]

**Other Hardware Sharing Approaches:** MPS allows CUDA kernels from different processes to potentially run concurrently on the same GPU via a hardware feature called Hyper-Q (mps, 2020). [cite: 493] MIG, which is currently only available on the most recent A100 GPUs, partitions a single GPU into multiple (up to 7) isolated GPU instances (GIs) where each job now run on a single GI (mig, 2020). [cite: 494] HFTA has already shown better performance than MIG and MPS since they both do not horizontally fuse operators across neural networks. [cite: 495] Further they have a high memory footprint, no memory optimization, restriction on the number of independent processes, and no multiplexing across processes. [cite: 496] Since neural network operations are highly deterministic, repetitive, and exhibit very specific memory usage patterns, it is important that schedulers make use of this information to drive the GPU utilization to high levels for achieving high efficiency, lowering training cost, and enabling the training of models larger than GPU. [cite: 497]

## [cite: 498] E FUTURE WORK

The question we address in this paper is when several models with identical architecture being trained, how can we

--- PAGE 22 ---

## [cite: 499] μ-TWO: 3× Faster Multi-Model Training with Orchestration and Memory Optimization

perform horizontal fusion for addressing compute underutilization and use memory optimization techniques to address memory limitation? [cite: 500] This question maps to numerous critical problems such as hyper-parameter tuning, ensemble learning, and neural architecture search which are typical cases where the model architecture stays the same but soft hyperparameters like learning rate, learning rate decay momentum, loss functions, and weight initializations need to vary. [cite: 501] However, training multiple models with heterogeneous architectures is also an exciting problem to pursue. [cite: 502] Firstly, if all models in a batch have different architectures, then horizontal fusion is not possible. [cite: 503] Secondly, technically only each sub-array should have identical architecture, and it can vary across sub-arrays. [cite: 504] But if we have multiple instances of different architectures, the problem of finding the sub-array splits itself explodes combinatorially and is an interesting direction for future work.
```