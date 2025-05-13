Below is a concise “contract” you can pin to your repo / project board.
The first table distills **what you must hand-in** for Stage 1 and Stage 2; the second table is a **tick-box QA checklist** you (or your TA) can run through to prove the implementation is correct and complete.

---

### 1. Required deliverables

| Stage                                      | What you build                                                                                                                                                                                                                                                                                                                                                                        | What you must submit                                                                                                                                                                                                                                                                                          | Where this is stated                                       |
| ------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------- |
| **1 — Graph Profiler**                     | • A PyTorch-based **computation-graph profiler** that:<br>  1. traces every op in a forward-backward-optimizer iteration;<br>  2. records per-op runtime & memory;<br>  3. tags each tensor as parameter / gradient / activation / optimiser-state / other;<br>  4. performs static first-use / last-use analysis on activations;<br>  5. produces a **peak-memory breakdown graph**. | 1. Profiler source code & unit tests<br>2. README (build + run instructions)<br>3. **CSV/JSON** with the collected metrics for both reference models (ResNet-152 & BERT-Base)<br>4. Image (e.g. PNG) of the peak-memory breakdown graph<br>5. Slide/demo clip showing profiler output on a mini-batch         | Project description — Phase 1 requirements & deliverables  |
| **2 — Activation Checkpointing algorithm** | • A **µ-TWO-style AC module** that, given profiler stats, greedily decides which activations to keep vs. discard/recompute and integrates with training.<br>• Works for both reference models.                                                                                                                                                                                        | 1. AC implementation & tests<br>2. README explaining activation-selection heuristic<br>3. **Bar chart**: peak memory vs. mini-batch size *with and without AC*<br>4. **Line chart**: iteration latency vs. mini-batch size *with and without AC*<br>5. Short demo video / gif of AC saving memory at run-time | Project description — Phase 2 + overall deliverables list  |

---

### 2. Verification checklist (tick every box before calling a stage “done”)

| ✔                                                                                                                                                              | **Stage 1 — Graph Profiler** |
| -------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------- |
| ☑ **Trace completeness**: Graph includes *all* forward, backward **and optimiser** ops for one iteration.                                                      |                              |
| ☑ **Node metadata**: For every op, `runtime_ms`, `active_mem_B`, and `peak_mem_B` are logged and values look sensible (spot-checked against `torch.profiler`). |                              |
| ☑ **Tensor labelling**: 100 % of tensors are classified into parameter / gradient / activation / optimiser-state / other.                                      |                              |
| ☑ **Static analysis**: For several random activations, ‘first-fw-use’ and ‘last-bw-use’ indices match manual inspection.                                       |                              |
| ☑ **Peak-memory plot** renders and shows the characteristic hump (grows through fwd, falls through bwd).                                                       |                              |
| ☐ Unit test: Running the profiler on a toy 3-layer MLP produces exactly 3 forward + 3 backward nodes and the expected memory curve.                            |                              |
| ☐ Code passes lint, README reproduces results in < 5 min on reference GPU.                                                                                     |                              |

| ✔                                                                                                                                                   | **Stage 2 — Activation Checkpointing** |
| --------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------- |
| ☑ **Uses profiler outputs** (does *not* hard-code layer indices).                                                                                   |                                        |
| ☐ **Correctness**: With AC on, loss and gradients (up to 1e-6 relative error) match the baseline no-AC run on both models.                          |                                        |
| ☐ **Memory saving**: Peak GPU memory drops by ≥ X % (set your target, e.g. ≥ 30 %) on a  batch size that previously OOM’d.                          |                                        |
| ☐ **Latency accounting**: Iteration time overhead (recompute + swap) is measured and plotted; numbers match those in the latency graph deliverable. |                                        |
| ☐ **Graphs generated**: Required bar/line charts appear in `reports/` and are referenced in the README.                                             |                                        |
| ☑ **Edge cases**: AC disabled itself gracefully when profiler shows activations already fit into memory.                                            |                                        |
| ☐ Unit test: Turning AC on for the toy MLP keeps memory < baseline and runs without error.                                                          |                                        |
| ☐ Code & docs pass review; demo script reproduces memory + latency graphs in < 10 min.                                                              |                                        |

---

Meet every deliverable and tick every checklist item, and you can confidently claim **Stage 1 and Stage 2 are complete**.
