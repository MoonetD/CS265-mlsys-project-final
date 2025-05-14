# Stage 1 Completion Checklist

This checklist tracks all tasks needed to complete Stage 1. Check off each item as you complete it.

## 1. GraphProfiler Instrumentation (`graph_prof.py`)

- [ ] **1.1 Track "active" memory before & after each op**
  - Add `pre_mem = torch.cuda.memory_allocated()` at start of `run_node()`
  - Store `active_mem_bytes = pre_mem` in a new dict `self.active_mem_node`
  - *Validation*: After a profile run, `max(active_mem_node.values())` should equal `torch.cuda.max_memory_allocated()` observed by `nvidia-smi` (± a few MiB)

- [ ] **1.2 Tag optimizer-state tensors**
  - Add `OPT_STATE` to `NodeType` enum
  - Detect optimizer moments: any node whose `target` matches `aten._fused_*` or has `"adam"`/`"sgd"` in its name
  - Update `node_types` accordingly
  - *Validation*: Verify `len([n for n,t in node_types.items() if t==NodeType.OPT_STATE]) > 0` for Adam

- [ ] **1.3 Capture parameter & gradient sizes**
  - In `__init__`, compute `self.param_sizes[name]` once
  - During first backward pass (detect via gtype == "backward"), record `grad_sizes`
  - *Validation*: Sum of `param_sizes` ≈ `230 MiB` for ResNet-152; `sum(grad_sizes)` within 5-10% of param

- [ ] **1.4 Handle non-tensor / tuple outputs**
  - Replace simple tensor check with proper flattening:
    ```python
    for t in torch.utils._pytree.tree_flatten(result)[0]:
        if isinstance(t, torch.Tensor) and t.device.type=='cuda':
            mem_size += t.nelement()*t.element_size()
    ```
  - *Validation*: Run profiler on a layer that returns `(out, indices)`—`median_memory_sizes[node]` must be > 0

- [ ] **1.5 Write active memory & node device into CSV**
  - Add `median_active_mem_bytes` and `device` columns
  - *Validation*: CSV row shows non-zero active mem and correct device string (`cuda:0`)

- [ ] **1.6 Docstring & type-hint cleanup** (optional)
  - *Validation*: `pylint graph_prof.py` score ≥ 8

## 2. Memory-breakdown Accuracy (`batch_memory_analysis.py`)

- [ ] **2.1 Compute weights/grad/activation bytes from CSV, not constants**
  - Replace hard-coded values with:
    ```python
    weight_mem = df[df.node_type=='parameter'].median_peak_mem_bytes.sum()/2**20
    grad_mem = df[df.node_type=='gradient'].median_peak_mem_bytes.sum()/2**20
    activation_mem = peak_memories_mib[i] - weight_mem - grad_mem
    ```
  - *Validation*: Bars stack exactly to the height printed for `peak_memories_mib[i]`

- [ ] **2.2 Remove BERT placeholder note** (since you're skipping it)

- [ ] **2.3 Latency-vs-batch plot** (optional)
  - In loop, store `iter_time = sum(graph_profiler.median_run_times.values())`
  - Plot batch size vs `iter_time`
  - *Validation*: Curve should increase roughly linearly

## 3. CSV / Plot Generation Robustness

- [ ] **3.1 Fail-fast if CSV missing**
  - Instead of silent placeholder creation, `raise RuntimeError` to surface Stage-1 errors early
  - *Validation*: Intentionally delete a CSV → script should stop with clear error

- [ ] **3.2 One-click "stage1_validate.sh"**
  - Create shell script that runs `python batch_memory_analysis.py`, checks for errors
  - *Validation*: CI job passes when all checks pass

## 4. Unit / Sanity Tests (`tests/test_profiler.py`)

- [ ] **4.1 `test_activation_liveness()`**
  - Assert `first_bw_use_rank > last_fw_use_rank` for every activation
  - Runtime < 5s

- [ ] **4.2 `test_memory_conservation()`**
  - Assert `sum(param+grad+peak_act) ≈ max_node_peak ±5%`
  - Runtime < 5s

- [ ] **4.3 `test_csv_schema()`**
  - Assert required headers present, no NaNs
  - Runtime < 1s

## 5. Documentation

- [ ] **5.1 Stage-1 README section**
  - Create `docs/stage1.md` with 1-page overview of profiler design & how to reproduce graphs

- [ ] **5.2 Update deliverables checklist**
  - Mark all completed items in the documentation

## Final Validation Run (Sign-off)

1. Run: `CUDA_VISIBLE_DEVICES=0 python batch_memory_analysis.py --batch-sizes 4 8 16 32 64`
2. Verify all checks pass in console summary
3. Confirm `reports/` contains five PNGs + two CSVs per batch
4. Push & tag commit as **stage1-final**

When the TA runs the validation on a V100 (or your local 4090), they should reproduce numbers within ±10 MiB and see no assertion failures.
