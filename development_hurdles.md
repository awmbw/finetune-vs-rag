# Project 2: Fine-Tune vs. RAG Decision Lab — Development Hurdles

> This document tracks all technical hurdles, blockers, debugging sessions, and lessons learned during the project build. Reference this for interview stories (STAR/PAR format).

---

## Legend
- 🔴 **Blocker** — Stopped progress entirely
- 🟡 **Challenge** — Slowed us down but we found a workaround
- 🟢 **Resolved** — Fixed and documented
- 💡 **Lesson Learned** — Key takeaway for interviews

---

## Hurdle #1 — Build Backend Import Error

🟢 **Resolved**

**Problem:** `pip install -e ".[dev]"` failed with `BackendUnavailable: Cannot import 'setuptools.backends._legacy'`

**Root Cause:** Incorrect `build-backend` path in `pyproject.toml`. Used `setuptools.backends._legacy:_Backend` instead of the correct PEP 517 backend `setuptools.build_meta`.

**Fix:** Changed `build-backend` to `"setuptools.build_meta"` in `pyproject.toml`.

💡 **Lesson Learned:** The `build-backend` field in `pyproject.toml` follows the PEP 517 standard. Always use `setuptools.build_meta` for setuptools-based projects. The error message "BackendUnavailable: Cannot import X" always means this field is wrong.

---

## Hurdle #2 — Dirty Source Data (Empty Records)

🟢 **Resolved**

**Problem:** Automated validation (`validate.py`) flagged hundreds of records with empty `input` and/or `output` fields across both training and evaluation sets. Source dataset (`medalpaca/medical_meadow_medical_flashcards`) contained 408 blank flashcard entries (~1.2% of total data).

**Impact:** Training on empty records teaches the model to output nothing. Evaluating on empty records produces meaningless accuracy metrics. Empty strings also caused false positives in the data leakage check (empty `""` == empty `""` counted as overlap).

**Root Cause:** The upstream HuggingFace dataset was not pre-cleaned. Real-world datasets are rarely clean — this is standard in production ML.

**Fix:** Added `is_valid_example()` filter function in `prepare.py` that strips whitespace and rejects records where `input` or `output` are empty. Applied as a cleaning step BEFORE the train/eval split.

💡 **Lesson Learned:** Never trust source data blindly. Always build automated validation into your data pipeline — it's the difference between a hobby project and production-grade ML. Google's internal ML pipelines have dozens of data validation checks (see: TensorFlow Data Validation / TFDV).

**STAR Format:** *Discovered 408 corrupt records (1.2%) in source medical dataset through automated validation pipeline, engineered data cleaning filter preventing garbage-in-garbage-out corruption of both LoRA training and RAG indexing pipelines.*

---

## Hurdle #3 — Data Leakage from Duplicate Questions

🟢 **Resolved**

**Problem:** After cleaning empty records, validation still flagged 76 evaluation questions that also appeared in the training set. This is **data leakage** — the model could memorize exact answers during training and appear artificially accurate on evaluation.

**Impact:** Data leakage inflates accuracy metrics, making the model appear better than it actually is. Any ML paper or interview panel would immediately reject results with known leakage.

**Root Cause:** The source dataset contains duplicate questions (same question appearing multiple times with identical or slightly different answers). The random 80/20 split placed some copies in train and others in eval.

**Fix:** Added deduplication step in `prepare.py` — removes duplicate questions (case-insensitive, whitespace-normalized) BEFORE the train/eval split, keeping only the first occurrence. This ensures each unique question exists in only one split.

💡 **Lesson Learned:** Data leakage is the #1 silent killer of ML experiments. Always validate that evaluation data is completely isolated from training data. Deduplication must happen BEFORE splitting, not after. This is a common interview question: *"How did you prevent data leakage in your evaluation?"*

**STAR Format:** *Identified 76 data leakage instances between train/eval splits through automated validation harness, implemented deduplication pipeline reducing 33,955 raw records to clean, leak-free datasets ensuring scientifically valid benchmarking results.*

---

## Hurdle #4 — Mixed Precision Type Mismatch (BFloat16 vs FP16)

🟢 **Resolved**

**Problem:** Training crashed immediately with `NotImplementedError: "_amp_foreach_non_finite_check_and_unscale_cuda" not implemented for 'BFloat16'`.

**Impact:** Unable to begin the training loop.

**Root Cause:** Mixed precision type mismatch between the QLoRA configuration and the Trainer configuration. The model was quantized with `bnb_4bit_compute_dtype=torch.bfloat16`, but the HuggingFace `SFTTrainer` was initialized with `fp16=True`. The PyTorch Automatic Mixed Precision (AMP) scaler attempted to unscale `bfloat16` gradients using an `fp16` function, causing the crash.

**Fix:** Changed `fp16=True` to `bf16=True` in the `SFTConfig` arguments inside `train.py`. Since the RTX 5060 uses the modern Ada Lovelace architecture, it natively supports `bfloat16`, which is superior for training stability compared to standard `fp16`.

💡 **Lesson Learned:** Always ensure your compute data types match across all layers of the training stack (Model loading, QLoRA config, and Trainer config). `bfloat16` should be the default for all modern GPUs (Ampere architecture and newer).

**STAR Format:** *Diagnosed and resolved a complex Automatic Mixed Precision (AMP) type mismatch between QLoRA gradient configurations and HuggingFace SFTTrainer arguments, preventing an immediate training crash and ensuring stable bfloat16 optimization on consumer hardware.*

---

## Hurdle #5 — TRL/Transformers API Breaking Changes (3 cascading failures)

🟢 **Resolved**

**Problem:** Training script crashed three times in a row with `TypeError` errors — `max_seq_length` was not recognized, `warmup_ratio` was deprecated, and `tokenizer` was no longer a valid keyword argument for `SFTTrainer`.

**Impact:** Unable to initialize the training loop despite correct model loading and data preparation.

**Root Cause:** We installed `trl==1.7.1` and `transformers==5.13.0`, which are bleeding-edge releases. Most online tutorials and documentation reference older API signatures. The library maintainers renamed several arguments between versions:
- `max_seq_length` → `max_length`
- `warmup_ratio` → `warmup_steps`
- `tokenizer` → `processing_class`

**Fix:** Updated all three arguments in `train.py` to match the current API. Each fix was iterative — run, read the error, fix, repeat.

💡 **Lesson Learned:** ML libraries (especially HuggingFace ecosystem) move extremely fast. Never blindly copy training code from tutorials — always check the installed library version and cross-reference the current API docs. Pin your dependency versions in `pyproject.toml` for reproducibility.

**STAR Format:** *Resolved three cascading API compatibility failures across the HuggingFace TRL and Transformers training stack by systematically reading error messages and updating deprecated function signatures, enabling successful QLoRA training on bleeding-edge library versions.*

---

## Hurdle #6 — Model Merge Crash (Meta Device Offloading + Tied Weights Bug)

🟢 **Resolved**

**Problem:** After successful training, the merge script (`merge.py`) crashed during `save_pretrained()` with `AttributeError: 'list' object has no attribute 'keys'`. Additionally, warnings indicated model parameters were on a "meta device" due to offloading.

**Impact:** Unable to save the merged model, blocking the entire downstream pipeline (evaluation, serving).

**Root Cause:** Two compounding issues:
1. **Meta Device Offloading:** Using `device_map="auto"` attempted to load the full 7.6GB bfloat16 model onto the 8GB GPU. Insufficient VRAM caused PyTorch to offload layers to a virtual "meta device," corrupting the save process.
2. **Transformers 5.x Bug:** The `_get_tied_weight_keys()` function in `transformers==5.13.0` expects tied weight keys to be a `dict`, but Phi-3.5's custom model code returns a `list`, causing the `AttributeError`.

**Fix:** 
- Removed `device_map="auto"` to load the model entirely on CPU RAM (22GB available, only ~7.6GB needed). GPU is not required for weight merging.
- Patched `merged_model._tied_weights_keys = None` to bypass the tied weights serialization bug.
- Added `max_shard_size="2GB"` to save in smaller chunks, reducing peak memory during serialization.

💡 **Lesson Learned:** Not every step in an ML pipeline needs a GPU. Merge operations are pure linear algebra on weight matrices — CPU RAM is sufficient and avoids GPU memory contention. When encountering framework bugs in bleeding-edge versions, reading the traceback and applying targeted monkey-patches is a valid production strategy.

**STAR Format:** *Diagnosed and resolved a compound failure during model weight merging caused by GPU memory offloading and a framework serialization bug in transformers 5.x, implementing CPU-only loading and a targeted monkey-patch to successfully produce a standalone fine-tuned Medical LLM.*

---

## Hurdle #7 — DynamicCache Incompatibility During Inference (KV Cache Bug)

🟢 **Resolved**

**Problem:** RAG generation crashed with `AttributeError: 'DynamicCache' object has no attribute 'seen_tokens'` when calling `model.generate()`.

**Impact:** Blocked the entire RAG generation pipeline — retrieval worked but the model could not produce any output.

**Root Cause:** Phi-3.5's custom model code (downloaded via `trust_remote_code=True`) references `past_key_values.seen_tokens` for KV cache management. In `transformers==5.13.0`, the `DynamicCache` class was refactored and this attribute was renamed/removed, breaking the custom code's assumptions.

**Fix:** Added `use_cache=False` to the `model.generate()` call. This disables KV caching entirely during generation, bypassing the incompatible `DynamicCache` code path. The tradeoff is slightly slower token-by-token generation, but for short medical answers (≤256 tokens) the difference is negligible.

💡 **Lesson Learned:** Using `trust_remote_code=True` downloads and executes Python code from the model's HuggingFace repository. This custom code may be outdated relative to the installed `transformers` version, creating version skew bugs. In production, pin both the model revision AND the library version.

**STAR Format:** *Resolved a KV cache incompatibility between Phi-3.5's custom model code and transformers 5.x by disabling cache during inference, unblocking the RAG generation pipeline while maintaining answer quality.*

---
