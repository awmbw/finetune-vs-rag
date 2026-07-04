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
