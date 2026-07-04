"""
Data preparation pipeline for Fine-Tune vs. RAG Decision Lab.

Downloads the medical flashcards dataset, splits it 80/20,
and saves processed data for both the fine-tuning and RAG pipelines.
"""

import json
from pathlib import Path

from datasets import load_dataset

# --- Configuration ---
DATASET_NAME = "medalpaca/medical_meadow_medical_flashcards"
SPLIT_RATIO = 0.8  # 80% knowledge, 20% evaluation
RANDOM_SEED = 42  # Fixed seed for reproducibility

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_EVAL = PROJECT_ROOT / "data" / "eval"

def format_for_finetuning(example: dict) -> dict:
    """Convert a raw example to instruction format for LoRA training."""
    return {
        "instruction": example["instruction"],
        "input": example["input"],
        "output": example["output"],
    }


def format_for_rag(example: dict) -> dict:
    """Convert a raw example to a document for RAG ingestion.

    Combines question and answer into a single searchable document.
    """
    return {
        "content": f"Question: {example['input']}\nAnswer: {example['output']}",
        "metadata": {
            "source": "medical_flashcards",
            "question": example["input"],
        },
    }


def format_for_eval(example: dict) -> dict:
    """Convert a raw example to evaluation format.

    Stores the question and the ground truth answer separately.
    """
    return {
        "question": example["input"],
        "ground_truth": example["output"],
        "instruction": example["instruction"],
    }

def is_valid_example(example: dict) -> bool:
    """Check if an example has non-empty input and output fields."""
    return bool(example["input"].strip() and example["output"].strip())

def deduplicate_by_input(dataset):
    """Remove duplicate questions, keeping only the first occurrence."""
    seen = set()
    indices_to_keep = []
    for i, example in enumerate(dataset):
        question = example["input"].strip().lower()
        if question not in seen:
            seen.add(question)
            indices_to_keep.append(i)
    return dataset.select(indices_to_keep)

def save_jsonl(data: list[dict], filepath: Path) -> None:
    """Save a list of dictionaries as a JSON Lines file."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
    print(f"  Saved {len(data)} examples to {filepath}")

def prepare_data() -> None:
    """Main data preparation pipeline."""
    print("=" * 60)
    print("DATA PREPARATION PIPELINE")
    print("=" * 60)

    # Step 1: Download dataset
    print(f"\n[1/6] Downloading dataset: {DATASET_NAME}")
    ds = load_dataset(DATASET_NAME)
    full_data = ds["train"]
    print(f"  Total examples: {len(full_data)}")

    # Step 2: Clean data — remove empty/invalid records
    print("\n[2/6] Cleaning data — removing empty records")
    original_count = len(full_data)
    full_data = full_data.filter(lambda ex: is_valid_example(ex))
    removed = original_count - len(full_data)
    print(f"  Removed {removed} invalid records ({removed/original_count:.1%})")
    print(f"  Remaining: {len(full_data)} clean examples")

    # Step 3: Deduplicate by question text
    print("\n[3/6] Deduplicating by question text")
    pre_dedup = len(full_data)
    full_data = deduplicate_by_input(full_data)
    dupes = pre_dedup - len(full_data)
    print(f"  Removed {dupes} duplicate questions ({dupes/pre_dedup:.1%})")
    print(f"  Remaining: {len(full_data)} unique examples")

    # Step 4: Split into knowledge (80%) and eval (20%)

    print(f"\n[4/6] Splitting data ({SPLIT_RATIO:.0%} knowledge / {1-SPLIT_RATIO:.0%} eval)")
    split = full_data.train_test_split(test_size=1 - SPLIT_RATIO, seed=RANDOM_SEED)
    knowledge_set = split["train"]
    eval_set = split["test"]
    print(f"  Knowledge set: {len(knowledge_set)} examples")
    print(f"  Evaluation set: {len(eval_set)} examples")

    # Step 5: Format and save training data (for fine-tuning)
    print("\n[5/6] Formatting and saving processed data")
    finetune_data = [format_for_finetuning(ex) for ex in knowledge_set]
    save_jsonl(finetune_data, DATA_PROCESSED / "train_finetune.jsonl")

    # Save RAG documents (for vector DB ingestion)
    rag_data = [format_for_rag(ex) for ex in knowledge_set]
    save_jsonl(rag_data, DATA_PROCESSED / "rag_documents.jsonl")

    # Step 6: Format and save evaluation data
    print("\n[6/6] Saving evaluation set")
    eval_data = [format_for_eval(ex) for ex in eval_set]
    save_jsonl(eval_data, DATA_EVAL / "eval_set.jsonl")

    # Summary
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  Fine-tuning data: {DATA_PROCESSED / 'train_finetune.jsonl'}")
    print(f"  RAG documents:    {DATA_PROCESSED / 'rag_documents.jsonl'}")
    print(f"  Evaluation set:   {DATA_EVAL / 'eval_set.jsonl'}")

if __name__ == "__main__":
    prepare_data()
