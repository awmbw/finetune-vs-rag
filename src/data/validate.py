"""
Data validation for Fine-Tune vs. RAG Decision Lab.

Runs quality checks on processed data to catch issues
before they propagate to training or evaluation.
"""

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_EVAL = PROJECT_ROOT / "data" / "eval"


def load_jsonl(filepath: Path) -> list[dict]:
    """Load a JSON Lines file into a list of dictionaries."""
    with open(filepath) as f:
        return [json.loads(line) for line in f]


def validate_no_empty_fields(data: list[dict], name: str) -> bool:
    """Check that no fields are empty or None."""
    issues = 0
    for i, item in enumerate(data):
        for key, value in item.items():
            if isinstance(value, str) and not value.strip():
                print(f"  ❌ {name}[{i}]: empty field '{key}'")
                issues += 1
            elif value is None:
                print(f"  ❌ {name}[{i}]: None field '{key}'")
                issues += 1
    if issues == 0:
        print(f"  ✅ {name}: no empty fields")
    return issues == 0


def validate_no_data_leakage(
    train_data: list[dict], eval_data: list[dict]
) -> bool:
    """Check that no evaluation questions appear in the training set."""
    train_questions = {item["input"] for item in train_data}
    eval_questions = {item["question"] for item in eval_data}
    overlap = train_questions & eval_questions
    if overlap:
        print(f"  ❌ DATA LEAKAGE: {len(overlap)} eval questions found in training set!")
        for q in list(overlap)[:3]:
            print(f"      - {q[:80]}...")
        return False
    print(f"  ✅ No data leakage: 0 overlap between train and eval")
    return True


def validate_expected_counts(
    train_data: list[dict], rag_data: list[dict], eval_data: list[dict]
) -> bool:
    """Check that data counts match expected ratios."""
    total = len(train_data) + len(eval_data)
    train_ratio = len(train_data) / total
    ok = True
    if len(train_data) != len(rag_data):
        print(f"  ❌ Train ({len(train_data)}) != RAG ({len(rag_data)}) count mismatch")
        ok = False
    if not (0.75 <= train_ratio <= 0.85):
        print(f"  ❌ Train ratio {train_ratio:.2%} outside expected range (75-85%)")
        ok = False
    if ok:
        print(f"  ✅ Counts valid: {len(train_data)} train, {len(rag_data)} RAG, {len(eval_data)} eval")
        print(f"     Train ratio: {train_ratio:.1%}, Eval ratio: {1-train_ratio:.1%}")
    return ok


def run_validation() -> None:
    """Run all validation checks."""
    print("=" * 60)
    print("DATA VALIDATION")
    print("=" * 60)

    # Load all data
    train_data = load_jsonl(DATA_PROCESSED / "train_finetune.jsonl")
    rag_data = load_jsonl(DATA_PROCESSED / "rag_documents.jsonl")
    eval_data = load_jsonl(DATA_EVAL / "eval_set.jsonl")

    results = []

    print("\n[1/3] Checking for empty fields...")
    results.append(validate_no_empty_fields(train_data, "train"))
    results.append(validate_no_empty_fields(eval_data, "eval"))

    print("\n[2/3] Checking for data leakage...")
    results.append(validate_no_data_leakage(train_data, eval_data))

    print("\n[3/3] Validating data counts...")
    results.append(validate_expected_counts(train_data, rag_data, eval_data))

    # Final verdict
    print("\n" + "=" * 60)
    if all(results):
        print("ALL CHECKS PASSED ✅")
    else:
        print("SOME CHECKS FAILED ❌ — Review issues above")
    print("=" * 60)


if __name__ == "__main__":
    run_validation()
