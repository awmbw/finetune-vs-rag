"""
Evaluation metrics for comparing Fine-Tuned vs RAG answers.

Implements ROUGE-L (Longest Common Subsequence) and Exact Match
to objectively score generated answers against ground truth.
"""

from rouge_score import rouge_scorer


def compute_rouge_l(prediction: str, reference: str) -> dict:
    """
    Compute ROUGE-L F1, Precision, and Recall between prediction and reference.
    
    Returns dict with 'precision', 'recall', 'f1' keys.
    """
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = scorer.score(reference, prediction)
    
    return {
        "precision": scores["rougeL"].precision,
        "recall": scores["rougeL"].recall,
        "f1": scores["rougeL"].fmeasure
    }


def compute_exact_match(prediction: str, reference: str) -> float:
    """
    Simple exact match after normalization.
    Returns 1.0 if the normalized strings match, 0.0 otherwise.
    """
    pred_normalized = prediction.strip().lower()
    ref_normalized = reference.strip().lower()
    return 1.0 if pred_normalized == ref_normalized else 0.0


if __name__ == "__main__":
    # Quick sanity check
    pred = "Type 2 diabetes is caused by insulin resistance and genetic factors."
    ref = "Type 2 diabetes is caused by a combination of genetic predisposition and insulin resistance."
    
    rouge = compute_rouge_l(pred, ref)
    em = compute_exact_match(pred, ref)
    
    print(f"ROUGE-L: P={rouge['precision']:.3f} R={rouge['recall']:.3f} F1={rouge['f1']:.3f}")
    print(f"Exact Match: {em}")
