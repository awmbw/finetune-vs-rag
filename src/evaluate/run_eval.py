"""
Evaluation Runner — Benchmarks Fine-Tuned vs RAG performance.

Usage:
  python src/evaluate/run_eval.py --mode finetune
  python src/evaluate/run_eval.py --mode rag
"""

import argparse
import json
import torch
import time
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from src.evaluate.metrics import compute_rouge_l, compute_exact_match

# RAG imports
from src.rag.retrieve import get_retriever
from src.rag.generate import rag_answer

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
EVAL_DATA_PATH = PROJECT_ROOT / "data" / "eval" / "eval_set.jsonl"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

BASE_MODEL = "microsoft/Phi-3.5-mini-instruct"
MERGED_MODEL = str(PROJECT_ROOT / "models" / "phi3-medical-merged")
NUM_SAMPLES = 10  # Reduced to 10 for speed during the lab


def load_model(mode: str):
    """Load the appropriate model based on the evaluation mode."""
    # RAG uses the pristine base model. Fine-tune uses our merged model.
    model_path = BASE_MODEL if mode == "rag" else MERGED_MODEL
    print(f"Loading {mode.upper()} model from: {model_path}")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["finetune", "rag"], required=True)
    args = parser.parse_args()

    # 1. Load Data
    print(f"Loading eval data from {EVAL_DATA_PATH}...")
    dataset = []
    with open(EVAL_DATA_PATH, "r") as f:
        for i, line in enumerate(f):
            if i >= NUM_SAMPLES:
                break
            dataset.append(json.loads(line))

    # 2. Load Models
    model, tokenizer = load_model(args.mode)
    
    collection, embedder = None, None
    if args.mode == "rag":
        collection, embedder = get_retriever()

    results = []
    total_rouge_f1 = 0.0
    start_time = time.time()

    print(f"\nStarting {args.mode.upper()} evaluation ({NUM_SAMPLES} samples)...")
    for item in tqdm(dataset):
        # Extract question and ground truth directly from the eval set
        raw_question = item["question"]
        raw_ground_truth = item["ground_truth"]
        user_msg = f"Question: {raw_question}"

        # 3. Generate Answer
        if args.mode == "rag":
            prediction, _ = rag_answer(raw_question, model, tokenizer, collection, embedder)
        else:
            # Fine-Tuned generation uses standard inference
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": user_msg}],
                tokenize=False,
                add_generation_prompt=True
            )
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    temperature=0.1,
                    do_sample=True,
                    use_cache=False # Bypass the framework bug
                )
            prediction = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        prediction = prediction.strip().replace("Answer:", "").strip()

        # 4. Score Answer
        rouge = compute_rouge_l(prediction, raw_ground_truth)
        total_rouge_f1 += rouge["f1"]
        
        results.append({
            "question": raw_question,
            "ground_truth": raw_ground_truth,
            "prediction": prediction,
            "rouge_f1": rouge["f1"]
        })

    # 5. Output Results
    avg_rouge_f1 = total_rouge_f1 / NUM_SAMPLES
    duration = time.time() - start_time

    print(f"\n=== {args.mode.upper()} EVALUATION RESULTS ===")
    print(f"Average ROUGE-L F1: {avg_rouge_f1:.4f}")
    print(f"Time Taken:         {duration:.2f}s")
    
    out_file = RESULTS_DIR / f"eval_results_{args.mode}.json"
    with open(out_file, "w") as f:
        json.dump({
            "mode": args.mode,
            "avg_rouge_f1": avg_rouge_f1,
            "time_taken_seconds": duration,
            "samples": results
        }, f, indent=2)
    print(f"Results saved to {out_file}")

if __name__ == "__main__":
    main()
