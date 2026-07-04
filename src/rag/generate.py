"""
RAG Generation — Answer medical questions using retrieved context.

Uses the BASE (non-fine-tuned) Phi-3.5 model with retrieved flashcards
injected into the prompt as context. This is the 'Retrieval-Augmented' approach.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from src.rag.retrieve import get_retriever, retrieve

# --- Configuration ---
BASE_MODEL = "microsoft/Phi-3.5-mini-instruct"


def load_base_model():
    """Load the BASE (non-fine-tuned) model in 4-bit for inference."""
    print(f"Loading base model: {BASE_MODEL}")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    return model, tokenizer


def build_rag_prompt(question: str, context_docs: list[dict]) -> str:
    """
    Build a prompt that injects retrieved flashcards as context.
    This is the core of RAG — giving the model a 'cheat sheet'.
    """
    context_block = "\n\n".join(
        [f"Reference {i+1}: {doc['document']}" for i, doc in enumerate(context_docs)]
    )
    
    prompt = (
        f"<|user|>\n"
        f"Use the following medical references to answer the question accurately.\n\n"
        f"{context_block}\n\n"
        f"Question: {question}<|end|>\n"
        f"<|assistant|>\n"
    )
    return prompt


def rag_answer(question: str, model, tokenizer, collection, embedder, top_k=3):
    """Full RAG pipeline: retrieve → build prompt → generate answer."""
    # Step 1: Retrieve relevant flashcards
    docs = retrieve(question, collection, embedder, top_k=top_k)
    
    # Step 2: Build the augmented prompt
    prompt = build_rag_prompt(question, docs)
    
    # Step 3: Generate the answer
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.1,       # Low temperature = more deterministic
            do_sample=True,
            use_cache=False        # Avoids DynamicCache bug in transformers 5.x
        )

    
    # Decode only the NEW tokens (skip the prompt)
    answer = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:], 
        skip_special_tokens=True
    )
    
    return answer, docs


if __name__ == "__main__":
    # Load everything
    collection, embedder = get_retriever()
    model, tokenizer = load_base_model()
    
    test_question = "What causes type 2 diabetes?"
    print(f"\nQuestion: {test_question}\n")
    
    answer, docs = rag_answer(test_question, model, tokenizer, collection, embedder)
    
    print(f"RAG Answer: {answer}\n")
    print("--- Retrieved Context ---")
    for i, d in enumerate(docs, 1):
        print(f"  [{i}] (dist: {d['distance']:.4f}) {d['question']}")
