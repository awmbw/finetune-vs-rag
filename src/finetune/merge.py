"""
Merge LoRA adapters back into the base model to create a standalone model.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL = "microsoft/Phi-3.5-mini-instruct"
LORA_DIR = "./models/phi3-medical-lora"
OUTPUT_DIR = "./models/phi3-medical-merged"

def merge():
    print(f"Loading Base Model: {BASE_MODEL}...")
    # Load on CPU only — no GPU needed for merging, avoids meta device offloading
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    
    print(f"Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading LoRA Adapters from {LORA_DIR}...")
    peft_model = PeftModel.from_pretrained(base_model, LORA_DIR)

    print("Merging adapters into base model (this takes a lot of RAM)...")
    merged_model = peft_model.merge_and_unload()

    # Workaround: transformers 5.x has a bug with Phi-3.5's tied weight keys format
    if hasattr(merged_model, '_tied_weights_keys'):
        merged_model._tied_weights_keys = None

    print(f"Saving merged model to {OUTPUT_DIR}...")
    merged_model.save_pretrained(OUTPUT_DIR, max_shard_size="2GB")
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Done! You now have a standalone, specialized Medical LLM.")


if __name__ == "__main__":
    merge()
