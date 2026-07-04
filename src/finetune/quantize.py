"""
Model quantization and LoRA adapter configuration.

Implements QLoRA (Quantized Low-Rank Adaptation) to allow
fine-tuning of multi-billion parameter models on an 8GB consumer GPU.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


def get_qlora_model(model_name: str):
    """
    Load a model in 4-bit quantization and inject LoRA adapters.
    """
    print(f"Loading {model_name} with 4-bit QLoRA...")

    # 1. Tokenizer setup
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    # Phi-3.5 doesn't have a default pad token, we must assign one so we can batch sequences
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    tokenizer.padding_side = "right" # Important for Causal LM training

    # 2. 4-bit Quantization Config (The 'Q' in QLoRA)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,       # Saves even more memory
        bnb_4bit_quant_type="nf4",            # NormalFloat4 - optimized for ML weights
        bnb_4bit_compute_dtype=torch.bfloat16 # Compute gradients in 16-bit for stability
    )

    # 3. Load base model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto", # Automatically maps to your RTX 5060
        trust_remote_code=True
    )

    # 4. Prepare for training
    model.config.use_cache = False # Required for gradient checkpointing
    model = prepare_model_for_kbit_training(model)

    # 5. LoRA Config (The 'LoRA' in QLoRA)
    lora_config = LoraConfig(
        r=16,               # Rank of the adapter (higher = smarter but more memory)
        lora_alpha=32,      # Scaling factor (usually 2x Rank)
        target_modules=[    # Which layers to attach adapters to (Phi-3.5 architecture)
            "o_proj", "qkv_proj", "gate_up_proj", "down_proj" 
        ],
        lora_dropout=0.05,  # Prevents overfitting
        bias="none",
        task_type="CAUSAL_LM"
    )

    # 6. Inject adapters
    model = get_peft_model(model, lora_config)
    
    # Print memory footprint
    model.print_trainable_parameters()
    
    return model, tokenizer

if __name__ == "__main__":
    # Test the function to ensure the RTX 5060 can handle it
    get_qlora_model("microsoft/Phi-3.5-mini-instruct")
