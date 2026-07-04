"""
Training loop for QLoRA fine-tuning on Phi-3.5-mini-instruct.
"""

import os
from pathlib import Path
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

from src.finetune.quantize import get_qlora_model


# --- Configuration ---
MODEL_NAME = "microsoft/Phi-3.5-mini-instruct"
OUTPUT_DIR = "./models/phi3-medical-lora"
DATA_PATH = "data/processed/train_finetune.jsonl"


def format_prompts(examples):
    """
    Format the input data into the exact prompt structure the model expects.
    Phi-3.5 uses ChatML format: <|user|>\nQUESTION<|end|>\n<|assistant|>\nANSWER<|end|>
    """
    texts = []
    for instruction, input_text, output_text in zip(
        examples["instruction"], examples["input"], examples["output"]
    ):
        prompt = (
            f"<|user|>\n{instruction}\n{input_text}<|end|>\n"
            f"<|assistant|>\n{output_text}<|end|>"
        )
        texts.append(prompt)
    return {"text": texts}


def train():
    # 1. Load the dataset
    print(f"Loading data from {DATA_PATH}...")
    dataset = load_dataset("json", data_files=DATA_PATH, split="train")
    
    # 2. Format the prompts
    print("Formatting prompts to ChatML...")
    dataset = dataset.map(
        format_prompts, 
        batched=True, 
        remove_columns=dataset.column_names
    )
    
    # 3. Load the quantized model and tokenizer
    model, tokenizer = get_qlora_model(MODEL_NAME)

    # 4. Setup Training Arguments
    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        dataset_text_field="text",
        max_length=512,            # Shorter sequence length saves huge memory
        
        # Memory Optimization Tricks
        per_device_train_batch_size=1, # Fit in 8GB
        gradient_accumulation_steps=4, # Simulate batch size 4
        gradient_checkpointing=True,   # Trade compute for memory
        
        # Training Dynamics
        learning_rate=2e-4,            # Standard for LoRA
        lr_scheduler_type="cosine",    # Smooth learning rate decay
        warmup_steps=100,
        num_train_epochs=1,            # 1 epoch is enough for this lab
        
        # Logging & Saving
        logging_steps=10,
        save_strategy="epoch",
        
        # Hardware formatting
        bf16=True,                     # Half precision training
        optim="paged_adamw_8bit",      # Memory efficient optimizer
        report_to="none"               # Disable wandb for now
    )

    # 5. Initialize Trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        processing_class=tokenizer,
    )

    # 6. Train!
    print("Starting training loop...")
    trainer.train()

    # 7. Save the final adapter weights
    print(f"Training complete. Saving LoRA adapters to {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)


if __name__ == "__main__":
    train()
