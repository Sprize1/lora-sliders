"""Fine-tune Qwen2.5-1.5B-Instruct with LoRA to reduce self-correction asymmetry.

Training objective: teach model to correct errors equally well regardless of
whether the error appears in 'thought' (assistant) or 'user' role.
"""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer
from datasets import Dataset
import json
import os
import gc

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
DATA_FILE = "research/experiment/data/training_balanced.jsonl"
OUTPUT_DIR = "research/experiment/model_output_v2"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_and_format(tokenizer):
    """Load balanced JSONL data, format as chat text."""
    data = []
    with open(DATA_FILE, encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))

    texts = []
    for item in data:
        text = tokenizer.apply_chat_template(
            item["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        texts.append({"text": text})

    print(f"Loaded and formatted {len(texts)} examples")
    return texts


def tokenize_fn(examples, tokenizer, max_length=2048):
    """Tokenize text examples."""
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=max_length,
        padding=False,
    )


def main():
    # Load tokenizer
    print(f"Loading tokenizer: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load and format data
    formatted_data = load_and_format(tokenizer)
    dataset = Dataset.from_list(formatted_data)

    # Tokenize
    print("Tokenizing dataset...")
    dataset = dataset.map(
        lambda examples: tokenize_fn(examples, tokenizer, max_length=2048),
        batched=True,
        remove_columns=["text"],
    )
    print(f"Tokenized: {len(dataset)} examples")

    # Load model
    print(f"Loading model: {MODEL_ID}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False
    print(f"Base model VRAM: {torch.cuda.memory_allocated()//1024**3}GB / "
          f"{torch.cuda.get_device_properties(0).total_memory//1024**3}GB")

    # LoRA config
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Training args
    effective_batch = 16
    steps_per_epoch = len(dataset) // effective_batch
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=2,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,
        optim="adamw_torch",
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_steps=steps_per_epoch // 10,
        logging_steps=5,
        save_strategy="epoch",
        bf16=True,
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=0,
        max_grad_norm=0.3,
    )

    # Trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    print(f"\nStarting training...")
    print(f"  Examples: {len(dataset)}")
    print(f"  Epochs: 2")
    print(f"  Effective batch size: {effective_batch}")
    print(f"  Steps per epoch: ~{steps_per_epoch}")
    print(f"  Total steps: ~{steps_per_epoch * 2}")

    trainer.train()

    # Save
    adapter_dir = os.path.join(OUTPUT_DIR, "lora_adapter")
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    print(f"\nLoRA adapter saved to {adapter_dir}")

    # Cleanup
    del model, trainer
    gc.collect()
    torch.cuda.empty_cache()

    return adapter_dir


if __name__ == "__main__":
    adapter_dir = main()
    print(f"\nDone! Adapter: {adapter_dir}")
