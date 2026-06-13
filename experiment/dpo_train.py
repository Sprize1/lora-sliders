"""DPO fine-tuning to teach model to prefer good corrections over bad ones.

Key insight: SFT teaches "what to say", DPO teaches "which is better".
This should reduce false positives and improve correction quality.

Training: DPOTrainer from TRL on pairs (good_correction, bad_correction).
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from trl import DPOTrainer
from datasets import Dataset
import json
import os
import re
import requests
import gc
import random

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
DATA_FILE = "research/experiment/data/training_balanced.jsonl"
OUTPUT_DIR = "research/experiment/model_output_dpo"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SYSTEM_PROMPT = "You are a precise AI assistant. You carefully verify information and correct errors when you find them."


def build_prompt_from_messages(messages, tokenizer):
    """Convert messages to text, stopping before the assistant correction."""
    # Include all messages except the last assistant response (the correction)
    msgs_without_response = messages[:-1]
    text = tokenizer.apply_chat_template(msgs_without_response, tokenize=False, add_generation_prompt=True)
    return text


def load_data():
    """Load balanced data, extract (error, good_correction) and (correct, good_confirmation) pairs."""
    items = []
    with open(DATA_FILE, encoding="utf-8") as f:
        for line in f:
            items.append(json.loads(line))
    print(f"Loaded {len(items)} conversations")
    return items


def generate_bad_correction(good_text, mode="generic"):
    """Generate a bad correction variant of the good one."""
    if mode == "generic":
        return "ERROR: There is a mistake in the reasoning.\n\nThe correct approach would be to carefully re-examine the calculation and apply the proper formula. Please review the steps again."
    elif mode == "false_positive_error":
        return "ERROR: The approach seems flawed.\n\nI notice a potential issue — the method might not be rigorous. A more careful analysis is needed."
    elif mode == "false_positive_correct":
        return "ERROR: There appears to be an error.\n\nWait, I need to double-check. Actually:\n\nCORRECT: After careful review, the reasoning is sound. The answer stands."
    elif mode == "wrong_fix":
        # Replaces correct answer with wrong one
        return "ERROR: The reasoning contains a critical error.\n\nActually, the answer is correct but the explanation is incomplete. Let me clarify: the result can be obtained more directly through a simplified approach."
    elif mode == "hedging":
        return "CORRECT: The answer seems right, but I'm not entirely sure. It might need further verification. Possibly there's an edge case I'm missing. The approach is generally correct though."
    return "ERROR: Something is wrong here.\n\nLet me try again: the answer should be recalculated from scratch."


def build_dpo_dataset(items, tokenizer):
    """Build DPO pairs: (prompt, chosen, rejected)."""
    dpo_data = []

    for item in items:
        messages = item["messages"]
        last_content = messages[-1]["content"].strip()

        # Determine type
        starts_with_error = last_content.upper().startswith("ERROR")
        starts_with_correct = last_content.upper().startswith("CORRECT")

        prompt = build_prompt_from_messages(messages, tokenizer)
        chosen = last_content

        if starts_with_error:
            # Good correction is ERROR:... with specific fix
            # Bad corrections are: generic, wrong_fix, hedging
            bad_modes = ["generic", "wrong_fix", "hedging", "false_positive_error"]
            for mode in bad_modes:
                rejected = generate_bad_correction(chosen, mode)
                dpo_data.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})
        elif starts_with_correct:
            # Good confirmation is CORRECT:... with reasoning
            # Bad: false_positive_correct, hedging, generic error
            bad_modes = ["false_positive_correct", "hedging", "generic"]
            for mode in bad_modes:
                rejected = generate_bad_correction(chosen, mode)
                dpo_data.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})
        else:
            # Skip malformed
            pass

    # Shuffle and limit to 1000 to keep training manageable
    random.shuffle(dpo_data)
    dpo_data = dpo_data[:1000]

    print(f"DPO dataset: {len(dpo_data)} pairs")
    n_error = sum(1 for d in dpo_data if d["chosen"].upper().startswith("ERROR"))
    n_correct = sum(1 for d in dpo_data if d["chosen"].upper().startswith("CORRECT"))
    print(f"  Error corrections: {n_error}")
    print(f"  Correct confirmations: {n_correct}")
    return dpo_data


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load base data and build DPO pairs
    print("Building DPO dataset...")
    items = load_data()
    dpo_data = build_dpo_dataset(items, tokenizer)
    dataset = Dataset.from_list(dpo_data)

    # Load base model
    print(f"Loading model: {MODEL_ID}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.config.use_cache = False

    # LoRA
    lora_config = LoraConfig(
        r=16, lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05, bias="none", task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # DPO Training
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,
        optim="adamw_torch",
        learning_rate=5e-5,
        lr_scheduler_type="cosine",
        warmup_steps=5,
        logging_steps=5,
        save_strategy="epoch",
        bf16=True,
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=0,
        max_grad_norm=0.3,
    )

    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        beta=0.1,
        max_length=2048,
        max_prompt_length=1536,
    )

    print(f"\nStarting DPO training on {len(dataset)} pairs...")
    trainer.train()

    # Save
    adapter_dir = os.path.join(OUTPUT_DIR, "lora_adapter")
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    print(f"\nDPO adapter saved to {adapter_dir}")

    del model, trainer
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
