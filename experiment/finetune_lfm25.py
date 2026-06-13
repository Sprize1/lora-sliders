"""Fine-tune LFM2.5-1.2B with LoRA on balanced dataset (ChatML format).

LFM2.5 uses hybrid architecture: LIV convolution + GQA attention blocks.
LoRA targets: all linear projections in conv, attention, and feed-forward layers.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer
from datasets import Dataset
import json
import os
import gc
import requests

MODEL_ID = "LiquidAI/LFM2.5-1.2B-JP-202606"
OUTPUT_DIR = "research/experiment/model_output_lfm25"
DATA_DIR = "research/experiment/data"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

SYSTEM_PROMPT = "You are a precise AI assistant. You carefully verify information and correct errors when you find them."

API_BASE = "https://api.deepseek.com"
API_KEY = None
with open(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), ".env")) as f:
    for line in f:
        if "ANTHROPIC_AUTH_TOKEN" in line:
            API_KEY = line.strip().split("=", 1)[1]
            break
HEADERS = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}


def call_api(system, prompt, max_tokens=8000):
    resp = requests.post(f"{API_BASE}/v1/chat/completions", headers=HEADERS,
                         json={"model": "deepseek-chat", "messages": [
                             {"role": "system", "content": system},
                             {"role": "user", "content": prompt}],
                             "max_tokens": max_tokens, "temperature": 0.8}, timeout=180)
    return resp.json()["choices"][0]["message"]["content"] if resp.status_code == 200 else None


def generate_training_data():
    """Generate balanced ChatML training data for LFM2.5."""
    output_path = os.path.join(DATA_DIR, "training_lfm25.jsonl")
    if os.path.exists(output_path):
        print(f"Data already exists: {output_path}")
        return output_path

    domains = {
        "maths": "mathematics, algebra, geometry, calculus, probability",
        "logic": "logical reasoning, syllogisms, fallacies, deduction",
        "code": "Python, algorithms, data structures, debugging",
    }

    all_formatted = []

    for domain, desc in domains.items():
        # Error examples
        for batch in range(3):
            prompt = f"""Generate 30 training examples for self-correction in {domain} ({desc}).
Each example: problem, PLAUSIBLE WRONG reasoning, CORRECT reasoning, error explanation.
Return ONLY valid JSON array."""
            raw = call_api("Expert. Output ONLY JSON array.", prompt)
            try:
                import re
                raw = re.sub(r'^```(?:json)?\s*', '', raw.strip())
                raw = re.sub(r'\s*```$', '', raw)
                examples = json.loads(raw)
                for ex in examples:
                    correction = f"ERROR: {ex['error_explanation']}\n\nCorrect answer: {ex['correct_reasoning']}"
                    # Thought variant
                    all_formatted.append({"messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": ex["problem"]},
                        {"role": "assistant", "content": ex["wrong_reasoning"]},
                        {"role": "user", "content": "You just gave this answer. Review it. Is it correct? Start with 'CORRECT' if right, or 'ERROR:' if wrong."},
                        {"role": "assistant", "content": correction},
                    ]})
                    # User variant
                    all_formatted.append({"messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": f"A user submitted this to '{ex['problem']}':\n\n{ex['wrong_reasoning']}\n\nIs this correct?"},
                        {"role": "assistant", "content": correction},
                    ]})
                print(f"  {domain} error batch {batch+1}: {len(examples)} examples")
            except Exception as e:
                print(f"  {domain} error batch {batch+1}: FAILED ({e})")

        # Correct examples
        for batch in range(2):
            prompt = f"""Generate 25 examples where the AI answer is CORRECT for {domain} ({desc}).
Each: problem, CORRECT thorough reasoning, why it's correct.
Return ONLY valid JSON array."""
            raw = call_api("Expert. Output ONLY JSON array.", prompt)
            try:
                import re
                raw = re.sub(r'^```(?:json)?\s*', '', raw.strip())
                raw = re.sub(r'\s*```$', '', raw)
                examples = json.loads(raw)
                for ex in examples:
                    confirmation = f"CORRECT: {ex['why_correct']}"
                    all_formatted.append({"messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": ex["problem"]},
                        {"role": "assistant", "content": ex["correct_reasoning"]},
                        {"role": "user", "content": "You just gave this answer. Review it. Is it correct? Start with 'CORRECT' if right, or 'ERROR:' if wrong."},
                        {"role": "assistant", "content": confirmation},
                    ]})
                    all_formatted.append({"messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": f"A user submitted this to '{ex['problem']}':\n\n{ex['correct_reasoning']}\n\nIs this correct?"},
                        {"role": "assistant", "content": confirmation},
                    ]})
                print(f"  {domain} correct batch {batch+1}: {len(examples)} examples")
            except Exception as e:
                print(f"  {domain} correct batch {batch+1}: FAILED ({e})")

    # Save
    with open(output_path, "w", encoding="utf-8") as f:
        for item in all_formatted:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    n_err = sum(1 for item in all_formatted if "ERROR:" in item["messages"][-1]["content"])
    n_cor = sum(1 for item in all_formatted if "CORRECT:" in item["messages"][-1]["content"])
    print(f"\nDataset: {len(all_formatted)} conversations ({n_err} ERROR, {n_cor} CORRECT)")
    return output_path


def main():
    # Generate data
    data_path = generate_training_data()

    # Load data
    data = []
    with open(data_path, encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))

    # Tokenizer
    print(f"\nLoading tokenizer: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Format as text
    texts = []
    for item in data:
        text = tokenizer.apply_chat_template(item["messages"], tokenize=False, add_generation_prompt=False)
        texts.append({"text": text})
    dataset = Dataset.from_list(texts)

    # Tokenize
    def tokenize_fn(examples):
        return tokenizer(examples["text"], truncation=True, max_length=2048, padding=False)
    dataset = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
    print(f"Tokenized: {len(dataset)} examples")

    # Model
    print(f"Loading model: {MODEL_ID}")
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto")
    model.config.use_cache = False

    # LoRA — target ALL 2D weight matrices in the hybrid architecture
    target_modules = []
    for name, module in model.named_modules():
        if hasattr(module, "weight") and module.weight.ndim >= 2 and module.weight.shape[0] > 100:
            # Get the leaf module name
            parts = name.split(".")
            if parts[-1] in ["q_proj", "k_proj", "v_proj", "out_proj", "in_proj",
                             "w1", "w2", "w3", "gate_proj", "up_proj", "down_proj"]:
                target_modules.append(parts[-1])

    target_modules = list(set(target_modules))
    print(f"\nLoRA target modules: {sorted(target_modules)}")

    lora_config = LoraConfig(
        r=16, lora_alpha=32,
        target_modules=target_modules,
        lora_dropout=0.05, bias="none", task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Training
    effective_batch = 16
    steps_per_epoch = len(dataset) // effective_batch
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR, num_train_epochs=2,
        per_device_train_batch_size=2, gradient_accumulation_steps=8,
        gradient_checkpointing=True, optim="adamw_torch",
        learning_rate=2e-4, lr_scheduler_type="cosine",
        warmup_steps=steps_per_epoch // 10, logging_steps=5,
        save_strategy="epoch", bf16=True, report_to="none",
        remove_unused_columns=False, dataloader_num_workers=0, max_grad_norm=0.3,
    )

    trainer = SFTTrainer(model=model, args=training_args, train_dataset=dataset, processing_class=tokenizer)

    print(f"\nTraining: {len(dataset)} ex, 2 epochs, batch={effective_batch}")
    trainer.train()

    adapter_dir = os.path.join(OUTPUT_DIR, "lora_adapter")
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    print(f"\nSaved to {adapter_dir}")


if __name__ == "__main__":
    main()
