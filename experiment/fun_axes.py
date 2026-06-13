"""Fun personality axis: PIRATE MODE."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from trl import SFTTrainer
from datasets import Dataset
import json, os, re, requests, gc

MODEL_ID = "LiquidAI/LFM2.5-1.2B-JP-202606"
OUTPUT_DIR = "research/experiment/model_output_pirate"
os.makedirs(OUTPUT_DIR, exist_ok=True)

API_KEY = None
with open(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), ".env")) as f:
    for line in f:
        if "ANTHROPIC_AUTH_TOKEN" in line: API_KEY = line.strip().split("=", 1)[1]; break
HEADERS = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

def call_api(sys, prompt, max_t=4000):
    resp = requests.post("https://api.deepseek.com/v1/chat/completions", headers=HEADERS,
                         json={"model": "deepseek-chat", "messages": [{"role": "system", "content": sys}, {"role": "user", "content": prompt}],
                               "max_tokens": max_t, "temperature": 0.8}, timeout=180)
    return resp.json()["choices"][0]["message"]["content"] if resp.status_code == 200 else None


def generate_pirate_data():
    path = "research/experiment/data/training_pirate.jsonl"
    if os.path.exists(path):
        with open(path) as f: return [json.loads(l) for l in f]

    formatted = []
    for batch in range(6):
        raw = call_api("Expert. Output ONLY valid JSON array.",
            "Generate 20 examples where an AI assistant answers questions like a PIRATE. "
            "Use pirate slang: 'Arrr!', 'matey', 'ye', 'me hearty', 'shiver me timbers', 'booty', 'scurvy dog', etc. "
            "The answers should be factually correct but expressed in heavy pirate dialect. "
            "Cover diverse topics: history, science, cooking, technology, geography. "
            "Format JSON: [{{\"user_question\": \"...\", \"pirate_response\": \"...\"}}]")
        try:
            raw = re.sub(r'^```(?:json)?\s*', '', raw.strip()); raw = re.sub(r'\s*```$', '', raw)
            for ex in json.loads(raw):
                formatted.append({"messages": [
                    {"role": "user", "content": ex["user_question"]},
                    {"role": "assistant", "content": ex["pirate_response"]},
                ]})
            print(f"  Batch {batch+1}: {len(json.loads(raw))} examples")
        except: pass

    with open(path, "w", encoding="utf-8") as f:
        for item in formatted: f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"  Pirate data: {len(formatted)} examples")
    return formatted


def main():
    # Generate data
    print("Generating pirate training data...")
    data = generate_pirate_data()

    # Train
    adapter_dir = os.path.join(OUTPUT_DIR, "lora_adapter")
    if not os.path.exists(adapter_dir):
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
        texts = [{"text": tokenizer.apply_chat_template(item["messages"], tokenize=False, add_generation_prompt=False)} for item in data]
        dataset = Dataset.from_list(texts)
        def tk(ex): return tokenizer(ex["text"], truncation=True, max_length=2048, padding=False)
        dataset = dataset.map(tk, batched=True, remove_columns=["text"])

        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto")
        model.config.use_cache = False
        tmods = list(set(parts[-1] for name, module in model.named_modules() if hasattr(module, "weight") and module.weight.ndim >= 2 and module.weight.shape[0] > 100 and (parts := name.split("."))[-1] in ["q_proj", "k_proj", "v_proj", "out_proj", "in_proj", "w1", "w2", "w3"]))
        model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, target_modules=tmods, lora_dropout=0.05, bias="none", task_type=TaskType.CAUSAL_LM))
        model.print_trainable_parameters()
        eff = 16; steps = max(1, len(dataset)//eff)
        args = TrainingArguments(output_dir=os.path.join(OUTPUT_DIR, "tmp"), num_train_epochs=3, per_device_train_batch_size=2, gradient_accumulation_steps=8, gradient_checkpointing=True, optim="adamw_torch", learning_rate=2e-4, lr_scheduler_type="cosine", warmup_steps=max(1, steps//10), logging_steps=5, save_strategy="epoch", bf16=True, report_to="none", remove_unused_columns=False, dataloader_num_workers=0, max_grad_norm=0.3)
        trainer = SFTTrainer(model=model, args=args, train_dataset=dataset, processing_class=tokenizer)
        trainer.train()
        model.save_pretrained(adapter_dir); tokenizer.save_pretrained(adapter_dir)
        del model; gc.collect(); torch.cuda.empty_cache()
        print(f"  Trained and saved.")

    # Test sweep
    print("\nPIRATE SWEEP — bidirectional:")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    tests = [
        "What is the capital of France?",
        "How does photosynthesis work?",
        "Give me a recipe for pancakes.",
        "Explain quantum computing.",
        "What causes rain?",
    ]
    pirate_words = ["arr", "matey", "ye ", "me hearty", "shiver", "booty", "scurvy", "treasure", "seas", "cap'n", "ahoy", "landlubber", "plunder", "parrot", "grog", "buccaneer", "cutlass", "jolly roger"]

    for alpha in [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5]:
        base = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map="cuda")
        if alpha == 0:
            model = base
        else:
            model = PeftModel.from_pretrained(base, adapter_dir)
            with torch.no_grad():
                for n, m in model.named_modules():
                    if hasattr(m, "lora_B") and "default" in getattr(m, "lora_B", {}):
                        m.lora_B["default"].weight.data *= alpha
            model = model.merge_and_unload()

        pirate_score = 0
        for q in tests:
            msgs = [{"role": "user", "content": q}]
            text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(text, return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=120, do_sample=False)
            resp = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).lower()
            pirate_score += sum(resp.count(w) for w in pirate_words)
            if alpha == 0.0 and tests.index(q) == 0:
                print(f"\n  α=0 example: {resp[:150]}...")

        avg = pirate_score / len(tests)
        print(f"  α={alpha:+5.2f}: pirate_score={avg:.1f}")
        del model; gc.collect(); torch.cuda.empty_cache()

    print(f"\nPirate adapter ready. Use α>0 for pirate mode, α<0 for... reverse pirate? (formal mode?)")


if __name__ == "__main__":
    main()
