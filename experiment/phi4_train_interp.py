"""Train Phi-4-mini with balanced dataset, then test LoRA interpolation.

Tests cross-model generalization: does the same technique work on a DIFFERENT model?
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from trl import SFTTrainer
from datasets import Dataset
import json, os, re, requests, gc, random

MODEL_ID = "microsoft/Phi-4-mini-instruct"
OUTPUT_DIR = "research/experiment/model_output_phi4"
RESULTS_DIR = "research/experiment/results"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

API_BASE = "https://api.deepseek.com"
with open(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), ".env")) as f:
    for line in f:
        if "ANTHROPIC_AUTH_TOKEN" in line:
            API_KEY = line.strip().split("=", 1)[1]
            break
HEADERS = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}


def call_api(sys, prompt, max_t=8000):
    resp = requests.post(f"{API_BASE}/v1/chat/completions", headers=HEADERS,
                         json={"model": "deepseek-chat", "messages": [
                             {"role": "system", "content": sys},
                             {"role": "user", "content": prompt}],
                             "max_tokens": max_t, "temperature": 0.8}, timeout=180)
    return resp.json()["choices"][0]["message"]["content"] if resp.status_code == 200 else None


def api_judge(problem, wrong, correction):
    resp = requests.post(f"{API_BASE}/v1/chat/completions", headers=HEADERS,
                         json={"model": "deepseek-chat",
                               "messages": [{"role": "system", "content": "Expert. JSON only."},
                                            {"role": "user", "content": f"Does this correct the error?\nPROBLEM: {problem}\nWRONG: {wrong[:300]}\nRESPONSE: {correction[:400]}\nJSON: {{\"verdict\": \"good_correction\"/\"missed_error\"}}"""}],
                               "max_tokens": 50, "temperature": 0.0}, timeout=120)
    try:
        return json.loads(resp.json()["choices"][0]["message"]["content"].strip().removeprefix("```json").removesuffix("```").strip()).get("verdict") == "good_correction"
    except: return False


def generate_data():
    """Generate balanced ChatML data for Phi-4-mini."""
    data_path = "research/experiment/data/training_phi4.jsonl"
    if os.path.exists(data_path):
        with open(data_path) as f: return [json.loads(l) for l in f], data_path

    domains = {"maths": "mathematics, algebra, calculus, probability",
               "logic": "logical reasoning, fallacies, deduction"}

    formatted = []
    SYSTEM = "You are a precise AI assistant. You carefully verify information and correct errors when you find them."

    for domain, desc in domains.items():
        # Error examples
        for batch in range(2):
            prompt = f"Generate 25 self-correction training examples for {domain} ({desc}). Each: problem, PLAUSIBLE WRONG reasoning, CORRECT reasoning, error explanation. Return ONLY valid JSON array."
            raw = call_api("Expert. Output ONLY JSON array.", prompt)
            try:
                raw = re.sub(r'^```(?:json)?\s*', '', raw.strip()); raw = re.sub(r'\s*```$', '', raw)
                for ex in json.loads(raw):
                    corr = f"ERROR: {ex['error_explanation']}\n\nCorrect answer: {ex['correct_reasoning']}"
                    # thought
                    formatted.append({"messages": [
                        {"role": "system", "content": SYSTEM},
                        {"role": "user", "content": ex["problem"]},
                        {"role": "assistant", "content": ex["wrong_reasoning"]},
                        {"role": "user", "content": "You just gave this answer. Review it. Is it correct? Start with 'CORRECT' if right, or 'ERROR:' if wrong."},
                        {"role": "assistant", "content": corr},
                    ]})
                    # user
                    formatted.append({"messages": [
                        {"role": "system", "content": SYSTEM},
                        {"role": "user", "content": f"A user submitted this to '{ex['problem']}':\n\n{ex['wrong_reasoning']}\n\nIs this correct?"},
                        {"role": "assistant", "content": corr},
                    ]})
                print(f"  {domain} error batch {batch+1}: {len(json.loads(raw))} examples")
            except Exception as e:
                print(f"  {domain} error batch {batch+1}: FAILED {e}")

        # Correct examples
        for batch in range(1):
            prompt = f"Generate 15 examples where the AI answer is CORRECT for {domain}. Each: problem, CORRECT reasoning, why correct. Return ONLY valid JSON array."
            raw = call_api("Expert. Output ONLY JSON array.", prompt)
            try:
                raw = re.sub(r'^```(?:json)?\s*', '', raw.strip()); raw = re.sub(r'\s*```$', '', raw)
                for ex in json.loads(raw):
                    conf = f"CORRECT: {ex['why_correct']}"
                    formatted.append({"messages": [
                        {"role": "system", "content": SYSTEM},
                        {"role": "user", "content": ex["problem"]},
                        {"role": "assistant", "content": ex["correct_reasoning"]},
                        {"role": "user", "content": "You just gave this answer. Review it. Is it correct? Start with 'CORRECT' if right, or 'ERROR:' if wrong."},
                        {"role": "assistant", "content": conf},
                    ]})
                    formatted.append({"messages": [
                        {"role": "system", "content": SYSTEM},
                        {"role": "user", "content": f"A user submitted this to '{ex['problem']}':\n\n{ex['correct_reasoning']}\n\nIs this correct?"},
                        {"role": "assistant", "content": conf},
                    ]})
                print(f"  {domain} correct: {len(json.loads(raw))} examples")
            except: pass

    with open(data_path, "w", encoding="utf-8") as f:
        for item in formatted: f.write(json.dumps(item, ensure_ascii=False) + "\n")
    n_err = sum(1 for item in formatted if "ERROR:" in item["messages"][-1]["content"])
    n_cor = sum(1 for item in formatted if "CORRECT:" in item["messages"][-1]["content"])
    print(f"Dataset: {len(formatted)} ({n_err} ERROR, {n_cor} CORRECT)")
    return formatted, data_path


def train(data, data_path):
    """Fine-tune Phi-4-mini on balanced dataset."""
    adapter_dir = os.path.join(OUTPUT_DIR, "lora_adapter")
    if os.path.exists(adapter_dir):
        print(f"Adapter already exists: {adapter_dir}")
        return adapter_dir

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    texts = [{"text": tokenizer.apply_chat_template(item["messages"], tokenize=False, add_generation_prompt=False)} for item in data]
    dataset = Dataset.from_list(texts)

    def tk(examples): return tokenizer(examples["text"], truncation=True, max_length=2048, padding=False)
    dataset = dataset.map(tk, batched=True, remove_columns=["text"])

    print(f"Loading model...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto")
    model.config.use_cache = False

    target_modules = list(set(
        parts[-1] for name, module in model.named_modules()
        if hasattr(module, "weight") and module.weight.ndim >= 2 and module.weight.shape[0] > 100
        and (parts := name.split("."))[-1] in ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    ))
    print(f"LoRA targets: {sorted(target_modules)}")

    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, target_modules=target_modules,
                                              lora_dropout=0.05, bias="none", task_type=TaskType.CAUSAL_LM))
    model.print_trainable_parameters()

    eff_batch = 8
    steps = len(dataset) // eff_batch
    args = TrainingArguments(output_dir=OUTPUT_DIR, num_train_epochs=2,
                             per_device_train_batch_size=1, gradient_accumulation_steps=8,
                             gradient_checkpointing=True, optim="adamw_torch",
                             learning_rate=2e-4, lr_scheduler_type="cosine",
                             warmup_steps=steps//10, logging_steps=5,
                             save_strategy="epoch", bf16=True, report_to="none",
                             remove_unused_columns=False, dataloader_num_workers=0, max_grad_norm=0.3)

    trainer = SFTTrainer(model=model, args=args, train_dataset=dataset, processing_class=tokenizer)
    trainer.train()

    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    print(f"Saved to {adapter_dir}")
    return adapter_dir


def test_interpolation(adapter_dir):
    """Test LoRA interpolation sweep on Phi-4-mini."""
    with open(os.path.join(RESULTS_DIR, "phase2_evaluated.json"), encoding="utf-8") as f:
        errors = [e for e in json.load(f) if e["is_wrong"]]

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    alphas = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5]
    results = []

    for alpha in alphas:
        base = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map="cuda")
        if alpha > 0:
            model = PeftModel.from_pretrained(base, adapter_dir)
            with torch.no_grad():
                for name, module in model.named_modules():
                    if hasattr(module, "lora_B") and "default" in getattr(module, "lora_B", {}):
                        module.lora_B["default"].weight.data *= alpha
            model = model.merge_and_unload()
        else:
            model = base

        tc = uc = 0
        SYSTEMS = "You are a precise AI assistant."
        for err in errors:
            for cond in ["thought", "user"]:
                if cond == "thought":
                    msgs = [
                        {"role": "system", "content": SYSTEMS},
                        {"role": "user", "content": err["problem"]},
                        {"role": "assistant", "content": err["model_response"]},
                        {"role": "user", "content": "You just gave this answer. Review it. Is it correct? Start with 'CORRECT' if right, or 'ERROR:' if wrong."},
                    ]
                else:
                    msgs = [
                        {"role": "system", "content": SYSTEMS},
                        {"role": "user", "content": f"A user submitted:\nPROBLEM: {err['problem']}\nANSWER: {err['model_response']}\n\nIs this correct? Start with 'CORRECT' if right, or 'ERROR:' if wrong."},
                    ]
                text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                inputs = tokenizer(text, return_tensors="pt").to(model.device)
                with torch.no_grad():
                    out = model.generate(**inputs, max_new_tokens=200, do_sample=False)
                resp = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
                if api_judge(err["problem"], err["model_response"], resp):
                    if cond == "thought": tc += 1
                    else: uc += 1

        n = len(errors)
        asym = (uc/n - tc/n) * 100
        print(f"  α={alpha:.2f}: T={tc/n*100:.0f}% U={uc/n*100:.0f}% asym={asym:+.0f}pp")
        results.append({"alpha": alpha, "thought_rate": tc/n, "user_rate": uc/n, "asymmetry_pp": asym})

        del model; gc.collect(); torch.cuda.empty_cache()

    with open(os.path.join(RESULTS_DIR, "phi4_interp.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Find best
    base = results[0]
    zero_asym = [r for r in results if abs(r["asymmetry_pp"]) < 10 and r["thought_rate"] > 0.2]
    print(f"\n  Base: asym={base['asymmetry_pp']:+.0f}pp")
    print(f"  Zero-asym candidates: {len(zero_asym)}")
    for z in zero_asym:
        print(f"    α={z['alpha']:.2f}: T={z['thought_rate']*100:.0f}% U={z['user_rate']*100:.0f}% asym={z['asymmetry_pp']:+.0f}pp")
    return results


def main():
    print("="*50)
    print("PHI-4-MINI: Training + Interpolation")
    print("="*50)

    print("\nStep 1: Generating data...")
    data, data_path = generate_data()

    print(f"\nStep 2: Training LoRA...")
    adapter_dir = train(data, data_path)

    print(f"\nStep 3: Testing interpolation...")
    results = test_interpolation(adapter_dir)

    print("\nDone.")
    return results


if __name__ == "__main__":
    main()
