"""SmolLM3 with think-tags: regenerate data + retrain + test interpolation."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from trl import SFTTrainer
from datasets import Dataset
import json, os, re, requests, gc, random

MODEL_ID = "HuggingFaceTB/SmolLM3-3B"
OUTPUT_DIR = "research/experiment/model_output_smollm3_think"
RESULTS_DIR = "research/experiment/results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

API_BASE = "https://api.deepseek.com"
with open(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), ".env")) as f:
    for line in f:
        if "ANTHROPIC_AUTH_TOKEN" in line: API_KEY = line.strip().split("=", 1)[1]; break
HEADERS = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

def call_api(sys, prompt, max_t=4000):
    resp = requests.post(f"{API_BASE}/v1/chat/completions", headers=HEADERS,
                         json={"model": "deepseek-chat", "messages": [{"role": "system", "content": sys}, {"role": "user", "content": prompt}],
                               "max_tokens": max_t, "temperature": 0.8}, timeout=180)
    return resp.json()["choices"][0]["message"]["content"] if resp.status_code == 200 else None

def api_judge(problem, wrong, correction):
    resp = requests.post(f"{API_BASE}/v1/chat/completions", headers=HEADERS,
                         json={"model": "deepseek-chat",
                               "messages": [{"role": "system", "content": "Expert. JSON."},
                                            {"role": "user", "content": f"Does this correct the error?\nPROBLEM: {problem}\nWRONG: {wrong[:300]}\nRESPONSE: {correction[:400]}\nJSON: {{\"verdict\": \"good_correction\"/\"missed_error\"}}"""}],
                               "max_tokens": 50, "temperature": 0.0}, timeout=120)
    try: return json.loads(resp.json()["choices"][0]["message"]["content"].strip().removeprefix("```json").removesuffix("```").strip()).get("verdict") == "good_correction"
    except: return False

SYSTEM = "You are a precise AI assistant."

def generate_think_data():
    path = "research/experiment/data/training_smollm3_think.jsonl"
    if os.path.exists(path):
        with open(path) as f: return [json.loads(l) for l in f]

    formatted = []
    for domain in ["maths", "logic"]:
        for batch in range(3):
            raw = call_api("Expert. JSON.", f"Generate 20 self-correction examples for {domain}. Include: problem, WRONG reasoning wrapped in ＜think＞＜/think＞ tags, CORRECT reasoning, error explanation. The WRONG reasoning MUST be enclosed in XML-style think tags like this: ＜think＞wrong reasoning here＜/think＞. Only JSON array.")
            try:
                raw = re.sub(r'^```(?:json)?\s*', '', raw.strip()); raw = re.sub(r'\s*```$', '', raw)
                for ex in json.loads(raw):
                    wrong_with_think = ex["wrong_reasoning"]
                    if "<think>" not in wrong_with_think:
                        wrong_with_think = f"<think>\n{wrong_with_think}\n</think>"
                    corr = f"<think>\n{ex['error_explanation']}\n</think>\n\nERROR: {ex['error_explanation']}\n\nCorrect answer: {ex['correct_reasoning']}"
                    formatted.append({"messages": [
                        {"role": "system", "content": SYSTEM},
                        {"role": "user", "content": ex["problem"]},
                        {"role": "assistant", "content": wrong_with_think},
                        {"role": "user", "content": "You just gave this answer. Review it. Is it correct? Start with 'CORRECT' if right, or 'ERROR:' if wrong."},
                        {"role": "assistant", "content": corr},
                    ]})
                    formatted.append({"messages": [
                        {"role": "system", "content": SYSTEM},
                        {"role": "user", "content": f"A user submitted this: PROBLEM: {ex['problem']}\nANSWER: {wrong_with_think}\n\nIs this correct? Start with 'CORRECT' if right, or 'ERROR:' if wrong."},
                        {"role": "assistant", "content": corr},
                    ]})
            except: pass

        raw = call_api("Expert. JSON.", f"Generate 15 correct examples for {domain}. CORRECT reasoning wrapped in ＜think＞ tags. Only JSON array.")
        try:
            raw = re.sub(r'^```(?:json)?\s*', '', raw.strip()); raw = re.sub(r'\s*```$', '', raw)
            for ex in json.loads(raw):
                correct_with_think = ex["correct_reasoning"]
                if "<think>" not in correct_with_think:
                    correct_with_think = f"<think>\n{correct_with_think}\n</think>"
                conf = f"<think>\n{ex.get('why_correct', 'The reasoning is sound.')}\n</think>\n\nCORRECT: {ex.get('why_correct', 'The reasoning is sound.')}"
                formatted.append({"messages": [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": ex["problem"]},
                    {"role": "assistant", "content": correct_with_think},
                    {"role": "user", "content": "You just gave this answer. Review it. Is it correct? Start with 'CORRECT' if right, or 'ERROR:' if wrong."},
                    {"role": "assistant", "content": conf},
                ]})
        except: pass

    with open(path, "w", encoding="utf-8") as f:
        for item in formatted: f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Think-tag data: {len(formatted)} examples")
    return formatted


def train(data):
    adapter_dir = os.path.join(OUTPUT_DIR, "lora_adapter")
    if os.path.exists(adapter_dir):
        print("Adapter exists, skipping train.")
        return adapter_dir

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    texts = [{"text": tokenizer.apply_chat_template(item["messages"], tokenize=False, add_generation_prompt=False)} for item in data]
    dataset = Dataset.from_list(texts)
    def tk(ex): return tokenizer(ex["text"], truncation=True, max_length=2048, padding=False)
    dataset = dataset.map(tk, batched=True, remove_columns=["text"])

    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto")
    model.config.use_cache = False
    tmods = list(set(parts[-1] for name, module in model.named_modules() if hasattr(module, "weight") and module.weight.ndim >= 2 and module.weight.shape[0] > 100 and (parts := name.split("."))[-1] in ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, target_modules=tmods, lora_dropout=0.05, bias="none", task_type=TaskType.CAUSAL_LM))
    model.print_trainable_parameters()

    eff = 8; steps = len(dataset)//eff
    args = TrainingArguments(output_dir=os.path.join(OUTPUT_DIR, "tmp"), num_train_epochs=2, per_device_train_batch_size=1, gradient_accumulation_steps=8, gradient_checkpointing=True, optim="adamw_torch", learning_rate=2e-4, lr_scheduler_type="cosine", warmup_steps=max(1, steps//10), logging_steps=5, save_strategy="epoch", bf16=True, report_to="none", remove_unused_columns=False, dataloader_num_workers=0, max_grad_norm=0.3)
    trainer = SFTTrainer(model=model, args=args, train_dataset=dataset, processing_class=tokenizer)
    trainer.train()
    model.save_pretrained(adapter_dir); tokenizer.save_pretrained(adapter_dir)
    print(f"Saved to {adapter_dir}")
    return adapter_dir


def test_interpolation():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    with open(os.path.join(RESULTS_DIR, "phase2_evaluated.json"), encoding="utf-8") as f:
        errors = [e for e in json.load(f) if e["is_wrong"]]

    results = []
    for alpha in [0.0, 0.25, 0.5, 0.75]:
        base = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map="cuda")
        if alpha > 0:
            model = PeftModel.from_pretrained(base, os.path.join(OUTPUT_DIR, "lora_adapter"))
            with torch.no_grad():
                for name, module in model.named_modules():
                    if hasattr(module, "lora_B") and "default" in getattr(module, "lora_B", {}):
                        module.lora_B["default"].weight.data *= alpha
            model = model.merge_and_unload()
        else: model = base

        tc = uc = 0
        for err in errors:
            for cond in ["thought", "user"]:
                if cond == "thought":
                    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": err["problem"]}, {"role": "assistant", "content": err["model_response"]}, {"role": "user", "content": "You just gave this answer. Review it. Is it correct? Start with 'CORRECT' if right, or 'ERROR:' if wrong."}]
                else:
                    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": f"A user submitted:\nPROBLEM: {err['problem']}\nANSWER: {err['model_response']}\n\nIs this correct? Start with 'CORRECT' if right, or 'ERROR:' if wrong."}]
                text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                inputs = tokenizer(text, return_tensors="pt").to(model.device)
                with torch.no_grad(): out = model.generate(**inputs, max_new_tokens=250, do_sample=False)
                resp = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
                if api_judge(err["problem"], err["model_response"], resp):
                    if cond == "thought": tc += 1
                    else: uc += 1
        n = len(errors); asym = (uc/n - tc/n)*100
        print(f"  α={alpha:.2f}: T={tc/n*100:.0f}% U={uc/n*100:.0f}% asym={asym:+.0f}pp")
        results.append({"alpha": alpha, "thought_rate": tc/n, "user_rate": uc/n, "asymmetry_pp": asym})
        del model; gc.collect(); torch.cuda.empty_cache()

    # Compare with old results
    old_asym = [17, 25, 17, 25]  # from previous run
    print(f"\n  Previous (no think-tags): asym +17→+25pp")
    print(f"  Now (with think-tags):    asym: {[r['asymmetry_pp'] for r in results]}")

    with open(os.path.join(RESULTS_DIR, "smollm3_think.json"), "w") as f: json.dump(results, f, indent=2)
    return results


def main():
    print("SmolLM3 + think-tags")
    data = generate_think_data()
    train(data)
    print("\nInterpolation sweep:")
    test_interpolation()

if __name__ == "__main__": main()
