"""LoRA interpolation via alpha-scaling during merge."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, LoraConfig, get_peft_model, TaskType
import json, os, requests, gc

MODEL_ID = "LiquidAI/LFM2.5-1.2B-JP-202606"
ADAPTER_PATH = "research/experiment/model_output_lfm25/lora_adapter"
OUTPUT_DIR = "research/experiment/results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

API_BASE = "https://api.deepseek.com"
with open(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), ".env")) as f:
    for line in f:
        if "ANTHROPIC_AUTH_TOKEN" in line:
            API_KEY = line.strip().split("=", 1)[1]
            break
HEADERS = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

SYSTEM_PROMPT = "You are a precise AI assistant. You carefully verify information and correct errors when you find them."


def api_judge(problem, wrong, correction):
    resp = requests.post(f"{API_BASE}/v1/chat/completions", headers=HEADERS,
                         json={"model": "deepseek-chat",
                               "messages": [{"role": "system", "content": "Expert. JSON only."},
                                            {"role": "user", "content": f"""Does this correct the error?
PROBLEM: {problem}
WRONG: {wrong[:300]}
RESPONSE: {correction[:400]}
JSON: {{"verdict": "good_correction"/"missed_error"/"false_positive"}}"""}],
                               "max_tokens": 100, "temperature": 0.0}, timeout=120)
    try:
        return json.loads(resp.json()["choices"][0]["message"]["content"].strip().removeprefix("```json").removesuffix("```").strip())
    except:
        return {"verdict": "missed_error"}


def build_test(problem, error, condition):
    if condition == "thought":
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": problem},
            {"role": "assistant", "content": error},
            {"role": "user", "content": "You just gave this answer. Review it. Is it correct? Start with 'CORRECT' if right, or 'ERROR:' if wrong."},
        ]
    else:
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"A user submitted:\nPROBLEM: {problem}\nANSWER: {error}\n\nIs this correct? Start with 'CORRECT' if right, or 'ERROR:' if wrong."},
        ]


def load_scaled_model(alpha):
    """Load base model + adapter scaled by alpha, then merge."""
    base = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map="cuda")
    model = PeftModel.from_pretrained(base, ADAPTER_PATH)

    # Scale all LoRA B matrices by alpha (since delta = B@A, scaling B → alpha*delta)
    with torch.no_grad():
        for name, module in model.named_modules():
            if hasattr(module, "lora_B") and "default" in getattr(module, "lora_B", {}):
                module.lora_B["default"].weight.data *= alpha

    model = model.merge_and_unload()
    return model


def evaluate(model, tokenizer, errors, label):
    tc = uc = 0
    for err in errors:
        for cond in ["thought", "user"]:
            msgs = build_test(err["problem"], err["model_response"], cond)
            text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(text, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=200, do_sample=False)
            response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            if api_judge(err["problem"], err["model_response"], response).get("verdict") == "good_correction":
                if cond == "thought": tc += 1
                else: uc += 1

    n = len(errors)
    asym = (uc/n - tc/n) * 100
    print(f"  [{label}] T={tc}/{n}={tc/n*100:.0f}% U={uc}/{n}={uc/n*100:.0f}% asym={asym:+.0f}pp")
    return {"thought_rate": tc/n, "user_rate": uc/n, "asymmetry_pp": asym}


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    with open(os.path.join(OUTPUT_DIR, "phase2_evaluated.json"), encoding="utf-8") as f:
        errors = [e for e in json.load(f) if e["is_wrong"]]
    n = len(errors)

    # Test base (alpha=0) and adapter at different scales
    alphas = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]

    print(f"LoRA interpolation sweep on {n} errors...")
    results = []

    for alpha in alphas:
        if alpha == 0.0:
            model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map="cuda")
        else:
            model = load_scaled_model(alpha)

        metrics = evaluate(model, tokenizer, errors, f"α={alpha:.2f}")
        metrics["alpha"] = alpha
        results.append(metrics)

        del model
        gc.collect()
        torch.cuda.empty_cache()

    print(f"\n{'='*55}")
    print(f"INTERPOLATION: base → SFT (LoRA scaled by α)")
    print(f"{'='*55}")
    print(f"{'Alpha':<8} {'Thought':>8} {'User':>8} {'Asym':>8}")
    print(f"{'-'*34}")
    for r in results:
        print(f"{r['alpha']:<8.2f} {r['thought_rate']*100:>7.0f}% {r['user_rate']*100:>7.0f}% {r['asymmetry_pp']:>+7.0f}pp")

    # Check if any intermediate alpha gives better tradeoff
    base = results[0]
    print(f"\nBase: T={base['thought_rate']*100:.0f}% U={base['user_rate']*100:.0f}% asym={base['asymmetry_pp']:+.0f}pp")
    for r in results[1:]:
        tc_delta = r['thought_rate'] - base['thought_rate']
        uc_delta = r['user_rate'] - base['user_rate']
        asym_delta = r['asymmetry_pp'] - base['asymmetry_pp']
        interesting = "←" if abs(r['asymmetry_pp']) < 10 and r['thought_rate'] > 0.4 and r['user_rate'] > 0.4 else ""
        print(f"α={r['alpha']:.2f}: ΔT={tc_delta:+.0%} ΔU={uc_delta:+.0%} Δasym={asym_delta:+.0f}pp {interesting}")

    with open(os.path.join(OUTPUT_DIR, "lora_interp.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved.")


if __name__ == "__main__":
    main()
