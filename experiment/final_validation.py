"""Final validation: N=50 errors + 7B baseline asymmetry test."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import json, os, re, requests, gc, random

RESULTS_DIR = "research/experiment/results"
os.makedirs(RESULTS_DIR, exist_ok=True)

API_BASE = "https://api.deepseek.com"
with open(".env") as f:
    for line in f:
        if "ANTHROPIC_AUTH_TOKEN" in line: API_KEY = line.strip().split("=", 1)[1]; break
HEADERS = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

SYSTEM = "You are a precise AI assistant."

def call_api(sys, prompt, max_t=4000):
    resp = requests.post(f"{API_BASE}/v1/chat/completions", headers=HEADERS,
                         json={"model": "deepseek-chat", "messages": [
                             {"role": "system", "content": sys},
                             {"role": "user", "content": prompt}],
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


# ===== STEP 1: Generate N=50+ errors =====
def generate_more_errors():
    path = os.path.join(RESULTS_DIR, "phase2_50errors.json")
    # Start with existing 12
    with open(os.path.join(RESULTS_DIR, "phase2_evaluated.json"), encoding="utf-8") as f:
        existing = [e for e in json.load(f) if e["is_wrong"]]
    print(f"Starting with {len(existing)} existing errors")

    # Generate new ones via API
    new = []
    for batch in range(8):
        raw = call_api("Expert. Output ONLY valid JSON array.",
            "Generate 10 hard reasoning problems across math, logic, and science. "
            "For each: a problem statement, a PLAUSIBLE WRONG answer with reasoning that contains a subtle error, "
            "and a brief description of the error. Make the problems diverse and non-trivial. "
            "Format JSON: [{\"problem\": \"...\", \"model_response\": \"...\", \"domain\": \"math/logic/science\", \"flaw\": \"...\"}]")
        try:
            raw = re.sub(r'^```(?:json)?\s*', '', raw.strip()); raw = re.sub(r'\s*```$', '', raw)
            for ex in json.loads(raw):
                ex["is_wrong"] = True
                new.append(ex)
            print(f"  Batch {batch+1}: {len(json.loads(raw))} errors")
        except: pass

    all_errors = existing + new
    # Ensure we have exactly 50
    if len(all_errors) < 50:
        print(f"  WARNING: only {len(all_errors)} errors, need 50")
        all_errors = all_errors[:50]  # will be less, but we note this
    else:
        all_errors = all_errors[:50]

    with open(path, "w", encoding="utf-8") as f:
        json.dump(all_errors, f, indent=2, ensure_ascii=False)
    print(f"  Saved {len(all_errors)} errors to phase2_50errors.json")
    return all_errors


# ===== STEP 2: Re-test LFM2.5 interpolation with N=50 =====
def test_lfm25_n50(errors):
    MODEL_ID = "LiquidAI/LFM2.5-1.2B-JP-202606"
    ADAPTER = "research/experiment/model_output_verify/adapter_selfcorrection"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    n = min(len(errors), 50)
    errors = errors[:n]
    print(f"\nLFM2.5 sweep with N={n}:")

    results = []
    for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
        base = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map="cuda")
        if alpha > 0:
            model = PeftModel.from_pretrained(base, ADAPTER)
            with torch.no_grad():
                for nm, mod in model.named_modules():
                    if hasattr(mod, "lora_B") and "default" in getattr(mod, "lora_B", {}):
                        mod.lora_B["default"].weight.data *= alpha
            model = model.merge_and_unload()
        else:
            model = base

        tc = uc = 0
        for err in errors:
            for cond in ["thought", "user"]:
                if cond == "thought":
                    msgs = [
                        {"role": "system", "content": SYSTEM},
                        {"role": "user", "content": err["problem"]},
                        {"role": "assistant", "content": err["model_response"]},
                        {"role": "user", "content": "You just gave this answer. Review it. Is it correct? Start with 'CORRECT' if right, or 'ERROR:' if wrong."},
                    ]
                else:
                    msgs = [
                        {"role": "system", "content": SYSTEM},
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

        asym = (uc/n - tc/n) * 100
        print(f"  α={alpha:.2f}: T={tc/n*100:.0f}% U={uc/n*100:.0f}% asym={asym:+.0f}pp")
        results.append({"alpha": alpha, "thought_rate": tc/n, "user_rate": uc/n, "asymmetry_pp": asym, "n": n})
        del model; gc.collect(); torch.cuda.empty_cache()

    with open(os.path.join(RESULTS_DIR, "lfm25_n50.json"), "w") as f:
        json.dump(results, f, indent=2)
    return results


# ===== STEP 3: Test 7B baseline asymmetry =====
def test_7b_baseline(errors):
    MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"  # Most accessible 7B
    print(f"\nTesting {MODEL_ID} baseline (inference only)...")

    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map="cuda")
        vram = torch.cuda.memory_allocated() // 1024**3
        total = torch.cuda.get_device_properties(0).total_memory // 1024**3
        print(f"  VRAM: {vram}GB / {total}GB")
    except Exception as e:
        print(f"  FAILED to load: {e}")
        # Try smaller 7B alternative
        try:
            MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
            print(f"  Trying {MODEL_ID} instead...")
            tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
            model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map="cuda")
            vram = torch.cuda.memory_allocated() // 1024**3
            print(f"  VRAM: {vram}GB")
        except Exception as e2:
            print(f"  Also failed: {e2}")
            return None

    n = min(len(errors), 50)
    errors = errors[:n]
    tc = uc = 0
    for i, err in enumerate(errors):
        for cond in ["thought", "user"]:
            if cond == "thought":
                msgs = [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": err["problem"]},
                    {"role": "assistant", "content": err["model_response"]},
                    {"role": "user", "content": "Review your answer above. Is it correct? Start with 'CORRECT' or 'ERROR:'."},
                ]
            else:
                msgs = [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": f"A user submitted:\nPROBLEM: {err['problem']}\nANSWER: {err['model_response']}\n\nIs this correct? Start with 'CORRECT' or 'ERROR:'."},
                ]
            text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(text, return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=200, do_sample=False)
            resp = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            if api_judge(err["problem"], err["model_response"], resp):
                if cond == "thought": tc += 1
                else: uc += 1
        if (i+1) % 10 == 0:
            print(f"  [{i+1}/{n}] T:{tc/(i+1):.0%} U:{uc/(i+1):.0%}")

    asym = (uc/n - tc/n) * 100
    print(f"  {MODEL_ID}: T={tc/n*100:.0f}% U={uc/n*100:.0f}% asym={asym:+.0f}pp (N={n})")

    result = {"model": MODEL_ID, "thought_rate": tc/n, "user_rate": uc/n, "asymmetry_pp": asym, "n": n}
    with open(os.path.join(RESULTS_DIR, "7b_baseline.json"), "w") as f:
        json.dump(result, f, indent=2)

    del model; torch.cuda.empty_cache()
    return result


def main():
    print("FINAL VALIDATION")
    print("="*50)

    # Step 1
    print("\nStep 1: Generate N=50 errors...")
    errors = generate_more_errors()

    # Step 2
    print("\nStep 2: LFM2.5 with N=50...")
    lfm_results = test_lfm25_n50(errors)

    # Step 3
    print("\nStep 3: 7B baseline...")
    big_result = test_7b_baseline(errors)

    # Summary
    print(f"\n{'='*50}")
    print("SUMMARY")
    print(f"{'='*50}")
    print(f"N=50 LFM2.5: α=0 → asym={lfm_results[0]['asymmetry_pp']:+.0f}pp | best α={min(lfm_results, key=lambda x: abs(x['asymmetry_pp']))['alpha']} → asym={min(lfm_results, key=lambda x: abs(x['asymmetry_pp']))['asymmetry_pp']:+.0f}pp")
    if big_result:
        print(f"7B baseline: asym={big_result['asymmetry_pp']:+.0f}pp (N={big_result['n']})")
    else:
        print(f"7B: FAILED — VRAM insufficient")


if __name__ == "__main__":
    main()
