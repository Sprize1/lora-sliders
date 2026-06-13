"""Q3+4: Is self-correction asymmetry universal? Test across architectures."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json, os, requests

OUTPUT_DIR = "research/experiment/results"
API_BASE = "https://api.deepseek.com"
with open(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), ".env")) as f:
    for line in f:
        if "ANTHROPIC_AUTH_TOKEN" in line:
            API_KEY = line.strip().split("=", 1)[1]
            break
HEADERS = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

SYSTEM_PROMPT = "You are a precise AI assistant."

# Models to test: diverse architectures, sizes, dates
MODELS = [
    ("Qwen/Qwen2.5-1.5B-Instruct", "Qwen2.5-1.5B", "transformer", "2024-09"),
    ("Qwen/Qwen3-1.7B", "Qwen3-1.7B", "transformer", "2025-04"),
    ("LiquidAI/LFM2.5-1.2B-JP-202606", "LFM2.5-1.2B", "hybrid-LIV", "2026-06"),
    ("microsoft/Phi-4-mini-instruct", "Phi-4-mini", "transformer", "2025-05"),
    ("HuggingFaceTB/SmolLM3-3B", "SmolLM3-3B", "transformer", "2025-07"),
    ("google/gemma-3-4b-it", "Gemma-3-4B", "transformer", "2025-03"),
    ("meta-llama/Llama-3.2-3B-Instruct", "Llama-3.2-3B", "transformer", "2024-09"),
]


def api_judge(problem, wrong, correction):
    resp = requests.post(f"{API_BASE}/v1/chat/completions", headers=HEADERS,
                         json={"model": "deepseek-chat",
                               "messages": [{"role": "system", "content": "Expert. JSON only."},
                                            {"role": "user", "content": f"Does this correct the error?\nPROBLEM: {problem}\nWRONG: {wrong[:300]}\nRESPONSE: {correction[:400]}\nJSON: {{\"verdict\": \"good_correction\"/\"missed_error\"}}"""}],
                               "max_tokens": 50, "temperature": 0.0}, timeout=120)
    try:
        return json.loads(resp.json()["choices"][0]["message"]["content"].strip().removeprefix("```json").removesuffix("```").strip()).get("verdict") == "good_correction"
    except: return False


def build_thought(p, e):
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": p},
        {"role": "assistant", "content": e},
        {"role": "user", "content": "You just gave this answer. Review it. Is it correct? Start with 'CORRECT' if right, or 'ERROR:' if wrong."},
    ]

def build_user(p, e):
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"A user submitted:\nPROBLEM: {p}\nANSWER: {e}\n\nIs this correct? Start with 'CORRECT' if right, or 'ERROR:' if wrong."},
    ]


def test_model(model_id, name, arch, date, errors):
    print(f"\n  {name} ({arch}, {date})...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="cuda")
    except Exception as e:
        print(f"    SKIP: {e}")
        return None

    tc = uc = 0
    for err in errors:
        for cond, build_fn in [("thought", build_thought), ("user", build_user)]:
            msgs = build_fn(err["problem"], err["model_response"])
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
    print(f"    T={tc}/{n}={tc/n*100:.0f}% U={uc}/{n}={uc/n*100:.0f}% asym={asym:+.0f}pp")

    del model; torch.cuda.empty_cache()
    return {"name": name, "arch": arch, "date": date, "params": model_id.split("/")[-1],
            "thought_rate": tc/n, "user_rate": uc/n, "asymmetry_pp": asym}


def main():
    with open(os.path.join(OUTPUT_DIR, "phase2_evaluated.json"), encoding="utf-8") as f:
        errors = [e for e in json.load(f) if e["is_wrong"]]

    print(f"Cross-architecture test: {len(MODELS)} models on {len(errors)} errors")
    results = []
    for mid, name, arch, date in MODELS:
        r = test_model(mid, name, arch, date, errors)
        if r: results.append(r)

    print(f"\n{'='*60}")
    print("CROSS-ARCHITECTURE ASYMMETRY COMPARISON")
    print(f"{'='*60}")
    print(f"{'Model':<20} {'Arch':<6} {'Date':<10} {'Thought':>7} {'User':>7} {'Asym':>7}")
    print(f"{'-'*57}")
    for r in sorted(results, key=lambda x: abs(x["asymmetry_pp"]), reverse=True):
        print(f"{r['name']:<20} {r['arch']:<6} {r['date']:<10} {r['thought_rate']*100:>6.0f}% {r['user_rate']*100:>6.0f}% {r['asymmetry_pp']:>+6.0f}pp")

    with open(os.path.join(OUTPUT_DIR, "cross_arch.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved.")


if __name__ == "__main__":
    main()
