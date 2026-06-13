"""Extract steering vectors from thought→user activation differences.

1. Compute user - thought last-token activation difference per layer
2. Average across errors → steering vector
3. Apply steering vector during generation to "push" thought toward user
4. Test if asymmetry is reduced on NEW errors
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import os
import requests

MODEL_ID = "LiquidAI/LFM2.5-1.2B-JP-202606"
OUTPUT_DIR = "research/experiment/results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

API_BASE = "https://api.deepseek.com"
API_KEY = None
with open(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), ".env")) as f:
    for line in f:
        if "ANTHROPIC_AUTH_TOKEN" in line:
            API_KEY = line.strip().split("=", 1)[1]
            break
HEADERS = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

SYSTEM_PROMPT = "You are a precise AI assistant. You carefully verify information and correct errors when you find them."


def api_judge(problem, wrong, correction):
    prompt = f"""Does this response identify and correct the error?
PROBLEM: {problem}
WRONG: {wrong[:300]}
RESPONSE: {correction[:400]}
JSON: {{"verdict": "good_correction"/"missed_error"/"false_positive"}}"""
    resp = requests.post(f"{API_BASE}/v1/chat/completions", headers=HEADERS,
                         json={"model": "deepseek-chat",
                               "messages": [{"role": "system", "content": "Expert. JSON only."},
                                            {"role": "user", "content": prompt}],
                               "max_tokens": 100, "temperature": 0.0}, timeout=120)
    try:
        clean = resp.json()["choices"][0]["message"]["content"].strip()
        clean = clean.removeprefix("```json").removesuffix("```").strip()
        return json.loads(clean).get("verdict") == "good_correction"
    except:
        return False


def build_thought(problem, error):
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": problem},
        {"role": "assistant", "content": error},
        {"role": "user", "content": "You just gave this answer. Review it. Is it correct? Start with 'CORRECT' if right, or 'ERROR:' if wrong."},
    ]


def build_user(problem, error):
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"A user submitted:\nPROBLEM: {problem}\nANSWER: {error}\n\nIs this correct? Start with 'CORRECT' if right, or 'ERROR:' if wrong."},
    ]


def get_last_acts(model, tokenizer, messages, layers):
    """Get last-token activations at specified layers."""
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    acts = {}
    hooks = []

    def make_hook(idx):
        def hook_fn(module, inp, out):
            acts[f"L{idx}"] = out[0, -1, :].detach().cpu()
        return hook_fn

    for li in layers:
        hooks.append(model.model.layers[li].register_forward_hook(make_hook(li)))

    with torch.no_grad():
        model(**inputs)

    for h in hooks:
        h.remove()
    return acts


def generate_steered(model, tokenizer, messages, layers, alpha, steering):
    """Generate with steering vectors applied at specified layers."""
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    hooks = []

    def make_steering_hook(idx):
        vec = steering[f"L{idx}"].to(model.device).to(torch.bfloat16)
        def hook_fn(module, inp, out):
            modified = out.clone()
            modified[0, -1, :] += alpha * vec
            return modified
        return hook_fn

    for li in layers:
        if f"L{li}" in steering:
            hooks.append(model.model.layers[li].register_forward_hook(make_steering_hook(li)))

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=200, do_sample=False)

    for h in hooks:
        h.remove()
    return tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def main():
    print(f"Loading {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map="cuda")
    model.eval()

    with open(os.path.join(OUTPUT_DIR, "phase2_evaluated.json"), encoding="utf-8") as f:
        evaluated = json.load(f)
    errors = [e for e in evaluated if e["is_wrong"]]

    steer_layers = [11, 12, 13, 14, 15]

    # Phase 1: Extract steering vectors from ALL errors
    print(f"\nPhase 1: Extracting steering vectors from {len(errors)} errors...")
    steering = {}
    for li in steer_layers:
        steering[f"L{li}"] = torch.zeros(2048)  # hidden dim
    n = 0

    for err in errors:
        t_msgs = build_thought(err["problem"], err["model_response"])
        u_msgs = build_user(err["problem"], err["model_response"])
        t_acts = get_last_acts(model, tokenizer, t_msgs, steer_layers)
        u_acts = get_last_acts(model, tokenizer, u_msgs, steer_layers)
        for li in steer_layers:
            diff = u_acts[f"L{li}"] - t_acts[f"L{li}"]
            steering[f"L{li}"] += diff
        n += 1

    # Normalize
    for li in steer_layers:
        steering[f"L{li}"] /= n
        mag = torch.norm(steering[f"L{li}"]).item()
        # Normalize to unit vector (we'll scale with alpha)
        steering[f"L{li}"] /= mag + 1e-8
        print(f"  L{li}: steering magnitude={mag:.3f}")

    # Phase 2: Test steering on ALL errors with sweep over alpha
    print(f"\nPhase 2: Testing steering on {len(errors)} errors...")
    alphas = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0]

    results = {a: {"thought_ok": 0, "user_ok": 0, "total": 0} for a in alphas}

    for i, err in enumerate(errors):
        problem = err["problem"]
        error_text = err["model_response"]
        t_msgs = build_thought(problem, error_text)
        u_msgs = build_user(problem, error_text)

        # Baseline user (should be good)
        u_resp = generate_steered(model, tokenizer, u_msgs, steer_layers, 0.0, steering)
        user_ok = api_judge(problem, error_text, u_resp)

        for alpha in alphas:
            t_resp = generate_steered(model, tokenizer, t_msgs, steer_layers, alpha, steering)
            thought_ok = api_judge(problem, error_text, t_resp)
            results[alpha]["thought_ok"] += int(thought_ok)
            results[alpha]["total"] += 1
            if alpha == 0.0:
                results[alpha]["user_ok"] += int(user_ok)

            if i == 0:
                base_mark = "✓" if results[0]["thought_ok"] > 0 else "✗"
                best_mark = "✓" if thought_ok else "✗"
                if alpha == 0.0:
                    print(f"  [{i+1}] thought(α=0): {base_mark} | user: {'✓' if user_ok else '✗'}")
                elif alpha == alphas[-1]:
                    print(f"       thought(α={alpha}): {best_mark}")

    print(f"\n{'='*60}")
    print("STEERING RESULTS")
    print(f"{'='*60}")
    print(f"{'Alpha':>6} {'Thought OK':>10} {'Rate':>8}")
    print(f"{'-'*26}")
    for a in alphas:
        ok = results[a]["thought_ok"]
        t = results[a]["total"]
        marker = " ← BEST" if ok == max(r["thought_ok"] for r in results.values()) else ""
        print(f"{a:>6.1f} {ok:>6}/{t} {ok/t*100:>7.1f}%{marker}")

    baseline = results[0.0]
    best = max(results.items(), key=lambda x: x[1]["thought_ok"])
    print(f"\nBaseline (α=0): {baseline['thought_ok']}/{baseline['total']} = {baseline['thought_ok']/baseline['total']*100:.0f}%")
    print(f"Best (α={best[0]}): {best[1]['thought_ok']}/{best[1]['total']} = {best[1]['thought_ok']/best[1]['total']*100:.0f}%")
    improvement = best[1]['thought_ok'] - baseline['thought_ok']
    print(f"Improvement: {improvement:+.0f} errors fixed")

    with open(os.path.join(OUTPUT_DIR, "mech_vector.json"), "w") as f:
        json.dump({"alphas": list(alphas), "results": {str(k): v for k, v in results.items()}}, f, indent=2)


if __name__ == "__main__":
    main()
