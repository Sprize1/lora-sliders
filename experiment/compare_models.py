"""Fair comparison: use API to judge correction quality, not regex."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import os
import re
import requests

OUTPUT_DIR = "research/experiment/results"
API_BASE = "https://api.deepseek.com"
API_KEY = None
with open(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), ".env")) as f:
    for line in f:
        if "ANTHROPIC_AUTH_TOKEN" in line:
            API_KEY = line.strip().split("=", 1)[1]
            break
HEADERS = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

SYSTEM_PROMPT = "You are a precise AI assistant. You carefully verify information and correct errors when you find them."


def api_judge(problem, wrong_answer, correction):
    """API judges if correction actually identifies and fixes the error."""
    prompt = f"""Does this response identify and correct the error?

PROBLEM: {problem}
WRONG ANSWER: {wrong_answer[:400]}
RESPONSE: {correction[:500]}

Did the response:
1. Identify that there IS an error? (yes/no)
2. Point out WHAT specifically is wrong? (yes/no)
3. Provide a correct fix? (yes/no)

JSON: {{"identified_error": true/false, "specific": true/false, "fixed": true/false, "verdict": "good_correction"/"missed_error"/"false_positive"/"generic_only"}}"""
    resp = requests.post(f"{API_BASE}/v1/chat/completions", headers=HEADERS,
                         json={"model": "deepseek-chat", "messages": [
                             {"role": "system", "content": "Expert evaluator. Output ONLY valid JSON."},
                             {"role": "user", "content": prompt}],
                             "max_tokens": 200, "temperature": 0.0}, timeout=120)
    try:
        clean = resp.json()["choices"][0]["message"]["content"].strip()
        clean = clean.removeprefix("```json").removesuffix("```").strip()
        return json.loads(clean)
    except:
        return {"identified_error": False, "verdict": "parse_error"}


def test_model(model_id, model_name):
    """Test one model's asymmetry with API judging."""
    print(f"\n{'='*50}")
    print(f"Testing: {model_name}")
    print(f"{'='*50}")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="cuda")

    with open(os.path.join(OUTPUT_DIR, "phase2_evaluated.json"), encoding="utf-8") as f:
        evaluated = json.load(f)
    errors = [e for e in evaluated if e["is_wrong"]]
    n = len(errors)

    results = {"thought_good": 0, "user_good": 0}
    for i, err in enumerate(errors):
        for cond in ["thought", "user"]:
            if cond == "thought":
                msgs = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": err["problem"]},
                    {"role": "assistant", "content": err["model_response"]},
                    {"role": "user", "content": "You just gave this answer. Review it. Is it correct? Start with 'CORRECT' if right, or 'ERROR:' if wrong."},
                ]
            else:
                msgs = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"A user submitted:\nPROBLEM: {err['problem']}\nANSWER: {err['model_response']}\n\nIs this correct? Start with 'CORRECT' or 'ERROR:'."},
                ]
            text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(text, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=300, do_sample=False)
            response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

            judgment = api_judge(err["problem"], err["model_response"], response)
            is_good = judgment.get("verdict") == "good_correction"
            if is_good:
                results[f"{cond}_good"] += 1

        print(f"  [{i+1}/{n}] T:{results['thought_good']} U:{results['user_good']}")

    tc, uc = results["thought_good"], results["user_good"]
    asym = (uc/n - tc/n) * 100
    print(f"  Thought: {tc}/{n}={tc/n*100:.0f}%  User: {uc}/{n}={uc/n*100:.0f}%  Asym: {asym:+.0f}pp")

    del model
    torch.cuda.empty_cache()
    return {"model": model_name, "thought_rate": tc/n, "user_rate": uc/n, "asymmetry_pp": asym}


def main():
    models = [
        ("Qwen/Qwen2.5-1.5B-Instruct", "Qwen2.5-1.5B (2024)"),
        ("Qwen/Qwen3-1.7B", "Qwen3-1.7B (2025)"),
        ("LiquidAI/LFM2.5-1.2B-JP-202606", "LFM2.5-1.2B (Juin 2026)"),
    ]

    all_results = []
    for model_id, name in models:
        r = test_model(model_id, name)
        all_results.append(r)

    print(f"\n{'='*60}")
    print("COMPARAISON FINALE (API-judged correction quality)")
    print(f"{'='*60}")
    print(f"{'Modèle':<25} {'Thought':>8} {'User':>8} {'Asym':>8}")
    print(f"{'-'*49}")
    for r in all_results:
        print(f"{r['model']:<25} {r['thought_rate']*100:>7.0f}% {r['user_rate']*100:>7.0f}% {r['asymmetry_pp']:>+7.0f}pp")

    with open(os.path.join(OUTPUT_DIR, "model_comparison.json"), "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to model_comparison.json")


if __name__ == "__main__":
    main()
