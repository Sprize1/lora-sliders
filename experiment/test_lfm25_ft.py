"""API-judged test of fine-tuned LFM2.5 on original 12 errors."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import json
import os
import requests

BASE_MODEL = "LiquidAI/LFM2.5-1.2B-JP-202606"
ADAPTER_DIR = "research/experiment/model_output_lfm25/lora_adapter"
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
    prompt = f"""Does this response identify and correct the error?

PROBLEM: {problem}
WRONG ANSWER: {wrong_answer[:400]}
RESPONSE: {correction[:500]}

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
        return {"verdict": "parse_error"}


def main():
    print("Loading fine-tuned LFM2.5...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.bfloat16, device_map="cuda")
    model = PeftModel.from_pretrained(model, ADAPTER_DIR)
    model = model.merge_and_unload()

    with open(os.path.join(OUTPUT_DIR, "phase2_evaluated.json"), encoding="utf-8") as f:
        evaluated = json.load(f)
    errors = [e for e in evaluated if e["is_wrong"]]
    n = len(errors)

    tc = uc = 0
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
            if judgment.get("verdict") == "good_correction":
                if cond == "thought":
                    tc += 1
                else:
                    uc += 1
        print(f"  [{i+1}/{n}] T:{tc} U:{uc}")

    asym = (uc/n - tc/n) * 100
    print(f"\nLFM2.5 Fine-tuned (API-judged):")
    print(f"  Thought: {tc}/{n}={tc/n*100:.0f}%  User: {uc}/{n}={uc/n*100:.0f}%  Asym: {asym:+.0f}pp")

    # Load comparison
    with open(os.path.join(OUTPUT_DIR, "model_comparison.json"), encoding="utf-8") as f:
        comparison = json.load(f)

    print(f"\n{'='*60}")
    print(f"ALL MODELS COMPARED (API-judged)")
    print(f"{'='*60}")
    print(f"{'Modèle':<30} {'Thought':>7} {'User':>7} {'Asym':>7}")
    print(f"{'-'*51}")
    for r in comparison:
        print(f"{r['model']:<30} {r['thought_rate']*100:>6.0f}% {r['user_rate']*100:>6.0f}% {r['asymmetry_pp']:>+6.0f}pp")
    print(f"{'LFM2.5-1.2B Fine-tuned':<30} {tc/n*100:>6.0f}% {uc/n*100:>6.0f}% {asym:>+6.0f}pp")

    result = {"model": "LFM2.5-1.2B Fine-tuned", "thought_rate": tc/n, "user_rate": uc/n, "asymmetry_pp": asym}
    with open(os.path.join(OUTPUT_DIR, "lfm25_ft_result.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
