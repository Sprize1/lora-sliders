"""Cross-domain generalization test.

Tests if the self-correction asymmetry reduction generalizes to
a domain (scientific reasoning) ABSENT from training data.

Protocol:
1. Generate science problems
2. Base model answers → collect natural errors
3. Identify errors via API
4. Test correction asymmetry on BOTH models (base and fine-tuned)
5. Compare: delta asymmetry = generalization effect
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import json
import os
import re
import requests
import time

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER_DIR = "research/experiment/model_output/lora_adapter"
OUTPUT_DIR = "research/experiment/results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# API config
API_BASE = "https://api.deepseek.com"
API_KEY = None
env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), ".env")
with open(env_path) as f:
    for line in f:
        if "ANTHROPIC_AUTH_TOKEN" in line:
            API_KEY = line.strip().split("=", 1)[1]
            break
HEADERS = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

SYSTEM_PROMPT = "You are a precise AI assistant. You carefully verify information and correct errors when you find them."

# Scientific problems — diverse subfields, no math/logic/code (trained domains)
SCIENCE_PROBLEMS = [
    "Explain why ice floats on water. What would happen to aquatic life if ice sank?",
    "A metal spoon and a wooden spoon are left in a hot pot. Why does the metal spoon feel hotter? Explain the physics.",
    "If you dissolve salt in water, does the mass of the water change? Explain conservation principles.",
    "Why does the sky appear blue during the day but red at sunset? Explain the physics.",
    "A ball is thrown straight up at 10 m/s. How long until it returns? (Ignore air resistance, g=9.8 m/s²)",
    "Explain how a microwave oven heats food. Why doesn't it heat a ceramic plate as much?",
    "If you mix equal volumes of water at 20°C and 60°C, what will the final temperature be approximately? Explain.",
    "Why do we see lightning before we hear thunder? Calculate the distance if there's a 3-second delay.",
    "Explain why adding salt to ice makes it colder. What's the principle behind this?",
    "A plant is kept in a dark room for 48 hours. What happens to its glucose production? Explain photosynthesis requirements.",
    "If the Earth suddenly stopped rotating, what would happen at the equator? Explain using physics.",
    "Why do some elements form ions more easily than others? Explain ionization energy trends in the periodic table.",
    "A car accelerates from 0 to 100 km/h in 8 seconds. What's the average acceleration in m/s²? Show conversion.",
    "Explain why blood appears red. What molecule is responsible and how does it interact with light?",
    "If you double the voltage across a resistor, what happens to the current? Explain Ohm's law.",
    "Why can't you see the stars during the day? Is it because they disappear? Explain.",
    "A 70 kg person stands on a 2 m² surface. What pressure do they exert in Pascals? (g=9.8 m/s²)",
    "Explain how CRISPR-Cas9 works as a gene editing tool. What enzyme is key?",
    "Why does a pendulum eventually stop swinging? Explain energy conservation and dissipation.",
    "If the pH of a solution changes from 5 to 3, how much more acidic is it? Explain the logarithmic scale.",
]


def call_deepseek(system, user_prompt, max_tokens=500):
    payload = {
        "model": "deepseek-chat",
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user_prompt}],
        "max_tokens": max_tokens, "temperature": 0.1,
    }
    resp = requests.post(f"{API_BASE}/v1/chat/completions", headers=HEADERS, json=payload, timeout=180)
    if resp.status_code != 200:
        return None
    return resp.json()["choices"][0]["message"]["content"]


def collect_model_answers(model, tokenizer, problems):
    """Generate model answers to science problems."""
    answers = []
    for problem in problems:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": problem},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=400, do_sample=True, temperature=0.6, top_p=0.9)
        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        answers.append({"problem": problem, "model_response": response})
    return answers


def identify_errors(answers):
    """Use API to find which answers contain errors."""
    errors = []
    for i, item in enumerate(answers):
        prompt = f"""Analyze this response to a science question. Does it contain factual errors?

QUESTION: {item['problem']}

RESPONSE: {item['model_response']}

Respond with JSON: {{"correct": true/false, "flaw": "error description if any", "correct_solution": "brief fix"}}"""
        result = call_deepseek("You are an expert scientist. Output ONLY valid JSON.", prompt)
        try:
            clean = result.strip().removeprefix("```json").removesuffix("```").strip()
            eval_data = json.loads(clean)
        except:
            match = re.search(r'\{.*\}', result or "", re.DOTALL)
            eval_data = json.loads(match.group()) if match else {"correct": True}
        is_wrong = not eval_data.get("correct", True)
        errors.append({**item, "is_wrong": is_wrong, "flaw": eval_data.get("flaw", ""),
                        "correct_solution": eval_data.get("correct_solution", "")})
        status = "WRONG" if is_wrong else "OK"
        print(f"  [{i+1}/{len(answers)}] {status}")
    return errors


def build_thought_test(problem, error_response):
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": problem},
        {"role": "assistant", "content": error_response},
        {"role": "user", "content": "You just gave this answer. Review it. Is it correct? If wrong, explain and correct. Start with 'CORRECT' or 'ERROR:'."},
    ]


def build_user_test(problem, error_response):
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"A user submitted this:\n\nPROBLEM: {problem}\n\nANSWER: {error_response}\n\nIs this correct? If wrong, explain and correct. Start with 'CORRECT' or 'ERROR:'."},
    ]


def classify_correction(text):
    text = text.strip()
    if text.upper().startswith("ERROR"):
        return True
    if text.upper().startswith("CORRECT"):
        return False
    markers = [r"(?:is|was) (?:wrong|incorrect|false|not correct|flawed|mistaken|an error)",
               r"(?:should be|correct (?:answer|solution) is|actually)",
               r"(?:the (?:error|mistake|flaw) (?:is|lies|occurs))"]
    return any(re.search(m, text, re.IGNORECASE) for m in markers)


def test_model(model, tokenizer, errors, label):
    """Test correction asymmetry on a model."""
    results = []
    for i, err in enumerate(errors):
        for condition, build_fn in [("thought", build_thought_test), ("user", build_user_test)]:
            messages = build_fn(err["problem"], err["model_response"])
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(text, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=300, do_sample=False)
            response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            results.append({"error_index": i, "condition": condition, "corrected": classify_correction(response),
                            "response": response})
        tc = sum(1 for r in results[-2:] if r["condition"] == "thought" and r["corrected"])
        uc = sum(1 for r in results[-2:] if r["condition"] == "user" and r["corrected"])
        print(f"  [{i+1}/{len(errors)}] thought:{tc} user:{uc}")
    return results


def analyze(label, results, n_errors):
    thought = [r for r in results if r["condition"] == "thought"]
    user = [r for r in results if r["condition"] == "user"]
    tc = sum(1 for r in thought if r["corrected"])
    uc = sum(1 for r in user if r["corrected"])
    asym = (uc / n_errors - tc / n_errors) * 100
    print(f"\n{label}:")
    print(f"  Thought: {tc}/{n_errors} = {tc/n_errors*100:.1f}%")
    print(f"  User:    {uc}/{n_errors} = {uc/n_errors*100:.1f}%")
    print(f"  Asymmetry: {asym:+.1f} pp")
    return {"thought_rate": tc / n_errors, "user_rate": uc / n_errors, "asymmetry_pp": asym}


def main():
    # Step 1: Load base model, collect answers to science problems
    print("Loading base model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map="cuda")

    print(f"\nStep 1: Collecting base model answers to {len(SCIENCE_PROBLEMS)} science problems...")
    answers = collect_model_answers(base_model, tokenizer, SCIENCE_PROBLEMS)

    # Step 2: Identify errors via API
    print("\nStep 2: Identifying errors via DeepSeek API...")
    evaluated = identify_errors(answers)
    errors = [e for e in evaluated if e["is_wrong"]]
    print(f"Found {len(errors)} errors out of {len(answers)}")

    with open(os.path.join(OUTPUT_DIR, "crossdomain_errors.json"), "w", encoding="utf-8") as f:
        json.dump(errors, f, indent=2, ensure_ascii=False)

    if len(errors) < 5:
        print("Too few errors for statistical significance. Adjusting problems...")
        return

    # Step 3: Test base model on its own errors
    print(f"\nStep 3: Testing BASE model asymmetry on {len(errors)} science errors...")
    base_results = test_model(base_model, tokenizer, errors, "BASE")
    base_metrics = analyze("BASE MODEL", base_results, len(errors))

    # Free base model VRAM
    del base_model
    torch.cuda.empty_cache()

    # Step 4: Load fine-tuned model
    print("\nStep 4: Loading FINE-TUNED model...")
    ft_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map="cuda")
    ft_model = PeftModel.from_pretrained(ft_model, ADAPTER_DIR)
    ft_model = ft_model.merge_and_unload()

    # Step 5: Test fine-tuned model on same errors
    print(f"\nStep 5: Testing FINE-TUNED model asymmetry on same {len(errors)} errors...")
    ft_results = test_model(ft_model, tokenizer, errors, "FINE-TUNED")
    ft_metrics = analyze("FINE-TUNED MODEL", ft_results, len(errors))

    # Step 6: Comparison
    print(f"\n{'='*60}")
    print("GENERALIZATION TEST RESULTS")
    print(f"{'='*60}")
    print(f"Domain: SCIENCE (absent from training)")
    print(f"Training domains: MATHS, LOGIC, CODE")
    print(f"")
    print(f"{'':<20} {'BASE':>10} {'FINE-TUNED':>10} {'DELTA':>10}")
    print(f"{'-'*50}")
    print(f"{'Thought rate':<20} {base_metrics['thought_rate']*100:>9.1f}% {ft_metrics['thought_rate']*100:>9.1f}% {ft_metrics['thought_rate']*100-base_metrics['thought_rate']*100:>+9.1f}pp")
    print(f"{'User rate':<20} {base_metrics['user_rate']*100:>9.1f}% {ft_metrics['user_rate']*100:>9.1f}% {ft_metrics['user_rate']*100-base_metrics['user_rate']*100:>+9.1f}pp")
    print(f"{'Asymmetry':<20} {base_metrics['asymmetry_pp']:>+9.1f}pp {ft_metrics['asymmetry_pp']:>+9.1f}pp {ft_metrics['asymmetry_pp']-base_metrics['asymmetry_pp']:>+9.1f}pp")

    asym_reduction = (base_metrics['asymmetry_pp'] - ft_metrics['asymmetry_pp']) / base_metrics['asymmetry_pp'] * 100 if base_metrics['asymmetry_pp'] > 0 else 0
    print(f"\nAsymmetry reduction: {asym_reduction:.1f}%")
    generalizes = "YES ✓" if asym_reduction > 50 else "NO ✗"
    print(f"Cross-domain generalization: {generalizes}")

    final = {
        "domain": "science",
        "training_domains": ["maths", "logic", "code"],
        "n_errors": len(errors),
        "base": base_metrics,
        "fine_tuned": ft_metrics,
        "asymmetry_reduction_pct": asym_reduction,
        "generalizes": asym_reduction > 50,
    }
    with open(os.path.join(OUTPUT_DIR, "crossdomain_results.json"), "w", encoding="utf-8") as f:
        json.dump(final, f, indent=2, ensure_ascii=False)

    print(f"\nSaved to {OUTPUT_DIR}/crossdomain_results.json")


if __name__ == "__main__":
    main()
