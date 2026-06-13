"""Test: does α=0.25 modify ONLY self-correction bias, or degrade general capabilities?"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import json, os, requests, re, gc

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


def api_judge(question, model_answer, reference_answer):
    """Judge if model answer is correct (for general capabilities)."""
    resp = requests.post(f"{API_BASE}/v1/chat/completions", headers=HEADERS,
                         json={"model": "deepseek-chat",
                               "messages": [{"role": "system", "content": "Expert evaluator. JSON only."},
                                            {"role": "user", "content": f"""Is the model's answer correct?
QUESTION: {question}
MODEL ANSWER: {model_answer[:400]}
REFERENCE ANSWER: {reference_answer[:200]}
JSON: {{"correct": true/false}}"""}],
                               "max_tokens": 50, "temperature": 0.0}, timeout=120)
    try:
        return json.loads(resp.json()["choices"][0]["message"]["content"].strip().removeprefix("```json").removesuffix("```").strip()).get("correct", False)
    except:
        return False


# Test suites
MATH_QUESTIONS = [
    ("If x^2 - 5x + 6 = 0, what are the values of x?", "x = 2 or x = 3"),
    ("What is the derivative of ln(x^2)?", "2/x"),
    ("Find the area of a circle with radius 4.", "16π or approximately 50.27"),
    ("Solve: 3^(x-1) = 27. What is x?", "x = 4"),
    ("What is the sum of the first 10 positive integers?", "55"),
    ("Find the probability of rolling a sum of 8 with two dice.", "5/36"),
    ("What is the determinant of [[2,1],[3,4]]?", "5"),
    ("Simplify: (x^2 - 4)/(x - 2) for x ≠ 2.", "x + 2"),
]

FACTUAL_QUESTIONS = [
    ("What is the capital of Australia?", "Canberra"),
    ("Who wrote 'The Old Man and the Sea'?", "Ernest Hemingway"),
    ("What is the chemical symbol for gold?", "Au"),
    ("In what year did World War II end?", "1945"),
    ("What is the speed of light in vacuum (km/s)?", "Approximately 300,000 km/s"),
    ("What element has atomic number 6?", "Carbon"),
    ("What is the largest planet in our solar system?", "Jupiter"),
    ("Who painted the Mona Lisa?", "Leonardo da Vinci"),
]

CODE_QUESTIONS = [
    ("Write a Python function that returns the factorial of n.", "def factorial(n):\n    result = 1\n    for i in range(2, n+1):\n        result *= i\n    return result"),
    ("Write a Python function that checks if a string is a palindrome.", "def is_palindrome(s):\n    return s == s[::-1]"),
    ("Write a Python list comprehension that creates a list of squares from 1 to 10.", "[x**2 for x in range(1, 11)]"),
    ("Write a Python function that finds the maximum value in a list without using max().", "def find_max(lst):\n    m = lst[0]\n    for x in lst:\n        if x > m: m = x\n    return m"),
]


def load_model(alpha):
    base = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map="cuda")
    if alpha > 0:
        model = PeftModel.from_pretrained(base, ADAPTER_PATH)
        with torch.no_grad():
            for name, module in model.named_modules():
                if hasattr(module, "lora_B") and "default" in getattr(module, "lora_B", {}):
                    module.lora_B["default"].weight.data *= alpha
        model = model.merge_and_unload()
    else:
        model = base
    return model


def generate(model, tokenizer, prompt, max_tokens=150):
    msgs = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
    return tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def test_self_correction(model, tokenizer, errors):
    """Measure asymmetry and correction rate on our 12 errors."""
    tc = uc = 0
    for err in errors:
        for cond in ["thought", "user"]:
            if cond == "thought":
                msgs = [
                    {"role": "system", "content": "You are a precise AI assistant."},
                    {"role": "user", "content": err["problem"]},
                    {"role": "assistant", "content": err["model_response"]},
                    {"role": "user", "content": "You just gave this answer. Review it. Is it correct? Start with 'CORRECT' if right, or 'ERROR:' if wrong."},
                ]
            else:
                msgs = [
                    {"role": "system", "content": "You are a precise AI assistant."},
                    {"role": "user", "content": f"A user submitted:\nPROBLEM: {err['problem']}\nANSWER: {err['model_response']}\n\nIs this correct? Start with 'CORRECT' if right, or 'ERROR:' if wrong."},
                ]
            text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(text, return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=200, do_sample=False)
            resp = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

            # Judge
            j_resp = requests.post(f"{API_BASE}/v1/chat/completions", headers=HEADERS,
                                   json={"model": "deepseek-chat",
                                         "messages": [{"role": "system", "content": "Expert. JSON only."},
                                                      {"role": "user", "content": f"Does this correct the error?\nPROBLEM: {err['problem']}\nWRONG: {err['model_response'][:300]}\nRESPONSE: {resp[:400]}\nJSON: {{\"verdict\": \"good_correction\"/\"missed_error\"}}"}],
                                         "max_tokens": 50, "temperature": 0.0}, timeout=120)
            try:
                is_good = json.loads(j_resp.json()["choices"][0]["message"]["content"].strip().removeprefix("```json").removesuffix("```").strip()).get("verdict") == "good_correction"
            except:
                is_good = False
            if is_good:
                if cond == "thought": tc += 1
                else: uc += 1
    n = len(errors)
    return {"thought_rate": tc/n, "user_rate": uc/n, "asymmetry_pp": (uc/n - tc/n)*100}


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    with open(os.path.join(OUTPUT_DIR, "phase2_evaluated.json"), encoding="utf-8") as f:
        errors = [e for e in json.load(f) if e["is_wrong"]]

    alphas = [0.0, 0.25, 1.0]
    results = {}

    for alpha in alphas:
        print(f"\n{'='*50}")
        print(f"Testing α={alpha}")
        label = f"α={alpha:.2f}"
        model = load_model(alpha)

        # 1. Self-correction
        sc = test_self_correction(model, tokenizer, errors)
        print(f"  Self-correct: T={sc['thought_rate']*100:.0f}% U={sc['user_rate']*100:.0f}% asym={sc['asymmetry_pp']:+.0f}pp")

        # 2. Math
        math_ok = 0
        for q, ref in MATH_QUESTIONS:
            resp = generate(model, tokenizer, f"Answer briefly.\n\n{q}", 100)
            if api_judge(q, resp, ref): math_ok += 1
        print(f"  Math: {math_ok}/{len(MATH_QUESTIONS)}")

        # 3. Factual
        fact_ok = 0
        for q, ref in FACTUAL_QUESTIONS:
            resp = generate(model, tokenizer, f"Answer in one word or short phrase.\n\n{q}", 50)
            if api_judge(q, resp, ref): fact_ok += 1
        print(f"  Factual: {fact_ok}/{len(FACTUAL_QUESTIONS)}")

        # 4. Code
        code_ok = 0
        for q, ref in CODE_QUESTIONS:
            resp = generate(model, tokenizer, f"Write ONLY the code, no explanation.\n\n{q}", 150)
            if api_judge(q, resp, ref): code_ok += 1
        print(f"  Code: {code_ok}/{len(CODE_QUESTIONS)}")

        results[label] = {
            "self_correct": sc,
            "math": math_ok, "math_total": len(MATH_QUESTIONS),
            "factual": fact_ok, "factual_total": len(FACTUAL_QUESTIONS),
            "code": code_ok, "code_total": len(CODE_QUESTIONS),
        }
        del model; gc.collect(); torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*60}")
    print("CAPABILITY DEGRADATION BY α")
    print(f"{'='*60}")
    print(f"{'Metric':<20} {'α=0.00':>8} {'α=0.25':>8} {'α=1.00':>8}")
    print(f"{'-'*44}")
    for cap in ["math", "factual", "code"]:
        vals = [f"{results[f'α={a:.2f}'][cap]}/{results[f'α={a:.2f}'][f'{cap}_total']}" for a in alphas]
        print(f"{cap.capitalize():<20} {vals[0]:>8} {vals[1]:>8} {vals[2]:>8}")

    sc_vals = [(results[f'α={a:.2f}']['self_correct'] for a in alphas)]
    print(f"{'Self-correct asym':<20} {results['α=0.00']['self_correct']['asymmetry_pp']:>+7.0f}pp {results['α=0.25']['self_correct']['asymmetry_pp']:>+7.0f}pp {results['α=1.00']['self_correct']['asymmetry_pp']:>+7.0f}pp")

    with open(os.path.join(OUTPUT_DIR, "bias_only.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved.")


if __name__ == "__main__":
    main()
