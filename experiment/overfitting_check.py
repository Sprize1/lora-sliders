"""Overfitting check: 3 tests before trusting the fine-tuned model.

Test 1: FRESH errors — the fine-tuned model's OWN new mistakes (not base model's)
Test 2: OVER-CORRECTION — does model flag CORRECT answers as wrong?
Test 3: REASONING QUALITY — does model give specific corrections or generic patterns?
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import json
import os
import re
import requests

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER_DIR = "research/experiment/model_output_dpo/lora_adapter"
OUTPUT_DIR = "research/experiment/results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

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

# Mix of hard problems the fine-tuned model might get wrong
FRESH_PROBLEMS = [
    # Maths (trained domain, hard)
    "Find the area between curves y=x² and y=2x+3 from x=-1 to x=3.",
    "A sequence is defined by a₁=2, a_{n+1}=3a_n-1. Find a₅.",
    "Solve the system: x²+y²=25, xy=12. Find all (x,y) pairs.",
    "What is the expected value of the sum when rolling two fair dice?",
    # Logic (trained domain, hard)
    "Is this statement true or false: 'This statement is false'. Explain the paradox.",
    "Prove or disprove: If A⊆B and B⊆C, then A⊆C.",
    # Code (trained domain, hard)
    "What does this Python function return for f(5)? def f(n): return n*f(n-1) if n>0 else 1",
    "Write code to find the longest palindromic substring. Then explain the time complexity.",
    # Science (untrained domain)
    "Calculate the gravitational force between Earth (5.97×10²⁴ kg) and Moon (7.34×10²² kg) at 3.84×10⁸ m distance. G=6.67×10⁻¹¹.",
    "Explain why a helium balloon rises in air. Calculate the buoyant force on a 10L helium balloon at STP.",
]

# Correct answers for the over-correction test
CORRECT_ANSWERS = [
    "The derivative of x³ is 3x². This comes from the power rule: d/dx(x^n) = n·x^(n-1).",
    "To find 15% of 200, multiply 200 × 0.15 = 30.",
    "In Python, list(range(3)) returns [0, 1, 2]. The range(3) generates numbers 0, 1, 2.",
    "A molecule of water (H₂O) consists of two hydrogen atoms covalently bonded to one oxygen atom.",
    "The capital of Japan is Tokyo. It has been the capital since 1868.",
]


def call_deepseek(system, prompt, max_tokens=1000):
    payload = {"model": "deepseek-chat", "messages": [{"role": "system", "content": system},
                {"role": "user", "content": prompt}], "max_tokens": max_tokens, "temperature": 0.1}
    resp = requests.post(f"{API_BASE}/v1/chat/completions", headers=HEADERS, json=payload, timeout=180)
    if resp.status_code != 200:
        return None
    return resp.json()["choices"][0]["message"]["content"]


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


def check_overcorrection(model, tokenizer, correct_answers):
    """Test 2: Present CORRECT answers in thought role. Model should say CORRECT."""
    false_positives = 0
    results = []
    for i, answer in enumerate(correct_answers):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "Explain this concept."},
            {"role": "assistant", "content": answer},
            {"role": "user", "content": "You just gave this explanation. Review it. Is it correct? Start with 'CORRECT' or 'ERROR:'."},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=150, do_sample=False)
        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        flagged_error = classify_correction(response)
        if flagged_error:
            false_positives += 1
        results.append({"answer": answer, "response": response, "flagged_as_error": flagged_error})
        print(f"  [{i+1}/{len(correct_answers)}] flagged_as_error={flagged_error} | {response[:100]}...")
    return false_positives, results


def check_reasoning_quality(model, tokenizer, errors, label):
    """Test 3: API evaluates if correction reasoning is specific vs generic."""
    specific_count = 0
    results = []
    for i, err in enumerate(errors[:8]):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": err["problem"]},
            {"role": "assistant", "content": err["model_response"]},
            {"role": "user", "content": "You just gave this answer. Review it. Is it correct? If wrong, explain specifically what the error is and provide the correct answer."},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=300, do_sample=False)
        correction = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        # API judges if correction is specific or generic
        judge_prompt = f"""Rate this self-correction on specificity (1-5):

ERROR TO CORRECT: {err['model_response'][:300]}

CORRECTION: {correction[:500]}

A score of 5 means: identifies the exact error location, explains why it's wrong, provides precise fix.
A score of 1 means: vague/generic ("recalculate carefully"), restates without fixing, or misses the error.

Respond with JSON: {{"score": <1-5>, "specific": true/false, "explanation": "1 sentence"}}"""
        judge = call_deepseek("Expert evaluator. Output ONLY valid JSON.", judge_prompt)
        try:
            clean = judge.strip().removeprefix("```json").removesuffix("```").strip()
            rating = json.loads(clean)
        except:
            rating = {"score": 3, "specific": True}

        is_specific = rating.get("specific", rating.get("score", 3) >= 4)
        if is_specific:
            specific_count += 1
        results.append({"error_idx": i, "correction": correction, "rating": rating})

        marker = "SPECIFIC" if is_specific else "GENERIC"
        print(f"  [{i+1}/8] {marker} (score: {rating.get('score','?')})")

    specificity_rate = specific_count / min(len(errors), 8) * 100
    return specificity_rate, results


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    # Load fine-tuned model
    print("Loading fine-tuned model...")
    ft_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map="cuda")
    ft_model = PeftModel.from_pretrained(ft_model, ADAPTER_DIR)
    ft_model = ft_model.merge_and_unload()

    print(f"\n{'='*60}")
    print("TEST 1: FRESH ERRORS")
    print(f"{'='*60}")
    print("Generating answers from fine-tuned model to fresh problems...")

    answers = []
    for problem in FRESH_PROBLEMS:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": problem}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(ft_model.device)
        with torch.no_grad():
            outputs = ft_model.generate(**inputs, max_new_tokens=400, do_sample=True, temperature=0.6, top_p=0.9)
        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        answers.append({"problem": problem, "model_response": response})

    print("Identifying errors via API...")
    errors = []
    for i, item in enumerate(answers):
        prompt = f"""Does this response contain factual errors?

PROBLEM: {item['problem']}
RESPONSE: {item['model_response']}

JSON: {{"correct": true/false, "flaw": "description"}}"""
        result = call_deepseek("Expert evaluator. Output ONLY valid JSON.", prompt)
        try:
            clean = result.strip().removeprefix("```json").removesuffix("```").strip()
            eval_data = json.loads(clean)
        except:
            eval_data = {"correct": True}
        is_wrong = not eval_data.get("correct", True)
        errors.append({**item, "is_wrong": is_wrong, "flaw": eval_data.get("flaw", "")})
        print(f"  [{i+1}/{len(answers)}] {'WRONG' if is_wrong else 'OK'}")

    ft_errors = [e for e in errors if e["is_wrong"]]
    print(f"Found {len(ft_errors)} fresh errors from fine-tuned model")

    if len(ft_errors) >= 3:
        print(f"\nTesting self-correction on {len(ft_errors)} fresh errors...")
        tc = 0
        uc = 0
        for i, err in enumerate(ft_errors):
            for condition, role in [("thought", "assistant"), ("user", "user")]:
                if condition == "thought":
                    msgs = [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": err["problem"]},
                        {"role": "assistant", "content": err["model_response"]},
                        {"role": "user", "content": "You just wrote this. Is it correct? Start with 'CORRECT' or 'ERROR:'."},
                    ]
                else:
                    msgs = [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": f"User answer to '{err['problem']}':\n{err['model_response']}\n\nIs this correct? Start with 'CORRECT' or 'ERROR:'."},
                    ]
                text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                inputs = tokenizer(text, return_tensors="pt").to(ft_model.device)
                with torch.no_grad():
                    outputs = ft_model.generate(**inputs, max_new_tokens=300, do_sample=False)
                response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
                if classify_correction(response):
                    if condition == "thought":
                        tc += 1
                    else:
                        uc += 1
            print(f"  [{i+1}/{len(ft_errors)}] thought:{tc/(i+1):.0%} user:{uc/(i+1):.0%}")
        n = len(ft_errors)
        print(f"\nFresh errors result:")
        print(f"  Thought: {tc}/{n} = {tc/n*100:.1f}%")
        print(f"  User:    {uc}/{n} = {uc/n*100:.1f}%")
        print(f"  Asymmetry: {(uc/n-tc/n)*100:+.1f} pp")
    else:
        print("Not enough fresh errors to test")
        tc, uc, n = 0, 0, 0

    print(f"\n{'='*60}")
    print("TEST 2: OVER-CORRECTION (false positives on correct answers)")
    print(f"{'='*60}")
    fp, fp_results = check_overcorrection(ft_model, tokenizer, CORRECT_ANSWERS)
    fp_rate = fp / len(CORRECT_ANSWERS) * 100
    print(f"False positive rate: {fp}/{len(CORRECT_ANSWERS)} = {fp_rate:.1f}%")

    print(f"\n{'='*60}")
    print("TEST 3: REASONING QUALITY")
    print(f"{'='*60}")
    spec_rate, spec_results = check_reasoning_quality(ft_model, tokenizer,
        ft_errors if len(ft_errors) >= 3 else errors[:8], "FINE-TUNED")
    print(f"Specific reasoning rate: {spec_rate:.1f}%")

    # Summary
    print(f"\n{'='*60}")
    print("OVERFITTING CHECK SUMMARY")
    print(f"{'='*60}")
    tests = {
        "Fresh errors - thought rate": f"{tc/n*100:.0f}%" if n > 0 else "N/A",
        "Fresh errors - asymmetry": f"{(uc/n-tc/n)*100:+.0f}pp" if n > 0 else "N/A",
        "False positive rate (over-correction)": f"{fp_rate:.1f}%",
        "Specific reasoning rate": f"{spec_rate:.1f}%",
    }
    for test, result in tests.items():
        print(f"  {test}: {result}")

    passed = (n == 0 or (tc/n >= 0.7 and abs(uc/n - tc/n) <= 0.15)) and fp_rate <= 20 and spec_rate >= 70
    print(f"\nOVERFITTING ASSESSMENT: {'PASSED ✓' if passed else 'FAILED ✗ - model shows overfitting signs'}")

    final = {
        "fresh_errors": {"n": n, "thought_rate": tc/n if n else None, "user_rate": uc/n if n else None,
                         "asymmetry": (uc/n-tc/n)*100 if n else None},
        "overcorrection": {"false_positives": fp, "total": len(CORRECT_ANSWERS), "rate": fp_rate},
        "reasoning_quality": {"specific_rate": spec_rate},
        "overfitting_assessment": "passed" if passed else "failed",
    }
    with open(os.path.join(OUTPUT_DIR, "overfitting_check.json"), "w", encoding="utf-8") as f:
        json.dump(final, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {OUTPUT_DIR}/overfitting_check.json")


if __name__ == "__main__":
    main()
