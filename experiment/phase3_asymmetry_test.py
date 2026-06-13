"""Phase 3: Test self-correction asymmetry on model's OWN errors.

For each verified error from Phase 2, test correction in:
- Condition THOUGHT: error appears as model's own assistant response
- Condition USER: same error appears as user's message
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import os
import re

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
OUTPUT_DIR = "experiment/results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SYSTEM_PROMPT = "You are a precise AI assistant. You carefully verify information and correct errors when you find them."


def build_thought_test(problem, model_error_response):
    """Error appears as model's own previous response."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": problem},
        {"role": "assistant", "content": model_error_response},
        {"role": "user", "content": "You just gave the answer above. Review it carefully. Is your answer correct? If you find any error, explain what is wrong and provide the correct answer. Start with 'CORRECT' if right, or 'ERROR:' if wrong."},
    ]
    return messages


def build_user_test(problem, model_error_response):
    """Same error appears as a user's message."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Another user submitted this answer to the problem:\n\nPROBLEM: {problem}\n\nUSER'S ANSWER: {model_error_response}\n\nIs the user's answer correct? If you find any error, explain what is wrong and provide the correct answer. Start with 'CORRECT' if right, or 'ERROR:' if wrong."},
    ]
    return messages


def classify_correction(response_text):
    """Classify whether model identified and corrected the error."""
    text = response_text.strip()
    # Direct markers
    if text.upper().startswith("ERROR"):
        return True
    if text.upper().startswith("CORRECT"):
        return False
    # Heuristics
    correction_markers = [
        r"(?:is|was|that(?:'s| is)) (?:wrong|incorrect|false|not correct|an error|flawed|mistaken)",
        r"(?:should be|correct (?:answer|solution|approach) is|actually)",
        r"(?:the (?:error|mistake|flaw|problem) (?:is|lies|occurs))",
        r"(?:fails to|does not|incorrectly)",
    ]
    for marker in correction_markers:
        if re.search(marker, text, re.IGNORECASE):
            return True
    return False


def run_correction_test(model, tokenizer, errors):
    """Run the asymmetry test on each error in both conditions."""
    results = []

    for i, error in enumerate(errors):
        problem = error["problem"]
        model_response = error["model_response"]
        domain = error["domain"]
        expected_flaw = error.get("flaw", "")

        print(f"\n[{i+1}/{len(errors)}] {domain}: {problem[:80]}...")

        for condition, build_fn in [("thought", build_thought_test), ("user", build_user_test)]:
            messages = build_fn(problem, model_response)
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(text, return_tensors="pt").to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=300,
                    do_sample=False,
                )
            response = tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            )

            corrected = classify_correction(response)
            results.append({
                "error_index": i,
                "domain": domain,
                "problem": problem,
                "condition": condition,
                "model_original_response": model_response,
                "correction_response": response,
                "corrected": corrected,
                "expected_flaw": expected_flaw,
            })

            marker = "✓ CORRECTED" if corrected else "✗ MISSED"
            print(f"  {condition}: {marker}")

    return results


def analyze(results, n_errors):
    """Compute asymmetry metrics."""
    thought_results = [r for r in results if r["condition"] == "thought"]
    user_results = [r for r in results if r["condition"] == "user"]

    thought_corrected = sum(1 for r in thought_results if r["corrected"])
    user_corrected = sum(1 for r in user_results if r["corrected"])

    # Per-domain breakdown
    domains = {}
    for tr, ur in zip(thought_results, user_results):
        d = tr["domain"]
        if d not in domains:
            domains[d] = {"thought": 0, "user": 0, "total": 0}
        domains[d]["total"] += 1
        if tr["corrected"]:
            domains[d]["thought"] += 1
        if ur["corrected"]:
            domains[d]["user"] += 1

    print(f"\n{'='*60}")
    print(f"PHASE 3: SELF-CORRECTION ASYMMETRY (MODEL'S OWN ERRORS)")
    print(f"{'='*60}")

    for d, s in sorted(domains.items()):
        tc, uc, t = s["thought"], s["user"], s["total"]
        asym = (uc/t - tc/t) * 100
        print(f"\n{d.upper()} ({t} errors):")
        print(f"  Thought correction: {tc}/{t} = {tc/t*100:.1f}%")
        print(f"  User correction:    {uc}/{t} = {uc/t*100:.1f}%")
        print(f"  Asymmetry:          {asym:+.1f} pp")

    overall_asym = (user_corrected/n_errors - thought_corrected/n_errors) * 100
    print(f"\n{'OVERALL':-^40}")
    print(f"  Thought: {thought_corrected}/{n_errors} = {thought_corrected/n_errors*100:.1f}%")
    print(f"  User:    {user_corrected}/{n_errors} = {user_corrected/n_errors*100:.1f}%")
    print(f"  Asymmetry: {overall_asym:+.1f} pp")

    return {
        "thought_rate": thought_corrected / n_errors,
        "user_rate": user_corrected / n_errors,
        "asymmetry_pp": overall_asym,
        "per_domain": {
            d: {
                "thought_rate": s["thought"] / s["total"],
                "user_rate": s["user"] / s["total"],
                "asymmetry_pp": (s["user"]/s["total"] - s["thought"]/s["total"]) * 100,
            }
            for d, s in domains.items()
        },
    }


def main():
    # Load errors from Phase 2
    with open(os.path.join(OUTPUT_DIR, "phase2_evaluated.json"), encoding="utf-8") as f:
        evaluated = json.load(f)

    errors = [e for e in evaluated if e["is_wrong"]]
    print(f"Loaded {len(errors)} verified errors from Phase 2")

    # Load model
    print(f"Loading model: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )

    # Run test
    results = run_correction_test(model, tokenizer, errors)

    # Save raw
    with open(os.path.join(OUTPUT_DIR, "phase3_correction_test.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Analyze
    metrics = analyze(results, len(errors))
    with open(os.path.join(OUTPUT_DIR, "phase3_asymmetry_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
