"""Post-test: measure self-correction asymmetry after LoRA fine-tuning.

Re-runs Phase 3 (correction test) on the SAME 12 errors using the fine-tuned model.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import json
import os
import re

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER_DIR = "experiment/model_output/lora_adapter"
OUTPUT_DIR = "experiment/results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SYSTEM_PROMPT = "You are a precise AI assistant. You carefully verify information and correct errors when you find them."


def build_thought_test(problem, model_error_response):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": problem},
        {"role": "assistant", "content": model_error_response},
        {"role": "user", "content": "You just gave the answer above. Review it carefully. Is your answer correct? If you find any error, explain what is wrong and provide the correct answer. Start with 'CORRECT' if right, or 'ERROR:' if wrong."},
    ]
    return messages


def build_user_test(problem, model_error_response):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Another user submitted this answer to the problem:\n\nPROBLEM: {problem}\n\nUSER'S ANSWER: {model_error_response}\n\nIs the user's answer correct? If you find any error, explain what is wrong and provide the correct answer. Start with 'CORRECT' if right, or 'ERROR:' if wrong."},
    ]
    return messages


def classify_correction(response_text):
    text = response_text.strip()
    if text.upper().startswith("ERROR"):
        return True
    if text.upper().startswith("CORRECT"):
        return False
    markers = [
        r"(?:is|was|that(?:'s| is)) (?:wrong|incorrect|false|not correct|an error|flawed|mistaken)",
        r"(?:should be|correct (?:answer|solution|approach) is|actually)",
        r"(?:the (?:error|mistake|flaw|problem) (?:is|lies|occurs))",
        r"(?:fails to|does not|incorrectly)",
    ]
    for m in markers:
        if re.search(m, text, re.IGNORECASE):
            return True
    return False


def main():
    # Load errors from Phase 2
    with open(os.path.join(OUTPUT_DIR, "phase2_evaluated.json"), encoding="utf-8") as f:
        evaluated = json.load(f)
    errors = [e for e in evaluated if e["is_wrong"]]
    print(f"Loaded {len(errors)} verified errors from Phase 2")

    # Load base model + LoRA adapter
    print(f"Loading base model: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )

    print(f"Loading LoRA adapter: {ADAPTER_DIR}")
    model = PeftModel.from_pretrained(model, ADAPTER_DIR)
    model = model.merge_and_unload()
    print(f"VRAM: {torch.cuda.memory_allocated()//1024**3}GB")

    # Run post-test
    print(f"\nRunning post-test on {len(errors)} errors...")
    results = []

    for i, error in enumerate(errors):
        problem = error["problem"]
        model_response = error["model_response"]
        domain = error["domain"]

        print(f"\n[{i+1}/{len(errors)}] {domain}: {problem[:80]}...")

        for condition, build_fn in [("thought", build_thought_test), ("user", build_user_test)]:
            messages = build_fn(problem, model_response)
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(text, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=300, do_sample=False)
            response = tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
            )
            corrected = classify_correction(response)
            results.append({
                "error_index": i, "domain": domain, "condition": condition,
                "corrected": corrected, "response": response,
            })
            marker = "✓ CORRECTED" if corrected else "✗ MISSED"
            print(f"  {condition}: {marker}")

    # Analyze
    thought_r = [r for r in results if r["condition"] == "thought"]
    user_r = [r for r in results if r["condition"] == "user"]
    tc = sum(1 for r in thought_r if r["corrected"])
    uc = sum(1 for r in user_r if r["corrected"])
    n = len(errors)

    # Load pre-test metrics for comparison
    with open(os.path.join(OUTPUT_DIR, "phase3_asymmetry_metrics.json"), encoding="utf-8") as f:
        pretest = json.load(f)

    print(f"\n{'='*60}")
    print(f"POST-TEST vs PRE-TEST COMPARISON")
    print(f"{'='*60}")
    print(f"{'':<20} {'PRE':>10} {'POST':>10} {'DELTA':>10}")
    print(f"{'-'*50}")
    print(f"{'Thought rate':<20} {pretest['thought_rate']*100:>9.1f}% {tc/n*100:>9.1f}% {tc/n*100-pretest['thought_rate']*100:>+9.1f}pp")
    print(f"{'User rate':<20} {pretest['user_rate']*100:>9.1f}% {uc/n*100:>9.1f}% {uc/n*100-pretest['user_rate']*100:>+9.1f}pp")
    post_asym = (uc/n - tc/n) * 100
    pre_asym = pretest['asymmetry_pp']
    print(f"{'Asymmetry':<20} {pre_asym:>+9.1f}pp {post_asym:>+9.1f}pp {post_asym-pre_asym:>+9.1f}pp")
    print(f"{'-'*50}")
    asym_reduction = (pretest['asymmetry_pp'] - post_asym) / pretest['asymmetry_pp'] * 100 if pretest['asymmetry_pp'] > 0 else 0
    print(f"Asymmetry reduction: {asym_reduction:.1f}%")

    # Save
    post_metrics = {
        "thought_rate": tc / n,
        "user_rate": uc / n,
        "asymmetry_pp": post_asym,
        "asymmetry_reduction_pct": asym_reduction,
        "pre_asymmetry_pp": pre_asym,
        "results": results,
    }
    with open(os.path.join(OUTPUT_DIR, "posttest_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(post_metrics, f, indent=2, ensure_ascii=False)

    print(f"\nSaved to {OUTPUT_DIR}/posttest_metrics.json")


if __name__ == "__main__":
    main()
