"""Quick baseline: test Qwen3-1.7B self-correction asymmetry on original 12 errors.

If Qwen3 (2025) shows less asymmetry than Qwen2.5 (2024), the problem is model quality, not parameter count.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import os
import re

MODEL_ID = "Qwen/Qwen3-1.7B"
OUTPUT_DIR = "research/experiment/results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SYSTEM_PROMPT = "You are a precise AI assistant. You carefully verify information and correct errors when you find them."


def classify_correction(text):
    text = text.strip()
    if text.upper().startswith("ERROR"):
        return True
    if text.upper().startswith("CORRECT"):
        return False
    markers = [r"(?:is|was) (?:wrong|incorrect|false|not correct|flawed|mistaken|an error)",
               r"(?:should be|correct (?:answer|solution) is|actually)"]
    return any(re.search(m, text, re.IGNORECASE) for m in markers)


def build_thought_test(problem, error_response):
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": problem},
        {"role": "assistant", "content": error_response},
        {"role": "user", "content": "You just gave this answer. Review it carefully. Is it correct? Start with 'CORRECT' if it's right, or 'ERROR:' if there's a mistake."},
    ]


def build_user_test(problem, error_response):
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"A user submitted this answer:\n\nPROBLEM: {problem}\n\nANSWER: {error_response}\n\nIs this correct? Start with 'CORRECT' if it's right, or 'ERROR:' if there's a mistake."},
    ]


def main():
    print(f"Loading {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    vram = torch.cuda.memory_allocated() // 1024**3
    print(f"VRAM: {vram}GB / {torch.cuda.get_device_properties(0).total_memory//1024**3}GB")

    # Load original 12 errors
    with open(os.path.join(OUTPUT_DIR, "phase2_evaluated.json"), encoding="utf-8") as f:
        evaluated = json.load(f)
    errors = [e for e in evaluated if e["is_wrong"]]
    print(f"Testing on {len(errors)} errors...")

    tc, uc = 0, 0
    for i, err in enumerate(errors):
        for cond, build_fn in [("thought", build_thought_test), ("user", build_user_test)]:
            msgs = build_fn(err["problem"], err["model_response"])
            text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(text, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=300, do_sample=False)
            response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            if classify_correction(response):
                if cond == "thought":
                    tc += 1
                else:
                    uc += 1
        print(f"  [{i+1}/12] thought:{tc/(i+1):.0%} user:{uc/(i+1):.0%}")

    n = len(errors)
    asym = (uc/n - tc/n) * 100
    print(f"\nQwen3-1.7B BASELINE:")
    print(f"  Thought: {tc}/{n} = {tc/n*100:.1f}%")
    print(f"  User:    {uc}/{n} = {uc/n*100:.1f}%")
    print(f"  Asymmetry: {asym:+.1f} pp")
    print(f"\n  Qwen2.5-1.5B was: Thought 25% / User 100% / Asymmetry +75pp")

    # Save
    result = {"model": MODEL_ID, "thought_rate": tc/n, "user_rate": uc/n, "asymmetry_pp": asym}
    with open(os.path.join(OUTPUT_DIR, "qwen3_baseline.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to qwen3_baseline.json")


if __name__ == "__main__":
    main()
