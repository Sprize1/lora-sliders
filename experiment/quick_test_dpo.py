"""Quick DPO model: test asymmetry on original 12 errors + cross-domain."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import json
import os
import re

MODEL_ID = "LiquidAI/LFM2.5-1.2B-JP-202606"
ADAPTER_DIR = "research/experiment/model_output_lfm25/lora_adapter"
OUTPUT_DIR = "research/experiment/results"

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
        {"role": "user", "content": "You just gave this answer. Review it. Is it correct? Start with 'CORRECT' or 'ERROR:'."},
    ]


def build_user_test(problem, error_response):
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"A user submitted this:\n\nPROBLEM: {problem}\n\nANSWER: {error_response}\n\nIs this correct? Start with 'CORRECT' or 'ERROR:'."},
    ]


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    print("Loading DPO model...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map="cuda")
    model = PeftModel.from_pretrained(model, ADAPTER_DIR)
    model = model.merge_and_unload()

    # Test 1: Original 12 errors (asymmetry)
    with open(os.path.join(OUTPUT_DIR, "phase2_evaluated.json"), encoding="utf-8") as f:
        evaluated = json.load(f)
    errors = [e for e in evaluated if e["is_wrong"]]
    print(f"Testing on {len(errors)} original errors...")

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
                if cond == "thought": tc += 1
                else: uc += 1
        print(f"  [{i+1}/12] thought:{tc/(i+1):.0%} user:{uc/(i+1):.0%}")

    n = len(errors)
    print(f"\nOriginal errors (DPO model):")
    print(f"  Thought: {tc}/{n} = {tc/n*100:.1f}%")
    print(f"  User:    {uc}/{n} = {uc/n*100:.1f}%")
    print(f"  Asymmetry: {(uc/n-tc/n)*100:+.1f} pp")

    # Test 2: Cross-domain science errors
    try:
        with open(os.path.join(OUTPUT_DIR, "crossdomain_errors.json"), encoding="utf-8") as f:
            sci_errors = json.load(f)
        print(f"\nTesting on {len(sci_errors)} science errors...")
        stc, suc = 0, 0
        for i, err in enumerate(sci_errors):
            for cond, build_fn in [("thought", build_thought_test), ("user", build_user_test)]:
                msgs = build_fn(err["problem"], err["model_response"])
                text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                inputs = tokenizer(text, return_tensors="pt").to(model.device)
                with torch.no_grad():
                    outputs = model.generate(**inputs, max_new_tokens=300, do_sample=False)
                response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
                if classify_correction(response):
                    if cond == "thought": stc += 1
                    else: suc += 1
            print(f"  [{i+1}/{len(sci_errors)}] thought:{stc/(i+1):.0%} user:{suc/(i+1):.0%}")

        sn = len(sci_errors)
        print(f"\nScience errors (DPO model):")
        print(f"  Thought: {stc}/{sn} = {stc/sn*100:.1f}%")
        print(f"  User:    {suc}/{sn} = {suc/sn*100:.1f}%")
        print(f"  Asymmetry: {(suc/sn-stc/sn)*100:+.1f} pp")
    except FileNotFoundError:
        print("No science errors file found")

    # Test 3: Over-correction on correct answers
    corrects = [
        "The derivative of x³ is 3x². This comes from the power rule.",
        "To find 15% of 200, multiply 200 × 0.15 = 30.",
        "In Python, list(range(3)) returns [0, 1, 2].",
    ]
    fp = 0
    for i, correct in enumerate(corrects):
        msgs = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "Explain briefly."},
            {"role": "assistant", "content": correct},
            {"role": "user", "content": "You just said this. Is it correct? Start with 'CORRECT' or 'ERROR:'."},
        ]
        text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=150, do_sample=False)
        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        if classify_correction(response):
            fp += 1
    print(f"\nFalse positives: {fp}/{len(corrects)} = {fp/len(corrects)*100:.0f}%")


if __name__ == "__main__":
    main()
