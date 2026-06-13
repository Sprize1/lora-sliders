"""SCoRe-style on-policy RL for self-correction.

Key difference from SFT/DPO: model learns from its OWN errors, not pre-written ones.
- Model generates answers → API identifies errors
- Model attempts correction → API rates quality
- REINFORCE with KL penalty to prevent collapse
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
import json
import os
import re
import requests
import random
import gc

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
OUTPUT_DIR = "research/experiment/model_output_score"
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

# Diverse problems across domains
PROBLEMS = [
    # Maths
    "Find x: 3^(x+1) = 81",
    "A car travels 240 km. First half at 60 km/h, second half at 80 km/h. Average speed?",
    "Probability of rolling sum 7 with two fair dice?",
    "Solve: |2x-5| = 9",
    "Find derivative of f(x) = x·sin(x)",
    "Sum of first 50 positive integers?",
    "Area of triangle with base 12 and height 8?",
    "If f(x) = 2x²-3x+1, find f(-2)",
    "Compound interest on $500 at 4% for 3 years?",
    "Solve: sqrt(x+5) = x-1",
    # Logic
    "If all A are B, and no B are C, can some A be C?",
    "Is 'If it rains, ground wet. Ground wet. Therefore it rained' valid?",
    "Three people: one always lies, one always tells truth, one random. How to identify?",
    "Prove: √2 is irrational",
    "If P→Q is true and P is false, what can we conclude about Q?",
    # Code
    "What does Python's enumerate(['a','b','c']) return?",
    "Time complexity of finding max element in unsorted array?",
    "Difference between list.append() and list.extend() in Python?",
    "What does 'lambda x: x**2' mean in Python?",
    "How to reverse a string in Python?",
    # Science
    "Why does ice float on water?",
    "Calculate force: 5kg mass, 2 m/s² acceleration",
    "What happens to pressure if volume halves at constant temp?",
    "Why is the sky blue?",
    "Difference between weight and mass?",
]


def call_api(system, prompt, max_tokens=1000, temp=0.1):
    payload = {"model": "deepseek-chat", "messages": [{"role": "system", "content": system},
                {"role": "user", "content": prompt}], "max_tokens": max_tokens, "temperature": temp}
    resp = requests.post(f"{API_BASE}/v1/chat/completions", headers=HEADERS, json=payload, timeout=120)
    return resp.json()["choices"][0]["message"]["content"] if resp.status_code == 200 else None


def generate_answer(model, tokenizer, problem):
    """Model generates initial answer."""
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": problem}]
    text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=200, do_sample=True, temperature=0.7, top_p=0.9)
    return tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def verify_answer(problem, answer):
    """API checks if answer is correct."""
    prompt = f"""Is this answer to the problem correct?

PROBLEM: {problem}
ANSWER: {answer}

JSON: {{"correct": true/false, "error_type": "factual/logic/calculation/none", "brief_fix": "if wrong, 1-sentence fix"}}"""
    result = call_api("Expert evaluator. Output ONLY valid JSON.", prompt, max_tokens=300)
    try:
        clean = result.strip().removeprefix("```json").removesuffix("```").strip()
        return json.loads(clean)
    except:
        return {"correct": True}


def generate_correction(model, tokenizer, problem, wrong_answer):
    """Model attempts to correct its own error (thought role)."""
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": problem},
        {"role": "assistant", "content": wrong_answer},
        {"role": "user", "content": "You just gave this answer. Review it carefully. Is it correct? If you find any error, explain specifically what is wrong and provide the correct answer. Start with 'CORRECT' if right, or 'ERROR:' if wrong."},
    ]
    text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=300, do_sample=True, temperature=0.7, top_p=0.9)
    return tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def rate_correction(problem, wrong_answer, correction, expected_fix):
    """API rates correction quality: 1-5 scale."""
    prompt = f"""Rate this self-correction on a 1-5 scale.

PROBLEM: {problem}
WRONG ANSWER: {wrong_answer[:300]}
EXPECTED FIX: {expected_fix}
CORRECTION: {correction[:500]}

Scoring:
5 = Exactly identifies the specific error, explains why it's wrong, provides precise correct solution
4 = Identifies error correctly, provides correct solution, but explanation could be clearer
3 = Identifies that there's an error and provides a correct-ish fix, but vague about what was wrong
2 = Says there's an error but fix is wrong or generic ("recalculate")
1 = Completely misses the error or says CORRECT when the answer has a clear error
0 = Says ERROR when the answer was actually correct (false positive - WORST case)

JSON: {{"score": <0-5 int>, "specific": true/false, "fixes_error": true/false, "explanation": "1 sentence"}}"""
    result = call_api("Expert evaluator. Output ONLY valid JSON.", prompt, max_tokens=300)
    try:
        clean = result.strip().removeprefix("```json").removesuffix("```").strip()
        return json.loads(clean)
    except:
        return {"score": 2, "specific": False, "fixes_error": False}


def compute_log_prob(model, tokenizer, prompt_text, response_text):
    """Compute log probability of response given prompt (with grad)."""
    full = prompt_text + response_text
    enc = tokenizer(full, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    prompt_enc = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=1536)
    prompt_len = prompt_enc["input_ids"].shape[1]

    outputs = model(**enc)
    logits = outputs.logits[:, prompt_len-1:-1, :]
    targets = enc["input_ids"][:, prompt_len:]
    lp = torch.nn.functional.log_softmax(logits, dim=-1)
    token_lp = lp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return token_lp.sum(-1)


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model + LoRA
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto")
    lora_config = LoraConfig(
        r=16, lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05, bias="none", task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Reference model (frozen base) for KL penalty
    print("Loading reference model...")
    ref_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto")
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    beta = 0.05  # KL penalty coefficient
    baseline_reward = 3.0  # moving baseline

    print(f"\nSCoRe-style RL: {len(PROBLEMS)} problems, ~50 iterations")
    all_problems = PROBLEMS * 3  # Reuse problems with different sampling
    random.shuffle(all_problems)

    for iteration in range(50):
        # Phase 1: Generate and verify
        problem = all_problems[iteration]
        answer = generate_answer(model, tokenizer, problem)
        verification = verify_answer(problem, answer)

        if verification.get("correct", True):
            # Answer is correct — generate CORRECT confirmation for training
            msgs_correct = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": problem},
                {"role": "assistant", "content": answer},
                {"role": "user", "content": "You just gave this answer. Is it correct? Start with 'CORRECT' or 'ERROR:'."},
            ]
            prompt_text = tokenizer.apply_chat_template(msgs_correct, tokenize=False, add_generation_prompt=True)
            response_text = "CORRECT: The reasoning is sound and the answer is correct."

            log_prob = compute_log_prob(model, tokenizer, prompt_text, response_text)
            with torch.no_grad():
                log_prob_ref = compute_log_prob(ref_model, tokenizer, prompt_text, response_text)

            reward = 1.0  # Small reward for correctly confirming
            kl_penalty = (log_prob - log_prob_ref).detach()
            loss = -log_prob * (reward - beta * kl_penalty)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if iteration % 5 == 0:
                print(f"  iter {iteration}: correct answer, reward={reward:.1f}")

        else:
            # Answer is wrong — attempt correction
            correction = generate_correction(model, tokenizer, problem, answer)
            rating = rate_correction(problem, answer, correction, verification.get("brief_fix", ""))

            score = rating.get("score", 2)
            reward = (score - 3) / 2.0  # Normalize: score 5→+1, 3→0, 1→-1, 0→-1.5
            if score == 0:
                reward = -2.0  # Heavy penalty for false positive

            # Build prompt for this correction
            msgs = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": problem},
                {"role": "assistant", "content": answer},
                {"role": "user", "content": "You just gave this answer. Review it. Is it correct? Start with 'CORRECT' or 'ERROR:'."},
            ]
            prompt_text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

            log_prob = compute_log_prob(model, tokenizer, prompt_text, correction)
            with torch.no_grad():
                log_prob_ref = compute_log_prob(ref_model, tokenizer, prompt_text, correction)

            kl_penalty = (log_prob - log_prob_ref).detach()
            adjusted_reward = reward - beta * kl_penalty

            loss = -log_prob * adjusted_reward
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Update baseline
            baseline_reward = 0.9 * baseline_reward + 0.1 * reward

            if iteration % 5 == 0:
                specifics = "SPECIFIC" if rating.get("specific") else "generic"
                print(f"  iter {iteration}: wrong, score={score}, reward={reward:+.1f}, {specifics}, "
                      f"KL={kl_penalty.item():.2f}, loss={loss.item():.4f}")

        # Free ref model VRAM occasionally
        if iteration % 10 == 9:
            torch.cuda.empty_cache()

    del ref_model
    gc.collect()
    torch.cuda.empty_cache()

    # Save
    adapter_dir = os.path.join(OUTPUT_DIR, "lora_adapter")
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    print(f"\nSCoRe adapter saved to {adapter_dir}")


if __name__ == "__main__":
    main()
