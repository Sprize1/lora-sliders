"""Generate BALANCED training dataset with both error and correct examples.

Fix: ~60% error examples (ERROR:), ~40% correct examples (CORRECT:)
This teaches model to actually evaluate, not reflexively say "ERROR".
"""

import json
import os
import re
import requests

API_BASE = "https://api.deepseek.com"
API_KEY = None
env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), ".env")
with open(env_path) as f:
    for line in f:
        if "ANTHROPIC_AUTH_TOKEN" in line:
            API_KEY = line.strip().split("=", 1)[1]
            break
HEADERS = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

OUTPUT_DIR = "research/experiment/data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SYSTEM_PROMPT = "You are a precise AI assistant. You carefully verify information and correct errors when you find them."


def call_deepseek(system, prompt, max_tokens=8000):
    payload = {"model": "deepseek-chat", "messages": [{"role": "system", "content": system},
                {"role": "user", "content": prompt}], "max_tokens": max_tokens, "temperature": 0.8}
    resp = requests.post(f"{API_BASE}/v1/chat/completions", headers=HEADERS, json=payload, timeout=180)
    if resp.status_code != 200:
        print(f"  API error: {resp.status_code}")
        return None
    return resp.json()["choices"][0]["message"]["content"]


def generate_error_examples(domain, domain_desc, batch_size=40):
    """Generate examples where the answer is WRONG (ERROR: examples)."""
    prompt = f"""Generate {batch_size} training examples for teaching self-correction.

DOMAIN: {domain} ({domain_desc})

For each example:
1. A problem/question
2. A PLAUSIBLE BUT INCORRECT answer with reasoning
3. The CORRECT answer with reasoning
4. Brief explanation of the error

The incorrect answer should be subtly wrong, not obviously so.

JSON array: [{{"problem": "...", "wrong_reasoning": "...", "correct_reasoning": "...", "error_explanation": "..."}}, ...]
Return ONLY valid JSON array, exactly {batch_size} items."""
    return call_deepseek("Expert curriculum designer. Output ONLY valid JSON.", prompt)


def generate_correct_examples(domain, domain_desc, batch_size=30):
    """Generate examples where the answer is CORRECT (CORRECT: examples)."""
    prompt = f"""Generate {batch_size} training examples where the AI gave a CORRECT answer and must confirm it.

DOMAIN: {domain} ({domain_desc})

For each example:
1. A problem/question
2. A CORRECT answer with thorough, accurate reasoning
3. A brief explanation of WHY it's correct (what makes it right)

The correctness should be verifiable - no ambiguity.

JSON array: [{{"problem": "...", "correct_reasoning": "...", "why_correct": "..."}}, ...]
Return ONLY valid JSON array, exactly {batch_size} items."""
    return call_deepseek("Expert curriculum designer. Output ONLY valid JSON.", prompt)


def parse_json(text):
    if text is None:
        return []
    text = text.strip()
    text = re.sub(r'^```(?:json)?\s*', '', text)
    text = re.sub(r'\s*```$', '', text)
    try:
        return json.loads(text) if isinstance(json.loads(text), list) else []
    except:
        match = re.search(r'\[.*\]', text, re.DOTALL)
        return json.loads(match.group()) if match else []


def format_error(example):
    """Format WRONG example in both roles."""
    correction_text = f"ERROR: {example['error_explanation']}\n\nCorrect answer: {example['correct_reasoning']}"
    thought = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": example["problem"]},
        {"role": "assistant", "content": example["wrong_reasoning"]},
        {"role": "user", "content": "You just gave this answer. Review it carefully. Is it correct? Start with 'CORRECT' if it's right, or 'ERROR:' if there's a mistake."},
        {"role": "assistant", "content": correction_text},
    ]
    user = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"A user submitted this to '{example['problem']}':\n\n{example['wrong_reasoning']}\n\nIs this correct? Start with 'CORRECT' if it's right, or 'ERROR:' if there's a mistake."},
        {"role": "assistant", "content": correction_text},
    ]
    return [{"messages": thought}, {"messages": user}]


def format_correct(example):
    """Format CORRECT example in both roles."""
    confirmation = f"CORRECT: {example['why_correct']}"
    thought = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": example["problem"]},
        {"role": "assistant", "content": example["correct_reasoning"]},
        {"role": "user", "content": "You just gave this answer. Review it carefully. Is it correct? Start with 'CORRECT' if it's right, or 'ERROR:' if there's a mistake."},
        {"role": "assistant", "content": confirmation},
    ]
    user = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"A user submitted this to '{example['problem']}':\n\n{example['correct_reasoning']}\n\nIs this correct? Start with 'CORRECT' if it's right, or 'ERROR:' if there's a mistake."},
        {"role": "assistant", "content": confirmation},
    ]
    return [{"messages": thought}, {"messages": user}]


def main():
    # Load existing error examples (we can reuse most of them)
    print("Loading existing error examples...")
    with open("research/experiment/data/training_raw.json", encoding="utf-8") as f:
        old_data = json.load(f)
    print(f"  {len(old_data)} existing error examples")

    # Generate new error examples to fill gaps
    all_formatted = []

    # Reuse existing errors (~200)
    for ex in old_data[:200]:
        all_formatted.extend(format_error(ex))

    # Generate fresh correct examples (need ~130, target 40% of ~330 total)
    print("\nGenerating CORRECT examples (target: ~130)...")
    domains = {
        "maths": "mathematics, algebra, geometry, probability, calculus",
        "logic": "logic, syllogisms, fallacies, formal reasoning, deduction",
        "code": "Python, algorithms, data structures, debugging, time complexity",
    }
    correct_examples = []
    for domain, desc in domains.items():
        for batch in range(2):  # 2 batches × 25 per domain
            print(f"  {domain} batch {batch+1}...")
            raw = generate_correct_examples(domain, desc, batch_size=25)
            parsed = parse_json(raw)
            print(f"    got {len(parsed)} correct examples")
            correct_examples.extend(parsed)

    print(f"\nTotal correct examples: {len(correct_examples)}")
    for ex in correct_examples:
        all_formatted.extend(format_correct(ex))

    # Stats
    n_correct = sum(1 for item in all_formatted if item["messages"][-1]["content"].startswith("CORRECT"))
    n_error = sum(1 for item in all_formatted if item["messages"][-1]["content"].startswith("ERROR"))
    print(f"\nBalanced dataset: {len(all_formatted)} conversations")
    print(f"  ERROR examples: {n_error} ({n_error/len(all_formatted)*100:.0f}%)")
    print(f"  CORRECT examples: {n_correct} ({n_correct/len(all_formatted)*100:.0f}%)")

    # Save
    with open(os.path.join(OUTPUT_DIR, "training_balanced.jsonl"), "w", encoding="utf-8") as f:
        for item in all_formatted:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"\nSaved to {OUTPUT_DIR}/training_balanced.jsonl")


if __name__ == "__main__":
    main()
