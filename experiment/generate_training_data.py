"""Generate training dataset: error-correction pairs in both thought/user roles.

Uses DeepSeek API to generate diverse (problem, incorrect reasoning, correction) tuples.
Formats each in both thought and user role variants.
Target: ~500 examples across maths, logic, code domains.
"""

import json
import os
import re
import time
import requests

# API config
API_BASE = "https://api.deepseek.com"
API_KEY = None
env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
with open(env_path) as f:
    for line in f:
        if "ANTHROPIC_AUTH_TOKEN" in line:
            API_KEY = line.strip().split("=", 1)[1]
            break

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}

OUTPUT_DIR = "experiment/data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SYSTEM_PROMPT = "You are a precise AI assistant. You carefully verify information and correct errors when you find them."

DOMAINS = {
    "maths": "mathematics, algebra, calculus, probability, geometry, number theory, word problems",
    "logic": "logical reasoning, syllogisms, fallacies, deductions, truth tables, paradoxes, formal logic",
    "code": "Python programming, algorithms, data structures, time complexity, debugging, programming concepts",
}


def call_deepseek(system, user_prompt, model="deepseek-chat", max_tokens=4000):
    """Call DeepSeek API v1/chat/completions."""
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.8,
    }
    resp = requests.post(
        f"{API_BASE}/v1/chat/completions",
        headers=HEADERS,
        json=payload,
        timeout=180,
    )
    if resp.status_code != 200:
        print(f"  API error {resp.status_code}: {resp.text[:200]}")
        return None
    return resp.json()["choices"][0]["message"]["content"]


def generate_batch(domain, domain_description, batch_size=50):
    """Generate a batch of training examples for a domain."""
    prompt = f"""Generate {batch_size} training examples for teaching an AI to self-correct.

DOMAIN: {domain} ({domain_description})

For each example, provide:
1. A problem/question
2. A PLAUSIBLE BUT INCORRECT answer with detailed reasoning (the kind a student or junior AI would give)
3. The CORRECT answer with proper reasoning
4. A brief explanation of what the error was and why it matters

The incorrect answer should:
- Sound confident and well-reasoned at first glance
- Contain a SPECIFIC factual, logical, or mathematical error
- NOT be obviously wrong (no "2+2=5" level errors)
- Be the kind of mistake a real person could make

Format as JSON array:
```json
[
  {{
    "problem": "question text",
    "wrong_reasoning": "detailed incorrect solution with the hidden flaw",
    "correct_reasoning": "proper solution with correct reasoning",
    "error_explanation": "concise explanation of what was wrong"
  }},
  ...
]
```

Return ONLY the valid JSON array, no other text. Generate EXACTLY {batch_size} diverse examples."""

    result = call_deepseek(
        "You are an expert curriculum designer. You generate high-quality training data. Output ONLY valid JSON.",
        prompt,
        max_tokens=8000,
    )
    return result


def parse_json_response(text):
    """Extract JSON array from API response."""
    if text is None:
        return []
    text = text.strip()
    # Remove markdown code blocks
    text = re.sub(r'^```(?:json)?\s*', '', text)
    text = re.sub(r'\s*```$', '', text)
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
        elif isinstance(data, dict) and "examples" in data:
            return data["examples"]
        return []
    except json.JSONDecodeError:
        # Try to find array
        match = re.search(r'\[.*\]', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        return []


def format_as_chat(example, variant="thought"):
    """Format an example as a chat conversation for fine-tuning.

    variant='thought': error appears as model's own assistant response
    variant='user': error appears as user's message
    """
    if variant == "thought":
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["problem"]},
            {"role": "assistant", "content": example["wrong_reasoning"]},
            {"role": "user", "content": "You just gave the answer above. Review it carefully. Is your answer correct? If you find any error, explain what is wrong and provide the correct answer."},
            {"role": "assistant", "content": f"ERROR: {example['error_explanation']}\n\nCorrect answer: {example['correct_reasoning']}"},
        ]
    else:  # user
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"A user submitted this answer to the problem:\n\nPROBLEM: {example['problem']}\n\nUSER'S ANSWER: {example['wrong_reasoning']}\n\nIs the user's answer correct? If you find any error, explain what is wrong and provide the correct answer."},
            {"role": "assistant", "content": f"ERROR: {example['error_explanation']}\n\nCorrect answer: {example['correct_reasoning']}"},
        ]
    return {"messages": messages}


def main():
    all_examples = []
    batches_per_domain = {  # Adjust to reach ~500 total
        "maths": 4,   # 4 × 50 = 200
        "logic": 3,   # 3 × 50 = 150
        "code": 3,    # 3 × 50 = 150
    }

    for domain, num_batches in batches_per_domain.items():
        print(f"\n{'='*60}")
        print(f"Generating {domain} training data ({num_batches} batches × 50)...")
        print(f"{'='*60}")

        domain_examples = []
        for batch_num in range(num_batches):
            print(f"  Batch {batch_num+1}/{num_batches}...")
            raw = generate_batch(domain, DOMAINS[domain], batch_size=50)
            parsed = parse_json_response(raw)
            print(f"    Parsed: {len(parsed)} examples")
            domain_examples.extend(parsed)
            if batch_num < num_batches - 1:
                time.sleep(2)  # Rate limit

        print(f"  Total {domain}: {len(domain_examples)} examples")
        all_examples.extend(domain_examples)

    print(f"\n{'='*60}")
    print(f"Generated {len(all_examples)} total raw examples")

    # Save raw
    with open(os.path.join(OUTPUT_DIR, "training_raw.json"), "w", encoding="utf-8") as f:
        json.dump(all_examples, f, indent=2, ensure_ascii=False)

    # Format as chat conversations in BOTH variants
    print(f"\nFormatting chat conversations (thought + user variants)...")
    thought_format = [format_as_chat(ex, "thought") for ex in all_examples]
    user_format = [format_as_chat(ex, "user") for ex in all_examples]

    # We want model to learn both formats, so we combine them
    # But with a 60/40 split favoring thought (we want to specifically
    # teach the model to correct its OWN errors)
    num_repeat_thought = 1  # 1 copy of thought
    num_repeat_user = 0     # No user examples needed for this experiment
    # Actually, we want the model to see errors in BOTH roles so it learns
    # that correction is role-independent. 50/50 split is better.

    all_formatted = thought_format + user_format
    print(f"  Total: {len(all_formatted)} formatted conversations")

    # Save formatted
    with open(os.path.join(OUTPUT_DIR, "training_formatted.json"), "w", encoding="utf-8") as f:
        json.dump(all_formatted, f, indent=2, ensure_ascii=False)

    # Also save as JSONL for easier training
    with open(os.path.join(OUTPUT_DIR, "training_formatted.jsonl"), "w", encoding="utf-8") as f:
        for item in all_formatted:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\nSaved to {OUTPUT_DIR}/")
    print(f"  training_raw.json: {len(all_examples)} raw examples")
    print(f"  training_formatted.json: {len(all_formatted)} chat conversations")
    print(f"  training_formatted.jsonl: {len(all_formatted)} lines")


if __name__ == "__main__":
    main()
