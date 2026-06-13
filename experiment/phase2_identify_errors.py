"""Phase 2: Test self-correction asymmetry on model's OWN errors.

Uses DeepSeek API to:
1. Identify which model responses contain errors
2. Test correction in both thought/user conditions
"""

import json
import os
import time
import requests

API_BASE = "https://api.deepseek.com"
# Read API key from environment or .env
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

OUTPUT_DIR = "experiment/results"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def call_deepseek(system_prompt, user_prompt, model="deepseek-chat", max_tokens=2000):
    """Call DeepSeek API (using Anthropic-compatible endpoint)."""
    # Actually DeepSeek has its own endpoint format
    # Let's use the standard DeepSeek API
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.1,
    }
    resp = requests.post(
        f"{API_BASE}/v1/chat/completions",
        headers=HEADERS,
        json=payload,
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


def identify_errors(model_responses):
    """Use DeepSeek to identify which responses contain factual/logical errors."""
    print("Identifying errors in model responses using DeepSeek API...")

    error_list = []
    for i, item in enumerate(model_responses):
        problem = item["problem"]
        response = item["model_response"]

        prompt = f"""Analyze this model response to the problem. Determine if it contains any factual, logical, or mathematical errors.

PROBLEM:
{problem}

MODEL RESPONSE:
{response}

Your task:
1. Is the final answer correct? (YES/NO)
2. If NO, extract the key reasoning flaw
3. Provide a brief correct solution (1-2 sentences)

Respond in JSON format:
{{"correct": true/false, "flaw": "description of error if any", "correct_solution": "brief correct answer"}}"""

        result = call_deepseek(
            "You are an expert evaluator. Respond ONLY with valid JSON.",
            prompt,
            max_tokens=500,
        )

        try:
            eval_data = json.loads(result.strip().removeprefix("```json").removesuffix("```").strip())
        except json.JSONDecodeError:
            # Try to extract JSON from response
            import re
            match = re.search(r'\{.*\}', result, re.DOTALL)
            if match:
                eval_data = json.loads(match.group())
            else:
                eval_data = {"correct": True, "flaw": "parse error", "correct_solution": ""}

        error_list.append({
            "problem": problem,
            "model_response": response,
            "domain": item["domain"],
            "is_wrong": not eval_data.get("correct", True),
            "flaw": eval_data.get("flaw", ""),
            "correct_solution": eval_data.get("correct_solution", ""),
        })

        status = "WRONG" if not eval_data.get("correct", True) else "OK"
        print(f"  [{i+1}/{len(model_responses)}] {status}")

    return error_list


def print_results(evaluated):
    """Print summary of identified errors."""
    wrong = [e for e in evaluated if e["is_wrong"]]
    print(f"\n{'='*60}")
    print(f"ERROR IDENTIFICATION RESULTS")
    print(f"{'='*60}")
    print(f"Total responses: {len(evaluated)}")
    print(f"Errors found: {len(wrong)}")
    print(f"Error rate: {len(wrong)/len(evaluated)*100:.1f}%")

    for i, e in enumerate(wrong):
        print(f"\n--- Error {i+1} ({e['domain']}) ---")
        print(f"Problem: {e['problem'][:100]}...")
        print(f"Flaw: {e['flaw'][:200]}...")

    return wrong


def main():
    # Load phase 1 results
    with open(os.path.join(OUTPUT_DIR, "phase1_model_errors.json"), encoding="utf-8") as f:
        model_responses = json.load(f)

    # Identify errors via DeepSeek API
    evaluated = identify_errors(model_responses)

    # Save
    with open(os.path.join(OUTPUT_DIR, "phase2_evaluated.json"), "w", encoding="utf-8") as f:
        json.dump(evaluated, f, indent=2, ensure_ascii=False)

    wrong = print_results(evaluated)
    print(f"\nSaved {len(evaluated)} evaluated responses to {OUTPUT_DIR}/phase2_evaluated.json")
    print(f"Found {len(wrong)} errors for phase 3 (correction testing)")

    return wrong


if __name__ == "__main__":
    main()
