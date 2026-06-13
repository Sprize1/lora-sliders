"""Self-improvement loop: model generates code → execute tests → reward → LoRA update.

Bypasses the self-correction asymmetry by using a deterministic verifier.
Signal is binary: code passes tests or not. No API judge, no subjectivity.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
import json
import os
import subprocess
import tempfile
import re
import gc

MODEL_ID = "LiquidAI/LFM2.5-1.2B-JP-202606"
OUTPUT_DIR = "research/experiment/model_output_selfimprove"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Problems: (description, test_input, expected_output)
PROBLEMS = [
    ("Write a function f(x) that returns x squared plus 2x minus 1.",
     "print(f(3))", "14"),
    ("Write a function is_even(n) that returns True if n is even, False otherwise.",
     "print(is_even(4)); print(is_even(7))", "True\nFalse"),
    ("Write a function factorial(n) that returns n! (n factorial). Use a loop.",
     "print(factorial(5))", "120"),
    ("Write a function count_vowels(s) that returns the number of vowels (aeiou) in string s.",
     "print(count_vowels('hello world'))", "3"),
    ("Write a function reverse_list(lst) that returns a new list with elements in reverse order.",
     "print(reverse_list([1, 2, 3, 4, 5]))", "[5, 4, 3, 2, 1]"),
    ("Write a function is_palindrome(s) that returns True if string s reads the same forwards and backwards.",
     "print(is_palindrome('radar')); print(is_palindrome('hello'))", "True\nFalse"),
    ("Write a function sum_digits(n) that returns the sum of all digits in integer n.",
     "print(sum_digits(12345))", "15"),
    ("Write a function fibonacci(n) that returns the nth Fibonacci number (0-indexed: fib(0)=0, fib(1)=1).",
     "print(fibonacci(10))", "55"),
    ("Write a function find_max(lst) that returns the maximum value in a list without using max().",
     "print(find_max([3, 7, 2, 9, 1]))", "9"),
    ("Write a function gcd(a, b) that returns the greatest common divisor using Euclid's algorithm.",
     "print(gcd(48, 18))", "6"),
    ("Write a function binary_search(arr, target) that returns the index of target in sorted arr, or -1 if not found.",
     "print(binary_search([1, 3, 5, 7, 9], 5)); print(binary_search([1, 3, 5, 7, 9], 6))", "2\n-1"),
    ("Write a function merge_sorted(a, b) that merges two sorted lists into one sorted list.",
     "print(merge_sorted([1, 3, 5], [2, 4, 6]))", "[1, 2, 3, 4, 5, 6]"),
    ("Write a function remove_duplicates(lst) that returns a new list with duplicates removed, preserving order.",
     "print(remove_duplicates([1, 2, 2, 3, 1, 4]))", "[1, 2, 3, 4]"),
    ("Write a function is_prime(n) that returns True if n is prime.",
     "print(is_prime(17)); print(is_prime(100))", "True\nFalse"),
    ("Write a function char_count(s) that returns a dict with count of each character in s.",
     "d = char_count('hello'); print(d['h'], d['e'], d['l'], d['o'])", "1 1 2 1"),
    ("Write a function flatten(lst) that flattens a nested list by one level.",
     "print(flatten([[1, 2], [3, 4], [5]]))", "[1, 2, 3, 4, 5]"),
    ("Write a function rotate_left(lst, k) that rotates list left by k positions.",
     "print(rotate_left([1, 2, 3, 4, 5], 2))", "[3, 4, 5, 1, 2]"),
    ("Write a function matrix_mult(A, B) that multiplies two 2x2 matrices represented as lists of lists.",
     "print(matrix_mult([[1, 2], [3, 4]], [[5, 6], [7, 8]]))", "[[19, 22], [43, 50]]"),
    ("Write a function longest_word(s) that returns the longest word in a string.",
     "print(longest_word('the quick brown fox jumped'))", "jumped"),
    ("Write a function solve_quadratic(a, b, c) that returns the real roots of ax²+bx+c=0 as a sorted list.",
     "print(solve_quadratic(1, -5, 6))", "[2.0, 3.0]"),
    ("Write a function pascal_row(n) that returns the nth row of Pascal's triangle as a list (0-indexed).",
     "print(pascal_row(4))", "[1, 4, 6, 4, 1]"),
    ("Write a function is_anagram(s1, s2) that returns True if s1 and s2 are anagrams.",
     "print(is_anagram('listen', 'silent')); print(is_anagram('hello', 'world'))", "True\nFalse"),
    ("Write a function running_sum(lst) that returns a list of running sums.",
     "print(running_sum([1, 2, 3, 4, 5]))", "[1, 3, 6, 10, 15]"),
    ("Write a function nth_prime(n) that returns the nth prime number (0-indexed: 0th=2, 1st=3).",
     "print(nth_prime(0)); print(nth_prime(5))", "2\n13"),
    ("Write a function missing_number(lst) that finds the missing number in a list of 0..n with one missing.",
     "print(missing_number([0, 1, 2, 4, 5]))", "3"),
]


def execute_and_test(code, test_code, expected_output):
    """Execute Python code and check output against expected. Returns (passed, output, error)."""
    full_code = f"{code}\n\n{test_code}"
    try:
        result = subprocess.run(
            ["python", "-c", full_code],
            capture_output=True, text=True, timeout=5,
            cwd=tempfile.gettempdir()
        )
        if result.returncode != 0:
            return False, "", result.stderr[:200]
        output = result.stdout.strip()
        expected = expected_output.strip()
        return output == expected, output, "" if output == expected else f"Expected:\n{expected}\nGot:\n{output}"
    except subprocess.TimeoutExpired:
        return False, "", "Timeout (>5s)"
    except Exception as e:
        return False, "", str(e)[:200]


def extract_code(text):
    """Extract Python code from model response (between ```python``` or just the whole text)."""
    # Try to get code from markdown blocks
    match = re.search(r'```(?:python)?\s*\n(.*?)```', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Otherwise return everything
    return text.strip()


def generate_solution(model, tokenizer, problem):
    """Model generates code solution."""
    prompt = f"Write a Python function that solves this problem. Output ONLY the code, no explanation.\n\n{problem}"
    msgs = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=300, do_sample=True, temperature=0.6, top_p=0.9)
    return tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def generate_correction(model, tokenizer, problem, wrong_code, error_msg):
    """Model attempts to fix broken code."""
    prompt = f"""This code failed the tests:

PROBLEM: {problem}

CODE:
```python
{wrong_code}
```

ERROR:
{error_msg}

Fix the code. Output ONLY the corrected code, no explanation."""
    msgs = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=300, do_sample=True, temperature=0.6, top_p=0.9)
    return tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def compute_seq_log_prob(model, tokenizer, prompt_text, response_text):
    """Log probability of response given prompt."""
    full = prompt_text + response_text
    enc = tokenizer(full, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    prompt_enc = tokenizer(prompt_text, return_tensors="pt").to(model.device)
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

    print(f"Loading {MODEL_ID}...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto")
    target_modules = list(set(
        parts[-1] for name, module in model.named_modules()
        if hasattr(module, "weight") and module.weight.ndim >= 2 and module.weight.shape[0] > 100
        and (parts := name.split("."))[-1] in ["q_proj", "k_proj", "v_proj", "out_proj", "in_proj",
                                                 "w1", "w2", "w3", "gate_proj", "up_proj", "down_proj"]
    ))
    lora_config = LoraConfig(r=16, lora_alpha=32, target_modules=target_modules,
                             lora_dropout=0.05, bias="none", task_type=TaskType.CAUSAL_LM)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    # Baseline: test all problems without training
    print(f"\nBaseline: testing {len(PROBLEMS)} problems...")
    baseline_pass = 0
    for problem, test_code, expected in PROBLEMS:
        raw = generate_solution(model, tokenizer, problem)
        code = extract_code(raw)
        passed, _, _ = execute_and_test(code, test_code, expected)
        if passed: baseline_pass += 1
    print(f"Baseline: {baseline_pass}/{len(PROBLEMS)} passed ({baseline_pass/len(PROBLEMS)*100:.0f}%)")

    # Self-improvement loop
    n_rounds = 5
    n_problems_per_round = 8
    print(f"\nSelf-improvement: {n_rounds} rounds × {n_problems_per_round} problems")

    for round_num in range(n_rounds):
        round_pass = 0
        round_fixed = 0
        round_loss = 0

        # Sample problems for this round
        indices = torch.randperm(len(PROBLEMS))[:n_problems_per_round].tolist()

        for idx in indices:
            problem, test_code, expected = PROBLEMS[idx]

            # Phase 1: Generate initial solution
            prompt = f"Write a Python function that solves this problem. Output ONLY the code, no explanation.\n\n{problem}"
            raw_solution = generate_solution(model, tokenizer, problem)
            code = extract_code(raw_solution)
            passed, output, error = execute_and_test(code, test_code, expected)

            if passed:
                # Positive reward
                prompt_text = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
                )
                log_prob = compute_seq_log_prob(model, tokenizer, prompt_text, raw_solution)
                reward = 1.0
                loss = -log_prob * reward
                loss.backward()
                round_loss += loss.item()
                round_pass += 1
            else:
                # Phase 2: Attempt correction
                raw_fix = generate_correction(model, tokenizer, problem, code, error)
                fix_code = extract_code(raw_fix)
                fix_passed, fix_output, fix_error = execute_and_test(fix_code, test_code, expected)

                correction_prompt = f"""This code failed the tests:

PROBLEM: {problem}

CODE:
```python
{code}
```

ERROR:
{error}

Fix the code. Output ONLY the corrected code, no explanation."""
                prompt_text = tokenizer.apply_chat_template(
                    [{"role": "user", "content": correction_prompt}], tokenize=False, add_generation_prompt=True
                )
                log_prob = compute_seq_log_prob(model, tokenizer, prompt_text, raw_fix)
                reward = 0.5 if fix_passed else -0.5
                loss = -log_prob * reward
                loss.backward()
                round_loss += loss.item()
                if fix_passed: round_fixed += 1

            # Update every 2 problems
            if (round_pass + round_fixed + (0 if passed else 1)) % 2 == 0:
                optimizer.step()
                optimizer.zero_grad()

        print(f"  Round {round_num+1}: pass={round_pass}/{n_problems_per_round}, "
              f"fixed={round_fixed}, loss={round_loss:.2f}")

    # Final test
    print(f"\nFinal test: all {len(PROBLEMS)} problems...")
    final_pass = 0
    for problem, test_code, expected in PROBLEMS:
        raw = generate_solution(model, tokenizer, problem)
        code = extract_code(raw)
        passed, _, _ = execute_and_test(code, test_code, expected)
        if passed: final_pass += 1
    print(f"Final: {final_pass}/{len(PROBLEMS)} passed ({final_pass/len(PROBLEMS)*100:.0f}%)")
    print(f"Baseline was: {baseline_pass}/{len(PROBLEMS)} ({baseline_pass/len(PROBLEMS)*100:.0f}%)")
    print(f"Delta: {final_pass - baseline_pass:+d}")

    adapter_dir = os.path.join(OUTPUT_DIR, "lora_adapter")
    model.save_pretrained(adapter_dir)
    print(f"\nSaved to {adapter_dir}")


if __name__ == "__main__":
    main()
