"""Self-improvement on math with sympy verification. v2: actually uses model's answer."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
import os, re, gc, random, subprocess, sympy as sp
import numpy as np

MODEL_ID = "LiquidAI/LFM2.5-1.2B-JP-202606"
OUTPUT_DIR = "research/experiment/model_output_selfimprove_math"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# (problem, expected_answer, type)
# type: "numeric" (float comparison), "exact" (string equality after normalize),
#        "sympy" (sympy equivalence), "set" (set of numbers)
PROBLEMS = [
    ("Solve for x: 2^x = 64.", "6", "numeric"),
    ("Find d/dx of x^3 * sin(x).", "3*x**2*sin(x) + x**3*cos(x)", "sympy"),
    ("Sum of arithmetic series 3 + 7 + 11 + ... + 39.", "210", "numeric"),
    ("Solve |2x-3|+|x+1|=7. List all real solutions separated by commas.", "-3, 2.333333", "numeric_set"),
    ("Probability of exactly 2 heads in 4 coin flips, as simplified fraction.", "3/8", "numeric"),
    ("Find log_2(32) + log_3(81).", "9", "numeric"),
    ("Determinant of [[2,3,1],[0,-1,4],[2,0,3]].", "-19", "numeric"),
    ("Area between y=x^2 and y=2x from x=0 to 2.", "1.333333", "numeric"),
    ("Sum of all primes less than 30.", "129", "numeric"),
    ("Expected value of sum when rolling two fair dice.", "7", "numeric"),
    ("20th Fibonacci number (0-indexed: F0=0, F1=1).", "6765", "numeric"),
    ("Compound interest: $1000 at 5% for 3 years, total amount.", "1157.625", "numeric"),
    ("Limit of sin(3x)/x as x approaches 0.", "3", "numeric"),
    ("Number of distinct ways to arrange letters in MISSISSIPPI.", "34650", "numeric"),
    ("Inverse of matrix [[1,2],[3,4]] as [[a,b],[c,d]].", "[[-2, 1], [1.5, -0.5]]", "matrix"),
    ("Taylor series of e^x up to x^3 term.", "1 + x + x**2/2 + x**3/6", "sympy"),
    ("Solve sqrt(x+4) = x-2.", "5", "numeric"),
    ("Angle between vectors (1,2,3) and (4,-5,6) in degrees.", "63.6", "numeric_approx"),
    ("Sum of series sum(1/(n*(n+1))) from n=1 to infinity.", "1", "numeric"),
    ("Eigenvalues of [[3,1],[1,3]] as set.", "2, 4", "numeric_set"),
    ("Volume when y=x^2 rotated around x-axis from x=0 to 1, expressed with pi.", "pi/5", "sympy"),
    ("Integers 1-1000 divisible by 3 or 5.", "467", "numeric"),
    ("7^100 mod 11.", "1", "numeric"),
    ("sin(15°) exact expression with sqrt.", "sqrt(6)/4 - sqrt(2)/4", "sympy"),
    ("Integral of cos(x) * e^x from x=0 to pi.", "(exp(pi) + 1) / 2", "sympy_special_case"),
]


def verify_answer(model_answer, expected, answer_type):
    """Check if model's answer is correct given expected answer and type."""
    answer = model_answer.strip()

    if answer_type == "numeric":
        try:
            # Extract first number from answer
            nums = re.findall(r'-?\d+\.?\d*(?:/\d+)?', answer)
            if not nums: return False
            val = float(eval(nums[0])) if '/' in nums[0] else float(nums[0])
            exp = float(eval(expected)) if '/' in expected else float(expected)
            return abs(val - exp) < 1e-9
        except: return False

    elif answer_type == "numeric_approx":
        try:
            val = float(re.findall(r'-?\d+\.?\d*', answer)[0])
            exp = float(expected)
            return abs(val - exp) < 0.5
        except: return False

    elif answer_type == "numeric_set":
        try:
            nums = [float(n) for n in re.findall(r'-?\d+\.?\d*(?:/\d+)?', answer)]
            nums = [float(eval(str(n))) if '/' in str(n) else n for n in nums]
            exp_nums = [float(n) for n in re.findall(r'-?\d+\.?\d*(?:/\d+)?', expected)]
            exp_nums = [float(eval(str(n))) if '/' in str(e) else n for n in exp_nums]
            return set(round(n, 2) for n in nums) == set(round(n, 2) for n in exp_nums)
        except: return False

    elif answer_type == "sympy":
        try:
            # Sanitize model answer for sympy
            clean = answer.replace('^', '**').replace('{', '(').replace('}', ')')
            x = sp.Symbol('x')
            model_expr = sp.sympify(clean)
            exp_expr = sp.sympify(expected.replace('^', '**'))
            return sp.simplify(model_expr - exp_expr) == 0
        except: return False

    elif answer_type == "matrix":
        try:
            clean = answer.replace(' ', '')
            exp_clean = expected.replace(' ', '')
            return clean == exp_clean
        except: return False

    elif answer_type == "sympy_special_case":
        # Special: compare numerically
        try:
            x = sp.Symbol('x')
            clean = answer.replace('^', '**')
            model_expr = sp.sympify(clean)
            # Evaluate numerically at x=pi and compare
            model_val = float(model_expr.evalf())
            exp_val = float((sp.exp(sp.pi) + 1) / 2)
            return abs(model_val - exp_val) < 0.01
        except: return False

    return False


def extract_answer(text):
    """Extract final answer from model's reasoning."""
    text = text.strip()
    # \boxed{...} — highest priority
    m = re.search(r'\\boxed\{([^}]+)\}', text)
    if m: return m.group(1).strip()
    # Answer: ...
    m = re.search(r'(?im)(?:Answer|answer)\s*:\s*(.+?)(?:\n|$)', text)
    if m: return m.group(1).strip()
    # Last short line
    lines = [l.strip() for l in text.split("\n") if l.strip() and len(l.strip()) < 50]
    return lines[-1] if lines else text


def generate(model, tokenizer, problem, do_sample=True):
    prompt = f"Solve this math problem concisely. Put the final answer in \\boxed{{}}.\n\n{problem}"
    msgs = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=250, do_sample=do_sample,
                                 temperature=0.6 if do_sample else None,
                                 top_p=0.9 if do_sample else None)
    return tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def seq_log_prob(model, tokenizer, prompt_text, response_text):
    full = prompt_text + response_text
    enc = tokenizer(full, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    prompt_len = tokenizer(prompt_text, return_tensors="pt").to(model.device)["input_ids"].shape[1]
    outputs = model(**enc)
    logits = outputs.logits[:, prompt_len-1:-1, :]
    targets = enc["input_ids"][:, prompt_len:]
    lp = torch.nn.functional.log_softmax(logits, dim=-1)
    return (lp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)).sum(-1)


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading {MODEL_ID}...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto")
    target_modules = list(set(
        parts[-1] for name, module in model.named_modules()
        if hasattr(module, "weight") and module.weight.ndim >= 2 and module.weight.shape[0] > 100
        and (parts := name.split("."))[-1] in ["q_proj", "k_proj", "v_proj", "out_proj", "in_proj",
                                                 "w1", "w2", "w3"]
    ))
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, target_modules=target_modules,
                         lora_dropout=0.05, bias="none", task_type=TaskType.CAUSAL_LM))
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    # Baseline
    random.seed(42)
    print(f"\nBaseline (greedy) on {len(PROBLEMS)} problems...")
    baseline_pass = 0
    for problem, expected, atype in PROBLEMS:
        raw = generate(model, tokenizer, problem, do_sample=False)
        ans = extract_answer(raw)
        passed = verify_answer(ans, expected, atype)
        if passed: baseline_pass += 1
        else: print(f"  FAIL [{atype}]: expected='{expected}' got='{ans[:80]}'")
    print(f"Baseline: {baseline_pass}/{len(PROBLEMS)} ({baseline_pass/len(PROBLEMS)*100:.0f}%)")

    if baseline_pass == 0:
        print("ABORT: 0% baseline - verification broken or model can't do math.")
        return

    # Self-improvement
    n_rounds = 10
    batch_size = 6
    print(f"\nSelf-improvement: {n_rounds} rounds × {batch_size} problems")

    for rnd in range(n_rounds):
        indices = random.sample(range(len(PROBLEMS)), batch_size)
        rnd_pass = 0

        for idx in indices:
            problem, expected, atype = PROBLEMS[idx]
            prompt = f"Solve this math problem. Show step-by-step, then 'Answer: <answer>'.\n\n{problem}"
            raw = generate(model, tokenizer, problem, do_sample=True)
            ans = extract_answer(raw)
            passed = verify_answer(ans, expected, atype)

            prompt_text = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
            )
            log_prob = seq_log_prob(model, tokenizer, prompt_text, raw)
            reward = 1.0 if passed else -1.0
            ( -log_prob * reward ).backward()
            if passed: rnd_pass += 1

            if (rnd_pass + (0 if passed else 1)) % 2 == 0:
                optimizer.step()
                optimizer.zero_grad()

        print(f"  Round {rnd+1}: pass={rnd_pass}/{batch_size}")

    # Final
    print(f"\nFinal (greedy):")
    final_pass = 0
    for problem, expected, atype in PROBLEMS:
        raw = generate(model, tokenizer, problem, do_sample=False)
        ans = extract_answer(raw)
        if verify_answer(ans, expected, atype): final_pass += 1
    print(f"Final: {final_pass}/{len(PROBLEMS)}")
    print(f"Baseline: {baseline_pass}/{len(PROBLEMS)}")
    print(f"Delta: {final_pass - baseline_pass:+d}")

    model.save_pretrained(os.path.join(OUTPUT_DIR, "lora_adapter"))
    print(f"Done.")


if __name__ == "__main__":
    main()
