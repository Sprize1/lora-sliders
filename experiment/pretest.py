"""Pre-test: measure thought/user self-correction asymmetry in Qwen2.5-1.5B."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import re
import os

# HSA override must be set BEFORE importing torch in the main process
# Run with: HSA_OVERRIDE_GFX_VERSION=11.0.0 python pretest.py

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
OUTPUT_DIR = "experiment/results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Test questions with deliberately wrong answers injected
# Each: (question, correct_answer, wrong_answer, domain)
TEST_ITEMS = [
    # === MATHS (subtle reasoning errors) ===
    ("A train travels 120 km in 1.5 hours. What is its average speed in km/h?",
     "80 km/h",
     "Average speed = 120 × 1.5 = 180 km/h. Because distance times time gives speed.",
     "maths"),
    ("If a shirt costs $40 after a 20% discount, what was the original price?",
     "$50",
     "If 20% off means the price is 80% of original, then original = $40 × 1.2 = $48.",
     "maths"),
    ("What is the probability of getting at least one head when flipping 3 fair coins?",
     "7/8",
     "Probability of at least one head = 1/2 + 1/2 + 1/2 = 3/2 = 150%, which is impossible so it must be 3/6 = 1/2.",
     "maths"),
    ("Find the value of x: 2^(x+1) = 32",
     "x = 4",
     "2^(x+1) = 32, so x+1 = 32/2 = 16, therefore x = 15.",
     "maths"),
    ("What is the slope of a line perpendicular to y = 3x + 2?",
     "-1/3",
     "A perpendicular line has the same slope but opposite sign, so slope = -3.",
     "maths"),
    ("If sin(θ) = 0.6, what is cos(θ)? (0° < θ < 90°)",
     "0.8",
     "sin²(θ) + cos²(θ) = 1, so cos²(θ) = 1 - 0.6 = 0.4, therefore cos(θ) = sqrt(0.4) ≈ 0.63.",
     "maths"),
    ("Find the compound interest on $1000 at 5% for 2 years.",
     "$102.50",
     "5% of $1000 is $50. Over 2 years: 2 × $50 = $100 interest.",
     "maths"),
    ("What is the determinant of matrix [[2, 3], [1, 4]]?",
     "5",
     "Determinant = 2 × 4 + 3 × 1 = 8 + 3 = 11.",
     "maths"),
    ("A bag has 4 red and 6 blue marbles. If 2 are drawn without replacement, what is P(both red)?",
     "4/10 × 3/9 = 12/90 = 2/15",
     "P(both red) = (4/10) × (4/10) = 16/100 = 4/25. Replacements don't matter for independent events.",
     "maths"),
    ("If f(x) = ln(x²), what is f'(x)?",
     "2/x",
     "f'(x) = 1/(x²) × 2x = 2/x? Wait, actually derivative of ln(x²) = 1/(x²). Because derivative of ln(g(x)) = 1/g(x).",
     "maths"),
    ("What is the sum of the arithmetic series: 3 + 7 + 11 + ... + 39?",
     "210",
     "Number of terms: (39-3)/4 = 9 terms. Sum = n(a1+an)/2 = 9(3+39)/2 = 9×42/2 = 9×21 = 189.",
     "maths"),
    ("Solve for x: sqrt(x+4) = x - 2",
     "x = 5",
     "Square both sides: x+4 = x² - 4, so x² - x - 8 = 0. Using quadratic formula: x = (1 ± sqrt(33))/2.",
     "maths"),

    # === LOGIC (subtle fallacies) ===
    ("If most A are B, and most B are C, what can we conclude about A and C?",
     "Nothing — 'most' is not transitive",
     "Most A are C, because if most A are B and most B are C, then by transitivity most A must be C.",
     "logic"),
    ("A study finds people who drink green tea live longer. Does this prove green tea causes longevity?",
     "No — correlation does not imply causation. Green tea drinkers might exercise more, have better diets, or be wealthier.",
     "Yes, the data clearly shows a link. Large sample sizes eliminate confounding variables.",
     "logic"),
    ("Is this argument valid? 'If it rains, I'll take an umbrella. I took an umbrella. Therefore it rained.'",
     "No, this is the fallacy of affirming the consequent",
     "Yes, the logic follows correctly. If the conditional is true and the consequent is true, the antecedent must be true.",
     "logic"),
    ("Critique this reasoning: 'Every time I wash my car, it rains the next day. Therefore washing my car causes rain.'",
     "This is the post hoc ergo propter hoc fallacy — correlation in time does not prove causation. Likely confirmation bias.",
     "This is valid inductive reasoning. Repeated observation confirms the causal link. The pattern is too consistent to be chance.",
     "logic"),
    ("Is this a valid deduction? 'No athletes are lazy. Some students are lazy. Therefore some students are not athletes.'",
     "Yes, this is valid (Celarent + conversion)",
     "No, because 'some students are lazy' doesn't exclude the possibility that all lazy students are also athletes.",
     "logic"),
    ("Evaluate: 'Either we raise taxes or the deficit will grow. We can't raise taxes. Therefore the deficit will grow.'",
     "Valid (disjunctive syllogism: P or Q, not P, therefore Q)",
     "This is a false dilemma. There might be other options like cutting spending or economic growth increasing revenue.",
     "logic"),
    ("'The theory of evolution says humans descended from monkeys. But monkeys still exist! So evolution is false.' What's wrong?",
     "This misrepresents evolution — it says humans and monkeys share a common ancestor, not that humans descended from modern monkeys.",
     "This is a valid point. If humans evolved from monkeys, monkeys should have evolved too or gone extinct. Their continued existence is problematic.",
     "logic"),
    ("Is this reasoning sound? 'All birds have wings. Bats have wings. Therefore bats are birds.'",
     "No, the first premise would need to be 'All creatures with wings are birds' which is false. The argument is valid but unsound.",
     "Yes, this follows logically. If all birds have wings and bats have wings, bats qualify as birds under the definition.",
     "logic"),

    # === CODE (subtle bugs) ===
    ("What does this Python code print? `x = [1, 2]; y = x; y.append(3); print(x)`",
     "[1, 2, 3]",
     "[1, 2] because y is a separate variable and modifying it doesn't affect x.",
     "code"),
    ("What's wrong with this code? `def f(a=[]): a.append(1); return a`",
     "The mutable default argument [] is created once and shared across all calls. f() returns [1], f() again returns [1,1].",
     "Nothing is wrong. Each call to f() creates a fresh empty list. Calling f() twice returns [1] both times.",
     "code"),
    ("What does this output? `print(0.1 + 0.2 == 0.3)`",
     "False (floating-point precision: 0.1+0.2 = 0.30000000000000004)",
     "True. Basic arithmetic: 0.1 + 0.2 = 0.3 exactly.",
     "code"),
    ("What does this list comprehension do? `[x for x in range(10) if x % 2 == 0]`",
     "[0, 2, 4, 6, 8]",
     "[2, 4, 6, 8, 10] — all even numbers up to and including 10.",
     "code"),
    ("What's the issue? `try: 1/0; except: pass` (all on one line in Python)",
     "After the semicolon, `except` is a syntax error — it must be on a new line or part of a proper try-except block.",
     "The code catches the ZeroDivisionError and continues normally. The pass statement handles it correctly.",
     "code"),
    ("What happens: `from math import *; print(ceil(3.1))`",
     "NameError — math imports 'ceil' not 'ceil'. The function is `ceil`.",
     "Prints 4. The math module exports all its functions including ceil().",
     "code"),
    ("What's the output? `print(type(lambda x: x))`",
     "<class 'function'>",
     "<class 'lambda'>. Lambda is a distinct type in Python like int or str.",
     "code"),
    ("What does `sorted(['11', '2', '1'])` return in Python?",
     "['1', '11', '2'] (lexicographic string sorting)",
     "['1', '2', '11'] because Python automatically converts numeric strings to numbers when sorting.",
     "code"),
]

SYSTEM_PROMPT = "You are a precise AI assistant. You carefully verify information and correct errors when you find them."

def build_thought_prompt(wrong_answer, question):
    """Error appears as model's own previous response (assistant role)."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
        {"role": "assistant", "content": wrong_answer},
        {"role": "user", "content": (
            "You just gave the answer above. Please review it carefully. "
            "Is it correct? If there is an error, explain what is wrong and provide the correct answer. "
            "Start with 'CORRECT' if the answer is right, or 'ERROR:' if there is a mistake."
        )},
    ]
    return messages


def build_user_prompt(wrong_answer, question):
    """Error appears as coming from a user (external role)."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": (
            f"A user submitted this answer to the question '{question}':\n\n"
            f"'{wrong_answer}'\n\n"
            f"Is this answer correct? If there is an error, explain what is wrong "
            f"and provide the correct answer. "
            f"Start with 'CORRECT' if the answer is right, or 'ERROR:' if there is a mistake."
        )},
    ]
    return messages


def classify_response(response_text):
    """Classify whether the model corrected the error or not."""
    # Check if model identifies the error
    text = response_text.strip().upper()
    if text.startswith("ERROR"):
        return True  # Model corrected the error
    elif text.startswith("CORRECT"):
        return False  # Model failed to identify the error
    # Fallback: check for correction patterns
    correction_patterns = [
        r"(?:is|was|that(?:'s| is)) (?:wrong|incorrect|false|not correct|an error)",
        r"(?:should be|correct answer is|actually)",
        r"(?:the error|the mistake|the problem)",
    ]
    for pattern in correction_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    return False


def run_pretest(model, tokenizer, test_items):
    """Run the asymmetry pre-test."""
    results = []

    for question, correct, wrong, domain in test_items:
        # Build both prompt variants
        thought_msgs = build_thought_prompt(wrong, question)
        user_msgs = build_user_prompt(wrong, question)

        # Generate responses
        responses = {}
        for condition, msgs in [("thought", thought_msgs), ("user", user_msgs)]:
            text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(text, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=150,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                )
            response = tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            )
            corrected = classify_response(response)
            responses[condition] = {
                "response": response,
                "corrected": corrected,
            }

        results.append({
            "question": question,
            "correct": correct,
            "wrong": wrong,
            "domain": domain,
            "thought": responses["thought"],
            "user": responses["user"],
        })

    return results


def analyze_results(results):
    """Compute asymmetry metrics."""
    domains = {}
    for r in results:
        d = r["domain"]
        if d not in domains:
            domains[d] = {"thought_corrected": 0, "user_corrected": 0, "total": 0}
        domains[d]["total"] += 1
        if r["thought"]["corrected"]:
            domains[d]["thought_corrected"] += 1
        if r["user"]["corrected"]:
            domains[d]["user_corrected"] += 1

    print("\n" + "=" * 60)
    print("PRE-TEST RESULTS: Self-Correction Asymmetry")
    print("=" * 60)

    total_thought = sum(d["thought_corrected"] for d in domains.values())
    total_user = sum(d["user_corrected"] for d in domains.values())
    total_items = sum(d["total"] for d in domains.values())

    for domain, stats in sorted(domains.items()):
        tc = stats["thought_corrected"]
        uc = stats["user_corrected"]
        t = stats["total"]
        asymmetry = (uc / t * 100) - (tc / t * 100)
        print(f"\n{domain.upper()} ({t} items):")
        print(f"  Thought correction rate: {tc}/{t} = {tc/t*100:.1f}%")
        print(f"  User correction rate:    {uc}/{t} = {uc/t*100:.1f}%")
        print(f"  Asymmetry (user - thought): {asymmetry:+.1f} pp")

    overall_asymmetry = (total_user / total_items * 100) - (total_thought / total_items * 100)
    print(f"\n{'OVERALL':-^40}")
    print(f"  Thought correction rate: {total_thought}/{total_items} = {total_thought/total_items*100:.1f}%")
    print(f"  User correction rate:    {total_user}/{total_items} = {total_user/total_items*100:.1f}%")
    print(f"  Asymmetry:               {overall_asymmetry:+.1f} pp")

    return {
        "overall_asymmetry": overall_asymmetry,
        "total_thought_rate": total_thought / total_items,
        "total_user_rate": total_user / total_items,
        "per_domain": {
            d: {
                "thought_rate": s["thought_corrected"] / s["total"],
                "user_rate": s["user_corrected"] / s["total"],
                "asymmetry": (s["user_corrected"] - s["thought_corrected"]) / s["total"] * 100,
            }
            for d, s in domains.items()
        },
    }


def main():
    print(f"Loading model: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    print(f"VRAM: {torch.cuda.memory_allocated()//1024**3}GB / {torch.cuda.get_device_properties(0).total_memory//1024**3}GB")

    print(f"\nRunning pre-test on {len(TEST_ITEMS)} items...")
    results = run_pretest(model, tokenizer, TEST_ITEMS)

    # Save raw results
    with open(os.path.join(OUTPUT_DIR, "pretest_raw.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Analyze
    metrics = analyze_results(results)
    with open(os.path.join(OUTPUT_DIR, "pretest_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nResults saved to {OUTPUT_DIR}/")
    return metrics


if __name__ == "__main__":
    main()
