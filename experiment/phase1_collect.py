"""Phase 1: Generate model's OWN errors on hard problems."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import os

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
OUTPUT_DIR = "experiment/results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

HARD_PROBLEMS = [
    # === MATHS (complex, multi-step) ===
    "A spherical balloon is being inflated at a rate of 100 cm³/s. How fast is the radius increasing when the radius is 5 cm? Show your work step by step.",
    "Find all real solutions: |2x - 3| + |x + 1| = 7. Show all cases.",
    "A fair coin is flipped 4 times. What is the probability of getting exactly 2 heads AND not having 2 consecutive heads? Show reasoning.",
    "Solve for x: log₂(x + 3) + log₂(x - 1) = 4. Check for extraneous solutions.",
    "Find the sum: 1/(1×2) + 1/(2×3) + 1/(3×4) + ... + 1/(99×100). Show the pattern.",
    "A bat and ball cost $1.10 total. The bat costs $1.00 more than the ball. How much does the ball cost? Think carefully.",
    "If 3^x = 7, find x. Express as a logarithm and compute numerically to 2 decimal places.",
    "A rectangle's perimeter is 50 cm. Its area is 150 cm². Find its dimensions. Show equations.",
    "Prove or disprove: √2 + √3 is irrational. Show your reasoning.",
    "How many distinct ways can you arrange the letters in MISSISSIPPI? Show the formula and calculation.",

    # === LOGIC (requires step-by-step deduction) ===
    "Alice, Bob, and Charlie each have a different profession: doctor, engineer, or teacher. The doctor is not Alice. The engineer is older than Bob. Charlie is not the teacher. Who has which profession? Show your deduction.",
    "In a room of 30 people, what is the probability that at least two share the same birthday? (Assume 365 days, equal probability). Show the calculation.",
    "If the statement 'If it is raining, then the game is cancelled' is FALSE, what must be true about the weather and the game? Explain using truth tables.",
    "Three boxes: one has only apples, one only oranges, one both. All labels are wrong. You can draw one fruit from one box. How do you determine all contents? Explain your strategy.",
    "Prove by contradiction: there is no largest prime number. Write out the full proof.",
    "Is this argument sound? 'If we have free will, determinism is false. Determinism is true. Therefore we do not have free will.' Analyze validity vs soundness.",
    "The unexpected hanging paradox: A judge tells a prisoner he will be hanged on a weekday next week, but it will be a surprise. The prisoner reasons he cannot be hanged. Explain the logical flaw (if any).",
    "Evaluate: 'All valid arguments have true conclusions. This argument has a true conclusion. Therefore this argument is valid.' Identify the fallacy.",
]

SYSTEM_PROMPT = "You are a precise AI assistant. Think step by step and show your full reasoning."


def collect_errors(model, tokenizer, problems, max_errors=30):
    """Run problems and collect the model's natural wrong answers."""
    errors = []

    for problem in problems:
        if len(errors) >= max_errors:
            break

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": problem},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.6,
                top_p=0.9,
            )
        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )

        # We'll store ALL responses (right or wrong) for later verification
        # During phase 2 we'll check which are actually wrong
        errors.append({
            "problem": problem,
            "model_response": response,
            "domain": "maths" if any(k in problem.lower() for k in ["solve", "find", "prove", "sum", "equation", "probability", "how many", "coin", "number"]) else "logic",
        })

    return errors


def main():
    print(f"Loading model: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )

    print(f"Collecting model responses on {len(HARD_PROBLEMS)} hard problems...")
    errors = collect_errors(model, tokenizer, HARD_PROBLEMS)

    with open(os.path.join(OUTPUT_DIR, "phase1_model_errors.json"), "w", encoding="utf-8") as f:
        json.dump(errors, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(errors)} responses to {OUTPUT_DIR}/phase1_model_errors.json")

    # Quick stats
    lengths = [len(e["model_response"]) for e in errors]
    print(f"Avg response length: {sum(lengths)/len(lengths):.0f} chars")
    print(f"Min: {min(lengths)}, Max: {max(lengths)}")


if __name__ == "__main__":
    main()
