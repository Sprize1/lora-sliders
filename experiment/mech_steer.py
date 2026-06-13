"""Activation steering: can we fix self-correction asymmetry by patching activations?

Step 1: For each error where model MISSED correction in thought condition:
  - Capture user-condition activation at L15
  - Patch it into thought-condition forward pass at L15
  - Generate with patched activations
  - Check if the model now corrects the error

If patching fixes the asymmetry → the "self-correction circuit" is localized.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import os
import requests

MODEL_ID = "LiquidAI/LFM2.5-1.2B-JP-202606"
OUTPUT_DIR = "research/experiment/results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

API_BASE = "https://api.deepseek.com"
API_KEY = None
with open(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), ".env")) as f:
    for line in f:
        if "ANTHROPIC_AUTH_TOKEN" in line:
            API_KEY = line.strip().split("=", 1)[1]
            break
HEADERS = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

SYSTEM_PROMPT = "You are a precise AI assistant. You carefully verify information and correct errors when you find them."


def api_judge(problem, wrong, correction):
    prompt = f"""Does this response identify and correct the error?

PROBLEM: {problem}
WRONG ANSWER: {wrong[:300]}
RESPONSE: {correction[:400]}

JSON: {{"identified_error": true/false, "verdict": "good_correction"/"missed_error"/"false_positive"}}"""
    resp = requests.post(f"{API_BASE}/v1/chat/completions", headers=HEADERS,
                         json={"model": "deepseek-chat",
                               "messages": [{"role": "system", "content": "Expert. Output ONLY valid JSON."},
                                            {"role": "user", "content": prompt}],
                               "max_tokens": 150, "temperature": 0.0}, timeout=120)
    try:
        clean = resp.json()["choices"][0]["message"]["content"].strip()
        clean = clean.removeprefix("```json").removesuffix("```").strip()
        return json.loads(clean)
    except:
        return {"verdict": "parse_error"}


def build_thought(problem, error):
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": problem},
        {"role": "assistant", "content": error},
        {"role": "user", "content": "You just gave this answer. Review it. Is it correct? Start with 'CORRECT' if it's right, or 'ERROR:' if there's a mistake."},
    ]


def build_user(problem, error):
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"A user submitted:\nPROBLEM: {problem}\nANSWER: {error}\n\nIs this correct? Start with 'CORRECT' if it's right, or 'ERROR:' if there's a mistake."},
    ]


def generate_with_patch(model, tokenizer, messages, patch_layer, patch_activation):
    """Generate with a patched activation at a specific layer."""
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    # We'll use a hook to replace activation
    replaced_hidden = None

    def patch_hook(module, input_tensor, output_tensor):
        nonlocal replaced_hidden
        # Replace the last token's hidden state with patch
        modified = output_tensor.clone()
        modified[0, -1, :] = patch_activation.to(modified.device).to(modified.dtype)
        replaced_hidden = modified
        return modified

    hook = model.model.layers[patch_layer].register_forward_hook(patch_hook)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=200, do_sample=False)

    hook.remove()
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return response


def generate_normal(model, tokenizer, messages):
    """Normal generation without patching."""
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=200, do_sample=False)
    return tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def get_last_activation(model, tokenizer, messages, layer_idx):
    """Get the activation at the last token of a specific layer."""
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    activation = None

    def hook_fn(module, input_tensor, output_tensor):
        nonlocal activation
        activation = output_tensor[0, -1, :].detach().cpu()

    hook = model.model.layers[layer_idx].register_forward_hook(hook_fn)
    with torch.no_grad():
        model(**inputs)
    hook.remove()
    return activation


def main():
    print(f"Loading {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    model.eval()
    n_layers = len(model.model.layers)

    # Load errors
    with open(os.path.join(OUTPUT_DIR, "phase2_evaluated.json"), encoding="utf-8") as f:
        evaluated = json.load(f)
    errors = [e for e in evaluated if e["is_wrong"]]

    # Patching experiment
    patch_layers = [15, 14, 13, 12, 11]  # Most divergent layers
    results = []

    print(f"\nActivation steering on all {len(errors)} errors...")
    print(f"Patching layers: {patch_layers}")
    print("(Only errors with asymmetry: thought MISS + user HIT will be patched)")

    for i, err in enumerate(errors):
        problem = err["problem"]
        error_text = err["model_response"]

        t_msgs = build_thought(problem, error_text)
        u_msgs = build_user(problem, error_text)

        # Baseline: normal generation
        thought_normal = generate_normal(model, tokenizer, t_msgs)
        user_normal = generate_normal(model, tokenizer, u_msgs)

        thought_ok = api_judge(problem, error_text, thought_normal).get("verdict") == "good_correction"
        user_ok = api_judge(problem, error_text, user_normal).get("verdict") == "good_correction"

        print(f"\n[{i+1}/6] Error: {problem[:80]}...")
        print(f"  Normal thought: {'✓' if thought_ok else '✗'} | user: {'✓' if user_ok else '✗'}")

        # Only try patching if thought missed but user caught it (asymmetry case)
        if not thought_ok and user_ok:
            # Get user activation at each patch layer
            for pl in patch_layers:
                user_act = get_last_activation(model, tokenizer, u_msgs, pl)
                patched_response = generate_with_patch(model, tokenizer, t_msgs, pl, user_act)
                patched_ok = api_judge(problem, error_text, patched_response).get("verdict") == "good_correction"
                marker = "★★★ FIXED" if patched_ok else "---"
                print(f"  L{pl} patch: {'✓' if patched_ok else '✗'} {marker}")
                results.append({
                    "error_idx": i, "condition": f"patch_L{pl}",
                    "thought_normal": thought_ok, "user_normal": user_ok,
                    "patched_ok": patched_ok, "layer": pl,
                })

    # Summary
    by_layer = {}
    for r in results:
        pl = r["layer"]
        if pl not in by_layer:
            by_layer[pl] = {"total": 0, "fixed": 0}
        by_layer[pl]["total"] += 1
        if r["patched_ok"]:
            by_layer[pl]["fixed"] += 1

    print(f"\n{'='*60}")
    print("PATCHING RESULTS: which layer fixes the asymmetry?")
    print(f"{'='*60}")
    for pl in sorted(by_layer.keys()):
        f = by_layer[pl]["fixed"]
        t = by_layer[pl]["total"]
        print(f"  L{pl}: {f}/{t} fixed ({f/t*100:.0f}%)")

    with open(os.path.join(OUTPUT_DIR, "mech_steer.json"), "w") as f:
        json.dump({"results": results, "by_layer": {str(k): v for k, v in by_layer.items()}}, f, indent=2)

    print(f"\nSaved to {OUTPUT_DIR}/mech_steer.json")


if __name__ == "__main__":
    main()
