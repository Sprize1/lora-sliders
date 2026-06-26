"""Trace where self-correction asymmetry lives in Qwen 7B layers."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json, os, numpy as np

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
SYSTEM = "You are a precise AI assistant."

def get_last_acts(model, tokenizer, messages, layers):
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    acts = {}
    hooks = []
    def make_hook(idx):
        def hook_fn(module, inp, out):
            acts[f"L{idx}"] = out[0, -1, :].detach().cpu()
        return hook_fn
    for li in layers:
        hooks.append(model.model.layers[li].register_forward_hook(make_hook(li)))
    with torch.no_grad():
        model(**inputs)
    for h in hooks: h.remove()
    return acts


def main():
    print(f"Loading {MODEL_ID} in 4-bit...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="cuda",
        load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    n_layers = len(model.model.layers)
    print(f"Model has {n_layers} layers. VRAM: {torch.cuda.memory_allocated()//1024**3}GB")

    # Load errors
    with open("research/experiment/results/phase2_evaluated.json", encoding="utf-8") as f:
        errors = [e for e in json.load(f) if e["is_wrong"]]

    # Trace every 2 layers (skip some for speed on 7B)
    step = max(1, n_layers // 16)  # ~16 sample points
    sample_layers = list(range(0, n_layers, step))
    if sample_layers[-1] != n_layers - 1:
        sample_layers.append(n_layers - 1)

    print(f"Tracing {len(errors)} errors at {len(sample_layers)} layers...")

    agg = {li: [] for li in sample_layers}
    for i, err in enumerate(errors):
        # Thought condition
        t_msgs = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": err["problem"]},
            {"role": "assistant", "content": err["model_response"]},
            {"role": "user", "content": "Review your answer. Is it correct?"},
        ]
        # User condition
        u_msgs = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"A user says: {err['model_response']}\nIs this correct?"},
        ]

        t_acts = get_last_acts(model, tokenizer, t_msgs, sample_layers)
        u_acts = get_last_acts(model, tokenizer, u_msgs, sample_layers)

        for li in sample_layers:
            diff = torch.norm(u_acts[f"L{li}"] - t_acts[f"L{li}"]).item()
            agg[li].append(diff)

        if (i+1) % 3 == 0:
            print(f"  [{i+1}/{len(errors)}] done")

    # Results
    print(f"\n{'='*60}")
    print(f"LAYER DIVERGENCE (thought vs user last-token L2 distance)")
    print(f"{'='*60}")
    print(f"{'Layer':<8} {'L2 dist':>10} {'% of max':>10}")
    print(f"{'-'*30}")

    max_l2 = max(np.mean(v) for v in agg.values())
    for li in sorted(agg.keys()):
        mean_l2 = np.mean(agg[li])
        pct = mean_l2 / max_l2 * 100 if max_l2 > 0 else 0
        bar = "█" * int(pct / 5)
        marker = " ← PEAK" if pct > 80 else ""
        print(f"  L{li:<6} {mean_l2:>10.4f} {pct:>9.0f}% {bar}{marker}")

    # Find where divergence concentrates
    # Top 25% layers
    l2_vals = [(li, np.mean(agg[li])) for li in sorted(agg.keys())]
    l2_vals.sort(key=lambda x: x[1], reverse=True)
    top_quarter = l2_vals[:len(l2_vals)//4]
    top_layers = [li for li, _ in top_quarter]

    print(f"\n  Top 25% layers (where behavior lives): L{min(top_layers)}-L{max(top_layers)}")
    print(f"  These are in the {'DEEP' if min(top_layers) > n_layers*0.6 else 'SHALLOW' if max(top_layers) < n_layers*0.3 else 'MIDDLE'} region of the model")

    shallow = [li for li in agg if li < n_layers * 0.3]
    deep = [li for li in agg if li > n_layers * 0.6]
    shallow_l2 = np.mean([np.mean(agg[li]) for li in shallow])
    deep_l2 = np.mean([np.mean(agg[li]) for li in deep])

    print(f"\n  Shallow layers (0-{int(n_layers*0.3)}): avg L2 = {shallow_l2:.4f}")
    print(f"  Deep layers ({int(n_layers*0.6)}-{n_layers-1}): avg L2 = {deep_l2:.4f}")
    print(f"  Deep/Shallow ratio: {deep_l2/shallow_l2:.1f}x")

    result = {
        "model": MODEL_ID,
        "n_layers": n_layers,
        "deep_shallow_ratio": float(deep_l2/shallow_l2),
        "per_layer": {str(li): float(np.mean(agg[li])) for li in sorted(agg.keys())},
    }
    with open("trace_7b_results.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to trace_7b_results.json")


if __name__ == "__main__":
    main()
