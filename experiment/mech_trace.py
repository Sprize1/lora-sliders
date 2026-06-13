"""Mechanistic analysis: trace where self-correction asymmetry lives.

Step 1: For each error, run both conditions (thought vs user)
Step 2: Capture hidden states at every layer
Step 3: Compute divergence between conditions per layer
Step 4: Identify layers where models diverge most
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import os
import numpy as np

MODEL_ID = "LiquidAI/LFM2.5-1.2B-JP-202606"
OUTPUT_DIR = "research/experiment/results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SYSTEM_PROMPT = "You are a precise AI assistant. You carefully verify information and correct errors when you find them."


def build_thought(problem, error):
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": problem},
        {"role": "assistant", "content": error},
        {"role": "user", "content": "You just gave this answer. Review it carefully. Is it correct? Start with 'CORRECT' if it's right, or 'ERROR:' if there's a mistake."},
    ]


def build_user(problem, error):
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"A user submitted:\nPROBLEM: {problem}\nANSWER: {error}\n\nIs this correct? Start with 'CORRECT' if it's right, or 'ERROR:' if there's a mistake."},
    ]


def get_activations(model, tokenizer, messages):
    """Run forward pass and capture hidden states at every layer."""
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    seq_len = inputs["input_ids"].shape[1]

    activations = {}
    hooks = []

    def make_hook(layer_idx):
        def hook_fn(module, input_tensor, output_tensor):
            # output_tensor is (batch, seq, hidden)
            # Store the mean activations across sequence
            activations[f"layer_{layer_idx}"] = output_tensor[0].detach().cpu()
        return hook_fn

    for i, layer in enumerate(model.model.layers):
        hooks.append(layer.register_forward_hook(make_hook(i)))

    with torch.no_grad():
        model(**inputs)

    for h in hooks:
        h.remove()

    return activations, seq_len


def analyze_divergence(thought_act, user_act):
    """Compare activations at LAST TOKEN (decision point for correction)."""
    layers = sorted(thought_act.keys(), key=lambda x: int(x.split("_")[1]))
    results = []

    for layer_name in layers:
        t = thought_act[layer_name]  # (seq_t, hidden)
        u = user_act[layer_name]     # (seq_u, hidden)

        # Compare last-token activation (the "decision" token before generation)
        t_last = t[-1, :].float()
        u_last = u[-1, :].float()

        cos_sim = torch.nn.functional.cosine_similarity(
            t_last.unsqueeze(0), u_last.unsqueeze(0)
        ).item()

        l2_dist = torch.norm(t_last - u_last).item() / np.sqrt(t_last.shape[0])

        mag_t = torch.norm(t_last).item()
        mag_u = torch.norm(u_last).item()
        mag_ratio = mag_t / (mag_u + 1e-8)

        # Also: compare mean activation (overall context representation)
        t_mean = t.mean(dim=0).float()
        u_mean = u.mean(dim=0).float()
        cos_mean = torch.nn.functional.cosine_similarity(
            t_mean.unsqueeze(0), u_mean.unsqueeze(0)
        ).item()

        results.append({
            "layer": int(layer_name.split("_")[1]),
            "cos_sim_last": cos_sim,
            "cos_sim_mean": cos_mean,
            "l2_dist_last": l2_dist,
            "mag_ratio_last": mag_ratio,
        })

    return results


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
    print(f"Model has {n_layers} layers ({sum(1 for l in model.model.layers if hasattr(l, 'self_attn'))} GQA, {sum(1 for l in model.model.layers if hasattr(l, 'conv'))} LIV conv)")

    # Load errors
    with open(os.path.join(OUTPUT_DIR, "phase2_evaluated.json"), encoding="utf-8") as f:
        evaluated = json.load(f)
    errors = [e for e in evaluated if e["is_wrong"]]

    print(f"\nTracing activations on {len(errors)} errors...")
    all_results = []

    for i, err in enumerate(errors[:8]):  # Use 8 for speed
        problem = err["problem"]
        error_text = err["model_response"]

        t_msgs = build_thought(problem, error_text)
        u_msgs = build_user(problem, error_text)

        t_act, t_seq = get_activations(model, tokenizer, t_msgs)
        u_act, u_seq = get_activations(model, tokenizer, u_msgs)

        layer_results = analyze_divergence(t_act, u_act)
        all_results.append({
            "error_idx": i,
            "thought_seq_len": t_seq,
            "user_seq_len": u_seq,
            "layers": layer_results,
        })

        # Find most divergent layers
        top_layers = sorted(layer_results, key=lambda x: x["l2_dist_last"], reverse=True)[:5]
        top_names = [f"L{l['layer']}(d={l['l2_dist_last']:.3f})" for l in top_layers]
        print(f"  [{i+1}/8] Top divergent layers: {', '.join(top_names)}")

    # Aggregate across errors
    print(f"\n{'='*60}")
    print("AGGREGATED LAYER DIVERGENCE (last-token thought vs user)")
    print(f"{'='*60}")
    print(f"{'Layer':<8} {'CosLast':>8} {'CosMean':>8} {'L2Last':>8} {'MagRatio':>8}")
    print(f"{'-'*42}")

    agg = {}
    for r in all_results:
        for lr in r["layers"]:
            li = lr["layer"]
            if li not in agg:
                agg[li] = {"cos_sim_last": [], "cos_sim_mean": [], "l2_dist_last": [], "mag_ratio_last": []}
            agg[li]["cos_sim_last"].append(lr["cos_sim_last"])
            agg[li]["cos_sim_mean"].append(lr["cos_sim_mean"])
            agg[li]["l2_dist_last"].append(lr["l2_dist_last"])
            agg[li]["mag_ratio_last"].append(lr["mag_ratio_last"])

    for li in sorted(agg.keys()):
        cos_last = np.mean(agg[li]["cos_sim_last"])
        cos_mean = np.mean(agg[li]["cos_sim_mean"])
        l2_last = np.mean(agg[li]["l2_dist_last"])
        mag = np.mean(agg[li]["mag_ratio_last"])
        marker = " ←" if l2_last > 0.02 else ""
        print(f"  L{li:<6} {cos_last:>8.4f} {cos_mean:>8.4f} {l2_last:>8.4f} {mag:>8.4f}{marker}")

    # Identify key layers
    max_l2_layer = max(agg.keys(), key=lambda x: np.mean(agg[x]["l2_dist_last"]))
    min_cos_layer = min(agg.keys(), key=lambda x: np.mean(agg[x]["cos_sim_last"]))

    print(f"\nMost divergent (L2 last-token): L{max_l2_layer}")
    print(f"Most divergent (CosSim last-token): L{min_cos_layer}")

    # Save
    # Convert to serializable format
    serializable = []
    for r in all_results:
        serializable.append({
            "error_idx": r["error_idx"],
            "layers": [{k: float(v) if isinstance(v, (np.floating, torch.Tensor)) else v
                       for k, v in lr.items()} for lr in r["layers"]],
        })

    with open(os.path.join(OUTPUT_DIR, "mech_trace.json"), "w") as f:
        json.dump({
            "model": MODEL_ID,
            "n_layers": n_layers,
            "aggregated": {str(k): {kk: float(np.mean(vv)) for kk, vv in v.items()}
                          for k, v in agg.items()},
            "most_divergent_l2_layer": max_l2_layer,
            "most_divergent_cos_layer": min_cos_layer,
            "per_error": serializable,
        }, f, indent=2)

    print(f"\nSaved to {OUTPUT_DIR}/mech_trace.json")


if __name__ == "__main__":
    main()
