"""Trace Mistral 7B layer activations to find behavioral circuit."""
import json, torch, numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
SYS = 'You are a precise AI assistant.'

print('Loading Mistral...', flush=True)
model = AutoModelForCausalLM.from_pretrained(
    'mistralai/Mistral-7B-Instruct-v0.3',
    quantization_config=bnb, device_map='cuda')
tokenizer = AutoTokenizer.from_pretrained('mistralai/Mistral-7B-Instruct-v0.3')
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

n = len(model.model.layers)
model.eval()
print(f'Layers: {n}', flush=True)

with open('/root/phase2_evaluated.json') as f:
    errors = [e for e in json.load(f) if e['is_wrong']]

# Trace ALL layers
def trace(msgs):
    text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors='pt').to(model.device)
    acts = {}
    hooks = []
    def make_hook(idx):
        def hook_fn(m, inp, out):
            acts[idx] = out[0, -1, :].detach().cpu()
        return hook_fn
    for li in range(n):
        hooks.append(model.model.layers[li].register_forward_hook(make_hook(li)))
    with torch.no_grad():
        model(**inputs)
    for h in hooks:
        h.remove()
    return acts

print(f'Tracing {len(errors[:8])} errors on {n} layers...', flush=True)
agg = {li: [] for li in range(n)}

for i, err in enumerate(errors[:8]):
    mr = err['model_response']
    ta = trace([
        {'role': 'system', 'content': SYS},
        {'role': 'user', 'content': err['problem']},
        {'role': 'assistant', 'content': mr},
        {'role': 'user', 'content': 'Review your answer. Is it correct?'}])
    ua = trace([
        {'role': 'system', 'content': SYS},
        {'role': 'user', 'content': f'A user says: {mr} Is this correct?'}])
    for li in range(n):
        agg[li].append(torch.norm(ua[li] - ta[li]).item())
    print(f'  [{i+1}/8]', flush=True)

# Analysis
print('\n=== LAYER DIVERGENCE ===', flush=True)
for li in range(n):
    mean_d = np.mean(agg[li])
    print(f'  L{li:02d}: {mean_d:.4f}', flush=True)

# Deep vs shallow ratio
shallow = [li for li in range(n) if li < n * 0.3]
deep = [li for li in range(n) if li > n * 0.7]
sv = np.mean([np.mean(agg[li]) for li in shallow]) if shallow else 0
dv = np.mean([np.mean(agg[li]) for li in deep]) if deep else 0
ratio = dv / sv if sv > 0 else 0
pred = 0.66 * n
delta = abs(ratio - pred) / pred * 100 if pred > 0 else 0

# Top-5 layers
means = [(li, np.mean(agg[li])) for li in range(n)]
means.sort(key=lambda x: x[1], reverse=True)
top5 = means[:5]

print(f'\n=== SUMMARY ===', flush=True)
print(f'Shallow (0-{int(n*0.3)-1}): {sv:.4f}', flush=True)
print(f'Deep ({int(n*0.7)+1}-{n-1}): {dv:.4f}', flush=True)
print(f'Ratio deep/shallow: {ratio:.1f}x (pred {pred:.1f}x, delta {delta:.0f}%)', flush=True)
print(f'Top-5 layers:', flush=True)
for li, d in top5:
    pct = d / max(d for _, d in means) * 100
    print(f'  L{li:02d}: {d:.4f} ({pct:.0f}%)', flush=True)

# Find best contiguous window of 25% layers
best_window = (0, 0)
best_val = 0
window_size = int(n * 0.25)  # 8 layers for 32
for start in range(n - window_size + 1):
    w_val = np.mean([np.mean(agg[li]) for li in range(start, start + window_size)])
    if w_val > best_val:
        best_val = w_val
        best_window = (start, start + window_size - 1)

print(f'Best 25% window: L{best_window[0]}-L{best_window[1]} (avg={best_val:.4f})', flush=True)

# Save results
result = {
    'model': 'Mistral-7B-Instruct-v0.3',
    'n_layers': n,
    'per_layer': {f'L{li}': float(np.mean(agg[li])) for li in range(n)},
    'shallow_mean': float(sv),
    'deep_mean': float(dv),
    'ratio': float(ratio),
    'predicted': pred,
    'delta_pct': delta,
    'top5': [(f'L{li}', float(d)) for li, d in top5],
    'best_25pct_window': [best_window[0], best_window[1]]
}
with open('/root/trace_mistral.json', 'w') as f:
    json.dump(result, f, indent=2)

print('\nSaved to /root/trace_mistral.json', flush=True)
print('DONE', flush=True)
