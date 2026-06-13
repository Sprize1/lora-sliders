"""Check: is user-thought activation difference LARGER on asymmetric errors?"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json, os, requests, numpy as np

MODEL_ID = "LiquidAI/LFM2.5-1.2B-JP-202606"
OUTPUT_DIR = "research/experiment/results"

API_BASE = "https://api.deepseek.com"
with open(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), ".env")) as f:
    for line in f:
        if "ANTHROPIC_AUTH_TOKEN" in line:
            API_KEY = line.strip().split("=", 1)[1]
            break
HEADERS = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

SYSTEM_PROMPT = "You are a precise AI assistant. You carefully verify information and correct errors when you find them."


def api_judge(problem, wrong, correction):
    resp = requests.post(f"{API_BASE}/v1/chat/completions", headers=HEADERS,
                         json={"model": "deepseek-chat",
                               "messages": [{"role": "system", "content": "Expert. JSON only."},
                                            {"role": "user", "content": f"""Does this correct the error?
PROBLEM: {problem}
WRONG: {wrong[:300]}
RESPONSE: {correction[:400]}
JSON: {{"verdict": "good_correction"/"missed_error"}}"""}],
                               "max_tokens": 100, "temperature": 0.0}, timeout=120)
    try:
        return json.loads(resp.json()["choices"][0]["message"]["content"].strip().removeprefix("```json").removesuffix("```").strip()).get("verdict") == "good_correction"
    except: return False


def build_thought(problem, error):
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": problem},
        {"role": "assistant", "content": error},
        {"role": "user", "content": "You just gave this answer. Review it. Is it correct? Start with 'CORRECT' if right, or 'ERROR:' if wrong."},
    ]

def build_user(problem, error):
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"A user submitted:\nPROBLEM: {problem}\nANSWER: {error}\n\nIs this correct? Start with 'CORRECT' if right, or 'ERROR:' if wrong."},
    ]


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


def generate_normal(model, tokenizer, messages):
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=200, do_sample=False)
    return tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map="cuda")
model.eval()
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

with open(os.path.join(OUTPUT_DIR, "phase2_evaluated.json"), encoding="utf-8") as f:
    errors = [e for e in json.load(f) if e["is_wrong"]]

steer_layers = [11, 12, 13, 14, 15]
asymmetric = []
symmetric = []

print(f"Classifying {len(errors)} errors as asymmetric or symmetric...")
for i, err in enumerate(errors):
    t_resp = generate_normal(model, tokenizer, build_thought(err["problem"], err["model_response"]))
    u_resp = generate_normal(model, tokenizer, build_user(err["problem"], err["model_response"]))
    t_ok = api_judge(err["problem"], err["model_response"], t_resp)
    u_ok = api_judge(err["problem"], err["model_response"], u_resp)

    t_acts = get_last_acts(model, tokenizer, build_thought(err["problem"], err["model_response"]), steer_layers)
    u_acts = get_last_acts(model, tokenizer, build_user(err["problem"], err["model_response"]), steer_layers)

    diffs = {}
    for li in steer_layers:
        diffs[f"L{li}"] = torch.norm(u_acts[f"L{li}"] - t_acts[f"L{li}"]).item()

    if not t_ok and u_ok:
        asymmetric.append({"idx": i, "diffs": diffs, "problem": err["problem"][:80]})
        print(f"  [{i}] ASYM: thought ✗ user ✓")
    elif t_ok and u_ok:
        symmetric.append({"idx": i, "diffs": diffs})
        print(f"  [{i}] SYM:  thought ✓ user ✓")
    elif t_ok and not u_ok:
        symmetric.append({"idx": i, "diffs": diffs})
        print(f"  [{i}] SYM*: thought ✓ user ✗ (inverse)")
    else:
        symmetric.append({"idx": i, "diffs": diffs})
        print(f"  [{i}] BOTH ✗")

print(f"\n{'='*60}")
print(f"Asymmetric: {len(asymmetric)} | Symmetric: {len(symmetric)}")
print(f"{'='*60}")

for li in steer_layers:
    asym_vals = [a["diffs"][f"L{li}"] for a in asymmetric]
    sym_vals = [s["diffs"][f"L{li}"] for s in symmetric]
    asym_mean = np.mean(asym_vals) if asym_vals else 0
    sym_mean = np.mean(sym_vals) if sym_vals else 0
    ratio = asym_mean / (sym_mean + 1e-8)
    bar = "█" * int(ratio) if ratio > 1 else "▁" * int(1/ratio) if ratio < 1 else "─"
    print(f"  L{li}: asym={asym_mean:.4f}  sym={sym_mean:.4f}  ratio={ratio:.1f}x  {bar}")

if asymmetric:
    print(f"\nAsymmetric errors (good candidates for steering):")
    for a in asymmetric:
        print(f"  [{a['idx']}] L15 diff={a['diffs']['L15']:.4f} | {a['problem']}")
