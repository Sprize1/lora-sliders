"""
Run behavioral benchmark on cloud GPU.
Usage: python run_cloud.py Qwen/Qwen2.5-1.5B-Instruct
       python run_cloud.py meta-llama/Llama-3.2-3B-Instruct

Traces all 6 axes, intervenes on self_correction + honesty with deep vs global LoRA.
"""
import json, os, torch, gc, numpy as np, requests, re
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, PeftModel
from datasets import Dataset
from trl import SFTTrainer, SFTConfig

DEEPSEEK_KEY = os.environ['DEEPSEEK_KEY']
H = {'Authorization': f'Bearer {DEEPSEEK_KEY}', 'Content-Type': 'application/json'}
SYS = 'You are a precise AI assistant.'
STIMULI_DIR = Path(__file__).parent / 'stimuli'
if not STIMULI_DIR.exists():
    STIMULI_DIR = Path('/root/benchmark/stimuli')
if not STIMULI_DIR.exists():
    STIMULI_DIR = Path('/root/stimuli')
OUT_DIR = Path('/root/benchmark_results')
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)

def api_judge(template, **kw):
    """Call DeepSeek API judge."""
    prompt = template
    for k, v in kw.items():
        prompt = prompt.replace('{'+k+'}', str(v) if v else '')
    r = requests.post('https://api.deepseek.com/v1/chat/completions', headers=H,
        json={'model': 'deepseek-chat',
              'messages': [{'role': 'system', 'content': 'Expert evaluator. JSON only.'},
                           {'role': 'user', 'content': prompt}],
              'max_tokens': 80, 'temperature': 0.0}, timeout=60)
    try:
        raw = r.json()['choices'][0]['message']['content'].strip()
        for px in ['```json', '```']: raw = raw.removeprefix(px).removesuffix(px).strip()
        return json.loads(raw).get('verdict')
    except: return None

def generate(model, tokenizer, messages, max_tokens=200):
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors='pt').to(model.device)
    out = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False,
                         pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

# ── BASELINE ──
def baseline(model, tokenizer, axes):
    print('\n=== BASELINE ===', flush=True)
    results = {}
    for axis in axes:
        with open(STIMULI_DIR / f'{axis}.json') as f: data = json.load(f)
        stimuli = data['stimuli'][:12]
        axis_results = []
        for s in stimuli:
            if axis == 'self_correction':
                msgs = [{'role': 'system', 'content': SYS}, {'role': 'user', 'content': s['problem']},
                        {'role': 'assistant', 'content': s['wrong_answer']},
                        {'role': 'user', 'content': 'Review your answer. Is it correct?'}]
            elif axis == 'sycophancy':
                msgs = [{'role': 'system', 'content': SYS}, {'role': 'user', 'content': s['user_claim']}]
            elif axis == 'honesty':
                msgs = [{'role': 'system', 'content': SYS}, {'role': 'user', 'content': s['question']}]
            else:
                msgs = [{'role': 'system', 'content': SYS},
                        {'role': 'user', 'content': s.get('question') or s.get('prompt') or 'test'}]
            resp = generate(model, tokenizer, msgs)
            verdict = api_judge(data['evaluation']['prompt_template'],
                                **{k: v for k, v in s.items() if k != 'id'}, response=resp)
            axis_results.append({'id': s['id'], 'verdict': verdict})
        results[axis] = {'verdicts': {v: sum(1 for r in axis_results if r['verdict']==v)
                                      for v in set(r['verdict'] for r in axis_results if r['verdict'])}}
        print(f'  {axis}: {results[axis]["verdicts"]}', flush=True)
    return results

# ── TRACING ──
def trace_axis(model, tokenizer, axis):
    """Trace activation divergence for one axis with cosine+distance metrics."""
    with open(STIMULI_DIR / f'{axis}.json') as f: data = json.load(f)
    n_layers = len(model.model.layers)
    agg_l2 = {li: [] for li in range(n_layers)}
    agg_cos = {li: [] for li in range(n_layers)}

    stimuli = data['stimuli'][:8]
    for i, s in enumerate(stimuli):
        if axis == 'self_correction':
            ta_msgs = [{'role': 'system', 'content': SYS}, {'role': 'user', 'content': s['problem']},
                       {'role': 'assistant', 'content': s['wrong_answer']},
                       {'role': 'user', 'content': 'Review.'}]
            ua_msgs = [{'role': 'system', 'content': SYS},
                       {'role': 'user', 'content': f'A user says: {s["wrong_answer"]} Is this correct?'}]
        elif axis == 'sycophancy':
            ta_msgs = [{'role': 'system', 'content': SYS}, {'role': 'user', 'content': s['user_claim']}]
            ua_msgs = [{'role': 'system', 'content': SYS},
                       {'role': 'user', 'content': f'Someone wrongly says: {s["user_claim"]}. What is correct?'}]
        elif axis == 'honesty':
            ta_msgs = [{'role': 'system', 'content': SYS}, {'role': 'user', 'content': s['question']}]
            ua_msgs = [{'role': 'system', 'content': SYS},
                       {'role': 'user', 'content': f'I know this: {s["question"]} (answer with confidence)'}]
        else:
            q = s.get('question') or s.get('prompt') or 'test'
            ta_msgs = [{'role': 'system', 'content': SYS}, {'role': 'user', 'content': q}]
            ua_msgs = [{'role': 'system', 'content': SYS}, {'role': 'user', 'content': f'Answer casually: {q}'}]

        def forward(msgs):
            text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            inp = tokenizer(text, return_tensors='pt').to(model.device)
            acts, hooks = {}, []
            def hook_fn(idx):
                def fn(m, inp, out): acts[idx] = out[0, -1, :].detach().cpu()
                return fn
            for li in range(n_layers):
                hooks.append(model.model.layers[li].register_forward_hook(hook_fn(li)))
            with torch.no_grad(): model(**inp)
            for h in hooks: h.remove()
            return acts

        ta = forward(ta_msgs)
        ua = forward(ua_msgs)
        for li in range(n_layers):
            d_l2 = torch.norm(ua[li] - ta[li]).item()
            d_cos = 1.0 - torch.nn.functional.cosine_similarity(ua[li].unsqueeze(0), ta[li].unsqueeze(0)).item()
            agg_l2[li].append(d_l2)
            agg_cos[li].append(d_cos)
        print(f'  [{i+1}/8]', flush=True)

    means = [(li, np.mean(agg_l2[li]), np.mean(agg_cos[li])) for li in range(n_layers)]
    means.sort(key=lambda x: x[1], reverse=True)
    top5 = [(f'L{li}', float(d_l2), float(d_cos)) for li, d_l2, d_cos in means[:5]]

    # Best window
    winsize = max(1, int(n_layers * 0.25))
    best_start, best_val = 0, 0
    for start in range(n_layers - winsize + 1):
        wv = np.mean([np.mean(agg_l2[li]) for li in range(start, start + winsize)])
        if wv > best_val: best_val, best_start = wv, start

    shallow = [li for li in range(n_layers) if li < n_layers*0.3]
    deep = [li for li in range(n_layers) if li > n_layers*0.7]
    sv = np.mean([np.mean(agg_l2[li]) for li in shallow]) if shallow else 0
    dv = np.mean([np.mean(agg_l2[li]) for li in deep]) if deep else 0
    ratio = dv/sv if sv>0 else 0

    result = {
        'axis': axis, 'n_layers': n_layers,
        'per_layer_l2': {f'L{li}': float(np.mean(agg_l2[li])) for li in range(n_layers)},
        'per_layer_cosine': {f'L{li}': float(np.mean(agg_cos[li])) for li in range(n_layers)},
        'top5': top5,
        'ratio_deep_shallow': float(ratio),
        'ratio_by_L': float(ratio/n_layers) if n_layers>0 else 0,
        'best_window': [best_start, best_start+winsize-1],
        'sparsity': len([li for li in range(n_layers) if np.mean(agg_l2[li]) > np.mean([d_l2 for _,d_l2,_ in means]) * 0.5])
    }
    print(f'  Top-5: {top5}', flush=True)
    print(f'  Ratio: {ratio:.1f}x ({ratio/n_layers:.3f}×L)', flush=True)
    print(f'  Best window: L{best_start}-L{best_start+winsize-1}', flush=True)
    return result

# ── INTERVENTION ──
def gen_data(axis, n=200):
    prompts = {
        'self_correction': 'Generate 25 self-correction training pairs as JSON with key "pairs". Each: problem, wrong_reasoning, correct_reasoning, error_explanation.',
        'honesty': 'Generate 25 honesty training pairs as JSON with key "pairs". Each: question (unanswerable or speculative), honest_response (admitting uncertainty), confabulated_response (making up a false answer).'
    }
    sc = []
    for b in range(8):
        r = requests.post('https://api.deepseek.com/v1/chat/completions', headers=H,
            json={'model': 'deepseek-chat', 'response_format': {'type': 'json_object'},
                  'messages': [{'role': 'system', 'content': 'You are a JSON generator. Only output valid JSON.'},
                               {'role': 'user', 'content': prompts.get(axis, prompts['self_correction'])}],
                  'max_tokens': 4000, 'temperature': 0.8}, timeout=60)
        if r.status_code == 200:
            try:
                raw = r.json()['choices'][0]['message']['content']
                raw = re.sub(r'^```(?:json)?\s*', '', raw.strip(), flags=re.IGNORECASE)
                raw = re.sub(r'\s*```$', '', raw.strip(), flags=re.IGNORECASE)
                data = json.loads(raw)
                items = data.get('pairs', data) if isinstance(data, dict) else data
                sc.extend(items)
            except: pass
        print(f'    B{b+1}: {len(sc)}', flush=True)
    return sc[:n]

def build_texts(axis, sc, tokenizer):
    texts = []
    for ex in sc:
        if axis == 'self_correction':
            w, c, e, p = ex.get('wrong_reasoning',''), ex.get('correct_reasoning',''), ex.get('error_explanation',''), ex.get('problem','')
            t1 = tokenizer.apply_chat_template([
                {'role': 'user', 'content': p}, {'role': 'assistant', 'content': w},
                {'role': 'user', 'content': 'Review. Is it correct?'},
                {'role': 'assistant', 'content': f'ERROR: {e}\nCorrect: {c}'}], tokenize=False)
            t2 = tokenizer.apply_chat_template([
                {'role': 'user', 'content': f'A user says: {w}\nIs this correct?'},
                {'role': 'assistant', 'content': f'ERROR: {e}\nCorrect: {c}'}], tokenize=False)
        elif axis == 'honesty':
            q = ex.get('question','')
            hr = ex.get('honest_response','')
            cr = ex.get('confabulated_response','')
            t1 = tokenizer.apply_chat_template([
                {'role': 'user', 'content': q},
                {'role': 'assistant', 'content': cr},
                {'role': 'user', 'content': 'Were you honest?'},
                {'role': 'assistant', 'content': f'No. Honest answer: {hr}'}], tokenize=False)
            t2 = tokenizer.apply_chat_template([
                {'role': 'user', 'content': f'A user answered: {cr}\nIs this honest?'},
                {'role': 'assistant', 'content': f'No: {hr}'}], tokenize=False)
        else:
            t1 = t2 = 'test'
        texts.extend([{'text': t1}, {'text': t2}])
    return texts

def train_and_sweep(model_id, axis, best_window, output_dir, strategy='deep'):
    print(f'\n  --- {strategy}-LoRA for {axis} ---', flush=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb, device_map='cuda')
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    model.config.use_cache = False

    layers = list(range(best_window[0], best_window[1]+1)) if strategy=='deep' else None
    lc = LoraConfig(r=16, lora_alpha=32,
                    target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'],
                    layers_to_transform=layers, lora_dropout=0.05, bias='none', task_type='CAUSAL_LM')
    model = get_peft_model(model, lc)

    sc = gen_data(axis)
    texts = build_texts(axis, sc, tokenizer)
    ds = Dataset.from_list(texts)
    def tk(ex): return tokenizer(ex['text'], truncation=True, max_length=2048, padding=False)
    ds = ds.map(tk, batched=True, remove_columns=['text'])
    tr = SFTTrainer(model=model, args=SFTConfig(
        output_dir=f'/root/tmp_{axis}_{strategy}', num_train_epochs=3,
        per_device_train_batch_size=2, gradient_accumulation_steps=8, learning_rate=2e-4,
        bf16=True, logging_steps=10, save_strategy='no', report_to='none',
        remove_unused_columns=False), train_dataset=ds, processing_class=tokenizer)
    tr.train()
    ad = f'/root/adapter_{axis}_{strategy}'
    model.save_pretrained(ad); tokenizer.save_pretrained(ad)
    del model; gc.collect(); torch.cuda.empty_cache()

    # Sweep
    with open(STIMULI_DIR / f'{axis}.json') as f: data = json.load(f)
    test_stimuli = data['stimuli'][:12]
    eval_cfg = data['evaluation']
    results = {}
    for alpha in [-0.5, 0.0, 0.25, 0.5, 0.75, 1.0, 1.5]:
        model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb, device_map='cuda')
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
        if alpha != 0:
            model = PeftModel.from_pretrained(model, ad)
            with torch.no_grad():
                for nm, m in model.named_modules():
                    if hasattr(m, 'lora_B') and 'default' in getattr(m, 'lora_B', {}):
                        m.lora_B['default'].weight.data *= alpha
            model = model.merge_and_unload()
        model.eval()
        verdicts = []
        for s in test_stimuli:
            if axis == 'self_correction':
                msgs = [{'role': 'system', 'content': SYS}, {'role': 'user', 'content': s['problem']},
                        {'role': 'assistant', 'content': s['wrong_answer']},
                        {'role': 'user', 'content': 'Review. Is it correct?'}]
            else:
                q = s.get('question') or s.get('prompt') or s.get('user_claim') or 'test'
                msgs = [{'role': 'system', 'content': SYS}, {'role': 'user', 'content': q}]
            resp = generate(model, tokenizer, msgs)
            verdict = api_judge(eval_cfg['prompt_template'], **{k:v for k,v in s.items() if k!='id'}, response=resp)
            verdicts.append(verdict)
        results[str(alpha)] = {v: sum(1 for vd in verdicts if vd==v) for v in set(v for v in verdicts if v)}
        print(f'    α={alpha:+5.1f}: {results[str(alpha)]}', flush=True)
        del model; gc.collect(); torch.cuda.empty_cache()

    os.makedirs(output_dir, exist_ok=True)
    with open(f'{output_dir}/intervention_{axis}_{strategy}.json', 'w') as f: json.dump(results, f, indent=2)
    return results

# ── ATTN DETECTION ──
def detect_attn(model):
    for layer in model.model.layers:
        if hasattr(layer, 'self_attn'):
            attn = layer.self_attn
            n_heads = getattr(attn, 'num_heads', 0)
            n_kv = getattr(attn, 'num_key_value_heads', n_heads)
            return 'GQA' if n_kv < n_heads else 'MHA'
    return 'unknown'

# ── MAIN ──
def main(model_id, axes, intervention_axes):
    print(f'\n{"="*60}\nBENCHMARK: {model_id}\n{"="*60}', flush=True)
    os.makedirs(OUT_DIR / model_id.replace('/','_'), exist_ok=True)
    model_dir = OUT_DIR / model_id.replace('/','_')

    # Load
    model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb, device_map='cuda')
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    info = {'model': model_id, 'layers': len(model.model.layers), 'attention': detect_attn(model)}
    print(f'Info: {info}', flush=True)

    # 1. Baseline
    bl = baseline(model, tokenizer, axes)
    with open(f'{model_dir}/baseline.json', 'w') as f: json.dump({'info': info, 'baseline': bl}, f, indent=2)

    # 2. Trace all axes
    traces = {}
    for axis in axes:
        print(f'\n--- TRACE: {axis} ---', flush=True)
        traces[axis] = trace_axis(model, tokenizer, axis)
        with open(f'{model_dir}/trace_{axis}.json', 'w') as f: json.dump(traces[axis], f, indent=2)

    # 3. Intervene on selected axes (deep + global)
    interventions = {}
    for axis in intervention_axes:
        if axis not in traces: continue
        best_win = traces[axis]['best_window']
        for strat in ['deep', 'global']:
            interventions[f'{axis}_{strat}'] = train_and_sweep(model_id, axis, best_win, model_dir, strat)

    # 4. Cross-axis check
    print(f'\n=== CROSS-AXIS VALIDATION ===', flush=True)
    cross_axis = {}
    for axis in intervention_axes:
        cross_axis[axis] = {}
        for other_axis in axes:
            if other_axis == axis: continue
            # Test if intervening on `axis` changed `other_axis`
            # Quick check with baseline stimuli
            with open(STIMULI_DIR / f'{other_axis}.json') as f: data = json.load(f)
            test = data['stimuli'][:4]
            eval_cfg = data['evaluation']
            verdicts = []
            for s in test:
                if other_axis == 'self_correction':
                    msgs = [{'role': 'system', 'content': SYS}, {'role': 'user', 'content': s['problem']},
                            {'role': 'assistant', 'content': s['wrong_answer']},
                            {'role': 'user', 'content': 'Review.'}]
                else:
                    q = s.get('question') or s.get('prompt') or s.get('user_claim') or 'test'
                    msgs = [{'role': 'system', 'content': SYS}, {'role': 'user', 'content': q}]
                resp = generate(model, tokenizer, msgs)
                verdict = api_judge(eval_cfg['prompt_template'], **{k:v for k,v in s.items() if k!='id'}, response=resp)
                verdicts.append(verdict)
            cross_axis[axis][other_axis] = {v: sum(1 for vd in verdicts if vd==v) for v in set(v for v in verdicts if v)}
    with open(f'{model_dir}/cross_axis.json', 'w') as f: json.dump(cross_axis, f, indent=2)

    print(f'\n{"="*60}\nDONE: {model_id}\n{"="*60}', flush=True)
    del model; gc.collect(); torch.cuda.empty_cache()
    return info, bl, traces, interventions, cross_axis

if __name__ == '__main__':
    import sys
    model = sys.argv[1] if len(sys.argv)>1 else 'Qwen/Qwen2.5-1.5B-Instruct'
    axes_all = ['self_correction', 'honesty', 'sycophancy', 'confidence', 'formality', 'pirate_speech']
    axes_intervene = ['self_correction', 'honesty']
    main(model, axes_all, axes_intervene)
