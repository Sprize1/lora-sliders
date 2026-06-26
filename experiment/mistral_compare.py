"""Mistral 7B: deep-LoRA vs global-LoRA comparison."""
import json, os, requests, torch, gc, re, numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, PeftModel
from datasets import Dataset
from trl import SFTTrainer, SFTConfig

bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
H = {'Authorization': 'Bearer ' + os.environ['DEEPSEEK_KEY'], 'Content-Type': 'application/json'}
SYS = 'You are a precise AI assistant.'
with open('/root/phase2_evaluated.json') as f:
    errors = [e for e in json.load(f) if e['is_wrong']]

def api_judge(p, w, c):
    r = requests.post('https://api.deepseek.com/v1/chat/completions', headers=H,
        json={'model': 'deepseek-chat', 'messages': [
            {'role': 'system', 'content': 'Expert. JSON only.'},
            {'role': 'user', 'content': 'Does this correct the error?\nPROBLEM: ' + p + '\nWRONG: ' + w[:300] + '\nRESPONSE: ' + c[:400] + '\nJSON: {"verdict": "good_correction"/"missed_error"}'}],
            'max_tokens': 50, 'temperature': 0.0}, timeout=60)
    try:
        raw = r.json()['choices'][0]['message']['content'].strip()
        for px in ['```json', '```']: raw = raw.removeprefix(px).removesuffix(px).strip()
        return json.loads(raw).get('verdict') == 'good_correction'
    except: return False

# Gen data
print('Gen data...', flush=True)
sc = []
for b in range(8):
    r = requests.post('https://api.deepseek.com/v1/chat/completions', headers=H,
        json={'model': 'deepseek-chat', 'response_format': {'type': 'json_object'},
              'messages': [
                  {'role': 'system', 'content': 'You are a JSON generator. Only output valid JSON.'},
                  {'role': 'user', 'content': 'Generate 25 self-correction training pairs as a JSON object with key "pairs" containing an array. Each object: problem, wrong_reasoning, correct_reasoning, error_explanation.'}],
              'max_tokens': 4000, 'temperature': 0.8}, timeout=60)
    if r.status_code == 200:
        try:
            raw = r.json()['choices'][0]['message']['content']
            raw = re.sub(r'^```(?:json)?\s*', '', raw.strip(), flags=re.IGNORECASE)
            raw = re.sub(r'\s*```$', '', raw.strip(), flags=re.IGNORECASE)
            data = json.loads(raw)
            items = data.get('pairs', data) if isinstance(data, dict) else data
            sc.extend(items)
        except Exception as e:
            print(f'  B{b+1} parse err: {str(e)[:80]}', flush=True)
    print(f'  B{b+1}: {len(sc)}', flush=True)
sc = sc[:200]
print(f'{len(sc)} ex', flush=True)

# Save data for reuse
with open('/root/mistral_training_data.json', 'w') as f:
    json.dump(sc, f)

def train_and_eval(name, layers_to_transform):
    print(f'\n=== {name} ===', flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        'mistralai/Mistral-7B-Instruct-v0.3',
        quantization_config=bnb, device_map='cuda')
    tokenizer = AutoTokenizer.from_pretrained('mistralai/Mistral-7B-Instruct-v0.3')
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    model.config.use_cache = False

    lc = LoraConfig(r=16, lora_alpha=32,
                    target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'],
                    layers_to_transform=layers_to_transform,
                    lora_dropout=0.05, bias='none', task_type='CAUSAL_LM')
    model = get_peft_model(model, lc)

    texts = []
    for ex in sc:
        w = ex['wrong_reasoning']; c = ex['correct_reasoning']; e = ex['error_explanation']; p = ex['problem']
        for v in ['thought', 'user']:
            if v == 'thought':
                t = tokenizer.apply_chat_template([
                    {'role': 'user', 'content': p},
                    {'role': 'assistant', 'content': w},
                    {'role': 'user', 'content': 'Review your answer. Is it correct?'},
                    {'role': 'assistant', 'content': f'ERROR: {e}\nCorrect: {c}'}], tokenize=False)
            else:
                t = tokenizer.apply_chat_template([
                    {'role': 'user', 'content': f'A user says: {w}\nIs this correct?'},
                    {'role': 'assistant', 'content': f'ERROR: {e}\nCorrect: {c}'}], tokenize=False)
            texts.append({'text': t})
    ds = Dataset.from_list(texts)
    def tk(ex): return tokenizer(ex['text'], truncation=True, max_length=2048, padding=False)
    ds = ds.map(tk, batched=True, remove_columns=['text'])
    tr = SFTTrainer(model=model, args=SFTConfig(
        output_dir=f'/root/tmp_{name}', num_train_epochs=3,
        per_device_train_batch_size=2, gradient_accumulation_steps=8, learning_rate=2e-4,
        bf16=True, logging_steps=10, save_strategy='no', report_to='none',
        remove_unused_columns=False), train_dataset=ds, processing_class=tokenizer)
    tr.train()
    ad = f'/root/adapter_{name}'
    model.save_pretrained(ad); tokenizer.save_pretrained(ad)
    print(f'Trained ({name})', flush=True)
    del model; gc.collect(); torch.cuda.empty_cache()

    # Sweep
    print(f'  Sweep {name}:', flush=True)
    results = {}
    for alpha in [-0.5, 0.0, 0.25, 0.5, 0.75, 1.0, 1.5]:
        model = AutoModelForCausalLM.from_pretrained(
            'mistralai/Mistral-7B-Instruct-v0.3',
            quantization_config=bnb, device_map='cuda')
        tokenizer = AutoTokenizer.from_pretrained('mistralai/Mistral-7B-Instruct-v0.3')
        if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
        if alpha != 0:
            model = PeftModel.from_pretrained(model, ad)
            with torch.no_grad():
                for nm, m in model.named_modules():
                    if hasattr(m, 'lora_B') and 'default' in getattr(m, 'lora_B', {}):
                        m.lora_B['default'].weight.data *= alpha
            model = model.merge_and_unload()
        model.eval()
        tc = uc = 0
        for err in errors[:12]:
            for cond in ['thought', 'user']:
                mr = err['model_response']
                if cond == 'thought':
                    msgs = [{'role': 'system', 'content': SYS}, {'role': 'user', 'content': err['problem']},
                            {'role': 'assistant', 'content': mr}, {'role': 'user', 'content': 'Review your answer. Is it correct?'}]
                else:
                    msgs = [{'role': 'system', 'content': SYS},
                            {'role': 'user', 'content': f'A user says: {mr} Is this correct?'}]
                text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                inputs = tokenizer(text, return_tensors='pt').to(model.device)
                out = model.generate(**inputs, max_new_tokens=200, do_sample=False)
                resp = tokenizer.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
                if api_judge(err['problem'], mr, resp):
                    if cond == 'thought': tc += 1
                    else: uc += 1
        asym = (uc/12 - tc/12) * 100
        results[alpha] = {'T': tc, 'U': uc, 'asym': asym}
        print(f'    a={alpha:+5.1f}: T={tc}/12 U={uc}/12 asym={asym:+.0f}pp', flush=True)
        del model; gc.collect(); torch.cuda.empty_cache()
    return results

# === RUN BOTH ===
results_deep = train_and_eval('deep_lora', list(range(24, 32)))
results_global = train_and_eval('global_lora', None)  # None = all layers

# Compare
print('\n=== COMPARISON ===', flush=True)
print(f'{"Alpha":>8}  {"Deep T":>7} {"Deep U":>7} {"Deep asym":>10}  {"Global T":>8} {"Global U":>8} {"Global asym":>11}', flush=True)
for alpha in [-0.5, 0.0, 0.25, 0.5, 0.75, 1.0, 1.5]:
    d = results_deep[alpha]
    g = results_global[alpha]
    print(f'{alpha:+8.1f}  {d["T"]:>7} {d["U"]:>7} {d["asym"]:+10.0f}pp  {g["T"]:>8} {g["U"]:>8} {g["asym"]:+11.0f}pp', flush=True)

# Save
with open('/root/mistral_compare.json', 'w') as f:
    json.dump({'deep_lora': {str(k): v for k, v in results_deep.items()},
               'global_lora': {str(k): v for k, v in results_global.items()}}, f, indent=2)

# Winner analysis
deep_range = max(v['asym'] for v in results_deep.values()) - min(v['asym'] for v in results_deep.values())
global_range = max(v['asym'] for v in results_global.values()) - min(v['asym'] for v in results_global.values())
print(f'\nDeep-LoRA range: {deep_range}pp', flush=True)
print(f'Global-LoRA range: {global_range}pp', flush=True)
if global_range > deep_range:
    print('GLOBAL LoRA wins for Mistral (distributed behavioral circuit)', flush=True)
else:
    print('DEEP LoRA wins for Mistral', flush=True)

with open('/root/mistral_compare_done', 'w') as f: f.write('DONE')
print('ALL DONE', flush=True)
