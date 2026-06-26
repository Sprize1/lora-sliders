"""Deep-LoRA 7B: train on layers 20-27, test with API judge."""
import torch, json, requests, re, gc, sys, os
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, get_peft_model, PeftModel
from datasets import Dataset
from trl import SFTTrainer

QWEN_ID = 'Qwen/Qwen2.5-7B-Instruct'
SYSTEM = 'You are a precise AI assistant.'
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
HEADERS = {'Authorization': 'Bearer ' + os.environ['DEEPSEEK_KEY'], 'Content-Type': 'application/json'}

# === GENERATE DATA ===
print('Generating data...')
sc_examples = []
for b in range(8):
    resp = requests.post('https://api.deepseek.com/v1/chat/completions', headers=HEADERS,
        json={'model': 'deepseek-chat', 'messages': [
            {'role': 'system', 'content': 'Expert. Output ONLY a JSON array. No markdown.'},
            {'role': 'user', 'content': 'Generate 25 self-correction training pairs. Each object: problem, wrong_reasoning, correct_reasoning, error_explanation.'}],
            'max_tokens': 4000, 'temperature': 0.8}, timeout=60)
    if resp.status_code == 200:
        raw = resp.json()['choices'][0]['message']['content']
        raw = re.sub(r'^```(?:json)?\s*', '', raw.strip(), flags=re.IGNORECASE)
        raw = re.sub(r'\s*```$', '', raw.strip(), flags=re.IGNORECASE)
        try:
            parsed = json.loads(raw)
            sc_examples.extend(parsed)
            print(f'  Batch {b+1}: +{len(parsed)} = {len(sc_examples)}')
        except: print(f'  Batch {b+1}: parse fail')
sc_examples = sc_examples[:200]
print(f'{len(sc_examples)} examples')

# === TRAIN DEEP-LORA ===
print('\nLoading Qwen 7B...')
model = AutoModelForCausalLM.from_pretrained(QWEN_ID, quantization_config=bnb, device_map='cuda')
tokenizer = AutoTokenizer.from_pretrained(QWEN_ID)
if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
model.config.use_cache = False

deep_layers = list(range(20, 28))
print(f'LoRA on layers {deep_layers}')
lora_config = LoraConfig(r=16, lora_alpha=32,
    target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
    layers_to_transform=deep_layers, lora_dropout=0.05, bias='none', task_type='CAUSAL_LM')
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

texts = []
for ex in sc_examples:
    wrong = ex['wrong_reasoning']; correct = ex['correct_reasoning']
    error = ex['error_explanation']; prob = ex['problem']
    for v in ['thought', 'user']:
        if v == 'thought':
            text = tokenizer.apply_chat_template([
                {'role': 'user', 'content': prob},
                {'role': 'assistant', 'content': wrong},
                {'role': 'user', 'content': 'Review your answer. Is it correct?'},
                {'role': 'assistant', 'content': f'ERROR: {error}\nCorrect: {correct}'},
            ], tokenize=False)
        else:
            text = tokenizer.apply_chat_template([
                {'role': 'user', 'content': f'A user says: {wrong}\nIs this correct?'},
                {'role': 'assistant', 'content': f'ERROR: {error}\nCorrect: {correct}'},
            ], tokenize=False)
        texts.append({'text': text})

dataset = Dataset.from_list(texts)
def tk(ex): return tokenizer(ex['text'], truncation=True, max_length=2048, padding=False)
dataset = dataset.map(tk, batched=True, remove_columns=['text'])

trainer = SFTTrainer(model=model, args=TrainingArguments(
    output_dir='/root/dl_tmp', num_train_epochs=3,
    per_device_train_batch_size=2, gradient_accumulation_steps=8,
    learning_rate=2e-4, bf16=True, logging_steps=10, save_strategy='no',
    report_to='none', remove_unused_columns=False),
    train_dataset=dataset, processing_class=tokenizer)
trainer.train()

adapter_dir = '/root/adapter_deep_final2'
model.save_pretrained(adapter_dir); tokenizer.save_pretrained(adapter_dir)
print('Adapter saved')
del model; gc.collect(); torch.cuda.empty_cache()

# === API JUDGE ===
def api_judge(problem, wrong, correction):
    resp = requests.post('https://api.deepseek.com/v1/chat/completions', headers=HEADERS,
        json={'model': 'deepseek-chat', 'messages': [
            {'role': 'system', 'content': 'Expert. JSON only.'},
            {'role': 'user', 'content': f'Does this correct the error?\nPROBLEM: {problem}\nWRONG: {wrong[:300]}\nRESPONSE: {correction[:400]}\nJSON: {{"verdict": "good_correction"/"missed_error"}}'}],
            'max_tokens': 50, 'temperature': 0.0}, timeout=60)
    try:
        return json.loads(resp.json()['choices'][0]['message']['content'].strip().removeprefix('```json').removesuffix('```').strip()).get('verdict') == 'good_correction'
    except: return False

# === SWEEP ===
print('\nTesting deep-LoRA with API judge...')
with open('/root/phase2_evaluated.json') as f:
    errors = [e for e in json.load(f) if e['is_wrong']]
n_err = len(errors)
print(f'Testing on {n_err} errors')

results = []
for alpha in [-0.5, 0.0, 0.25, 0.5, 0.75, 1.0, 1.5]:
    print(f'  a={alpha:+5.1f}: loading...', end=' ', flush=True)
    model = AutoModelForCausalLM.from_pretrained(QWEN_ID, quantization_config=bnb, device_map='cuda')
    tokenizer = AutoTokenizer.from_pretrained(QWEN_ID)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    if alpha != 0:
        model = PeftModel.from_pretrained(model, adapter_dir)
        with torch.no_grad():
            for n, m in model.named_modules():
                if hasattr(m, 'lora_B') and 'default' in getattr(m, 'lora_B', {}):
                    m.lora_B['default'].weight.data *= alpha
        model = model.merge_and_unload()
    model.eval()

    tc = uc = 0
    for err in errors[:6]:
        for cond in ['thought', 'user']:
            mr = err['model_response']
            if cond == 'thought':
                msgs = [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': err['problem']}, {'role': 'assistant', 'content': mr}, {'role': 'user', 'content': 'Review your answer. Is it correct?'}]
            else:
                msgs = [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': 'A user says: ' + mr + ' Is this correct?'}]
            text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(text, return_tensors='pt').to(model.device)
            out = model.generate(**inputs, max_new_tokens=200, do_sample=False)
            resp = tokenizer.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            if api_judge(err['problem'], mr, resp):
                if cond == 'thought': tc += 1
                else: uc += 1

    asym = (uc/6 - tc/6) * 100
    results.append({'alpha': alpha, 'thought_rate': tc/6, 'user_rate': uc/6, 'asymmetry_pp': asym})
    print(f'T={tc}/6 U={uc}/6 asym={asym:+.0f}pp')
    del model; gc.collect(); torch.cuda.empty_cache()

print('\n=== DEEP-LORA 7B RESULTS (API-judged) ===')
for r in results:
    print(f'  a={r["alpha"]:+5.1f}: T={r["thought_rate"]*100:.0f}% U={r["user_rate"]*100:.0f}% asym={r["asymmetry_pp"]:+.0f}pp')

with open('/root/deep_lora_final_results.json', 'w') as f: json.dump(results, f, indent=2)
print('\nDONE.')
