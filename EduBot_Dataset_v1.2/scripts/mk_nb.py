import json, sys
cells = []
def md(s): cells.append(dict(cell_type="markdown", metadata={}, source=s))
def code(s): cells.append(dict(cell_type="code", metadata={}, execution_count=None, outputs=[], source=s))

md("""# EduBot: AASTMT Computer Engineering AI Advisor (Colab starter)

This notebook covers five steps:
1. Load the dataset (9 CSV tables plus the advising tickets).
2. Look at a ticket and the **student record** the bot sees.
3. Build the prompt for **any** student straight from the tables, using the same code that created the training data.
4. Fine-tune a chat model with LoRA (TRL + PEFT).
5. Ask the bot questions and run a simple accuracy check on held-out students.

**Runtime:** Runtime → Change runtime type → **T4 GPU** (free) or better.""")
code("""# Upload EduBot_Dataset_v3.zip to the Colab file panel (or copy it from Google Drive), then:
!unzip -q -o EduBot_Dataset_v3.zip -d edubot
import sys; sys.path.append('/content/edubot/scripts')   # edubot_context.py lives here
%cd /content/edubot""")
md("## 1. Load the tables and tickets")
code("""import pandas as pd, json
students = pd.read_csv('Students.csv', dtype=str)
qa = pd.read_csv('Advising_QA.csv')                      # one row per message (ticket_id, turn_no, role, message)
tickets = pd.read_json('EduBot_Training/tickets.jsonl', lines=True)
print(students.shape, qa.shape, tickets.shape)
tickets['intent'].value_counts()""")
md("## 2. One ticket and the record EduBot sees")
code("""t = tickets.sample(1, random_state=1).iloc[0]
print(t.ticket_id, '|', t.intent, '|', t.status, '| student', t.student_id)
for m in t.messages:
    print(f"\\n[{m['role']}] {m['content']}")""")
code("""from edubot_context import EduBotDB, SYSTEM_PROMPT
db = EduBotDB('.')
rec = db.public(db.record(t.student_id))
print(json.dumps(rec['student'], indent=2))
print('keys in record:', list(rec))""")
md("""## 3. Training files

`EduBot_Training/sft_{train,validation,test}.jsonl` hold one conversation per line in the standard chat format:
`{"messages": [{"role":"system", ...policy + STUDENT RECORD...}, {"role":"user"}, {"role":"assistant"}, ...]}`.
The split is **by student**, so no test student appears in training.""")
code("""from datasets import load_dataset
ds = load_dataset('json', data_files={'train': 'EduBot_Training/sft_train.jsonl',
                                      'validation': 'EduBot_Training/sft_validation.jsonl',
                                      'test': 'EduBot_Training/sft_test.jsonl'})
ds""")
md("""## 4. Fine-tune with LoRA

`Qwen/Qwen2.5-1.5B-Instruct` is a good free-Colab default. Swap in Llama-3.2-3B-Instruct, Gemma-2-2b-it, etc. if you have a bigger GPU.
Each example is about 4k tokens, because the student record is in the prompt. On a T4, start with a subset (for example 1,500 tickets) to test the pipeline.""")
code("""!pip -q install -U transformers trl peft accelerate bitsandbytes datasets""")
code("""import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

MODEL = 'Qwen/Qwen2.5-1.5B-Instruct'
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, device_map='auto',
    quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16))

train = ds['train'].shuffle(seed=0).select(range(1500))      # remove .select(...) to use everything
val = ds['validation'].select(range(100))

cfg = SFTConfig(output_dir='edubot-lora', num_train_epochs=1, per_device_train_batch_size=1,
                gradient_accumulation_steps=8, learning_rate=2e-4, logging_steps=20,
                max_length=6144,                 # older TRL versions: max_seq_length=6144
                gradient_checkpointing=True, fp16=True, eval_strategy='steps', eval_steps=200,
                assistant_only_loss=True)        # learn only the bot's replies; drop this if your TRL/template doesn't support it
peft = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules='all-linear', task_type='CAUSAL_LM')
trainer = SFTTrainer(model=model, args=cfg, train_dataset=train, eval_dataset=val,
                     processing_class=tok, peft_config=peft)
trainer.train()
trainer.save_model('edubot-lora')""")
md("""## 5. Talk to EduBot

At inference time the portal does exactly what the training data did: it builds the student's record from the database and puts it in the system message.""")
code("""def ask(student_id, history):
    \"\"\"history: list of {'role': 'user'|'assistant', 'content': str}\"\"\"
    msgs = [{'role': 'system', 'content': db.system_message(student_id)}] + history
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors='pt').to(model.device)
    out = model.generate(ids, max_new_tokens=400, do_sample=False)
    return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)

sid = students.sample(1, random_state=7).student_id.iloc[0]
print(ask(sid, [{'role': 'user', 'content': 'what do I still need to graduate?'}]))""")
md("""### Quick check on held-out students
For each test ticket, generate the first reply and check that the numbers in the reference answer (GPA, credits, marks) also appear in the model's answer.""")
code("""import re
def nums(s): return set(re.findall(r'\\d+(?:\\.\\d+)?', s))
hits = 0; N = 50
for ex in ds['test'].select(range(N)):
    m = ex['messages']
    ref = m[2]['content']
    msgs = m[:2]
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors='pt').to(model.device)
    pred = tok.decode(model.generate(ids, max_new_tokens=400, do_sample=False)[0][ids.shape[1]:], skip_special_tokens=True)
    r, p = nums(ref), nums(pred)
    hits += len(r & p) / max(1, len(r))
print('numeric recall:', round(hits / N, 3))""")
nb = dict(cells=cells, metadata=dict(kernelspec=dict(name="python3", display_name="Python 3"), accelerator="GPU",
          colab=dict(provenance=[], gpuType="T4")), nbformat=4, nbformat_minor=5)
json.dump(nb, open(sys.argv[1], "w"), indent=1)
