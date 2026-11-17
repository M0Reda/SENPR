# EduBot — Local Academic Advisor Chatbot

A fully **offline** hybrid-RAG chatbot for the AASTMT Computer Engineering Department.
No cloud API calls at runtime. Everything runs on your local machine.

---

## Architecture

```
Student message
      │
      ▼
┌─────────────┐    blocks PII/injection
│ Safety Rules│────────────────────────────► Blocked reply
└──────┬──────┘
       │ passes
       ▼
┌─────────────┐    TF-IDF + LogisticRegression
│Intent Router│────────────────────────────► intent + confidence
└──────┬──────┘
       │
       ▼
┌─────────────┐    ChromaDB (qa_turns + policy_cards + course_cards)
│  Retrieval  │    all-MiniLM-L6-v2 embeddings → cross-encoder rerank
└──────┬──────┘
       │
       ▼
┌─────────────┐    Deterministic pandas tools (EduBotDB)
│  Tools      │    GPA, transcripts, prereqs, overload, graduation plan
└──────┬──────┘
       │
       ▼
┌─────────────┐    System prompt + student record + context + tool results
│ Ollama LLM  │    qwen2.5:7b-instruct (or llama3.2:3b fallback)
│ (scratchpad)│    Private reasoning in <think>...</think> tags
└──────┬──────┘
       │
       ▼
  Final answer (warm, first-person, no hallucination)
```

---

## Setup

### 1. Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com) installed and running

### 2. Install Ollama and pull the model

```powershell
# Download from https://ollama.com/download
# Then pull the model:
ollama pull qwen2.5:7b-instruct
# Optional smaller fallback:
ollama pull llama3.2:3b
```

### 3. Install Python dependencies

```powershell
cd edubot
pip install -r requirements.txt
```

### 4. Data

The dataset lives at `../EduBot_Dataset_v1.2/` relative to `edubot/`.
The path is set in `config.yaml → data_dir`.
**`Users.csv` is never indexed or passed to the LLM.**

---

## Running the Pipeline

### Step 1 — Data audit (run once, check the report)

```powershell
python -X utf8 src/data_audit.py
# Output: reports/data_audit_report.md
```

### Step 2 — Build the vector index

```powershell
python src/build_index.py
# Downloads all-MiniLM-L6-v2 (~90 MB, once)
# Builds chroma_db/ with 3 collections
```

### Step 3 — Train the intent router

```powershell
python src/router.py
# Trains TF-IDF + LogisticRegression on 80% of TRAIN tickets
# Saves chroma_db/router.pkl
```

### Step 4 — Run unit tests

```powershell
pip install pytest pytest-cov
pytest tests/ -v --tb=short
```

### Step 5 — Evaluate on validation set (tune here, not on test)

```powershell
python src/evaluate.py --split validation --skip-llm
# Add --paraphrase to build the hard test set (requires Ollama)
```

### Step 6 — Launch the chat UI

**Option A: Local Python**
```powershell
pip install gradio
python src/app.py
# Open http://localhost:7860
# Log in with any student ID from Students.csv (e.g. 201003465)
```

**Option B: Docker Service (Recommended)**
```powershell
# Build and start all services (EduBot + Ollama + automatic model puller):
docker compose up -d

# Check status:
docker compose ps

# View logs:
docker compose logs -f edubot

# Open http://localhost:7860 in your browser!
# To shut down:
docker compose down
```

### Step 7 — Final test evaluation (touch once)

```powershell
python src/evaluate.py --split test
# Output: reports/eval_test.md + reports/eval_test.json
```

### Step 8 — (Optional) LoRA fine-tune

```powershell
# Enable in config.yaml: lora.enabled: true
# Install GPU deps:
pip install unsloth trl peft bitsandbytes transformers datasets accelerate
python src/train_lora.py
```

---

## Project Structure

```
edubot/
├── config.yaml          ← all tunable knobs
├── requirements.txt
├── README.md
├── src/
│   ├── data_audit.py    ← Stage 1: data checks
│   ├── build_index.py   ← Stage 2: ChromaDB index
│   ├── tools.py         ← Stage 3: deterministic pandas tools
│   ├── router.py        ← Stage 4: intent classifier
│   ├── bot.py           ← Stage 5: full RAG pipeline
│   ├── evaluate.py      ← Stage 6: evaluation suite
│   ├── train_lora.py    ← Stage 7: optional LoRA fine-tune
│   └── app.py           ← Stage 8: Gradio chat UI
├── tests/
│   └── test_tools.py    ← pytest unit tests for tools
├── chroma_db/           ← vector index + router.pkl (generated)
├── logs/                ← per-session debug logs (JSONL)
└── reports/             ← audit + eval reports (Markdown)
```

---

## Configuration (`config.yaml`)

| Key | Default | Description |
|---|---|---|
| `model` | `qwen2.5:7b-instruct` | Primary Ollama model |
| `fallback_model` | `llama3.2:3b` | Used if primary is OOM |
| `embed_model` | `all-MiniLM-L6-v2` | Sentence-transformers embedding |
| `rerank_model` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder reranker |
| `retrieval.top_k` | `10` | ChromaDB initial fetch |
| `retrieval.rerank_top_n` | `4` | After reranking, keep this many |
| `router.threshold` | `0.50` | Below this → ask clarifying question |
| `lora.enabled` | `false` | Enable LoRA fine-tuning |

---

## Privacy

- `Users.csv` is **never** loaded by `build_index.py`, `bot.py`, or any index file.
- Student email is **stripped** from the system prompt before passing to the LLM.
- Each session is scoped to **one authenticated student ID**. Tools raise `PermissionError` if another student's data is requested.
- Advisor email is also excluded; only the office room is shown.

---

## Data Notes

| Dataset fact | Explanation |
|---|---|
| `Enrollments (31,821) > Transcripts (25,910)` | 5,911 current-term (Fall 2026-2027) in-progress courses; they'll move to Transcripts when grades are released |
| `Students.gpa` has 164 nulls | Semester-1 students with no completed term yet |
| `professor_id` vs `instructor_id` | Professor owns the course (exams, grades); instructor delivers the lab section |
| QA data is templated | Expect inflated retrieval scores on the original test set; use the paraphrased hard set for real generalisation |

---

## Evaluation Metrics

| Metric | Tool |
|---|---|
| Retrieval recall@k, MRR | `evaluate.py` |
| Router intent accuracy | `router.py` + `evaluate.py` |
| Semantic similarity | `sentence-transformers` cosine |
| LLM-as-judge (1-5) | Ollama rubric |
| Rule-based correctness | pandas against CSVs |
| Adversarial robustness | Predefined test cases |
| Paraphrased hard test | Ollama paraphrase + re-eval |
