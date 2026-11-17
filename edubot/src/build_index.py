"""
build_index.py -- Stage 2: Build ChromaDB Vector Index
=======================================================
Builds three ChromaDB collections:
  1. qa_turns       -- EduBot responses from TRAIN tickets
  2. policy_cards   -- One distilled policy card per intent
  3. course_cards   -- One card per course (code + title + prereqs + offerings)

Run:
    cd edubot
    python src/build_index.py

The index is persisted in ./chroma_db/ and reused by bot.py at inference time.
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str((ROOT / "../EduBot_Dataset_v1.2/scripts").resolve()))

cfg: dict[str, Any] = yaml.safe_load((ROOT / "config.yaml").read_text())
DATA_DIR  = (ROOT / cfg["data_dir"]).resolve()
INDEX_DIR = ROOT / cfg["index_dir"]
INDEX_DIR.mkdir(parents=True, exist_ok=True)

EMBED_MODEL = cfg["embed_model"]
CURRENT_TERM = "First Semester / 2026-2027"


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_train_sids() -> set[str]:
    """Return the set of student IDs in the training split."""
    train_path = DATA_DIR / "EduBot_Training" / "sft_train.jsonl"
    sids: set[str] = set()
    with open(train_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            conv = json.loads(line)
            msgs = conv.get("messages", conv.get("conversations", []))
            for m in msgs:
                hit = re.search(r'"student_id"\s*:\s*"(\d+)"', m.get("content", ""))
                if hit:
                    sids.add(hit.group(1))
                    break
    print(f"  Train split: {len(sids):,} students")
    return sids


def chroma_client():
    """Return a persistent ChromaDB client."""
    import chromadb
    return chromadb.PersistentClient(path=str(INDEX_DIR))


def embedding_fn():
    """Return a ChromaDB-compatible sentence-transformers embedding function."""
    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
    return SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)


# ── Collection 1: QA turns (TRAIN only) ──────────────────────────────────────

def build_qa_turns(client, ef, train_sids: set[str]) -> None:
    """Index EduBot response turns from TRAIN tickets."""
    print("\n  Building collection: qa_turns ...")

    # Drop and recreate for a clean rebuild
    try:
        client.delete_collection("qa_turns")
    except Exception:
        pass
    col = client.create_collection("qa_turns", embedding_function=ef,
                                   metadata={"hnsw:space": "cosine"})

    qa = pd.read_csv(str(DATA_DIR / "Advising_QA.csv"))
    # Only TRAIN students, only EduBot (advisor) turns
    train_qa = qa[
        (qa["student_id"].astype(str).isin(train_sids)) &
        (qa["role"] == "edubot")
    ].copy()

    # Group by ticket — build a document that is "student_question\nbot_answer"
    # so retrieval hits on natural language questions, not just bot responses
    student_qa = qa[qa["role"] == "student"][["ticket_id", "turn_no", "message"]].copy()
    student_qa.columns = ["ticket_id", "student_turn_no", "student_message"]

    # Match each bot turn to the preceding student turn
    docs, metas, ids = [], [], []
    for _, row in train_qa.iterrows():
        tid  = row["ticket_id"]
        tno  = int(row["turn_no"])
        bot_msg = str(row["message"])

        # Find the most recent student turn before this bot turn
        prev = student_qa[
            (student_qa["ticket_id"] == tid) &
            (student_qa["student_turn_no"] < tno)
        ]
        student_msg = prev.iloc[-1]["student_message"] if not prev.empty else ""

        # Composite document for embedding
        doc = f"Student: {student_msg}\nAdvisor: {bot_msg}" if student_msg else bot_msg

        docs.append(doc)
        metas.append({
            "ticket_id":       tid,
            "student_id":      str(row["student_id"]),
            "intent_category": str(row["intent_category"]),
            "turn_no":         tno,
            "bot_response":    bot_msg,         # stored for retrieval display
        })
        ids.append(f"{tid}_t{tno}")

    # Batch upsert (ChromaDB limit: 5,000 per call)
    BATCH = 2000
    total = len(docs)
    for i in range(0, total, BATCH):
        col.upsert(
            documents=docs[i:i+BATCH],
            metadatas=metas[i:i+BATCH],
            ids=ids[i:i+BATCH],
        )
        print(f"    Upserted {min(i+BATCH, total):,}/{total:,}", end="\r")
    print(f"  qa_turns: {total:,} documents indexed.                    ")


# ── Collection 2: Policy cards (one per intent) ───────────────────────────────

def build_policy_cards(client, ef, train_sids: set[str]) -> None:
    """
    Distil one policy card per intent from the SYSTEM_PROMPT + representative
    TRAIN bot responses. Each card is a clean, dense summary of the rules.
    """
    print("\n  Building collection: policy_cards ...")
    try:
        client.delete_collection("policy_cards")
    except Exception:
        pass
    col = client.create_collection("policy_cards", embedding_function=ef,
                                   metadata={"hnsw:space": "cosine"})

    qa = pd.read_csv(str(DATA_DIR / "Advising_QA.csv"))
    train_bot = qa[
        (qa["student_id"].astype(str).isin(train_sids)) &
        (qa["role"] == "edubot")
    ]

    # Import the regulations text
    sys.path.insert(0, str(DATA_DIR / "scripts"))
    from edubot_context import SYSTEM_PROMPT

    # Extract just the REGULATIONS block from the system prompt
    regulations_text = SYSTEM_PROMPT.split("REGULATIONS\n")[-1].strip()

    # Group bot responses by intent and take a diverse sample
    docs, metas, ids = [], [], []
    for intent, grp in train_bot.groupby("intent_category"):
        responses = grp["message"].dropna().tolist()
        # Sample up to 10 diverse responses for the card body
        sample = responses[:10]
        source_tickets = grp["ticket_id"].unique().tolist()[:20]

        card = (
            f"INTENT: {intent}\n\n"
            f"REGULATIONS:\n{regulations_text}\n\n"
            f"EXAMPLE ADVISOR RESPONSES FOR THIS INTENT:\n"
            + "\n---\n".join(f"• {r}" for r in sample)
        )

        docs.append(card)
        metas.append({
            "intent": intent,
            "source_ticket_ids": json.dumps(source_tickets),
            "n_examples": len(sample),
        })
        ids.append(f"policy_{re.sub(r'[^a-z0-9]', '_', intent.lower())}")

    col.upsert(documents=docs, metadatas=metas, ids=ids)
    print(f"  policy_cards: {len(docs)} intent cards indexed.")


# ── Collection 3: Course cards (full catalogue) ───────────────────────────────

def build_course_cards(client, ef) -> None:
    """One document per course: code + title + type + prereqs + recent offerings."""
    print("\n  Building collection: course_cards ...")
    try:
        client.delete_collection("course_cards")
    except Exception:
        pass
    col = client.create_collection("course_cards", embedding_function=ef,
                                   metadata={"hnsw:space": "cosine"})

    courses  = pd.read_csv(str(DATA_DIR / "Courses.csv"))
    prereqs  = pd.read_csv(str(DATA_DIR / "Course_Prerequisites.csv"))
    offerings= pd.read_csv(str(DATA_DIR / "Course_Offerings.csv"))

    # Build prereq map
    prereq_map: dict[str, list[str]] = defaultdict(list)
    for _, row in prereqs.iterrows():
        prereq_map[row["course_code"]].append(row["prereq_course_code"])

    # Build offering summary (most recent 3 offerings per course)
    off_summary: dict[str, list[str]] = defaultdict(list)
    def off_label(row: pd.Series) -> str:
        term_map = {"Fall": "First Semester", "Spring": "Second Semester", "Summer": "Summer Semester"}
        return f"{term_map.get(row['semester_term'], row['semester_term'])} / {row['academic_year']}"

    for _, row in offerings.iterrows():
        off_summary[row["course_code"]].append(off_label(row))

    docs, metas, ids = [], [], []
    for _, c in courses.iterrows():
        code     = c["course_code"]
        title    = c["course_title"]
        cr       = int(c["credit_hours"])
        ctype    = c["course_type"]
        rec_term = c.get("recommended_term", "")
        pqs      = prereq_map.get(code, [])
        offs     = sorted(set(off_summary.get(code, [])))[-3:]  # 3 most recent

        card = (
            f"Course: {code} — {title}\n"
            f"Credit hours: {cr}\n"
            f"Type: {ctype}\n"
            f"Recommended term: {rec_term}\n"
            f"Prerequisites: {', '.join(pqs) if pqs else 'None'}\n"
            f"Recent offerings: {', '.join(offs) if offs else 'Not yet offered'}\n"
        )

        docs.append(card)
        metas.append({
            "course_code": code,
            "course_title": title,
            "credit_hours": cr,
            "course_type": ctype,
        })
        ids.append(f"course_{code.lower()}")

    col.upsert(documents=docs, metadatas=metas, ids=ids)
    print(f"  course_cards: {len(docs)} courses indexed.")


# ── Verification ──────────────────────────────────────────────────────────────

def verify_index(client, ef) -> None:
    """Run a quick sanity-check query on each collection."""
    print("\n  Verifying index ...")
    test_queries = {
        "qa_turns":      "how is my GPA calculated?",
        "policy_cards":  "what is the maximum credit hours for overload?",
        "course_cards":  "what are the prerequisites for embedded systems?",
    }
    for col_name, query in test_queries.items():
        col = client.get_collection(col_name, embedding_function=ef)
        res = col.query(query_texts=[query], n_results=2)
        docs = res["documents"][0]
        print(f"  [{col_name}] Query: '{query}'")
        for i, d in enumerate(docs):
            snippet = d[:120].replace("\n", " ")
            print(f"    {i+1}. {snippet} ...")
    print("\n  Verification complete.")


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\nBuilding EduBot vector index ...")
    print(f"  Embedding model : {EMBED_MODEL}")
    print(f"  Index directory : {INDEX_DIR}")

    print("\n  Loading train split ...")
    train_sids = get_train_sids()

    client = chroma_client()
    ef     = embedding_fn()

    build_qa_turns(client, ef, train_sids)
    build_policy_cards(client, ef, train_sids)
    build_course_cards(client, ef)
    verify_index(client, ef)

    # Collection summary
    print("\n  Index summary:")
    for name in ["qa_turns", "policy_cards", "course_cards"]:
        col = client.get_collection(name, embedding_function=ef)
        print(f"    {name}: {col.count():,} documents")

    print("\nDone. Index ready.")
