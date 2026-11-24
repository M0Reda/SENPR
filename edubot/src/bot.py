"""
bot.py -- Stage 5: RAG Answer Pipeline
========================================
Orchestrates the full answer flow per turn:
  Auth → Safety → Route → Retrieve → Rerank → Tools → Prompt → LLM → Answer

Usage:
    from src.bot import EduBot
    bot = EduBot(session_sid="201003465")
    answer = bot.chat("how is my GPA calculated?")
    print(answer)
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str((ROOT / "../EduBot_Dataset_v1.2/scripts").resolve()))

cfg: dict[str, Any] = yaml.safe_load((ROOT / "config.yaml").read_text())
DATA_DIR   = (ROOT / cfg["data_dir"]).resolve()
INDEX_DIR  = ROOT / cfg["index_dir"]
EMBED_MODEL= cfg["embed_model"]
TOP_K      = cfg["retrieval"]["top_k"]
RERANK_N   = cfg["retrieval"]["rerank_top_n"]
MIN_CONF   = cfg["retrieval"]["min_confidence"]
MODEL      = os.environ.get("OLLAMA_MODEL", cfg["model"])
FALLBACK   = cfg["fallback_model"]
TEMPERATURE= cfg["temperature"]
MAX_TOKENS = cfg["max_tokens"]
OLLAMA_URL = cfg["ollama_base_url"]
NUM_CTX    = cfg.get("num_ctx", 8192)
USE_EXAMPLES = cfg["retrieval"].get("use_training_examples", False)

# ── Logging ───────────────────────────────────────────────────────────────────

logger = logging.getLogger("edubot")
logger.setLevel(logging.DEBUG)
_fh = logging.FileHandler(LOG_DIR / "bot.log", encoding="utf-8")
_fh.setLevel(logging.DEBUG)
logger.addHandler(_fh)

# ── Lazy imports (heavy; only load once) ─────────────────────────────────────

_db      = None
_chroma  = None
_ef      = None
_reranker= None
_ollama  = None


_db_stamp = None


def _data_stamp() -> tuple:
    """Fingerprint of the CSV tables (mtime+size) so we notice newly created students."""
    out = []
    for f in sorted(DATA_DIR.glob("*.csv")):
        try:
            st = f.stat()
            out.append((f.name, st.st_mtime_ns, st.st_size))
        except OSError:
            pass
    return tuple(out)


def _get_db(sid: str | None = None):
    """Return the DB, reloading it whenever the data files changed or `sid` is unknown."""
    global _db, _db_stamp
    stamp = _data_stamp()
    stale = _db is None or stamp != _db_stamp or (sid is not None and sid not in _db.students)
    if stale:
        from edubot_context import EduBotDB
        _db = EduBotDB(str(DATA_DIR))
        _db_stamp = stamp
    return _db


def _get_chroma():
    global _chroma, _ef
    if _chroma is None:
        import chromadb
        from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
        _chroma = chromadb.PersistentClient(path=str(INDEX_DIR))
        _ef     = SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
    return _chroma, _ef


def _get_reranker():
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder
        _reranker = CrossEncoder(cfg["rerank_model"])
    return _reranker


def _get_ollama():
    global _ollama
    if _ollama is None:
        import os
        import ollama
        host = os.environ.get("OLLAMA_HOST", OLLAMA_URL)
        _ollama = ollama.Client(host=host)
    return _ollama


# ── Retrieval ─────────────────────────────────────────────────────────────────

def retrieve(query: str, intent: str | None, k: int = TOP_K) -> list[dict[str, Any]]:
    """
    Fetch top-k documents from qa_turns and policy_cards.
    If intent is known, filter qa_turns by intent metadata for better precision.
    """
    client, ef = _get_chroma()
    results: list[dict] = []

    # QA turns
    qa_col = client.get_collection("qa_turns", embedding_function=ef)
    where  = {"intent_category": intent} if intent and intent not in ("__LOW_CONFIDENCE__",
              "__PRIVACY_BLOCK__", "__INJECTION_BLOCK__") else None
    qa_res = qa_col.query(
        query_texts=[query],
        n_results=min(k, qa_col.count()),
        where=where if where else None,
        include=["documents", "metadatas", "distances"],
    )
    for doc, meta, dist in zip(
        qa_res["documents"][0],
        qa_res["metadatas"][0],
        qa_res["distances"][0],
    ):
        results.append({
            "source":   "qa_turns",
            "document": doc,
            "meta":     meta,
            "score":    1 - dist,   # cosine similarity
        })

    # Policy card for this specific intent
    if intent and intent not in ("__LOW_CONFIDENCE__", "__PRIVACY_BLOCK__", "__INJECTION_BLOCK__"):
        import re
        pol_col  = client.get_collection("policy_cards", embedding_function=ef)
        pol_id   = f"policy_{re.sub(r'[^a-z0-9]', '_', intent.lower())}"
        try:
            pol_res = pol_col.get(ids=[pol_id], include=["documents", "metadatas"])
            if pol_res["documents"]:
                results.append({
                    "source":   "policy_cards",
                    "document": pol_res["documents"][0],
                    "meta":     pol_res["metadatas"][0],
                    "score":    1.0,   # exact intent match
                })
        except Exception:
            pass

    # Fallback: also query policy_cards by semantic similarity
    pol_col = client.get_collection("policy_cards", embedding_function=ef)
    pol_res2 = pol_col.query(
        query_texts=[query], n_results=2,
        include=["documents", "metadatas", "distances"],
    )
    for doc, meta, dist in zip(
        pol_res2["documents"][0],
        pol_res2["metadatas"][0],
        pol_res2["distances"][0],
    ):
        if {"source": "policy_cards", "document": doc} not in [{"source": r["source"], "document": r["document"]} for r in results]:
            results.append({
                "source":   "policy_cards",
                "document": doc,
                "meta":     meta,
                "score":    1 - dist,
            })

    return results


def rerank(query: str, results: list[dict]) -> list[dict]:
    """Cross-encoder reranking; returns top RERANK_N results."""
    if not results:
        return results
    reranker = _get_reranker()
    pairs = [(query, r["document"]) for r in results]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(results, scores), key=lambda x: x[1], reverse=True)
    for r, s in ranked:
        r["rerank_score"] = float(s)
    return [r for r, _ in ranked[:RERANK_N]]


# ── LLM calls ─────────────────────────────────────────────────────────────────

def _ensure_model(client, model: str) -> None:
    """Pull the model on demand if the Ollama server does not have it yet."""
    try:
        names = {m.get("model") or m.get("name") for m in client.list().get("models", [])}
        if model in names or f"{model}:latest" in names:
            return
    except Exception:
        return
    logger.warning(f"Model {model} not found on Ollama server; pulling (first run only)...")
    client.pull(model)


def _ollama_chat(messages: list[dict], model: str, stream: bool = False) -> str:
    """Call Ollama chat API; fall back to smaller model on OOM."""
    ollama = _get_ollama()
    _ensure_model(ollama, model)
    try:
        resp = ollama.chat(
            model=model,
            messages=messages,
            options={"temperature": TEMPERATURE, "num_predict": MAX_TOKENS, "num_ctx": NUM_CTX},
            stream=stream,
        )
        if stream:
            parts = []
            for chunk in resp:
                parts.append(chunk["message"]["content"])
            return "".join(parts)
        return resp["message"]["content"]
    except Exception as e:
        err_str = str(e).lower()
        if "memory" in err_str or "oom" in err_str or model != FALLBACK:
            logger.warning(f"Falling back to {FALLBACK} due to: {e}")
            try:
                resp = ollama.chat(
                    model=FALLBACK,
                    messages=messages,
                    options={"temperature": TEMPERATURE, "num_predict": MAX_TOKENS, "num_ctx": NUM_CTX},
                )
                return resp["message"]["content"]
            except Exception as e2:
                raise RuntimeError(f"Both {model} and {FALLBACK} failed: {e2}") from e2
        raise


def _build_prompt(
    system_msg: str,
    history: list[dict],
    user_msg: str,
    context_passages: list[dict],
    tool_results: dict[str, Any],
) -> list[dict]:
    """
    Assemble the full chat message list for the LLM.

    Structure:
      [system] system_prompt + student_record
      ... (conversation history)
      [user]   CONTEXT: <retrieved> | TOOL RESULTS: <json> | QUESTION: <message>
    """
    context_block = ""
    if context_passages:
        context_block = "RETRIEVED CONTEXT:\n"
        for i, p in enumerate(context_passages, 1):
            intent = p["meta"].get("intent_category", p["meta"].get("intent", ""))
            context_block += f"[{i}] ({p['source']}; intent: {intent})\n{p['document'][:600]}\n\n"

    tool_block = ""
    if tool_results:
        tool_block = "TOOL RESULTS:\n" + json.dumps(tool_results, ensure_ascii=False, indent=2, default=str) + "\n\n"

    user_content = (
        f"{context_block}"
        f"{tool_block}"
        f"STUDENT'S QUESTION:\n{user_msg}"
    )

    messages = [{"role": "system", "content": system_msg}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_content})
    return messages


SCRATCHPAD_SUFFIX = (
    "\n\nBefore answering, write a private scratchpad inside <think>...</think> tags:\n"
    "1. What exactly is the student asking?\n"
    "2. What facts/rules do I need from the context and tools?\n"
    "3. Apply the rule or arithmetic step by step.\n"
    "4. Double-check any numbers.\n"
    "Then write a short, warm, first-person answer OUTSIDE the <think> tags."
)

CLARIFY_PROMPT = (
    "The student's question is ambiguous or I don't have enough confidence to answer. "
    "Ask ONE short, friendly clarifying question to get the specific information needed. "
    "Do not answer yet."
)

BLOCKED_RESPONSES = {
    "__PRIVACY_BLOCK__":   (
        "I can only share information about your own academic record. "
        "If you need to contact another student or a staff member, "
        "please reach out to your advisor."
    ),
    "__INJECTION_BLOCK__": (
        "I'm here to help with your academic questions — "
        "let's keep things on track! What would you like to know?"
    ),
}


# ── Main bot class ─────────────────────────────────────────────────────────────

class EduBot:
    """
    Session-scoped academic advisor chatbot.

    Parameters
    ----------
    session_sid : str   the authenticated student ID
    stream      : bool  stream tokens to stdout if True
    """

    def __init__(self, session_sid: str, stream: bool = False) -> None:
        self._sid    = str(session_sid)
        self._stream = stream
        self._history: deque[dict] = deque(maxlen=20)  # 10 turns = 20 messages
        self._debug_log: list[dict] = []

        # Lazy-load all heavy components
        self._router = self._load_router()
        self._refresh_student()

    def _refresh_student(self) -> None:
        """Re-read the student's data so answers are never stale / from another student."""
        from tools import StudentTools
        self._db = _get_db(self._sid)
        if self._sid not in self._db.students:
            raise ValueError(f"Unknown student ID: {self._sid}")
        self._tools = StudentTools(self._db, session_sid=self._sid)

    @staticmethod
    def _load_router():
        from router import IntentRouter
        if IntentRouter.is_trained():
            return IntentRouter.load()
        # If not trained yet, return a no-op router
        logger.warning("Router not trained — using keyword fallback only.")
        return None

    # ── Main entry point ──────────────────────────────────────────────────────

    def chat(self, message: str) -> str:
        """
        Process one student message and return the advisor's reply.

        The scratchpad (reasoning) is saved to the debug log but never returned.
        """
        start = time.time()
        self._refresh_student()
        step: dict[str, Any] = {
            "ts":           datetime.utcnow().isoformat(),
            "sid":          self._sid,
            "user_message": message,
        }

        # ── 1. Route ──────────────────────────────────────────────────────────
        route = self._route(message)
        step["route"] = {
            "intent": route.intent,
            "confidence": route.confidence,
            "blocked": route.blocked,
        }

        # ── 2. Handle blocked messages ────────────────────────────────────────
        if route.blocked:
            reply = BLOCKED_RESPONSES.get(route.intent,
                    "I'm not able to help with that. Is there something else I can do for you?")
            self._save_debug(step | {"reply": reply, "elapsed": time.time() - start})
            return reply

        # ── 3. Retrieve ───────────────────────────────────────────────────────
        # Training data (other students' Q&A) is NOT an answer source; it is only
        # optional style guidance and is off by default.
        if USE_EXAMPLES:
            ranked = rerank(message, retrieve(message, route.intent))
        else:
            ranked = []
        step["retrieved"] = [
            {"source": r["source"], "intent": r["meta"].get("intent_category", ""),
             "score": round(r.get("rerank_score", r["score"]), 4),
             "snippet": r["document"][:200]}
            for r in ranked
        ]

        # ── 4. Call tools if needed ───────────────────────────────────────────
        tool_results: dict[str, Any] = {}
        if route.needs_tool:
            tool_results = self._call_tools(route.intent, message)
        step["tool_results"] = tool_results

        # ── 5. Build system message (student record) ──────────────────────────
        system_msg = self._tools.build_system_message(strip_email=True)
        prof = self._tools.get_student_profile()
        system_msg += (
            f"\n\nYOU ARE CURRENTLY TALKING TO: {prof['name']} (student ID {self._sid}). "
            "The STUDENT RECORD above is the ONLY source of facts about this student. "
            "Never use any other student's data. Use TOOL RESULTS (pre-computed, exact) "
            "for numbers; copy them, do not recompute. If a fact is not in the record or "
            "tool results, say you don't have it."
        )

        # ── 6. Build prompt ───────────────────────────────────────────────────
        if route.intent == "__LOW_CONFIDENCE__":
            system_msg += f"\n\nINSTRUCTION: {CLARIFY_PROMPT}"

        prompt_msgs = _build_prompt(
            system_msg   = system_msg + SCRATCHPAD_SUFFIX,
            history      = list(self._history),
            user_msg     = message,
            context_passages = ranked,
            tool_results = tool_results,
        )

        # ── 7. LLM call ───────────────────────────────────────────────────────
        raw_reply = _ollama_chat(prompt_msgs, model=MODEL, stream=self._stream)
        step["raw_llm_reply"] = raw_reply[:2000]

        # ── 8. Strip scratchpad ───────────────────────────────────────────────
        reply = self._strip_scratchpad(raw_reply)
        step["final_reply"] = reply
        step["elapsed_s"]   = round(time.time() - start, 2)

        # ── 9. Update memory ──────────────────────────────────────────────────
        self._history.append({"role": "user",      "content": message})
        self._history.append({"role": "assistant",  "content": reply})

        # ── 10. Save debug log ────────────────────────────────────────────────
        self._save_debug(step)

        return reply

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _route(self, message: str):
        from router import RouteResult, LOW_CONFIDENCE
        if self._router is None:
            # Fallback: always "needs_tool" with low confidence
            return RouteResult(
                intent=LOW_CONFIDENCE, confidence=0.0,
                needs_tool=True, blocked=False, block_reason=None, top3=[],
            )
        return self._router.route(message, session_sid=self._sid)

    def _call_tools(self, intent: str, message: str) -> dict[str, Any]:
        """Dispatch to the appropriate tool(s) based on intent."""
        t = self._tools
        results: dict[str, Any] = {}

        intent_tool_map: dict[str, Any] = {
            "GPA Projection":              lambda: {"gpa": t.calculate_gpa(), "current": t.get_current_marks()},
            "GPA Calculation Methodology": lambda: {"gpa": t.calculate_gpa(), "history": t.get_semester_history()},
            "Required Final Exam Score":   lambda: self._required_final(message),
            "Current Marks Summary":       lambda: {"marks": t.get_current_marks()},
            "Standing Classification":     lambda: {"standing": t.check_standing()},
            "Semester Level & Credits":    lambda: {"profile": t.get_student_profile()},
            "Transcript Summary":          lambda: {"transcript": t.get_transcript()},
            "Graduation Requirements Audit": lambda: {"profile": t.get_student_profile(),
                                                       "remaining": t.remaining_required_courses()},
            "Graduation Plan":             lambda: {"plan": t.get_graduation_plan(),
                                                     "profile": t.get_student_profile()},
            "Next Term Registration":      lambda: {"overload": t.check_overload_eligibility(),
                                                     "remaining": t.remaining_required_courses()},
            "Overload Permission":         lambda: {"overload": t.check_overload_eligibility()},
            "Summer Graduating Overload Exception": lambda: {"profile": t.get_student_profile(),
                                                              "overload": t.check_overload_eligibility()},
            "Probation Load Restriction":  lambda: {"overload": t.check_overload_eligibility(),
                                                     "standing": t.check_standing()},
            "Probation Consequences":      lambda: {"standing": t.check_standing()},
            "Honors Eligibility Criteria": lambda: {"honors": t.get_honors_status(),
                                                     "profile": t.get_student_profile()},
            "Prerequisite Dependency Check": lambda: self._prereq_dispatch(message),
            "Course Improvement Window Policy": lambda: self._improvement_dispatch(message),
            "Retake Grade Cap":            lambda: {"profile": t.get_student_profile(),
                                                     "transcript": t.get_transcript()},
            "Failed Courses to Repeat":    lambda: {"profile": t.get_student_profile()},
            "Failed Final Exam Threshold": lambda: {"marks": t.get_current_marks()},
            "Absence & Forced Withdrawal Warning": lambda: {"marks": t.get_current_marks()},
            "Advisor & Staff Contact":     lambda: {"advisor": t.get_advisor()},
        }

        try:
            fn = intent_tool_map.get(intent)
            if fn:
                results = fn()
            else:
                # Generic fallback: include profile + marks
                results = {
                    "profile": t.get_student_profile(),
                    "marks":   t.get_current_marks(),
                }
        except Exception as e:
            logger.error(f"Tool call failed for intent '{intent}': {e}")
            results = {"tool_error": str(e)}

        return results

    def _required_final(self, message: str) -> dict:
        """Try to extract course code from message, else return all current courses."""
        import re
        t = self._tools
        cur = t.get_current_marks()
        codes = [c["code"] for c in cur]
        # Look for a code pattern in the message
        found = re.findall(r'\b([A-Z]{2,3}\d{3,})\b', message)
        if found and found[0] in codes:
            return {"final_calc": t.calculate_required_final(found[0])}
        # Return calcs for all current courses
        return {"final_calcs": [t.calculate_required_final(c) for c in codes]}

    def _prereq_dispatch(self, message: str) -> dict:
        """Extract course code from message, else return a generic prereq check."""
        import re
        t = self._tools
        found = re.findall(r'\b([A-Z]{2,3}\d{3,})\b', message)
        if found:
            return {"prereqs": t.check_prereqs(found[0])}
        return {"remaining": t.remaining_required_courses()}

    def _improvement_dispatch(self, message: str) -> dict:
        import re
        t = self._tools
        found = re.findall(r'\b([A-Z]{2,3}\d{3,})\b', message)
        if found:
            return {"window": t.check_improvement_window(found[0])}
        return {"transcript": t.get_transcript()}

    @staticmethod
    def _strip_scratchpad(text: str) -> str:
        """Remove <think>...</think> blocks from the LLM output."""
        import re
        cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        cleaned = re.sub(r"<think>.*", "", cleaned, flags=re.DOTALL)  # unclosed block
        cleaned = cleaned.replace("</think>", "").strip()
        return cleaned or text.replace("<think>", "").replace("</think>", "").strip()

    def _save_debug(self, step: dict) -> None:
        """Append this turn's debug info to the session log."""
        log_path = LOG_DIR / f"debug_{self._sid}.jsonl"
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(step, ensure_ascii=False, default=str) + "\n")

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def session_sid(self) -> str:
        return self._sid

    def clear_history(self) -> None:
        self._history.clear()
