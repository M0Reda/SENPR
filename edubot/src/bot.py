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
    Fetch top-k authoritative documents from policy_cards and course_cards.
    Training qa_turns (other students' Q&A) are only included when USE_EXAMPLES is True.
    """
    import re
    client, ef = _get_chroma()
    results: list[dict] = []

    # 1. QA turns (only when explicitly enabled; never an answer source by default)
    if USE_EXAMPLES:
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
                "score":    1 - dist,
            })

    def _clean_policy(doc: str) -> str:
        if USE_EXAMPLES:
            return doc
        # Strip other students' example responses; keep only authoritative regulations
        return doc.split("\n\nEXAMPLE ADVISOR RESPONSES")[0].strip()

    # 2. Policy card for this specific intent
    pol_col = client.get_collection("policy_cards", embedding_function=ef)
    if intent and intent not in ("__LOW_CONFIDENCE__", "__PRIVACY_BLOCK__", "__INJECTION_BLOCK__"):
        pol_id = f"policy_{re.sub(r'[^a-z0-9]', '_', intent.lower())}"
        try:
            pol_res = pol_col.get(ids=[pol_id], include=["documents", "metadatas"])
            if pol_res["documents"]:
                results.append({
                    "source":   "policy_cards",
                    "document": _clean_policy(pol_res["documents"][0]),
                    "meta":     pol_res["metadatas"][0],
                    "score":    1.0,
                })
        except Exception:
            pass

    # 3. Semantic policy_cards lookup
    pol_res2 = pol_col.query(
        query_texts=[query], n_results=min(2, pol_col.count()),
        include=["documents", "metadatas", "distances"],
    )
    for doc, meta, dist in zip(
        pol_res2["documents"][0],
        pol_res2["metadatas"][0],
        pol_res2["distances"][0],
    ):
        cleaned = _clean_policy(doc)
        if not any(r["source"] == "policy_cards" and r["document"] == cleaned for r in results):
            results.append({
                "source":   "policy_cards",
                "document": cleaned,
                "meta":     meta,
                "score":    1 - dist,
            })

    # 4. Course catalog cards (codes, titles, credits, prerequisites, offerings)
    try:
        crs_col = client.get_collection("course_cards", embedding_function=ef)
        crs_res = crs_col.query(
            query_texts=[query], n_results=min(4, crs_col.count()),
            include=["documents", "metadatas", "distances"],
        )
        for doc, meta, dist in zip(
            crs_res["documents"][0],
            crs_res["metadatas"][0],
            crs_res["distances"][0],
        ):
            results.append({
                "source":   "course_cards",
                "document": doc,
                "meta":     meta,
                "score":    1 - dist,
            })
    except Exception:
        pass

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


def _ollama_chat_stream(messages: list[dict], model: str):
    """Stream tokens from Ollama chat API; fall back to smaller model on OOM."""
    ollama = _get_ollama()
    _ensure_model(ollama, model)
    try:
        resp = ollama.chat(
            model=model,
            messages=messages,
            options={"temperature": TEMPERATURE, "num_predict": MAX_TOKENS, "num_ctx": NUM_CTX},
            stream=True,
        )
        for chunk in resp:
            yield chunk["message"]["content"]
    except Exception as e:
        err_str = str(e).lower()
        if "memory" in err_str or "oom" in err_str or model != FALLBACK:
            logger.warning(f"Falling back to {FALLBACK} due to: {e}")
            try:
                resp = ollama.chat(
                    model=FALLBACK,
                    messages=messages,
                    options={"temperature": TEMPERATURE, "num_predict": MAX_TOKENS, "num_ctx": NUM_CTX},
                    stream=True,
                )
                for chunk in resp:
                    yield chunk["message"]["content"]
            except Exception as e2:
                raise RuntimeError(f"Both {model} and {FALLBACK} failed: {e2}") from e2
        else:
            raise


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
        tool_block = "TOOL RESULTS:\n" + json.dumps(tool_results, ensure_ascii=False, separators=(",", ":"), default=str) + "\n\n"

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
    "1. What exact fact(s) is the student asking for?\n"
    "2. Copy the exact values (course codes, titles, grades, numbers) from TOOL RESULTS.\n"
    "3. Verify every single course code and grade matches TOOL RESULTS verbatim.\n"
    "Then write a clear, warm, first-person answer OUTSIDE the <think> tags using ONLY those verified facts."
)

KEYWORD_INTENTS = [
    (("graduat", "degree", "left to finish", "remaining courses", "what do i need"),
     "Graduation Requirements Audit"),
    (("final exam", "need on the final", "needed to pass", "score to pass", "pass the final"),
     "Required Final Exam Score"),
    (("7th", "12th", "marks", "coursework", "current course"), "Current Marks Summary"),
    (("prereq", "pre-req", "eligible to take", "can i take", "register", "enroll"),
     "Prerequisite Dependency Check"),
    (("improv",), "Course Improvement Window Policy"),
    (("fail", "retake", "repeat"), "Failed Courses to Repeat"),
    (("absen", "attend", "withdraw"), "Absence & Forced Withdrawal Warning"),
    (("standing",), "Standing Classification"),
    (("probation", "warning"), "Probation Consequences"),
    (("honor",), "Honors Eligibility Criteria"),
    (("gpa",), "GPA Calculation Methodology"),
    (("advisor", "adviser", "office"), "Advisor & Staff Contact"),
    (("overload", "max credit", "how many credit", "credit limit"), "Overload Permission"),
    (("transcript", "grade", "grades", "highest", "lowest", "best", "worst",
      "courses have i taken", "courses i took", "already finished", "list all", "full list", "all the courses"),
     "Transcript Summary"),
]

# Indexed record sections per intent: only send the slice of the student record
# actually needed for the question (cuts prompt size ~3x and prevents distraction).
INTENT_SECTIONS: dict[str, set[str]] = {
    "GPA Projection":                     {"student", "current_courses"},
    "GPA Calculation Methodology":        {"student", "semester_history"},
    "Required Final Exam Score":          {"student", "current_courses"},
    "Current Marks Summary":              {"student", "current_courses"},
    "Standing Classification":            {"student", "registration_rules_now", "repeats"},
    "Semester Level & Credits":           {"student", "registration_rules_now"},
    "Transcript Summary":                 {"student", "repeats"},
    "Graduation Requirements Audit":      {"student", "registration_rules_now", "current_courses", "remaining", "graduation_plan_projection", "repeats"},
    "Graduation Plan":                    {"student", "registration_rules_now", "current_courses", "remaining", "graduation_plan_projection"},
    "Next Term Registration":             {"student", "registration_rules_now", "current_courses", "remaining", "repeats"},
    "Overload Permission":                {"student", "registration_rules_now"},
    "Summer Graduating Overload Exception": {"student", "registration_rules_now", "remaining"},
    "Probation Load Restriction":         {"student", "registration_rules_now", "repeats"},
    "Probation Consequences":             {"student", "repeats"},
    "Honors Eligibility Criteria":        {"student", "honors_check"},
    "Prerequisite Dependency Check":      {"student", "current_courses", "remaining"},
    "Course Improvement Window Policy":   {"student", "repeats"},
    "Retake Grade Cap":                   {"student", "repeats", "current_courses"},
    "Failed Courses to Repeat":           {"student", "repeats", "current_courses"},
    "Failed Final Exam Threshold":        {"student", "current_courses"},
    "Absence & Forced Withdrawal Warning": {"student", "current_courses"},
    "Advisor & Staff Contact":            {"student", "advisor"},
}
DEFAULT_SECTIONS: set[str] = {"student", "registration_rules_now", "advisor", "current_courses", "repeats"}

CLARIFY_PROMPT = (
    "The question may be vague. Answer it as best you can from the STUDENT RECORD and "
    "TOOL RESULTS. Ask a short clarifying question ONLY if you truly cannot tell what "
    "is being asked; never ask the student for information that is already in their record."
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

def warmup() -> None:
    """
    Pre-load everything that is normally lazy so the FIRST student message is as
    fast as later ones: database, router, embedder, reranker, and the Ollama model
    (kept resident in GPU memory).
    """
    t0 = time.time()
    try:
        _get_db()
        from router import IntentRouter
        if IntentRouter.is_trained():
            IntentRouter.load()
        ranked = rerank("warm up", retrieve("warm up", None))   # loads embedder + reranker + chroma
        logger.info(f"warmup: retrieval ready ({len(ranked)} docs) in {time.time() - t0:.1f}s")
    except Exception as e:
        logger.warning(f"warmup (retrieval) failed: {e}")
    try:
        client = _get_ollama()
        _ensure_model(client, MODEL)
        # Same num_ctx as real requests, otherwise Ollama would reload the model.
        client.chat(
            model=MODEL,
            messages=[{"role": "user", "content": "hi"}],
            options={"num_predict": 1, "num_ctx": NUM_CTX},
            keep_alive=-1,
        )
        logger.info(f"warmup: LLM '{MODEL}' resident in {time.time() - t0:.1f}s")
    except Exception as e:
        logger.warning(f"warmup (LLM) failed: {e}")


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

    def _system_base(self, intent: str | None = None, message: str = "") -> str:
        """System prompt + indexed slice of this student's record + identity/rules."""
        sections = set(INTENT_SECTIONS.get(intent or "", DEFAULT_SECTIONS))
        if any(w in message.lower() for w in ("full transcript", "all courses", "list all", "every course")):
            sections.add("transcript")
        system_msg = self._tools.build_system_message(strip_email=True, include_sections=sections)
        prof = self._tools.get_student_profile()
        system_msg += (
            f"\n\nYOU ARE CURRENTLY TALKING TO: {prof['name']} (student ID {self._sid}). "
            "The STUDENT RECORD above is the ONLY source of facts about this student. "
            "Never use any other student's data. Use TOOL RESULTS (pre-computed, exact) "
            "for numbers; copy them, do not recompute. If a fact is not in the record or "
            "tool results, say you don't have it. "
            "Answer ONLY the current question; do not repeat answers from earlier turns. "
            "Never ask the student for course prerequisites or transcript facts — check TOOL RESULTS. "
            "The maximum credit load is only a LIMIT, never what the student needs. "
            "Only mention electives if electives_still_needed > 0. "
            "Credits remaining that are covered by courses currently in progress are NOT extra courses to find."
        )
        return system_msg

    def prefill(self) -> None:
        """
        Pre-evaluate this student's (large) system prompt in Ollama right after login,
        so Ollama's KV cache already holds it when the first question arrives.
        """
        try:
            self._refresh_student()
            client = _get_ollama()
            client.chat(
                model=MODEL,
                messages=[
                    {"role": "system", "content": self._system_base() + SCRATCHPAD_SUFFIX},
                    {"role": "user", "content": "hi"},
                ],
                options={"temperature": TEMPERATURE, "num_predict": 1, "num_ctx": NUM_CTX},
                keep_alive=-1,
            )
        except Exception as e:
            logger.warning(f"prefill failed for {self._sid}: {e}")

    def _prepare_turn(self, message: str) -> tuple[dict[str, Any], list[dict] | None, str | None]:
        """Run Route → Retrieve → Rerank → Tools → Prompt; returns (step, prompt_msgs, blocked_reply)."""
        self._refresh_student()
        step: dict[str, Any] = {
            "ts":           datetime.utcnow().isoformat(),
            "sid":          self._sid,
            "user_message": message,
        }

        # ── 1. Route ──────────────────────────────────────────────────────────
        t0 = time.time()
        route = self._route(message)
        t_route = time.time() - t0
        step["route"] = {
            "intent": route.intent,
            "confidence": route.confidence,
            "blocked": route.blocked,
        }

        # ── 2. Handle blocked messages ────────────────────────────────────────
        if route.blocked:
            reply = BLOCKED_RESPONSES.get(route.intent,
                    "I'm not able to help with that. Is there something else I can do for you?")
            return step, None, reply

        # ── 3. Retrieve (policy_cards + course_cards; qa_turns only if enabled) ─
        t1 = time.time()
        raw_results = retrieve(message, route.intent)
        ranked      = rerank(message, raw_results)
        t_rag = time.time() - t1
        step["retrieved"] = [
            {"source": r["source"], "intent": r["meta"].get("intent_category", ""),
             "score": round(r.get("rerank_score", r["score"]), 4),
             "snippet": r["document"][:200]}
            for r in ranked
        ]

        # ── 4. Call tools (always include verified baseline + intent tools) ───
        t2 = time.time()
        tool_results: dict[str, Any] = self._baseline_facts(message)
        if route.needs_tool:
            tool_results.update(self._call_tools(route.intent, message))
        t_tools = time.time() - t2
        step["tool_results"] = tool_results
        step["timing_s"] = {"route": round(t_route, 2), "rag": round(t_rag, 2), "tools": round(t_tools, 2)}

        # ── 5. Build system message (indexed student record slice) ────────────
        system_msg = self._system_base(intent=route.intent, message=message)

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
        return step, prompt_msgs, None

    def chat(self, message: str) -> str:
        """
        Process one student message and return the advisor's reply.

        The scratchpad (reasoning) is saved to the debug log but never returned.
        """
        start = time.time()
        step, prompt_msgs, blocked_reply = self._prepare_turn(message)
        if blocked_reply is not None or prompt_msgs is None:
            self._save_debug(step | {"reply": blocked_reply, "elapsed": time.time() - start})
            return blocked_reply or ""

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

    def chat_stream(self, message: str):
        """
        Stream the advisor's visible reply chunk-by-chunk (hiding <think>...</think>).
        Yields the accumulated visible reply text as it is generated.
        """
        start = time.time()
        step, prompt_msgs, blocked_reply = self._prepare_turn(message)
        if blocked_reply is not None or prompt_msgs is None:
            self._save_debug(step | {"reply": blocked_reply, "elapsed": time.time() - start})
            yield blocked_reply or ""
            return

        raw_parts: list[str] = []
        last_visible = ""
        for chunk in _ollama_chat_stream(prompt_msgs, model=MODEL):
            raw_parts.append(chunk)
            raw_so_far = "".join(raw_parts)
            # Hold back output only while still inside an opening <think> tag
            if "<think>" in raw_so_far and "</think>" not in raw_so_far:
                continue
            visible = self._strip_scratchpad(raw_so_far)
            if visible and visible != last_visible:
                last_visible = visible
                yield visible

        raw_reply = "".join(raw_parts)
        reply = self._strip_scratchpad(raw_reply)
        if reply != last_visible:
            yield reply

        step["raw_llm_reply"] = raw_reply[:2000]
        step["final_reply"] = reply
        step["elapsed_s"]   = round(time.time() - start, 2)
        self._history.append({"role": "user",      "content": message})
        self._history.append({"role": "assistant", "content": reply})
        self._save_debug(step)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _route(self, message: str):
        from router import RouteResult, LOW_CONFIDENCE
        if self._router is None:
            # Fallback: always "needs_tool" with low confidence
            return RouteResult(
                intent=LOW_CONFIDENCE, confidence=0.0,
                needs_tool=True, blocked=False, block_reason=None, top3=[],
            )
        res = self._router.route(message, session_sid=self._sid)
        if not res.blocked and self._requested_grade(message):
            # "which courses did I get an A in" is a transcript lookup, whatever the classifier says
            return RouteResult(
                intent="Transcript Summary", confidence=max(res.confidence, 0.9),
                needs_tool=True, blocked=False, block_reason=None, top3=res.top3,
            )
        if res.intent == LOW_CONFIDENCE and not res.blocked:
            # Keyword safety net: the classifier is unsure, but the topic is obvious.
            m = message.lower()
            for kws, intent in KEYWORD_INTENTS:
                if any(k in m for k in kws):
                    return RouteResult(
                        intent=intent, confidence=res.confidence,
                        needs_tool=True, blocked=False, block_reason=None, top3=res.top3,
                    )
            # Still unsure: give the model the student's facts anyway (never answer blind).
            return RouteResult(
                intent=res.intent, confidence=res.confidence,
                needs_tool=True, blocked=False, block_reason=None, top3=res.top3,
            )
        return res

    def _baseline_facts(self, message: str) -> dict[str, Any]:
        """
        Compact, pre-computed facts included on every turn so the LLM never
        has to guess or misread raw transcript JSON, even if routing is imperfect.
        """
        import re
        t = self._tools
        by_letter = t.grades_by_letter()
        letter_grades = [g for g in by_letter if g != "P"]
        sem_hist = t.get_semester_history()
        reg_terms = sum(1 for h in sem_hist if not h["term"].startswith("Summer"))
        sum_terms = sum(1 for h in sem_hist if h["term"].startswith("Summer"))
        facts: dict[str, Any] = {
            "verified_profile": t.get_student_profile(),
            "verified_standing_and_bands": t.check_standing(),
            "verified_semesters_attended": {
                "completed_regular_semesters": reg_terms,
                "completed_summer_semesters": sum_terms,
                "completed_total_terms": len(sem_hist),
                "current_term_in_progress": "First Semester / 2026-2027",
                "note": "semester_level is credit-based standing, NOT the count of semesters attended.",
            },
            "verified_advisor": t.get_advisor(),
            "verified_honors":  t.get_honors_status(),
            "verified_current_marks": t.get_current_marks(),
            "verified_highest_grade": (
                {letter_grades[0]: by_letter[letter_grades[0]]} if letter_grades else {}
            ),
            "verified_lowest_grade": (
                {letter_grades[-1]: by_letter[letter_grades[-1]]} if letter_grades else {}
            ),
        }
        # Auto-attach course-specific lookups for any course codes or titles mentioned
        codes = t.find_courses_in_message(message)
        if codes:
            facts["mentioned_courses"] = {
                code: t.course_status(code) for code in codes[:3]
            }
        return facts

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
            "Transcript Summary":          lambda: self._transcript_dispatch(message),
            "Graduation Requirements Audit": lambda: {"graduation_status": t.graduation_status()},
            "Graduation Plan":             lambda: {"graduation_status": t.graduation_status()},
            "Next Term Registration":      lambda: {"overload": t.check_overload_eligibility(),
                                                     "graduation_status": t.graduation_status()},
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
                    "graduation_status": t.graduation_status(),
                }
        except Exception as e:
            logger.error(f"Tool call failed for intent '{intent}': {e}")
            results = {"tool_error": str(e)}

        return results

    @staticmethod
    def _requested_grade(message: str) -> str | None:
        """Return a letter grade (e.g. 'A', 'B+') if the message asks about one."""
        import re
        m = re.search(
            r"\b(?:grade of|got an?|get an?|achieved? an?|received? an?|earned? an?|with an?)\s+([A-Da-dFf][+-]?)(?![A-Za-z])"
            r"|(?<![A-Za-z])([A-DF][+-]?)\s+(?:grade|in\b)",
            message,
        )
        if not m:
            return None
        g = (m.group(1) or m.group(2)).upper()
        return g if g in {"A+", "A", "A-", "B+", "B", "B-", "C+", "C", "C-", "D+", "D", "F"} else None

    def _transcript_dispatch(self, message: str) -> dict:
        t = self._tools
        by_letter = t.grades_by_letter()
        out: dict = {}
        letter_grades = [g for g in by_letter if g != "P"]
        if letter_grades:
            out["highest_grade"] = letter_grades[0]
            out["highest_grade_courses"] = by_letter[letter_grades[0]]
            out["lowest_grade"] = letter_grades[-1]
            out["lowest_grade_courses"] = by_letter[letter_grades[-1]]
        g = self._requested_grade(message)
        if g:
            out["requested_grade"] = g
            out["courses_with_exactly_that_grade"] = by_letter.get(g, [])
            out["note"] = (
                f"Match the grade EXACTLY ({g}); '{g}-' and '{g}+' are different grades. "
                "If the list is empty, say there are none."
            )
        out["grades_by_letter_latest_attempts"] = by_letter
        return out

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
        """Extract course code or title from message, else return remaining courses."""
        t = self._tools
        found = t.find_courses_in_message(message)
        if found:
            return {"course_statuses": {c: t.course_status(c) for c in found[:3]}}
        return {"remaining": t.remaining_required_courses()}

    def _improvement_dispatch(self, message: str) -> dict:
        t = self._tools
        found = t.find_courses_in_message(message)
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
