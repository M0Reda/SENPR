"""
router.py -- Stage 4: Intent Router
=====================================
Two-layer routing:
  1. ML classifier (TF-IDF + LogisticRegression) trained on TRAIN tickets.
  2. Rule layer: detects privacy violations, prompt injection, low-confidence.

Usage:
    from src.router import IntentRouter
    router = IntentRouter.load()                    # loads from disk
    result = router.route(message="my GPA?", session_sid="201003465")
    # -> RouteResult(intent="GPA Projection", needs_tool=True, confidence=0.92)

Training:
    python src/router.py  (trains and saves the classifier)
"""

from __future__ import annotations

import json
import pickle
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

cfg: dict[str, Any] = yaml.safe_load((ROOT / "config.yaml").read_text())
DATA_DIR = (ROOT / cfg["data_dir"]).resolve()
ROUTER_DIR = ROOT / "chroma_db"   # store classifier next to the index

# ── Intent constants ──────────────────────────────────────────────────────────

# Special synthetic labels (not real intents)
PRIVACY_BLOCK    = "__PRIVACY_BLOCK__"
INJECTION_BLOCK  = "__INJECTION_BLOCK__"
LOW_CONFIDENCE   = "__LOW_CONFIDENCE__"
OUT_OF_SCOPE     = "__OUT_OF_SCOPE__"

# Intents that require a live student-data lookup (tools.py call)
TOOL_INTENTS: frozenset[str] = frozenset({
    "GPA Projection",
    "Required Final Exam Score",
    "Current Marks Summary",
    "Prerequisite Dependency Check",
    "Graduation Plan",
    "Graduation Requirements Audit",
    "Next Term Registration",
    "Standing Classification",
    "GPA Calculation Methodology",
    "Semester Level & Credits",
    "Transcript Summary",
    "Honors Eligibility Criteria",
    "Retake Grade Cap",
    "Course Improvement Window Policy",
    "Failed Courses to Repeat",
    "Failed Final Exam Threshold",
    "Overload Permission",
    "Summer Graduating Overload Exception",
    "Probation Load Restriction",
    "Probation Consequences",
    "Absence & Forced Withdrawal Warning",
    "Elective Selection",
    "Advisor & Staff Contact",
})

# ── Safety patterns ───────────────────────────────────────────────────────────

_INJECTION_PATTERNS = re.compile(
    r"\b(ignore|disregard|forget|override|bypass|pretend|act as|you are now|"
    r"system prompt|new instructions?|jailbreak|DAN|developer mode)\b",
    re.IGNORECASE,
)

_PII_PATTERNS = re.compile(
    r"\b(email|password|passwd|phone|address|national\s*id|social\s*security|"
    r"credit\s*card|bank|ssn)\b",
    re.IGNORECASE,
)

_OTHER_STUDENT_PATTERN = re.compile(r"\b(20\d{7}|21\d{7}|22\d{7}|23\d{7}|24\d{7})\b")

# ── Result type ───────────────────────────────────────────────────────────────

@dataclass
class RouteResult:
    intent: str                  # intent label (or PRIVACY_BLOCK etc.)
    confidence: float            # classifier softmax probability [0, 1]
    needs_tool: bool             # whether to call tools.py
    blocked: bool                # True if safety filter fired
    block_reason: str | None     # human-readable reason if blocked
    top3: list[tuple[str, float]]  # top-3 intent candidates


# ── Router ────────────────────────────────────────────────────────────────────

class IntentRouter:
    """
    Two-layer intent router.

    Attributes
    ----------
    _clf : sklearn Pipeline (TF-IDF + LogisticRegression)
    _threshold : float
    """

    _SAVE_PATH = ROUTER_DIR / "router.pkl"

    def __init__(self, clf, threshold: float, label_names: list[str]) -> None:
        self._clf         = clf
        self._threshold   = threshold
        self._label_names = label_names

    # ── Rule layer (runs before ML) ───────────────────────────────────────────

    @staticmethod
    def _safety_check(message: str, session_sid: str) -> RouteResult | None:
        """Return a blocked RouteResult if any safety rule fires, else None."""
        # 1. Prompt injection
        if _INJECTION_PATTERNS.search(message):
            return RouteResult(
                intent=INJECTION_BLOCK, confidence=1.0, needs_tool=False,
                blocked=True, block_reason="Potential prompt injection detected.",
                top3=[],
            )
        # 2. PII request
        if _PII_PATTERNS.search(message):
            return RouteResult(
                intent=PRIVACY_BLOCK, confidence=1.0, needs_tool=False,
                blocked=True, block_reason="Request appears to seek personal/private information.",
                top3=[],
            )
        # 3. Other student's ID in the message
        ids_in_msg = _OTHER_STUDENT_PATTERN.findall(message)
        for sid in ids_in_msg:
            if sid != session_sid:
                return RouteResult(
                    intent=PRIVACY_BLOCK, confidence=1.0, needs_tool=False,
                    blocked=True,
                    block_reason=f"Request references another student's ID ({sid}).",
                    top3=[],
                )
        return None

    # ── ML layer ──────────────────────────────────────────────────────────────

    def _classify(self, message: str) -> RouteResult:
        proba = self._clf.predict_proba([message])[0]
        top_idx = proba.argsort()[::-1]
        top3 = [(self._label_names[i], float(proba[i])) for i in top_idx[:3]]
        best_label, best_conf = top3[0]

        if best_conf < self._threshold:
            return RouteResult(
                intent=LOW_CONFIDENCE, confidence=best_conf, needs_tool=False,
                blocked=False, block_reason=None, top3=top3,
            )

        return RouteResult(
            intent=best_label,
            confidence=best_conf,
            needs_tool=best_label in TOOL_INTENTS,
            blocked=False,
            block_reason=None,
            top3=top3,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def route(self, message: str, session_sid: str) -> RouteResult:
        """
        Route a student message.

        Parameters
        ----------
        message    : str   the student's raw message
        session_sid: str   the authenticated student ID for this session

        Returns
        -------
        RouteResult
        """
        # Safety first
        blocked = self._safety_check(message, session_sid)
        if blocked:
            return blocked
        # ML
        return self._classify(message)

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self) -> None:
        ROUTER_DIR.mkdir(parents=True, exist_ok=True)
        with open(self._SAVE_PATH, "wb") as f:
            pickle.dump({"clf": self._clf, "threshold": self._threshold,
                         "labels": self._label_names}, f)
        print(f"  Router saved to {self._SAVE_PATH}")

    @classmethod
    def load(cls) -> "IntentRouter":
        with open(cls._SAVE_PATH, "rb") as f:
            d = pickle.load(f)
        return cls(d["clf"], d["threshold"], d["labels"])

    @classmethod
    def is_trained(cls) -> bool:
        return cls._SAVE_PATH.exists()


# ── Training ──────────────────────────────────────────────────────────────────

def _load_train_messages() -> tuple[list[str], list[str]]:
    """Load (message, intent) pairs from the training JSONL (student turns only)."""
    train_path = DATA_DIR / "EduBot_Training" / "sft_train.jsonl"
    # Also read from the flat Advising_QA for any student in the train split
    import pandas as pd

    # Get train student IDs from the JSONL
    train_sids: set[str] = set()
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
                    train_sids.add(hit.group(1))
                    break

    # Load QA and filter to train students
    qa = pd.read_csv(str(DATA_DIR / "Advising_QA.csv"))
    train_qa = qa[(qa["student_id"].astype(str).isin(train_sids)) & (qa["role"] == "student")]
    messages = train_qa["message"].tolist()
    labels   = train_qa["intent_category"].tolist()
    print(f"  Training router on {len(messages):,} student turns "
          f"across {len(train_sids):,} students, "
          f"{len(set(labels))} intents.")
    return messages, labels


def train_router(threshold: float | None = None) -> IntentRouter:
    """Train the intent classifier and return a fitted IntentRouter."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.model_selection import cross_val_score

    if threshold is None:
        threshold = cfg["router"]["threshold"]
    max_features = cfg["router"]["max_features"]

    messages, labels = _load_train_messages()

    clf = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=max_features,
            sublinear_tf=True,
            min_df=2,
        )),
        ("lr", LogisticRegression(
            max_iter=1000,
            C=5.0,
            class_weight="balanced",
            solver="lbfgs",
        )),
    ])
    clf.fit(messages, labels)

    # Quick cross-val sanity check (3-fold)
    scores = cross_val_score(clf, messages, labels, cv=3, scoring="accuracy")
    print(f"  3-fold CV accuracy: {scores.mean():.3f} ± {scores.std():.3f}")

    label_names = list(clf.classes_)
    router = IntentRouter(clf, threshold, label_names)
    router.save()
    return router


# ── Evaluation helper ─────────────────────────────────────────────────────────

def evaluate_router(router: IntentRouter, split: str = "validation") -> dict[str, Any]:
    """Evaluate router accuracy and return a classification report dict."""
    from sklearn.metrics import classification_report, confusion_matrix
    import pandas as pd

    split_path = DATA_DIR / "EduBot_Training" / f"sft_{split}.jsonl"
    split_sids: set[str] = set()
    with open(split_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            conv = json.loads(line)
            msgs = conv.get("messages", conv.get("conversations", []))
            for m in msgs:
                hit = re.search(r'"student_id"\s*:\s*"(\d+)"', m.get("content", ""))
                if hit:
                    split_sids.add(hit.group(1))
                    break

    qa = pd.read_csv(str(DATA_DIR / "Advising_QA.csv"))
    val_qa = qa[(qa["student_id"].astype(str).isin(split_sids)) & (qa["role"] == "student")]
    messages = val_qa["message"].tolist()
    labels   = val_qa["intent_category"].tolist()

    preds = []
    for msg in messages:
        r = router.route(msg, session_sid="000000000")  # dummy sid; no safety concern
        preds.append(r.intent if not r.blocked else "__BLOCKED__")

    report = classification_report(labels, preds, output_dict=True, zero_division=0)
    print(f"\n  Router evaluation on {split} set ({len(messages)} turns):")
    print(f"  Accuracy: {report['accuracy']:.3f}")
    return {
        "split": split,
        "n_samples": len(messages),
        "report": report,
        "confusion_matrix": confusion_matrix(labels, preds,
                                              labels=sorted(set(labels))).tolist(),
        "label_order": sorted(set(labels)),
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\nTraining intent router ...")
    router = train_router()

    print("\nEvaluating on validation set ...")
    results = evaluate_router(router, split="validation")
    print(f"  Accuracy: {results['report']['accuracy']:.3f}")

    # Save eval results
    out = ROOT / "reports" / "router_eval.json"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"  Eval results saved to {out}")
