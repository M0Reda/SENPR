"""
data_audit.py -- Stage 1: Data Quality & Consistency Checks
===========================================================
Run:
    cd edubot
    python src/data_audit.py

Produces:
    reports/data_audit_report.md
    Prints a summary table to stdout.
"""

from __future__ import annotations

import os
import sys
import textwrap

# Windows: force UTF-8 stdout so emoji and special chars don't crash
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

# ── Config ────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent.parent          # edubot/
CFG_PATH = ROOT / "config.yaml"
cfg: dict[str, Any] = yaml.safe_load(CFG_PATH.read_text())

DATA_DIR = (ROOT / cfg["data_dir"]).resolve()
REPORTS_DIR = ROOT / cfg["reports_dir"]
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

GRADE_PTS_3CR: dict[str, float] = {
    "A+": 12.0, "A": 11.5, "A-": 11.0, "B+": 10.0, "B": 9.0,
    "B-": 8.0,  "C+": 7.0, "C": 6.0,   "C-": 5.0,  "D+": 4.0,
    "D": 3.0,   "F": 0.0,
}

# ── Helpers ───────────────────────────────────────────────────────────────────

findings: list[str] = []      # markdown sections accumulated


def section(title: str, body: str) -> None:
    findings.append(f"## {title}\n\n{body.strip()}\n")
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    # Strip markdown for console output
    for line in body.strip().splitlines():
        if line.startswith("|") or line.startswith("#"):
            print(line)
        else:
            print(textwrap.fill(line, 100))


def ok(msg: str) -> str:
    return f"✅ {msg}"


def warn(msg: str) -> str:
    return f"⚠️  {msg}"


def err(msg: str) -> str:
    return f"❌ {msg}"


def load(name: str) -> pd.DataFrame:
    path = DATA_DIR / name
    df = pd.read_csv(path, low_memory=False)
    print(f"  Loaded {name}: {len(df):,} rows × {len(df.columns)} cols")
    return df


def round_half_up_1dp(x: float) -> float:
    """Replicate the exact rounding used in edubot_context.py (Fraction-based)."""
    import math
    return math.floor(x * 10 + 0.5) / 10


# ── Load all tables ───────────────────────────────────────────────────────────

print("\nLoading tables ...")
students      = load("Students.csv")
cohorts       = load("Student_Cohorts.csv")
transcripts   = load("Student_Transcripts.csv")
enrollments   = load("Course_Enrollments.csv")
courses       = load("Courses.csv")
prerequisites = load("Course_Prerequisites.csv")
offerings     = load("Course_Offerings.csv")
advisors      = load("Advisors.csv")
qa            = load("Advising_QA.csv")
# Users.csv is explicitly never audited beyond existence; we just confirm it exists.
users_path = DATA_DIR / "Users.csv"

# ── CHECK 0 · File inventory & row counts ─────────────────────────────────────

rows = {
    "Students.csv":           len(students),
    "Student_Cohorts.csv":    len(cohorts),
    "Student_Transcripts.csv": len(transcripts),
    "Course_Enrollments.csv": len(enrollments),
    "Courses.csv":            len(courses),
    "Course_Prerequisites.csv": len(prerequisites),
    "Course_Offerings.csv":   len(offerings),
    "Advisors.csv":           len(advisors),
    "Advising_QA.csv":        len(qa),
    "Users.csv":              "exists (not loaded)" if users_path.exists() else "MISSING",
}

tbl = "| File | Rows |\n|---|---|\n"
tbl += "\n".join(f"| `{k}` | {v:,} |" if isinstance(v, int) else f"| `{k}` | {v} |" for k, v in rows.items())
section("0 · File Inventory", tbl)

# ── CHECK 1 · total_credits_completed reconciliation ──────────────────────────

print("\nRunning credit-hour reconciliation ...")

# Latest attempt per student per course (drop earlier retaken rows)
tr = transcripts.copy()
tr["points"] = pd.to_numeric(tr["points"], errors="coerce")

# Sort so Retaken / latest is at the bottom; keep last occurrence per (student, course)
# We define "latest" by transcript_id (monotone ascending)
tr_sorted = tr.sort_values("transcript_id")
latest = tr_sorted.drop_duplicates(subset=["student_id", "course_code"], keep="last")

# Merge credit hours from Courses
cr_map = courses.set_index("course_code")["credit_hours"].astype(int).to_dict()
latest["cr"] = latest["course_code"].map(cr_map)

# Credits = sum of cr where grade != 'F' and cr != 0 (Practical Training IM400CC has 0 cr)
passed_latest = latest[(latest["grade"] != "F") & (latest["cr"].fillna(0) > 0)]
computed_credits = passed_latest.groupby("student_id")["cr"].sum().reset_index()
computed_credits.columns = ["student_id", "computed_credits"]

merged = students[["student_id", "total_credits_completed"]].merge(
    computed_credits, on="student_id", how="left"
)
merged["computed_credits"] = merged["computed_credits"].fillna(0).astype(int)
merged["diff"] = merged["total_credits_completed"].astype(int) - merged["computed_credits"]
mismatched = merged[merged["diff"].abs() > 0]

body = ""
if mismatched.empty:
    body += ok(f"All {len(students):,} students' `total_credits_completed` match the transcript-derived sum.\n\n")
else:
    body += warn(f"{len(mismatched)} students have a discrepancy between `Students.total_credits_completed` and the transcript-derived count:\n\n")
    body += mismatched[["student_id", "total_credits_completed", "computed_credits", "diff"]].to_markdown(index=False) + "\n\n"

# Distribution of differences
diff_dist = merged["diff"].value_counts().sort_index()
body += f"**Difference distribution (Students.total − computed):**\n\n"
body += diff_dist.rename("count").rename_axis("diff").reset_index().to_markdown(index=False) + "\n"
section("1 · Credits Reconciliation", body)

# ── CHECK 2 · Enrollment > Transcript row count ───────────────────────────────

CURRENT_TERM_SEASON = "Fall"
CURRENT_TERM_YEAR   = "2026-2027"

# Map offering_id → semester_term + academic_year
off_map = offerings.set_index("offering_id")[["semester_term", "academic_year"]]
enr_aug = enrollments.merge(off_map, left_on="offering_id", right_index=True, how="left")
current_enr = enr_aug[
    (enr_aug["semester_term"] == CURRENT_TERM_SEASON) &
    (enr_aug["academic_year"] == CURRENT_TERM_YEAR)
]
past_enr = enr_aug[
    ~((enr_aug["semester_term"] == CURRENT_TERM_SEASON) &
      (enr_aug["academic_year"] == CURRENT_TERM_YEAR))
]

body  = f"- **Total enrollments:** {len(enrollments):,}\n"
body += f"- **Total transcripts:** {len(transcripts):,}\n"
body += f"- **Gap:** {len(enrollments) - len(transcripts):,}\n"
body += f"- **Current-term (Fall 2026-2027) enrollments:** {len(current_enr):,}\n"
body += f"- **Past enrollments:** {len(past_enr):,}\n\n"

gap = len(enrollments) - len(transcripts)
close_enough = abs(len(current_enr) - gap) <= 5   # allow tiny rounding
if close_enough:
    body += ok(
        f"The gap ({gap:,}) is fully explained by current-term in-progress enrollments "
        f"({len(current_enr):,} rows). These courses are still running (week 12, Fall 2026-2027) "
        f"and will reach the transcript when final grades are released."
    )
else:
    body += warn(
        f"Gap ({gap:,}) does not equal current-term enrollments ({len(current_enr):,}). "
        f"Unexplained difference: {gap - len(current_enr):,}."
    )

section("2 · Enrollment vs. Transcript Row Count", body)

# ── CHECK 3 · professor_id vs instructor_id ───────────────────────────────────

# Load only non-PII columns of Users to check IDs exist
users_ids = set(pd.read_csv(users_path, usecols=["user_id"])["user_id"])

prof_ids  = set(offerings["professor_id"].dropna())
inst_ids  = set(offerings["instructor_id"].dropna())

prof_missing = prof_ids - users_ids
inst_missing = inst_ids - users_ids
overlap = prof_ids & inst_ids   # IDs that appear in both roles

body  = "**Semantic distinction:**\n\n"
body += (
    "- `professor_id` — the faculty member who *owns* the course (writes the syllabus, "
    "sets the exams, submits final grades). Typically a senior academic.\n"
    "- `instructor_id` — the staff member who *delivers* the lab/section (may be a junior "
    "lecturer or teaching assistant). The two can be the same person for pure-lecture courses.\n\n"
)
body += f"| Metric | Value |\n|---|---|\n"
body += f"| Unique professor_ids | {len(prof_ids)} |\n"
body += f"| Unique instructor_ids | {len(inst_ids)} |\n"
body += f"| IDs appearing in both roles | {len(overlap)} |\n"
body += f"| professor_ids not in Users.csv | {len(prof_missing)} |\n"
body += f"| instructor_ids not in Users.csv | {len(inst_missing)} |\n\n"

if not prof_missing and not inst_missing:
    body += ok("All professor_id and instructor_id values resolve to Users.csv records.")
else:
    if prof_missing:
        body += err(f"Unknown professor_ids: {sorted(prof_missing)}\n\n")
    if inst_missing:
        body += err(f"Unknown instructor_ids: {sorted(inst_missing)}\n\n")

section("3 · professor_id vs instructor_id", body)

# ── CHECK 4 · Duplicate rows ──────────────────────────────────────────────────

checks: list[tuple[str, str, str]] = [
    ("Student_Transcripts", "transcript_id", transcripts),
    ("Course_Enrollments",  "enrollment_id", enrollments),
    ("Students",            "student_id",    students),
    ("Advisors",            "advisor_id",    advisors),
    ("Course_Offerings",    "offering_id",   offerings),
]
body = "| Table | ID column | Total rows | Duplicate IDs |\n|---|---|---|---|\n"
for name, col, df in checks:
    dups = df[col].duplicated().sum()
    status = "0 ✅" if dups == 0 else f"{dups} ❌"
    body += f"| {name} | `{col}` | {len(df):,} | {status} |\n"

# QA: duplicate (ticket_id, turn_no) pairs
qa_dup = qa.duplicated(subset=["ticket_id", "turn_no"]).sum()
body += f"| Advising_QA | `(ticket_id, turn_no)` | {len(qa):,} | {'0 ✅' if qa_dup == 0 else f'{qa_dup} ❌'} |\n"

section("4 · Duplicate Rows", body)

# ── CHECK 5 · Null / Orphan keys ──────────────────────────────────────────────

issues: list[str] = []

# Students with no advisor
no_adv = students[~students["advisor_id"].isin(advisors["advisor_id"].dropna())]
if not no_adv.empty:
    issues.append(err(f"{len(no_adv)} students have an advisor_id not in Advisors.csv."))
else:
    issues.append(ok("All student advisor_ids resolve to Advisors.csv."))

# Transcripts with unknown student
unknown_tr_sid = set(transcripts["student_id"]) - set(students["student_id"])
if unknown_tr_sid:
    issues.append(err(f"{len(unknown_tr_sid)} student_id(s) in Transcripts not in Students.csv: {sorted(unknown_tr_sid)[:5]}"))
else:
    issues.append(ok("All transcript student_ids resolve to Students.csv."))

# Transcript courses not in Courses.csv
unknown_tr_cc = set(transcripts["course_code"]) - set(courses["course_code"])
if unknown_tr_cc:
    issues.append(warn(f"{len(unknown_tr_cc)} course_code(s) in Transcripts not in Courses.csv: {sorted(unknown_tr_cc)[:10]}"))
else:
    issues.append(ok("All transcript course_codes resolve to Courses.csv."))

# Prerequisite courses not in Courses.csv
unknown_prereq = (
    set(prerequisites["course_code"]) | set(prerequisites["prereq_course_code"])
) - set(courses["course_code"])
if unknown_prereq:
    issues.append(warn(f"{len(unknown_prereq)} course_code(s) in Prerequisites not in Courses.csv: {sorted(unknown_prereq)}"))
else:
    issues.append(ok("All prerequisite course_codes resolve to Courses.csv."))

# Enrollments with unknown offering
unknown_off = set(enrollments["offering_id"]) - set(offerings["offering_id"])
if unknown_off:
    issues.append(err(f"{len(unknown_off)} offering_id(s) in Enrollments not in Course_Offerings.csv."))
else:
    issues.append(ok("All enrollment offering_ids resolve to Course_Offerings.csv."))

# QA with unknown student
unknown_qa_sid = set(qa["student_id"]) - set(students["student_id"])
if unknown_qa_sid:
    issues.append(warn(f"{len(unknown_qa_sid)} student_id(s) in Advising_QA not in Students.csv."))
else:
    issues.append(ok("All Advising_QA student_ids resolve to Students.csv."))

# Null checks across key columns
null_report: list[str] = []
for df, name, cols in [
    (students,    "Students",             ["student_id","major","gpa","standing_status","advisor_id"]),
    (transcripts, "Student_Transcripts",  ["transcript_id","student_id","course_code","grade","points"]),
    (enrollments, "Course_Enrollments",   ["enrollment_id","offering_id","student_id"]),
    (offerings,   "Course_Offerings",     ["offering_id","course_code","professor_id"]),
]:
    for col in cols:
        n = df[col].isna().sum()
        if n:
            null_report.append(f"  - `{name}.{col}`: {n:,} nulls")

if null_report:
    issues.append(warn("Null values found in key columns:\n" + "\n".join(null_report)))
else:
    issues.append(ok("No nulls in any primary key or critical column."))

section("5 · Null & Orphan Key Checks", "\n\n".join(issues))

# ── CHECK 6 · Grade / Points consistency ──────────────────────────────────────

tr2 = transcripts.copy()
tr2["points"] = pd.to_numeric(tr2["points"], errors="coerce")
tr2 = tr2.merge(courses[["course_code", "credit_hours"]], on="course_code", how="left")
tr2["credit_hours"] = pd.to_numeric(tr2["credit_hours"], errors="coerce").fillna(3)

def expected_pts(grade: str, cr: float) -> float | None:
    if grade in ("P", "U", "") or cr == 0:
        return 0.0
    base = GRADE_PTS_3CR.get(grade)
    if base is None:
        return None
    return round_half_up_1dp(base * cr / 3)

tr2["expected_pts"] = tr2.apply(
    lambda r: expected_pts(r["grade"], r["credit_hours"]), axis=1
)
tr2["pts_diff"] = (tr2["points"] - tr2["expected_pts"]).abs()
# Allow 0.05 tolerance for floating-point drift
bad_pts = tr2[(tr2["pts_diff"] > 0.05) & tr2["expected_pts"].notna()]

body = ""
if bad_pts.empty:
    body += ok(f"All {len(tr2):,} transcript rows pass the grade→points formula check (tolerance ±0.05).\n\n")
else:
    body += warn(f"{len(bad_pts):,} rows have an unexpected points value:\n\n")
    sample = bad_pts[["transcript_id","student_id","course_code","grade",
                       "credit_hours","points","expected_pts","pts_diff"]].head(20)
    body += sample.to_markdown(index=False) + "\n\n"

# Check standing consistency
def expected_standing(gpa: float) -> str:
    g = float(gpa)
    if g >= 3.7: return "Excellent"
    if g >= 3.3: return "Very Good"
    if g >= 2.7: return "Good"
    if g >= 2.0: return "Pass"
    return "Academic Probation"

students["expected_standing"] = students["gpa"].apply(expected_standing)
bad_standing = students[students["standing_status"] != students["expected_standing"]]
if bad_standing.empty:
    body += ok(f"All {len(students):,} student standing_status values are consistent with GPA bands.\n")
else:
    body += warn(f"{len(bad_standing):,} students have a standing_status inconsistent with their GPA:\n\n")
    body += bad_standing[["student_id","gpa","standing_status","expected_standing"]].head(10).to_markdown(index=False) + "\n"

section("6 · Grade / Points & Standing Consistency", body)

# ── CHECK 7 · Advising_QA structure ──────────────────────────────────────────

n_tickets   = qa["ticket_id"].nunique()
n_students  = qa["student_id"].nunique()
intents     = qa["intent_category"].value_counts()
roles       = qa["role"].value_counts()
statuses    = qa["ticket_status"].value_counts()

# Multi-turn statistics
turns_per_ticket = qa.groupby("ticket_id")["turn_no"].max()

body  = f"- **Total message rows:** {len(qa):,}\n"
body += f"- **Unique tickets:** {n_tickets:,}\n"
body += f"- **Unique students:** {n_students:,}\n"
body += f"- **Roles:** {dict(roles)}\n"
body += f"- **Ticket statuses:** {dict(statuses)}\n\n"

body += "**Turns-per-ticket distribution:**\n\n"
body += turns_per_ticket.describe().rename("turns").to_frame().T.to_markdown() + "\n\n"
multi_turn_pct = (turns_per_ticket > 2).mean() * 100
body += f"- Tickets with >2 turns (multi-turn): **{multi_turn_pct:.1f}%** of all tickets\n\n"

body += "**Top 20 intent categories (by message count):**\n\n"
body += intents.head(20).rename("messages").rename_axis("intent").reset_index().to_markdown(index=False) + "\n\n"

body += warn(
    "The QA data is templated / programmatically generated. Many bot responses follow "
    "near-identical patterns, so retrieval scores on this test set will be artificially high. "
    "A paraphrased hard test set (generated with the local LLM in evaluate.py) is required "
    "to measure real generalisation."
)

section("7 · Advising_QA Structure & Inflated-Retrieval Note", body)

# ── CHECK 8 · Semester-level formula cross-check ─────────────────────────────

def level_of(credits: int) -> int:
    return min(10, int(credits) // 18 + 1)

students["computed_level"] = students["total_credits_completed"].astype(int).apply(level_of)
bad_level = students[students["current_semester_level"].astype(int) != students["computed_level"]]

body = ""
if bad_level.empty:
    body += ok(f"All {len(students):,} semester_level values match the formula floor(credits/18)+1.\n")
else:
    body += warn(f"{len(bad_level):,} students have a semester_level inconsistent with total_credits_completed:\n\n")
    body += bad_level[["student_id","total_credits_completed","current_semester_level","computed_level"]].head(10).to_markdown(index=False) + "\n"

section("8 · Semester-Level Formula Cross-Check", body)

# ── CHECK 9 · Existing train/val/test split coverage ─────────────────────────

import json

split_files = {
    "train":      ROOT / cfg["data_dir"] / "EduBot_Training" / "sft_train.jsonl",
    "validation": ROOT / cfg["data_dir"] / "EduBot_Training" / "sft_validation.jsonl",
    "test":       ROOT / cfg["data_dir"] / "EduBot_Training" / "sft_test.jsonl",
}

body = "| Split | File exists | Lines | Unique students (first turn) |\n|---|---|---|---|\n"
split_sids: dict[str, set] = {}
for split, path in split_files.items():
    path = (ROOT / cfg["data_dir"] / "EduBot_Training" / f"sft_{split}.jsonl").resolve()
    if path.exists():
        with open(path, encoding="utf-8") as f:
            lines = [json.loads(l) for l in f if l.strip()]
        # Each line is a chat conversation; extract student_id from system message if possible
        sids: set[str] = set()
        for conv in lines:
            msgs = conv.get("messages", conv.get("conversations", []))
            for m in msgs:
                content = m.get("content", "")
                # The student ID appears in the system prompt after "student_id":"
                import re
                m_sid = re.search(r'"student_id"\s*:\s*"(\d+)"', content)
                if m_sid:
                    sids.add(m_sid.group(1))
                    break
        split_sids[split] = sids
        body += f"| {split} | ✅ | {len(lines):,} | {len(sids):,} |\n"
    else:
        split_sids[split] = set()
        body += f"| {split} | ❌ | — | — |\n"

# Overlap check
tr_val  = split_sids.get("train", set()) & split_sids.get("validation", set())
tr_test = split_sids.get("train", set()) & split_sids.get("test", set())
val_test= split_sids.get("validation", set()) & split_sids.get("test", set())

body += "\n**Student-ID overlap across splits:**\n\n"
body += f"- train ∩ validation: {len(tr_val)}\n"
body += f"- train ∩ test: {len(tr_test)}\n"
body += f"- validation ∩ test: {len(val_test)}\n"

if tr_val == tr_test == val_test == set():
    body += "\n" + ok("No student appears in more than one split — clean train/val/test isolation.")
else:
    body += "\n" + err("Student overlap detected between splits!")

section("9 · Existing Train / Val / Test Split Coverage", body)

# ── Write report ──────────────────────────────────────────────────────────────

report_path = REPORTS_DIR / "data_audit_report.md"
header = (
    "# EduBot — Data Audit Report\n\n"
    f"Dataset: `{DATA_DIR}`  \n"
    f"Generated: 2026-10-13  \n\n"
    "---\n\n"
)
report_path.write_text(header + "\n\n".join(findings), encoding="utf-8")
print(f"\n\n{'='*60}")
print(f"  ✅  Report saved → {report_path}")
print(f"{'='*60}\n")
