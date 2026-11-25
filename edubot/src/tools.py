"""
tools.py -- Stage 3: Deterministic Pandas Tools
================================================
All student-specific lookups and calculations.

Every tool is a pure function that accepts a session_sid (the
authenticated student) and raises PermissionError if the caller
tries to fetch another student's data.

Usage:
    from src.tools import StudentTools
    t = StudentTools(db, session_sid="201003465")
    profile = t.get_student_profile()
    gpa     = t.calculate_gpa()
"""

from __future__ import annotations

import math
import sys
from fractions import Fraction as Fr
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

# ---------------------------------------------------------------------------
# Config + DB import
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT.parent / "EduBot_Dataset_v1.2" / "scripts"))

# We import the authoritative context builder from the dataset's own scripts.
# It already contains the exact GPA/points/standing formulas used to generate
# the training data, so training and production see identical computations.
try:
    from edubot_context import EduBotDB, SYSTEM_PROMPT, rnd1, pts, standing_of, term_key, level_of
except ModuleNotFoundError:
    # Fallback: add the scripts folder dynamically from config
    cfg: dict[str, Any] = yaml.safe_load((ROOT / "config.yaml").read_text())
    DATA_DIR = (ROOT / cfg["data_dir"]).resolve()
    sys.path.insert(0, str(DATA_DIR / "scripts"))
    from edubot_context import EduBotDB, SYSTEM_PROMPT, rnd1, pts, standing_of, term_key, level_of


CURRENT_TERM = "First Semester / 2026-2027"
TOTAL_CREDITS = 180
FINAL_PASS_SCORE = 12          # minimum final-exam score out of 40 to pass
COURSE_PASS_TOTAL = 60         # minimum total score out of 100

GRADE_ORDER = ["F", "D", "D+", "C-", "C", "C+", "B-", "B", "B+", "A-", "A", "A+"]

# ── Helpers ─────────────────────────────────────────────────────────────────

def _rnd1(x: float) -> float:
    """Round-half-up to 1 decimal (matches edubot_context.rnd1)."""
    return float(math.floor(x * 10 + 0.5) / 10)


# ── Main tools class ─────────────────────────────────────────────────────────

class StudentTools:
    """
    Deterministic tools scoped to a single authenticated student.

    Parameters
    ----------
    db : EduBotDB
        Loaded database instance (shared across the session).
    session_sid : str
        The student ID that was authenticated at session start.
    """

    def __init__(self, db: "EduBotDB", session_sid: str) -> None:
        self._db = db
        self._sid = str(session_sid)
        # Pre-compute the full record once; cache it for the session
        self._record: dict[str, Any] | None = None

    # ── Auth guard ────────────────────────────────────────────────────────────

    def _check(self, sid: str | None = None) -> str:
        """Return the session sid, raising PermissionError if sid != session."""
        target = str(sid) if sid is not None else self._sid
        if target != self._sid:
            raise PermissionError(
                f"Access denied: student {target} is not the authenticated user "
                f"({self._sid}). Each session may only query its own data."
            )
        return self._sid

    def _rec(self) -> dict[str, Any]:
        """Return the full EduBotDB record, memoised for the session."""
        if self._record is None:
            self._record = self._db.record(self._sid)
        return self._record

    # ── Tool 1: get_student_profile ───────────────────────────────────────────

    def get_student_profile(self) -> dict[str, Any]:
        """
        Return the student's academic profile (no PII beyond name).

        Returns
        -------
        dict with keys: student_id, name, semester_level, enrollment_phase,
            credits_completed, credits_remaining, cumulative_gpa, standing,
            consecutive_probation_warnings, max_load_hours, load_reason,
            registered_this_term_hours.
        """
        self._check()
        r = self._rec()
        s = r["student"]
        reg = r["registration_rules_now"]
        return {
            "student_id":                    s["student_id"],
            "name":                          s["name"],
            "semester_level":                s["semester_level"],
            "enrollment_phase":              s["enrollment_phase"],
            "credits_completed":             s["credits_completed"],
            "credits_remaining":             s["credits_remaining"],
            "cumulative_gpa":                s["cumulative_gpa"],
            "standing":                      s["standing"],
            "consecutive_probation_warnings": s["consecutive_probation_warnings"],
            "max_load_hours":                reg["max_credit_hours_fall_spring"],
            "load_reason":                   reg["reason"],
            "registered_this_term_hours":    reg["registered_this_term"],
        }

    # ── Tool 2: get_transcript ────────────────────────────────────────────────

    def get_transcript(self) -> list[str]:
        """
        Return the student's full transcript as human-readable lines.

        Each line: "<term> | <code> <title> | <cr> cr | <grade> | <points> pts | <status>"
        Failed rows include mark details where available.
        """
        self._check()
        return self._rec()["transcript"]

    # ── Tool 3: get_current_marks ─────────────────────────────────────────────

    def get_current_marks(self) -> list[dict[str, Any]]:
        """
        Return current-term course marks (7th-week + 12th-week so far).

        Returns a list of dicts: code, title, credit_hours, attempt,
        seventh_week, twelfth_week, marks_so_far, instructor_feedback.
        """
        self._check()
        return self._rec()["current_courses"]

    # ── Tool 4: calculate_gpa ─────────────────────────────────────────────────

    def calculate_gpa(self, scenario: dict[str, str] | None = None) -> dict[str, Any]:
        """
        Calculate (or project) cumulative and semester GPA.

        Parameters
        ----------
        scenario : dict[course_code -> hypothetical_grade], optional
            If provided, overlay these grades on top of existing current-course
            marks to project what the GPA *would* be.

        Returns
        -------
        dict with: cumulative_gpa, cumulative_points, cumulative_credit_hours,
            standing, semester_gpa (current term), scenario_used.
        """
        self._check()
        r   = self._rec()
        internal = r["_internal"]
        P   = Fr(internal["P"])
        A   = Fr(internal["A"])
        gpa = internal["gpa"]

        if scenario is None:
            return {
                "cumulative_gpa":            str(gpa) if gpa else "N/A",
                "cumulative_gpa_rounded":    float(gpa) if gpa else None,
                "cumulative_points":         r["student"]["cumulative_points"],
                "cumulative_credit_hours":   A,
                "standing":                  r["student"]["standing"],
                "scenario_used":             None,
            }

        # Apply hypothetical grades
        cur_courses = self._rec()["current_courses"]
        added_pts = Fr(0)
        added_cr  = 0
        breakdown: list[str] = []
        for item in cur_courses:
            code = item["code"]
            if code not in scenario:
                continue
            grade = scenario[code]
            cr    = item["credit_hours"]
            p     = pts(grade, cr)
            added_pts += p
            # Only add credit hours for courses not already passed (retry doesn't add cr)
            latest = internal["latest"].get(code)
            if latest is None or latest["grade"] == "F":
                added_cr += cr
            breakdown.append(f"{code}: {grade} → {float(p):.1f} pts, {cr} cr")

        new_P = P + added_pts
        new_A = A + added_cr
        new_gpa = rnd1(new_P / new_A) if new_A else Fr(0)
        sem_pts  = added_pts
        sem_cr   = added_cr + sum(
            item["credit_hours"] for item in cur_courses
            if item["code"] not in scenario
        )
        sem_gpa = rnd1(sem_pts / sem_cr) if sem_cr else Fr(0)

        return {
            "cumulative_gpa":           float(new_gpa),
            "cumulative_points":        float(new_P),
            "cumulative_credit_hours":  int(new_A),
            "standing":                 standing_of(new_gpa),
            "semester_gpa_projection":  float(sem_gpa),
            "breakdown":                breakdown,
            "scenario_used":            scenario,
        }

    # ── Tool 5: check_standing ────────────────────────────────────────────────

    def check_standing(self) -> dict[str, Any]:
        """Return current standing, GPA bands and probation warning count."""
        self._check()
        s = self._rec()["student"]
        return {
            "standing":                      s["standing"],
            "cumulative_gpa":                s["cumulative_gpa"],
            "consecutive_probation_warnings": s["consecutive_probation_warnings"],
            "bands": {
                "Excellent":           "3.7 – 4.0",
                "Very Good":           "3.3 – 3.6",
                "Good":                "2.7 – 3.2",
                "Pass":                "2.0 – 2.6",
                "Academic Probation":  "< 2.0",
            },
        }

    # ── Tool 6: check_prereqs ─────────────────────────────────────────────────

    def check_prereqs(self, course_code: str) -> dict[str, Any]:
        """
        Recursively check whether the student meets all prerequisites.

        Returns
        -------
        dict with: course_code, title, direct_prereqs, all_prereqs_met,
            met[], unmet[].
        """
        self._check()
        db = self._db
        passed = self._rec()["_internal"]["passed"]
        cur    = self._rec()["_internal"]["current_codes"]
        available = passed | cur  # prereq satisfied if passed OR currently enrolled

        def resolve(code: str, depth: int = 0, seen: set[str] | None = None) -> list[dict]:
            if seen is None:
                seen = set()
            if code in seen or depth > 10:
                return []
            seen.add(code)
            result = []
            for prereq in db.prereq.get(code, []):
                is_met = prereq in available
                result.append({
                    "prereq": prereq,
                    "title":  db.courses.get(prereq, {}).get("course_title", "?"),
                    "met":    is_met,
                    "depth":  depth,
                })
                result.extend(resolve(prereq, depth + 1, seen))
            return result

        if course_code not in db.courses:
            return {"error": f"Course {course_code} not found in course catalogue."}

        chain = resolve(course_code)
        met   = [r for r in chain if r["met"]]
        unmet = [r for r in chain if not r["met"]]

        return {
            "course_code":       course_code,
            "title":             db.courses[course_code]["course_title"],
            "direct_prereqs":    db.prereq.get(course_code, []),
            "full_prereq_chain": chain,
            "all_prereqs_met":   len(unmet) == 0,
            "met":               [r["prereq"] for r in met],
            "unmet":             [r["prereq"] for r in unmet],
        }

    # ── Tool 7: get_offerings ─────────────────────────────────────────────────

    def get_offerings(self, course_code: str, semester: str | None = None) -> list[dict]:
        """
        Return recent offerings for a course.

        Parameters
        ----------
        course_code : str
        semester    : str | None  e.g. "First Semester / 2026-2027"
        """
        # No student-scope restriction — course catalogue is public
        db = self._db
        results = []
        for oid, o in db.offerings.items():
            if o["course_code"] != course_code:
                continue
            label = db.off_label(o)
            if semester and label != semester:
                continue
            results.append({
                "offering_id":   oid,
                "semester":      label,
                "professor_id":  o["professor_id"],
                "instructor_id": o["instructor_id"],
            })
        return sorted(results, key=lambda x: x["semester"], reverse=True)

    # ── Tool 8: get_advisor ───────────────────────────────────────────────────

    def get_advisor(self) -> dict[str, Any]:
        """Return the student's advisor name and office room (never email)."""
        self._check()
        r   = self._rec()
        adv = r["advisor"]
        return {
            "advisor_id": adv["advisor_id"],
            "name":       adv["name"],
            "office":     adv["office"],
            # Email deliberately excluded per privacy policy
        }

    # ── Tool 9: remaining_required_courses ────────────────────────────────────

    def remaining_required_courses(self) -> dict[str, Any]:
        """Return all required courses not yet passed and elective status."""
        self._check()
        return self._rec()["remaining"]

    # ── Tool 10: calculate_required_final ─────────────────────────────────────

    def calculate_required_final(self, course_code: str) -> dict[str, Any]:
        """
        Calculate the minimum final-exam score needed to pass a course.

        Considers the student's current 7th-week + 12th-week marks and
        hypothetical pre-final scores (3 scenarios: 10, 7, 5 out of 10).

        Returns
        -------
        dict with: course_code, marks_so_far, max_coursework, scenarios[],
            note (grade cap if retaking a failed course).
        """
        self._check()
        cur = {c["code"]: c for c in self._rec()["current_courses"]}
        if course_code not in cur:
            return {"error": f"{course_code} is not in your current-term courses."}

        item = cur[course_code]
        w7  = float(item.get("seventh_week", 0))
        w12 = float(item.get("twelfth_week", 0))
        pre_max = 10
        marks_so_far = w7 + w12  # out of 50

        scenarios = []
        for pre_final in [pre_max, 7, 5]:
            coursework = marks_so_far + pre_final   # out of 60
            needed_final = COURSE_PASS_TOTAL - coursework  # needed out of 40
            needed_final = max(needed_final, FINAL_PASS_SCORE)  # floor = 12
            feasible = needed_final <= 40
            scenarios.append({
                "pre_final_score":   pre_final,
                "total_before_exam": coursework,
                "final_needed":      round(needed_final, 1),
                "feasible":          feasible,
            })

        return {
            "course_code":    course_code,
            "course_title":   item.get("title", ""),
            "seventh_week":   w7,
            "twelfth_week":   w12,
            "marks_so_far":   f"{marks_so_far:.1f}/50",
            "attempt":        item.get("attempt", ""),
            "scenarios":      scenarios,
            "note": (
                "Grade is capped at B+ (this is a repeat of a failed course)."
                if "repeat of a failed" in item.get("attempt", "")
                else None
            ),
        }

    # ── Tool 11: check_overload_eligibility ───────────────────────────────────

    def check_overload_eligibility(self) -> dict[str, Any]:
        """Return the load cap and the rule that determines it."""
        self._check()
        reg = self._rec()["registration_rules_now"]
        return {
            "max_credit_hours_fall_spring": reg["max_credit_hours_fall_spring"],
            "reason":                       reg["reason"],
            "registered_this_term":         reg["registered_this_term"],
            "overload_allowed": reg["max_credit_hours_fall_spring"] > 18,
        }

    # ── Tool 12: check_improvement_window ────────────────────────────────────

    def check_improvement_window(self, course_code: str) -> dict[str, Any]:
        """
        Check if the 3-regular-semester improvement-repeat window is still open.

        A student may retake a *passed* course to improve their grade only within
        3 Fall/Spring semesters of the term they first completed it.
        """
        self._check()
        internal = self._rec()["_internal"]
        latest   = internal["latest"]

        if course_code not in latest:
            return {"error": f"{course_code} not found in your transcript."}

        row = latest[course_code]
        if row["grade"] == "F":
            return {
                "course_code":      course_code,
                "applicable":       False,
                "reason":           "Course was failed — must be retaken (not an improvement repeat).",
            }

        completed_in = row["semester_taken"]
        # Count regular (Fall / Spring) semesters from completion to now
        from edubot_context import regular_semesters_between
        semesters_elapsed = regular_semesters_between(completed_in, CURRENT_TERM)
        window_open = semesters_elapsed <= 3

        return {
            "course_code":        course_code,
            "course_title":       self._db.courses[course_code]["course_title"],
            "grade":              row["grade"],
            "completed_in":       completed_in,
            "semesters_elapsed":  semesters_elapsed,
            "window_open":        window_open,
            "reason": (
                f"Completed {semesters_elapsed} regular semester(s) ago; window closes at 3."
                if window_open
                else f"Window closed ({semesters_elapsed} semesters since completion; max 3)."
            ),
        }

    # ── Tool 13: get_graduation_plan ─────────────────────────────────────────

    def get_graduation_plan(self) -> dict[str, Any]:
        """
        Return the greedy term-by-term graduation plan projection.

        Assumes all current-term courses are passed.
        """
        self._check()
        return self._rec()["graduation_plan_projection"]

    def find_courses_in_message(self, message: str) -> list[str]:
        """Extract course codes mentioned by code OR by course title."""
        import re
        found: list[str] = re.findall(r"\b([A-Z]{2,3}\d{3,})\b", message.upper())
        msg_low = f" {re.sub(r'[^a-z0-9]+', ' ', message.lower())} "
        # Longest titles first so 'Computer Architecture' matches before shorter substrings
        for code, info in sorted(
            self._db.courses.items(),
            key=lambda kv: len(kv[1]["course_title"]),
            reverse=True,
        ):
            title_norm = re.sub(r"[^a-z0-9]+", " ", info["course_title"].lower()).strip()
            if len(title_norm) >= 5 and f" {title_norm} " in msg_low:
                if code not in found:
                    found.append(code)
        return [c for c in dict.fromkeys(found) if c in self._db.courses]

    def course_status(self, course_code: str) -> dict[str, Any]:
        """Full deterministic status for a single course (taken, passed, prereqs, repeat)."""
        self._check()
        if course_code not in self._db.courses:
            return {"error": f"Course {course_code} not found."}
        internal = self._rec()["_internal"]
        latest = internal["latest"].get(course_code)
        cur_codes = internal["current_codes"]
        status: dict[str, Any] = {
            "course_code": course_code,
            "course_title": self._db.courses[course_code]["course_title"],
            "credit_hours": self._db.courses[course_code]["credit_hours"],
            "already_passed": course_code in internal["passed"],
            "currently_enrolled": course_code in cur_codes,
            "latest_transcript_attempt": (
                {
                    "semester": latest["semester_taken"],
                    "grade": latest["grade"],
                    "status": latest["status"],
                }
                if latest
                else None
            ),
            "prereqs": self.check_prereqs(course_code),
        }
        if course_code in cur_codes:
            status["required_final"] = self.calculate_required_final(course_code)
        win = self.check_improvement_window(course_code)
        if "error" not in win:
            status["improvement_window"] = win
        return status

    def grades_by_letter(self) -> dict[str, list[str]]:
        """Courses grouped by letter grade (latest attempt of each course only)."""
        self._check()
        latest = self._rec()["_internal"]["latest"]
        out: dict[str, list[str]] = {}
        for code, row in latest.items():
            g = row["grade"]
            out.setdefault(g, []).append(
                f"{code} {self._db.courses[code]['course_title']} ({row['semester_taken']})"
            )
        order = ["A+", "A", "A-", "B+", "B", "B-", "C+", "C", "C-", "D+", "D", "F", "P"]
        return {g: sorted(out[g]) for g in order if g in out}

    # ── Tool: graduation_status (deterministic summary) ──────────────────────

    def graduation_status(self) -> dict[str, Any]:
        """Exact, pre-computed answer to 'what do I have left to graduate?'."""
        self._check()
        r = self._rec()
        s = r["student"]
        rem = r["remaining"]
        passed = r["_internal"]["passed"]
        in_progress = [
            c for c in r["current_courses"] if c["code"] not in passed
        ]
        ip_credits = sum(c["credit_hours"] for c in in_progress)
        left = s["credits_remaining"]
        after = max(0, left - ip_credits)
        el_needed = rem["electives_still_needed"]
        required_left = rem["required_courses_not_taken"]
        out: dict[str, Any] = {
            "credits_completed": s["credits_completed"],
            "credits_remaining_now": left,
            "courses_in_progress_this_term": [
                f"{c['code']} {c['title']} ({c['credit_hours']} cr, {c['attempt']})"
                for c in in_progress
            ],
            "credits_in_progress": ip_credits,
            "credits_still_unscheduled_after_this_term": after,
            "required_courses_not_taken_and_not_in_progress": required_left,
            "electives_completed": rem["electives_completed"],
            "electives_in_progress": rem["electives_in_progress"],
            "electives_still_needed": el_needed,
            "outstanding_failed_courses": r["repeats"]["outstanding_failed_courses"],
            "cumulative_gpa": s["cumulative_gpa"],
            "gpa_requirement_met": bool(s["cumulative_gpa"]) and float(s["cumulative_gpa"]) >= 2.0,
            "graduation_plan_terms": r["graduation_plan_projection"]["terms"],
            "notes": [
                "max_credit_hours is only an upper LIMIT on registration, never the amount the student needs.",
            ],
        }
        if el_needed > 0:
            out["electives_eligible"] = rem["electives_eligible_after_this_term"]
        else:
            out["notes"].append("All 6 electives are done: do NOT suggest or list more electives.")
        if after == 0 and not required_left and el_needed == 0:
            out["conclusion"] = (
                "If the student passes the course(s) in progress "
                f"({', '.join(c['code'] for c in in_progress) or 'none'}), "
                "all degree requirements are met and the student graduates at the end of this term "
                "(also needs cumulative GPA >= 2.0)."
            )
        return out

    # ── Tool 14: get_honors_status ────────────────────────────────────────────

    def get_honors_status(self) -> dict[str, Any]:
        """
        Return honors eligibility status.

        Honors requires: semester GPA >= 3.3 in EVERY semester and no F ever.
        """
        self._check()
        h = self._rec()["honors_check"]
        eligible = len(h["f_grades"]) == 0 and len(h["semesters_below_3_3"]) == 0
        return {
            "eligible_so_far": eligible,
            "f_grades":        h["f_grades"],
            "semesters_below_3_3": h["semesters_below_3_3"],
            "rule": "Semester GPA >= 3.3 in every semester AND no F at any time.",
        }

    # ── Tool 15: get_semester_history ─────────────────────────────────────────

    def get_semester_history(self) -> list[dict[str, Any]]:
        """Return per-semester GPA history with cumulative totals."""
        self._check()
        return self._rec()["semester_history"]

    # ── Convenience: build system message for this student ───────────────────

    def build_system_message(
        self,
        strip_email: bool = True,
        include_sections: set[str] | None = None,
    ) -> str:
        """
        Build the full system prompt + student record JSON.

        Parameters
        ----------
        strip_email : bool
            If True (default), remove 'university_email' from the student block
            per the Users.csv privacy policy.
        include_sections : set[str] | None
            If provided, include only these top-level keys from the student
            record (indexed lookup) instead of dumping the entire record.
        """
        self._check()
        import json
        rec = self._db.public(self._rec())
        st = rec.get("student", {})
        if "cumulative_credit_hours" in st:
            # Counts every attempt (including failed ones) for GPA only; it is NOT
            # credits completed and misleads small models, so rename it.
            st["credit_hours_counted_in_gpa_only"] = st.pop("cumulative_credit_hours")
        rem = rec.get("remaining", {})
        if rem.get("electives_still_needed", 0) == 0:
            rem["electives_eligible_after_this_term"] = []
        if strip_email:
            if "student" in rec and "university_email" in rec["student"]:
                del rec["student"]["university_email"]
        if include_sections is not None:
            rec = {k: v for k, v in rec.items() if k in include_sections}
        return SYSTEM_PROMPT + "\n\nSTUDENT RECORD (JSON):\n" + json.dumps(
            rec, ensure_ascii=False, separators=(",", ":"), default=str
        )
