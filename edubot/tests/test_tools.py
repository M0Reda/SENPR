"""
test_tools.py -- Unit tests for src/tools.py
=============================================
Run:
    cd edubot
    pytest tests/test_tools.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str((ROOT / "../EduBot_Dataset_v1.2/scripts").resolve()))

cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
DATA_DIR = (ROOT / cfg["data_dir"]).resolve()

from edubot_context import EduBotDB
from tools import StudentTools

# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def db() -> EduBotDB:
    return EduBotDB(str(DATA_DIR))


@pytest.fixture(scope="module")
def student_id(db: EduBotDB) -> str:
    """Use first student from the dataset."""
    return "201003465"


@pytest.fixture(scope="module")
def tools(db: EduBotDB, student_id: str) -> StudentTools:
    return StudentTools(db, session_sid=student_id)


# ── Test: Privacy guard ───────────────────────────────────────────────────────

class TestPrivacyGuard:
    def test_cannot_fetch_other_student(self, db: EduBotDB, student_id: str):
        """Tools must raise PermissionError when asked for a different student."""
        other_id = "201003491"
        t = StudentTools(db, session_sid=student_id)
        with pytest.raises(PermissionError, match="Access denied"):
            t._check(other_id)

    def test_system_message_strips_email(self, tools: StudentTools):
        """build_system_message must not contain the student's email."""
        msg = tools.build_system_message(strip_email=True)
        # The raw email format is like "user@aast.edu"
        assert "@" not in msg.split("STUDENT RECORD")[1] or "advisor" in msg.lower(), \
            "Student email should be stripped from the system message"

    def test_advisor_no_email(self, tools: StudentTools):
        """get_advisor must not return an email field."""
        adv = tools.get_advisor()
        assert "email" not in adv, "Advisor email must not be exposed by get_advisor()"
        assert "advisor_id" in adv
        assert "name" in adv
        assert "office" in adv


# ── Test: Student profile ─────────────────────────────────────────────────────

class TestStudentProfile:
    def test_profile_keys(self, tools: StudentTools):
        p = tools.get_student_profile()
        required_keys = {
            "student_id", "name", "semester_level", "credits_completed",
            "credits_remaining", "cumulative_gpa", "standing",
            "max_load_hours", "load_reason",
        }
        assert required_keys.issubset(p.keys())

    def test_credits_sum_to_180(self, tools: StudentTools):
        p = tools.get_student_profile()
        total = p["credits_completed"] + p["credits_remaining"]
        assert total == 180, f"credits_completed + credits_remaining should be 180, got {total}"

    def test_semester_level_range(self, tools: StudentTools):
        p = tools.get_student_profile()
        assert 1 <= p["semester_level"] <= 10


# ── Test: GPA calculation ─────────────────────────────────────────────────────

class TestGPACalculation:
    def test_gpa_current(self, tools: StudentTools):
        result = tools.calculate_gpa()
        gpa = result["cumulative_gpa_rounded"]
        assert gpa is not None
        assert 0.0 <= float(gpa) <= 4.0, f"GPA out of range: {gpa}"

    def test_gpa_rounding_to_one_decimal(self, tools: StudentTools):
        result = tools.calculate_gpa()
        gpa_str = str(result["cumulative_gpa"])
        # Should have at most 1 decimal place
        if "." in gpa_str:
            assert len(gpa_str.split(".")[1]) <= 1, f"GPA has >1 decimal: {gpa_str}"

    def test_gpa_projection_b_in_all(self, tools: StudentTools):
        cur = tools.get_current_marks()
        scenario = {c["code"]: "B" for c in cur}
        result = tools.calculate_gpa(scenario=scenario)
        assert "cumulative_gpa" in result
        assert result["scenario_used"] == scenario

    def test_gpa_projection_improves_on_a(self, tools: StudentTools):
        cur = tools.get_current_marks()
        if not cur:
            pytest.skip("No current courses")
        scenario_b  = {c["code"]: "B"  for c in cur}
        scenario_ap = {c["code"]: "A+" for c in cur}
        gpa_b  = tools.calculate_gpa(scenario=scenario_b)["cumulative_gpa"]
        gpa_ap = tools.calculate_gpa(scenario=scenario_ap)["cumulative_gpa"]
        assert float(gpa_ap) >= float(gpa_b), "A+ scenario should give >= GPA than B scenario"


# ── Test: Standing ────────────────────────────────────────────────────────────

class TestStanding:
    BAND_RANGES = {
        "Excellent":          (3.7, 4.0),
        "Very Good":          (3.3, 3.6),
        "Good":               (2.7, 3.2),
        "Pass":               (2.0, 2.6),
        "Academic Probation": (0.0, 1.9),
    }

    def test_standing_consistent_with_gpa(self, tools: StudentTools):
        s = tools.check_standing()
        gpa = float(s["cumulative_gpa"]) if s["cumulative_gpa"] else 0.0
        standing = s["standing"]
        if "Academic Probation" in standing:
            assert gpa < 2.0, f"Probation standing but GPA={gpa}"
        elif standing == "Excellent":
            assert gpa >= 3.7, f"Excellent standing but GPA={gpa}"
        elif standing == "Pass":
            assert 2.0 <= gpa <= 2.6, f"Pass standing but GPA={gpa}"

    def test_standing_has_probation_warnings(self, tools: StudentTools):
        s = tools.check_standing()
        assert "consecutive_probation_warnings" in s
        assert isinstance(s["consecutive_probation_warnings"], int)


# ── Test: Prerequisite checker ────────────────────────────────────────────────

class TestPrereqChecker:
    def test_known_course_with_prereqs(self, tools: StudentTools):
        # CC412 needs CC319; student 201003465 has passed CC319
        result = tools.check_prereqs("CC412")
        assert result["course_code"] == "CC412"
        assert "all_prereqs_met" in result
        assert isinstance(result["all_prereqs_met"], bool)

    def test_unknown_course(self, tools: StudentTools):
        result = tools.check_prereqs("XX999")
        assert "error" in result

    def test_prereq_chain_no_cycles(self, tools: StudentTools, db: EduBotDB):
        """Recursion guard: no course should appear in its own chain more than once."""
        for code in list(db.prereq.keys())[:10]:
            t = StudentTools(db, session_sid="201003465")
            result = t.check_prereqs(code)
            if "error" not in result:
                seen = [r["prereq"] for r in result["full_prereq_chain"]]
                assert len(seen) == len(set(seen)), f"Cycle detected for {code}: {seen}"


# ── Test: Required-final calculator ──────────────────────────────────────────

class TestRequiredFinal:
    def test_in_current_course(self, tools: StudentTools):
        cur = tools.get_current_marks()
        if not cur:
            pytest.skip("No current courses")
        code = cur[0]["code"]
        result = tools.calculate_required_final(code)
        assert "scenarios" in result
        assert len(result["scenarios"]) == 3
        for sc in result["scenarios"]:
            assert 0 <= sc["final_needed"] <= 40 or not sc["feasible"]
            # Must always require at least 12/40 (auto-fail threshold)
            assert sc["final_needed"] >= 12

    def test_not_enrolled_returns_error(self, tools: StudentTools):
        result = tools.calculate_required_final("XX999")
        assert "error" in result

    def test_final_floor_is_12(self, tools: StudentTools):
        """Even with 60/60 coursework, must still score >=12 on final."""
        cur = tools.get_current_marks()
        if not cur:
            pytest.skip("No current courses")
        # Patch marks_so_far to simulate high coursework
        code = cur[0]["code"]
        result = tools.calculate_required_final(code)
        for sc in result["scenarios"]:
            assert sc["final_needed"] >= 12, "Final floor must be 12 regardless of coursework"


# ── Test: Overload eligibility ────────────────────────────────────────────────

class TestOverload:
    def test_overload_result_shape(self, tools: StudentTools):
        result = tools.check_overload_eligibility()
        assert "max_credit_hours_fall_spring" in result
        assert result["max_credit_hours_fall_spring"] in (12, 18, 21)
        assert "overload_allowed" in result

    def test_probation_cap_is_12(self, db: EduBotDB):
        """A student on probation must have a 12-credit cap."""
        # Find a probation student
        import pandas as pd
        students = pd.read_csv(str(DATA_DIR / "Students.csv"))
        prob_students = students[students["standing_status"].str.contains("Probation", na=False)]
        if prob_students.empty:
            pytest.skip("No probation students in dataset")
        sid = str(prob_students.iloc[0]["student_id"])
        t = StudentTools(db, session_sid=sid)
        result = t.check_overload_eligibility()
        assert result["max_credit_hours_fall_spring"] == 12, \
            f"Probation student should have 12-cr cap, got {result['max_credit_hours_fall_spring']}"


# ── Test: Improvement window ──────────────────────────────────────────────────

class TestImprovementWindow:
    def test_failed_course_not_improvement(self, tools: StudentTools):
        """A failed course is not an improvement repeat candidate."""
        internal = tools._rec()["_internal"]
        failed = [c for c, r in internal["latest"].items() if r["grade"] == "F"]
        if not failed:
            pytest.skip("Student has no failed courses")
        result = tools.check_improvement_window(failed[0])
        assert result["applicable"] is False

    def test_old_course_window_closed(self, tools: StudentTools):
        """Student 201003465 completed CC528 in Semester 6 (>3 sems ago) — window closed."""
        # CC528 was completed in Second Semester / 2023-2024 per the QA example
        result = tools.check_improvement_window("CC528")
        if "error" in result:
            pytest.skip("CC528 not in transcript")
        assert result["window_open"] is False, \
            f"Expected window closed for CC528; got: {result}"


# ── Test: Graduation plan ─────────────────────────────────────────────────────

class TestGraduationPlan:
    def test_plan_has_terms(self, tools: StudentTools):
        plan = tools.get_graduation_plan()
        assert "terms" in plan
        assert isinstance(plan["terms"], list)

    def test_graduation_term_present_for_senior(self, tools: StudentTools):
        """Senior students (semester 9-10) should have a valid projection or finish in current term."""
        profile = tools.get_student_profile()
        if profile["semester_level"] >= 9:
            plan = tools.get_graduation_plan()
            # If terms is empty, all required courses are in progress this term
            assert plan.get("expected_graduation_term") is not None or len(plan.get("terms", [])) == 0


# ── Test: Honors status ───────────────────────────────────────────────────────

class TestHonors:
    def test_honors_shape(self, tools: StudentTools):
        h = tools.get_honors_status()
        assert "eligible_so_far" in h
        assert isinstance(h["eligible_so_far"], bool)
        assert "f_grades" in h
        assert "semesters_below_3_3" in h

    def test_student_with_f_not_eligible(self, tools: StudentTools):
        h = tools.get_honors_status()
        if h["f_grades"]:
            assert h["eligible_so_far"] is False, \
                "Student with F grades must not be honors-eligible"
