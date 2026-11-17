"""EduBot context builder.

Builds the STUDENT RECORD that EduBot sees, straight from the 9 CSV tables.
The same code is used to create the training tickets and to build prompts at
inference time (Colab / portal), so training and production see identical inputs.

    from edubot_context import EduBotDB, SYSTEM_PROMPT
    db = EduBotDB("path/to/csv_folder")
    rec = db.record("241004053")          # dict
    prompt = db.system_message("241004053")  # SYSTEM_PROMPT + record as JSON
"""
import csv, json, math, os
from fractions import Fraction as Fr

CURRENT_TERM = "First Semester / 2026-2027"
NEXT_TERMS = ["Second Semester / 2026-2027", "First Semester / 2027-2028", "Second Semester / 2027-2028",
              "First Semester / 2028-2029", "Second Semester / 2028-2029", "First Semester / 2029-2030",
              "Second Semester / 2029-2030", "First Semester / 2030-2031", "Second Semester / 2030-2031",
              "First Semester / 2031-2032", "Second Semester / 2031-2032"]
PTS3 = {"A+": Fr(12), "A": Fr(23, 2), "A-": Fr(11), "B+": Fr(10), "B": Fr(9), "B-": Fr(8), "C+": Fr(7),
        "C": Fr(6), "C-": Fr(5), "D+": Fr(4), "D": Fr(3), "F": Fr(0)}
GRADE_ORDER = ["F", "D", "D+", "C-", "C", "C+", "B-", "B", "B+", "A-", "A", "A+"]
BANDS = [(97, "A+"), (93, "A"), (89, "A-"), (84, "B+"), (80, "B"), (76, "B-"), (73, "C+"), (70, "C"),
         (67, "C-"), (64, "D+"), (60, "D")]
N_ELECTIVES = 6
TOTAL_CREDITS = 180

SYSTEM_PROMPT = """You are EduBot, the AI academic advisor of the Computer Engineering Department, College of Engineering & Technology, Arab Academy for Science, Technology & Maritime Transport (AASTMT).
You answer the student whose STUDENT RECORD is attached below. Use only that record and the regulations here. Never invent grades, marks or courses. Show your arithmetic when you give a number. If the question is ambiguous (for example, it is not clear which course is meant), ask one short clarifying question before answering. If the request needs a human (changing a mark, approving an exception, account problems), explain what you can and route the ticket to the right person.

REGULATIONS
- Assessment out of 100: 7th-week 30, 12th-week 20, pre-final 10, final exam 40. The instructor announces the lecture/lab split in week 1. A final exam score below 12/40 is an automatic F, whatever the coursework.
- Letter grades: A+ >=97, A >=93, A- >=89, B+ >=84, B >=80, B- >=76, C+ >=73, C >=70, C- >=67, D+ >=64, D >=60, F <60.
- Points for a 3-credit course: A+ 12, A 11.5, A- 11, B+ 10, B 9, B- 8, C+ 7, C 6, C- 5, D+ 4, D 3, F 0. Other credit hours scale by ratio, rounded to 1 decimal (A in a 2-credit course = 7.7, in the 6-credit Senior Project II = 23). Practical Training (IM400CC, 0 cr) is P/U and is not counted.
- Semester GPA = points achieved / credit hours taken that semester (an F counts with 0 points). Cumulative GPA uses the same formula over all terms; when a course is repeated the latest attempt replaces the earlier one. GPAs are rounded half-up to 1 decimal (1.95 -> 2.0, 1.85 -> 1.9) and the rounded value is used for every rule.
- Standing: Excellent 3.7-4.0, Very Good 3.3-3.6, Good 2.7-3.2, Pass 2.0-2.6, below 2.0 = Academic Probation. Probation warnings count consecutive terms; a 3rd consecutive probation term means a mandatory Change of Major, and 3 more consecutive probation terms in the new major means dismissal.
- Semester level = completed credit hours / 18 rounded down + 1 (0-17 -> 1, 18-35 -> 2 ... 162-179 -> 10). 180 completed credit hours with GPA >= 2.0 = graduate.
- Degree: 180 credit hours, cumulative GPA >= 2.0, 4.5 to 10 years, at least 50% of credits at AASTMT. 6 electives (1 in term 7, 1 in term 8, 2 in term 9, 2 in term 10). Senior Project I (CC501, 3 cr) before Senior Project II (CC503, 6 cr). Practical Training IM400CC in the final term.
- Load (Fall/Spring): normal 18, minimum 9. GPA >= 3.0 may take up to 21. Final-year students (semesters 9-10) with GPA above 2.0 may take up to 21. Probation (GPA < 2.0): maximum 12, registered with the academic advisor.
- Summer: maximum 2 courses; a graduating senior who is behind may be approved for 3 courses in the final summer.
- Repeats: a failed course must be repeated, and the repeat grade is capped at B+. Regular students may repeat up to 5 passed courses to improve their GPA, within 3 semesters of first completing the course. Students on probation may repeat as many as needed under advisor supervision.
- Prerequisites are hard requirements: they cannot be bypassed or taken in the same term as the dependent course.
- Attendance: at least 85%. Absence above 15% of scheduled classes means forced withdrawal from the course.
- Honors degree: semester GPA >= 3.3 in every semester and no F at any time."""


def rnd1(x):
    return Fr(math.floor(Fr(x) * 10 + Fr(1, 2)), 10)

def pts(grade, cr):
    if grade in ("P", "U", "") or cr == 0:
        return Fr(0)
    return rnd1(PTS3[grade] * cr / 3)

def letter(total, final):
    if final < 12:
        return "F"
    for lo, g in BANDS:
        if total >= lo:
            return g
    return "F"

def standing_of(g):
    if g >= Fr(37, 10): return "Excellent"
    if g >= Fr(33, 10): return "Very Good"
    if g >= Fr(27, 10): return "Good"
    if g >= 2: return "Pass"
    return "Academic Probation"

def term_key(label):
    s, y = label.split(" / ")
    return int(y[:4]) * 3 + {"First Semester": 0, "Second Semester": 1, "Summer Semester": 2}[s]

def regular_semesters_between(after_label, upto_label):
    """Fall/Spring semesters on the calendar strictly after `after_label`, up to and including `upto_label`."""
    return sum(1 for k in range(term_key(after_label) + 1, term_key(upto_label) + 1) if k % 3 != 2)

def level_of(credits):
    return min(10, credits // 18 + 1)

def g1(x):
    return f"{float(x):.1f}"


class EduBotDB:
    def __init__(self, folder):
        rd = lambda n: list(csv.DictReader(open(os.path.join(folder, n), encoding="utf-8")))
        self.users = {u["user_id"]: u for u in rd("Users.csv")}
        self.advisors = {a["advisor_id"]: a for a in rd("Advisors.csv")}
        self.students = {s["student_id"]: s for s in rd("Students.csv")}
        self.courses = {c["course_code"]: dict(c, credit_hours=int(c["credit_hours"]), catalog_order=i)
                        for i, c in enumerate(rd("Courses.csv"))}
        self.prereq = {}
        for p in rd("Course_Prerequisites.csv"):
            self.prereq.setdefault(p["course_code"], []).append(p["prereq_course_code"])
        self.offerings = {o["offering_id"]: o for o in rd("Course_Offerings.csv")}
        self.enr = {}
        for e in rd("Course_Enrollments.csv"):
            self.enr.setdefault(e["student_id"], []).append(e)
        self.trs = {}
        for t in rd("Student_Transcripts.csv"):
            self.trs.setdefault(t["student_id"], []).append(t)
        # curriculum order (plan term) for planning; electives sorted by code
        self.plan_order = sorted([c for c, v in self.courses.items() if v["course_type"] == "Required"],
                                 key=lambda c: (int(self.courses[c]["recommended_term"]), self.courses[c]["catalog_order"]))
        self.electives = sorted(c for c, v in self.courses.items() if v["course_type"] == "Elective")

    def name(self, uid):
        u = self.users[uid]
        return f"{u['first_name']} {u['last_name']}"

    def cr(self, c):
        return self.courses[c]["credit_hours"]

    def title(self, c):
        return self.courses[c]["course_title"]

    @staticmethod
    def off_label(o):
        y = o["academic_year"]
        return {"Fall": "First Semester", "Spring": "Second Semester", "Summer": "Summer Semester"}[o["semester_term"]] + " / " + y

    # ------------------------------------------------------------------
    def record(self, sid):
        s = self.students[sid]
        u = self.users[sid]
        tl = sorted(self.trs.get(sid, []), key=lambda t: (term_key(t["semester_taken"]), t["transcript_id"]))
        marks = {}
        current = []
        for e in self.enr.get(sid, []):
            o = self.offerings[e["offering_id"]]
            lab = self.off_label(o)
            if lab == CURRENT_TERM:
                current.append((o, e))
            else:
                marks[(o["course_code"], lab)] = e
        # chronological replay of the transcript
        latest, history, pc, improve_used, any_F = {}, [], 0, 0, []
        terms = sorted({t["semester_taken"] for t in tl}, key=term_key)
        for term in terms:
            rows = [t for t in tl if t["semester_taken"] == term]
            for r in rows:
                prev = latest.get(r["course_code"])
                if prev is not None and prev["grade"] != "F" and pc == 0:
                    improve_used += 1
                latest[r["course_code"]] = r
                if r["grade"] == "F":
                    any_F.append(f"{r['course_code']} ({term})")
            sem = [r for r in rows if r["grade"] != "P"]
            sA = sum(self.cr(r["course_code"]) for r in sem)
            sP = sum(pts(r["grade"], self.cr(r["course_code"])) for r in sem)
            cnt = [r for r in latest.values() if r["grade"] != "P"]
            cA = sum(self.cr(r["course_code"]) for r in cnt)
            cP = sum(pts(r["grade"], self.cr(r["course_code"])) for r in cnt)
            cg = rnd1(cP / cA)
            if not term.startswith("Summer"):
                pc = pc + 1 if cg < 2 else 0
            elif cg >= 2:
                pc = 0
            elif pc == 0:
                pc = 1
            history.append(dict(term=term, credit_hours=sA, points=g1(sP),
                                semester_gpa=g1(rnd1(sP / sA)) if sA else None,
                                cumulative_credit_hours=cA, cumulative_points=g1(cP), cumulative_gpa=g1(cg),
                                standing=(f"Academic Probation ({['1st', '2nd'][pc - 1]} Warning)" if pc else standing_of(cg))))
        passed = {c for c, r in latest.items() if r["grade"] != "F"}
        credits = sum(self.cr(c) for c in passed)
        cnt = [r for r in latest.values() if r["grade"] != "P"]
        A = sum(self.cr(r["course_code"]) for r in cnt)
        P = sum(pts(r["grade"], self.cr(r["course_code"])) for r in cnt)
        gpa = rnd1(P / A) if A else None
        lvl = level_of(credits)
        on_prob = gpa is not None and gpa < 2
        if gpa is None:
            cap, why = 18, "first semester: standard load"
        elif on_prob:
            cap, why = 12, "academic probation (GPA below 2.0): maximum 12, registered with the advisor"
        elif gpa >= 3:
            cap, why = 21, "GPA 3.0 or higher: up to 21 allowed"
        elif lvl >= 9 and gpa > 2:
            cap, why = 21, "final-year student with GPA above 2.0: up to 21 allowed"
        else:
            cap, why = 18, "standard maximum load"

        def tline(t):
            c = t["course_code"]
            line = f"{t['semester_taken']} | {c} {self.title(c)} | {self.cr(c)} cr | {t['grade']} | {g1(Fr(t['points']))} pts | {t['status']}"
            if t["grade"] == "F":
                e = marks.get((c, t["semester_taken"]))
                if e and e["total_score"]:
                    line += (f" | marks {e['seventh_week_score']}+{e['twelfth_week_score']}+{e['prefinal_score']}"
                             f"+{e['final_exam_score']}={e['total_score']}")
            return line

        cur = []
        cur_codes = set()
        for o, e in sorted(current, key=lambda x: x[0]["offering_id"]):
            c = o["course_code"]
            cur_codes.add(c)
            prev = latest.get(c)
            if prev is None:
                att = "first attempt"
            elif prev["grade"] == "F":
                att = "repeat of a failed course (grade capped at B+)"
            else:
                att = f"improvement repeat (previous grade {prev['grade']})"
            item = dict(code=c, title=self.title(c), credit_hours=self.cr(c),
                        professor=self.name(o["professor_id"]), instructor=self.name(o["instructor_id"]),
                        attempt=att)
            if e["seventh_week_score"] != "":
                item.update(seventh_week=float(e["seventh_week_score"]), twelfth_week=float(e["twelfth_week_score"]),
                            marks_so_far=f"{float(e['seventh_week_score']) + float(e['twelfth_week_score']):g}/50")
            if e["instructor_feedback"] != "None":
                item["instructor_feedback"] = e["instructor_feedback"]
            cur.append(item)

        after = passed | cur_codes
        req_left = [c for c in self.plan_order if c not in passed]
        el_done = sum(1 for c in passed if self.courses[c]["course_type"] == "Elective")
        el_now = sum(1 for c in cur_codes if self.courses[c]["course_type"] == "Elective" and c not in passed)
        remaining_req = []
        for c in req_left:
            if c in cur_codes:
                continue
            remaining_req.append(dict(code=c, title=self.title(c), credit_hours=self.cr(c),
                                      prerequisites=self.prereq.get(c, []),
                                      prerequisites_met_if_current_courses_passed=all(p in after for p in self.prereq.get(c, []))))
        el_need = max(0, N_ELECTIVES - el_done - el_now)
        eligible_el = [f"{c} {self.title(c)}" for c in self.electives
                       if c not in after and all(p in after for p in self.prereq.get(c, []))]
        plan = self._plan(passed, cur_codes, credits, gpa, on_prob)
        adv = s["advisor_id"]
        rec = dict(
            student=dict(student_id=sid, name=self.name(sid), university_email=u["university_email"],
                         university_email_activated=u["is_email_activated"] == "TRUE",
                         current_term=CURRENT_TERM, semester_level=lvl, enrollment_phase=s["enrollment_phase"],
                         credits_completed=credits, credits_remaining=TOTAL_CREDITS - credits,
                         cumulative_points=g1(P) if A else None, cumulative_credit_hours=A,
                         cumulative_gpa=g1(gpa) if gpa is not None else None,
                         standing=s["standing_status"], consecutive_probation_warnings=pc),
            advisor=dict(advisor_id=adv, name=self.name(adv), email=self.users[adv]["university_email"],
                         office=self.advisors[adv]["office_room"]),
            registration_rules_now=dict(max_credit_hours_fall_spring=cap, reason=why,
                                        registered_this_term=sum(self.cr(x["code"]) for x in cur)),
            repeats=dict(outstanding_failed_courses=[c for c, r in latest.items() if r["grade"] == "F"],
                         improvement_repeats_used=improve_used,
                         improvement_repeats_left=(None if on_prob else max(0, 5 - improve_used))),
            honors_check=dict(f_grades=any_F,
                              semesters_below_3_3=[h["term"] for h in history if h["semester_gpa"] and Fr(h["semester_gpa"]) < Fr(33, 10)]),
            semester_history=history,
            transcript=[tline(t) for t in tl],
            current_courses=cur,
            remaining=dict(required_courses_not_taken=[
                f"{r['code']} {r['title']} ({r['credit_hours']} cr) | prereq: {' & '.join(r['prerequisites']) or 'none'} | "
                f"prereqs met after this term: {'yes' if r['prerequisites_met_if_current_courses_passed'] else 'no'}" for r in remaining_req],
                           electives_completed=el_done,
                           electives_in_progress=el_now, electives_still_needed=el_need,
                           electives_eligible_after_this_term=eligible_el),
            graduation_plan_projection=plan,
            _internal=dict(remaining_req=remaining_req, rows=tl, latest=latest, passed=passed, marks=marks, current_codes=cur_codes,
                           P=P, A=A, gpa=gpa, on_prob=on_prob, credits=credits),
        )
        return rec

    def _plan(self, passed, cur_codes, credits, gpa, on_prob):
        """Greedy term-by-term plan from next semester, assuming every current course is passed."""
        done = set(passed) | set(cur_codes)
        cred = credits + sum(self.cr(c) for c in cur_codes if c not in passed)
        req = [c for c in self.plan_order if c not in done]
        el_need = max(0, N_ELECTIVES - sum(1 for c in done if self.courses[c]["course_type"] == "Elective"))
        can21 = gpa is not None and (gpa >= 3 or gpa > 2)
        terms = []
        for i, term in enumerate(NEXT_TERMS):
            if not req and not el_need:
                break
            lvl = level_of(cred)
            left = sum(self.cr(c) for c in req) + 3 * el_need
            if i == 0 and on_prob:
                cap = 12
            elif gpa is not None and (gpa >= 3 or (lvl >= 9 and gpa > 2)) and left <= 21:
                cap = 21
            else:
                cap = 18
            pick, crs = [], 0
            for c in req:
                if c == "IM400CC" or (c == "CC501" and lvl < 8):
                    continue
                if all(p in done for p in self.prereq.get(c, [])) and crs + self.cr(c) <= cap:
                    pick.append(c); crs += self.cr(c)
            n_el = 0
            if lvl >= 6:
                for e in self.electives:
                    if n_el >= el_need or crs + 3 > cap:
                        break
                    if e not in done and all(p in done for p in self.prereq.get(e, [])):
                        pick.append(e); crs += 3; n_el += 1
            if "IM400CC" in req and not [c for c in req if c not in pick and c != "IM400CC"] and el_need - n_el == 0:
                pick.append("IM400CC")
            if not pick:
                break
            done |= set(pick); cred += crs; el_need -= n_el
            req = [c for c in req if c not in pick]
            terms.append(dict(term=term, credit_hours=crs, max_load_assumed=cap,
                              courses=[f"{c} {self.title(c)} ({self.cr(c)} cr)" + (" [elective]" if self.courses[c]["course_type"] == "Elective" else "") for c in pick]))
        return dict(assumptions="passes all current courses; Fall/Spring only (summer can shorten it); "
                                + ("12-credit probation cap in the first term, normal load after" if on_prob else "normal load rules"),
                    terms=terms, expected_graduation_term=(terms[-1]["term"] if terms and not req and not el_need else None))

    def system_message(self, sid):
        return self.system_from_record(self.record(sid))

    @staticmethod
    def public(rec):
        return {k: v for k, v in rec.items() if not k.startswith("_")}

    @classmethod
    def system_from_record(cls, rec):
        return SYSTEM_PROMPT + "\n\nSTUDENT RECORD (JSON):\n" + json.dumps(cls.public(rec), ensure_ascii=False, separators=(",", ":"))
