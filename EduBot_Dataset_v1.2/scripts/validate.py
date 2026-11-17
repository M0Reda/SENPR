import csv, sys, collections
from fractions import Fraction as Fr
D = sys.argv[1]
def load(n): return list(csv.DictReader(open(f"{D}/{n}", encoding="utf-8")))
U, A, S, C, PR, O, E, T, Q, K = [load(x) for x in ["Users.csv", "Advisors.csv", "Students.csv", "Courses.csv",
    "Course_Prerequisites.csv", "Course_Offerings.csv", "Course_Enrollments.csv", "Student_Transcripts.csv",
    "Advising_QA.csv", "Student_Cohorts.csv"]]
err = collections.Counter(); ex = {}
def bad(k, m):
    err[k] += 1; ex.setdefault(k, m)
uid = {u["user_id"]: u for u in U}
assert len(uid) == len(U)
assert len({u["university_email"] for u in U}) == len(U)
assert len({u["personal_email"] for u in U}) == len(U)
assert "contact_phone" not in U[0]
print("roles", collections.Counter(u["role"] for u in U))
cr = {c["course_code"]: int(c["credit_hours"]) for c in C}
for a in A:
    if uid.get(a["advisor_id"], {}).get("role") != "Advisor": bad("advisor fk", a)
for s in S:
    if uid.get(s["student_id"], {}).get("role") != "Student": bad("student fk", s)
    if s["advisor_id"] not in {a["advisor_id"] for a in A}: bad("stu advisor fk", s)
for p in PR:
    if p["course_code"] not in cr or p["prereq_course_code"] not in cr: bad("prereq fk", p)
off = {o["offering_id"]: o for o in O}
for o in O:
    if o["course_code"] not in cr: bad("off course", o)
    if uid.get(o["professor_id"], {}).get("role") != "Professor": bad("off prof", o)
    if uid.get(o["instructor_id"], {}).get("role") != "Instructor": bad("off inst", o)
sids = {s["student_id"] for s in S}
for e in E:
    if e["offering_id"] not in off: bad("enr off", e)
    if e["student_id"] not in sids: bad("enr stu", e)
for t in T:
    if t["course_code"] not in cr: bad("tr course", t)
    if t["student_id"] not in sids: bad("tr stu", t)
for q in Q:
    if q["student_id"] not in sids: bad("qa stu", q)

import math
P3 = {"A+": 12, "A": 11.5, "A-": 11, "B+": 10, "B": 9, "B-": 8, "C+": 7, "C": 6, "C-": 5, "D+": 4, "D": 3, "F": 0, "P": 0}
def r1(x): return Fr(math.floor(Fr(x) * 10 + Fr(1, 2)), 10)
def PTS(g, c): return r1(Fr(str(P3[g])) * c / 3)
# spot checks from the brief
assert PTS("A", 3) == Fr(23, 2) and PTS("A", 2) == Fr(77, 10) and PTS("A+", 6) == 24
assert r1(Fr("1.95")) == 2 and r1(Fr("0.96")) == 1 and r1(Fr("1.85")) == Fr("1.9")
BANDS = [(97, "A+"), (93, "A"), (89, "A-"), (84, "B+"), (80, "B"), (76, "B-"), (73, "C+"), (70, "C"), (67, "C-"), (64, "D+"), (60, "D")]
def letter(tot, fin):
    if fin < 12: return "F"
    for lo, g in BANDS:
        if tot >= lo: return g
    return "F"
ORDER = ["F", "D", "D+", "C-", "C", "C+", "B-", "B", "B+", "A-", "A", "A+"]
def tkey(label):
    s, y = label.split(" / "); y0 = int(y[:4])
    return y0 * 3 + {"First Semester": 0, "Second Semester": 1, "Summer Semester": 2}[s]
def okey(o):
    return int(o["academic_year"][:4]) * 3 + {"Fall": 0, "Spring": 1, "Summer": 2}[o["semester_term"]]
CURK = 2026 * 3

for t in T:
    if Fr(t["points"]) != PTS(t["grade"], cr[t["course_code"]]): bad("points", t)

prereq = collections.defaultdict(list)
for p in PR: prereq[p["course_code"]].append(p["prereq_course_code"])

trs = collections.defaultdict(list)
for t in T: trs[t["student_id"]].append(t)
ens = collections.defaultdict(list)
for e in E: ens[e["student_id"]].append(e)
stu = {s["student_id"]: s for s in S}
coh = {k["student_id"]: k["cohort"] for k in K}

for sid, s in stu.items():
    tl = sorted(trs[sid], key=lambda t: (tkey(t["semester_taken"]), t["transcript_id"]))
    el = ens[sid]
    # map enrollment (course, term) -> enrollment
    emap = {}
    for e in el:
        o = off[e["offering_id"]]
        emap[(o["course_code"], okey(o))] = e
    hist_e = [k for k in emap if k[1] < CURK]
    if len(hist_e) != len(tl): bad("enr/transcript count", sid)
    # chronological replay
    latest = {}
    seen_F = set()
    terms = sorted({tkey(t["semester_taken"]) for t in tl})
    improve = 0
    for k in terms + [CURK]:
        cnt = [latest[c] for c in latest if latest[c]["grade"] != "P"]
        Aa = sum(cr[t["course_code"]] for t in cnt); Pp = sum(PTS(t["grade"], cr[t["course_code"]]) for t in cnt)
        gpa = r1(Pp / Aa) if Aa else None
        passed = {c for c, t in latest.items() if t["grade"] != "F"}
        level = min(10, sum(cr[c] for c in passed) // 18 + 1)
        rows = [t for t in tl if tkey(t["semester_taken"]) == k] if k != CURK else \
               [dict(course_code=c) for (c, kk) in emap if kk == CURK]
        load = sum(cr[r["course_code"]] for r in rows)
        if k % 3 == 2:
            fin3 = level >= 9
            if len(rows) > (3 if fin3 else 2): bad("summer courses", (sid, k, len(rows)))
        else:
            cap = 18 if gpa is None else (12 if gpa < 2 else (21 if gpa >= 3 or (level >= 9 and gpa > 2) else 18))
            if load > cap: bad("load cap", (sid, k, load, cap, gpa))
        for r in rows:
            c = r["course_code"]
            for p in prereq[c]:
                if p not in passed: bad("prereq violated", (sid, c, p, k))
            if c in passed and latest.get(c, {}).get("grade") not in ("D", "D+", "C-"):
                bad("repeat of good grade", (sid, c))
        for r in rows if k != CURK else []:
            c = r["course_code"]
            e = emap[(c, k)]
            if c in cr and r["grade"] != "P":
                tot = float(e["total_score"]); fin = float(e["final_exam_score"])
                parts = sum(float(e[x]) for x in ["seventh_week_score", "twelfth_week_score", "prefinal_score", "final_exam_score"])
                if abs(parts - tot) > 0.01: bad("total sum", e)
                exp = letter(tot, fin)
                if c in latest and latest[c]["grade"] == "F" and ORDER.index(exp) > ORDER.index("B+"): exp = "B+"
                if exp != r["grade"]: bad("grade mismatch", (sid, c, tot, fin, r["grade"], exp))
            if c in latest and latest[c]["status"] != "Retaken": bad("retaken status", (sid, c))
            latest[c] = r
    for c, t in latest.items():
        if t["status"] == "Retaken": bad("latest marked retaken", (sid, c))
        if t["status"] != ("Failed" if t["grade"] == "F" else "Passed"): bad("status", t)
    cnt = [t for t in latest.values() if t["grade"] != "P"]
    Aa = sum(cr[t["course_code"]] for t in cnt); Pp = sum(PTS(t["grade"], cr[t["course_code"]]) for t in cnt)
    gpa = r1(Pp / Aa) if Aa else None
    credits = sum(cr[c] for c, t in latest.items() if t["grade"] != "F")
    if int(s["total_credits_completed"]) != credits: bad("credits", (sid, s["total_credits_completed"], credits))
    if gpa is None:
        if s["gpa"] != "": bad("gpa blank", sid)
    else:
        if s["gpa"] != f"{float(gpa):.1f}": bad("gpa", (sid, s["gpa"], float(gpa)))
        st = s["standing_status"]
        exp = ("Academic Probation" if gpa < 2 else "Excellent" if gpa >= Fr(37, 10) else "Very Good" if gpa >= Fr(33, 10)
               else "Good" if gpa >= Fr(27, 10) else "Pass")
        if not st.startswith(exp): bad("standing", (sid, st, float(gpa)))
    lvl = int(s["current_semester_level"])
    if lvl != min(10, credits // 18 + 1): bad("level formula", (sid, lvl, credits))
    c0 = coh[sid]
    if c0.startswith("Academic") != s["standing_status"].startswith("Academic"): bad("cohort prob", sid)
    if c0.startswith("Graduation") and lvl < 9: bad("cohort grad", sid)
    # per-student transcript file agrees with master transcript and Students row
    pf = list(csv.DictReader(open(f"{D}/Student_Transcripts/{sid}.csv", encoding="utf-8")))
    done_rows = [r for r in pf if r["status"] != "In Progress"]
    if len(done_rows) != len(tl): bad("file rows", sid)
    for r in done_rows:
        if r["semester_gpa"] and Fr(r["semester_gpa"]) != r1(Fr(r["semester_points"]) / int(r["semester_credit_hours"])): bad("file sem gpa", r)
        if Fr(r["cumulative_gpa"]) != r1(Fr(r["cumulative_points"]) / int(r["cumulative_credit_hours"])): bad("file cum gpa", r)
    if done_rows and (done_rows[-1]["cumulative_gpa"] != s["gpa"] or done_rows[-1]["standing_after_semester"] != s["standing_status"]):
        bad("file final", (sid, done_rows[-1]["cumulative_gpa"], s["gpa"], done_rows[-1]["standing_after_semester"], s["standing_status"]))
    # current enrollments have only 7th/12th
    for (c, kk), e in emap.items():
        if kk == CURK and (e["final_exam_score"] or e["total_score"]): bad("current has final", e)
    # anomaly feedback
    for e in el:
        if e["instructor_feedback"] != "None" and (e["seventh_week_score"] == "" or float(e["seventh_week_score"]) >= 18):
            bad("feedback w/o low 7th", e)

print("ERRORS:", dict(err) or "none")
for k, v in ex.items(): print(" ", k, v)
print("cohorts", collections.Counter(coh.values()))
print("standing", collections.Counter(s["standing_status"] for s in S))
print("level", sorted(collections.Counter(int(s["current_semester_level"]) for s in S).items()))
print("phase", collections.Counter(s["enrollment_phase"] for s in S))
print("intents", collections.Counter(q["intent_category"] for q in Q))
print("feedback notes", sum(e["instructor_feedback"] != "None" for e in E))
print("grades", collections.Counter(t["grade"] for t in T))
print("status", collections.Counter(t["status"] for t in T))
print("intakes", sorted(collections.Counter(s[:3] for s in stu).items()))

import statistics
print("prob by level", sorted(collections.Counter(int(s["current_semester_level"]) for s in S if s["standing_status"].startswith("Academic")).items()))
print("cohort file", collections.Counter(k["cohort"] for k in K))

# ---- tickets: spot-audit facts quoted by EduBot against the tables
import json, re
TK = [json.loads(l) for l in open(f"{D}/EduBot_Training/tickets.jsonl", encoding="utf-8")]
bad_t = collections.Counter()
for t in TK:
    s = stu[t["student_id"]]
    for m in t["messages"]:
        if m["role"] != "assistant": continue
        for g in re.findall(r"Your cumulative GPA is (\d\.\d)", m["content"]):
            if g != s["gpa"]: bad_t["gpa quote"] += 1
        for c in re.findall(r"You've completed (\d+) credit hours", m["content"]):
            if c != s["total_credits_completed"]: bad_t["credits quote"] += 1
        for st in re.findall(r"Your standing is ([A-Za-z ()0-9]+)\.", m["content"]):
            if st != s["standing_status"]: bad_t["standing quote"] += 1
    if t["messages"][0]["role"] != "user" or t["messages"][-1]["role"] != "assistant": bad_t["turn order"] += 1
print("ticket audit:", dict(bad_t) or "no mismatches", "| tickets", len(TK))
