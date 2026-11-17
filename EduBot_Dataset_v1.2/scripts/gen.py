#!/usr/bin/env python3
"""EduBot synthetic dataset generator — AASTMT Computer Engineering.
Term-by-term simulation: every grade, GPA, standing and Q&A answer is derived from
simulated records, so all 9 tables stay mutually consistent."""
import csv, os, random, sys, re
from fractions import Fraction as Fr
import openpyxl

SEED = 20261011
rng = random.Random(SEED)
OUT = sys.argv[2] if len(sys.argv) > 2 else "out"
os.makedirs(OUT, exist_ok=True)

# ------------------------------------------------------------------ curriculum
wb = openpyxl.load_workbook(sys.argv[1], data_only=True)
ws = wb.active
COURSES = {}       # code -> dict(title, cr, type, term)
PREREQ = {}        # code -> [codes]
PLAN = []          # ordered list of (code|'ELEC', plan_term)
ELECTIVES = []
for r in ws.iter_rows(values_only=True):
    if not r or r[1] is None or r[0] in ("Term / Category", "Category"):
        continue
    cat, code, title, cr, pre = r
    if not isinstance(cr, (int, float)):
        continue
    code = code.replace("*", "").strip()
    if cat.startswith("Term"):
        t = int(cat.split()[1])
        if code == "CCXXX":
            PLAN.append(("ELEC", t)); continue
        COURSES[code] = dict(title=title, cr=int(cr), type="Required", term=t)
        PLAN.append((code, t))
    else:
        COURSES[code] = dict(title=title, cr=int(cr), type="Elective", term=7)
        ELECTIVES.append(code)
    p = str(pre).strip()
    if p == "None":
        PREREQ[code] = []
    else:
        # "A / B" = either; only first exists in catalogue (CC213 is not in the plan)
        p = p.split("/")[0] if "/" in p and "&" not in p else p
        PREREQ[code] = [x.strip() for x in p.split("&")]
for c, ps in PREREQ.items():
    for p in ps:
        assert p in COURSES, (c, p)
assert len(ELECTIVES) == 27
N_ELEC = 6
assert sum(v["cr"] for v in COURSES.values() if v["type"] == "Required") + N_ELEC * 3 == 180

# ------------------------------------------------------------------ grading
GRADES = [(97, "A+", 12), (93, "A", Fr(23, 2)), (89, "A-", 11), (84, "B+", 10), (80, "B", 9),
          (76, "B-", 8), (73, "C+", 7), (70, "C", 6), (67, "C-", 5), (64, "D+", 4), (60, "D", 3)]
FACTOR = {g: Fr(n) / 3 for _, g, n in GRADES}
FACTOR["F"] = Fr(0)
ORDER = ["F", "D", "D+", "C-", "C", "C+", "B-", "B", "B+", "A-", "A", "A+"]

import math
PTS3 = {"A+": Fr(12), "A": Fr(23, 2), "A-": Fr(11), "B+": Fr(10), "B": Fr(9), "B-": Fr(8), "C+": Fr(7),
        "C": Fr(6), "C-": Fr(5), "D+": Fr(4), "D": Fr(3), "F": Fr(0)}
def rnd1(x):
    """round half-up to 1 decimal (1.95 -> 2.0, 1.85 -> 1.9, 0.96 -> 1.0)"""
    return Fr(math.floor(x * 10 + Fr(1, 2)), 10)
def pts(gr, cr):
    """course points: 3-cr table, other credit hours by ratio, rounded to 1 decimal (A in 2 cr = 7.7)"""
    if gr == "P" or cr == 0: return Fr(0)
    return rnd1(PTS3[gr] * cr / 3)

def letter(total, final):
    if final < 12:
        return "F"
    for lo, g, _ in GRADES:
        if total >= lo:
            return g
    return "F"

def standing(g):
    if g >= Fr(37, 10): return "Excellent"
    if g >= Fr(33, 10): return "Very Good"
    if g >= Fr(27, 10): return "Good"
    if g >= 2: return "Pass"
    return "Probation"

def fmt(x, d=2):
    return f"{float(x):.{d}f}"

# ------------------------------------------------------------------ calendar
# term = (academic_year_start, k)  k: 0 Fall/First, 1 Spring/Second, 2 Summer
CURRENT = (2026, 0)
def key(t): return t[0] * 3 + t[1]
def nxt(t): return (t[0], t[1] + 1) if t[1] < 2 else (t[0] + 1, 0)
def ay(t): return f"{t[0]}-{t[0]+1}"
def sem_label(t): return ["First Semester", "Second Semester", "Summer Semester"][t[1]] + " / " + ay(t)
def short(t): return ["Fall", "Spring", "Summer"][t[1]] + f" {t[0] + (0 if t[1]==0 else 1)}"
TERMNAME = ["Fall", "Spring", "Summer"]

# per-course difficulty
DIFF = {c: rng.gauss(0, 3) for c in COURSES}
for c in COURSES:
    if c.startswith("BA") and c not in ("BA118",): DIFF[c] -= 3
    if c in ("CC418", "CC410", "CC311", "CC317", "EC339", "EE418", "CC319"): DIFF[c] -= 2.5
    if c.startswith(("LH", "IM", "NE")): DIFF[c] += 4

def r05(x, hi):
    return max(0.0, min(hi, round(x * 2) / 2))

# ------------------------------------------------------------------ simulation
class Reject(Exception): pass

def simulate(intake, mu, sd, summer_p, high_load_p, ff_p, dip_p, s_rng, shock_at=None, shock=0):
    att = []            # completed attempts
    latest = {}         # course -> index into att of latest attempt
    improve_used = 0
    prob_count = 0
    prob_hist = []      # (term, count) after each term
    reg_terms = []      # regular term records with semester gpa
    t = intake
    cur_mu = mu
    first_regular_done = False

    def counted():
        return [a for c, i in latest.items() for a in [att[i]] if a["grade"] != "P"]
    def cum():
        cs = counted()
        A = sum(COURSES[a["course"]]["cr"] for a in cs)
        P = sum(pts(a["grade"], COURSES[a["course"]]["cr"]) for a in cs)
        return P, A, (rnd1(P / A) if A else None)
    def passed():
        return {c for c, i in latest.items() if att[i]["grade"] != "F"}
    def credits_done():
        return sum(COURSES[c]["cr"] for c in passed())
    def level():
        return min(10, credits_done() // 18 + 1)
    def outstanding_F():
        return [c for c, i in latest.items() if att[i]["grade"] == "F"]
    def reg_count_since(term, now):
        """Fall/Spring semesters on the calendar after `term`, up to and including `now`"""
        return sum(1 for k in range(key(term) + 1, key(now) + 1) if k % 3 != 2)

    def choose(term, summer, cap_cr, target_cr, cap_courses=None):
        nonlocal improve_used
        P_, A_, g = cum()
        pas = passed()
        lvl = level()
        chosen = []
        crs = 0
        def can_add(c):
            return crs + COURSES[c]["cr"] <= cap_cr and (cap_courses is None or len(chosen) < cap_courses)
        for c in outstanding_F():
            if can_add(c):
                chosen.append((c, "repeatF")); crs += COURSES[c]["cr"]
        # improvement repeats (within 3 regular semesters of the attempt)
        on_prob = g is not None and g < 2
        if not summer and g is not None:
            p_imp = 0.30 if on_prob else 0.07
            if (on_prob or improve_used < 5) and s_rng.random() < p_imp:
                cands = [c for c in pas if att[latest[c]]["grade"] in ("D", "D+", "C-")
                         and not att[latest[c]].get("was_retake_F")
                         and reg_count_since(att[latest[c]]["term"], term) <= 3
                         and COURSES[c]["cr"] > 0]
                s_rng.shuffle(cands)
                if cands and can_add(cands[0]):
                    chosen.append((cands[0], "improve")); crs += COURSES[cands[0]]["cr"]
                    if not on_prob: improve_used += 1
        n_elec_have = sum(1 for c in latest if COURSES[c]["type"] == "Elective" and c in pas)
        n_elec_have += sum(1 for c, _ in chosen if COURSES[c]["type"] == "Elective")
        elec_slot = 0
        for item, pt in PLAN:
            if crs >= target_cr or (cap_courses and len(chosen) >= cap_courses):
                break
            if pt > lvl + 1:
                break
            if item == "ELEC":
                elec_slot += 1
                if elec_slot <= n_elec_have:
                    continue
                el = [e for e in ELECTIVES if e not in latest and e not in [c for c, _ in chosen]
                      and all(p in pas for p in PREREQ[e])]
                if el and can_add(el[0]):
                    e = s_rng.choice(el)
                    chosen.append((e, "new")); crs += 3; n_elec_have += 1
                continue
            c = item
            if c in latest or c in [x for x, _ in chosen]:
                continue
            if c == "IM400CC":
                continue
            if c == "CC501" and lvl < 8: continue
            if all(p in pas for p in PREREQ[c]) and can_add(c):
                chosen.append((c, "new")); crs += COURSES[c]["cr"]
        # Practical Training (0 cr) is registered with the student's final courses
        if not summer and "IM400CC" not in latest and lvl >= 9:
            got = credits_done() + sum(COURSES[c]["cr"] for c, _ in chosen if c not in pas)
            if got >= 180 or lvl >= 10:
                chosen.append(("IM400CC", "new"))
        return chosen, crs

    def score(course, kind, prior_g, term_mu):
        target = term_mu + DIFF[course] + s_rng.gauss(0, sd) + (3 if kind != "new" else 0)
        target = max(20, min(99.5, target))
        f = target / 100
        dip = prior_g is not None and prior_g >= Fr(33, 10) and s_rng.random() < dip_p
        s7 = r05(30 * (f + s_rng.gauss(0, 0.06)) - (s_rng.uniform(9, 15) if dip else 0), 30)
        s12 = r05(20 * (f + s_rng.gauss(0, 0.06)), 20)
        pf = r05(10 * (f + s_rng.gauss(0, 0.07)), 10)
        fin = r05(40 * (f + s_rng.gauss(0, 0.07)), 40)
        if s7 + s12 + pf >= 34 and s_rng.random() < ff_p:
            fin = r05(s_rng.uniform(5, 11.5), 40)
        return s7, s12, pf, fin

    while key(t) < key(CURRENT):
        summer = t[1] == 2
        P_, A_, g = cum()
        if summer:
            if not att or s_rng.random() > summer_p:
                t = nxt(t); continue
            rem = remaining_courses(passed(), latest, att)
            ccap = 3 if (level() >= 9 and len(rem) <= 3) else 2
            chosen, crs = choose(t, True, 99, 99, ccap)
        else:
            on_prob = g is not None and g < 2
            if g is None: cap = 18
            elif on_prob: cap = 12
            elif g >= 3 or (level() >= 9 and g > 2): cap = 21
            else: cap = 18
            target = 12 if on_prob else (21 if cap == 21 and s_rng.random() < high_load_p else 18)
            if on_prob and s_rng.random() < 0.3: target = 9
            chosen, crs = choose(t, False, cap, target)
            if crs == 0 and not chosen:
                raise Reject("stuck")
        if not chosen:
            t = nxt(t); continue
        if shock_at is not None and key(t) == shock_at:
            cur_mu -= shock
        term_mu = cur_mu + s_rng.gauss(0, 2.5)
        prior_g = g
        for c, kind in chosen:
            a = dict(course=c, term=t, kind=kind)
            if c == "IM400CC":
                a.update(s7=None, s12=None, pf=None, fin=None, total=None, grade="P")
            else:
                for _try in range(40):
                    s7, s12, pf, fin = score(c, kind, prior_g, term_mu)
                    tot = s7 + s12 + pf + fin
                    gr = letter(tot, fin)
                    if kind != "improve" or ORDER.index(gr) > ORDER.index(att[latest[c]]["grade"]):
                        break
                    term_mu += 2
                else:
                    raise Reject("improve")
                if kind == "repeatF":
                    a["was_retake_F"] = True
                    if ORDER.index(gr) > ORDER.index("B+"):
                        gr = "B+"; a["capped"] = True
                a.update(s7=s7, s12=s12, pf=pf, fin=fin, total=tot, grade=gr)
            a["prior_gpa"] = prior_g
            if c in latest:
                att[latest[c]]["superseded"] = True
            att.append(a); latest[c] = len(att) - 1
        # semester gpa
        sem = [a for a in att if a["term"] == t and a["grade"] != "P"]
        sA = sum(COURSES[a["course"]]["cr"] for a in sem)
        sP = sum(pts(a["grade"], COURSES[a["course"]]["cr"]) for a in sem)
        P_, A_, g = cum()
        if not summer:
            prob_count = prob_count + 1 if g < 2 else 0
            reg_terms.append(dict(term=t, sgpa=rnd1(sP / sA) if sA else None, cgpa=g, credits=sA))
        else:
            if g >= 2: prob_count = 0
            elif prob_count == 0: prob_count = 1
        if prob_count >= 3:
            raise Reject("change of major")
        prob_hist.append((t, prob_count))
        # ability drift; probation shock motivates some
        cur_mu += s_rng.gauss(0, 1.5) + (1.2 if prob_count else 0)
        t = nxt(t)

    # current term registration (in progress)
    P_, A_, g = cum()
    if g is None: cap, target = 18, 18
    elif g < 2: cap, target = 12, (9 if s_rng.random() < 0.25 else 12)
    elif g >= 3 or (level() >= 9 and g > 2):
        cap = 21; target = 21 if s_rng.random() < high_load_p else 18
    else: cap, target = 18, 18
    chosen, crs = choose(CURRENT, False, cap, target)
    current = []
    term_mu = cur_mu + s_rng.gauss(0, 2.5)
    for c, kind in chosen:
        if c == "IM400CC":
            current.append(dict(course=c, kind=kind, s7=None, s12=None, prior_gpa=g)); continue
        s7, s12, _, _ = score(c, kind, g, term_mu)
        current.append(dict(course=c, kind=kind, s7=s7, s12=s12, prior_gpa=g))
    return dict(att=att, latest=latest, current=current, cur_cr=crs, prob_count=prob_count,
                reg_terms=reg_terms, improve_used=improve_used, P=P_, A=A_, gpa=g,
                credits=credits_done(), level=level(), passed=passed())

def remaining_courses(pas, latest, att):
    req = [c for c, _ in PLAN if c != "ELEC" and c not in pas]
    n_el = sum(1 for c in pas if COURSES[c]["type"] == "Elective")
    return req + ["ELEC"] * max(0, N_ELEC - n_el)

# ------------------------------------------------------------------ people
MALE = "Ahmed Mohamed Omar Youssef Mahmoud Mostafa Karim Khaled Hassan Hussein Ali Amr Tarek Ziad Seif Adham Marwan Yassin Hamza Abdelrahman Mazen Islam Sherif Ibrahim Moaz Nour Belal Fady Kirollos Mina Bishoy Ayman Hatem Waleed Saif Yehia Osama Ehab Tamer Abdallah Anas Ammar Hazem Eslam Ramy Galal Kareem Sameh Wael Nader Shady Medhat Haitham Taha Zeyad Moataz Hesham Adel Fares Malek".split()
FEMALE = "Nour Salma Mariam Farida Habiba Malak Jana Hana Rana Yasmin Sara Aya Nada Menna Dina Reem Lina Laila Shahd Rawan Hagar Esraa Mai Heba Nadine Jomana Rokaya Basmala Fatma Aisha Mirna Marina Sandra Carol Nourhan Alaa Dalia Ghada Noha Rahma Rodaina Judy Tasneem Asmaa Zeina Lama Kenzy Logain Haya Talia".split()
LAST = "Hassan Mahmoud Ibrahim ElSayed Abdelrahman Mostafa Khalil Farouk Saleh Mansour Fahmy Ghanem Shaker Nasser Hamdy Ezzat Soliman Abdelaziz Zaki Rashad Badawy Sabry Fouad Kamel Lotfy Naguib Amin Helmy Shawky Hafez Morsy Ashour Gaber Tawfik Shehata Radwan Barakat Sharaf Moussa Hegazy Atef Fathy Saad Elshamy Ragab Osman Darwish Seif Hamed Nabil Galal Anwar Samir Wahba Selim Yousry Kassem Elbanna Tolba Gamal Hosny Abdelhamid Awad Metwally Emam Abdelmoneim Rizk Zaghloul Elmasry Habib Ayad Girgis Boutros Hanna Tadros Mikhail".split()
used_uemail = set()
used_pemail = set()

def person(p_rng, female_p=0.3):
    fem = p_rng.random() < female_p
    return (p_rng.choice(FEMALE if fem else MALE), p_rng.choice(LAST))

def slug(s): return re.sub(r"[^a-z]", "", s.lower())

def uemail(f, l, domain):
    base = f"{slug(f)}.{slug(l)}"
    e, n = f"{base}@{domain}", 1
    while e in used_uemail:
        n += 1; e = f"{base}{n}@{domain}"
    used_uemail.add(e); return e

def pemail(f, l, p_rng):
    while True:
        style = p_rng.randint(0, 3)
        num = p_rng.randint(1, 2008)
        loc = [f"{slug(f)}.{slug(l)}{num}", f"{slug(f)}{slug(l)}{num}", f"{slug(f)}_{slug(l)[:4]}{num}",
               f"{slug(f)[0]}{slug(l)}{num}"][style]
        dom = p_rng.choices(["gmail.com", "yahoo.com", "outlook.com", "hotmail.com", "icloud.com"],
                            [60, 12, 14, 10, 4])[0]
        e = f"{loc}@{dom}"
        if e not in used_pemail:
            used_pemail.add(e); return e


users = []
staff = {"Admin": [], "Advisor": [], "Professor": [], "Instructor": []}
emp_seq = {}
def emp_id(yy):
    emp_seq[yy] = emp_seq.get(yy, 100) + rng.randint(1, 40)
    return f"EMP{yy:02d}{emp_seq[yy]:04d}"

for role, n, yrs, title in [("Admin", 6, (8, 18), None), ("Advisor", 10, (6, 20), None),
                             ("Professor", 5, (5, 15), None), ("Instructor", 5, (17, 24), None)]:
    for _ in range(n):
        f, l = person(rng, 0.4)
        uid = emp_id(rng.randint(*yrs))
        users.append(dict(user_id=uid, role=role, first_name=f, last_name=l,
                          personal_email=pemail(f, l, rng), university_email=uemail(f, l, "aast.edu"),
                          is_email_activated="TRUE"))
        staff[role].append(uid)
NAME = {u["user_id"]: f'{u["first_name"]} {u["last_name"]}' for u in users}
advisors = []
for i, a in enumerate(staff["Advisor"]):
    bld = rng.choice(["Building C", "Building D", "Hall B"])
    advisors.append(dict(advisor_id=a, department="Computer Engineering",
                         office_room=f"{bld} - Room {rng.choice([1,2,3])}{rng.randint(0,2)}{rng.randint(1,9)}"))

# course staff pools
COURSE_PROF = {c: rng.sample(staff["Professor"], 2) for c in COURSES}
COURSE_INSTR = {c: rng.sample(staff["Instructor"], 2) for c in COURSES}

# ------------------------------------------------------------------ population
N_TOTAL = 1000
students = []
queue = {}
def new_sid(intake):
    yy, k = intake[0] % 100, intake[1] + 1
    q = queue.get((yy, k), rng.randint(1000, 4500))
    q += rng.randint(1, 37)
    queue[(yy, k)] = q
    return f"{yy:02d}{k}{q:06d}"

# intake weights: Fall intakes are larger than Spring; older intakes only remain if delayed
INTAKES = [((2026, 0), 125), ((2025, 1), 45), ((2025, 0), 120), ((2024, 1), 42), ((2024, 0), 115),
           ((2023, 1), 40), ((2023, 0), 110), ((2022, 1), 40), ((2022, 0), 108), ((2021, 1), 60),
           ((2021, 0), 60), ((2020, 1), 25), ((2020, 0), 20)]
def build_one():
    while True:
        s_rng = random.Random(rng.random())
        intake = s_rng.choices([i for i, _ in INTAKES], [w for _, w in INTAKES])[0]
        mu = max(45, min(98, s_rng.gauss(79, 10)))
        regs = [(y, k) for y in range(2019, 2027) for k in (0, 1) if key(intake) <= key((y, k)) < key(CURRENT)]
        sh, sha = None, 0
        if len(regs) >= 3 and s_rng.random() < 0.18:      # a bad patch mid-degree (illness, work, family)
            sh, sha = key(s_rng.choice(regs[1:])), s_rng.uniform(8, 18)
        try:
            sim = simulate(intake, mu, 7, 0.3, 0.25, 0.015, 0.05, s_rng, sh, sha)
        except Reject:
            continue          # left the major (3 consecutive probation terms) or stalled
        if sim["credits"] >= 180 or not sim["current"] or not remaining_courses(sim["passed"], sim["latest"], sim["att"]):
            continue          # already graduated -> alumni, not in the active database
        return intake, sim

def category(intake, sim):
    if sim["gpa"] is None: return "newcomer_L1"
    if sim["prob_count"]: return "probation"
    if key(intake) >= key((2025, 1)): return "newcomer_L2"
    if sim["level"] >= 9: return "grad"
    return "regular"

for _ in range(N_TOTAL):
    intake, sim = build_one()
    sid = new_sid(intake)
    f, l = person(rng, 0.3)
    students.append(dict(sid=sid, cat=category(intake, sim), intake=intake, sim=sim, first=f, last=l))
students.sort(key=lambda s: s["sid"])
assert len({s["sid"] for s in students}) == len(students)

# advisor assignment
for i, s in enumerate(students):
    s["advisor"] = staff["Advisor"][rng.randrange(len(staff["Advisor"]))]

for s in students:
    act = "TRUE" if (rng.random() < (0.7 if s["cat"] == "newcomer_L1" else 0.97)) else "FALSE"
    users.append(dict(user_id=s["sid"], role="Student", first_name=s["first"], last_name=s["last"],
                      personal_email=pemail(s["first"], s["last"], rng),
                      university_email=uemail(s["first"], s["last"], "student.aast.edu"),
                      is_email_activated=act))
NAME.update({s["sid"]: f'{s["first"]} {s["last"]}' for s in students})

# ------------------------------------------------------------------ offerings
off_keys = set()
for s in students:
    for a in s["sim"]["att"]:
        off_keys.add((a["course"], a["term"]))
    for a in s["sim"]["current"]:
        off_keys.add((a["course"], CURRENT))
plan_idx = {c: i for i, (c, _) in enumerate(PLAN)}
def ckey(c): return (COURSES[c]["term"], plan_idx.get(c, 999), c)
offerings = {}
by_term = {}
for c, t in off_keys:
    by_term.setdefault(t, []).append(c)
off_rows = []
for t in sorted(by_term, key=key):
    for i, c in enumerate(sorted(by_term[t], key=ckey), 1):
        oid = f"OFF{t[0]}{'FSU'[t[1]]}{i:03d}"
        h = (t[0] * 3 + t[1])
        prof = COURSE_PROF[c][h % 2]
        inst = COURSE_INSTR[c][(h // 2) % 2]
        offerings[(c, t)] = dict(offering_id=oid, course_code=c, professor_id=prof, instructor_id=inst,
                                 semester_term=TERMNAME[t[1]], academic_year=ay(t))
        off_rows.append(offerings[(c, t)])

# ------------------------------------------------------------------ feedback notes
TOPICS = {
    "BA123": ["limits and continuity", "differentiation rules", "applications of derivatives"],
    "BA124": ["integration by parts", "partial fractions", "improper integrals"],
    "BA223": ["multiple integrals", "vector calculus", "series convergence"],
    "BA224": ["Laplace transforms", "second-order ODEs", "systems of ODEs"],
    "BA323": ["probability distributions", "hypothesis testing"],
    "BA326": ["Fourier series", "complex variables"],
    "CC111": ["number systems", "basic algorithms and flowcharts"],
    "CC112": ["loops and nested conditions", "arrays and functions", "pointer tracing"],
    "CC212": ["file handling", "structs and modular design"],
    "CC215": ["linked-list operations", "tree traversal", "stack/queue implementation"],
    "CC216": ["Karnaugh-map simplification", "combinational circuit design", "flip-flop timing"],
    "CC218": ["proof by induction", "graph theory", "relations and functions"],
    "CC317": ["finite state machine design", "HDL modelling"],
    "CC319": ["recursion", "dynamic memory and templates"],
    "CC311": ["pipeline hazards", "cache mapping calculations"],
    "CC316": ["inheritance and polymorphism", "UML class design"],
    "CC331": ["error detection (CRC)", "multiplexing calculations"],
    "CC410": ["assemblers and linkers", "macro processors"],
    "CC414": ["normalisation", "SQL joins and subqueries", "ER modelling"],
    "CC418": ["CPU scheduling calculations", "deadlock detection", "page replacement"],
    "CC421": ["interrupt handling", "memory interfacing"],
    "CC431": ["subnetting", "TCP congestion control", "routing algorithms"],
    "EE231": ["nodal and mesh analysis", "Thevenin equivalents"],
    "EE232": ["AC phasor analysis", "three-phase power"],
    "EC238": ["diode circuits", "BJT biasing"],
    "EC339": ["amplifier frequency response", "op-amp configurations"],
    "EC320": ["AM/FM modulation", "Fourier analysis of signals"],
    "EE418": ["root locus", "Bode plots", "steady-state error"],
    "CC511": ["search algorithms", "knowledge representation"],
}
def feedback(c, s7, f_rng):
    tp = f_rng.choice(TOPICS.get(c, ["the problem-solving section", "the long-answer questions",
                                         "the applied design question"]))
    lost = 30 - s7
    tmpl = f_rng.choice([
        "Lost {lost:g} marks mainly on {tp}; the method was correct but the working was incomplete. Recommend office hours before the 12th-week assessment.",
        "Most marks lost on the {tp} question (left partly unanswered). The rest of the paper was solid; this looks like time management, not understanding.",
        "Lost marks on {tp} because of calculation errors carried through several parts. The approach was right. Advised to review the tutorial sheet.",
        "Weak answer on {tp}, well below the student's usual standard. Student said they were ill during exam week; referred to advisor.",
        "Marks lost on {tp} and one missing sub-question. Discussed with the student; a revision plan was agreed before the 12th-week exam.",
    ])
    return tmpl.format(lost=lost, tp=tp)

# ------------------------------------------------------------------ tables
enr_rows, tr_rows = [], []
eid = tid = 0
for s in students:
    sim = s["sim"]
    s["enr_ref"] = []
    f_rng = random.Random(int(s["sid"]))
    for i, a in enumerate(sim["att"]):
        eid += 1
        c = a["course"]
        fb = "None"
        if a["s7"] is not None and a["prior_gpa"] is not None and a["prior_gpa"] >= Fr(33, 10) and a["s7"] < 18:
            fb = feedback(c, a["s7"], f_rng)
        enr_rows.append(dict(enrollment_id=f"ENR{eid:06d}", offering_id=offerings[(c, a["term"])]["offering_id"],
                             student_id=s["sid"],
                             seventh_week_score="" if a["s7"] is None else f"{a['s7']:.1f}",
                             twelfth_week_score="" if a["s12"] is None else f"{a['s12']:.1f}",
                             prefinal_score="" if a["pf"] is None else f"{a['pf']:.1f}",
                             final_exam_score="" if a["fin"] is None else f"{a['fin']:.1f}",
                             total_score="" if a["total"] is None else f"{a['total']:.1f}",
                             instructor_feedback=fb))
        a["enr_id"] = f"ENR{eid:06d}"
        tid += 1
        status = "Retaken" if a.get("superseded") else ("Failed" if a["grade"] == "F" else "Passed")
        a["status"] = status
        tr_rows.append(dict(transcript_id=f"TR{tid:07d}", student_id=s["sid"], course_code=c,
                            semester_taken=sem_label(a["term"]), grade=a["grade"],
                            points=fmt(pts(a["grade"], COURSES[c]["cr"]), 1), status=status))
    for a in sim["current"]:
        eid += 1
        c = a["course"]
        fb = "None"
        if a["s7"] is not None and a["prior_gpa"] is not None and a["prior_gpa"] >= Fr(33, 10) and a["s7"] < 18:
            fb = feedback(c, a["s7"], f_rng)
        enr_rows.append(dict(enrollment_id=f"ENR{eid:06d}", offering_id=offerings[(c, CURRENT)]["offering_id"],
                             student_id=s["sid"],
                             seventh_week_score="" if a["s7"] is None else f"{a['s7']:.1f}",
                             twelfth_week_score="" if a["s12"] is None else f"{a['s12']:.1f}",
                             prefinal_score="", final_exam_score="", total_score="", instructor_feedback=fb))

def phase(level, credits, gpa, done=False):
    if done and gpa is not None and gpa >= 2: return "Eligible for Graduation"
    if level <= 2: return "Freshman"
    if level <= 4: return "Sophomore"
    if level <= 8: return "Junior"
    return "Senior Year"

stu_rows = []
for s in students:
    sim = s["sim"]
    g = sim["gpa"]
    if g is None:
        st, gp = "Not Yet Classified", ""
    elif sim["prob_count"]:
        st, gp = f"Academic Probation ({['1st', '2nd'][sim['prob_count'] - 1]} Warning)", fmt(g, 1)
    else:
        st, gp = standing(g), fmt(g, 1)
    s["standing"] = st
    stu_rows.append(dict(student_id=s["sid"], major="Computer Engineering", current_semester_level=sim["level"],
                         total_credits_completed=sim["credits"], gpa=gp, standing_status=st,
                         enrollment_phase=phase(sim["level"], sim["credits"], g), advisor_id=s["advisor"]))

course_rows = [dict(course_code=c, course_title=v["title"], credit_hours=v["cr"], course_type=v["type"],
                    recommended_term=(v["term"] if v["type"] == "Required" else "Elective"))
               for c, v in sorted(COURSES.items(), key=lambda kv: ckey(kv[0]))]
pre_rows, pid = [], 0
for r in course_rows:
    c = r["course_code"]
    for p in PREREQ[c]:
        pid += 1
        pre_rows.append(dict(prereq_id=f"PR{pid:03d}", course_code=c, prereq_course_code=p))

# ------------------------------------------------------------------ write
def write(name, rows, cols):
    with open(os.path.join(OUT, name), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, quoting=csv.QUOTE_MINIMAL)
        w.writeheader()
        for r in rows: w.writerow({k: r[k] for k in cols})

role_order = {"Admin": 0, "Advisor": 1, "Professor": 2, "Instructor": 3, "Student": 4}
users.sort(key=lambda u: (role_order[u["role"]], u["user_id"]))
write("Users.csv", users, "user_id,role,first_name,last_name,personal_email,university_email,is_email_activated".split(","))
write("Advisors.csv", advisors, "advisor_id,department,office_room".split(","))
write("Students.csv", stu_rows, "student_id,major,current_semester_level,total_credits_completed,gpa,standing_status,enrollment_phase,advisor_id".split(","))
write("Courses.csv", course_rows, "course_code,course_title,credit_hours,course_type,recommended_term".split(","))
write("Course_Prerequisites.csv", pre_rows, "prereq_id,course_code,prereq_course_code".split(","))
write("Course_Offerings.csv", off_rows, "offering_id,course_code,professor_id,instructor_id,semester_term,academic_year".split(","))
write("Course_Enrollments.csv", enr_rows, "enrollment_id,offering_id,student_id,seventh_week_score,twelfth_week_score,prefinal_score,final_exam_score,total_score,instructor_feedback".split(","))
write("Student_Transcripts.csv", tr_rows, "transcript_id,student_id,course_code,semester_taken,grade,points,status".split(","))

# cohort key (extra helper file)
CAT = {"probation": "Academic Probation (Half Load)", "newcomer_L1": "Newcomer", "newcomer_L2": "Newcomer",
       "grad": "Graduation Year (Semester 9-10)", "regular": "Regular"}
write("Student_Cohorts.csv", [dict(student_id=s["sid"], cohort=CAT[s["cat"]], current_semester_level=s["sim"]["level"],
      intake_term=sem_label(s["intake"])) for s in students],
      ["student_id", "cohort", "current_semester_level", "intake_term"])

# ------------------------------------------------------------------ one transcript file per student
TDIR = os.path.join(OUT, "Student_Transcripts")
os.makedirs(TDIR, exist_ok=True)
TCOLS = ("student_id,student_name,semester_taken,course_code,course_title,credit_hours,grade,points,status,"
         "semester_credit_hours,semester_points,semester_gpa,cumulative_credit_hours,cumulative_points,"
         "cumulative_gpa,credits_completed,standing_after_semester").split(",")
for s in students:
    sim = s["sim"]
    rows = []
    latest = {}
    pc = 0
    terms = sorted({a["term"] for a in sim["att"]}, key=key)
    for t in terms:
        ta = [a for a in sim["att"] if a["term"] == t]
        for a in ta: latest[a["course"]] = a
        sem = [a for a in ta if a["grade"] != "P"]
        sA = sum(COURSES[a["course"]]["cr"] for a in sem)
        sP = sum(pts(a["grade"], COURSES[a["course"]]["cr"]) for a in sem)
        cnt = [a for a in latest.values() if a["grade"] != "P"]
        cA = sum(COURSES[a["course"]]["cr"] for a in cnt)
        cP = sum(pts(a["grade"], COURSES[a["course"]]["cr"]) for a in cnt)
        cg = rnd1(cP / cA) if cA else None
        done = sum(COURSES[c]["cr"] for c, a in latest.items() if a["grade"] != "F")
        if t[1] != 2: pc = pc + 1 if cg < 2 else 0
        elif cg >= 2: pc = 0
        elif pc == 0: pc = 1
        st = f"Academic Probation ({['1st', '2nd'][pc - 1]} Warning)" if pc else standing(cg)
        for a in ta:
            c = a["course"]
            rows.append(dict(student_id=s["sid"], student_name=NAME[s["sid"]], semester_taken=sem_label(t),
                             course_code=c, course_title=COURSES[c]["title"], credit_hours=COURSES[c]["cr"],
                             grade=a["grade"], points=fmt(pts(a["grade"], COURSES[c]["cr"]), 1), status=a["status"],
                             semester_credit_hours=sA, semester_points=fmt(sP, 1),
                             semester_gpa=fmt(rnd1(sP / sA), 1) if sA else "",
                             cumulative_credit_hours=cA, cumulative_points=fmt(cP, 1),
                             cumulative_gpa=fmt(cg, 1), credits_completed=done, standing_after_semester=st))
    assert (cg if terms else None) == sim["gpa"] and (not terms or done == sim["credits"])
    for a in sim["current"]:
        c = a["course"]
        rows.append(dict(student_id=s["sid"], student_name=NAME[s["sid"]], semester_taken=sem_label(CURRENT),
                         course_code=c, course_title=COURSES[c]["title"], credit_hours=COURSES[c]["cr"],
                         grade="", points="", status="In Progress", **{k: "" for k in TCOLS[9:]}))
    write(os.path.join("Student_Transcripts", f"{s['sid']}.csv"), rows, TCOLS)

# ------------------------------------------------------------------ advising tickets (built from the CSVs)
import json
from edubot_context import EduBotDB, SYSTEM_PROMPT
import tickets as TK
db = EduBotDB(OUT)
records = {s["sid"]: db.record(s["sid"]) for s in students}
for s in students:     # parity check: record == Students.csv
    r = records[s["sid"]]["student"]
    assert r["credits_completed"] == s["sim"]["credits"] and r["semester_level"] == s["sim"]["level"]
    assert (r["cumulative_gpa"] or "") == next(x["gpa"] for x in stu_rows if x["student_id"] == s["sid"])
tks = TK.generate(db, records, random.Random(SEED + 11), per_student=6)
qa_rows = []
for t in tks:
    for i, (role, msg) in enumerate(t["turns"], 1):
        qa_rows.append(dict(ticket_id=t["ticket_id"], student_id=t["student_id"], intent_category=t["intent"], turn_no=i,
                            role="student" if role == "user" else "edubot", message=msg, ticket_status=t["status"],
                            assigned_to=t["assigned_to"]))
write("Advising_QA.csv", qa_rows, "ticket_id,student_id,intent_category,turn_no,role,message,ticket_status,assigned_to".split(","))

TD = os.path.join(OUT, "EduBot_Training")
os.makedirs(TD, exist_ok=True)
with open(os.path.join(TD, "student_records.jsonl"), "w", encoding="utf-8") as fh:
    for sid, r in records.items():
        fh.write(json.dumps(EduBotDB.public(r), ensure_ascii=False) + "\n")
with open(os.path.join(TD, "tickets.jsonl"), "w", encoding="utf-8") as fh:
    for t in tks:
        fh.write(json.dumps(dict(ticket_id=t["ticket_id"], student_id=t["student_id"], term=t["term"], academic_week=t["academic_week"],
                                 channel=t["channel"], intent=t["intent"], priority=t["priority"], status=t["status"],
                                 assigned_to=t["assigned_to"], data_used=t["data_used"],
                                 messages=[dict(role=r_, content=m) for r_, m in t["turns"]]), ensure_ascii=False) + "\n")
# train / validation / test split by student (no student appears in two splits)
sids = sorted(records); rs = random.Random(SEED + 3); rs.shuffle(sids)
split = {sid: ("train" if i < 0.8 * len(sids) else "validation" if i < 0.9 * len(sids) else "test") for i, sid in enumerate(sids)}
fhs = {k: open(os.path.join(TD, f"sft_{k}.jsonl"), "w", encoding="utf-8") for k in ("train", "validation", "test")}
sysmsg = {sid: EduBotDB.system_from_record(r) for sid, r in records.items()}
for t in tks:
    msgs = [dict(role="system", content=sysmsg[t["student_id"]])] + [dict(role=r_, content=m) for r_, m in t["turns"]]
    fhs[split[t["student_id"]]].write(json.dumps(dict(ticket_id=t["ticket_id"], intent=t["intent"], messages=msgs), ensure_ascii=False) + "\n")
for f in fhs.values(): f.close()
print("students", len(students), "enrollments", len(enr_rows), "transcripts", len(tr_rows),
      "offerings", len(off_rows), "tickets", len(tks), "qa rows", len(qa_rows), "users", len(users))
