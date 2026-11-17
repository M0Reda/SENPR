"""EduBot advising tickets: multi-turn, record-grounded conversations for model training.
Every assistant answer is computed from the same STUDENT RECORD that the model sees in its prompt."""
from fractions import Fraction as Fr
from edubot_context import (pts, rnd1, g1, PTS3, GRADE_ORDER, standing_of, regular_semesters_between,
                            level_of, NEXT_TERMS, CURRENT_TERM)

NEXT = NEXT_TERMS[0]
TARGETS = [("pass the course", 60, "D"), ("get at least a C", 70, "C"), ("get a B", 80, "B"),
           ("get a B+", 84, "B+"), ("get an A", 93, "A")]

def f3(x): return f"{float(x):.3f}"


class Ctx:
    def __init__(self, db, rec, rng):
        self.db, self.rec, self.rng = db, rec, rng
        self.st = rec["student"]
        self.I = rec["_internal"]
        self.cur = rec["current_courses"]
        self.adv = rec["advisor"]

    def pick(self, xs): return self.rng.choice(xs)
    def hi(self): return self.pick(["", "", "Hi, ", "Hello, ", "Hey, ", "Good morning, ", "Good evening, ", "Hi EduBot, ", "Salam, "])
    def thanks(self): return self.pick(["", "", "", " Thanks!", " Thank you.", " Thanks in advance.", " pls help"])
    def T(self, c): return self.db.title(c)
    def CR(self, c): return self.db.cr(c)
    def name_first(self): return self.st["name"].split()[0]


def ticket(intent, turns, data, status="Resolved", priority="Normal", assigned="EduBot"):
    return dict(intent=intent, turns=turns, data_used=data, status=status, priority=priority, assigned_to=assigned)

# ======================================================================= helpers
def graded_current(x):
    return x.cr(x.cur[0]["code"]) if x.cur else 0

def projection(x, grade_for):
    """grade_for: code -> letter.  Returns (semP, semA, newP, newA) applying repeat caps & replacement."""
    I = x.I
    P, A = I["P"], I["A"]
    sP = sA = 0
    rm_P = 0
    add_A = 0
    for c in x.cur:
        code = c["code"]
        if x.CR(code) == 0:
            continue
        g = grade_for[code]
        prev = I["latest"].get(code)
        if prev is not None and prev["grade"] == "F" and GRADE_ORDER.index(g) > GRADE_ORDER.index("B+"):
            g = "B+"
        p = pts(g, x.CR(code))
        sP += p; sA += x.CR(code)
        if prev is not None and prev["grade"] != "P":
            rm_P += pts(prev["grade"], x.CR(code))
        else:
            add_A += x.CR(code)
    return sP, sA, P - rm_P + sP, A + add_A

def need_final(so_far, pf, target):
    return max(Fr(12), Fr(target) - Fr(so_far) - Fr(pf))

def course_line(x, c):
    return f"{c} ({x.T(c)})"

# ======================================================================= intents
def probation_load(x):
    if not x.I["on_prob"]: return None
    want = x.pick([15, 16, 17, 18, 18, 21])
    q = x.pick([
        f"Why is the registration portal refusing to let me register {want} credit hours? It keeps saying max load reached.",
        f"{x.hi()}I tried to add {want} credits and the portal says 'maximum load exceeded'. Is this a bug?",
        f"why can't I take {want} hours this term?? my friends all registered 18",
        f"{x.hi()}I could not register more than 12 credit hours this semester although I need to catch up. Could you explain why?",
        f"the system blocked me at 12 credits, I need {want} so I don't fall more behind. can you override it",
    ])
    st = x.st
    regs = ", ".join(f"{c['code']} ({c['credit_hours']} cr)" for c in x.cur if c["credit_hours"])
    rep = [c["code"] for c in x.cur if c["attempt"].startswith("repeat")]
    a = (f"It isn't a portal error. Your cumulative GPA is {st['cumulative_gpa']} ({st['cumulative_points']} points ÷ {st['cumulative_credit_hours']} credit hours), "
         f"which is below 2.0, so you are on Academic Probation (warning {st['consecutive_probation_warnings']} in a row). "
         f"Students on probation can register at most 12 credit hours in a Fall or Spring semester, and registration is done with your academic advisor ({x.adv['name']}), not directly in the portal. "
         f"I can't override this limit. This term you are registered for {x.rec['registration_rules_now']['registered_this_term']} credit hours: {regs}.")
    if rep:
        a += f" You're repeating {', '.join(rep)}. Passing these replaces the F grades in your GPA, which is the fastest way back above 2.0."
    a += " Once your cumulative GPA is 2.0 or higher, the normal 18-credit limit applies again."
    t = [("user", q), ("assistant", a)]
    if x.rng.random() < 0.35:
        t += [("user", x.pick(["ok so how do I book the session with my advisor?", "where can I find my advisor?", "who is my advisor again?"])),
              ("assistant", f"Your advisor is {x.adv['name']}, office {x.adv['office']}, email {x.adv['email']}. Email them to book a registration session, and bring your current marks so you can plan your next term together.")]
    return ticket("Probation Load Restriction", t, ["student", "registration_rules_now", "current_courses", "advisor"], priority="High")

def probation_consequences(x):
    if not x.I["on_prob"]: return None
    st = x.st
    w = st["consecutive_probation_warnings"]
    q = x.pick([
        f"This is my {'second' if w == 2 else 'first'} semester on academic probation and my midterms aren't great. What happens if I don't get above 2.0 this term?",
        "I'm scared. what's the worst case if my GPA stays under 2?",
        "Is it true they move you out of computer engineering if you stay on probation?",
        f"{x.hi()}I would like to understand the consequences of remaining on academic probation.",
    ])
    sP_needed = None
    graded = [c for c in x.cur if c["credit_hours"]]
    calc = ""
    if graded:
        P, A = x.I["P"], x.I["A"]
        rmP = 0; addA = 0; c_new = 0
        for c in graded:
            prev = x.I["latest"].get(c["code"])
            c_new += c["credit_hours"]
            if prev is not None: rmP += pts(prev["grade"], c["credit_hours"])
            else: addA += c["credit_hours"]
        need = (Fr(195, 100) * (A + addA) - (P - rmP)) / c_new
        if need <= 0:
            calc = " Your current registrations would be enough to get back to 2.0 if you pass them all (and score at least 12/40 on every final)."
        elif need <= 4:
            calc = (f" To get your cumulative GPA back to 2.0 (an exact value of 1.95 or more rounds up to 2.0), you need a term GPA of about {float(need):.2f} "
                    f"across your {c_new} registered credit hours. That's roughly a {nearest(need)} average.")
            if any(c["attempt"].startswith("repeat") for c in graded):
                calc += " Remember that repeated failed courses are capped at B+."
        else:
            calc = (f" Being honest about the arithmetic: with {c_new} registered credit hours you can't reach 2.0 this term (you would need a term GPA of {float(need):.2f}, above the 4.0 maximum). "
                    f"Please meet your advisor soon to plan the next terms.")
    if w >= 2:
        head = ("This is your 2nd consecutive probation warning. If your cumulative GPA is still below 2.0 at the end of this term, that will be a 3rd consecutive probation term, "
                "which means a mandatory Change of Major. (Three more consecutive probation terms in the new major lead to dismissal.)")
    else:
        head = ("You're on your 1st probation warning. The rule is that 3 consecutive probation terms trigger a mandatory Change of Major, "
                "and 3 more consecutive probation terms in the new major lead to dismissal. You still have time, but every term counts.")
    a = (f"Your cumulative GPA is {st['cumulative_gpa']}. " + head + calc +
         f" While you're on probation you're limited to 12 credit hours, registered with {x.adv['name']}, and you can repeat as many courses as needed to raise your GPA. Keep attendance above 85% so no course turns into a forced withdrawal.")
    return ticket("Probation Consequences", [("user", q), ("assistant", a)],
                  ["student", "current_courses", "repeats"], priority="High")

def nearest(x):
    return min((g for g in GRADE_ORDER if g != "F"), key=lambda g: abs(PTS3[g] / 3 - x))

def grad_audit(x):
    st = x.st
    if st["cumulative_gpa"] is None and x.rng.random() < 0.7: return None
    rem = x.rec["remaining"]
    q = x.pick(["Can you run a graduation audit for me? What do I still need?",
                "how many credits until I graduate? and which courses are left",
                f"{x.hi()}please confirm what remains for me to graduate in Computer Engineering.",
                "am I on track to graduate? I lost count of what I took",
                "what's left in my degree?"])
    cur_cr = sum(c["credit_hours"] for c in x.cur if c["code"] not in x.I["passed"])
    req = [r for r in x.I["remaining_req"]]
    req_cr = sum(r["credit_hours"] for r in req)
    left_after = req_cr + 3 * rem["electives_still_needed"]
    lst = "; ".join(f"{r['code']} {r['title']} ({r['credit_hours']} cr)" for r in req if r["credit_hours"]) or "none"
    gp = st["cumulative_gpa"]
    gline = (f"Your cumulative GPA is {gp}, so the GPA condition (2.0 or higher) is currently met." if gp and Fr(gp) >= 2 else
             f"Your cumulative GPA is {gp}, below the 2.0 needed to graduate, so raising it is part of your graduation plan." if gp else
             "You don't have a GPA yet because this is your first term.")
    a = (f"To graduate you need 180 credit hours with a cumulative GPA of at least 2.0. You've completed {st['credits_completed']} credit hours ({st['credits_remaining']} to go), and you're in semester {st['semester_level']}. {gline} "
         f"This term you're taking {cur_cr} new credit hours ({', '.join(c['code'] for c in x.cur)}).")
    if left_after:
        a += f" If you pass them all, {left_after} credit hours will remain: required courses: {lst}"
        if rem["electives_still_needed"]:
            a += f"; plus {rem['electives_still_needed']} elective(s) of 3 credit hours each"
        a += "."
    if any(r["code"] == "IM400CC" for r in req):
        a += " Practical Training (IM400CC, 0 cr, P/U) is also required in your final term."
    if left_after == 0:
        a += " These are your last courses: pass them all and you'll have completed the 180 credit hours and graduate at the end of this term. Congratulations in advance!"
    t = [("user", q), ("assistant", a)]
    if left_after and x.rng.random() < 0.4:
        p = x.rec["graduation_plan_projection"]
        t += [("user", x.pick(["can you turn that into a semester-by-semester plan?", "ok and when would I graduate?", "make me a plan pls"])),
              ("assistant", plan_text(x, p))]
    return ticket("Graduation Requirements Audit", t, ["student", "remaining", "current_courses", "graduation_plan_projection"])

def plan_text(x, p):
    if not p["terms"]:
        return "I couldn't build a plan from your record. Please check with your advisor."
    lines = [f"{t['term']}: {', '.join(t['courses'])} = {t['credit_hours']} cr" for t in p["terms"]]
    s = "Here's a plan that respects every prerequisite and your load limit, assuming you pass all your current courses:\n" + "\n".join(f"- {l}" for l in lines)
    if p["expected_graduation_term"]:
        s += f"\nExpected graduation: end of {p['expected_graduation_term']}."
    if x.I["on_prob"]:
        s += "\nThe first term is capped at 12 credit hours because you're on probation. The later terms assume your GPA is back to 2.0 or higher."
    if len(p["terms"]) > 1:
        s += " Taking courses in summer (up to 2 courses) could shorten this."
    s += " Electives can be swapped for any other elective whose prerequisites you meet. Please confirm with your advisor before registration."
    return s

def grad_plan(x):
    p = x.rec["graduation_plan_projection"]
    if not p["terms"]: return None
    q = x.pick(["Can you make me a graduation plan?", "what should I take each semester until I graduate?",
                f"{x.hi()}I'd like a semester-by-semester plan to finish my degree.", "when will I graduate if everything goes fine?",
                "plan my remaining semesters please"])
    t = [("user", q), ("assistant", plan_text(x, p))]
    if x.rng.random() < 0.3 and len(p["terms"]) > 1:
        t += [("user", "can I finish faster?"),
              ("assistant", faster_text(x, p))]
    return ticket("Graduation Plan", t, ["student", "remaining", "graduation_plan_projection", "registration_rules_now"])

def faster_text(x, p):
    gp = x.st["cumulative_gpa"]
    opts = ["take up to 2 courses in each summer semester (the plan above uses Fall and Spring only)"]
    if gp and (Fr(gp) >= 3):
        opts.append("register up to 21 credit hours per semester, which your GPA of 3.0 or higher allows")
    elif gp and Fr(gp) > 2:
        opts.append("register up to 21 credit hours once you reach your final year (semesters 9-10), since your GPA is above 2.0")
    s = "There are two levers: " + "; and ".join(opts) + "."
    s += " Prerequisite chains still set a minimum pace. For example, Senior Project II (CC503) can only come after Senior Project I (CC501). Your advisor can check which of your remaining courses run in summer."
    return s

def next_term(x):
    p = x.rec["graduation_plan_projection"]
    if not p["terms"]: return None
    t0 = p["terms"][0]
    q = x.pick([f"What should I register next semester?", "which courses should I take in spring?",
                f"{x.hi()}can you recommend my courses for {NEXT}?", "what can I register next term?"])
    rule = x.rec["registration_rules_now"]
    a = (f"Assuming you pass your current courses, a good registration for {t0['term']} is: {', '.join(t0['courses'])}, {t0['credit_hours']} credit hours in total. "
         f"All their prerequisites will be complete by then. Your current limit is {rule['max_credit_hours_fall_spring']} credit hours ({rule['reason']}).")
    rep = x.rec["repeats"]["outstanding_failed_courses"]
    rep_not_now = [c for c in rep if c not in x.I["current_codes"]]
    if rep_not_now:
        a += f" Failed courses must be repeated, so {', '.join(rep_not_now)} take priority."
    a += " If you fail anything this term, the plan changes, especially for any course that is a prerequisite. Let's review it again when the results are out."
    return ticket("Next Term Registration", [("user", q), ("assistant", a)], ["current_courses", "remaining", "graduation_plan_projection", "registration_rules_now"])

def overload(x):
    gp = x.st["cumulative_gpa"]
    if gp is None or x.I["on_prob"]: return None
    G = Fr(gp); lvl = x.st["semester_level"]
    q = x.pick(["Can I register 21 credit hours next semester to graduate faster?", "my GPA is decent, can I take 21 hours next term instead of 18?",
                f"{x.hi()}I'd like permission to register 21 credit hours next term. Am I eligible?", "is overload allowed for me?"])
    if G >= 3:
        a = f"Yes. Your cumulative GPA is {gp}, and students with a GPA of 3.0 or higher may register up to 21 credit hours in a Fall or Spring semester."
    elif lvl >= 9 and G > 2:
        a = f"Yes. You're in semester {lvl} (final year) with a GPA of {gp}, which is above 2.0, so you may register up to 21 credit hours."
    else:
        a = (f"Not at the moment. Up to 21 credit hours is allowed with a GPA of 3.0 or higher, or for final-year students (semesters 9-10) with a GPA above 2.0. "
             f"You're in semester {lvl} with a GPA of {gp}, so your maximum is 18.")
        if lvl >= 7:
            a += " Once you reach semester 9 with a GPA above 2.0, the 21-hour option opens."
    a += " Every course still needs its prerequisites completed first."
    return ticket("Overload Permission", [("user", q), ("assistant", a)], ["student", "registration_rules_now"])

def summer_exception(x):
    rem = x.rec["remaining"]
    req = [r for r in x.I["remaining_req"] if r["credit_hours"]]
    n = len(req) + rem["electives_still_needed"]
    if x.st["semester_level"] < 8 or n < 2 or n > 5: return None
    crs = sum(r["credit_hours"] for r in req) + 3 * rem["electives_still_needed"]
    q = x.pick([f"After this term I'll have {n} courses left ({crs} credit hours). Can I take them all in the summer or do I have to wait till fall?",
                f"if I pass everything now I have {n} subjects left. can I do them all in summer?",
                f"{x.hi()}does the summer overload exception apply to me?"])
    names = [r["code"] for r in req] + ["elective"] * rem["electives_still_needed"]
    chain = [(r["code"], p) for r in req for p in r["prerequisites"] if p in [z["code"] for z in req]]
    ch = (" Note: " + "; ".join(f"{c} requires {p}, so they can't be in the same term" for c, p in chain) + ".") if chain else ""
    if n <= 2:
        a = f"Yes. The normal summer limit is 2 courses, so your remaining {n} courses ({', '.join(names)}) fit within it, provided they're offered in summer and the prerequisites are met.{ch}"
    elif n == 3:
        a = (f"The normal summer limit is 2 courses. However, a graduating senior who is behind schedule can be approved for 3 courses in the final summer to complete the 180 credit hours. "
             f"You'd have exactly 3 left ({', '.join(names)}), so you can apply through your advisor {x.adv['name']}. Approval also depends on the courses being offered in summer.{ch}")
    else:
        a = (f"Not all of them. Even with the graduating-senior exception, the summer maximum is 3 courses, and you'd have {n} left ({crs} credit hours). "
             f"A realistic option is to take them in {NEXT} (up to 21 credit hours if your GPA is above 2.0 and you're in your final year), or to split them between Spring and Summer.{ch}")
    return ticket("Summer Graduating Overload Exception", [("user", q), ("assistant", a)], ["remaining", "student"])

def retake_cap(x):
    failed = x.rec["repeats"]["outstanding_failed_courses"]
    capped = [r for r in x.I["rows"] if r["grade"] == "B+" and any(o["course_code"] == r["course_code"] and o["grade"] == "F" and o is not r for o in x.I["rows"])]
    if not failed and not capped: return None
    if capped and (not failed or x.rng.random() < 0.4):
        r = capped[-1]; c = r["course_code"]
        q = x.pick([f"Why is my {c} grade a B+? I did really well when I repeated it.",
                    f"I repeated {x.T(c)} and I'm sure I scored above 90, so why does my transcript show B+?"])
        a = (f"{c} ({x.T(c)}) was a repeat of a course you had failed (F), and the grade for a repeated failed course is capped at B+ whatever the score. "
             f"It's recorded as B+ = {g1(pts('B+', x.CR(c)))} points for {x.CR(c)} credit hours, and that grade replaces the F in your cumulative GPA.")
        return ticket("Retake Grade Cap", [("user", q), ("assistant", a)], ["transcript"])
    t = []
    if len(failed) > 1 and x.rng.random() < 0.4:
        t += [("user", x.pick(["If I retake the course I failed and get an A+, will it show as A+?", "what's the max grade on a retake?"])),
              ("assistant", f"Quick check: you have more than one failed course on record ({', '.join(failed)}). Which one do you mean? The rule is the same for all of them, but I'll give you the exact numbers.")]
        c = x.pick(failed)
        t += [("user", x.pick([c, f"{c}", f"the {x.T(c)} one", f"{c} please"]))]
    else:
        c = x.pick(failed)
        sc = x.pick([95, 97, 98, 100])
        t += [("user", x.pick([f"If I failed {c} and I study super hard and get {sc}%, will my transcript show an A+?",
                                f"I'm retaking {x.T(c)}. if I ace it will the F become an A+?",
                                f"{x.hi()}what's the highest grade I can get when I repeat {c}?"]))]
    now = c in x.I["current_codes"]
    a = (f"No. Because {c} ({x.T(c)}) was failed, the grade for the repeat is capped at B+, even with a near-perfect score. "
         f"For {x.CR(c)} credit hours that's at most {g1(pts('B+', x.CR(c)))} points (an A+ would have been {g1(pts('A+', x.CR(c)))}). "
         f"The new grade replaces the F in your cumulative GPA, so passing still helps a lot."
         + (" You're repeating it this term." if now else " You must register it again, as failed courses have to be repeated.")
         + " You'll also need at least 12/40 on the final exam to pass.")
    t.append(("assistant", a))
    return ticket("Retake Grade Cap", t, ["repeats", "transcript", "current_courses"])

def improve_window(x):
    rows = x.I["rows"]; latest = x.I["latest"]
    cands = []
    for c, r in latest.items():
        if r["grade"] in ("D", "D+", "C-") and x.CR(c) and c not in x.I["current_codes"]:
            earlier_F = any(o["course_code"] == c and o["grade"] == "F" and o is not r for o in rows)
            if not earlier_F:
                cands.append(r)
    if not cands: return None
    r = x.pick(cands); c = r["course_code"]
    el = regular_semesters_between(r["semester_taken"], NEXT)
    q = x.pick([f"I got a {r['grade']} in {c}. Can I retake it next semester to boost my GPA?",
                f"is it too late to repeat {x.T(c)} to improve my {r['grade']}?",
                f"{x.hi()}I'd like to repeat {c} ({r['grade']}) for grade improvement. Is that possible?"])
    left = x.rec["repeats"]["improvement_repeats_left"]
    if el <= 3 and (left is None or left > 0):
        quota = ("you're on probation, so there's no limit on the number of repeats (under advisor supervision)" if left is None
                 else f"you have {left} of your 5 improvement repeats left")
        a = (f"Yes. You completed {c} in {r['semester_taken']}, and {NEXT} would be semester {el} after that, inside the 3-semester improvement window. Also, {quota}. "
             f"The new grade replaces the {r['grade']} in your cumulative GPA.")
    elif el > 3:
        a = (f"No, it's too late for that course. Improvement repeats must be taken within 3 semesters of first completing the course. You completed {c} in {r['semester_taken']}, "
             f"so {NEXT} would be semester {el} after it, outside the window. The {r['grade']} stays, so your best GPA lever now is your remaining courses.")
    else:
        a = "No. You've already used all 5 improvement repeats allowed for regular students."
    return ticket("Course Improvement Window Policy", [("user", q), ("assistant", a)], ["transcript", "repeats"])

def prereq_question(x):
    rem = x.I["remaining_req"] + [
        dict(code=e.split()[0], title=" ".join(e.split()[1:]), prerequisites=x.db.prereq.get(e.split()[0], []),
             prerequisites_met_if_current_courses_passed=True) for e in x.rec["remaining"]["electives_eligible_after_this_term"][:3]]
    rem = [r for r in rem if r["prerequisites"] and r["code"] != "IM400CC"]
    if not rem: return None
    blocked = [r for r in rem if not r["prerequisites_met_if_current_courses_passed"]]
    failed = set(x.rec["repeats"]["outstanding_failed_courses"])
    hard = [r for r in blocked if any(p in failed and p not in x.I["current_codes"] for p in r["prerequisites"])]
    r = x.pick(hard or blocked or rem) if (hard or blocked) and x.rng.random() < 0.7 else x.pick(rem)
    c = r["code"]
    t = []
    if x.rng.random() < 0.3:
        t += [("user", x.pick(["can I take it next term?", "am I allowed to register that course?", "can I register the next course in the chain?"])),
              ("assistant", "Which course do you mean? Send me the course code or name, and I'll check its prerequisites against your record.")]
        t += [("user", x.pick([c, f"{c} {r['title']}", r["title"]]))]
    else:
        miss = [p for p in r["prerequisites"] if p not in x.I["passed"]]
        if miss and any(p in failed for p in miss) and x.rng.random() < 0.6:
            p = [p for p in miss if p in failed][0]
            t += [("user", x.pick([f"Hey, can you open {c} ({r['title']}) for me? I failed {p} but I really want to take them together.",
                                   f"can I take {c} and {p} at the same time? I'm retaking {p} anyway",
                                   f"is there a prerequisite override for {r['title']}? I didn't pass {x.T(p)} but I can handle both"]))]
        else:
            t += [("user", x.pick([f"Can I register {c} next semester?", f"am I eligible for {r['title']} in spring?",
                                   f"{x.hi()}what are the prerequisites for {c} and do I have them?"]))]
    pre = r["prerequisites"]
    status = []
    for p in pre:
        if p in x.I["passed"]:
            status.append(f"{p} ({x.T(p)}): passed")
        elif p in x.I["current_codes"]:
            status.append(f"{p} ({x.T(p)}): registered this term, not passed yet")
        elif p in failed:
            status.append(f"{p} ({x.T(p)}): failed, must be repeated")
        else:
            status.append(f"{p} ({x.T(p)}): not taken yet")
    ok_now = all(p in x.I["passed"] for p in pre)
    ok_after = all(p in x.I["passed"] or p in x.I["current_codes"] for p in pre)
    head = f"{c} ({r['title']}) requires: " + "; ".join(status) + ". "
    if ok_now:
        a = head + f"All prerequisites are complete, so you can register {c} next semester (within your credit-hour limit)."
    elif ok_after:
        a = head + f"If you pass the prerequisite(s) you're taking this term, {c} opens for {NEXT}. If you don't pass, it stays blocked, because prerequisites can't be taken in the same term as the course that needs them."
    else:
        a = head + f"So you can't register {c} yet. Prerequisites are hard requirements: they can't be bypassed or taken in the same term. Complete the missing one(s) first, and {c} opens the following term."
    t.append(("assistant", a))
    return ticket("Prerequisite Dependency Check", t, ["remaining", "transcript", "current_courses"])

def honors(x):
    gp = x.st["cumulative_gpa"]
    if gp is None: return None
    h = x.rec["honors_check"]
    q = x.pick([f"My GPA is {gp}. Can I still graduate with honors?", "what do I need for an honors degree? am I still eligible?",
                f"{x.hi()}am I on track for honors?"])
    a = "An honors degree needs a semester GPA of at least 3.3 in every semester and no F grade at any time. Your overall GPA alone doesn't decide it. "
    if h["f_grades"]:
        a += f"Your record has an F in {h['f_grades'][0]}, and even if it was repeated later, it rules out honors."
    elif h["semesters_below_3_3"]:
        term = h["semesters_below_3_3"][0]
        sg = [s for s in x.rec["semester_history"] if s["term"] == term][0]["semester_gpa"]
        a += f"Your semester GPA in {term} was {sg}, below 3.3, so the every-semester condition is no longer met."
    else:
        low = min(Fr(s["semester_gpa"]) for s in x.rec["semester_history"] if s["semester_gpa"])
        a += f"So far you meet both conditions: no F grades, and every semester GPA is 3.3 or higher (your lowest is {g1(low)}). Keep at least 3.3 this term and every term after, with no F."
    return ticket("Honors Eligibility Criteria", [("user", q), ("assistant", a)], ["honors_check", "semester_history"])

def mark_breakdown(x):
    q = x.pick(["How is the 100% total grade split up for our subjects? How much does the TA control?",
                "first year here, how are marks distributed in a course? is the final everything?",
                f"{x.hi()}could you explain the assessment breakdown and the instructor's role in grading?",
                "what's the mark distribution? midterm vs final"])
    a = ("Every course is out of 100: 7th-week assessment 30, 12th-week assessment 20, pre-final 10 and final exam 40. "
         "How the marks are split between the professor and the instructor (lecture, tutorial, lab) is set by the course professor and announced in the first week. "
         "Important: you need at least 12/40 on the final exam to pass, whatever your coursework total.")
    ex = [c for c in x.cur if "seventh_week" in c]
    if ex:
        c = x.pick(ex)
        a += f" For example, in {c['code']} ({c['title']}) you have {c['seventh_week']:g}/30 + {c['twelfth_week']:g}/20 = {c['marks_so_far']} so far, with the pre-final (10) and the final (40) still to come."
    return ticket("Course Assessment Mark Breakdown", [("user", q), ("assistant", a)], ["current_courses"])

def standing(x):
    st = x.st
    if st["cumulative_gpa"] is None: return None
    P, A = x.I["P"], x.I["A"]
    q = x.pick([f"My cumulative points are {g1(P)} over {A} credit hours. What's my standing?",
                "what's my academic standing right now and what are the cutoffs?",
                f"{x.hi()}kindly confirm my current academic classification.", "am I Good or Very Good?"])
    a = (f"{g1(P)} points ÷ {A} credit hours = {f3(P / A)}, which rounds (half-up, one decimal) to {st['cumulative_gpa']}. "
         f"Bands: Excellent 3.7-4.0, Very Good 3.3-3.6, Good 2.7-3.2, Pass 2.0-2.6, and below 2.0 is Academic Probation. "
         f"Your standing is {st['standing']}.")
    t = [("user", q), ("assistant", a)]
    G = Fr(st["cumulative_gpa"])
    nxt = [(Fr(2), "Pass"), (Fr(27, 10), "Good"), (Fr(33, 10), "Very Good"), (Fr(37, 10), "Excellent")]
    up = [(b, n) for b, n in nxt if G < b]
    if up and x.rng.random() < 0.4:
        b, n = up[0]
        need_P = (b - Fr(5, 100)) * A - P
        t += [("user", x.pick([f"how far am I from {n}?", f"what do I need to reach {n}?"])),
              ("assistant", f"{n} starts at {g1(b)}, and an exact GPA of {float(b - Fr(5, 100)):.2f} or more rounds up to {g1(b)}. On your current {A} credit hours you'd need about {float(need_P):.1f} more points. "
                            f"In practice your new courses add both points and credit hours, so strong grades this term (and any repeats of low grades, which replace the old grade) are what move you up. I can project this term's GPA if you tell me the grades you expect.")]
    return ticket("Standing Classification", t, ["student"])

def gpa_method(x):
    hist = x.rec["semester_history"]
    if not hist:
        q = x.pick(["how will my GPA be calculated?", "how does GPA work here? first semester"])
        cur = [c for c in x.cur if c["credit_hours"]]
        tot = sum(c["credit_hours"] for c in cur)
        a = ("Each course gives points from its letter grade. For a 3-credit course: A+ 12, A 11.5, A- 11, B+ 10, B 9, B- 8, C+ 7, C 6, C- 5, D+ 4, D 3, F 0. "
             "Other credit hours scale by ratio, rounded to 1 decimal (an A in a 2-credit course = 7.7). Semester GPA = total points ÷ total credit hours taken, rounded half-up to 1 decimal. "
             f"This term you have {tot} credit hours, so your semester GPA will be your total points ÷ {tot}. For example, a B in every course would give exactly 3.0.")
        return ticket("GPA Calculation Methodology", [("user", q), ("assistant", a)], ["current_courses"])
    h = hist[-1]
    rows = [r for r in x.I["rows"] if r["semester_taken"] == h["term"] and r["grade"] != "P"]
    parts = "; ".join(f"{r['course_code']} ({x.CR(r['course_code'])} cr) {r['grade']} = {g1(pts(r['grade'], x.CR(r['course_code'])))}" for r in rows)
    q = x.pick([f"How exactly was my GPA for {h['term']} calculated?", "can you explain how semester GPA vs cumulative GPA works",
                "how do you get the GPA from the letter grades? is it just an average?", f"{x.hi()}I want to verify my last semester GPA."])
    a = (f"Course points come from the grade (for 3 credit hours: A+ 12, A 11.5, A- 11, B+ 10, B 9, B- 8, C+ 7, C 6, C- 5, D+ 4, D 3, F 0, and other credit hours scale by ratio, rounded to 1 decimal). "
         f"Semester GPA = total points ÷ credit hours taken. For {h['term']}: {parts}. Total {h['points']} ÷ {h['credit_hours']} = {f3(Fr(h['points']) / h['credit_hours'])}, rounded to {h['semester_gpa']}. "
         f"Your cumulative GPA uses the same formula over all terms, with a repeated course counting only its latest attempt: {x.st['cumulative_points']} ÷ {x.st['cumulative_credit_hours']} = "
         f"{f3(x.I['P'] / x.I['A'])}, rounded to {x.st['cumulative_gpa']}.")
    return ticket("GPA Calculation Methodology", [("user", q), ("assistant", a)], ["semester_history", "transcript"])

def gpa_projection(x):
    cur = [c for c in x.cur if c["credit_hours"]]
    if not cur: return None
    t = []
    g = x.pick(["A", "B+", "B", "B", "C+", "C"])
    if x.rng.random() < 0.3:
        t += [("user", x.pick(["what will my GPA be after this term?", "can you predict my GPA?"])),
              ("assistant", "I can project it. Your final grades aren't out yet, so tell me the grades you expect in each course (or one grade for all of them) and I'll calculate it.")]
        t += [("user", x.pick([f"say {g} in everything", f"{g} in all", f"assume {g} for all courses"]))]
    else:
        t += [("user", x.pick([f"If I get {g} in all my courses this term, what will my GPA be?", f"what's my GPA if I get {g}s this semester?",
                               f"{x.hi()}could you project my cumulative GPA assuming {g} in every course?"]))]
    t.append(("assistant", projection_text(x, g)))
    if x.rng.random() < 0.35:
        g2 = x.pick([z for z in ["A", "B+", "B", "C+", "C", "D+"] if z != g])
        t += [("user", x.pick([f"and if I get {g2} instead?", f"what about {g2}s?"])), ("assistant", projection_text(x, g2, short=True))]
    return ticket("GPA Projection", t, ["student", "current_courses", "transcript"])

def projection_text(x, g, short=False):
    cur = [c for c in x.cur if c["credit_hours"]]
    sP, sA, nP, nA = projection(x, {c["code"]: g for c in cur})
    caps = [c["code"] for c in cur if c["attempt"].startswith("repeat") and GRADE_ORDER.index(g) > GRADE_ORDER.index("B+")]
    reps = [c["code"] for c in cur if not c["attempt"].startswith("first")]
    s = (f"With {g} in all {len(cur)} courses: this term = {g1(sP)} points ÷ {sA} credit hours = {f3(sP / sA)} → semester GPA {g1(rnd1(sP / sA))}. "
         f"Cumulative = {g1(nP)} ÷ {nA} = {f3(nP / nA)} → {g1(rnd1(nP / nA))} ({standing_of(rnd1(nP / nA))})")
    if x.st["cumulative_gpa"]:
        s += f", compared with {x.st['cumulative_gpa']} now."
    else:
        s += "."
    if caps:
        s += f" {', '.join(caps)} counted as B+ because the repeat of a failed course is capped."
    if reps and not short:
        s += f" Repeated courses ({', '.join(reps)}) replace their earlier grade, so they don't add new credit hours to the cumulative total."
    if not short:
        s += " This is a projection, not your official result."
    return s

def required_final(x):
    cur = [c for c in x.cur if "seventh_week" in c]
    if not cur: return None
    c = x.pick(cur)
    t = []
    if x.rng.random() < 0.3:
        t += [("user", x.pick(["how much do I need in the final to pass?", "what do I need on the final exam?"])),
              ("assistant", f"Which course? You're registered in: {', '.join(z['code'] for z in x.cur)}.")]
        t += [("user", x.pick([c["code"], c["title"], f"{c['code']} please"]))]
        tgt = TARGETS[0]
    else:
        tgt = x.pick(TARGETS[:4])
        t += [("user", x.pick([f"How much do I need in the {c['title']} final to {tgt[0]}?",
                               f"what do I need on the {c['code']} final to {tgt[0]}?",
                               f"{x.hi()}I have {c['marks_so_far']} in {c['code']} so far. What final exam score do I need to {tgt[0]}?"]))]
    t.append(("assistant", final_text(c, tgt)))
    if x.rng.random() < 0.35:
        tg2 = x.pick([z for z in TARGETS if z != tgt])
        t += [("user", x.pick([f"and to {tg2[0]}?", f"what about to {tg2[0]}?"])), ("assistant", final_text(c, tg2, short=True))]
    return ticket("Required Final Exam Score", t, ["current_courses"])

def final_text(c, tgt, short=False):
    so = c["seventh_week"] + c["twelfth_week"]
    rep = c["attempt"].startswith("repeat")
    name, T, L = tgt
    if rep and T > 84:
        return f"{c['code']} is a repeat of a failed course, so the highest grade you can get is B+ (84 or more), even if you score higher. Aim for 84 total."
    lines = []
    for pf in (10, 7, 5):
        n = need_final(so, pf, T)
        lines.append(f"pre-final {pf}/10 → final {('%.1f' % float(n)).rstrip('0').rstrip('.')}/40" if n <= 40 else f"pre-final {pf}/10 → not reachable (would need {float(n):g}/40)")
    s = (f"In {c['code']} you have {c['seventh_week']:g}/30 + {c['twelfth_week']:g}/20 = {so:g}/50. To {name} you need a total of {T}/100"
         + (", and at least 12/40 on the final in any case" if not short else "") + ": " + "; ".join(lines) + ".")
    if not short and rep:
        s += " Since this is a repeat of a failed course, your grade is capped at B+."
    return s

def current_marks(x):
    cur = [c for c in x.cur if "seventh_week" in c]
    if not cur: return None
    q = x.pick(["how am I doing this term?", "can you show me all my marks so far?", f"{x.hi()}a summary of my current coursework marks please.",
                "where do I stand in my courses right now?"])
    lines = [f"{c['code']} {c['title']}: {c['seventh_week']:g}/30 + {c['twelfth_week']:g}/20 = {c['marks_so_far']}" for c in cur]
    weak = sorted(cur, key=lambda c: c["seventh_week"] + c["twelfth_week"])[0]
    so = weak["seventh_week"] + weak["twelfth_week"]
    a = "Your marks so far (7th week + 12th week, out of 50):\n" + "\n".join(f"- {l}" for l in lines)
    a += (f"\nYour weakest is {weak['code']} at {so:g}/50. To pass it you need 60/100 overall and at least 12/40 on the final. "
          f"For example, with 7/10 in the pre-final you'd need {float(need_final(so, 7, 60)):g}/40 on the final.")
    fb = [c for c in cur if "instructor_feedback" in c]
    if fb:
        a += f"\nYour professor left a note on your {fb[0]['code']} 7th-week exam: \"{fb[0]['instructor_feedback']}\""
    return ticket("Current Marks Summary", [("user", q), ("assistant", a)], ["current_courses"])

def absence(x):
    cur = [c for c in x.cur if c["credit_hours"]]
    if not cur: return None
    c = x.pick(cur); n = x.pick([2, 3, 4, 5, 6])
    reason = x.pick(["because of an internship", "because I was sick", "due to a family emergency", "because of transport issues", ""])
    t = []
    if x.rng.random() < 0.3:
        t += [("user", f"I've missed {n} lectures {reason}. am I in danger?".replace("  ", " ")),
              ("assistant", f"Which course were the absences in? The limit is per course. You're registered in: {', '.join(z['code'] for z in cur)}.")]
        t += [("user", x.pick([c["code"], c["title"]]))]
    else:
        t += [("user", x.pick([f"I've missed {n} lectures in {c['title']} {reason}. Am I in danger of getting banned?".replace("  ", " "),
                               f"how many absences can I have in {c['code']} before I'm forced to withdraw? I'm at {n}",
                               f"{x.hi()}I was absent {n} times in {c['title']}. Will this affect my final exam eligibility?"]))]
    sessions = 28 if c["credit_hours"] >= 3 else 20
    mx = int(sessions * 0.15)
    st = ("you've reached the limit, so you can't miss another session" if n >= mx else f"you can miss {mx - n} more at most")
    if n > mx:
        st = "you may already be over the limit, so please contact your professor and advisor today"
    a = (f"You need at least 85% attendance in each course. Absence above 15% of scheduled classes means forced withdrawal from {c['code']}, whatever the reason. "
         f"For example, if {c['code']} has {sessions} sessions this term, 15% is {mx}, so with {n} absences {st}. "
         f"The official count is your professor's attendance record ({c['professor']}). If you have medical or emergency documents, give them to your advisor {x.adv['name']} right away.")
    t.append(("assistant", a))
    return ticket("Absence & Forced Withdrawal Warning", t, ["current_courses", "advisor"], priority="High" if n >= mx else "Normal")

def failed_final(x):
    rows = []
    for r in x.I["rows"]:
        e = x.I["marks"].get((r["course_code"], r["semester_taken"]))
        if r["grade"] == "F" and e and e["final_exam_score"] and float(e["final_exam_score"]) < 12:
            cw = float(e["seventh_week_score"]) + float(e["twelfth_week_score"]) + float(e["prefinal_score"])
            if cw >= 30:
                rows.append((r, e, cw))
    if not rows: return None
    r, e, cw = rows[-1]; c = r["course_code"]
    q = x.pick([f"I had {cw:g} out of 60 before the final in {x.T(c)}! Why does my transcript say F?!",
                f"there must be a mistake, I got {cw:g}/60 coursework in {c} and my total is {float(e['total_score']):g}. how is that an F",
                f"Can you explain my {c} result? Total {float(e['total_score']):g}/100 but grade F."])
    a = (f"I understand the shock, but the grade is correct. In {c} ({r['semester_taken']}) your coursework was {e['seventh_week_score']}/30 + {e['twelfth_week_score']}/20 + {e['prefinal_score']}/10 = {cw:g}/60, "
         f"and the final exam was {e['final_exam_score']}/40, for a total of {e['total_score']}. You need at least 12/40 on the final exam to pass, so a final below 12 is an automatic F regardless of coursework. ")
    a += ("You've since repeated it and the latest attempt replaced this F in your GPA." if r["status"] == "Retaken" else
          "You need to repeat the course. The repeat is capped at B+ and will replace the F in your GPA.")
    t = [("user", q), ("assistant", a)]
    if x.rng.random() < 0.25:
        t += [("user", "can I appeal the final exam mark?"),
              ("assistant", f"I can't change marks. You can submit a formal re-check request to your professor through your advisor {x.adv['name']}, who will tell you the deadline and the form. I've marked this ticket for advisor follow-up.")]
        return ticket("Failed Final Exam Threshold", t, ["transcript"], status="Escalated", assigned=x.adv["advisor_id"])
    return ticket("Failed Final Exam Threshold", t, ["transcript"])

def electives(x):
    rem = x.rec["remaining"]
    if rem["electives_still_needed"] == 0 or x.st["semester_level"] < 5: return None
    el = rem["electives_eligible_after_this_term"]
    q = x.pick(["Which electives can I take?", "what electives am I eligible for next term?", f"{x.hi()}could you list the electives I can register?",
                "I need to pick electives, what are my options"])
    if el:
        a = (f"You need {rem['electives_still_needed']} more elective(s) ({rem['electives_completed']} completed, {rem['electives_in_progress']} in progress). "
             f"Assuming you pass this term's courses, you'll meet the prerequisites for: {'; '.join(el[:10])}"
             + (f" and {len(el) - 10} more" if len(el) > 10 else "") + ". The curriculum places electives in terms 7, 8, 9 (2) and 10 (2).")
    else:
        a = f"You need {rem['electives_still_needed']} elective(s), but none of their prerequisites are complete yet. Most electives need CC319, CC311, CC431 or CC418 first."
    return ticket("Elective Selection", [("user", q), ("assistant", a)], ["remaining"])

def semester_level(x):
    st = x.st
    q = x.pick(["what semester am I in?", "why does the portal say I'm in semester " + str(st["semester_level"]) + "?",
                "how is the semester level calculated?", f"{x.hi()}which level am I officially at?"])
    c = st["credits_completed"]
    a = (f"Your semester level comes from completed credit hours: completed ÷ 18, rounded down, + 1 (0-17 → semester 1, 18-35 → 2, ... 162-179 → 10). "
         f"You've completed {c} credit hours, and credits are only added once a term's grades are released, so you're in semester {st['semester_level']}. "
         f"You have {st['credits_remaining']} credit hours left of the 180 needed to graduate.")
    if x.cur:
        add = sum(z["credit_hours"] for z in x.cur if z["code"] not in x.I["passed"])
        a += f" If you pass this term's {add} new credit hours, you'll move to semester {level_of(c + add) if c + add < 180 else 'graduate status'}."
    return ticket("Semester Level & Credits", [("user", q), ("assistant", a)], ["student", "current_courses"])

def contact(x):
    if x.cur and x.rng.random() < 0.55:
        c = x.pick(x.cur)
        t = []
        if x.rng.random() < 0.3:
            t += [("user", "who teaches my course?"), ("assistant", f"Which one? You're registered in: {', '.join(z['code'] for z in x.cur)}.")]
            t += [("user", x.pick([c["code"], c["title"]]))]
        else:
            t += [("user", x.pick([f"Who is teaching {c['title']} this term?", f"who's the professor and instructor for {c['code']}?"]))]
        t.append(("assistant", f"{c['code']} ({c['title']}) this term: professor {c['professor']}, instructor {c['instructor']}. For questions about marks, contact them first."))
        return ticket("Advisor & Staff Contact", t, ["current_courses"])
    q = x.pick(["who is my academic advisor?", "how do I contact my advisor?", "where is my advisor's office?"])
    a = f"Your academic advisor is {x.adv['name']}, office {x.adv['office']}, email {x.adv['email']}."
    return ticket("Advisor & Staff Contact", [("user", q), ("assistant", a)], ["advisor"])

def email_activation(x):
    st = x.st
    if st["university_email_activated"]:
        if x.rng.random() < 0.7: return None
        q = x.pick(["I can't log into the portal, it says invalid password", "I forgot my portal password"])
        a = (f"Your university email ({st['university_email']}) is active, so this is a password issue. I can't reset passwords, so I've forwarded this ticket to the portal administrators. "
             "They'll contact you on your university email. Meanwhile, please don't share your password with anyone.")
        return ticket("Portal Account & Email", [("user", q), ("assistant", a)], ["student"], status="Escalated", assigned="Admin")
    q = x.pick(["I'm not receiving any university emails", "my university email doesn't work", "how do I activate my student email?"])
    a = (f"Your university email ({st['university_email']}) isn't activated yet, which is why nothing arrives. Activation is done by the portal administrators. "
         "I've forwarded this ticket to them. Until then, official notices (including advisor messages) may only appear in the portal.")
    return ticket("Portal Account & Email", [("user", q), ("assistant", a)], ["student"], status="Escalated", assigned="Admin")

def escalation(x):
    kind = x.pick(["mark", "grade", "medical", "transfer", "load"])
    if kind == "mark" and [c for c in x.cur if "seventh_week" in c]:
        c = x.pick([c for c in x.cur if "seventh_week" in c])
        q = f"My 7th week mark in {c['code']} shows {c['seventh_week']:g}/30 but I think it was graded wrong. Can you fix it?"
        a = (f"I can't change marks. Only the course staff can. Your {c['code']} 7th-week mark is recorded as {c['seventh_week']:g}/30. "
             f"Please ask professor {c['professor']} for a review of your paper. I've logged this ticket so your advisor {x.adv['name']} can follow up if needed.")
        return ticket("Escalation: Mark Review", [("user", q), ("assistant", a)], ["current_courses"], status="Escalated", assigned=x.adv["advisor_id"])
    if kind == "grade" and x.I["rows"]:
        r = x.pick(x.I["rows"])
        q = f"Can you change my {r['course_code']} grade from {r['grade']} to a higher one? I really need it for my GPA."
        a = (f"I'm not able to change grades. Your {r['course_code']} grade ({r['grade']}, {r['semester_taken']}) is part of your official transcript. "
             "Grades can only change through a formal re-check request made through your advisor, and only if a marking error is found. "
             "If you want to raise your GPA, I can show you which repeat options or upcoming courses would help most.")
        return ticket("Escalation: Grade Change Request", [("user", q), ("assistant", a)], ["transcript"], status="Declined")
    if kind == "medical":
        q = x.pick(["I was in hospital during the 12th week exam, what do I do?", "I missed an exam because I was sick, can I retake it?"])
        a = (f"Sorry to hear that, I hope you're feeling better. Make-up exams need an approved medical excuse. Send your medical report to your advisor {x.adv['name']} ({x.adv['email']}) as soon as possible. "
             "They'll pass it to the department for a decision. I've forwarded this ticket to them.")
        return ticket("Escalation: Medical Excuse", [("user", q), ("assistant", a)], ["advisor"], status="Escalated", assigned=x.adv["advisor_id"], priority="High")
    if kind == "transfer":
        q = "I took a course at another university, can it be transferred to my record?"
        a = ("Transfer credit is possible if the course is from an accredited institution and the Computer Engineering department approves it as equivalent, and at least 50% of your 180 credit hours must be completed at AASTMT. "
             f"I can't approve transfers myself. Please submit the course syllabus and official transcript through your advisor {x.adv['name']}. I've forwarded this ticket.")
        return ticket("Escalation: Credit Transfer", [("user", q), ("assistant", a)], ["advisor"], status="Escalated", assigned=x.adv["advisor_id"])
    if x.I["on_prob"]:
        q = "can my advisor approve 15 credit hours for me even though I'm on probation?"
        a = (f"No. The 12-credit maximum for students on probation (GPA below 2.0) is a regulation, not an advisor decision, so it can't be raised. Your GPA is {x.st['cumulative_gpa']}. "
             f"Your advisor {x.adv['name']} can help you choose the 12 credit hours that will raise your GPA the most.")
        return ticket("Escalation: Load Exception", [("user", q), ("assistant", a)], ["student", "registration_rules_now"], status="Declined")
    return None

def failed_list(x):
    f = x.rec["repeats"]["outstanding_failed_courses"]
    hist_F = x.rec["honors_check"]["f_grades"]
    if not hist_F: return None
    q = x.pick(["which courses do I still have to repeat?", "what failed courses are on my record?", f"{x.hi()}do I have any F's I still need to clear?"])
    if f:
        now = [c for c in f if c in x.I["current_codes"]]
        later = [c for c in f if c not in x.I["current_codes"]]
        a = f"Courses with an F that you haven't passed yet: {', '.join(course_line(x, c) for c in f)}."
        if now: a += f" You're repeating {', '.join(now)} this term."
        if later: a += f" {', '.join(later)} still need to be registered, and failed courses take priority in your next registration."
        a += " Repeats are capped at B+, and the new grade replaces the F in your GPA."
    else:
        a = f"You have no outstanding F grades. Every course you failed ({', '.join(hist_F)}) has since been repeated and passed. The F still counts against honors eligibility, though."
    return ticket("Failed Courses to Repeat", [("user", q), ("assistant", a)], ["repeats", "honors_check"])

def transcript_summary(x):
    h = x.rec["semester_history"]
    if len(h) < 2: return None
    q = x.pick(["can you show my GPA history?", "how has my GPA changed over the semesters?", f"{x.hi()}a semester-by-semester summary of my results please."])
    lines = [f"{s['term']}: {s['credit_hours']} cr, semester GPA {s['semester_gpa'] or '-'}, cumulative {s['cumulative_gpa']} ({s['standing']})" for s in h]
    best = max((s for s in h if s["semester_gpa"]), key=lambda s: Fr(s["semester_gpa"]))
    a = "Here's your record term by term:\n" + "\n".join(f"- {l}" for l in lines) + f"\nYour best semester was {best['term']} ({best['semester_gpa']}). Current cumulative GPA: {x.st['cumulative_gpa']}."
    return ticket("Transcript Summary", [("user", q), ("assistant", a)], ["semester_history"])

INTENTS = [
    (probation_load, 3), (probation_consequences, 3), (grad_audit, 2), (grad_plan, 2), (next_term, 2), (overload, 1),
    (summer_exception, 2), (retake_cap, 2), (improve_window, 1.5), (prereq_question, 2.5), (honors, 1), (mark_breakdown, 1),
    (standing, 1.5), (gpa_method, 1.5), (gpa_projection, 2.5), (required_final, 2.5), (current_marks, 1.5), (absence, 1),
    (failed_final, 1.5), (electives, 1), (semester_level, 1), (contact, 0.8), (email_activation, 0.6), (escalation, 0.7),
    (failed_list, 1), (transcript_summary, 1),
]

def generate(db, records, rng, per_student=6):
    out = []
    n = 0
    for sid, rec in records.items():
        x = Ctx(db, rec, rng)
        pool = INTENTS[:]
        got = 0
        tries = 0
        while got < per_student and pool and tries < 60:
            tries += 1
            fn, w = rng.choices(pool, [w for _, w in pool])[0]
            pool = [p for p in pool if p[0] is not fn]
            tk = fn(x)
            if tk:
                n += 1
                tk.update(ticket_id=f"TKT-{n:06d}", student_id=sid,
                          channel=rng.choice(["Portal Chat", "Portal Chat", "Mobile App"]),
                          academic_week=rng.choice([12, 12, 13]), term=CURRENT_TERM)
                out.append(tk)
                got += 1
    return out
