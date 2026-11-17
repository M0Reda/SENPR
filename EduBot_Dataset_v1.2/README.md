# EduBot dataset v3: AASTMT Computer Engineering

There are 1,000 active students spread across semesters 1 to 10. The data is a snapshot of Fall 2026-2027 (week 12).

## What changed in v3
- **Graduation rule:** credit hours are only added once a term's grades are out. A student who reaches 180 credit hours is a graduate and is no longer in the active data. The highest value in Students.csv is now 177. Practical Training (IM400CC) is taken in the student's final term.
- **Semester level:** 0–17 credits → semester 1, 18–35 → 2, 36–53 → 3, ... 162–179 → 10, and 180 means graduated.
- **Advising_QA is now ticket-based:** multi-turn conversations. The bot asks a clarifying question when the student is vague, answers follow-ups, and routes cases it can't handle to the advisor or an admin.
- **Courses.csv** has a new `recommended_term` column (1–10, or "Elective"), used by the graduation planner.

## Why Course_Enrollments has more rows than Student_Transcripts
Course_Enrollments also holds this term's (Fall 2026) courses. These are still in progress with only the 7th-week and 12th-week marks, and they reach the transcript when the final grades are released. Every finished enrollment has exactly one transcript row.

## Files
| File | What it is |
|---|---|
| `Users.csv`, `Advisors.csv`, `Students.csv`, `Courses.csv`, `Course_Prerequisites.csv`, `Course_Offerings.csv`, `Course_Enrollments.csv`, `Student_Transcripts.csv` | The relational tables |
| `Advising_QA.csv` | Tickets flattened to one row per message: ticket_id, student_id, intent_category, turn_no, role (student/edubot), message, ticket_status, assigned_to |
| `Student_Transcripts/<id>.csv` | One progressive transcript per student, term by term |
| `EduBot_Training/tickets.jsonl` | 6,000 tickets with metadata (intent, priority, status, assigned_to, data_used, messages) |
| `EduBot_Training/sft_train/validation/test.jsonl` | Ready-to-train chat format: system (regulations + the student's record) → user → assistant…, split 80/10/10 by student |
| `EduBot_Training/student_records.jsonl` | The record EduBot sees for each student |
| `EduBot_Colab_Starter.ipynb` | Load → inspect → LoRA fine-tune → chat → quick accuracy check |
| `scripts/edubot_context.py` | Builds a student's record and prompt from the CSVs. The same code made the training data, so use it in production too |
| `scripts/gen.py`, `tickets.py`, `validate.py` | Generator and independent validator (no errors) |

## Ticket types (30)
Graduation Plan, Graduation Requirements Audit, Next Term Registration, GPA Projection, Required Final Exam Score, Current Marks Summary, Prerequisite Dependency Check, Standing Classification, GPA Calculation Methodology, Transcript Summary, Semester Level & Credits, Honors Eligibility, Retake Grade Cap, Course Improvement Window, Failed Courses to Repeat, Failed Final Exam Threshold, Elective Selection, Overload Permission, Summer Graduating Overload Exception, Probation Load Restriction, Probation Consequences, Absence & Forced Withdrawal, Course Assessment Mark Breakdown, Advisor & Staff Contact, Portal Account & Email, and escalations (mark review, grade change, medical excuse, credit transfer, load exception).

About 20% of tickets are multi-turn: clarifying questions, follow-ups, or an escalation after an answer.
