# EduBot — Data Audit Report

Dataset: `D:\M.Reda\SENPR\EduBot_Dataset_v1.2`  
Generated: 2026-10-13  

---

## 0 · File Inventory

| File | Rows |
|---|---|
| `Students.csv` | 1,000 |
| `Student_Cohorts.csv` | 1,000 |
| `Student_Transcripts.csv` | 25,910 |
| `Course_Enrollments.csv` | 31,821 |
| `Courses.csv` | 83 |
| `Course_Prerequisites.csv` | 81 |
| `Course_Offerings.csv` | 775 |
| `Advisors.csv` | 10 |
| `Advising_QA.csv` | 14,650 |
| `Users.csv` | exists (not loaded) |


## 1 · Credits Reconciliation

✅ All 1,000 students' `total_credits_completed` match the transcript-derived sum.

**Difference distribution (Students.total − computed):**

|   diff |   count |
|-------:|--------:|
|      0 |    1000 |


## 2 · Enrollment vs. Transcript Row Count

- **Total enrollments:** 31,821
- **Total transcripts:** 25,910
- **Gap:** 5,911
- **Current-term (Fall 2026-2027) enrollments:** 5,911
- **Past enrollments:** 25,910

✅ The gap (5,911) is fully explained by current-term in-progress enrollments (5,911 rows). These courses are still running (week 12, Fall 2026-2027) and will reach the transcript when final grades are released.


## 3 · professor_id vs instructor_id

**Semantic distinction:**

- `professor_id` — the faculty member who *owns* the course (writes the syllabus, sets the exams, submits final grades). Typically a senior academic.
- `instructor_id` — the staff member who *delivers* the lab/section (may be a junior lecturer or teaching assistant). The two can be the same person for pure-lecture courses.

| Metric | Value |
|---|---|
| Unique professor_ids | 5 |
| Unique instructor_ids | 5 |
| IDs appearing in both roles | 0 |
| professor_ids not in Users.csv | 0 |
| instructor_ids not in Users.csv | 0 |

✅ All professor_id and instructor_id values resolve to Users.csv records.


## 4 · Duplicate Rows

| Table | ID column | Total rows | Duplicate IDs |
|---|---|---|---|
| Student_Transcripts | `transcript_id` | 25,910 | 0 ✅ |
| Course_Enrollments | `enrollment_id` | 31,821 | 0 ✅ |
| Students | `student_id` | 1,000 | 0 ✅ |
| Advisors | `advisor_id` | 10 | 0 ✅ |
| Course_Offerings | `offering_id` | 775 | 0 ✅ |
| Advising_QA | `(ticket_id, turn_no)` | 14,650 | 0 ✅ |


## 5 · Null & Orphan Key Checks

✅ All student advisor_ids resolve to Advisors.csv.

✅ All transcript student_ids resolve to Students.csv.

✅ All transcript course_codes resolve to Courses.csv.

✅ All prerequisite course_codes resolve to Courses.csv.

✅ All enrollment offering_ids resolve to Course_Offerings.csv.

✅ All Advising_QA student_ids resolve to Students.csv.

⚠️  Null values found in key columns:
  - `Students.gpa`: 164 nulls


## 6 · Grade / Points & Standing Consistency

✅ All 25,910 transcript rows pass the grade→points formula check (tolerance ±0.05).

⚠️  257 students have a standing_status inconsistent with their GPA:

|   student_id |   gpa | standing_status                  | expected_standing   |
|-------------:|------:|:---------------------------------|:--------------------|
|    202003449 |   1.8 | Academic Probation (2nd Warning) | Academic Probation  |
|    202003474 |   1.9 | Academic Probation (1st Warning) | Academic Probation  |
|    211004166 |   1.9 | Academic Probation (1st Warning) | Academic Probation  |
|    212001635 |   1.9 | Academic Probation (1st Warning) | Academic Probation  |
|    212001868 |   1.9 | Academic Probation (1st Warning) | Academic Probation  |
|    221004168 |   1.9 | Academic Probation (1st Warning) | Academic Probation  |
|    221005040 |   1.8 | Academic Probation (1st Warning) | Academic Probation  |
|    222002264 |   1.8 | Academic Probation (1st Warning) | Academic Probation  |
|    222002557 |   1.9 | Academic Probation (2nd Warning) | Academic Probation  |
|    222002593 |   1.9 | Academic Probation (1st Warning) | Academic Probation  |


## 7 · Advising_QA Structure & Inflated-Retrieval Note

- **Total message rows:** 14,650
- **Unique tickets:** 6,000
- **Unique students:** 1,000
- **Roles:** {'student': np.int64(7325), 'edubot': np.int64(7325)}
- **Ticket statuses:** {'Resolved': np.int64(14212), 'Escalated': np.int64(382), 'Declined': np.int64(56)}

**Turns-per-ticket distribution:**

|       |   count |    mean |      std |   min |   25% |   50% |   75% |   max |
|:------|--------:|--------:|---------:|------:|------:|------:|------:|------:|
| turns |    6000 | 2.44167 | 0.903549 |     2 |     2 |     2 |     2 |     6 |

- Tickets with >2 turns (multi-turn): **20.5%** of all tickets

**Top 20 intent categories (by message count):**

| intent                              |   messages |
|:------------------------------------|-----------:|
| GPA Projection                      |       1664 |
| Required Final Exam Score           |       1662 |
| Prerequisite Dependency Check       |       1284 |
| Graduation Plan                     |       1066 |
| Graduation Requirements Audit       |        912 |
| Next Term Registration              |        808 |
| Standing Classification             |        798 |
| Current Marks Summary               |        696 |
| GPA Calculation Methodology         |        692 |
| Absence & Forced Withdrawal Warning |        632 |
| Semester Level & Credits            |        522 |
| Course Assessment Mark Breakdown    |        478 |
| Advisor & Staff Contact             |        460 |
| Course Improvement Window Policy    |        426 |
| Honors Eligibility Criteria         |        330 |
| Transcript Summary                  |        328 |
| Overload Permission                 |        328 |
| Retake Grade Cap                    |        316 |
| Failed Courses to Repeat            |        244 |
| Failed Final Exam Threshold         |        216 |

⚠️  The QA data is templated / programmatically generated. Many bot responses follow near-identical patterns, so retrieval scores on this test set will be artificially high. A paraphrased hard test set (generated with the local LLM in evaluate.py) is required to measure real generalisation.


## 8 · Semester-Level Formula Cross-Check

✅ All 1,000 semester_level values match the formula floor(credits/18)+1.


## 9 · Existing Train / Val / Test Split Coverage

| Split | File exists | Lines | Unique students (first turn) |
|---|---|---|---|
| train | ✅ | 4,800 | 800 |
| validation | ✅ | 600 | 100 |
| test | ✅ | 600 | 100 |

**Student-ID overlap across splits:**

- train ∩ validation: 0
- train ∩ test: 0
- validation ∩ test: 0

✅ No student appears in more than one split — clean train/val/test isolation.
