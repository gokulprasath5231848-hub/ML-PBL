"""
Evaluation script for the report (Chapter 6).

Runs the REAL screening_engine.py unchanged -- there is no separate demo
pipeline. Reports, on the 12-resume demo batch in sample_data/:
  (a) the baseline TF-IDF score against the synonym-normalised hybrid score
      the application already computes for every candidate,
  (b) how far the ranking moves between the two scorings,
  (c) the skill checklist extracted from the job description itself, and
  (d) one constructed case where a resume spells every skill out in full
      against a job description that uses the abbreviated forms, to isolate
      what the synonym-normalisation step is for.
"""
import pandas as pd
from screening_engine import (extract_skills_from_jd, pool_insights,
                              screen_resumes)

JD_PATH = "sample_data/sample_job_description.txt"
CSV_PATH = "sample_data/sample_resumes_large.csv"

with open(JD_PATH) as f:
    jd_text = f.read()

resumes_df = pd.read_csv(CSV_PATH)
required = extract_skills_from_jd(jd_text)
results_df = screen_resumes(jd_text, resumes_df, required)
records = results_df.to_dict("records")

lines = []
lines.append("$ python eval_real.py")
lines.append(f"Job description : {JD_PATH}")
lines.append(f"Resumes         : {CSV_PATH}  ({len(resumes_df)} candidates)")
lines.append(f"Skills extracted: {len(required)} from the JD -> {', '.join(required)}")
lines.append("")
lines.append(f"{'Rank':<5}{'Candidate':<22}{'Baseline':<11}{'Hybrid':<9}"
             f"{'Delta':<8}{'Skills':<8}{'Verdict'}")
for r in records:
    lines.append(
        f"{r['Rank']:<5}{r['Candidate']:<22}{r['Baseline_TFIDF_%']:<11}"
        f"{r['Hybrid_Match_%']:<9}{r['Delta']:<+8}"
        f"{str(r['Matched_Count']) + '/' + str(r['Total_Skills']):<8}{r['Verdict']}"
    )

# --- how much the synonym step actually moves the shortlist ---------------
hybrid_order = [r["Candidate"] for r in records]
baseline_order = [r["Candidate"] for r in
                  sorted(records, key=lambda r: -r["Baseline_TFIDF_%"])]
moved = sum(1 for i, c in enumerate(hybrid_order) if baseline_order.index(c) != i)
lifted = [r for r in records if r["Delta"] > 0]

lines.append("")
lines.append("--- Effect of synonym normalisation on the ranking ---")
lines.append(f"Candidates whose score rose : {len(lifted)}/{len(records)}")
lines.append(f"Largest single gain         : "
             f"{max(r['Delta'] for r in records):+.1f} points "
             f"({max(records, key=lambda r: r['Delta'])['Candidate']})")
lines.append(f"Candidates that change rank : {moved}/{len(records)}")

# --- pool-level skill gaps ------------------------------------------------
gaps = pool_insights(records, required)
lines.append("")
lines.append("--- Scarcest requirements across the pool ---")
for g in gaps[:3]:
    lines.append(f"{g['skill']:<30}{g['have']}/{len(records)} candidates ({g['pct']}%)")

# --- isolated synonym case ------------------------------------------------
lines.append("")
lines.append("--- Synonym case (spelled-out resume vs abbreviated JD) ---")
case_jd = ("Looking for a Machine Learning Engineer with strong Python skills. "
           "Must know NLP, AI and cloud platforms such as AWS. SQL is a plus.")
case_resume = pd.DataFrame([{
    "candidate_name": "Case: spelled-out resume",
    "resume_text": (
        "Experienced in Python and Machine Learning. Built Natural Language "
        "Processing pipelines and worked on Artificial Intelligence projects. "
        "Comfortable with Structured Query Language and Amazon Web Services."
    ),
}])
case = screen_resumes(case_jd, case_resume).iloc[0]
lines.append(f"Baseline TF-IDF : {case['Baseline_TFIDF_%']}%  (literal keyword overlap)")
lines.append(f"Hybrid Match    : {case['Hybrid_Match_%']}%  (after normalize_text)")
lines.append(f"Gain            : {case['Delta']:+.1f} points")
lines.append(f"Checklist       : {case['Matched_Count']}/{case['Total_Skills']}"
             f"  ->  {case['Verdict']}")

output_text = "\n".join(lines)
print(output_text)
with open("eval_real_output.txt", "w") as f:
    f.write(output_text)
