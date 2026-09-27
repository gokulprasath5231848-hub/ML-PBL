import io
import json

import pandas as pd
from flask import (Flask, jsonify, redirect, render_template, request,
                   send_file, url_for)

from screening_engine import (DEFAULT_REQUIRED_SKILLS, MODERATE_THRESHOLD,
                              STRONG_THRESHOLD, extract_skills_from_jd,
                              pool_insights, screen_resumes)
from file_parser import candidate_name_from_filename, extract_text_from_file

app = Flask(__name__)

# Single-user demo storage: keeps the last run's results in memory so the
# results page and CSV download route can both read it. Fine for a local
# demo / single reviewer session; not meant for concurrent multi-user use.
_last_results_df = None


def _compose(error=None, job_desc_text="", skills=None, notices=None):
    """Render the compose screen, preserving whatever the user already typed."""
    return render_template(
        "index.html",
        error=error,
        job_desc_text=job_desc_text,
        skills=skills or [],
        notices=notices or [],
    )


@app.route("/", methods=["GET"])
def index():
    return _compose()


@app.route("/api/extract-skills", methods=["POST"])
def api_extract_skills():
    """Live skill extraction for the compose screen.

    The checklist is seeded from the job description as the user types it,
    then stays editable — the extracted list is a starting point, not a
    fixed requirement.
    """
    payload = request.get_json(silent=True) or {}
    skills = extract_skills_from_jd(payload.get("job_desc_text", ""))
    return jsonify({"skills": skills})


@app.route("/screen", methods=["POST"])
def screen():
    global _last_results_df

    # ---- Job description: pasted text takes priority, else uploaded .txt ----
    job_desc_text = (request.form.get("job_desc_text") or "").strip()
    job_desc_file = request.files.get("job_desc_file")
    if not job_desc_text and job_desc_file and job_desc_file.filename:
        job_desc_text = job_desc_file.read().decode("utf-8", errors="ignore").strip()

    if not job_desc_text:
        return _compose(error="Please paste a job description or upload a .txt file.")

    # ---- Required skills: the edited chip list, else derived from the JD ----
    posted_skills = [s.strip().lower() for s in
                     (request.form.get("required_skills") or "").split(",") if s.strip()]
    required_skills = posted_skills or extract_skills_from_jd(job_desc_text)

    # ---- Resumes: CSV rows + individually uploaded files, combined ----
    rows, notices = [], []

    resumes_csv = request.files.get("resumes_csv")
    if resumes_csv and resumes_csv.filename:
        try:
            csv_df = pd.read_csv(resumes_csv)
            missing_cols = {"candidate_name", "resume_text"} - set(csv_df.columns)
            if missing_cols:
                return _compose(
                    error=f"CSV is missing the column(s): {', '.join(sorted(missing_cols))}. "
                          "It needs 'candidate_name' and 'resume_text'.",
                    job_desc_text=job_desc_text, skills=required_skills)
            for _, row in csv_df.iterrows():
                rows.append({"candidate_name": row["candidate_name"],
                             "resume_text": row["resume_text"]})
        except Exception as exc:
            return _compose(error=f"Couldn't read the CSV file: {exc}",
                            job_desc_text=job_desc_text, skills=required_skills)

    resume_files = [f for f in request.files.getlist("resume_files") if f and f.filename]
    for f in resume_files:
        try:
            text = extract_text_from_file(f)
        except ValueError as exc:
            return _compose(error=str(exc), job_desc_text=job_desc_text,
                            skills=required_skills)
        if not text.strip():
            # A scanned / image-only PDF. Reported separately so the reviewer
            # knows the file was unreadable rather than the candidate unqualified.
            notices.append(f"{f.filename}: no extractable text (scanned or image-only "
                           "PDF) — skipped, not scored.")
            continue
        rows.append({"candidate_name": candidate_name_from_filename(f.filename),
                     "resume_text": text})

    if not rows:
        return _compose(error="Add resumes via CSV, individual files, or both.",
                        job_desc_text=job_desc_text, skills=required_skills,
                        notices=notices)

    resumes_df = pd.DataFrame(rows)
    results_df = screen_resumes(job_desc_text, resumes_df, required_skills)
    _last_results_df = results_df

    results = results_df.to_dict("records")
    verdict_counts = {
        "strong": sum(1 for r in results if r["Verdict"] == "STRONG MATCH"),
        "moderate": sum(1 for r in results if r["Verdict"] == "MODERATE MATCH"),
        "weak": sum(1 for r in results if r["Verdict"] == "WEAK MATCH"),
    }
    lifted = sum(1 for r in results if r["Delta"] > 0)

    return render_template(
        "results.html",
        results=results,
        results_json=json.dumps(results),
        insights=pool_insights(results, required_skills),
        required_skills=required_skills,
        total_candidates=len(results),
        verdict_counts=verdict_counts,
        lifted=lifted,
        notices=notices,
        strong_threshold=STRONG_THRESHOLD,
        moderate_threshold=MODERATE_THRESHOLD,
    )


@app.route("/download", methods=["GET"])
def download():
    global _last_results_df
    if _last_results_df is None:
        return redirect(url_for("index"))

    export_df = _last_results_df.copy()
    export_df["Matched_Skills"] = export_df["Matched_Skills"].apply(", ".join)
    export_df["Missing_Skills"] = export_df["Missing_Skills"].apply(", ".join)
    export_df["Top_Contributing_Terms"] = export_df["Top_Contributing_Terms"].apply(
        lambda terms: ", ".join(f"{t['term']} ({t['share']}%)" for t in terms))

    buffer = io.BytesIO()
    export_df.to_csv(buffer, index=False)
    buffer.seek(0)
    return send_file(buffer, mimetype="text/csv", as_attachment=True,
                     download_name="screening_results.csv")


if __name__ == "__main__":
    app.run(debug=False, port=5000)
