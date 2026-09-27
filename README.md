# AI Powered Resume Screening and Ranking System

A Flask application that scores resumes against a job description twice — a
plain TF-IDF baseline and a synonym-aware hybrid score — checks each resume
against a skill checklist read from the job description itself, and explains
every ranking it produces.

Self-contained: no CDN, no build step, no internet access required at runtime.

## Run locally

```bash
pip install -r requirements.txt
python app.py
```

Then open http://127.0.0.1:5000

Try it with the sample data in `sample_data/`: press **Load sample** to fill
the job description, then upload `sample_resumes_large.csv` (12 candidates —
enough for the pool and compare views to be useful). `sample_resumes.csv` is
the original 3-candidate set.

## What it does

**Scoring.** Every resume is vectorised twice with scikit-learn's
`TfidfVectorizer` and scored by cosine similarity: once as written (the
baseline) and once after `normalize_text()` folds synonymous skill terms onto
a single token ("ML" and "Machine Learning" become the same term). Both scores
are kept so they can be compared directly.

**The checklist is read from the job description.** `extract_skills_from_jd()`
scans the description against a 32-skill lexicon and returns what it finds, in
the order the recruiter wrote it. The list stays editable before the run, so an
unusual requirement can be added by hand. Coverage of that checklist — not the
similarity score — decides the Strong / Moderate / Weak verdict.

**Explainability.** `TfidfVectorizer` L2-normalises each document vector, so
cosine similarity equals the dot product of the two vectors. That means the
score decomposes exactly, term by term, with no approximation: the interface
plots each term's true share of the score. Every candidate also gets a one-line
written reason built from the same numbers shown beside it.

**Review interface.** A ranked rail with coverage meters and verdicts on the
left, candidate detail on the right. Switch between baseline and hybrid
scoring and the list physically reorders. Three detail views: the candidate,
pool insights (which requirements are scarcest across all candidates), and a
side-by-side comparison of up to three candidates. A threshold control
re-labels every verdict as the Strong cut-off moves — that is a client-side
pass, since coverage is already computed.

## Project structure

- `app.py` — Flask routes: compose form (`/`), live skill extraction
  (`/api/extract-skills`), screening (`/screen`), CSV export (`/download`)
- `screening_engine.py` — skill lexicon and extraction, baseline + hybrid
  scoring, checklist matching and verdicts, exact term-contribution
  explainability, written reasons, pool skill-gap analysis
- `file_parser.py` — extracts text from uploaded PDF (pdfplumber) and DOCX
  (python-docx) resumes
- `eval_real.py` — reproducible evaluation script, run against the real engine
- `static/app.css` — design tokens, layout and components; `static/inter.ttf`
  is bundled so no font CDN is needed
- `templates/` — page shell, compose screen, review screen
- `sample_data/` — job description, 3-resume CSV, 12-resume demonstration batch

## Interface design notes

Colour never carries meaning on its own: each verdict has its own shape
(circle, square, outlined triangle) and label, so the three states stay
distinguishable in greyscale and for colour-blind reviewers. Every text and
background pair meets the WCAG AAA 7:1 contrast ratio.

Coverage and similarity are deliberately drawn differently. Coverage is a
fraction, so it is a segmented meter; similarity is a position on a scale, so
it is a marker on a 0–100 track with the pool median marked. Drawing both as
coloured badges — as an earlier version did — made them look contradictory
whenever they disagreed, which they legitimately can.

## Known limitations

- Skill extraction is lexicon-based, so a skill outside the 32-entry lexicon
  will not be picked up automatically. That is why the checklist is editable.
- The similarity score is partly driven by generic job-description vocabulary
  (run the app and look at the contribution bars for a mid-ranked candidate —
  words like "experience" can be the largest single contributor). Read it
  alongside the coverage verdict, not on its own.
- Results are held in memory for the current server process, not a database,
  so the app suits one reviewer at a time.
- Scanned or image-only PDFs have no extractable text and no OCR is applied;
  they are reported as unreadable rather than scored zero.
- Matching is lexical and synonym-based, not a learned language model. It will
  miss a semantic match with no shared vocabulary.
- The system is a shortlisting aid. It makes no hiring decision and has not
  been audited for fairness or bias.
