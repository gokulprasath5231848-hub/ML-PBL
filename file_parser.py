"""
Extracts plain text from uploaded resume files (PDF / DOCX) so a real
uploaded file and a CSV row become identical inputs to the screening
engine.
"""

import os
import pdfplumber
import docx


def extract_text_from_pdf(file_storage) -> str:
    text_chunks = []
    with pdfplumber.open(file_storage) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text_chunks.append(page_text)
    return "\n".join(text_chunks)


def extract_text_from_docx(file_storage) -> str:
    document = docx.Document(file_storage)
    return "\n".join(p.text for p in document.paragraphs)


def extract_text_from_file(file_storage) -> str:
    """file_storage: a Werkzeug FileStorage object (from request.files)."""
    filename = file_storage.filename or ""
    ext = os.path.splitext(filename)[1].lower()

    if ext == ".pdf":
        return extract_text_from_pdf(file_storage)
    elif ext == ".docx":
        return extract_text_from_docx(file_storage)
    elif ext == ".txt":
        return file_storage.read().decode("utf-8", errors="ignore")
    else:
        raise ValueError(f"Unsupported file type: {ext or '(no extension)'}")


def candidate_name_from_filename(filename: str) -> str:
    """Turns 'john_doe_resume.pdf' into 'John Doe Resume'."""
    stem = os.path.splitext(os.path.basename(filename))[0]
    stem = stem.replace("_", " ").replace("-", " ")
    return stem.title()
