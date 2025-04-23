import sys
import os
from docx import Document
import fitz  # PyMuPDF

def extract_text_docx(path):
    doc = Document(path)
    return "\n".join([p.text for p in doc.paragraphs])

def extract_text_pdf(path):
    doc = fitz.open(path)
    return "\n".join([page.get_text() for page in doc])

file_path = sys.argv[1]
ext = os.path.splitext(file_path)[1].lower()

if ext == ".docx":
    print(extract_text_docx(file_path))
elif ext == ".pdf":
    print(extract_text_pdf(file_path))
else:
    print("Unsupported file type", file=sys.stderr)
