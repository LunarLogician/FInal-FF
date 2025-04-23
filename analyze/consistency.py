from fastapi import APIRouter, Request, File, UploadFile
from pydantic import BaseModel
from typing import List, Union
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re
import string
import os

import openai
openai.api_key = os.environ.get("OPENAI_API_KEY")


import fitz  # PyMuPDF
from docx import Document
import langid  # 🔥 Language detection

from textstat import flesch_reading_ease
from nltk import word_tokenize
from nltk.util import ngrams
from nltk.corpus import stopwords

import nltk
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

router = APIRouter()

class ConsistencyInput(BaseModel):
    chunks: List[str]  # Still accepts chunks, but we'll treat them as full texts

def is_english(text: str) -> bool:
    lang, _ = langid.classify(text)
    return lang == "en"

def preprocess(text: str) -> str:
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    return text.translate(str.maketrans('', '', string.punctuation))

def split_into_segments(text: str, segment_length: int) -> List[str]:
    """Split text into segments of approximately equal length."""
    words = text.split()
    segments = []
    
    for i in range(0, len(words), segment_length):
        segment = " ".join(words[i:i + segment_length])
        if segment:  # Only add non-empty segments
            segments.append(segment)
    
    return segments


def compute_tfidf_consistency(segments: List[str]) -> Union[dict, None]:
    if len(segments) <= 1:
        return None

    try:
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(segments)
        sim_matrix = cosine_similarity(tfidf_matrix)
        tril_indices = np.tril_indices_from(sim_matrix, k=-1)
        values = sim_matrix[tril_indices]

        mean_sim = float(np.mean(values))
        std_dev_sim = float(np.std(values))
        return {
            "consistency_score": mean_sim,
            "consistency_variability": std_dev_sim
        }
    except Exception as e:
        print("TF-IDF Consistency error:", e)
        return None



from nltk.tokenize import sent_tokenize

def compute_readability(text: str) -> float:
    try:
        # Reconstruct clean sentence-delimited text
        sentences = sent_tokenize(text)
        reconstructed = " ".join(sentences)

        score = flesch_reading_ease(reconstructed)
        print("Raw Flesch score:", score)

        # Normalize: Flesch score ~ [0–100]; higher = more readable
        normalized = max(0.0, min(score / 100.0, 1.0))
        print("Normalized readability:", normalized)
        return normalized
    except Exception as e:
        print("Readability error:", e)
        return 0.0



import math

def compute_clarity(text: str) -> float:
    try:
        words = word_tokenize(text.lower())
        words = [w for w in words if w.isalpha()]
        total = len(words)
        unique = len(set(words))
        if total == 0:
            print("No valid words found for CTTR")
            return 0.0
        cttr = unique / math.sqrt(2 * total)
        print("CTTR raw:", cttr, "| Total:", total, "| Unique:", unique)

        # Normalize for range 0–25
        normalized = cttr / 25.0
        normalized = max(0.0, min(normalized, 1.0))
        print("Normalized clarity:", normalized)
        return float(normalized)
    except Exception as e:
        print("Clarity error:", e)
        return 0.0




@router.post("/consistency")
async def compute_consistency(data: ConsistencyInput):
    full_text = " ".join(data.chunks)  # Assume single document was passed

    if not is_english(full_text):
        return {"error": "Non-English text detected. Only English documents are supported."}

    cleaned = preprocess(full_text)

    segment_lengths = [500, 1000, len(cleaned.split())]  # include full doc

    scores = []
    variabilities = [] 
    for length in segment_lengths:
        segments = split_into_segments(cleaned, length) if length != len(cleaned.split()) else [cleaned]
        embedding_result = compute_tfidf_consistency(segments)
        if embedding_result is not None:
            scores.append(embedding_result["consistency_score"])
            variabilities.append(embedding_result["consistency_variability"])


    consistency_score = float(np.mean(scores)) if scores else None
    consistency_variability = float(np.mean(variabilities)) if scores else None
    readability_score = compute_readability(full_text)
    clarity_score = compute_clarity(full_text)

    print("RETURNING CONSISTENCY RESULTS:")
    print("  consistency_score:", consistency_score)
    print("  readability_score:", readability_score)
    print("  clarity_score:", clarity_score)


    return {
    "consistency_score": consistency_score,
    "consistency_variability": consistency_variability,
    "readability_score": readability_score,
    "clarity_score": clarity_score
    }


# ========== NEW PDF/DOCX SUPPORT ==========

def extract_text_from_pdf(file_stream) -> str:
    doc = fitz.open(stream=file_stream.read(), filetype="pdf")
    return "\n".join([page.get_text() for page in doc])

def extract_text_from_docx(file_stream) -> str:
    document = Document(file_stream)
    return "\n".join([p.text for p in document.paragraphs])

@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    filename = file.filename.lower()
    if filename.endswith(".pdf"):
        text = extract_text_from_pdf(file.file)
    elif filename.endswith(".docx"):
        text = extract_text_from_docx(file.file)
    else:
        return {"error": "Unsupported file type. Please upload a PDF or Word document (.pdf, .docx)"}

    return {"name": file.filename, "text": text}
