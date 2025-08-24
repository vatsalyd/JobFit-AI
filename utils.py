import re
import os
from pathlib import Path
from typing import Set, List, Dict, Any
import numpy as np
import spacy
from rapidfuzz import fuzz
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer, util
_embedding_model = None
import spacy
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

def load_skill_set(file_path: str) -> Set[str]:
    p = Path(file_path)
    if not p.exists():
        raise FileNotFoundError(f"Skill file not found: {file_path}")
    with p.open("r", encoding="utf-8") as f:
        skills = {line.strip().lower() for line in f if line.strip()}
    return skills


def extract_text_from_pdf(uploaded_file) -> str:
    reader = PdfReader(uploaded_file)
    parts: List[str] = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            parts.append(text)
    return "\n".join(parts)


def clean_text(text: str) -> str:
    if not text:
        return ""
    txt = text.replace('\r', ' ').replace('\t', ' ')
    txt = re.sub(r"[-–—]+", "-", txt)  # normalize different dashes
    txt = re.sub(r"\s+", " ", txt).strip()
    return txt

def ensure_spacy_model():
    global nlp
    if nlp is None:
        raise RuntimeError(
            "spaCy model not loaded. Run: python -m spacy download en_core_web_sm and restart."
        )


def lemmatize_text(text: str) -> str:
    """Return lemmatized lowercase text using spaCy."""
    ensure_spacy_model()
    doc = nlp(text.lower())
    return " ".join(token.lemma_ for token in doc)


def extract_named_entities(text: str, relevant_labels: Set[str] = None) -> Set[str]:
    """Extract named entities and filter by relevant labels.

    By default we keep ORG, PRODUCT, and LANGUAGE entities — adjust `relevant_labels` if desired.
    """
    ensure_spacy_model()
    if relevant_labels is None:
        relevant_labels = {"ORG", "PRODUCT", "LANGUAGE"}

    doc = nlp(text)
    ents = set()
    for ent in doc.ents:
        if ent.label_ in relevant_labels:
            ents.add(ent.text.lower())
    return ents

def fuzzy_match_skills(text: str, skill_set: Set[str], threshold: int = 85) -> Set[str]:
    text = text.lower()
    found = set()
    for skill in skill_set:
        score = fuzz.partial_ratio(skill, text)
        if score >= threshold:
            found.add(skill)
    return found


def extract_skills_advanced(text: str, skill_set: Set[str], threshold: int = 85) -> Set[str]:
    txt = clean_text(text or "")
    lemm = lemmatize_text(txt)
    exact_matches = {s for s in skill_set if s in lemm}
    fuzzy_matches = fuzzy_match_skills(lemm, skill_set, threshold)
    ner_matches = extract_named_entities(txt)
    all_matches = set(map(str.lower, exact_matches)) | set(map(str.lower, fuzzy_matches)) | set(map(str.lower, ner_matches))
    cleaned = {s for s in all_matches if s in skill_set}
    return cleaned

def load_embedding_model(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(model_name)
    return _embedding_model

def find_missing_skills(resume_text, jd_text, skill_set):
    resume_skills = extract_skills_advanced(resume_text, skill_set)
    jd_skills_raw = extract_skills_advanced(jd_text, skill_set)
    jd_skills = set(skill for skill in jd_skills_raw if skill in skill_set)
    matched_skills = resume_skills & jd_skills
    missing_skills = jd_skills - resume_skills
    match_score = round((len(matched_skills) / max(len(jd_skills), 1)) * 100, 1)
    return sorted(resume_skills), sorted(jd_skills), sorted(missing_skills), match_score



def embed_text(text: str, model: SentenceTransformer = None) -> np.ndarray:
    if model is None:
        model = load_embedding_model()
    emb = model.encode(str(text), convert_to_tensor=False)
    return np.asarray(emb, dtype=np.float32)

def compute_features(resume_text: str, jd_text: str, skill_set: Set[str], model: SentenceTransformer = None) -> Dict[str, Any]:
    resume_skills = extract_skills_advanced(resume_text, skill_set)
    jd_skills = extract_skills_advanced(jd_text, skill_set)
    skill_overlap = len(resume_skills & jd_skills)
    missing_skills = len(jd_skills - resume_skills)
    if model is None:
        model = load_embedding_model()
    emb1 = embed_text(resume_text, model)
    emb2 = embed_text(jd_text, model)
    norm1 = np.linalg.norm(emb1)
    norm2 = np.linalg.norm(emb2)
    bert_sim = float(np.dot(emb1, emb2) / (max(1e-8, norm1 * norm2)))
    resume_len = len(clean_text(resume_text).split())
    jd_len = len(clean_text(jd_text).split())
    skill_density = skill_overlap / max(1, len(jd_skills))

    return {
        "skill_overlap": skill_overlap,
        "missing_skills": missing_skills,
        "bert_similarity": bert_sim,
        "resume_len": resume_len,
        "jd_len": jd_len,
        "skill_density": skill_density,
        "resume_skills": sorted(resume_skills),
        "jd_skills": sorted(jd_skills),
    }
    
def highlight_skills(text: str, skills: List[str], color: str = "#e6f7ff") -> str:
    if not text:
        return ""
    highlighted = text
    for skill in sorted(set(skills), key=lambda s: -len(s)):
        if not skill:
            continue
        safe_skill = re.escape(skill)
        pattern = re.compile(rf"\b({safe_skill})\b", flags=re.IGNORECASE)
        highlighted = pattern.sub(rf"<span style='background-color: {color}; padding:2px'>\1</span>", highlighted)
    return highlighted

__all__ = [
    "load_skill_set",
    "extract_text_from_pdf",
    "clean_text",
    "lemmatize_text",
    "extract_named_entities",
    "fuzzy_match_skills",
    "extract_skills_advanced",
    "load_embedding_model",
    "embed_text",
    "compute_features",
    "highlight_skills",
]
