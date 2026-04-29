import re
from pathlib import Path
from typing import Set, List, Dict, Any, Optional

import numpy as np
import spacy
from rapidfuzz import fuzz
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ── Globals ──────────────────────────────────────────────────────────────────
_embedding_model: Optional[SentenceTransformer] = None

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# Skills that are too short for fuzzy matching (≤2 chars) — use exact word-boundary only
_SHORT_SKILL_THRESHOLD = 2


# ── Skill Loading ────────────────────────────────────────────────────────────
def load_skill_set(file_path: str) -> Set[str]:
    p = Path(file_path)
    if not p.exists():
        raise FileNotFoundError(f"Skill file not found: {file_path}")
    with p.open("r", encoding="utf-8") as f:
        skills = {line.strip().lower() for line in f if line.strip()}
    return skills


# ── PDF Extraction ───────────────────────────────────────────────────────────
def extract_text_from_pdf(uploaded_file) -> str:
    """Extract text from a PDF file, with error handling for corrupt files."""
    try:
        reader = PdfReader(uploaded_file)
        parts: List[str] = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                parts.append(text)
        result = "\n".join(parts)
        if not result.strip():
            raise ValueError("PDF appears to contain no extractable text.")
        return result
    except Exception as e:
        raise RuntimeError(f"Failed to parse PDF: {e}") from e


# ── Text Cleaning ────────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    if not text:
        return ""
    txt = text.replace('\r', ' ').replace('\t', ' ')
    txt = re.sub(r"[-–—]+", "-", txt)          # normalize dashes
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


# ── Named Entity Extraction ─────────────────────────────────────────────────
def extract_named_entities(text: str, relevant_labels: Set[str] = None) -> Set[str]:
    """Extract named entities filtered by label type."""
    ensure_spacy_model()
    if relevant_labels is None:
        relevant_labels = {"ORG", "PRODUCT", "LANGUAGE"}
    doc = nlp(text)
    ents = set()
    for ent in doc.ents:
        if ent.label_ in relevant_labels:
            ents.add(ent.text.lower())
    return ents


# ── Skill Extraction (FIXED: no more short-skill false positives) ────────────
def _exact_word_boundary_match(skill: str, text: str) -> bool:
    """Check if a skill appears as a whole word/phrase in text."""
    pattern = re.compile(rf"\b{re.escape(skill)}\b", re.IGNORECASE)
    return bool(pattern.search(text))


def fuzzy_match_skills(text: str, skill_set: Set[str], threshold: int = 90) -> Set[str]:
    """Fuzzy-match skills against text.

    Short skills (≤2 chars) are skipped here — they are matched only via
    exact word-boundary matching in `extract_skills_advanced()`.
    Threshold raised from 85 → 90 to reduce false positives.
    """
    text = text.lower()
    found = set()
    for skill in skill_set:
        if len(skill) <= _SHORT_SKILL_THRESHOLD:
            continue  # short skills handled via exact match only
        # Use token_sort_ratio for multi-word skills, partial_ratio for single-word
        if " " in skill:
            score = fuzz.token_sort_ratio(skill, text)
        else:
            score = fuzz.partial_ratio(skill, text)
        if score >= threshold:
            found.add(skill)
    return found


def extract_skills_advanced(text: str, skill_set: Set[str], threshold: int = 90) -> Set[str]:
    """Extract skills from text using exact match, fuzzy match, and NER.

    Short skills (≤2 chars like 'c', 'r', 'go') use word-boundary exact match
    to avoid catastrophic false positives from fuzzy matching.
    """
    txt = clean_text(text or "")
    lemm = lemmatize_text(txt)

    # --- Exact substring match in lemmatized text (long skills) ---
    exact_matches = {s for s in skill_set if len(s) > _SHORT_SKILL_THRESHOLD and s in lemm}

    # --- Word-boundary exact match for short skills ---
    short_exact = {
        s for s in skill_set
        if len(s) <= _SHORT_SKILL_THRESHOLD and _exact_word_boundary_match(s, txt)
    }

    # --- Fuzzy match (long skills only, higher threshold) ---
    fuzzy_matches = fuzzy_match_skills(lemm, skill_set, threshold)

    # --- NER (keep only those in our known skill set) ---
    ner_matches = extract_named_entities(txt)

    all_matches = (
        set(map(str.lower, exact_matches))
        | set(map(str.lower, short_exact))
        | set(map(str.lower, fuzzy_matches))
        | set(map(str.lower, ner_matches))
    )
    return {s for s in all_matches if s in skill_set}


# ── Embedding Utilities ─────────────────────────────────────────────────────
def load_embedding_model(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(model_name)
    return _embedding_model


def embed_text(text: str, model: SentenceTransformer = None) -> np.ndarray:
    if model is None:
        model = load_embedding_model()
    emb = model.encode(str(text), convert_to_tensor=False)
    return np.asarray(emb, dtype=np.float32)


# ── Feature Computation ─────────────────────────────────────────────────────
def compute_tfidf_cosine(text_a: str, text_b: str) -> float:
    """Compute TF-IDF cosine similarity between two texts."""
    try:
        vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
        tfidf = vectorizer.fit_transform([text_a, text_b])
        sim = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
        return float(sim)
    except ValueError:
        return 0.0


def compute_jaccard_similarity(set_a: Set[str], set_b: Set[str]) -> float:
    """Jaccard index: |A ∩ B| / |A ∪ B|."""
    if not set_a and not set_b:
        return 0.0
    return len(set_a & set_b) / max(len(set_a | set_b), 1)


def compute_features(
    resume_text: str,
    jd_text: str,
    skill_set: Set[str],
    model: SentenceTransformer = None,
) -> Dict[str, Any]:
    """Compute a rich feature dictionary for resume–JD matching.

    Returns original features PLUS new features:
      - jaccard_similarity, tfidf_cosine
      - resume_skill_count, jd_skill_count
    """
    resume_skills = extract_skills_advanced(resume_text, skill_set)
    jd_skills = extract_skills_advanced(jd_text, skill_set)

    skill_overlap = len(resume_skills & jd_skills)
    missing_skills_count = len(jd_skills - resume_skills)

    # BERT embedding similarity
    if model is None:
        model = load_embedding_model()
    emb1 = embed_text(resume_text, model)
    emb2 = embed_text(jd_text, model)
    norm1 = np.linalg.norm(emb1)
    norm2 = np.linalg.norm(emb2)
    bert_sim = float(np.dot(emb1, emb2) / max(1e-8, norm1 * norm2))

    resume_clean = clean_text(resume_text)
    jd_clean = clean_text(jd_text)
    resume_len = len(resume_clean.split())
    jd_len = len(jd_clean.split())

    skill_density = skill_overlap / max(1, len(jd_skills))

    # ── New features ──
    jaccard_sim = compute_jaccard_similarity(resume_skills, jd_skills)
    tfidf_cos = compute_tfidf_cosine(resume_clean, jd_clean)
    resume_skill_count = len(resume_skills)
    jd_skill_count = len(jd_skills)

    return {
        # Original features
        "skill_overlap":      skill_overlap,
        "missing_skills":     missing_skills_count,
        "bert_similarity":    bert_sim,
        "resume_len":         resume_len,
        "jd_len":             jd_len,
        "skill_density":      skill_density,
        # New features
        "jaccard_similarity": jaccard_sim,
        "tfidf_cosine":       tfidf_cos,
        "resume_skill_count": resume_skill_count,
        "jd_skill_count":     jd_skill_count,
        # Metadata (not used as ML features)
        "resume_skills":      sorted(resume_skills),
        "jd_skills":          sorted(jd_skills),
    }


# ── Module exports ───────────────────────────────────────────────────────────
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
    "compute_tfidf_cosine",
    "compute_jaccard_similarity",
]
