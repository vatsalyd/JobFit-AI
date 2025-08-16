import streamlit as st
import joblib
from pathlib import Path
from utils import (
    load_skill_set,
    extract_text_from_pdf,
    clean_text,
    compute_features
)
from dl_model_wrapper import DLModelWrapper

SKILL_FILE = "data/skills.txt"
ML_MODEL_FILE = "models/ml_model.joblib"
DL_MODEL_DIR = "models/dl_resume_match"

st.set_page_config(page_title="JobFtt AI", layout="wide")
st.title("📄 JobFit AI")


@st.cache_resource
def load_resources():
    skills = load_skill_set(SKILL_FILE)
    ml_model = joblib.load(ML_MODEL_FILE) if Path(ML_MODEL_FILE).exists() else None
    dl_model = DLModelWrapper(DL_MODEL_DIR) if Path(DL_MODEL_DIR).exists() else None
    return skills, ml_model, dl_model
skills, ml_model, dl_model = load_resources()

resume_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])
job_description = st.text_area("Paste Job Description here")

if st.button("Analyze"):
    if not resume_file or not job_description.strip():
        st.warning("Please upload a resume and paste a job description.")
    else:
        resume_text = clean_text(extract_text_from_pdf(resume_file))
        jd_text = clean_text(job_description)

        feats = compute_features(resume_text, jd_text, skills, None)
        resume_skills = feats["resume_skills"]
        jd_skills = feats["jd_skills"]
        missing_skills = sorted(set(jd_skills) - set(resume_skills))

        rule_score = round(feats["skill_overlap"] / max(1, len(jd_skills)) * 100, 1)

        ml_score = None
        if ml_model:
            feat_vector = [
                feats["skill_overlap"],
                feats["missing_skills"],
                feats["bert_similarity"],
                feats["resume_len"],
                feats["jd_len"],
                feats["skill_density"],
            ]
            ml_score = round(float(ml_model.predict([feat_vector])[0]), 1)

        dl_score = None
        if dl_model:
            dl_score = dl_model.predict(resume_text, jd_text)

        st.subheader("Match Scores")
        col1, col2, col3 = st.columns(3)
        col1.metric("Rule-based", f"{rule_score}%")
        col2.metric("ML-based", f"{ml_score}%" if ml_score is not None else "N/A")
        col3.metric("Deep Learning", f"{dl_score}%" if dl_score is not None else "N/A")
        st.subheader("✅ Skills in Resume")
        st.write(", ".join(resume_skills) if resume_skills else "No skills detected.")
        st.subheader("📌 Skills in Job Description")
        st.write(", ".join(jd_skills) if jd_skills else "No skills detected.")
        st.subheader("❌ Missing Skills")
        st.write(", ".join(missing_skills) if missing_skills else "All JD skills are present in the resume!")
