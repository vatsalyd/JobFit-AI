import pandas as pd
from utils import load_skill_set
from utils import load_embedding_model
from utils import compute_features
from utils import clean_text
SYNTHETIC_FILE = "data/raw/synthetic_dataset.csv"
REAL_FILE = "data/raw/real_dataset.csv"
SKILL_FILE = "data/skills.txt"
OUT_SYNTH = "synthetic_clean_v3.csv"
OUT_REAL = "real_clean_v3.csv"
OUT_MERGED = "merged_training_v3.csv"
skills = load_skill_set(SKILL_FILE)
embed_model = load_embedding_model()

def process_dataset(df):
    df = df.dropna(subset=["resume_text", "jd_text"]).copy()
    df["resume_text"] = df["resume_text"].astype(str).apply(clean_text)
    df["jd_text"] = df["jd_text"].astype(str).apply(clean_text)

    df = df[df["resume_text"].str.split().str.len() > 5]
    df = df[df["jd_text"].str.split().str.len() > 5]
    df = df.drop_duplicates(subset=["resume_text", "jd_text"])

    results = []
    for _, row in df.iterrows():
        feats = compute_features(row["resume_text"], row["jd_text"], skills, embed_model)

        matched_skills = feats["skill_overlap"]
        total_jd_skills = max((feats["missing_skills"]) + matched_skills, 1)
        skill_overlap_ratio = matched_skills / total_jd_skills

        length_ratio = min(feats["resume_len"] / feats["jd_len"], 1)

        match_score = (feats["bert_similarity"] * 0.5 +skill_overlap_ratio * 0.3 +length_ratio * 0.2) * 100
        match_score = round(match_score, 1)

        feats["match_score"] = match_score
        feats["resume_text"] = row["resume_text"]
        feats["jd_text"] = row["jd_text"]
        results.append(feats)

    return pd.DataFrame(results)

df_synth = pd.read_csv(SYNTHETIC_FILE)
df_real = pd.read_csv(REAL_FILE)
df_synth_clean = process_dataset(df_synth)
df_synth_clean.to_csv(OUT_SYNTH, index=False)
df_real_clean = process_dataset(df_real)
df_real_clean.to_csv(OUT_REAL, index=False)
df_merged = pd.concat([df_synth_clean, df_real_clean], ignore_index=True)
df_merged.to_csv(OUT_MERGED, index=False)
print("Data complete! ")
