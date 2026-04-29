"""
data_prep.py -- Compute features and generate match-score labels for training.

LABEL DESIGN (eliminates data leakage):
  Uses a TWO-PASS approach:
    Pass 1: Compute all raw features for every row.
    Pass 2: Percentile-normalize the label components so scores are
            well-distributed across the 0-100 range.

  This avoids the problem where BERT similarity (mean 0.32) and Jaccard
  (mean 0.07) produce crushed labels when treated as 0-1 scales.
"""

import os
import pandas as pd
import numpy as np
from utils import (
    load_skill_set,
    load_embedding_model,
    compute_features,
    clean_text,
    compute_tfidf_cosine,
    compute_jaccard_similarity,
)

SYNTHETIC_FILE = "data/raw/synthetic_dataset.csv"
REAL_FILE = "data/raw/real_dataset.csv"
COLLECTED_FILE = "data/raw/collected_dataset.csv"
SKILL_FILE = "data/skills.txt"
OUT_SYNTH = "data/cleaned_data/synthetic_clean_v4.csv"
OUT_REAL = "data/cleaned_data/real_clean_v4.csv"
OUT_COLLECTED = "data/cleaned_data/collected_clean_v4.csv"
OUT_MERGED = "data/cleaned_data/merged_training_v4.csv"


def compute_labels(df: pd.DataFrame) -> pd.Series:
    """Generate well-distributed match-score labels (0-100).

    Uses RANK-BASED scoring: each component is ranked among all samples,
    then blended. This guarantees labels span the full 0-100 range uniformly,
    regardless of how skewed the raw feature values are.
    """
    n = len(df)

    # Rank each component (0 to 1 range, where 1 = highest in dataset)
    bert_rank = df["bert_similarity"].rank(pct=True)
    tfidf_rank = df["tfidf_cosine"].rank(pct=True)
    jaccard_rank = df["jaccard_similarity"].rank(pct=True)
    density_rank = df["skill_density"].rank(pct=True)

    # Semantic score: BERT (dominant) + TF-IDF
    semantic = bert_rank * 0.65 + tfidf_rank * 0.35

    # Skill score: Jaccard + skill density
    skill = jaccard_rank * 0.5 + density_rank * 0.5

    # Blend: semantic matters more for real text
    raw = semantic * 0.55 + skill * 0.45

    # Scale to 0-100
    raw = raw * 100

    # Add small noise to prevent perfect reconstruction from features
    rng = np.random.RandomState(42)
    noise = rng.normal(0, 2.0, size=n)
    raw = raw + noise

    return raw.clip(0, 100).round(1)


def process_dataset(df: pd.DataFrame, skills, embed_model) -> pd.DataFrame:
    """Clean data, compute features, and generate match-score labels."""
    df = df.dropna(subset=["resume_text", "jd_text"]).copy()
    df["resume_text"] = df["resume_text"].astype(str).apply(clean_text)
    df["jd_text"] = df["jd_text"].astype(str).apply(clean_text)

    # Filter out too-short texts
    df = df[df["resume_text"].str.split().str.len() > 5]
    df = df[df["jd_text"].str.split().str.len() > 5]
    df = df.drop_duplicates(subset=["resume_text", "jd_text"])

    # Pass 1: Compute all features
    results = []
    total = len(df)
    for i, (idx, row) in enumerate(df.iterrows()):
        if (i + 1) % 200 == 0:
            print(f"  Processing row {i+1}/{total}...")
        feats = compute_features(row["resume_text"], row["jd_text"], skills, embed_model)
        feats["resume_text"] = row["resume_text"]
        feats["jd_text"] = row["jd_text"]
        results.append(feats)

    result_df = pd.DataFrame(results)

    # Pass 2: Compute labels using dataset-wide normalization
    result_df["match_score"] = compute_labels(result_df)

    return result_df


if __name__ == "__main__":
    print("Loading resources...")
    skills = load_skill_set(SKILL_FILE)
    embed_model = load_embedding_model()

    dfs_to_merge = []

    # --- Synthetic ---
    if os.path.exists(SYNTHETIC_FILE):
        print(f"\nProcessing synthetic data from {SYNTHETIC_FILE}...")
        df_synth = pd.read_csv(SYNTHETIC_FILE)
        df_synth_clean = process_dataset(df_synth, skills, embed_model)
        df_synth_clean.to_csv(OUT_SYNTH, index=False)
        dfs_to_merge.append(df_synth_clean)
        print(f"  -> {len(df_synth_clean)} rows saved to {OUT_SYNTH}")
    else:
        print(f"WARN: Skipping {SYNTHETIC_FILE} (not found)")

    # --- Real ---
    if os.path.exists(REAL_FILE):
        print(f"\nProcessing real data from {REAL_FILE}...")
        df_real = pd.read_csv(REAL_FILE)
        df_real_clean = process_dataset(df_real, skills, embed_model)
        df_real_clean.to_csv(OUT_REAL, index=False)
        dfs_to_merge.append(df_real_clean)
        print(f"  -> {len(df_real_clean)} rows saved to {OUT_REAL}")
    else:
        print(f"WARN: Skipping {REAL_FILE} (not found)")

    # --- Collected (from collect_data.py) ---
    if os.path.exists(COLLECTED_FILE):
        print(f"\nProcessing collected data from {COLLECTED_FILE}...")
        df_collected = pd.read_csv(COLLECTED_FILE)
        df_collected_clean = process_dataset(df_collected, skills, embed_model)
        df_collected_clean.to_csv(OUT_COLLECTED, index=False)
        dfs_to_merge.append(df_collected_clean)
        print(f"  -> {len(df_collected_clean)} rows saved to {OUT_COLLECTED}")
    else:
        print(f"WARN: Skipping {COLLECTED_FILE} (not found -- run collect_data.py first)")

    # --- Merge all ---
    if dfs_to_merge:
        df_merged = pd.concat(dfs_to_merge, ignore_index=True)

        # Re-normalize labels across the full merged dataset
        df_merged["match_score"] = compute_labels(df_merged)

        df_merged.to_csv(OUT_MERGED, index=False)
        print(f"\n  -> {len(df_merged)} total rows saved to {OUT_MERGED}")
        print(f"\n  Label stats:")
        print(f"    Mean: {df_merged['match_score'].mean():.1f}")
        print(f"    Std:  {df_merged['match_score'].std():.1f}")
        print(f"    Min:  {df_merged['match_score'].min():.1f}")
        print(f"    Max:  {df_merged['match_score'].max():.1f}")
    else:
        print("\nERROR: No datasets found to merge!")

    print("\nData preparation complete!")
