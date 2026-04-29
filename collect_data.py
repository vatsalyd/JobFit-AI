"""
collect_data.py — Build real resume–JD training pairs from locally downloaded Kaggle datasets.

Expected files in data/raw/:
  - Resume/Resume.csv            (Kaggle: snehaanbhawal/resume-dataset)
  - DataScientist.csv            (Kaggle: andrewmvd/data-scientist-jobs)
  - job_descriptions.csv         (Kaggle: ravindrasinghrana/job-description-dataset)

Usage:
    python collect_data.py
"""

import os
import re
import random
import pandas as pd

# ── Config ───────────────────────────────────────────────────────────────────
RESUME_FILE = "data/raw/Resume/Resume.csv"
DS_JD_FILE = "data/raw/DataScientist.csv"
MULTI_JD_FILE = "data/raw/job_descriptions.csv"

OUTPUT_FILE = "data/raw/collected_dataset.csv"

PAIRS_PER_MATCH = 10      # matched-domain pairs per resume
PAIRS_PER_MISMATCH = 3    # mismatched-domain pairs per resume
MAX_RESUMES_PER_CAT = 50  # cap resumes per category to keep dataset balanced
MAX_JD_ROWS = 30000       # increased cap for better coverage
RANDOM_SEED = 42

# Map ALL 24 resume categories → JD search keywords
# Keywords are searched in BOTH role column AND full JD text
CATEGORY_TO_KEYWORDS = {
    "accountant":              ["accountant", "accounting", "bookkeeper", "auditor", "tax", "financial analyst", "finance"],
    "advocate":                ["lawyer", "attorney", "legal", "paralegal", "advocate", "law firm", "litigation", "family law"],
    "agriculture":             ["agriculture", "farm", "agronomist", "crop", "horticulture", "agricultural", "livestock"],
    "apparel":                 ["apparel", "fashion", "textile", "garment", "merchandise", "retail buyer", "clothing"],
    "arts":                    ["graphic designer", "illustrator", "creative director", "visual designer", "art director", "animation", "interaction designer"],
    "automobile":              ["automobile", "automotive", "vehicle", "mechanic", "car", "dealership", "auto repair"],
    "aviation":                ["aviation", "airline", "pilot", "aircraft", "flight", "airport", "aerospace"],
    "banking":                 ["banking", "bank", "loan", "mortgage", "credit", "investment", "financial advisor", "portfolio manager"],
    "bpo":                     ["bpo", "call center", "customer service", "customer support", "telemarketing", "customer success"],
    "business-development":    ["business development", "account executive", "sales manager", "partnership", "client relations", "inside sales"],
    "chef":                    ["chef", "cook", "culinary", "restaurant", "food", "kitchen", "catering", "bakery"],
    "construction":            ["construction", "civil engineer", "structural", "building", "site engineer", "surveyor", "contractor"],
    "consultant":              ["consultant", "consulting", "advisory", "strategy", "management consultant", "business analyst"],
    "designer":                ["designer", "ui designer", "ux designer", "product designer", "user experience", "user interface", "web designer", "ux/ui"],
    "digital-media":           ["digital media", "social media", "content creator", "content marketing", "digital marketing", "seo", "social media manager"],
    "engineering":             ["engineer", "engineering", "devops", "backend developer", "frontend developer", "software", "data engineer", "systems"],
    "finance":                 ["finance", "financial", "investment", "portfolio", "wealth", "retirement planner", "fund", "equity"],
    "fitness":                 ["fitness", "personal trainer", "gym", "wellness", "health coach", "sports", "exercise", "yoga"],
    "healthcare":              ["healthcare", "nurse", "medical", "doctor", "clinical", "hospital", "patient", "pharmacy", "health"],
    "hr":                      ["human resources", "hr ", "recruiter", "talent acquisition", "training coordinator", "benefits coordinator", "payroll"],
    "information-technology":  ["information technology", "it manager", "it project", "systems administrator", "network administrator", "it support", "help desk", "it analyst"],
    "public-relations":        ["public relations", "communications", "pr manager", "media relations", "press", "corporate communications", "event planner"],
    "sales":                   ["sales", "account executive", "sales manager", "sales representative", "inside sales", "business development", "account manager"],
    "teacher":                 ["teacher", "instructor", "professor", "tutor", "education", "training", "academic", "teaching"],
}


def clean_html(text: str) -> str:
    """Strip HTML tags and excessive whitespace."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_resumes() -> pd.DataFrame:
    """Load Resume.csv → columns: [category, resume_text]."""
    print(f"📄 Loading resumes from {RESUME_FILE}...")
    df = pd.read_csv(RESUME_FILE)
    df = df.rename(columns={"Resume_str": "resume_text", "Category": "category"})
    df["resume_text"] = df["resume_text"].apply(clean_html)
    df["category"] = df["category"].str.lower().str.strip()

    # Filter out very short resumes
    df = df[df["resume_text"].str.split().str.len() > 50]

    print(f"  ✅ {len(df)} resumes, {df['category'].nunique()} categories")
    print(f"  Avg length: {df['resume_text'].str.split().str.len().mean():.0f} words")
    return df[["category", "resume_text"]]


def load_jds() -> pd.DataFrame:
    """Load and combine both JD datasets → columns: [role, jd_text]."""
    jd_frames = []

    # --- DataScientist.csv (high-quality, longer JDs) ---
    if os.path.exists(DS_JD_FILE):
        print(f"\n📋 Loading JDs from {DS_JD_FILE}...")
        ds = pd.read_csv(DS_JD_FILE)
        ds = ds.rename(columns={"Job Title": "role", "Job Description": "jd_text"})
        ds["jd_text"] = ds["jd_text"].apply(clean_html)
        ds["role"] = ds["role"].fillna("").str.lower().str.strip()
        ds = ds[ds["jd_text"].str.split().str.len() > 50]
        jd_frames.append(ds[["role", "jd_text"]])
        print(f"  ✅ {len(ds)} JDs, avg {ds['jd_text'].str.split().str.len().mean():.0f} words")

    # --- job_descriptions.csv (big file — combine description + responsibilities) ---
    if os.path.exists(MULTI_JD_FILE):
        print(f"\n📋 Loading JDs from {MULTI_JD_FILE} (sampling {MAX_JD_ROWS} rows)...")
        chunks = pd.read_csv(
            MULTI_JD_FILE,
            usecols=["Job Title", "Role", "Job Description", "Responsibilities", "skills"],
            nrows=MAX_JD_ROWS,
        )
        # Combine description + responsibilities for longer JD text
        chunks["jd_text"] = (
            chunks["Job Description"].fillna("").astype(str) + " " +
            chunks["Responsibilities"].fillna("").astype(str) + " " +
            "Required skills: " + chunks["skills"].fillna("").astype(str)
        )
        chunks["jd_text"] = chunks["jd_text"].apply(clean_html)
        chunks["role"] = chunks["Role"].fillna(chunks["Job Title"]).fillna("").str.lower().str.strip()
        chunks = chunks[chunks["jd_text"].str.split().str.len() > 30]
        jd_frames.append(chunks[["role", "jd_text"]])
        print(f"  ✅ {len(chunks)} JDs, avg {chunks['jd_text'].str.split().str.len().mean():.0f} words")

    if not jd_frames:
        raise FileNotFoundError("No JD datasets found in data/raw/!")

    combined = pd.concat(jd_frames, ignore_index=True)
    print(f"\n  📊 Total JDs: {len(combined)}")
    return combined


def find_jds_for_category(jd_df: pd.DataFrame, keywords: list, n: int) -> pd.DataFrame:
    """Find JDs whose role OR jd_text matches any keyword."""
    mask = pd.Series(False, index=jd_df.index)
    for kw in keywords:
        # Search in BOTH role and full JD text for better recall
        mask |= jd_df["role"].str.contains(kw, case=False, na=False, regex=False)
        mask |= jd_df["jd_text"].str.contains(kw, case=False, na=False, regex=False)
    matches = jd_df[mask]
    if len(matches) == 0:
        return pd.DataFrame(columns=jd_df.columns)
    return matches.sample(n=min(n, len(matches)), random_state=RANDOM_SEED)


def create_pairs(resume_df: pd.DataFrame, jd_df: pd.DataFrame) -> pd.DataFrame:
    """Create matched and mismatched resume–JD pairs."""
    random.seed(RANDOM_SEED)
    pairs = []
    categories = resume_df["category"].unique()

    for cat in categories:
        cat_resumes = resume_df[resume_df["category"] == cat]
        if len(cat_resumes) > MAX_RESUMES_PER_CAT:
            cat_resumes = cat_resumes.sample(MAX_RESUMES_PER_CAT, random_state=RANDOM_SEED)

        keywords = CATEGORY_TO_KEYWORDS.get(cat, [cat])
        matched_jds = find_jds_for_category(jd_df, keywords, n=300)

        # Pick keywords from distant categories for mismatches
        distant_cats = [c for c in categories if c != cat]
        random.shuffle(distant_cats)
        other_keywords = []
        for dc in distant_cats[:5]:
            other_keywords.extend(CATEGORY_TO_KEYWORDS.get(dc, [dc])[:2])
        mismatched_jds = find_jds_for_category(jd_df, other_keywords, n=150)

        cat_match = 0
        cat_mismatch = 0

        for _, r_row in cat_resumes.iterrows():
            # --- Matched pairs ---
            if len(matched_jds) > 0:
                sample_n = min(PAIRS_PER_MATCH, len(matched_jds))
                for _, jd_row in matched_jds.sample(sample_n, random_state=random.randint(0, 9999)).iterrows():
                    pairs.append({
                        "resume_text": r_row["resume_text"],
                        "jd_text": jd_row["jd_text"],
                        "resume_category": cat,
                        "pair_type": "match",
                    })
                    cat_match += 1

            # --- Mismatched pairs ---
            if len(mismatched_jds) > 0:
                sample_n = min(PAIRS_PER_MISMATCH, len(mismatched_jds))
                for _, jd_row in mismatched_jds.sample(sample_n, random_state=random.randint(0, 9999)).iterrows():
                    pairs.append({
                        "resume_text": r_row["resume_text"],
                        "jd_text": jd_row["jd_text"],
                        "resume_category": cat,
                        "pair_type": "mismatch",
                    })
                    cat_mismatch += 1

        status = "✅" if cat_match > 0 else "❌"
        print(f"  {status} {cat:<30s}  {len(cat_resumes)} resumes → {cat_match} match + {cat_mismatch} mismatch")

    return pd.DataFrame(pairs)


def main():
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    resume_df = load_resumes()
    jd_df = load_jds()

    print(f"\n🔗 Creating resume–JD pairs...")
    pairs_df = create_pairs(resume_df, jd_df)
    pairs_df = pairs_df.drop_duplicates(subset=["resume_text", "jd_text"])

    match_count = int((pairs_df["pair_type"] == "match").sum())
    mismatch_count = int((pairs_df["pair_type"] == "mismatch").sum())
    avg_resume = pairs_df["resume_text"].str.split().str.len().mean()
    avg_jd = pairs_df["jd_text"].str.split().str.len().mean()

    print(f"\n{'─' * 55}")
    print(f"  Total pairs:       {len(pairs_df)}")
    print(f"  Matching pairs:    {match_count}")
    print(f"  Mismatching pairs: {mismatch_count}")
    print(f"  Avg resume length: {avg_resume:.0f} words")
    print(f"  Avg JD length:     {avg_jd:.0f} words")
    print(f"{'─' * 55}")

    pairs_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n💾 Saved → {OUTPUT_FILE}")
    print(f"\n✅ Next steps:")
    print(f"   1. python data_prep.py   (compute features + labels)")
    print(f"   2. python train_ml.py    (train ML model)")
    print(f"   3. streamlit run app.py  (run the app)")


if __name__ == "__main__":
    main()
