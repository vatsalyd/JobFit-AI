# JobFit AI

**JobFit AI** is an intelligent resume-job description matching system.
It extracts skills, compares resumes against job postings, and predicts a **match score** using **rule-based, machine learning, and fine-tuned deep learning models**.

---

## Features

- **Resume Parsing (PDF)** - extract and clean text from resumes with error handling.
- **Skill Extraction** - SpaCy NER + word-boundary exact matching + fuzzy matching (430+ skills across 24 domains).
- **Rule-based Match Score** - skill overlap-based scoring.
- **ML Model** - GradientBoosting (best of RF/GBR/XGBoost via GridSearchCV) on 10 engineered features.
- **Deep Learning Model** - fine-tuned Sentence-BERT dual-encoder with deeper regression head.
- **Missing Skills Detection** - highlights gaps between JD and resume.
- **Streamlit Web App** - interactive UI to upload resume and paste JD text.

---

## Project Structure

```
JOBFIT-AI/
|-- app.py                   # Streamlit web app
|-- utils.py                 # Skill extraction, 10 feature computations
|-- dl_model_wrapper.py      # DL model inference wrapper
|-- data_prep.py             # Feature engineering + rank-based label generation
|-- collect_data.py          # Real data collection from Kaggle datasets
|-- train_ml.py              # ML training (RF, GBR, XGBoost + GridSearchCV)
|-- train_dl.py              # DL training (SBERT fine-tuning)
|-- Dockerfile               # Docker deployment config
|-- data/
|   |-- skills.txt           # 430+ skills across 24 job domains
|-- models/
|   |-- ml_model.joblib      # Trained GradientBoosting model
|   |-- ml_scaler.joblib     # StandardScaler for features
|   |-- ml_metrics.json      # Evaluation metrics
|   |-- dl_resume_match/     # Fine-tuned SBERT model weights
|-- requirements.txt         # Dependencies
```

---

## Quick Start

### Run Locally
```bash
git clone https://github.com/vatsalyd/JobFit-AI.git
cd JobFit-AI
pip install -r requirements.txt
streamlit run app.py
```

### Run with Docker
```bash
docker build -t jobfit-ai .
docker run -p 8501:8501 jobfit-ai
```

Open **http://localhost:8501** in your browser.

---

## Train the Models

### 1. Collect Data
```bash
python collect_data.py
```
Pairs 13,000+ real resumes with job descriptions across 24 categories from Kaggle datasets.

### 2. Prepare Features
```bash
python data_prep.py
```
Computes 10 ML features + rank-based labels for all pairs.

### 3. Train ML Model
```bash
python train_ml.py
```
Compares RandomForest, GradientBoosting, and XGBoost with GridSearchCV. Saves the best model.

### 4. Train DL Model (GPU recommended)
```bash
python train_dl.py
```
Fine-tunes SBERT with OneCycleLR, gradient clipping, and early stopping. Use `notebooks/train_dl_colab.ipynb` for free GPU on Google Colab.

---

## Models

| Model | Description |
|---|---|
| **Rule-based** | Skill overlap ratio (`matched / total_jd_skills * 100`) |
| **ML (GradientBoosting)** | Trained on 10 features with 5-fold CV, StandardScaler |
| **Deep Learning (SBERT)** | `all-MiniLM-L6-v2` dual-encoder -> `512 -> BN -> GELU -> 128 -> GELU -> 1 -> Sigmoid` |

### ML Features (10)

| Feature | Description |
|---|---|
| `skill_overlap` | Number of matched skills |
| `missing_skills` | Number of JD skills not in resume |
| `bert_similarity` | SBERT cosine similarity |
| `resume_len` | Word count of resume |
| `jd_len` | Word count of JD |
| `skill_density` | `skill_overlap / jd_skill_count` |
| `jaccard_similarity` | Jaccard index on skill sets |
| `tfidf_cosine` | TF-IDF cosine similarity |
| `resume_skill_count` | Total skills found in resume |
| `jd_skill_count` | Total skills in JD |

---

## Deployment

Deployable on AWS EC2 with Docker:

```bash
# On EC2 (Ubuntu)
git clone https://github.com/vatsalyd/JobFit-AI.git
cd JobFit-AI
docker build -t jobfit-ai .
docker run -d --restart unless-stopped -p 8501:8501 jobfit-ai
```

---

## Tech Stack

- **Frontend**: Streamlit
- **ML**: scikit-learn, XGBoost
- **DL**: PyTorch, Sentence-Transformers (SBERT)
- **NLP**: spaCy, RapidFuzz, TF-IDF
- **Deployment**: Docker, AWS EC2

---

## License

MIT License
