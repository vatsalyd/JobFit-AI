# 💼 JobFit AI

🚀 **JOBFIT AI** is an intelligent resume–job description (JD) matching system.  
It extracts skills, compares resumes against job postings, and predicts a **match score** using **rule-based, machine learning, and fine-tuned deep learning models**.  

---

## ✨ Features  

- 📄 **Resume Parsing (PDF)** – extract and clean text from resumes.  
- 🛠 **Skill Extraction** – using SpaCy, fuzzy matching, and lemmatization.  
- 📊 **Rule-based Match Score** – quick overlap-based scoring.  
- 🤖 **Machine Learning Model** – trained with handcrafted features (skill overlap, density, lengths, etc.).  
- 🧠 **Deep Learning Model** – fine-tuned **Sentence-BERT (all-MiniLM-L6-v2)** for semantic similarity.  
- ❌ **Missing Skills Detection** – highlights gaps between JD and resume.  
- 🌐 **Streamlit Web App** – interactive UI to upload resume & paste JD text.  

---

## 📂 Project Structure  

```
JOBFIT-AI/
│── app.py                   # Streamlit app
│── utils.py                 # Skill extraction & feature utilities
│── dl_model_wrapper.py      # Wrapper for deep learning model
│── train_deep_model.py      # Script to fine-tune SBERT
│── ml_train.ipynb           # Training ML-based model
│── text_extraction.ipynb    # Resume/JD parsing experiments
│── data/
│   ├── skills.txt           # List of technical skills
│   ├── real_dataset.csv     # Real-world dataset
│   ├── synthetic_clean.csv  # Synthetic dataset for training
│── models/
│   ├── ml_model.joblib      # Trained ML model
│   ├── dl_resume_match/     # Fine-tuned deep learning model
│── requirements.txt         # Dependencies
│── README.md                # Project documentation
```

---

## ⚡ Installation  

Clone the repo and install dependencies:  

```bash
git clone https://github.com/your-username/JOBFIT-AI.git
cd JOBFIT-AI
pip install -r requirements.txt
```

---

## 🚀 Run the App  

```bash
streamlit run app.py
```

Then open your browser at **http://localhost:8501**.  

---

## 🧠 Models  

- **Rule-based Model** → Uses skill overlap ratio.  
- **ML Model** → Trained using Random Forests / XGBoost on engineered features.  
- **Deep Learning Model** → Fine-tuned **Sentence-BERT (all-MiniLM-L6-v2)** to predict match score on a 0–100 scale.  

---

## 📊 Example Output  

- ✅ **Resume Skills Extracted**  
- 📌 **JD Skills Extracted**  
- ❌ **Missing Skills**  
- 📈 **Match Scores** (Rule-based, ML-based, Deep Learning)  

---

## 🔮 Future Improvements  

- Add Named Entity Recognition (NER) with domain-specific models.  
- Expand skill dictionary with emerging tech skills.  
- Deploy as an **ATS (Applicant Tracking System)** SaaS with multiple resumes.  

---

## 🤝 Contributing  

Pull requests are welcome! For major changes, please open an issue first to discuss.  

---

## 📜 License  

MIT License © 2025  
