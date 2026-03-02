# EduSense — Aspect-Based Sentiment Analysis for Educational Feedback

## What This System Does
Analyzes student feedback and identifies sentiment for 5 aspects:
- Course Content, Teaching Quality, Assessment, Resources, Infrastructure

**Input:** "Great course content but difficult exams and poor lab equipment"  
**Output:**
- Course Content → 🟢 Positive
- Assessment → 🔴 Negative
- Infrastructure → 🔴 Negative

---

## Setup (5 minutes)

```bash
pip install -r requirements.txt
python -m textblob.download_corpora
```

---

## Run the Project

### Option A — Run everything at once
```bash
python run_all.py
```

### Option B — Step by step
```bash
python step1_prepare_data.py     # Create and preprocess dataset
python step2_aspect_extraction.py # Demo keyword-based aspect detection
python step3_train_models.py      # Train NB, SVM, Random Forest
python step4_baseline_comparison.py # Compare vs TextBlob and VADER
```

### Launch Dashboard
```bash
streamlit run dashboard.py
```
Open http://localhost:8501 in your browser.

---

## File Structure

```
absa_project/
├── step1_prepare_data.py       # Dataset creation + preprocessing
├── step2_aspect_extraction.py  # Keyword-based aspect detection
├── step3_train_models.py       # ML model training (NB / SVM / RF)
├── step4_baseline_comparison.py # Compare with TextBlob & VADER
├── predictor.py                # Core prediction logic (used by dashboard)
├── dashboard.py                # Streamlit web dashboard
├── run_all.py                  # Run entire pipeline
├── requirements.txt
│
├── data/
│   └── dataset.csv             # Generated dataset (70 labeled reviews)
│
├── models/
│   └── *_model.pkl             # Trained model per aspect (5 files)
│
└── outputs/
    ├── model_comparison.csv    # NB vs SVM vs RF accuracy
    └── baseline_comparison.csv # Our system vs TextBlob vs VADER
```

---

## Technology Stack
- **NLP:** NLTK (preprocessing), keyword matching (aspect extraction)
- **ML:** scikit-learn (Naive Bayes, SVM, Random Forest), TF-IDF features
- **Baselines:** TextBlob, VADER
- **Dashboard:** Streamlit

---

## Expected Results
- Accuracy: 75–85% per aspect
- Improvement over TextBlob: +15–25%
- Dashboard: Single review in <1 second, bulk 1000 reviews in <2 minutes
