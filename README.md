# Mental Health Detection

A Streamlit-based application that predicts a mental health-related category from user text using a calibrated SVM model with hybrid TF‑IDF features (word + character n‑grams).

> Disclaimer: This project is for educational purposes and is **NOT** a medical diagnosis.

## Features
- Calm Night UI theme (modern dark + aurora background)
- Predict category + top probabilities
- Bar chart (top‑5) + Donut chart distribution
- Input quality checks (unclear/gibberish text warning)
- Safety handling for **Suicidal** label (shown only if self-harm triggers exist)
- Extra non-clinical mood tags
- Sidebar developer panel

## Project Structure
```text
.
├─ app.py
├─ requirements.txt
├─ README.md
├─ .gitignore
├─ assets/
│  └─ developer.jpg
├─ models/
│  ├─ tfidf_word.joblib
│  ├─ tfidf_char.joblib
│  └─ mental_health_svm_hybrid_calibrated.joblib
├─ data/
│  ├─ mental_health.csv
│  └─ mental_health_large.csv
└─ notebook/
   └─ preprocessing_and_baseline.ipynb
```

## Run Locally

### 1) Create and activate a virtual environment
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

### 2) Install dependencies
```bash
pip install -r requirements.txt
```

### 3) Required files
Make sure these exist:

**Models**
- `models/tfidf_word.joblib`
- `models/tfidf_char.joblib`
- `models/mental_health_svm_hybrid_calibrated.joblib`

**Developer photo**
- `assets/developer.jpg`

### 4) Run the app
```bash
streamlit run app.py
```

## Developer
**MD. MOSTAFIZUR RAHMAN**  
Email: mdmrmanik1000@gmail.com