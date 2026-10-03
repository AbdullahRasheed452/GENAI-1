# Resume Rater & Classifier

An NLP-powered tool that analyzes how well a resume matches a job description — and predicts the resume's job category using a trained machine learning model.

Paste in a resume and a job description, and get:
- A **match score** (how closely the resume aligns with the job description)
- **Missing keywords** — important terms from the job description that don't appear in the resume
- A **predicted job category** (e.g. HR, Engineering, Sales) — or an honest "uncertain" flag when the resume doesn't clearly match any category the model was trained on

## Demo

*(Add a screenshot or GIF of the Gradio app here, and a link to your live Hugging Face Spaces demo once deployed)*

## How it works

The project combines two techniques:

**1. Resume-to-job matching (TF-IDF + Cosine Similarity)**
The resume and job description are converted into TF-IDF vectors (numerical representations that weigh words by importance), and compared using cosine similarity to produce a match score. The tool also identifies important job-description keywords that are missing from the resume.

**2. Job category classification (trained ML model)**
A Linear SVM classifier, trained on ~2,700 labeled resumes across 32 job categories, predicts what field a resume belongs to. The model uses TF-IDF features with bigrams (word pairs) and class balancing to handle uneven category sizes.

To avoid confidently guessing wrong on resumes outside its training categories (e.g. a retail resume being forced into "Aviation"), the model only returns a category prediction when its confidence score is above a calibrated threshold — otherwise it reports "uncertain" rather than a misleading guess.

## Model performance

- **Accuracy:** ~71% across 32 categories (a random guess would score ~3%)
- Iteratively improved from a 65% baseline through:
  - Comparing Logistic Regression vs. Linear SVM (SVM performed notably better on high-dimensional text data)
  - Class balancing to reduce bias toward larger categories
  - Adding bigrams (word-pair features) to capture phrases like "customer service" as single units
- **Strongest categories:** roles with distinctive technical vocabulary (e.g. Cloud Engineer, Data Scientist, HR, Chef) — precision/recall above 0.85 even with limited training samples
- **Weakest categories:** roles with overlapping business language (e.g. Consultant vs. Business Development vs. Sales), where vocabulary genuinely overlaps

## Known limitations

- The classifier can only predict categories it was trained on. Resumes from fields outside those 32 categories (e.g. Retail, Hospitality) will be flagged "uncertain" rather than forced into an incorrect category — a deliberate design choice, not a bug.
- Some categories have small sample sizes (under 20 resumes), which limits reliability for those specific predictions.
- The match score is based on keyword/phrase overlap, not deep semantic understanding — a resume could be a strong real-world fit while scoring lower if it uses different terminology than the job description.

## Tech stack

- Python, pandas, scikit-learn (TF-IDF, Linear SVM, Logistic Regression)
- Gradio (interactive web app)
- Jupyter Notebook (exploration and model development)

## Project structure

```
resume_analyzer/
├── notebooks/
│   └── 01_exploration.ipynb   # EDA, cleaning, model training & evaluation
├── src/
│   ├── app.py                  # Gradio app
│   ├── classifier_model.pkl    # Trained SVM model
│   └── tfidf_vectorizer.pkl    # Fitted TF-IDF vectorizer
├── data/                       # Dataset (not tracked in git)
├── requirements.txt
└── README.md
```

## Running locally

```bash
git clone https://github.com/YOUR_USERNAME/resume_analyzer.git
cd resume_analyzer
pip install -r requirements.txt
cd src
python app.py
```

Then open the local URL Gradio provides (typically `http://127.0.0.1:7860`).

## What I learned

This was my first end-to-end ML project, and it taught me the full workflow beyond just calling `.fit()`:
- Real-world text data is messy — cleaning and preprocessing takes real thought, not just boilerplate code
- Model choice matters: switching from Logistic Regression to SVM alone improved accuracy by 6 points
- A model is only as honest as its handling of uncertainty — forcing predictions outside a model's training scope produces confidently wrong answers, which a calibrated confidence threshold fixes
- Data quality matters more than data quantity — I evaluated and rejected a synthetic dataset that would have taught the model false patterns

## Future improvements

- Expand category coverage with more real-world data (e.g. Retail, Customer Service, Education)
- Try sentence embeddings (e.g. `sentence-transformers`) for more semantic matching, beyond keyword overlap
- Deploy publicly with persistent hosting
