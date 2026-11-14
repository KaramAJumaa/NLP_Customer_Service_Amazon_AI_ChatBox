# Intent Classification for Conversational AI (Amazon MASSIVE)

[![Course](https://img.shields.io/badge/Course-A0443404%20Natural%20Language%20Processing-blue.svg)](file:///docs/NLP_Course_Project_PBL_Description.pdf)
[![Institution](https://img.shields.io/badge/University-Middle%20East%20University%20(MEU)-darkred.svg)](https://meu.edu.jo)
[![Python](https://img.shields.io/badge/Python-3.12%20%7C%203.14-blue.svg)](https://python.org)
[![Phase](https://img.shields.io/badge/Project%20Status-Phase%201%20Complete-brightgreen.svg)](#)
[![Dataset](https://img.shields.io/badge/Dataset-Amazon%20MASSIVE%20(en--US)-orange.svg)](https://github.com/alexa/massive)

---

## 📌 Project Overview

This repository hosts **Phase 1 (Proposal & NLP Data Preparation)** of the Problem-Based Learning (PBL) project for the **Natural Language Processing course (A0443404)** at **Middle East University (MEU)**, Faculty of Information Technology.

The goal of this project is to build an end-to-end, high-performance Intent Classification system for task-oriented conversational AI and smart voice assistants. Spoken and typed user queries are often noisy, informal, and varied. This pipeline processes user utterances and accurately maps them into one of **60 target intent categories** across **18 functional domains/scenarios**.

---

## 👥 Team Members

| Name | Role | Responsibilities |
|---|---|---|
| **Karam Abdallah Abdelkarim Abu-Juma'a** | **Group Leader** | System architecture, automated data ingestion (`download_data.py`), stratified splitting logic, EDA analysis, documentation & proposal lead. |
| **Omar** | Team Member | Contraction dictionary expansion, regex text normalization, negation-aware stop-word retention logic. |
| **Osama** | Team Member | WordNet lemmatization integration, tokenization benchmarking, dataset validation, quality control. |

**Instructors:** Dr. Israa Al Badarneh · Dr. Zaid Alhalhouli  
**Semester:** First Semester 2026/2027

---

## 📂 Repository Directory Tree

```
NLP-Intent-Classification/
├── Data/                             # Dataset storage (excluded from git)
│   ├── .gitkeep                      # Preserves directory in git tracking
│   ├── massive_english.csv           # Raw downloaded en-US MASSIVE dataset (16,521 rows)
│   ├── train_cleaned.csv             # Stratified train split (80% / 13,214 rows)
│   ├── val_cleaned.csv               # Stratified validation split (10% / 1,652 rows)
│   └── test_cleaned.csv              # Stratified test split (10% / 1,655 rows)
├── docs/                             # Academic proposals & course documentation
│   ├── NLP_Course_Project_PBL_Description.pdf # Official MEU course project specification
│   ├── NLP_Project_Proposal_Form_Phase1.docx  # Phase 1 submission proposal template
│   └── Project_Proposal_Phase1.md             # Complete, formatted Phase 1 Proposal
├── outputs/                          # Automated EDA metrics & JSON artifacts
│   ├── .gitkeep                      # Preserves directory in git tracking
│   └── preprocessing_report.json     # Comprehensive preprocessing and EDA audit
├── src/                              # Core Python pipeline modules
│   ├── download_data.py              # Zero-compilation Amazon MASSIVE download script
│   └── preprocess.py                 # Preprocessing, normalization, EDA, & 80/10/10 split
├── .gitignore                        # Git exclusion rules (.venv, caches, raw CSVs)
├── README.md                         # Project documentation and reproduction guide
└── requirements.txt                  # Python dependencies
```

---

## 💾 Flash-Drive Portability (Zero Compilation Barriers)

To guarantee seamless evaluation and submission across different hardware environments and operating system letters (e.g. running directly from `D:\`, `E:\`, or `F:\` on university lab machines):

- **Anchor Relative Paths:** All Python scripts locate directories dynamically using `os.path.dirname(os.path.dirname(os.path.abspath(__file__)))`. No hardcoded drive letters or absolute paths exist anywhere in the codebase.
- **Pure Python Dependencies:** Zero compiler toolchains (such as MSVC or C++ wheels) are required. All standard operations rely on `pandas`, `numpy`, `scikit-learn`, `nltk`, and `urllib`.

---

## 📊 Dataset Specification (Amazon MASSIVE)

We utilize the **Amazon MASSIVE (en-US localization)** dataset:
- **Total Records:** 16,521 utterances.
- **Number of Classes:** 60 discrete user intents.
- **Semantic Domains:** 18 scenarios (`calendar`, `play`, `qa`, `email`, `iot`, `general`, `weather`, `transport`, `lists`, `news`, `recommendation`, `datetime`, `social`, `alarm`, `music`, `audio`, `takeaway`, `cooking`).
- **Class Balance:** Naturally imbalanced. Ranging from 1,150 samples (`calendar_set`) down to 6 samples (`cooking_query`).
- **License:** Creative Commons Attribution 4.0 International (CC-BY 4.0).

---

## ⚙️ Phase 1 Data Pipeline Implementation

### 1. Robust Data Download (`src/download_data.py`)
- Streams the compressed Amazon MASSIVE tar archive (`amazon-massive-dataset-1.0.tar.gz`) directly over HTTP using Python's standard `urllib.request`.
- Selectively extracts member `1.0/data/en-US.jsonl` in memory, avoiding saving massive multilingual archives (~1M rows in 51 other languages) to disk.
- Outputs `Data/massive_english.csv` with fields: `id`, `utt`, `intent`, `scenario`, `partition`, `locale`.

### 2. Preprocessing & Text Normalization (`src/preprocess.py`)
- **Lowercasing & Normalization:** Standardizes letter casing and normalizes curly quotes (`’` $\to$ `'`).
- **Contraction Expansion:** Replaces colloquial English contractions using an exhaustive mapping dictionary (e.g., `don't` $\to$ `do not`, `can't` $\to$ `cannot`, `what's` $\to$ `what is`).
- **Regex Sanitization:** Strips external URLs, email addresses, and non-alphanumeric noise using `re.sub(r"[^a-z0-9\s]", " ", text)`.
- **Negation-Preserving Stop-Word Removal:** Crucial for intent classification: while standard stop-word lists discard words like `not` and `no`, our pipeline explicitly preserves negation markers (`not`, `no`, `nor`, `never`, `neither`, `none`) so commands like *"do not cancel my alarm"* retain their opposite semantic meaning.
- **WordNet Lemmatization:** Morphologically reduces inflected word variations (e.g., `alarms` $\to$ `alarm`, `setting` $\to$ `set`) without aggressive or destructive character truncation.

### 3. Stratified 80/10/10 Split
- Custom stratified 3-way partition distributing 80% to training, 10% to validation, and 10% to testing.
- Guarantees **all 60 intent classes** are represented across all three partitions, even rare tail classes.
  - **Train split:** 13,214 samples (79.98%)
  - **Validation split:** 1,652 samples (10.00%)
  - **Test split:** 1,655 samples (10.02%)

---

## 📈 Automated EDA & Preprocessing Metrics

The automated pipeline outputs detailed quantitative diagnostics to `outputs/preprocessing_report.json`:

| Metric | Raw Utterances | Cleaned Utterances | Change / Effect |
|---|---|---|---|
| **Vocabulary Size** | 6,328 words | 5,420 words | **-14.35% reduction** (noise reduced) |
| **Mean Utterance Length** | 6.90 words | 3.88 words | Concentrated semantic tokens |
| **Word Count Std. Dev.** | 3.29 words | 1.83 words | More uniform sequence length |
| **Min / Max Word Count** | 1 / 61 words | 0 / 31 words | Eliminated outlier spam lengths |
| **Mean Character Length**| 34.92 chars | 23.34 chars | Cleaned morphological footprint |

### Sample Before & After

```text
[alarm_set]
  Raw:   "wake me up at five am this week"
  Clean: "wake five week"

[alarm_set]
  Raw:   "set an alarm for two hours from now"
  Clean: "set alarm two hour"

[audio_volume_mute]
  Raw:   "olly quiet"
  Clean: "olly quiet"
```

---

## 🚀 Reproduction & Setup Guide

### 1. Clone & Initialize Environment
```bash
# Clone the repository
git clone <repo-url>
cd Project

# Create virtual environment (Python 3.12 or 3.14)
python -m venv .venv

# Activate virtual environment
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# Linux / macOS:
source .venv/bin/activate

# Install required dependencies
pip install -r requirements.txt
```

### 2. Download Dataset
```bash
python src/download_data.py
```
*Output: Saves raw English data to `Data/massive_english.csv` (16,521 rows).*

### 3. Run Preprocessing, EDA, and Splits
```bash
python src/preprocess.py
```
*Outputs:*
- `Data/train_cleaned.csv`
- `Data/val_cleaned.csv`
- `Data/test_cleaned.csv`
- `outputs/preprocessing_report.json`

---

## 🔮 Scope Notice & Phase 2 Outlook

Per course project guidelines, **Phase 1** focuses strictly on Problem Formulation, Related Literature Review, Data Ingestion, and Preprocessing. **No model training code is included in this phase.**

**Upcoming in Phase 2 (Week 9):**
- Vectorization & Text Representations: Bag-of-Words (BoW), TF-IDF n-grams, and Dense Contextual Embeddings (Sentence-Transformers).
- Baseline Modeling: Logistic Regression & Linear Support Vector Classifiers (LinearSVC).
- Deep Learning / Transformer Benchmarks: Fine-tuning DistilBERT / RoBERTa models.
- Macro F1 and Accuracy multi-class evaluation.
