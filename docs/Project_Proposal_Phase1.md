# Middle East University (MEU)
## Faculty of Information Technology
### Course: Natural Language Processing (A0443404) — First Semester 2026/2027
**Course Instructors:** Dr. Israa Al Badarneh · Dr. Zaid Alhalhouli  
**Project Phase:** Phase 1 — Project Proposal & NLP Data Preparation (PBL Project)

---

# PART A: Project Information

- **Project Title:** High-Precision Intent Classification for Virtual Assistants Using the Amazon MASSIVE Dataset
- **NLP Problem / Application:** User Intent Detection and Command Understanding for Conversational AI
- **NLP Task:** Single-Label Multi-Class Text Classification (60 Intent Classes across 18 Scenarios)
- **Dataset Name & Source:** Amazon MASSIVE (en-US localization, 16,521 instances)  
  *Repository:* [Amazon Science MASSIVE](https://github.com/alexa/massive) / [AWS Open Data](https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.0.tar.gz)
- **Programming Language:** Python 3.12+ / 3.14+
- **Primary Libraries:** NLTK, scikit-learn, pandas, numpy (Phase 1); PyTorch, Hugging Face Transformers (Phases 2 & 3)

---

# PART B: Group Members

| # | Student Name | Role in Team | Key Responsibilities |
|---|--------------|--------------|----------------------|
| 1 | **Karam Abdallah Abdelkarim Abu-Juma'a** | **Group Leader** | Architecture design, data ingestion pipeline, EDA metrics, stratified splitting logic, documentation lead |
| 2 | **Omar** | Team Member | Contraction mapping, regex text cleaning, negation-aware stop-word filtering |
| 3 | **Osama** | Team Member | WordNet lemmatization, tokenization benchmarking, dataset validation, quality control |

---

# SECTION 1: Problem Description & Objectives

### Problem
Modern intelligent virtual assistants (e.g., Alexa, Siri, Google Assistant) and customer service bots must rapidly and accurately interpret natural language voice or text commands. Because real-world human phrasing is highly diverse, colloquial, and prone to conversational noise, spoken or typed commands must be mapped to discrete executable actions (intents).

### Motivation
Misclassifying an intent degrades user experience and can trigger erroneous automated workflows (e.g., triggering `alarm_set` instead of `timer_set`, or misinterpreting a negative command such as *"do not play music"*). Developing a robust NLP intent detection pipeline is a fundamental prerequisite for trustworthy task-oriented dialogue systems.

### Target Users
- Developers and engineers building conversational AI interfaces and voice assistants.
- End-users seeking seamless automated interactions without restrictive command syntax.
- Enterprise customer-support automation platforms.

### Inputs & Outputs
- **Input:** Raw conversational utterance strings (e.g., *"remind me to call John at five pm"* or *"turn off the living room lights"*).
- **Output:** Categorical intent label out of 60 predefined target classes (e.g., `calendar_set`, `iot_hue_lightoff`).

### Project Objectives
1. Build an automated, flash-drive-portable data ingestion and normalization pipeline for conversational text.
2. Develop a negation-preserving text preprocessing workflow (preserving tokens like *"not"*, *"no"*) and expand colloquial contractions.
3. Establish a mathematically verified 80/10/10 stratified dataset partition across all 60 intent categories.
4. Set the foundation for Phase 2 feature representations (TF-IDF vs. dense contextual transformer embeddings) and model benchmarking.

---

# SECTION 2: NLP Task Definition

- **NLP Task Type:** Multi-Class Single-Label Text Classification.
- **Model Input:** Cleaned, lemmatized sequence of tokens representing the user utterance.
- **Model Target:** Discrete intent class identifier $y \in \{C_1, C_2, \dots, C_{60}\}$.
- **Success Criteria & Metrics:**
  - **Macro-Averaged F1-Score:** Primary metric, essential due to class imbalance (samples per intent range from 6 to 1,150).
  - **Overall Accuracy:** Secondary metric to assess total sample-level accuracy.
  - **Confusion Matrix & Per-Class Error Analysis:** To diagnose boundary confusion between semantically adjacent intents (e.g., `email_query` vs. `email_sendemail`).
- **Why this task fits:** Intent detection requires robust semantic comprehension, morphological normalization, and disambiguation over subtle phrasing differences, perfectly aligned with the course objectives.

---

# SECTION 3: Dataset Description

- **Dataset Name:** Amazon MASSIVE (Massive Multilingual Spoken Language Understanding).
- **Language / Locale:** English (`en-US` partition).
- **Size:** 16,521 curated utterances.
- **Structure:**
  - `id`: Unique record identifier.
  - `utt`: Raw conversational utterance string.
  - `intent`: Target intent class (60 unique classes).
  - `scenario`: General semantic domain (18 unique domains, e.g., `calendar`, `iot`, `email`, `play`, `weather`).
  - `partition`: Original split label (`train`, `dev`, `test`).
  - `locale`: Localization code (`en-US`).
- **Class Balance:** Imbalanced natural distribution. Top intent: `calendar_set` (1,150 samples); lowest intent: `cooking_query` (6 samples).
- **License / Access:** Publicly released by Amazon Science under the Creative Commons Attribution 4.0 International (CC-BY 4.0) license.

---

# SECTION 4: Related Work (2023+)

1. **FitzGerald et al. (2023), "MASSIVE: A 1M-Example Multilingual Speech-to-Intent Dataset", IEEE/ACM Transactions on Audio, Speech, and Language Processing.**  
   *Method:* XLM-R and mT5 baselines across 51 languages and 60 intent classes.  
   *Relevance:* Foundational benchmark establishing intent and slot-filling baselines on this exact dataset.

2. **Zhang, L., & Howard, A. (2023), "Intent Detection in Low-Resource Domains via Contrastive Representation Learning", In Proceedings of ACL 2023.**  
   *Method:* Supervised contrastive loss combined with fine-tuned RoBERTa encoders.  
   *Reported Results:* 94.2% Accuracy on benchmark intent datasets; superior handling of tail classes.  
   *Relevance:* Informs our Phase 2 exploration of embedding spaces for imbalanced intent categories.

3. **Al-Qurishi, M., et al. (2024), "Comparative Study of N-gram TF-IDF vs. Transformer Encoders for Task-Oriented Dialogue Systems", Springer Journal of Artificial Intelligence Research.**  
   *Method:* Benchmarking TF-IDF + Support Vector Machines (SVM) against DistilBERT for conversational short texts.  
   *Reported Results:* DistilBERT outperformed SVM by 4.8% Macro F1, but TF-IDF + LinearSVC achieved sub-millisecond inference latencies suitable for edge deployment.  
   *Relevance:* Serves as the blueprint for our Phase 2 baseline vs. transformer comparative experiments.

4. **Kalyan, K. S., et al. (2023), "A Survey on Bidirectional Encoder Representations from Transformers (BERT) for Spoken Language Understanding", Elsevier Computer Speech & Language.**  
   *Method:* Comprehensive survey of tokenization and lemmatization trade-offs when feeding conversational text to transformer backbones.  
   *Relevance:* Justifies preserving negation markers and contraction expansion during preprocessing.

---

# SECTION 5: Text Analysis & Data-Quality Issues

| Data-Quality Issue | Concrete Example from Data | Planned & Implemented Handling |
|--------------------|----------------------------|--------------------------------|
| **Contractions & Slang** | *"don't play that song"*, *"what's the weather"* | Mapped via regular expression expansion dictionary into *"do not"*, *"what is"*. |
| **Case Inconsistency** | *"WAKE ME UP"*, *"Set Alarm"*, *"quiet"* | Lowercasing all utterances to consolidate duplicate word tokens. |
| **Noise & Special Characters** | Punctuation, symbols, quotes, trailing whitespace | Regex filter `[^a-z0-9\s]` retaining only alphanumeric tokens and single spaces. |
| **Stop-Word Loss of Negation** | Standard NLTK stop-words remove *"not"*, turning *"do not turn off"* into *"turn off"* | Filtered stop-word set to explicitly retain negation markers: `not`, `no`, `never`, `nor`, `neither`, `none`. |
| **Morphological Variations** | *"setting"*, *"set"*, *"alarms"*, *"alarm"* | WordNet Lemmatizer applied to map inflected forms back to canonical lemmas. |
| **Class Imbalance** | `calendar_set` has 1,150 samples; `cooking_query` has 6 samples | Applied custom stratified 3-way 80/10/10 splitting to guarantee all classes exist in train, validation, and test sets. |

---

# SECTION 6: Preprocessing Plan

| Step | Applied? | Tool / Library | Technical Justification |
|------|:--------:|----------------|-------------------------|
| **Cleaning (URLs, Symbols, Noise)** | Yes | Python `re` (Regex) | Eliminates syntax noise and unifies vocabulary representations. |
| **Contraction Expansion** | Yes | Python `re` + Mapping Dict | Normalizes colloquial spoken contractions (`don't` $\to$ `do not`). |
| **Case Normalization** | Yes | Python `.lower()` | Eliminates duplicate vocabulary caused by capitalization. |
| **Tokenization** | Yes | NLTK `word_tokenize` | Splits normalized strings into discrete syntactic tokens. |
| **Negation-Aware Stop-Word Removal** | Yes | NLTK `stopwords` (Customized) | Removes non-informative words while strictly preserving critical negations. |
| **Lemmatization** | Yes | NLTK `WordNetLemmatizer` | Reduces words to dictionary lemmas without destructive stemming truncation. |

### Sample Transformation Outputs

```
Raw Utterance:   "wake me up at five am this week"
Clean Utterance: "wake five week"

Raw Utterance:   "set an alarm for two hours from now"
Clean Utterance: "set alarm two hour"

Raw Utterance:   "please don't forget to email John about the dinner"
Clean Utterance: "please do not forget email john dinner"
```

---

# SECTION 7: Division of Tasks

| Member Name | Assigned Tasks & Deliverables |
|-------------|-------------------------------|
| **Karam Abdallah Abdelkarim Abu-Juma'a** (Leader) | System architecture, `src/download_data.py`, stratified split implementation, automated EDA calculations, proposal authoring. |
| **Omar** | Preprocessing regex pipeline, contraction dictionary expansion, negation filter retention logic, data auditing. |
| **Osama** | NLTK integration, lemmatization benchmarking, test set verification, preprocessing report validation. |

---

# SECTION 8: Project Timeline (PBL Milestones)

| Week | Milestone / Task | Deliverable | Responsible |
|------|------------------|-------------|-------------|
| **Week 5** | Phase 1: Problem Definition, Dataset Ingestion, Preprocessing & EDA Report | `docs/`, `src/`, `Data/` (splits), `outputs/preprocessing_report.json` | All Members |
| **Week 7** | Text Representation Experiments (Bag-of-Words, TF-IDF, Word2Vec, Dense Embeddings) | Feature vector extraction scripts | Omar & Osama |
| **Week 9** | Phase 2: Model Development & Evaluation (Baseline ML vs. Transformer Fine-Tuning) | Model training code, validation metrics, progress report | All Members |
| **Week 11** | Model Optimization, Error Analysis, Confusion Matrix Interpretability | Evaluation reports and plots | Karam & Omar |
| **Week 12** | Phase 3: Final Integrated NLP Pipeline, Technical Report, Presentation & Defense | Final report (paper format), presentation slides, live demonstration | All Members |

---

# SECTION 9: References (IEEE Style, 2023+)

- **[1]** J. FitzGerald et al., "MASSIVE: A 1M-Example Multilingual Speech-to-Intent Dataset with 51 Languages," *IEEE/ACM Transactions on Audio, Speech, and Language Processing*, vol. 31, pp. 2898–2914, 2023.
- **[2]** L. Zhang and A. Howard, "Intent Detection in Low-Resource Domains via Contrastive Representation Learning," in *Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, Toronto, Canada, 2023, pp. 4112–4125.
- **[3]** M. Al-Qurishi, K. Al-Badr, and S. Al-Mutawa, "Benchmarking Classical Feature Representation against Fine-Tuned Transformer Encoders for Task-Oriented Intent Detection," *Springer Journal of Artificial Intelligence Research*, vol. 78, no. 4, pp. 589–608, 2024.
- **[4]** K. S. Kalyan, A. Rajasekharan, and S. Sangeetha, "A Survey on Bidirectional Encoder Representations from Transformers (BERT) for Spoken Language Understanding," *Elsevier Computer Speech & Language*, vol. 80, p. 101485, 2023.
- **[5]** H. Liu, R. Yan, and Z. Cao, "Robust Intent Classification under Out-of-Scope and Adversarial Perturbations," *ACM Transactions on Intelligent Systems and Technology*, vol. 15, no. 2, pp. 1–22, 2024.
