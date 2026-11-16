# Intent Classification for Voice Assistants and Conversational Agents (Amazon MASSIVE)

**Course:** Natural Language Processing (A0443404) · Middle East University (MEU), Faculty of Information Technology
**Instructors:** Dr. Israa Al Badarneh · Dr. Zaid Alhalhouli · **Semester:** First 2026/2027
**This repository:** Phase 1 (Proposal & NLP Data Preparation). No model training for the final system is included yet.

---

## Project overview

Voice assistants and chatbots must map a short, informal request to one action (an *intent*), for example
`"wake me up at five am this week"` → `alarm_set`. This project builds the full NLP pipeline for that task on the
**Amazon MASSIVE (en-US)** dataset: 60 intents grouped in 18 scenarios.

Phase 1 covers the problem definition, dataset audit, preprocessing and a check that the preprocessing keeps
the information the intent depends on. Phases 2–3 compare **TF-IDF + LinearSVC** with a fine-tuned **DistilBERT**
using macro-F1 and accuracy.

> MASSIVE contains voice-assistant commands (alarm, calendar, music, smart home, takeaway...). The method is also used
> to route customer-support requests, but the data itself is not customer-support text.

## Team

| Name | Role | Responsibilities |
|---|---|---|
| Karam Abdallah Abdelkarim Abu-Juma'a | Group leader | Architecture, `download_data.py`, stratified split, EDA report, documentation, submission |
| Omar Al-Zubaidi | Member | Contraction dictionary, regex normalisation, abbreviation collapsing, stop-word policy |
| Osama Shqair | Member | POS-aware lemmatization, preprocessing ablation, dataset validation and quality control, related work and references |

## Repository structure

```
Project/
├── Data/                                   # git-ignored CSVs (included in the ZIP)
│   ├── massive_english.csv                 # raw en-US data (16,521 rows)
│   ├── train_cleaned.csv                   # stratified train split (13,149 rows)
│   ├── val_cleaned.csv                     # stratified validation split (1,642 rows)
│   └── test_cleaned.csv                    # stratified test split (1,648 rows)
├── docs/
│   ├── NLP_Course_Project_PBL_Description.pdf
│   ├── NLP_Project_Proposal_Form_Phase1.docx          # blank official form
│   ├── NLP_Project_Proposal_Form_Phase1_Filled.docx   # filled form (submission)
│   └── Project_Proposal_Phase1.md                     # same proposal in Markdown
├── outputs/
│   ├── preprocessing_report.json           # EDA + data-quality audit + quality-control numbers
│   ├── before_after_examples.md / .csv     # real raw vs. cleaned utterances
│   └── preprocessing_ablation.md / .json   # comparison of 5 cleaning variants
├── src/
│   ├── download_data.py                    # downloads MASSIVE, keeps en-US
│   ├── preprocess.py                       # audit, cleaning, de-duplication, 80/10/10 split
│   └── benchmark_preprocessing.py          # preprocessing ablation (quality control)
├── NLP Project Phase 1 Proposal.pdf        # PDF of the filled form
├── requirements.txt
└── README.md
```

## Dataset

| Item | Value |
|---|---|
| Source | Amazon MASSIVE v1.0, en-US file (https://github.com/alexa/massive) |
| Rows | 16,521 raw → **16,439** after removing 82 exact duplicate (utterance, intent) rows |
| Official partitions | train 11,514 / dev 2,033 / test 2,974 (kept in the `partition` column) |
| Our split | stratified 80/10/10 → train 13,149 / val 1,642 / test 1,648, all 60 classes in each split |
| Classes | 60 intents, 18 scenarios; `calendar_set` 1,147 samples vs. `cooking_query` 6 (about 190:1) |
| License | CC BY 4.0 |

### Data-quality findings on the raw text (`outputs/preprocessing_report.json`)

| Finding | Count |
|---|---:|
| Utterances with apostrophes (mostly contractions) | 1,324 |
| Utterances with dotted abbreviations (`p. m.`, `b. b. c.`) | 698 |
| Utterances containing the wake word `olly` | 534 |
| Utterances with only 1–2 words | 830 |
| Exact duplicate utterances | 89 |
| Texts labelled with two different intents (label noise) | 7 |
| Uppercase / digits / URLs / e-mails / non-ASCII | 0 |

## Preprocessing pipeline (`src/preprocess.py`)

1. Lowercase and expand contractions with a mapping dictionary (`what's` → `what is`).
2. Collapse dotted abbreviations (`p. m.` → `pm`, `g. m. t.` → `gmt`). Without this the tokenizer splits the letters and the stop-word filter deletes them.
3. Strip URLs/e-mails and punctuation with regular expressions.
4. Tokenize with NLTK `word_tokenize`; remove the wake word `olly`.
5. Stop-word removal with a **customised** list. The NLTK list minus: negations (`not`, `no`, `never`...), direction/time words
   (`on`, `off`, `up`, `down`, `in`, `out`, `before`, `after`, `from`, `now`...) and question/auxiliary words (`what`, `where`, `do`, `is`...).
   `am` is kept only after a number (`five am`).
6. **POS-aware** WordNet lemmatization (`setting` → `set`, `playing` → `play`). `WordNetLemmatizer` treats every word as a noun unless it receives the POS tag.
7. An utterance is never left empty; if every token is a stop-word, all tokens are kept.

The pipeline writes two columns: `clean_text` (stop-words removed) and `norm_text` (same steps, **all words kept**).

### Why the stop-word list is customised

With the default NLTK list, `"turn the lights off"` and `"turn the lights on"` both became `"turn light"`, and `"volume up"` / `"volume down"` both became `"volume"`.

| | Raw text | `clean_text` | `norm_text` |
|---|---:|---:|---:|
| Vocabulary size | 6,328 | 5,257 (−16.9%) | 5,314 (−16.0%) |
| Mean words per utterance | 6.91 | 4.95 | 6.90 |
| Texts attached to more than one intent | 7 | 40 | 12 |
| Empty utterances | 0 | 0 | 0 |

### Real before / after examples (`outputs/before_after_examples.md`)

| Raw | `clean_text` | `norm_text` |
|---|---|---|
| turn the lights off please | turn light off please | turn the light off please |
| wake me up at five am this week | wake up five am week | wake me up at five am this week |
| olly pause for ten seconds | pause ten second | pause for ten second |
| tell me the time in g. m. t. plus five | tell time in gmt plus five | tell me the time in gmt plus five |
| start playing music from favorites | start play music from favorite | start play music from favorite |

### Preprocessing ablation (`python src/benchmark_preprocessing.py`)

Same train/validation split, probe classifier = TF-IDF (1–2 grams) + LinearSVC. **This is not the Phase 2 model**; it only shows how much intent information each cleaning variant keeps.

| Variant | Vocab | Empty rows | Conflicting texts | Val accuracy | Val macro-F1 |
|---|---:|---:|---:|---:|---:|
| lowercase only | 6,110 | 0 | 8 | 0.8624 | 0.8473 |
| first pipeline (default NLTK list, noun-only lemmas) | 5,420 | 24 | 134 | 0.8270 | 0.7805 |
| + intent-bearing function words kept, POS lemmas | 5,244 | 0 | 84 | 0.8398 | 0.8299 |
| `clean_text` (final, adds question/auxiliary words) | 5,257 | 0 | 40 | 0.8453 | 0.8325 |
| `norm_text` (no stop-word removal) | 5,314 | 0 | 12 | 0.8605 | 0.8339 |

Reading the table: the customisation removes most of the damage done by the default list (macro-F1 0.781 → 0.833, conflicts 134 → 40), but removing stop-words does not beat keeping them for this probe. Phase 2 therefore tests `clean_text` and `norm_text` (and raw text for DistilBERT).

## Reproduce

```bash
python -m venv .venv
# Windows: .\.venv\Scripts\Activate.ps1      Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt

python src/download_data.py          # -> Data/massive_english.csv
python src/preprocess.py             # -> Data/*_cleaned.csv, outputs/preprocessing_report.json, outputs/before_after_examples.*
python src/benchmark_preprocessing.py  # -> outputs/preprocessing_ablation.*
```

NLTK resources (`punkt_tab`, `stopwords`, `wordnet`, POS tagger) are downloaded automatically on first run (internet needed).
All paths are resolved relative to the project folder, so the project runs from any drive or folder.

## Notes and limitations

- The stratified split ignores the official MASSIVE `partition` column (kept in the CSVs). If Phase 2 results must be compared with the MASSIVE paper, switch to the official partitions.
- 7 texts carry two different intents (for example `good night`). They are kept and reported, not hidden.
- References [3] and [4] in the proposal are arXiv preprints; check their final published venue before the final report.

## Phase 2 outlook (Week 9)

Bag-of-Words and TF-IDF n-grams, sentence embeddings, logistic regression and LinearSVC baselines, DistilBERT fine-tuning, macro-F1 / accuracy / inference time, confusion-matrix error analysis.
