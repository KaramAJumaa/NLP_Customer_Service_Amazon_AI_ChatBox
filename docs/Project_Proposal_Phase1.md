# Middle East University (MEU) — Faculty of Information Technology
## Natural Language Processing (A0443404) · First Semester 2026/2027
**Instructors:** Dr. Israa Al Badarneh · Dr. Zaid Alhalhouli
**Phase 1:** Project Proposal & NLP Data Preparation (Week 5, due 16/11/2026)

> The submitted version is the filled official form (`NLP_Project_Proposal_Form_Phase1_Filled.docx`, PDF copy in the project root). This file is the same content in Markdown.

---

## Part A: Group information

| Item | Value |
|---|---|
| Project title | Intent Classification for Voice Assistants and Conversational Agents on the Amazon MASSIVE Dataset (TF-IDF + LinearSVC vs. DistilBERT) |
| NLP problem / application | User intent detection for task-oriented voice assistants and chatbots; also usable for routing customer-support requests |
| NLP task | Single-label multi-class text classification (60 intents, 18 scenarios) |
| Dataset | Amazon MASSIVE v1.0, en-US subset (https://github.com/alexa/massive) |
| Language / libraries | Python 3.12; Phase 1: NLTK, pandas, scikit-learn; Phases 2–3: scikit-learn, Hugging Face Transformers, PyTorch |

## Part B: Group members

| # | Student | Role |
|---|---|---|
| 1 | Karam Abdallah Abdelkarim Abu-Juma'a (leader) | Data ingestion, stratified split, EDA report, documentation |
| 2 | Omar Al-Zubaidi | Contraction map, regex normalisation, stop-word policy |
| 3 | Osama Shqair | Lemmatization, preprocessing ablation, dataset validation and quality control |

---

## Section 1: Problem description & objectives

- **Problem:** voice assistants and chatbots must map a short, informal request to one action (an intent). Wording varies a lot, so keyword matching fails.
- **Motivation:** a wrong intent triggers the wrong action (setting an alarm instead of querying alarms, lights on instead of off). Intent detection is the first step of every task-oriented dialogue system.
- **Users:** developers of voice assistants and chatbots, customer-support teams, end users.
- **Input:** one short English utterance (mean 6.9 words), e.g. *"wake me up at five am this week"*.
- **Output:** one of 60 intent labels, e.g. `alarm_set`, `iot_hue_lightoff`.
- **Objectives:** (1) define the problem and prepare MASSIVE en-US (audit, de-duplicate, clean, split 80/10/10); (2) build a justified preprocessing pipeline that keeps intent-bearing words; (3) verify it with an ablation; (4) in Phases 2–3 compare TF-IDF + LinearSVC with fine-tuned DistilBERT.

## Section 2: NLP task definition

- **Task type:** text classification (multi-class, single label).
- **Model input:** the utterance after preprocessing (`clean_text` or `norm_text`); transformers read the raw text.
- **Target:** one intent class among 60.
- **Success criterion:** macro-F1 (primary, classes are very imbalanced), accuracy, per-class errors and confusion matrix on the held-out test split; inference time for the TF-IDF vs. DistilBERT comparison.
- **Why this task fits:** it needs a text representation, a classifier and task-appropriate metrics, the comparison the course project asks for.

## Section 3: Dataset description

| Item | Value |
|---|---|
| Source | Amazon MASSIVE v1.0 (FitzGerald et al., ACL 2023), en-US file |
| Size | 16,521 utterances (official: train 11,514 / dev 2,033 / test 2,974); 16,439 after removing 82 exact duplicates; our split train 13,149 / val 1,642 / test 1,648 |
| Fields | `id`, `utt`, `intent`, `scenario`, `partition`, `locale` (+ `clean_text`, `norm_text`) |
| Classes | 60 intents, 18 scenarios; `calendar_set` 1,147 vs. `cooking_query` 6 (about 190:1) |
| Language | English (en-US) |
| Characteristics | 1–61 words (mean 6.9), lowercase ASCII, no digits/URLs/emojis; 1,324 utterances with apostrophes, 698 dotted abbreviations, 534 with the wake word "olly", 830 with 1–2 words, 7 texts with two intents |
| License | CC BY 4.0 |

## Section 4: Related work (2023+)

| # | Reference | Method | Dataset | Reported results | Relevance |
|---|---|---|---|---|---|
| 1 | FitzGerald et al., 2023, ACL 2023 | MASSIVE release; XLM-R and mT5 baselines | MASSIVE (1M utterances, 51 languages, 60 intents) | Exact match, intent accuracy and slot F1 per language | Our dataset and its official task |
| 2 | Zhang et al., 2023, Findings of ACL 2023 | Direct fine-tuning of PLMs (DFT, DFT++) for few-shot intent detection | Real-world few-shot intent benchmarks | With 2+ labeled samples per class, direct fine-tuning beats many continual-pre-training baselines | Supports fine-tuning DistilBERT directly |
| 3 | Arora, Jain, Merugu, 2024, arXiv:2410.01627 | SetFit vs. 7 LLMs (in-context learning); hybrid routing | HINT3 + 3 internal e-commerce datasets | Avg F1: SetFit 0.600 (0.658 with negative augmentation) vs. best LLM 0.736; SetFit ~56x faster | Accuracy vs. latency trade-off |
| 4 | Lichouri, Lounnas, Amziane, 2024, arXiv:2407.13565 | TF-IDF + LinearSVC, LSTM, transformers | ArBanking77 | Best micro-F1 93.02% dev, 67.21% test | Same TF-IDF + LinearSVC baseline; dev–test gap |

## Section 5: Text analysis & data-quality issues

| Issue | Example | Handling |
|---|---|---|
| Contractions and apostrophes | "what's the time in australia" (1,324) | Contraction dictionary |
| Dotted abbreviations / time markers | "four p. m.", "b. b. c." (698) | Collapse to `pm`, `bbc` |
| Wake-word noise | "olly pause for ten seconds" (534) | Remove "olly" |
| Duplicates and label noise | 89 duplicates; "good night" has two intents (7 texts) | Drop 82 exact duplicates; report conflicting texts |
| Class imbalance | 1,147 vs. 6 samples | Stratified split, macro-F1, class weights in Phase 2 |
| Very short utterances | 830 with 1–2 words | Keep; never delete intent-bearing words |
| Intent words listed as stop-words | "turn the lights off/on" both became "turn light" | Customised stop-word list (conflicts 134 → 40) |

## Section 6: Preprocessing plan

| Step | Use? | Tool | Why |
|---|:---:|---|---|
| Cleaning (URLs, special characters) | Yes | `re` | Removes symbol noise (0 URLs in this data, kept for safety) |
| Tokenization | Yes | NLTK `word_tokenize` | Tokens for filtering and lemmatization |
| Normalization | Yes | `str.lower`, `re`, contraction map | Lowercase, contractions, abbreviations, wake word |
| Stop-word handling | Yes | NLTK (customised) | Keeps negations, on/off/up/down/in/out/from/now, question words |
| Lemmatization | Yes | WordNetLemmatizer + `pos_tag` | Merges word forms without broken stems |
| Punctuation & numbers | Yes | `re` | Numbers are spoken words; 0 digit utterances |
| Emoji handling | No | – | 0 non-ASCII characters |
| Encoding / language filtering | Yes | pandas, UTF-8 | Only en-US extracted; plain ASCII |

Real samples (`clean_text`):

| Raw | After preprocessing |
|---|---|
| turn the lights off please | turn light off please |
| wake me up at five am this week | wake up five am week |
| olly pause for ten seconds | pause ten second |
| tell me the time in g. m. t. plus five | tell time in gmt plus five |
| start playing music from favorites | start play music from favorite |

Ablation (validation split, TF-IDF 1–2 grams + LinearSVC probe, **not** the Phase 2 model), accuracy / macro-F1: lowercase only 0.862 / 0.847; first pipeline 0.827 / 0.781; final `clean_text` 0.845 / 0.833; `norm_text` 0.861 / 0.834. Stop-word removal shrinks the vocabulary by 16.9% but does not improve this probe, so Phase 2 tests both columns.

## Section 7: Division of tasks

| Member | Responsibilities |
|---|---|
| Karam Abdallah Abdelkarim Abu-Juma'a | Architecture, `download_data.py`, stratified split, EDA report, proposal and submission |
| Omar Al-Zubaidi | Contraction dictionary, regex normalisation, abbreviation collapsing, stop-word policy |
| Osama Shqair | POS-aware lemmatization, ablation (`benchmark_preprocessing.py`), dataset validation and quality control, related work and references |

## Section 8: Timeline

| Week | Milestone | Responsible | Deliverable |
|---|---|---|---|
| 5 | Phase 1: proposal, audit, preprocessing, ablation | All | Form, `src/`, `outputs/` |
| 7 | Text representation experiments (BoW, TF-IDF, embeddings) | Omar, Osama | Feature scripts, first baselines |
| 9 | Phase 2: baselines and DistilBERT fine-tuning | All | Progress report, model code |
| 11 | Error analysis and tuning | Karam, Omar | Evaluation tables and plots |
| 12 | Phase 3: integrated pipeline, demo, report, slides, defense | All | Final report, slides, demo |

## Section 9: References

1. J. FitzGerald et al., "MASSIVE: A 1M-Example Multilingual Natural Language Understanding Dataset with 51 Typologically-Diverse Languages," in *Proc. 61st Annu. Meeting Assoc. Comput. Linguistics (Vol. 1: Long Papers)*, Toronto, Canada, Jul. 2023, pp. 4277–4302, doi: 10.18653/v1/2023.acl-long.235.
2. H. Zhang, H. Liang, L. Zhan, A. Y. S. Lam, and X.-M. Wu, "Revisit few-shot intent classification with PLMs: Direct fine-tuning vs. continual pre-training," in *Findings Assoc. Comput. Linguistics: ACL 2023*, Toronto, Canada, Jul. 2023, pp. 11105–11121, doi: 10.18653/v1/2023.findings-acl.706.
3. G. Arora, S. Jain, and S. Merugu, "Intent detection in the age of LLMs," arXiv:2410.01627 [cs.CL], Oct. 2024.
4. M. Lichouri, K. Lounnas, and M. Z. Amziane, "dzFinNlp at AraFinNLP: Improving intent detection in financial conversational agents," arXiv:2407.13565 [cs.CL], Jul. 2024.
