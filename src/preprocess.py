"""
Text Preprocessing & Exploratory Data Analysis (EDA) Pipeline
Project: Intent Classification
Course: Natural Language Processing (A0443404) - Middle East University (MEU)
Dataset: Amazon MASSIVE (en-US Localization)

Design notes (v2)
-----------------
* Stop-word removal keeps negations AND the function words that change the
  intent of a command (on / off / up / down / in / out / before / after ...).
  With the default NLTK list, "turn on the lights" and "turn off the lights"
  both became "turn light".
* Spoken time markers ("p. m.", "a. m.") are normalised to "pm" / "am" and "am"
  is kept only after a number ("five am"), so "i am" is still removed.
* Lemmatization is POS-aware (WordNetLemmatizer defaults to nouns, so
  "setting" stayed "setting").
* The wake word "olly" (534 utterances) is removed because it carries no intent.
* An utterance is never left empty: if every token is a stop-word we fall back
  to the full token list.
"""

import os
import re
import json
from collections import Counter
import pandas as pd
import numpy as np

import nltk
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Flash-drive portable relative path resolution
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "Data")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

RAW_DATA_PATH = os.path.join(DATA_DIR, "massive_english.csv")
TRAIN_OUTPUT_PATH = os.path.join(DATA_DIR, "train_cleaned.csv")
VAL_OUTPUT_PATH = os.path.join(DATA_DIR, "val_cleaned.csv")
TEST_OUTPUT_PATH = os.path.join(DATA_DIR, "test_cleaned.csv")
REPORT_OUTPUT_PATH = os.path.join(OUTPUTS_DIR, "preprocessing_report.json")
EXAMPLES_MD_PATH = os.path.join(OUTPUTS_DIR, "before_after_examples.md")
EXAMPLES_CSV_PATH = os.path.join(OUTPUTS_DIR, "before_after_examples.csv")

# Comprehensive English contraction map
CONTRACTION_MAP = {
    "ain't": "is not",
    "aren't": "are not",
    "can't": "cannot",
    "can't've": "cannot have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'll": "he will",
    "he's": "he is",
    "how'd": "how did",
    "how'll": "how will",
    "how's": "how is",
    "i'd": "i would",
    "i'll": "i will",
    "i'm": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it'd": "it would",
    "it'll": "it will",
    "it's": "it is",
    "let's": "let us",
    "might've": "might have",
    "mustn't": "must not",
    "shan't": "shall not",
    "she'd": "she would",
    "she'll": "she will",
    "she's": "she is",
    "shouldn't": "should not",
    "that's": "that is",
    "there's": "there is",
    "they'd": "they would",
    "they'll": "they will",
    "they're": "they are",
    "they've": "they have",
    "wasn't": "was not",
    "we'd": "we would",
    "we'll": "we will",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "where'd": "where did",
    "where's": "where is",
    "who'll": "who will",
    "who's": "who is",
    "won't": "will not",
    "wouldn't": "would not",
    "you'd": "you would",
    "you'll": "you will",
    "you're": "you are",
    "you've": "you have"
}

CONTRACTION_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in CONTRACTION_MAP.keys()) + r")\b",
    flags=re.IGNORECASE
)


def expand_contractions(text: str) -> str:
    """Normalize apostrophes and expand common English contractions."""
    text = text.replace("’", "'").replace("`", "'")
    def replace(match):
        contracted = match.group(0).lower()
        expanded = CONTRACTION_MAP.get(contracted, contracted)
        if match.group(0)[0].isupper():
            expanded = expanded.capitalize()
        return expanded
    return CONTRACTION_PATTERN.sub(replace, text)



# ---------------------------------------------------------------------------
# Stop-word policy
# ---------------------------------------------------------------------------
# In intent classification, negations carry crucial semantics
# (e.g. 'do not cancel order' vs 'cancel order').
NEGATION_TOKENS = {"not", "no", "nor", "never", "neither", "none"}

# NLTK lists these as stop-words, but removing them destroys the contrast
# between intents such as iot_hue_lighton / iot_hue_lightoff or
# audio_volume_up / audio_volume_down, or the time reference of a command.
SEMANTIC_FUNCTION_WORDS = {
    "on", "off", "up", "down", "in", "out", "over", "under", "above", "below",
    "before", "after", "from", "until", "between", "again", "more", "most",
    "few", "all", "now",
}

# The voice-assistant wake word that appears in hundreds of utterances.
# It says nothing about the intent, so it is removed (unless it is the only token).
WAKE_WORDS = {"olly"}

# Question words and auxiliaries separate a QUERY from a COMMAND:
# "do i have any alarms" (alarm_query) vs "alarm for ten am" (alarm_set),
# "did i add milk to the list" (lists_query) vs "add milk to my list" (lists_createoradd).
QUESTION_AND_AUXILIARY_WORDS = {
    "what", "when", "where", "who", "whom", "which", "why", "how",
    "do", "does", "did", "is", "are", "was", "were",
    "can", "could", "will", "would", "should", "have", "has", "had",
}

PROTECTED_WORDS = NEGATION_TOKENS | SEMANTIC_FUNCTION_WORDS | QUESTION_AND_AUXILIARY_WORDS

# "am" is an NLTK stop-word (as in "i am"), but "five am" is a time marker.
NUMBER_WORDS = {
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "ten", "eleven", "twelve", "fifteen", "twenty", "thirty", "forty", "fifty",
    "noon", "oh", "clock", "quarter", "half",
}

# Dotted abbreviations are collapsed: "p. m." -> "pm", "a. m." -> "am",
# "b. b. c." -> "bbc", "g. m. t." -> "gmt".  Without this, the single letters
# are split by the tokenizer and then dropped as stop-words ("m", "t", "s").
ABBREVIATION_PATTERN = re.compile(r"\b(?:[a-z]\s?\.\s?){2,}")


def collapse_abbreviations(text: str) -> str:
    """Join dotted letter sequences into one token (keeps a space around it)."""
    return ABBREVIATION_PATTERN.sub(lambda m: " " + re.sub(r"[^a-z]", "", m.group(0)) + " ", text)


def setup_nltk():
    """Ensure essential NLTK resources and tokenizers are available."""
    packages = [
        "punkt", "punkt_tab", "stopwords", "wordnet",
        "averaged_perceptron_tagger", "averaged_perceptron_tagger_eng",
    ]
    for pkg in packages:
        try:
            nltk.download(pkg, quiet=True)
        except Exception as e:
            print(f"[WARN] NLTK download '{pkg}' encountered: {e}")


def initialize_nlp_tools(protected_words=None):
    """Build the stop-word set (negations + intent-bearing function words kept)
    and initialize the lemmatizer."""
    setup_nltk()
    if protected_words is None:
        protected_words = PROTECTED_WORDS
    stop_words = set(stopwords.words("english")) - set(protected_words)
    lemmatizer = WordNetLemmatizer()
    return stop_words, lemmatizer


def _wordnet_pos(treebank_tag: str) -> str:
    """Map a Penn Treebank POS tag to a WordNet POS constant."""
    if treebank_tag.startswith("J"):
        return wordnet.ADJ
    if treebank_tag.startswith("V"):
        return wordnet.VERB
    if treebank_tag.startswith("R"):
        return wordnet.ADV
    return wordnet.NOUN


def _pos_tag(tokens):
    """POS-tag tokens; return None if the tagger resource is unavailable."""
    try:
        return nltk.pos_tag(tokens)
    except LookupError:
        return None


def _lemmatize(tokens, lemmatizer):
    """POS-aware lemmatization (falls back to verb-then-noun if no tagger)."""
    tagged = _pos_tag(tokens)
    if tagged is not None:
        return [lemmatizer.lemmatize(tok, _wordnet_pos(tag)) for tok, tag in tagged]
    lemmas = []
    for tok in tokens:
        as_verb = lemmatizer.lemmatize(tok, wordnet.VERB)
        lemmas.append(as_verb if as_verb != tok else lemmatizer.lemmatize(tok, wordnet.NOUN))
    return lemmas


def clean_text_pipeline(text: str, stop_words: set, lemmatizer: WordNetLemmatizer) -> str:
    """
    Preprocessing pipeline:
    1. String cast & lowercasing
    2. Contraction expansion (e.g., don't -> do not)
    3. Dotted-abbreviation / time-marker normalisation (p. m. -> pm, b. b. c. -> bbc)
    4. URL and Email stripping
    5. Alphanumeric regex cleaning (retaining alphanumeric & spaces)
    6. NLTK word tokenization, then removal of the wake word "olly"
    7. Stop-word filtering (negations + intent-bearing function words kept;
       "am" kept only as a time marker)
    8. POS-aware WordNet lemmatization
    9. Fallback: never return an empty string
    """
    if not isinstance(text, str):
        return ""

    # 1. Lowercase
    text = text.lower().strip()

    # 2. Contraction expansion
    text = expand_contractions(text)

    # 3. Dotted abbreviations / time markers must be handled before punctuation is stripped
    text = collapse_abbreviations(text)

    # 4. URL & Email strip
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)

    # 5. Alphanumeric regex cleaning
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    # 6. Tokenization
    tokens = word_tokenize(text)
    if not tokens:
        return ""

    # 6b. Wake-word removal (keep the tokens if nothing else would remain)
    without_wake = [t for t in tokens if t not in WAKE_WORDS]
    if without_wake:
        tokens = without_wake

    # 7. Stop-word removal ("am" survives only right after a number / number word)
    kept = []
    for i, token in enumerate(tokens):
        if token == "am" and i > 0 and (tokens[i - 1].isdigit() or tokens[i - 1] in NUMBER_WORDS):
            kept.append(token)
        elif token not in stop_words:
            kept.append(token)

    # 9. Fallback: an utterance made only of stop-words keeps its tokens
    if not kept:
        kept = tokens

    # 8. Lemmatization (POS-aware); the time marker "am" must not become "be"
    lemmas = _lemmatize(kept, lemmatizer)
    for i, token in enumerate(kept):
        if token == "am" and i > 0 and (kept[i - 1].isdigit() or kept[i - 1] in NUMBER_WORDS):
            lemmas[i] = "am"
    return " ".join(lemmas)


def compute_vocabulary(series: pd.Series) -> set:
    """Extract unique vocabulary set from text series."""
    vocab = set()
    for text in series.dropna():
        for token in str(text).split():
            vocab.add(token)
    return vocab


def count_label_conflicts(texts: pd.Series, labels: pd.Series):
    """Number of distinct texts that are attached to more than one intent.

    Used as a quality-control check: preprocessing must not *create* conflicts
    that do not exist in the raw data (e.g. 'turn on/off the light')."""
    tmp = pd.DataFrame({"t": texts.astype(str).str.strip().str.lower(), "y": labels.values})
    per_text = tmp.groupby("t")["y"].nunique()
    return int((per_text > 1).sum()), int(len(per_text))


def data_quality_audit(df: pd.DataFrame, text_col: str, intent_col: str) -> dict:
    """Concrete, countable data-quality findings on the RAW utterances."""
    s = df[text_col].astype(str)
    counts = df[intent_col].value_counts()
    conflicts, unique_texts = count_label_conflicts(s, df[intent_col])
    return {
        "rows": int(len(df)),
        "missing_values": int(df[[text_col, intent_col]].isna().sum().sum()),
        "exact_duplicate_utterances": int(s.duplicated().sum()),
        "utterances_with_uppercase": int(s.str.contains(r"[A-Z]").sum()),
        "utterances_with_apostrophe_contraction": int(s.str.contains(r"['’]").sum()),
        "utterances_with_punctuation_other_than_apostrophe": int(s.str.contains(r"[^\w\s'’]").sum()),
        "utterances_with_dotted_time_or_abbreviation": int(s.str.contains(r"\b[a-z]\. [a-z]\.", case=False).sum()),
        "utterances_with_digits": int(s.str.contains(r"\d").sum()),
        "utterances_with_url_or_email": int(s.str.contains(r"https?://|www\.|\S+@\S+").sum()),
        "utterances_with_non_ascii": int(s.apply(lambda x: any(ord(c) > 127 for c in x)).sum()),
        "utterances_with_one_or_two_words": int((s.str.split().str.len() <= 2).sum()),
        "utterances_containing_wake_word_olly": int(s.str.contains(r"\bolly\b", case=False).sum()),
        "raw_texts_with_conflicting_intents": conflicts,
        "unique_raw_texts": unique_texts,
        "class_imbalance_ratio_max_over_min": round(float(counts.max() / counts.min()), 1),
    }


def stratified_split_80_10_10(df: pd.DataFrame, target_col: str, random_state: int = 42):

    """
    Stratifies dataset into 80% train, 10% validation, 10% test.
    Guarantees every class with >= 3 samples is distributed across all partitions,
    safely handling low-frequency classes without throwing sklearn exceptions.
    """
    train_indices = []
    val_indices = []
    test_indices = []

    rng = np.random.RandomState(random_state)

    for _, group in df.groupby(target_col):
        indices = group.index.values.copy()
        rng.shuffle(indices)
        n = len(indices)

        if n == 1:
            train_indices.extend(indices)
        elif n == 2:
            train_indices.append(indices[0])
            test_indices.append(indices[1])
        else:
            n_train = max(1, int(round(n * 0.80)))
            n_val = max(1, int(round(n * 0.10)))
            if n_train + n_val >= n:
                n_train = n - 2
                n_val = 1
                n_test = 1
            else:
                n_test = n - n_train - n_val

            train_indices.extend(indices[:n_train])
            val_indices.extend(indices[n_train:n_train + n_val])
            test_indices.extend(indices[n_train + n_val:])

    train_df = df.loc[train_indices].sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    val_df = df.loc[val_indices].sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    test_df = df.loc[test_indices].sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    return train_df, val_df, test_df




# Categories used to build the "before / after" examples for the proposal.
EXAMPLE_CATEGORIES = [
    ("Contraction expansion",        r"\b(?:don't|can't|what's|i'm|won't|isn't)\b", None),
    ("Negation preserved",           r"\b(?:don't|not|never)\b", None),
    ("on/off kept (light on)",       r"\bon\b", "iot_hue_lighton"),
    ("on/off kept (light off)",      r"\boff\b", "iot_hue_lightoff"),
    ("up/down kept (volume up)",     r"\bup\b", "audio_volume_up"),
    ("up/down kept (volume down)",   r"\bdown\b", "audio_volume_down"),
    ("Time marker (p. m. -> pm)",    r"\b[ap]\. m\.", None),
    ("Dotted abbreviation",          r"\b[a-z]\. [a-z]\. [a-z]\.", None),
    ("Verb forms (lemmatization)",   r"\b(?:setting|playing|waking|reminding|turning|ordering)\b", None),
    ("Wake word 'olly' removed",     r"^olly\b", None),
    ("Time marker (am after a number)", r"\bfive am\b|\bten am\b|\bseven am\b|\bsix am\b", None),
]


def build_before_after_examples(df: pd.DataFrame, text_col: str, intent_col: str) -> pd.DataFrame:
    """Pick one real utterance per category (deterministic) for documentation."""
    rows = []
    word_counts = df[text_col].astype(str).str.split().str.len()
    for label, pattern, intent in EXAMPLE_CATEGORIES:
        mask = df[text_col].astype(str).str.contains(pattern, regex=True) & word_counts.between(4, 10)
        if intent is not None:
            mask &= df[intent_col] == intent
        hit = df[mask]
        if hit.empty:
            continue
        r = hit.iloc[0]
        rows.append({
            "category": label,
            "intent": r[intent_col],
            "raw_text": r[text_col],
            "after_preprocessing": r["clean_text"],
            "norm_text_no_stopword_removal": r["norm_text"],
        })
    return pd.DataFrame(rows)


def export_before_after_examples(examples: pd.DataFrame) -> None:
    examples.to_csv(EXAMPLES_CSV_PATH, index=False, encoding="utf-8")
    lines = ["# Before / after preprocessing (real utterances from Amazon MASSIVE en-US)", "",
             "| Category | Intent | Raw text | clean_text (stop-words removed) | norm_text (all words kept) |",
             "|---|---|---|---|---|"]
    for r in examples.itertuples(index=False):
        lines.append(
            f"| {r.category} | `{r.intent}` | {r.raw_text} | {r.after_preprocessing} | {r.norm_text_no_stopword_removal} |"
        )
    with open(EXAMPLES_MD_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def run_preprocessing_pipeline():
    """Executes data ingestion, text preprocessing, EDA calculation, and export."""
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    if not os.path.exists(RAW_DATA_PATH):
        raise FileNotFoundError(
            f"Raw dataset not found at: {RAW_DATA_PATH}\n"
            "Please execute 'src/download_data.py' first."
        )

    print(f"[INFO] Loading raw dataset from: {RAW_DATA_PATH}")
    df = pd.read_csv(RAW_DATA_PATH)

    # Resolve text column ('utt' or 'text')
    text_col = "utt" if "utt" in df.columns else ("text" if "text" in df.columns else df.columns[1])
    intent_col = "intent" if "intent" in df.columns else "label"

    print(f"[INFO] Dataset loaded with {len(df):,} records.")
    print(f"[INFO] Text column: '{text_col}', Intent column: '{intent_col}'")

    # Drop null utterances or intents
    initial_count = len(df)
    df = df.dropna(subset=[text_col, intent_col]).copy()
    dropped_nulls = initial_count - len(df)
    if dropped_nulls > 0:
        print(f"[INFO] Dropped {dropped_nulls} records with null values.")

    # Raw data-quality audit (counts used in Section 5 of the proposal)
    print("[INFO] Auditing raw data quality...")
    audit = data_quality_audit(df, text_col, intent_col)

    # De-duplication: identical (utterance, intent) pairs would leak between train and test.
    # Same text with a DIFFERENT intent is label noise and is kept (it is reported, not hidden).
    rows_before_dedup = len(df)
    df = df.drop_duplicates(subset=[text_col, intent_col]).reset_index(drop=True)
    rows_removed = rows_before_dedup - len(df)
    print(f"[INFO] Removed {rows_removed} exact duplicate (utterance, intent) rows -> {len(df):,} rows remain.")

    # Initialize NLP tools
    stop_words, lemmatizer = initialize_nlp_tools()

    # Raw EDA statistics
    print("[INFO] Computing raw EDA metrics...")
    raw_vocab = compute_vocabulary(df[text_col])
    raw_word_counts = df[text_col].apply(lambda s: len(str(s).split()))
    raw_char_counts = df[text_col].apply(lambda s: len(str(s)))

    # Apply preprocessing pipeline
    print("[INFO] Applying normalization, contraction expansion, stop-word filtering & POS-aware lemmatization...")
    df["clean_text"] = df[text_col].apply(
        lambda t: clean_text_pipeline(t, stop_words, lemmatizer)
    )

    # Second representation: same normalisation + lemmatization, but NO stop-word removal.
    # The ablation (src/benchmark_preprocessing.py) shows that stop-word removal hurts intent
    # classification on short commands, so 'norm_text' is the recommended input for TF-IDF/BoW.
    df["norm_text"] = df[text_col].apply(
        lambda t: clean_text_pipeline(t, set(), lemmatizer)
    )

    # Cleaned EDA statistics
    print("[INFO] Computing processed EDA metrics...")
    clean_vocab = compute_vocabulary(df["clean_text"])
    clean_word_counts = df["clean_text"].apply(lambda s: len(str(s).split()))
    clean_char_counts = df["clean_text"].apply(lambda s: len(str(s)))

    # Quality control: preprocessing must not create empty rows or new label conflicts
    empty_clean = int((df["clean_text"].str.strip() == "").sum())
    raw_conflicts, _ = count_label_conflicts(df[text_col], df[intent_col])
    clean_conflicts, clean_unique = count_label_conflicts(df["clean_text"], df[intent_col])
    norm_conflicts, _ = count_label_conflicts(df["norm_text"], df[intent_col])
    norm_vocab = compute_vocabulary(df["norm_text"])
    print(f"[QC] Empty cleaned utterances: {empty_clean}")
    print(f"[QC] Texts with conflicting intents  raw: {raw_conflicts} | clean_text: {clean_conflicts} | norm_text: {norm_conflicts}")

    # Class distribution analysis
    intent_counts = df[intent_col].value_counts().to_dict()
    num_classes = len(intent_counts)
    min_class_samples = min(intent_counts.values())
    max_class_samples = max(intent_counts.values())
    avg_class_samples = float(np.mean(list(intent_counts.values())))

    scenario_counts = {}
    if "scenario" in df.columns:
        scenario_counts = df["scenario"].value_counts().to_dict()

    eda_summary = {
        "dataset_name": "Amazon MASSIVE (en-US subset)",
        "course": "Natural Language Processing (A0443404) - Middle East University (MEU)",
        "task": "Intent Classification (Single-label Multi-class)",
        "total_samples": int(len(df)),
        "num_classes": int(num_classes),
        "raw_data_quality_audit": audit,
        "deduplication": {
            "rows_before": int(rows_before_dedup),
            "exact_duplicate_rows_removed": int(rows_removed),
            "rows_after": int(len(df)),
        },
        "preprocessing_config": {
            "lowercase": True,
            "contraction_expansion": True,
            "abbreviation_normalisation": "p. m. -> pm, a. m. -> am, b. b. c. -> bbc",
            "stop_words": "NLTK English minus protected words",
            "protected_words_kept": sorted(PROTECTED_WORDS),
            "am_rule": "kept only after a digit or number word (and never lemmatized to 'be')",
            "wake_words_removed": sorted(WAKE_WORDS),
            "lemmatization": "WordNet, POS-aware (NLTK pos_tag)",
            "empty_result_fallback": "keep all tokens"
        },
        "raw_text_metrics": {
            "vocab_size": int(len(raw_vocab)),
            "avg_word_count": round(float(raw_word_counts.mean()), 2),
            "std_word_count": round(float(raw_word_counts.std()), 2),
            "min_word_count": int(raw_word_counts.min()),
            "max_word_count": int(raw_word_counts.max()),
            "avg_char_length": round(float(raw_char_counts.mean()), 2)
        },
        "clean_text_metrics": {
            "vocab_size": int(len(clean_vocab)),
            "avg_word_count": round(float(clean_word_counts.mean()), 2),
            "std_word_count": round(float(clean_word_counts.std()), 2),
            "min_word_count": int(clean_word_counts.min()),
            "max_word_count": int(clean_word_counts.max()),
            "avg_char_length": round(float(clean_char_counts.mean()), 2),
            "vocab_reduction_pct": round((1.0 - (len(clean_vocab) / len(raw_vocab))) * 100, 2)
        },
        "norm_text_metrics": {
            "description": "same normalisation + POS-aware lemmatization, no stop-word removal",
            "vocab_size": int(len(norm_vocab)),
            "avg_word_count": round(float(df["norm_text"].str.split().str.len().mean()), 2)
        },
        "quality_control": {
            "empty_cleaned_utterances": empty_clean,
            "texts_with_conflicting_intents_raw": raw_conflicts,
            "texts_with_conflicting_intents_clean_text": clean_conflicts,
            "texts_with_conflicting_intents_norm_text": norm_conflicts,
            "conflicts_created_by_stopword_removal": clean_conflicts - norm_conflicts
        },
        "class_distribution_summary": {
            "num_unique_intents": num_classes,
            "min_samples_per_intent": min_class_samples,
            "max_samples_per_intent": max_class_samples,
            "mean_samples_per_intent": avg_class_samples,
            "is_balanced": bool(min_class_samples == max_class_samples)
        },
        "top_10_intents": dict(list(intent_counts.items())[:10]),
        "bottom_5_intents": dict(list(intent_counts.items())[-5:]),
        "scenario_distribution": scenario_counts
    }

    # Stratified 80/10/10 Train/Validation/Test Split
    print("[INFO] Performing stratified 80/10/10 train/val/test split...")
    train_df, val_df, test_df = stratified_split_80_10_10(
        df,
        target_col=intent_col,
        random_state=42
    )

    eda_summary["split_distribution"] = {
        "train_samples": int(len(train_df)),
        "train_pct": round(len(train_df) / len(df) * 100, 2),
        "val_samples": int(len(val_df)),
        "val_pct": round(len(val_df) / len(df) * 100, 2),
        "test_samples": int(len(test_df)),
        "test_pct": round(len(test_df) / len(df) * 100, 2),
        "classes_in_train": int(train_df[intent_col].nunique()),
        "classes_in_val": int(val_df[intent_col].nunique()),
        "classes_in_test": int(test_df[intent_col].nunique())
    }

    # Save cleaned splits
    train_df.to_csv(TRAIN_OUTPUT_PATH, index=False, encoding="utf-8")
    val_df.to_csv(VAL_OUTPUT_PATH, index=False, encoding="utf-8")
    test_df.to_csv(TEST_OUTPUT_PATH, index=False, encoding="utf-8")

    print(f"[SUCCESS] Exported train partition to: {TRAIN_OUTPUT_PATH} ({len(train_df):,} rows)")
    print(f"[SUCCESS] Exported val partition to:   {VAL_OUTPUT_PATH} ({len(val_df):,} rows)")
    print(f"[SUCCESS] Exported test partition to:  {TEST_OUTPUT_PATH} ({len(test_df):,} rows)")

    # Export JSON report
    with open(REPORT_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(eda_summary, f, indent=4)
    print(f"[SUCCESS] Exported preprocessing & EDA report to: {REPORT_OUTPUT_PATH}")

    # Export real before/after examples (used in the proposal, Section 6)
    examples = build_before_after_examples(df, text_col, intent_col)
    export_before_after_examples(examples)
    print(f"[SUCCESS] Exported before/after examples to: {EXAMPLES_MD_PATH}")

    # Display sample transformations
    print("\n--- SAMPLE PREPROCESSING TRANSFORMATIONS ---")
    for r in examples.itertuples(index=False):
        print(f"[{r.intent}] ({r.category})")
        print(f"  Raw:   \"{r.raw_text}\"")
        print(f"  Clean: \"{r.after_preprocessing}\"")
        print(f"  Norm:  \"{r.norm_text_no_stopword_removal}\"")
    print("-------------------------------------------\n")

    return eda_summary


if __name__ == "__main__":
    run_preprocessing_pipeline()
