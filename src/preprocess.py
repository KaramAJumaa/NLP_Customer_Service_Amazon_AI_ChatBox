"""
Text Preprocessing & Exploratory Data Analysis (EDA) Pipeline
Project: Intent Classification
Course: Natural Language Processing (A0443404) - Middle East University (MEU)
Dataset: Amazon MASSIVE (en-US Localization)
"""

import os
import re
import json
from collections import Counter
import pandas as pd
import numpy as np

import nltk
from nltk.corpus import stopwords
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


def setup_nltk():
    """Ensure essential NLTK resources and tokenizers are available."""
    packages = ["punkt", "punkt_tab", "stopwords", "wordnet"]
    for pkg in packages:
        try:
            nltk.download(pkg, quiet=True)
        except Exception as e:
            print(f"[WARN] NLTK download '{pkg}' encountered: {e}")


def initialize_nlp_tools():
    """Build stopwords set retaining critical negation tokens and initialize lemmatizer."""
    setup_nltk()
    stop_words = set(stopwords.words("english"))
    # In intent classification, negations carry crucial semantics
    # (e.g., 'do not cancel order' vs 'cancel order')
    negation_tokens = {"not", "no", "nor", "never", "neither", "none"}
    stop_words = stop_words - negation_tokens
    lemmatizer = WordNetLemmatizer()
    return stop_words, lemmatizer


def clean_text_pipeline(text: str, stop_words: set, lemmatizer: WordNetLemmatizer) -> str:
    """
    Production preprocessing pipeline:
    1. String cast & lowercasing
    2. Contraction expansion (e.g., don't -> do not)
    3. URL and Email stripping
    4. Alphanumeric regex cleaning (retaining alphanumeric & spaces)
    5. NLTK word tokenization
    6. Stop-word filtering (preserving negation markers)
    7. WordNet lemmatization
    """
    if not isinstance(text, str):
        return ""

    # 1. Lowercase
    text = text.lower().strip()

    # 2. Contraction expansion
    text = expand_contractions(text)

    # 3. URL & Email strip
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)

    # 4. Alphanumeric regex cleaning
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    # 5. Tokenization
    tokens = word_tokenize(text)

    # 6. Stop-word removal & 7. Lemmatization
    cleaned_tokens = [
        lemmatizer.lemmatize(token)
        for token in tokens
        if token not in stop_words and len(token) > 0
    ]

    return " ".join(cleaned_tokens)


def compute_vocabulary(series: pd.Series) -> set:
    """Extract unique vocabulary set from text series."""
    vocab = set()
    for text in series.dropna():
        for token in str(text).split():
            vocab.add(token)
    return vocab


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

    # Initialize NLP tools
    stop_words, lemmatizer = initialize_nlp_tools()

    # Raw EDA statistics
    print("[INFO] Computing raw EDA metrics...")
    raw_vocab = compute_vocabulary(df[text_col])
    raw_word_counts = df[text_col].apply(lambda s: len(str(s).split()))
    raw_char_counts = df[text_col].apply(lambda s: len(str(s)))

    # Apply preprocessing pipeline
    print("[INFO] Applying text normalization, contraction expansion, stop-word filtering & lemmatization...")
    df["clean_text"] = df[text_col].apply(
        lambda t: clean_text_pipeline(t, stop_words, lemmatizer)
    )

    # Cleaned EDA statistics
    print("[INFO] Computing processed EDA metrics...")
    clean_vocab = compute_vocabulary(df["clean_text"])
    clean_word_counts = df["clean_text"].apply(lambda s: len(str(s).split()))
    clean_char_counts = df["clean_text"].apply(lambda s: len(str(s)))

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

    # Display sample transformations
    print("\n--- SAMPLE PREPROCESSING TRANSFORMATIONS ---")
    for idx in range(min(5, len(df))):
        raw_val = df[text_col].iloc[idx]
        clean_val = df["clean_text"].iloc[idx]
        intent_val = df[intent_col].iloc[idx]
        print(f"[{intent_val}]")
        print(f"  Raw:   \"{raw_val}\"")
        print(f"  Clean: \"{clean_val}\"")
    print("-------------------------------------------\n")

    return eda_summary


if __name__ == "__main__":
    run_preprocessing_pipeline()
