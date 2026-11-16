"""
Preprocessing ablation (Phase 1 quality-control benchmark)
Course: Natural Language Processing (A0443404) - Middle East University (MEU)

Purpose
-------
Justify the preprocessing choices with evidence instead of assumptions.
Four cleaning variants are applied to the SAME train / validation split and
compared with (a) data-level quality checks and (b) a deliberately simple
TF-IDF + LinearSVC sanity classifier.

IMPORTANT: this is NOT the Phase 2 model. The classifier is only a quick probe
that shows how much intent information each variant keeps. The full model
development (BoW / TF-IDF / Sentence-Transformers, LogisticRegression, SVM,
DistilBERT) is left for Phase 2.

Variants
--------
1. lowercase_only      : lowercase + contractions + punctuation removal (no stop-word removal, no lemmatization)
2. legacy_v1           : NLTK stop-words (only negations kept) + noun-only lemmatization (first pipeline)
3. v2_function_words   : + on/off/up/down/in/out... kept, POS-aware lemmatization, abbreviation collapsing
4. v2_final            : v2_function_words + question words / auxiliaries kept (clean_text column)
5. v2_keep_all_words   : same normalisation + POS-aware lemmatization, NO stop-word removal (norm_text column)

Run:  python src/benchmark_preprocessing.py
"""

import os
import re
import json
import warnings

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.svm import LinearSVC
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

import preprocess as P

warnings.filterwarnings("ignore")

BASE_DIR = P.BASE_DIR
ABLATION_PATH = os.path.join(P.OUTPUTS_DIR, "preprocessing_ablation.json")
ABLATION_MD_PATH = os.path.join(P.OUTPUTS_DIR, "preprocessing_ablation.md")


# ---------------------------------------------------------------------------
# Cleaners
# ---------------------------------------------------------------------------
def make_lowercase_only():
    def clean(text):
        text = P.expand_contractions(str(text).lower().strip())
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        return re.sub(r"\s+", " ", text).strip()
    return clean


def make_legacy_v1(lemmatizer):
    """First pipeline: default NLTK stop-words (negations kept), noun-only lemmatization."""
    stop = set(stopwords.words("english")) - P.NEGATION_TOKENS

    def clean(text):
        text = P.expand_contractions(str(text).lower().strip())
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return " ".join(lemmatizer.lemmatize(t) for t in word_tokenize(text) if t not in stop)
    return clean


def make_v2(protected_words):
    stop_words, lemmatizer = P.initialize_nlp_tools(protected_words)
    return lambda text: P.clean_text_pipeline(text, stop_words, lemmatizer)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def evaluate_variant(name, cleaner, raw_all, intents_all, train_raw, train_y, val_raw, val_y):
    clean_all = raw_all.apply(cleaner)
    conflicts, _ = P.count_label_conflicts(clean_all, intents_all)
    empty_rows = int((clean_all.str.strip() == "").sum())
    vocab = len(P.compute_vocabulary(clean_all))

    train_x = train_raw.apply(cleaner)
    val_x = val_raw.apply(cleaner)
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
    xtr = vec.fit_transform(train_x)
    xva = vec.transform(val_x)
    clf = LinearSVC(C=1.0, random_state=42).fit(xtr, train_y)
    pred = clf.predict(xva)

    return {
        "variant": name,
        "vocab_size": vocab,
        "empty_utterances": empty_rows,
        "texts_with_conflicting_intents": conflicts,
        "val_accuracy": round(float(accuracy_score(val_y, pred)), 4),
        "val_macro_f1": round(float(f1_score(val_y, pred, average="macro")), 4),
    }


def run_ablation():
    P.setup_nltk()
    raw = pd.read_csv(P.RAW_DATA_PATH).drop_duplicates(subset=["utt", "intent"]).reset_index(drop=True)
    train = pd.read_csv(P.TRAIN_OUTPUT_PATH)
    val = pd.read_csv(P.VAL_OUTPUT_PATH)

    _, lemmatizer = P.initialize_nlp_tools()
    variants = [
        ("lowercase_only", make_lowercase_only()),
        ("legacy_v1", make_legacy_v1(lemmatizer)),
        ("v2_function_words", make_v2(P.NEGATION_TOKENS | P.SEMANTIC_FUNCTION_WORDS)),
        ("v2_final", make_v2(P.PROTECTED_WORDS)),
        ("v2_keep_all_words", make_v2(set(stopwords.words("english")))),
    ]

    results = []
    for name, cleaner in variants:
        print(f"[INFO] Evaluating variant: {name}")
        results.append(
            evaluate_variant(
                name, cleaner,
                raw["utt"], raw["intent"],
                train["utt"], train["intent"], val["utt"], val["intent"],
            )
        )

    raw_conflicts, _ = P.count_label_conflicts(raw["utt"], raw["intent"])
    report = {
        "note": "TF-IDF (1-2 grams) + LinearSVC is only a sanity probe for preprocessing quality; "
                "it is not the Phase 2 model.",
        "raw_texts_with_conflicting_intents": raw_conflicts,
        "split": {"train_rows": int(len(train)), "val_rows": int(len(val))},
        "results": results,
    }
    with open(ABLATION_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)

    lines = [
        "# Preprocessing ablation (sanity probe: TF-IDF 1-2 grams + LinearSVC, validation split)", "",
        f"Raw data already contains {raw_conflicts} texts attached to more than one intent (label noise).", "",
        "| Variant | Vocab | Empty rows | Conflicting texts | Val accuracy | Val macro-F1 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        lines.append(
            f"| {r['variant']} | {r['vocab_size']:,} | {r['empty_utterances']} | "
            f"{r['texts_with_conflicting_intents']} | {r['val_accuracy']:.4f} | {r['val_macro_f1']:.4f} |"
        )
    with open(ABLATION_MD_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print("\n" + "\n".join(lines))
    print(f"\n[SUCCESS] Saved: {ABLATION_PATH}")
    return report


if __name__ == "__main__":
    run_ablation()
