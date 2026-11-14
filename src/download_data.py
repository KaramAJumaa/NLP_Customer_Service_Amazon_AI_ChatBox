"""
Data Ingestion Pipeline for Intent Classification Project
Dataset: Amazon MASSIVE (English subset: en-US)
Course: Natural Language Processing (A0443404) - Middle East University (MEU)
"""

import os
import io
import json
import tarfile
import urllib.request
import pandas as pd

# Flash-drive portable relative path resolution
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "Data")
OUTPUT_PATH = os.path.join(DATA_DIR, "massive_english.csv")

DATASET_URL = "https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.0.tar.gz"
TARGET_MEMBER = "1.0/data/en-US.jsonl"


def download_massive_english(force_download=False):
    """
    Downloads Amazon MASSIVE dataset archive and extracts exclusively the
    en-US (English) localization with zero third-party compilation barriers.
    """
    os.makedirs(DATA_DIR, exist_ok=True)

    # Check if valid dataset already exists
    if os.path.exists(OUTPUT_PATH) and not force_download:
        try:
            existing_df = pd.read_csv(OUTPUT_PATH)
            if "utt" in existing_df.columns and "intent" in existing_df.columns:
                # Check that it is purely English subset (~16,521 rows)
                if len(existing_df) == 16521 or ("locale" in existing_df.columns and (existing_df["locale"] == "en-US").all()):
                    print(f"[INFO] Verified existing valid dataset at: {OUTPUT_PATH}")
                    print(f"[INFO] Total records: {len(existing_df):,} | Unique intents: {existing_df['intent'].nunique()}")
                    return OUTPUT_PATH
        except Exception:
            print("[WARN] Existing file corrupted or invalid format. Re-downloading...")

    print(f"[INFO] Downloading Amazon MASSIVE archive from:\n       {DATASET_URL}")
    print("[INFO] Streaming and extracting English partition (en-US)...")

    req = urllib.request.Request(
        DATASET_URL,
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    )

    records = []
    with urllib.request.urlopen(req) as response:
        # Stream the compressed tarfile directly without writing full 40MB tar to disk
        with tarfile.open(fileobj=response, mode="r|gz") as tar:
            found = False
            for member in tar:
                if member.name.endswith("en-US.jsonl"):
                    found = True
                    print(f"[INFO] Found target archive member: {member.name}")
                    extracted = tar.extractfile(member)
                    if extracted:
                        for line in extracted:
                            if not line.strip():
                                continue
                            try:
                                item = json.loads(line.decode("utf-8"))
                                records.append({
                                    "id": item.get("id", ""),
                                    "utt": item.get("utt", "").strip(),
                                    "intent": item.get("intent", "").strip(),
                                    "scenario": item.get("scenario", "").strip(),
                                    "partition": item.get("partition", "").strip(),
                                    "locale": item.get("locale", "en-US")
                                })
                            except json.JSONDecodeError:
                                continue
                    break

            if not found:
                raise RuntimeError(f"Target member '{TARGET_MEMBER}' not found in downloaded archive.")

    df = pd.DataFrame(records)

    # Sanity checks
    if df.empty:
        raise ValueError("Extraction yielded 0 records from en-US.jsonl.")

    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")
    print(f"[SUCCESS] Saved raw English dataset to: {OUTPUT_PATH}")
    print(f"[INFO] Dimensions: {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"[INFO] Target Classes (Intents): {df['intent'].nunique()} classes across {df['scenario'].nunique()} scenarios.")
    return OUTPUT_PATH


if __name__ == "__main__":
    download_massive_english(force_download=True)
