# scripts/prepare_dataset.py
import os
import re

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ─────────────────────────────────────────────
# NLTK setup
# ─────────────────────────────────────────────
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("punkt", quiet=True)

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

ASPECTS = [
    "course_content",
    "teaching_quality",
    "assessment",
    "resources",
    "infrastructure",
]


def clean_text(text: str) -> str:
    """Lowercase, remove non-letters, remove stopwords, lemmatize."""
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", "", text)
    tokens = [lemmatizer.lemmatize(w) for w in text.split() if w not in stop_words]
    return " ".join(tokens)


def prepare_dataset(csv_path: str):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found at {csv_path}. Please check path.")

    df = pd.read_csv(csv_path)

    # Clean text
    df["clean_text"] = df["text"].apply(clean_text)

    # Keep only relevant columns for training
    keep_cols = ["text", "clean_text"] + ASPECTS
    df = df[keep_cols]

    print(f"✅ Dataset loaded and cleaned: {len(df)} reviews")
    print("Sample rows:")
    print(df.head(3).to_string())

    # Save cleaned dataset
    df.to_csv(csv_path, index=False)
    print(f"\n✅ Saved cleaned dataset to {csv_path}")
    return df


if __name__ == "__main__":
    # Relative path from scripts/ to data/
    csv_file = r"D:\_programming\edu-opinion-miner\data\dataset.csv"
    prepare_dataset(csv_file)
