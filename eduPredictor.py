"""
PREDICTOR MODULE
================
The main interface used by the dashboard.
Load trained models and predict sentiment for any input text.
"""

import os
import pickle
import re

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

ASPECTS = [
    "course_content",
    "teaching_quality",
    "assessment",
    "resources",
    "infrastructure",
]

ASPECT_KEYWORDS = {
    "course_content": [
        "syllabus",
        "curriculum",
        "course",
        "content",
        "topic",
        "subject",
        "module",
    ],
    "teaching_quality": [
        "teacher",
        "professor",
        "faculty",
        "instructor",
        "teaching",
        "lecture",
        "explain",
        "class",
        "staff",
    ],
    "assessment": [
        "exam",
        "test",
        "assignment",
        "quiz",
        "grade",
        "grading",
        "marks",
        "evaluation",
        "deadline",
        "paper",
    ],
    "resources": [
        "library",
        "book",
        "material",
        "resource",
        "journal",
        "reference",
        "notes",
        "handout",
    ],
    "infrastructure": [
        "lab",
        "classroom",
        "wifi",
        "internet",
        "equipment",
        "facility",
        "campus",
        "projector",
        "computer",
        "infrastructure",
    ],
}

ASPECT_DISPLAY = {
    "course_content": "📚 Course Content",
    "teaching_quality": "👨‍🏫 Teaching Quality",
    "assessment": "📝 Assessment",
    "resources": "📖 Resources",
    "infrastructure": "🏫 Infrastructure",
}

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))


def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words]
    return " ".join(tokens)


def is_aspect_mentioned(text, aspect):
    text_lower = text.lower()
    return any(kw in text_lower for kw in ASPECT_KEYWORDS[aspect])


def load_models():
    # Load all saved models from disk
    models = {}

    base_path = os.path.join(os.path.dirname(__file__), "models")

    for aspect in ASPECTS:
        path = os.path.join(base_path, f"{aspect}_model.pkl")

        if os.path.exists(path):
            with open(path, "rb") as f:
                models[aspect] = pickle.load(f)
        else:
            print(f"Missing model: {path}")

    return models


def predict_single(text, models):
    """
    Given a review text, return aspect-wise sentiment.
    Returns: dict of {aspect: sentiment or 'not mentioned'}
    """
    cleaned = clean_text(text)
    results = {}

    for aspect in ASPECTS:
        if not is_aspect_mentioned(text, aspect):
            results[aspect] = "not mentioned"
        else:
            if aspect in models:
                vectorizer = models[aspect]["vectorizer"]
                model = models[aspect]["model"]
                X = vectorizer.transform([cleaned])
                prediction = model.predict(X)[0]
                results[aspect] = prediction
            else:
                results[aspect] = "model not loaded"

    return results


def predict_bulk(texts, models):
    """Predict for a list of texts. Returns list of result dicts."""
    return [predict_single(t, models) for t in texts]


def summarize_bulk(results_list):
    """
    Aggregate bulk predictions.
    Returns: {aspect: {positive: N, negative: N, neutral: N, total_mentioned: N}}
    """
    summary = {
        asp: {"positive": 0, "negative": 0, "neutral": 0, "total": 0} for asp in ASPECTS
    }

    for result in results_list:
        for aspect, sentiment in result.items():
            if sentiment in ("positive", "negative", "neutral"):
                summary[aspect][sentiment] += 1
                summary[aspect]["total"] += 1

    return summary
