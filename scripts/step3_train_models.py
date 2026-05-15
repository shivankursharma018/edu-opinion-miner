"""
STEP 3: MODEL TRAINING + EVALUATION
=====================================
Trains Naive Bayes, SVM, Random Forest for each aspect.
Outputs a full comparison table with Accuracy, Precision, Recall, F1.
This table goes directly into your research paper (Table II or Table III).

Pipeline:
  text → TF-IDF features → ML model → Positive / Negative / Neutral
"""

import pandas as pd
import numpy as np
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate, StratifiedKFold, train_test_split
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

ASPECTS = ["course_content", "teaching_quality", "assessment", "resources", "infrastructure"]

MODELS = {
    "Naive Bayes":   MultinomialNB(),
    "SVM":           LinearSVC(random_state=42, max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
}


def load_aspect_data(df, aspect):
    """Return X (clean text) and y (label) for rows where aspect is mentioned."""
    subset = df[df[aspect] != 'none'].copy()
    return subset['clean_text'].values, subset[aspect].values


def evaluate_model(clf, X_tfidf, y):
    """
    Use 5-fold cross-validation for reliable evaluation on small datasets.
    Returns: accuracy, precision, recall, f1 (all macro-averaged).
    Falls back to simple split if too few samples for cross-val.
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
    try:
        scores = cross_validate(clf, X_tfidf, y, cv=cv, scoring=scoring)
        return {
            "accuracy":  np.mean(scores['test_accuracy']),
            "precision": np.mean(scores['test_precision_macro']),
            "recall":    np.mean(scores['test_recall_macro']),
            "f1":        np.mean(scores['test_f1_macro']),
        }
    except Exception:
        X_tr, X_te, y_tr, y_te = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)
        clf.fit(X_tr, y_tr)
        y_pred = clf.predict(X_te)
        p, r, f, _ = precision_recall_fscore_support(y_te, y_pred, average='macro', zero_division=0)
        return {
            "accuracy":  accuracy_score(y_te, y_pred),
            "precision": p, "recall": r, "f1": f,
        }


def train_system():
    print("=" * 65)
    print("MODEL TRAINING & EVALUATION")
    print("=" * 65)

    df = pd.read_csv("data/dataset.csv")
    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    all_rows = []

    for aspect in ASPECTS:
        X, y = load_aspect_data(df, aspect)
        print(f"\n[{aspect.upper()}]  ({len(X)} samples)")

        vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=500, min_df=1)
        vectorizer.fit(X)
        X_tfidf = vectorizer.transform(X)

        best_f1 = -1
        best_name = None
        best_clf = None

        for model_name, clf in MODELS.items():
            metrics = evaluate_model(clf, X_tfidf, y)
            all_rows.append({
                "Aspect":    aspect,
                "Model":     model_name,
                "Accuracy":  round(metrics["accuracy"]  * 100, 1),
                "Precision": round(metrics["precision"] * 100, 1),
                "Recall":    round(metrics["recall"]    * 100, 1),
                "F1-Score":  round(metrics["f1"]        * 100, 1),
            })
            marker = ""
            if metrics["f1"] > best_f1:
                best_f1 = metrics["f1"]
                best_name = model_name
                # Refit on ALL data for saving
                clf.fit(X_tfidf, y)
                best_clf = clf
                marker = " ← BEST"
            print(f"  {model_name:15s}  Acc:{metrics['accuracy']:.2f}  "
                  f"P:{metrics['precision']:.2f}  R:{metrics['recall']:.2f}  "
                  f"F1:{metrics['f1']:.2f}{marker}")

        # Save best model
        bundle = {"vectorizer": vectorizer, "model": best_clf}
        with open(f"models/{aspect}_model.pkl", "wb") as f:
            pickle.dump(bundle, f)

    results_df = pd.DataFrame(all_rows)
    results_df.to_csv("outputs/model_comparison.csv", index=False)

    print("\n" + "=" * 65)
    print("PAPER-READY RESULTS TABLE (values in %)")
    print("=" * 65)
    print(results_df.to_string(index=False))

    print("\n" + "=" * 65)
    print("AVERAGE F1 PER MODEL (headline result for your paper)")
    print("=" * 65)
    avg = results_df.groupby("Model")["F1-Score"].mean().sort_values(ascending=False)
    for m, f1 in avg.items():
        print(f"  {m:15s}: {f1:.1f}%")
    print(f"\n→ Best overall model: {avg.idxmax()} ({avg.max():.1f}% avg F1)")
    print("\n✅ Saved to outputs/model_comparison.csv")
    return results_df


if __name__ == "__main__":
    os.makedirs("outputs", exist_ok=True)
    train_system()
