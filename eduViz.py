# scripts/eduViz.py
"""
Visualization module for EduSense Dashboard
============================================
Provides functions to generate interactive charts from aspect-based sentiment predictions.
"""

import pandas as pd
import plotly.express as px


def plot_aspect_distribution(df, ASPECT_DISPLAY, ASPECTS):
    """
    Plot distribution of aspects in the dataset.
    df: DataFrame containing aspect columns
    ASPECT_DISPLAY: dict mapping aspect keys to display names
    ASPECTS: list of aspect keys
    """
    aspect_counts = {
        ASPECT_DISPLAY[a]: df[a].replace("none", pd.NA).count() for a in ASPECTS
    }
    aspect_df = pd.DataFrame(
        {"Aspect": list(aspect_counts.keys()), "Count": list(aspect_counts.values())}
    )

    fig = px.bar(
        aspect_df,
        x="Aspect",
        y="Count",
        text="Count",
        title="Aspect Distribution in Dataset",
    )
    fig.update_traces(textposition="outside")
    return fig


def plot_model_performance_heatmap(model_df):
    """
    Plot heatmap of F1 scores for different models across aspects.
    model_df: DataFrame with columns ["Aspect", "Model", "F1-Score"]
    """
    pivot_df = model_df.pivot(index="Aspect", columns="Model", values="F1-Score")
    fig = px.imshow(
        pivot_df,
        text_auto=True,
        aspect="auto",
        title="Model F1-Score Heatmap Across Aspects",
    )
    return fig


def plot_avg_model_performance(model_df):
    """
    Plot average F1-score per model.
    model_df: DataFrame with columns ["Aspect", "Model", "F1-Score"]
    """
    avg_scores = model_df.groupby("Model")["F1-Score"].mean().reset_index()
    fig = px.bar(
        avg_scores,
        x="Model",
        y="F1-Score",
        text="F1-Score",
        title="Average Model Performance (F1-Score)",
    )
    fig.update_traces(textposition="outside")
    return fig


def plot_baseline_comparison(baseline_df):
    """
    Plot baseline performance comparison.
    baseline_df: DataFrame with columns ["Method", "Accuracy"]
    """
    fig = px.bar(
        baseline_df,
        x="Method",
        y="Accuracy",
        text="Accuracy",
        title="Baseline vs ABSA System Accuracy",
    )
    fig.update_traces(textposition="outside")
    return fig


def plot_aspect_sentiment_bar(results, ASPECT_DISPLAY, ASPECT_KEYWORDS, text):
    """
    Plot horizontal bar chart showing sentiment scores for mentioned aspects.
    results: dict returned by predict_single
    text: original review text
    """
    lower_text = text.lower()
    score_map = {"positive": 1, "neutral": 0, "negative": -1}

    mentioned = {
        a: s
        for a, s in results.items()
        if s not in ("not mentioned", "model not loaded")
    }
    # if not mentioned:
    #     return None  # nothing to plot
    if not mentioned:
        # Return empty chart
        chart_df = pd.DataFrame({"Aspect": [], "Score": [], "Sentiment": []})
        fig = px.bar(
            chart_df,
            x="Score",
            y="Aspect",
            orientation="h",
            color="Sentiment",
            title="Aspect Sentiment Scores (No aspects detected)",
        )
        return fig

    aspect_scores = {}
    for aspect, sentiment in mentioned.items():
        total_score = sum(
            score_map[sentiment] for kw in ASPECT_KEYWORDS[aspect] if kw in lower_text
        )
        aspect_scores[ASPECT_DISPLAY[aspect]] = total_score

    chart_df = pd.DataFrame(
        {
            "Aspect": list(aspect_scores.keys()),
            "Score": list(aspect_scores.values()),
            "Sentiment": [
                "positive" if s > 0 else "negative" if s < 0 else "neutral"
                for s in aspect_scores.values()
            ],
        }
    )

    fig = px.bar(
        chart_df,
        x="Score",
        y="Aspect",
        orientation="h",
        color="Sentiment",
        text="Score",
        color_discrete_map={"positive": "green", "neutral": "gray", "negative": "red"},
        title="Aspect Sentiment Scores",
    )
    fig.update_traces(textposition="inside")
    fig.update_layout(
        xaxis_title="Sentiment Score (Stacked Mentions)", yaxis_title="Aspect"
    )
    return fig


def plot_keyword_contributions(results, ASPECT_DISPLAY, ASPECT_KEYWORDS, text):
    """
    Plot contributions of individual keywords to aspect sentiment.
    results: dict returned by predict_single
    text: original review text
    """
    lower_text = text.lower()
    score_map = {"positive": 1, "neutral": 0, "negative": -1}

    word_rows = []
    for aspect, sentiment in results.items():
        if sentiment in ("not mentioned", "model not loaded"):
            continue
        for kw in ASPECT_KEYWORDS[aspect]:
            if kw in lower_text:
                word_rows.append(
                    {
                        "Word": kw,
                        "Aspect": ASPECT_DISPLAY[aspect],
                        "Sentiment": sentiment,
                        "Score": score_map[sentiment],
                        "Position": lower_text.find(kw),
                    }
                )

    # if not word_rows:
    #     return None
    if not word_rows:
        chart_df = pd.DataFrame({"Word": [], "Score": [], "Sentiment": []})
        fig = px.bar(
            chart_df,
            x="Score",
            y="Word",
            orientation="h",
            title="Keyword Contribution (No keywords detected)",
        )
        return fig

    word_df = pd.DataFrame(word_rows).sort_values("Position")

    fig = px.bar(
        word_df,
        x="Score",
        y="Word",
        orientation="h",
        color="Sentiment",
        text="Aspect",
        color_discrete_map={"positive": "green", "neutral": "gray", "negative": "red"},
        category_orders={"Word": word_df["Word"].tolist()},
        title="Keyword Contribution to Sentiment",
    )
    fig.update_layout(
        xaxis_title="Sentiment Contribution", yaxis_title="Detected Keywords"
    )
    fig.update_xaxes(range=[-1, 1], dtick=1)
    fig.update_traces(textposition="inside")
    return fig
