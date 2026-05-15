# eduDashboard.py
import os

import pandas as pd
import plotly.express as px
import streamlit as st

from eduPredictor import (
    ASPECT_DISPLAY,
    ASPECT_KEYWORDS,
    ASPECTS,
    load_models,
    predict_single,
)
from eduViz import (
    plot_aspect_distribution,
    plot_aspect_sentiment_bar,
    plot_avg_model_performance,
    plot_baseline_comparison,
    plot_keyword_contributions,
    plot_model_performance_heatmap,
)

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(page_title="EduSense Research Dashboard", layout="wide")

# ─────────────────────────────────────────────
# IMPORT LOCAL MODULES
# ─────────────────────────────────────────────
base_path = os.path.join(os.path.dirname(__file__), "models")


# ─────────────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────────────
@st.cache_resource
def get_models():
    return load_models()


models = get_models()

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
st.sidebar.title("EduSense")
section = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "Dataset Analytics",
        "Model Training",
        "Baseline Comparison",
        "Live Inference",
    ],
)

# ═════════════════════════════════════════════
# OVERVIEW PAGE
# ═════════════════════════════════════════════
if section == "Overview":
    st.title("EduSense Research Dashboard")
    st.markdown("""
    Aspect-Based Sentiment Analysis for Educational Feedback

    This dashboard demonstrates the complete pipeline:
    dataset creation → model training → benchmarking → inference
    """)
    st.markdown("---")

    st.subheader("Pipeline Overview")
    c1, c2, c3, c4, c5 = st.columns(5)
    for col, step in zip(
        [c1, c2, c3, c4, c5],
        [
            "Dataset Created",
            "Aspect Extraction",
            "Model Training",
            "Baseline Evaluation",
            "Inference Ready",
        ],
    ):
        col.success(step)

    st.markdown("---")
    st.subheader("Project Summary")
    st.info(
        "Educational reviews often contain multiple opinions "
        "about teaching quality, infrastructure, assessments, "
        "and course content. Traditional sentiment analysis "
        "fails to capture these aspect-specific sentiments. "
        "This project uses Aspect-Based Sentiment Analysis "
        "(ABSA) to classify sentiment separately for each "
        "educational aspect."
    )

    # ─────────────────────────────────────────
    # QUICK METRICS
    # ─────────────────────────────────────────
    dataset_df = pd.read_csv("data/dataset.csv")
    model_df = pd.read_csv("outputs/model_comparison.csv")
    baseline_df = pd.read_csv("outputs/baseline_comparison.csv")

    avg_scores = model_df.groupby("Model")["F1-Score"].mean().reset_index()
    best_model = avg_scores.loc[avg_scores["F1-Score"].idxmax()]

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Reviews", len(dataset_df))
    c2.metric("Best Model", best_model["Model"])
    c3.metric("Best Avg F1", f"{best_model['F1-Score']:.1f}%")

# ═════════════════════════════════════════════
# DATASET ANALYTICS
# ═════════════════════════════════════════════
elif section == "Dataset Analytics":
    st.title("Dataset Analytics")
    df = pd.read_csv("data/dataset.csv")

    st.info("Visualizing dataset structure and aspect distribution.")

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Reviews", len(df))
    c2.metric("Tracked Aspects", len(ASPECTS))
    c3.metric("Columns", len(df.columns))

    st.markdown("---")
    st.subheader("Aspect Distribution")
    fig = plot_aspect_distribution(df, ASPECT_DISPLAY, ASPECTS)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)

# ═════════════════════════════════════════════
# MODEL TRAINING
# ═════════════════════════════════════════════
elif section == "Model Training":
    st.title("Model Training & Evaluation")
    st.info(
        "Multiple ML models trained independently for aspect-level sentiment analysis."
    )

    model_df = pd.read_csv("outputs/model_comparison.csv")

    avg_scores = model_df.groupby("Model")["F1-Score"].mean().reset_index()
    best_model = avg_scores.loc[avg_scores["F1-Score"].idxmax()]
    st.success(
        f"Key Finding: {best_model['Model']} achieved highest average F1-score ({best_model['F1-Score']:.1f}%)."
    )

    st.subheader("F1-Score Heatmap")
    fig = plot_model_performance_heatmap(model_df)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Average Model Performance")
    fig2 = plot_avg_model_performance(model_df)
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Detailed Evaluation Table")
    st.dataframe(model_df, use_container_width=True)

# ═════════════════════════════════════════════
# BASELINE COMPARISON
# ═════════════════════════════════════════════
elif section == "Baseline Comparison":
    st.title("Baseline Comparison")
    st.info("Compare final ABSA system against generic sentiment tools.")

    baseline_df = pd.read_csv("outputs/baseline_comparison.csv")

    st.subheader("Performance Comparison")
    fig = plot_baseline_comparison(baseline_df)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Comparison Table")
    st.dataframe(baseline_df, use_container_width=True)


# ═════════════════════════════════════════════
# LIVE INFERENCE
# ═════════════════════════════════════════════

elif section == "Live Inference":
    st.title("Live Sentiment Inference")
    st.info("Run real-time aspect-based sentiment analysis on educational feedback.")

    # ─────────────────────────────────────────
    # INPUT
    # ─────────────────────────────────────────
    sample = st.selectbox(
        "Quick Examples",
        [
            "",
            "The professor explains concepts clearly but assignments are extremely difficult.",
            "Excellent syllabus but poor classroom infrastructure.",
            "Library resources are outdated and internet connectivity is terrible.",
        ],
    )

    user_input = st.text_area("Enter student feedback", value=sample, height=150)

    if st.button("Analyze"):
        if not user_input.strip():
            st.warning("Please enter feedback text.")
        else:
            results = predict_single(user_input, models)
            lower_text = user_input.lower()

            # ─────────────────────────────────────────
            # ASPECT PREDICTIONS
            # ─────────────────────────────────────────
            st.subheader("Aspect Predictions")
            pred_df = pd.DataFrame(
                [
                    {"Aspect": ASPECT_DISPLAY[a], "Prediction": s}
                    for a, s in results.items()
                ]
            )
            st.dataframe(pred_df, use_container_width=True, hide_index=True)

            # ─────────────────────────────────────────
            # FILTER MENTIONED ASPECTS
            # ─────────────────────────────────────────
            mentioned = {
                a: s
                for a, s in results.items()
                if s in ("positive", "neutral", "negative")
            }

            if mentioned:
                score_map = {"positive": 1, "neutral": 0, "negative": -1}

                # 1️⃣ Aspect sentiment scores for plotting
                aspect_scores = {}
                for aspect, sentiment in mentioned.items():
                    total_score = 0
                    for kw in ASPECT_KEYWORDS[aspect]:
                        if kw in lower_text:
                            total_score += score_map[sentiment]
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
                    color_discrete_map={
                        "positive": "green",
                        "neutral": "gray",
                        "negative": "red",
                    },
                )
                fig.update_traces(textposition="inside")
                st.plotly_chart(fig, use_container_width=True)

                # 2️⃣ Keyword contribution plot
                word_rows = []
                for aspect, sentiment in mentioned.items():
                    for kw in ASPECT_KEYWORDS[aspect]:
                        pos = lower_text.find(kw)
                        if pos != -1:
                            word_rows.append(
                                {
                                    "Word": kw,
                                    "Aspect": ASPECT_DISPLAY[aspect],
                                    "Sentiment": sentiment,
                                    "Score": score_map[sentiment],
                                    "Position": pos,
                                }
                            )

                if word_rows:
                    word_df = pd.DataFrame(word_rows).sort_values("Position")
                    fig2 = px.bar(
                        word_df,
                        x="Score",
                        y="Word",
                        orientation="h",
                        color="Sentiment",
                        text="Aspect",
                        color_discrete_map={
                            "positive": "green",
                            "neutral": "gray",
                            "negative": "red",
                        },
                        category_orders={"Word": word_df["Word"].tolist()},
                    )
                    fig2.update_traces(textposition="inside")
                    fig2.update_layout(
                        xaxis_title="Sentiment Contribution",
                        yaxis_title="Detected Keywords",
                    )
                    st.plotly_chart(fig2, use_container_width=True)

            # ─────────────────────────────────────────
            # DETECTED KEYWORDS TABLE
            # ─────────────────────────────────────────
            keyword_rows = []
            for aspect, keywords in ASPECT_KEYWORDS.items():
                matched = [kw for kw in keywords if kw in lower_text]
                if matched:
                    keyword_rows.append(
                        {
                            "Aspect": ASPECT_DISPLAY[aspect],
                            "Keywords": ", ".join(matched),
                        }
                    )

            if keyword_rows:
                st.subheader("Detected Keywords")
                keyword_df = pd.DataFrame(keyword_rows)
                st.dataframe(keyword_df, use_container_width=True, hide_index=True)
