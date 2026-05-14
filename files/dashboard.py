import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

import os
import sys

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="EduSense Research Dashboard",
    layout="wide"
)

# ─────────────────────────────────────────────
# IMPORT LOCAL MODULES
# ─────────────────────────────────────────────

sys.path.insert(0, os.path.dirname(__file__))

from predictor import (
    load_models,
    predict_single,
    ASPECTS,
    ASPECT_DISPLAY,
    ASPECT_KEYWORDS
)

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
        "Live Inference"
    ]
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

    # ─────────────────────────────────────────
    # PIPELINE FLOW
    # ─────────────────────────────────────────

    st.subheader("Pipeline Overview")

    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        st.success("Dataset Created")

    with c2:
        st.success("Aspect Extraction")

    with c3:
        st.success("Model Training")

    with c4:
        st.success("Baseline Evaluation")

    with c5:
        st.success("Inference Ready")

    st.markdown("---")

    # ─────────────────────────────────────────
    # PROJECT SUMMARY
    # ─────────────────────────────────────────

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

    model_df = pd.read_csv(
        "outputs/model_comparison.csv"
    )

    baseline_df = pd.read_csv(
        "outputs/baseline_comparison.csv"
    )

    avg_scores = (
        model_df
        .groupby("Model")["F1-Score"]
        .mean()
        .reset_index()
    )

    best_model = avg_scores.loc[
        avg_scores["F1-Score"].idxmax()
    ]

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(
            "Total Reviews",
            len(dataset_df)
        )

    with c2:
        st.metric(
            "Best Model",
            best_model["Model"]
        )

    with c3:
        st.metric(
            "Best Avg F1",
            f"{best_model['F1-Score']:.1f}%"
        )

    st.markdown("---")

    st.subheader("Execution Flow")

    st.markdown("""
    1. Dataset is generated and labeled by aspect  
    2. Aspect keywords are extracted from reviews  
    3. Multiple ML models are trained independently  
    4. Models are benchmarked using F1-score  
    5. Final system is compared against TextBlob and VADER  
    6. Users can perform real-time inference on reviews  
    """)

# ═════════════════════════════════════════════
# DATASET ANALYTICS
# ═════════════════════════════════════════════

elif section == "Dataset Analytics":

    st.title("Dataset Analytics")

    df = pd.read_csv("data/dataset.csv")

    st.info(
        "This section visualizes the structure and distribution "
        "of the educational feedback dataset."
    )

    # ─────────────────────────────────────────
    # METRICS
    # ─────────────────────────────────────────

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric("Total Reviews", len(df))

    with c2:
        st.metric("Tracked Aspects", len(ASPECTS))

    with c3:
        st.metric("Columns", len(df.columns))

    st.markdown("---")

    # ─────────────────────────────────────────
    # ASPECT DISTRIBUTION
    # ─────────────────────────────────────────

    st.subheader("Aspect Distribution")

    aspect_counts = {}

    for aspect in ASPECTS:

        count = (
            df[aspect]
            .replace("none", np.nan)
            .count()
        )

        aspect_counts[
            ASPECT_DISPLAY[aspect]
        ] = count

    aspect_df = pd.DataFrame({
        "Aspect": aspect_counts.keys(),
        "Count": aspect_counts.values()
    })

    fig = px.bar(
        aspect_df,
        x="Aspect",
        y="Count",
        text="Count"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.caption(
        "Observation: Some educational aspects appear "
        "more frequently than others, creating "
        "class imbalance challenges."
    )

    st.markdown("---")

    # ─────────────────────────────────────────
    # DATASET PREVIEW
    # ─────────────────────────────────────────

    st.subheader("Dataset Preview")

    st.dataframe(
        df.head(10),
        use_container_width=True
    )

    st.success(
        "Next Step → Reviews are passed through "
        "aspect extraction and model training."
    )

# ═════════════════════════════════════════════
# MODEL TRAINING
# ═════════════════════════════════════════════

elif section == "Model Training":

    st.title("Model Training & Evaluation")

    st.info(
        "Multiple machine learning models are trained "
        "independently for aspect-level sentiment analysis."
    )

    model_df = pd.read_csv(
        "outputs/model_comparison.csv"
    )

    # ─────────────────────────────────────────
    # MAIN INSIGHT
    # ─────────────────────────────────────────

    avg_scores = (
        model_df
        .groupby("Model")["F1-Score"]
        .mean()
        .reset_index()
    )

    best_model = avg_scores.loc[
        avg_scores["F1-Score"].idxmax()
    ]

    st.success(
        f"Key Finding: {best_model['Model']} "
        f"achieved the highest average F1-score "
        f"({best_model['F1-Score']:.1f}%)."
    )

    # ─────────────────────────────────────────
    # HEATMAP
    # ─────────────────────────────────────────

    st.subheader("F1-Score Heatmap")

    pivot_df = model_df.pivot(
        index="Aspect",
        columns="Model",
        values="F1-Score"
    )

    fig = px.imshow(
        pivot_df,
        text_auto=True,
        aspect="auto"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.caption(
        "Observation: Model performance varies "
        "significantly across educational aspects."
    )

    st.markdown("---")

    # ─────────────────────────────────────────
    # AVERAGE PERFORMANCE
    # ─────────────────────────────────────────

    st.subheader("Average Model Performance")

    fig2 = px.bar(
        avg_scores,
        x="Model",
        y="F1-Score",
        text="F1-Score"
    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )

    st.caption(
        "Observation: SVM generalizes better "
        "across aspect categories."
    )

    st.markdown("---")

    st.subheader("Detailed Evaluation Table")

    st.dataframe(
        model_df,
        use_container_width=True
    )

    st.success(
        "Next Step → Compare the trained system "
        "against generic sentiment baselines."
    )


# ═════════════════════════════════════════════
# BASELINE COMPARISON
# ═════════════════════════════════════════════

elif section == "Baseline Comparison":

    st.title("Baseline Comparison")

    st.info(
        "The final ABSA system is compared against "
        "generic sentiment analysis tools."
    )

    baseline_df = pd.read_csv(
        "outputs/baseline_comparison.csv"
    )

    st.subheader("Performance Comparison")

    fig = px.bar(
        baseline_df,
        x="Method",
        y="Accuracy",
        text="Accuracy"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.caption(
        "Observation: Domain-trained aspect-aware "
        "classification significantly outperforms "
        "generic sentiment systems."
    )

    st.markdown("---")

    st.subheader("Comparison Table")

    st.dataframe(
        baseline_df,
        use_container_width=True
    )

    improvement = 80.00 - 60.67

    st.success(
        f"Our ABSA system improved performance "
        f"over TextBlob by {improvement:.1f}%."
    )

    st.success(
        "Next Step → Perform live inference "
        "using the trained model."
    )

# ═════════════════════════════════════════════
# LIVE INFERENCE
# ═════════════════════════════════════════════

elif section == "Live Inference":

    st.title("Live Sentiment Inference")

    st.info(
        "Run real-time aspect-based sentiment "
        "analysis on educational feedback."
    )

    sample = st.selectbox(
        "Quick Examples",
        [
            "",
            "The professor explains concepts clearly but assignments are extremely difficult.",
            "Excellent syllabus but poor classroom infrastructure.",
            "Library resources are outdated and internet connectivity is terrible."
        ]
    )

    user_input = st.text_area(
        "Enter student feedback",
        value=sample,
        height=150
    )

    analyze_btn = st.button("Analyze")

    if analyze_btn:

        if not user_input.strip():

            st.warning("Please enter feedback text.")

        else:

            results = predict_single(
                user_input,
                models
            )

            st.subheader("Aspect Predictions")

            rows = []

            for aspect, sentiment in results.items():

                rows.append({
                    "Aspect": ASPECT_DISPLAY[aspect],
                    "Prediction": sentiment
                })

            pred_df = pd.DataFrame(rows)

            st.dataframe(
                pred_df,
                use_container_width=True,
                hide_index=True
            )

            mentioned = {
                k: v for k, v in results.items()
                if v != "not mentioned"
            }

            if mentioned:

                score_map = {
                    "positive": 1,
                    "neutral": 0,
                    "negative": -1
                }

                chart_df = pd.DataFrame({
                    "Aspect": [
                        ASPECT_DISPLAY[a]
                        for a in mentioned
                    ],
                    "Score": [
                        score_map[s]
                        for s in mentioned.values()
                    ],
                    "Sentiment": list(
                        mentioned.values()
                    )
                })

                fig = px.bar(
                    chart_df,
                    x="Aspect",
                    y="Score",
                    color="Sentiment",
                    text="Sentiment"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

            st.subheader("Detected Keywords")

            lower_text = user_input.lower()

            keyword_rows = []

            for aspect, keywords in ASPECT_KEYWORDS.items():

                matched = [
                    kw for kw in keywords
                    if kw in lower_text
                ]

                if matched:

                    keyword_rows.append({
                        "Aspect": ASPECT_DISPLAY[aspect],
                        "Keywords": ", ".join(matched)
                    })

            if keyword_rows:

                keyword_df = pd.DataFrame(keyword_rows)

                st.dataframe(
                    keyword_df,
                    use_container_width=True,
                    hide_index=True
                )

