"""
STEP 5: STREAMLIT DASHBOARD
============================
Run with: streamlit run dashboard.py

Features:
- Single review analysis
- Bulk CSV upload + analysis
- Charts and export
"""

import streamlit as st
import pandas as pd
import os
import sys
import io

# ── Page config ────────────────────────────────────────────
st.set_page_config(
    page_title="EduSense - Educational Feedback Analyzer",
    page_icon="🎓",
    layout="wide"
)

# Add project root to path so we can import predictor
sys.path.insert(0, os.path.dirname(__file__))
from predictor import (
    load_models, predict_single, predict_bulk,
    summarize_bulk, ASPECTS, ASPECT_DISPLAY
)

# ── Load models (cached so it only loads once) ──────────────
@st.cache_resource
def get_models():
    return load_models()

models = get_models()

# ── Colour coding ────────────────────────────────────────────
SENTIMENT_COLOR = {
    "positive": "🟢",
    "negative": "🔴",
    "neutral": "🟡",
    "not mentioned": "⚪"
}

SENTIMENT_BG = {
    "positive": "#d4edda",
    "negative": "#f8d7da",
    "neutral": "#fff3cd",
    "not mentioned": "#f8f9fa"
}

# ── Title ────────────────────────────────────────────────────
st.title("🎓 EduSense — Educational Feedback Analyzer")
st.markdown("**Aspect-Based Sentiment Analysis** | Analyzes feedback across 5 key aspects")
st.divider()

# ── Tabs ─────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📝 Single Review", "📊 Bulk Analysis", "ℹ️ How It Works"])

# ════════════════════════════════════════════════════════════
# TAB 1: SINGLE REVIEW
# ════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Analyze a Single Feedback")
    
    # Example buttons
    st.markdown("**Quick examples:**")
    col_ex1, col_ex2, col_ex3 = st.columns(3)
    
    example_texts = {
        "Example 1": "Great course content but the exams are too tough and lab equipment is outdated.",
        "Example 2": "Professor explains very well but library resources are insufficient.",
        "Example 3": "The syllabus is outdated and classroom infrastructure needs improvement.",
    }
    
    if col_ex1.button("Example 1"):
        st.session_state['input_text'] = example_texts["Example 1"]
    if col_ex2.button("Example 2"):
        st.session_state['input_text'] = example_texts["Example 2"]
    if col_ex3.button("Example 3"):
        st.session_state['input_text'] = example_texts["Example 3"]
    
    user_input = st.text_area(
        "Enter student feedback:",
        value=st.session_state.get('input_text', ''),
        height=100,
        placeholder="e.g. Great course content but difficult exams and poor lab equipment..."
    )
    
    analyze_btn = st.button("🔍 Analyze", type="primary", use_container_width=True)
    
    if analyze_btn and user_input.strip():
        with st.spinner("Analyzing..."):
            results = predict_single(user_input, models)
        
        st.subheader("Results")
        
        cols = st.columns(5)
        for i, aspect in enumerate(ASPECTS):
            sentiment = results[aspect]
            with cols[i]:
                st.markdown(f"""
                <div style='background:{SENTIMENT_BG[sentiment]};padding:16px;
                            border-radius:10px;text-align:center;'>
                    <div style='font-size:24px'>{SENTIMENT_COLOR[sentiment]}</div>
                    <div style='font-weight:bold;font-size:13px'>{ASPECT_DISPLAY[aspect]}</div>
                    <div style='color:#555;font-size:13px;margin-top:4px'>{sentiment.upper()}</div>
                </div>
                """, unsafe_allow_html=True)
        
        # Bar chart of detected sentiments
        mentioned = {k: v for k, v in results.items() if v != "not mentioned"}
        if mentioned:
            st.markdown("\n**Detected Sentiments:**")
            chart_data = pd.DataFrame({
                "Aspect": [ASPECT_DISPLAY[a] for a in mentioned],
                "Sentiment": list(mentioned.values()),
                "Score": [1 if s == "positive" else (-1 if s == "negative" else 0)
                          for s in mentioned.values()]
            })
            st.bar_chart(chart_data.set_index("Aspect")["Score"],
                        color="#4A90D9", height=250)
            st.caption("Score: 1 = Positive, 0 = Neutral, -1 = Negative")
    
    elif analyze_btn:
        st.warning("Please enter some feedback text first.")


# ════════════════════════════════════════════════════════════
# TAB 2: BULK ANALYSIS
# ════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Bulk Analysis — Upload CSV")
    
    st.markdown("""
    Upload a CSV file with a column named **`text`** containing student feedback.
    The system will analyze all rows and show aggregate statistics.
    """)
    
    # Download sample CSV
    sample_df = pd.DataFrame({
        "text": [
            "Great course content but difficult exams.",
            "Professor is excellent and lab facilities are good.",
            "Syllabus is outdated and library lacks books.",
            "Teaching quality is poor but assignments are fair.",
        ]
    })
    sample_csv = sample_df.to_csv(index=False)
    st.download_button("⬇️ Download Sample CSV", sample_csv,
                      "sample_feedback.csv", "text/csv")
    
    uploaded = st.file_uploader("Upload feedback CSV", type=["csv"])
    
    if uploaded:
        df_upload = pd.read_csv(uploaded)
        
        if "text" not in df_upload.columns:
            st.error("CSV must have a column named 'text'")
        else:
            st.success(f"✅ Loaded {len(df_upload)} reviews")
            
            with st.spinner(f"Analyzing {len(df_upload)} reviews..."):
                texts = df_upload['text'].fillna('').tolist()
                all_results = predict_bulk(texts, models)
                summary = summarize_bulk(all_results)
            
            # ── Summary metrics ──────────────────────────────
            st.subheader("📊 Aggregate Results")
            
            cols = st.columns(5)
            for i, aspect in enumerate(ASPECTS):
                total = summary[aspect]["total"]
                pos = summary[aspect]["positive"]
                neg = summary[aspect]["negative"]
                
                if total > 0:
                    pos_pct = pos / total * 100
                    neg_pct = neg / total * 100
                    health = "🟢" if pos_pct > 60 else ("🔴" if neg_pct > 60 else "🟡")
                else:
                    pos_pct = neg_pct = 0
                    health = "⚪"
                
                with cols[i]:
                    st.metric(
                        label=ASPECT_DISPLAY[aspect],
                        value=f"{health} {pos_pct:.0f}% Positive",
                        delta=f"{neg_pct:.0f}% Negative"
                    )
            
            # ── Stacked bar chart ────────────────────────────
            st.subheader("Sentiment Distribution by Aspect")
            chart_data = {
                "Aspect": [],
                "Positive": [],
                "Negative": [],
                "Neutral": []
            }
            for aspect in ASPECTS:
                total = summary[aspect]["total"] or 1
                chart_data["Aspect"].append(ASPECT_DISPLAY[aspect])
                chart_data["Positive"].append(round(summary[aspect]["positive"] / total * 100))
                chart_data["Negative"].append(round(summary[aspect]["negative"] / total * 100))
                chart_data["Neutral"].append(round(summary[aspect]["neutral"] / total * 100))
            
            chart_df = pd.DataFrame(chart_data).set_index("Aspect")
            st.bar_chart(chart_df, color=["#28a745", "#dc3545", "#ffc107"])
            
            # ── Row-level results ────────────────────────────
            st.subheader("Row-Level Results")
            result_rows = []
            for idx, (text, res) in enumerate(zip(texts, all_results)):
                row = {"Review #": idx + 1, "Text": text[:80] + "..." if len(text) > 80 else text}
                for aspect in ASPECTS:
                    row[aspect.replace("_", " ").title()] = SENTIMENT_COLOR.get(res[aspect], "") + " " + res[aspect]
                result_rows.append(row)
            
            result_df = pd.DataFrame(result_rows)
            st.dataframe(result_df, use_container_width=True)
            
            # Export
            csv_out = result_df.to_csv(index=False)
            st.download_button("⬇️ Export Results as CSV", csv_out,
                              "absa_results.csv", "text/csv",
                              type="primary")


# ════════════════════════════════════════════════════════════
# TAB 3: HOW IT WORKS
# ════════════════════════════════════════════════════════════
with tab3:
    st.subheader("System Architecture")
    
    st.markdown("""
    ### How EduSense Works
    
    ```
    Student Feedback Text
           │
           ▼
    ┌─────────────────────┐
    │  Text Preprocessing  │  → lowercase, remove noise, lemmatize
    └─────────┬───────────┘
              │
              ▼
    ┌─────────────────────┐
    │  Aspect Detection   │  → keyword matching for 5 aspects
    └─────────┬───────────┘
              │
              ▼  (for each detected aspect)
    ┌─────────────────────┐
    │  TF-IDF Features    │  → convert text to numbers
    └─────────┬───────────┘
              │
              ▼
    ┌─────────────────────┐
    │  ML Classifier      │  → Naive Bayes / SVM / Random Forest
    └─────────┬───────────┘
              │
              ▼
    Positive / Negative / Neutral  (per aspect)
    ```
    
    ### 5 Aspects Analyzed
    | Aspect | What it covers |
    |--------|---------------|
    | 📚 Course Content | Syllabus, curriculum, topics, relevance |
    | 👨‍🏫 Teaching Quality | Professors, teaching style, clarity |
    | 📝 Assessment | Exams, assignments, grading, deadlines |
    | 📖 Resources | Library, books, study materials, e-resources |
    | 🏫 Infrastructure | Labs, classrooms, WiFi, equipment |
    
    ### Why Better Than TextBlob / VADER?
    | Feature | TextBlob / VADER | Our System |
    |---------|-----------------|------------|
    | Aspect-aware | ❌ No | ✅ Yes |
    | Domain-trained | ❌ General | ✅ Education |
    | Per-aspect output | ❌ Single score | ✅ 5 scores |
    | Actionable insights | ❌ Limited | ✅ Yes |
    """)
