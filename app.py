import streamlit as st
import pickle
import os
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from src.preprocessor import clean_text

# --- PAGE CONFIG ---
st.set_page_config(page_title="EduFeedback AI", layout="wide")

# --- PATHS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'sentiment_model.pkl')
VECTORIZER_PATH = os.path.join(BASE_DIR, 'models', 'tfidf_vectorizer.pkl')

# --- LOAD MODEL ---
@st.cache_resource
def load_artifacts():
    if not os.path.exists(MODEL_PATH):
        return None, None
    model = pickle.load(open(MODEL_PATH, 'rb'))
    tfidf = pickle.load(open(VECTORIZER_PATH, 'rb'))
    return model, tfidf

model, tfidf = load_artifacts()

# --- UI ---
st.title("🎓 Educational Feedback Opinion Mining")
st.markdown("Analyzing student feedback using NLP & Random Forest (Group 22)")

if model is None:
    st.error("Model files not found! Please run 'python src/train.py' first.")
else:
    # Sidebar for project info
    st.sidebar.header("Project Details")
    st.sidebar.info("This system uses TF-IDF and Random Forest to classify feedback into Positive, Negative, or Neutral categories.")

    # Main Input
    user_input = st.text_area("Enter student feedback here:", placeholder="e.g., The professor explains concepts clearly but the lab is too small.")

    if st.button("Analyze Sentiment"):
        if user_input:
            # 1. Preprocess
            cleaned = clean_text(user_input)
            # 2. Vectorize
            vec = tfidf.transform([cleaned])
            # 3. Predict
            prediction = model.predict(vec)[0]
            
            # Display Result
            st.divider()
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Result")
                if prediction == "Positive":
                    st.success(f"Sentiment: **{prediction}**")
                elif prediction == "Negative":
                    st.error(f"Sentiment: **{prediction}**")
                else:
                    st.warning(f"Sentiment: **{prediction}**")
            
            with col2:
                st.subheader("Processed Text")
                st.write(f"*{cleaned}*")
                
            # Bonus: WordCloud (For Review-4 Visuals)
            st.divider()
            st.subheader("Word Importance Visualization")
            wordcloud = WordCloud(background_color="white", width=800, height=400).generate(cleaned if cleaned else "Empty")
            fig, ax = plt.subplots()
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis("off")
            st.pyplot(fig)
        else:
            st.warning("Please enter some text.")

# Terminal MVP Fallback (Hidden in code)
# To run via terminal only: prediction = model.predict(tfidf.transform([clean_text("text")]))
