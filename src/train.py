import os
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
# Ensure preprocessor.py is in the same folder (src)
from preprocessor import clean_text 

# --- PATH FIX: This finds the project root automatically ---
# Get the absolute path of the current script (train.py)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Go up one level to the project root (major_projekt)
BASE_DIR = os.path.dirname(SCRIPT_DIR)

DATA_PATH = os.path.join(BASE_DIR, 'data', 'raw_feedback.csv')
MODEL_DIR = os.path.join(BASE_DIR, 'models')

# Create 'models' directory if it doesn't exist
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

# Load Dataset
if not os.path.exists(DATA_PATH):
    print(f"CRITICAL ERROR: File not found at {DATA_PATH}")
    print("Make sure you have a folder named 'data' with 'raw_feedback.csv' inside it.")
    exit()

df = pd.read_csv(DATA_PATH)
# ---------------------------------------------------------

print("Cleaning data...")
df['cleaned_text'] = df['text'].apply(clean_text)

print("Training model...")
tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
X = tfidf.fit_transform(df['cleaned_text'])
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Save the artifacts using absolute paths
pickle.dump(model, open(os.path.join(MODEL_DIR, 'sentiment_model.pkl'), 'wb'))
pickle.dump(tfidf, open(os.path.join(MODEL_DIR, 'tfidf_vectorizer.pkl'), 'wb'))

print(f"Success! Model and Vectorizer saved in: {MODEL_DIR}")


