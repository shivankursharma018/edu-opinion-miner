"""
STEP 1: DATA PREPARATION
========================
Since EduRABSA dataset may not be freely available,
we create a realistic synthetic dataset of student feedback.
Each review is labeled for 5 aspects.
"""

import pandas as pd
import numpy as np
import re
import json
import os
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK data
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)

# ─────────────────────────────────────────────
# SYNTHETIC DATASET
# ─────────────────────────────────────────────
# Format: (text, course_content, teaching_quality, assessment, resources, infrastructure)
# Labels: positive / negative / neutral / none (aspect not mentioned)

RAW_DATA = [
    # course_content positive
    ("The syllabus is very well structured and covers all important topics.", "positive", "none", "none", "none", "none"),
    ("Course material is up to date and highly relevant to industry.", "positive", "none", "none", "none", "none"),
    ("Topics are explained in great depth with real-world examples.", "positive", "none", "none", "none", "none"),
    ("The curriculum is excellent, very comprehensive.", "positive", "none", "none", "none", "none"),
    ("Course content is amazing, covers both theory and practice.", "positive", "none", "none", "none", "none"),
    # course_content negative
    ("The syllabus is outdated and not useful for placements.", "negative", "none", "none", "none", "none"),
    ("Topics covered in class are irrelevant and poorly chosen.", "negative", "none", "none", "none", "none"),
    ("The course content is too basic and lacks depth.", "negative", "none", "none", "none", "none"),
    ("Curriculum does not match current industry requirements.", "negative", "none", "none", "none", "none"),
    ("The subjects taught are boring and of no practical use.", "negative", "none", "none", "none", "none"),

    # teaching_quality positive
    ("The professor explains concepts very clearly and patiently.", "none", "positive", "none", "none", "none"),
    ("Faculty is very knowledgeable and always willing to help.", "none", "positive", "none", "none", "none"),
    ("Teaching is excellent, the instructor makes every class engaging.", "none", "positive", "none", "none", "none"),
    ("The teacher is very approachable and gives great feedback.", "none", "positive", "none", "none", "none"),
    ("Lectures are interactive and the professor is very supportive.", "none", "positive", "none", "none", "none"),
    # teaching_quality negative
    ("The professor rushes through topics without explanation.", "none", "negative", "none", "none", "none"),
    ("Faculty is unapproachable and does not respond to queries.", "none", "negative", "none", "none", "none"),
    ("Teaching quality is very poor, instructor just reads slides.", "none", "negative", "none", "none", "none"),
    ("The teacher is not prepared for class and wastes our time.", "none", "negative", "none", "none", "none"),
    ("Lectures are boring and the professor shows no interest in students.", "none", "negative", "none", "none", "none"),

    # assessment positive
    ("Exams are fair and test actual understanding of concepts.", "none", "none", "positive", "none", "none"),
    ("Assignments are challenging but very useful for learning.", "none", "none", "positive", "none", "none"),
    ("The grading system is transparent and fair.", "none", "none", "positive", "none", "none"),
    ("Quizzes help reinforce what we learn in class.", "none", "none", "positive", "none", "none"),
    ("The evaluation is balanced between theory and practical work.", "none", "none", "positive", "none", "none"),
    # assessment negative
    ("Exams are extremely difficult and not aligned with what is taught.", "none", "none", "negative", "none", "none"),
    ("Too many assignments with unrealistic deadlines.", "none", "none", "negative", "none", "none"),
    ("Grading is very strict and subjective.", "none", "none", "negative", "none", "none"),
    ("The exam paper was out of syllabus, very unfair.", "none", "none", "negative", "none", "none"),
    ("Assessment methods are outdated and do not test real skills.", "none", "none", "negative", "none", "none"),

    # resources positive
    ("The library has excellent books and journals for reference.", "none", "none", "none", "positive", "none"),
    ("Online study materials provided are very helpful.", "none", "none", "none", "positive", "none"),
    ("Reference materials and e-resources are well organized.", "none", "none", "none", "positive", "none"),
    ("The college provides access to great research databases.", "none", "none", "none", "positive", "none"),
    ("Study materials and handouts are very comprehensive.", "none", "none", "none", "positive", "none"),
    # resources negative
    ("Library lacks important textbooks and reference material.", "none", "none", "none", "negative", "none"),
    ("Online resources are outdated and poorly maintained.", "none", "none", "none", "negative", "none"),
    ("No proper study material is provided by teachers.", "none", "none", "none", "negative", "none"),
    ("The college does not have sufficient books or journals.", "none", "none", "none", "negative", "none"),
    ("Digital resources are inaccessible and poorly organized.", "none", "none", "none", "negative", "none"),

    # infrastructure positive
    ("The classrooms are spacious and well equipped with projectors.", "none", "none", "none", "none", "positive"),
    ("Labs have modern computers and fast internet connection.", "none", "none", "none", "none", "positive"),
    ("The campus infrastructure is excellent and well maintained.", "none", "none", "none", "none", "positive"),
    ("Facilities like wifi and smart boards are very good.", "none", "none", "none", "none", "positive"),
    ("The computer lab is well maintained with latest software.", "none", "none", "none", "none", "positive"),
    # infrastructure negative
    ("Lab equipment is old and breaks down frequently.", "none", "none", "none", "none", "negative"),
    ("Classrooms are too small and overcrowded.", "none", "none", "none", "none", "negative"),
    ("Internet connection in campus is very slow and unreliable.", "none", "none", "none", "none", "negative"),
    ("The infrastructure is poorly maintained and outdated.", "none", "none", "none", "none", "negative"),
    ("Computer lab has outdated machines that crash often.", "none", "none", "none", "none", "negative"),

    # MULTI-ASPECT REVIEWS (realistic feedback)
    ("Great course content but the exams are too tough.", "positive", "none", "negative", "none", "none"),
    ("Teaching is excellent but lab equipment is very poor.", "none", "positive", "none", "none", "negative"),
    ("Good study material but the grading is unfair.", "none", "none", "negative", "positive", "none"),
    ("The professor is brilliant but the syllabus is outdated.", "negative", "positive", "none", "none", "none"),
    ("Exams are fair and facilities are great.", "none", "none", "positive", "none", "positive"),
    ("Course is well designed and faculty is supportive.", "positive", "positive", "none", "none", "none"),
    ("Poor infrastructure and ineffective teaching quality.", "none", "negative", "none", "none", "negative"),
    ("Assignments are helpful but internet is too slow on campus.", "none", "none", "positive", "none", "negative"),
    ("Library resources are good but syllabus needs update.", "negative", "none", "none", "positive", "none"),
    ("The teacher is good but exams are very difficult and unfair.", "none", "positive", "negative", "none", "none"),
    ("Excellent course content, good faculty, but poor labs.", "positive", "positive", "none", "none", "negative"),
    ("Grading is strict and study material is insufficient.", "none", "none", "negative", "negative", "none"),
    ("Overall the course is decent, nothing extraordinary.", "neutral", "neutral", "none", "none", "none"),
    ("Infrastructure is okay but could be improved.", "none", "none", "none", "none", "neutral"),
    ("Assignments are okay, not too hard not too easy.", "none", "none", "neutral", "none", "none"),
    ("The syllabus is good but teaching needs improvement.", "positive", "negative", "none", "none", "none"),
    ("Resources are average, labs need better equipment.", "none", "none", "none", "neutral", "negative"),
    ("Faculty is great and the course content is also excellent.", "positive", "positive", "none", "none", "none"),
    ("Lab facilities are good but the syllabus is very outdated.", "negative", "none", "none", "none", "positive"),
    ("Good professors but too many assignments with short deadlines.", "none", "positive", "negative", "none", "none"),
]

COLUMNS = ["text", "course_content", "teaching_quality", "assessment", "resources", "infrastructure"]
ASPECTS = ["course_content", "teaching_quality", "assessment", "resources", "infrastructure"]


# ─────────────────────────────────────────────
# TEXT PREPROCESSING
# ─────────────────────────────────────────────
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    """Remove noise, lowercase, lemmatize."""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)          # keep only letters
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words]
    return " ".join(tokens)


def prepare_dataset():
    df = pd.DataFrame(RAW_DATA, columns=COLUMNS)
    df['clean_text'] = df['text'].apply(clean_text)
    print(f"✅ Dataset created: {len(df)} reviews")
    print(f"   Aspects tracked: {ASPECTS}")
    print("\nSample rows:")
    print(df[['text', 'course_content', 'teaching_quality']].head(3).to_string())
    return df


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    df = prepare_dataset()
    df.to_csv("data/dataset.csv", index=False)
    print("\n✅ Saved to data/dataset.csv")
