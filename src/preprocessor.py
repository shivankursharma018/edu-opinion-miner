import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# We remove negation words from stopwords because they change sentiment
negation_words = {'not', 'no', 'never', 'neither', 'nor', 'but'}
stop_words = stop_words - negation_words

def clean_text(text):
    # 1. Remove special characters and numbers (Handling 'noise' mentioned in PPT)
    text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
    
    # 2. Tokenization and Stopword removal
    words = text.split()
    cleaned_words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    
    return " ".join(cleaned_words)

