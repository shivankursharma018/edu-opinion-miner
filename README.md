# Educational Feedback Opinion Mining

## Quick Start

1. Activate virtual environment:
   ```
   venv\Scripts\activate
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"
   python run_once.py
   ```

   Install dataset from [here](https://www.kaggle.com/datasets/shivankursharma018/educational-feedback-opinions/)

3. Directory structure:
   ```
   edu-opinion-miner/
    │
    ├── app.py
    ├── run_once.py
    ├── requirements.txt
    ├── README.md
    ├── LICENSE
    ├── .gitignore
    │
    ├── data/
    ├── models/
    ├── notebooks/
    ├── src/
    └── venv/ # Virtual environment (not required in repo)
   ```

4. Train model:
   ```
   python src/train_model.py
   ```

5. Run dashboard:
   ```
   streamlit run app/streamlit_app.py
   ```

## Project Structure

- `data/` - Datasets (raw, processed, sample)
- `models/` - Trained ML models
- `src/` - Core Python modules

## Usage

### Single Review Analysis
Run the Streamlit app and paste a review in the text area.

### Batch Analysis
Upload a CSV file with a 'review_text' column.
