# Educational Feedback Opinion Mining

### What This System Does

Analyzes student feedback and identifies sentiment for 5 aspects:
- Course Content, 
- Teaching Quality, 
- Assessment, 
- Resources, 
- Infrastructure

**Input:** "Great course content but difficult exams and poor lab equipment"  

**Output:**
- Course Content → 🟢 Positive
- Assessment     → 🔴 Negative
- Infrastructure → 🔴 Negative

## Quick Start

1. Activate virtual environment:
   ```
   python -m venv venv
   venv\Scripts\activate
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"
   ```

   Install dataset from [here](https://www.kaggle.com/datasets/shivankursharma018/educational-feedback-opinions/)

3. Run scripts:
   ```
   python run_all.py
   ```

4. Run dashboard:
   ```
   streamlit run eduDashboard.py
   ```

## Project Structure

- `data/` - Datasets (csv)
- `models/` - Trained ML models
- `scripts/` - Core Python modules/scripts

## Usage

### Single Review Analysis
Run the Streamlit app and paste a review in the text area.

### Some sample reviews

> Internet connection in campus is very slow and unreliable. the teaching methods are nice. teacher is bad. course material is insufficient and the exams are hard to pass. the lab equipment is latest and up to date. 

> The syllabus is very clear and well-structured, and the examples are helpful. However, some topics are confusing, outdated, and poorly explained. The professor is engaging and explains well, but sometimes they rush through lectures, and their feedback is inconsistent. Assignments are fair, but the tests are extremely difficult, and some quizzes are unfair, too. The library is well-stocked, and the lab equipment works fine, but the computers are slow, the internet is terrible, and the classroom is noisy.

> The syllabus is very clear and includes helpful examples, and the course content is engaging. However, some topics are confusing, outdated, and poorly explained. The professor explains concepts clearly and encourages participation, but sometimes lectures are rushed, feedback is inconsistent, and explanations are unclear. Assignments are fair and quizzes are well-structured, but the tests are extremely difficult, some homework is confusing, and grading is inconsistent. The library has great books and the lab equipment works perfectly, yet the computers are slow, the internet is terrible, and some software is outdated. Classrooms are comfortable and seating is adequate, but the environment is noisy, the projector often fails, and lighting is poor.

> The syllabus is detailed and includes interesting examples, and the course material is well-organized. However, some sections are outdated, confusing, and poorly explained. The professor is enthusiastic and answers questions clearly, but sometimes they rush through the lectures, provide inconsistent feedback, and skip important topics. Assignments are manageable and quizzes are fair, but tests are too difficult, homework instructions are unclear, and grading is inconsistent. The library has excellent resources and the lab equipment functions well, yet computers are slow, the internet connection is terrible, and some lab software is outdated. Classrooms are spacious and seating is comfortable, but the environment is noisy, lighting is poor, and the projectors often malfunction.

> The syllabus is thorough and covers essential topics, and the course content is engaging. However, some modules are confusing, outdated, and poorly structured. The professor explains concepts clearly and interacts well with students, but lectures are rushed, feedback is inconsistent, and explanations on some topics are unclear. Assignments are reasonable and quizzes are fair, but tests are overly difficult, homework is confusing, and grading lacks transparency. The library is well-stocked and lab equipment works fine, but the computers are slow, the internet is unreliable, and some lab software is outdated. Classrooms are comfortable and seating is adequate, yet the environment is noisy, projectors fail frequently, and the lighting is insufficient.

### Batch Analysis
Upload a CSV file with a 'review_text' column.
