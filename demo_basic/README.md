# Mini Project Demo

This is a **mini-project demonstration** of the edu-opinion-miner, aspect-based sentiment analysis system.
It allows you to train models, preprocess data, and run a real-time Streamlit dashboard for educational feedback analysis.

## Features

* Train aspect-based sentiment analysis models (`train.py`)
* Preprocess educational feedback dataset (`preprocessor.py`)
* Run a live inference dashboard (`app.py`) using Streamlit

## Running the Mini Project

1. **Preprocess the dataset**

```bash
python preprocessor.py
```

2. **Train models**

```bash
python train.py
```

3. **Run the Streamlit dashboard**

```bash
streamlit run app.py
```

Open the URL shown in your terminal (usually `http://localhost:8501`) to interact with the dashboard.

## Notes

* Ensure the `data/` folder contains your dataset CSV.
* Models will be saved to `models/` after training.
* Quick example feedback is available in the Live Inference section of the dashboard.
