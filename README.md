
# ⚽ FootPredict — Premier League Match Prediction

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![ML](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-orange)
![Dashboard](https://img.shields.io/badge/Dashboard-Streamlit-red)
![Status](https://img.shields.io/badge/Project-Active-brightgreen)

---

# Overview

**FootPredict** is a machine learning project that predicts the probability of a **home team victory** in Premier League matches.

The system builds a full ML pipeline including:

- data collection
- feature engineering
- model training
- walk‑forward backtesting
- prediction of upcoming matches
- interactive Streamlit dashboard

The goal is to explore whether statistical features such as **Elo rating, team form, and goal statistics** can outperform random guessing.

---

# Project Architecture

```
            ┌─────────────────────┐
            │  SofaScore API      │
            └──────────┬──────────┘
                       │
                       ▼
            ┌─────────────────────┐
            │  Dataset Builder    │
            │ scripts/build_dataset│
            └──────────┬──────────┘
                       │
                       ▼
            ┌─────────────────────┐
            │ Feature Engineering │
            │   src/features/     │
            └──────────┬──────────┘
                       │
                       ▼
            ┌─────────────────────┐
            │ Model Training      │
            │ train_backtest.py   │
            └──────────┬──────────┘
                       │
                       ▼
            ┌─────────────────────┐
            │ Prediction Engine   │
            │ predict_next.py     │
            └──────────┬──────────┘
                       │
                       ▼
            ┌─────────────────────┐
            │ Streamlit Dashboard │
            │ webapp/             │
            └─────────────────────┘
```

---

# Project Structure

```
footpredict-pl
│
├── data
│   ├── raw
│   ├── processed
│   └── reports
│
├── scripts
│   ├── build_dataset.py
│   ├── train_backtest.py
│   ├── predict_next.py
│   ├── analyze_features.py
│   └── run_all.ps1
│
├── src
│   ├── api
│   ├── data
│   ├── features
│   ├── models
│   └── config.py
│
├── webapp
│   └── streamlit_app.py
│
└── README.md
```

---

# Installation

Clone repository

```
git clone https://github.com/YOUR_USERNAME/footpredict-pl.git
cd footpredict-pl
```

Install dependencies

```
pip install -e .
```

---

# Environment Variables

Create a `.env` file:

```
RAPIDAPI_KEY=YOUR_KEY
```

---

# Run the Full Pipeline

```
powershell -ExecutionPolicy Bypass -File scripts/run_all.ps1
```

This command will:

1. build dataset
2. train model
3. evaluate model
4. generate predictions
5. launch Streamlit dashboard

---

# Feature Engineering

The model uses several groups of features.

### Elo Rating

Measures team strength dynamically.

Features

- elo_home
- elo_away
- elo_diff
- elo_exp_home

### Rolling Form

Recent team performance.

- home_pts_mean
- away_pts_mean
- home_gf_mean
- away_gf_mean
- home_ga_mean
- away_ga_mean

### Goal Difference

- home_goal_diff_mean
- away_goal_diff_mean

### Rest Days

- home_rest_days
- away_rest_days

---

# Model

Algorithm

```
Logistic Regression
```

Pipeline

```
StandardScaler
→ LogisticRegression
```

Target

```
y_homewin
```

---

# Walk Forward Backtesting

Instead of random splits, the project uses **time‑based evaluation**.

Example

```
Train : Matches 1‑200
Test  : Matches 201‑240

Train : Matches 1‑240
Test  : Matches 241‑280
```

This simulates real prediction conditions.

---

# Model Benchmark

| Model | Accuracy |
|------|------|
Random baseline | 50% |
Current model | ~55% |

Metrics

| Metric | Value |
|------|------|
Accuracy | ~0.55 |
LogLoss | ~0.72 |
Brier | ~0.26 |

The model performs **better than random guessing**.

---

# Streamlit Dashboard

Launch

```
streamlit run webapp/streamlit_app.py
```

Pages

- Dashboard
- Next Matches
- Model Performance
- Match Explainer
- Model Insights

Add screenshot

```
assets/dashboard.png
```

---

# Example Prediction

Match

```
Arsenal vs Tottenham
```

Prediction

```
Home Win Probability: 0.61
Confidence: Medium
```

---

# Advanced Technical Sections

### Data Pipeline

The project separates responsibilities:

- API layer
- feature engineering layer
- modeling layer
- presentation layer

This modular design makes experimentation easier.

### Feature Validation

Before training, the pipeline checks:

- duplicates
- missing values
- dataset consistency

### Model Explainability

Feature importance is exported to

```
feature_importance.csv
```

and visualized in Streamlit.

---

# Roadmap

Future improvements

### Data

- Multi‑season dataset
- Expected goals (xG)
- Shot statistics

### Features

- home vs away form
- team streaks
- player ratings

### Models

- Gradient Boosting
- XGBoost
- LightGBM

### Betting Analysis

Compare model probabilities with bookmaker odds.

Compute

- expected value
- Kelly criterion

---

# Author

Mathieu Alassoeur

Engineering student specializing in

- Data Science
- Machine Learning
- Business Intelligence

---

# License

MIT License
