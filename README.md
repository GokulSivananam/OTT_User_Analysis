# OTT User Retention Analysis
**Project ID:** DS_Day01_41 | **Domain:** Streaming and OTT

---

## Problem Statement

An OTT (Over-The-Top) streaming platform has noticed that many users stop watching after a few weeks. The platform wants to understand **why users churn** and **how to retain them** using data-driven strategies.

---

## Objective

1. Clean and validate the user dataset.
2. Perform exploratory data analysis (EDA).
3. Identify different viewing behaviour patterns.
4. Calculate average watch time by customer segment.
5. Examine whether content diversity influences engagement.
6. Identify early signs of declining engagement.
7. Compare subscription plans.
8. Build and evaluate a Random Forest model for Retention_Status prediction.
9. Generate 5–7 data-driven insights.
10. Provide a practical action plan for improving OTT engagement.

---

## Dataset Description

**File:** `data/OTT_User_Data.csv`

| Column | Description |
|---|---|
| User_ID | Unique identifier for each user |
| Viewing_History | Viewing frequency category (Low / Medium / High) |
| Session_Duration | Duration of each viewing session in minutes |
| Content_Genre | Genre of content watched in that session |
| Subscription_Plan | User's subscription tier (Basic / Standard / Premium) |
| Inactivity_Period | Number of days the user has been inactive |
| Retention_Status | Whether the user is Retained or Churned |

- **Total Records:** 763 (multiple rows per user)
- **Unique Users:** 250
- **Missing Values:** 7 in Session_Duration, 7 in Content_Genre
- **Duplicate Rows:** 7

---

## Technologies Used

| Library | Purpose |
|---|---|
| Python 3 | Core programming language |
| pandas | Data loading, cleaning, manipulation |
| numpy | Numerical operations |
| matplotlib | Data visualizations |
| seaborn | Statistical visualizations |
| scikit-learn | Machine learning (Random Forest, preprocessing) |
| joblib | Model serialization |
| openpyxl | Excel compatibility |

---

## Project Structure

```
OTT_Retention_Analysis/
│
├── data/
│   └── OTT_User_Data.csv          ← Raw dataset
│
├── src/
│   ├── data_cleaning.py           ← Phase 1 & 2: Inspection + Cleaning
│   ├── eda.py                     ← Phase 3 & 4: EDA + Behaviour Patterns
│   ├── feature_engineering.py     ← Phase 5–8: Segments, Diversity, Engagement
│   ├── model.py                   ← Phase 9–13: Visualizations + RF Model
│   └── main.py                    ← Orchestrator: runs all phases
│
├── outputs/
│   ├── watch_time_segment.png     ← Avg session by subscription plan
│   ├── genre_engagement.png       ← Avg session by content genre
│   ├── inactivity_retention.png   ← Inactivity distribution by retention
│   ├── content_diversity.png      ← Content diversity vs session duration
│   ├── confusion_matrix.png       ← Random Forest confusion matrix
│   └── feature_importance.png     ← Random Forest feature importances
│
├── model/
│   ├── random_forest_model.pkl    ← Trained pipeline (preprocessor + RF)
│   └── label_encoder.pkl          ← LabelEncoder for Retention_Status
│
├── requirements.txt               ← Python dependencies
└── README.md                      ← Project documentation
```

---

## Data Cleaning Process

1. **Strip whitespace** from column names and string values.
2. **Remove duplicate rows** — 7 duplicates removed.
3. **Convert to numeric** — Session_Duration and Inactivity_Period verified.
4. **Handle missing numerical values** — Session_Duration: 7 NaN → filled with **median**.
5. **Handle missing categorical values** — Content_Genre: 7 NaN → filled with **mode**.
6. **Validate categorical values** — Checked for unexpected entries in all category columns.
7. **Check for negative values** — No impossible negative values found.
8. **Preserve User_ID** as string type throughout.

---

## EDA Methodology

- Descriptive statistics for all numerical columns.
- Value counts and percentage distributions for all categorical columns.
- Cross-tabulation of Subscription_Plan × Retention_Status.
- Cross-tabulation of Viewing_History × Retention_Status.
- Average session duration grouped by genre, plan, viewing history.
- Inactivity period statistics grouped by retention status.

---

## Feature Engineering

### Content Diversity
Since each User_ID appears multiple times (one row per viewing session), **Content_Diversity** is calculated as the number of unique Content_Genre values watched by each user. This feature is merged back into the dataset.

### Declining Engagement
Using **median-based thresholds**:
- **At Risk** = Session_Duration < median AND Inactivity_Period > median
- **Active** = all other users

This approach is transparent, data-driven, and avoids arbitrary hard-coding.

---

## Machine Learning Methodology

### Target Variable
`Retention_Status` (Retained / Churned)

### Features Used
| Feature | Type |
|---|---|
| Session_Duration | Numerical |
| Inactivity_Period | Numerical |
| Content_Diversity | Numerical |
| Viewing_History | Categorical |
| Content_Genre | Categorical |
| Subscription_Plan | Categorical |

### Preprocessing
- **Numerical:** Passed through directly (no scaling needed for RF)
- **Categorical:** OneHotEncoder (handle_unknown="ignore")
- **ColumnTransformer** combines both
- Wrapped in a **scikit-learn Pipeline** for clean prediction

---

## Random Forest Model

```python
RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced"
)
```

- `class_weight="balanced"` accounts for any class imbalance.
- `stratify=y` used in train_test_split for balanced class representation.
- `test_size=0.20`, `random_state=42`

---

## Evaluation Metrics

The model is evaluated using:
- **Accuracy** — overall correctness
- **Precision** — of predicted churned/retained, how many are correct
- **Recall** — of actual churned/retained, how many were found
- **F1-Score** — harmonic mean of precision and recall
- **Confusion Matrix** — visual breakdown of predictions

---

## Visualization Descriptions

| Chart | Description |
|---|---|
| watch_time_segment.png | Bar chart: avg session duration per subscription plan |
| genre_engagement.png | Bar chart: avg session duration per content genre |
| inactivity_retention.png | Boxplot: inactivity days split by retained vs churned |
| content_diversity.png | Scatter + line: content diversity vs avg session duration |
| confusion_matrix.png | Heatmap of RF classification predictions |
| feature_importance.png | Horizontal bar chart of RF feature importances |

---

## Key Findings

(Generated from actual dataset — see terminal output when running main.py)

1. Average session duration and median inactivity are the primary churn indicators.
2. Premium plan users tend to have higher session durations.
3. Genre preference significantly impacts watch time.
4. High inactivity strongly correlates with churned status.
5. Users with more content diversity show different engagement patterns.
6. A measurable percentage of users show early declining engagement signs.
7. Random Forest identifies Inactivity_Period and Session_Duration as the top predictors.

---

## Business Action Plan

1. **Re-engagement notifications** for users inactive beyond threshold.
2. **Genre-based personalized recommendations** for high-engagement genres.
3. **At-risk user campaigns** with exclusive content or discounts.
4. **Content diversity encouragement** via gamification (badges, streaks).
5. **Subscription plan improvement** for the lowest-retention tier.
6. **Churn prediction integration** into production pipeline using the trained model.
7. **Session duration improvement** via autoplay and adaptive streaming.

---

## How to Run the Project

```bash
# Navigate to project folder
cd OTT_Retention_Analysis

# Install dependencies
pip install -r requirements.txt

# Run the complete analysis
python src/main.py
```

All outputs (charts, model, insights) will be generated automatically.

---

## Notes and Limitations

- Dataset has 763 rows across 250 unique users — suitable for a college project.
- No timestamp data, so time-series trend analysis is not possible.
- Retention_Status appears to reflect a point-in-time snapshot.
- Model performance should be validated on fresh data before production deployment.

---

*Project developed for DS Skill-Based Assessment — College Data Science Lab*
