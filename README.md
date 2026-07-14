# Breast Cancer Classification — Decision Tree

An end-to-end machine learning pipeline that classifies breast tumors as **Malignant** or **Benign** using a **Decision Tree Classifier**.

## 📌 Problem Statement
- **Objective:** Classify breast tumors as malignant or benign based on cell nuclei measurements from digitized images
- **Dataset:** Breast Cancer Wisconsin dataset (`data.csv`) — 569 records, 33 columns
- **Problem Type:** Classification
- **Target Variable:** `diagnosis` (M = Malignant, B = Benign)
- **Business Relevance:** Assists in early and accurate cancer diagnosis, supporting doctors in making faster, more informed treatment decisions
- **Success Metric:** F1-score > 0.80 → **Achieved: F1 = 0.9208** (before tuning) / **93.86% accuracy** (after tuning)

## 📊 Dataset
- **Rows:** 569
- **Dropped columns:** `id` (identifier), `unnamed: 32` (constant/empty column)
- **Class distribution:** Benign 62.74% | Malignant 37.26% (reasonably balanced)
- **Features:** 30 numeric measurements (radius, texture, perimeter, area, smoothness, compactness, concavity, symmetry, etc. — mean, standard error, and worst-case values)
- **Dropped highly correlated features (>0.95):** `perimeter_mean`, `area_mean`, `perimeter_se`, `area_se`, `radius_worst`, `perimeter_worst`, `area_worst`
- No missing values or duplicate rows detected

## 🛠 Tech Stack
- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn
- XGBoost
- Joblib

## 🔍 Pipeline / Methodology
1. **Data Loading & Validation** — shape, info, summary statistics
2. **Data Cleaning** — standardized column names, dropped ID and constant columns
3. **Exploratory Data Analysis** — dashboard overview, class distribution check, correlation analysis, skewness check
4. **Outlier Treatment** — boxplot-based capping applied to numeric features
5. **Feature Engineering** — removed 7 highly correlated features to reduce redundancy
6. **Preprocessing Pipeline** — median imputation + standard scaling (via `ColumnTransformer`)
7. **Train-Test Split** — 80/20 split (455 train / 114 test)
8. **Model Comparison** — benchmarked 10 classification algorithms using 5-fold cross-validation
9. **Model Training** — final pipeline trained using **Decision Tree**
10. **Model Evaluation** — Accuracy, Precision, Recall, F1-score, Confusion Matrix, Classification Report
11. **Sanity Checks** — train vs test score comparison, feature importance dominance check
12. **Cross Validation** — 5-fold CV for robustness check
13. **Hyperparameter Tuning** — GridSearchCV & RandomizedSearchCV to tune `max_depth`

## 📈 Results

| Metric | Before Tuning | After Tuning |
|--------|---------------|--------------|
| **Accuracy** | 0.9211 | **0.9386** |
| **Precision** | 0.9208 | — |
| **Recall** | 0.9211 | — |
| **F1-Score** | 0.9208 | — |
| Best Hyperparameter (GridSearchCV) | — | `max_depth = 3` |
| Cross-Validation Mean Accuracy | 0.9156 (± 0.0214) | — |
| Train Score vs Test Score (untuned) | 1.0000 vs 0.9211 | — |

**Classification Report (before tuning):**

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| B (Benign) | 0.93 | 0.94 | 0.94 |
| M (Malignant) | 0.90 | 0.88 | 0.89 |

**Model Comparison (Top Results — CV Accuracy):**

| Model | CV Accuracy |
|-------|-------------|
| SVM | 0.9737 |
| XGBoost | 0.9684 |
| AdaBoost | 0.9684 |
| Logistic Regression | 0.9667 |
| **Decision Tree (selected)** | 0.9156 |

## 🔑 Key Insights
- **Best Model overall:** SVM achieved the highest cross-validation score (97.37%), notably outperforming the Decision Tree
- The **untuned Decision Tree showed clear overfitting** — a perfect training score of 1.0000 against a test score of 0.9211, a classic sign of a single tree memorizing the training data
- **Hyperparameter tuning fixed this:** restricting `max_depth` to 3 via GridSearchCV improved the final test accuracy to **93.86%** while reducing overfitting

## 📂 Folder Structure
```
Breast-Cancer-Classification/
│
├── data/
│   └── data.csv
├── decision-tree-breast-cancer-classification.ipynb
└── README.md
```

## 👤 Author
**Nimra Muhammad Imran**
