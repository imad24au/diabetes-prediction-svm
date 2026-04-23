# Diabetes Prediction — Support Vector Machine (SVM)

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![sklearn](https://img.shields.io/badge/scikit--learn-1.0+-orange)
![Status](https://img.shields.io/badge/Status-Complete-green)
![Domain](https://img.shields.io/badge/Domain-Healthcare-red)

## Project Overview

This project builds a binary classification model to predict whether 
a patient has diabetes based on medical measurements.

This is a real-world healthcare problem — early detection of diabetes 
can significantly improve patient outcomes and reduce long-term 
health complications.

---

## Dataset

- **Source:** Pima Indians Diabetes Dataset
- **Link:** https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv
- **Size:** 768 female patients aged 21 and above
- **Target:** `Outcome` — 1 = has diabetes, 0 = no diabetes

### Features

| Feature | Description |
|---------|-------------|
| `Pregnancies` | Number of times pregnant |
| `Glucose` | Plasma glucose concentration |
| `BloodPressure` | Diastolic blood pressure (mm Hg) |
| `SkinThickness` | Triceps skin fold thickness (mm) |
| `Insulin` | 2 hour serum insulin |
| `BMI` | Body mass index |
| `DiabetesPedigree` | Diabetes pedigree function — family history score |
| `Age` | Age in years |

---

## Problem Statement

The dataset is **imbalanced**:
- 65% of patients do NOT have diabetes
- 35% of patients DO have diabetes

A naive model predicting "no diabetes" for everyone would achieve 
65% accuracy without learning anything useful.

In healthcare, **Recall is the most critical metric** — missing a 
diabetic patient (false negative) is far more dangerous than a 
false alarm. This guided our evaluation approach throughout.

---

## Data Cleaning

The dataset contained **hidden missing values** — five columns 
(`Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`) 
used zeros as placeholders for missing data. A person cannot 
have 0 blood pressure or 0 BMI and be alive!

These zeros were replaced with the **column median** — a robust 
measure that handles outliers better than the mean.

---

## Project Structure

```
diabetes-prediction-svm/
│
├── diabetes_prediction_svm.ipynb   ← Full analysis notebook
└── README.md                       ← Project description
```

## Methodology

### 1. Data Loading and Exploration
- Loaded dataset directly from web — no manual download needed
- Inspected shape, data types and distributions
- Identified and fixed hidden zero values in 5 columns
- Visualised class imbalance

### 2. Preprocessing
- Defined X (8 features) and y (target)
- Split data 80/20 with stratification to preserve class balance
- Applied StandardScaler — critical for SVM which uses distances!

### 3. Kernel Comparison
- Trained Linear, Polynomial and RBF kernels
- Compared accuracy across all three
- RBF kernel outperformed others consistently

### 4. Best Model Evaluation
- Full classification report — Precision, Recall, F1 per class
- Confusion matrix with heatmap visualisation
- Analysed false negatives — the most dangerous error type

### 5. Hyperparameter Experiment
- Tested 16 combinations of C and gamma values
- Found performance ceiling around 73-74% for basic SVM
- Lower C values with gamma='scale' most reliable

---

## Results

### Kernel Comparison

| Kernel | Accuracy |
|--------|----------|
| Linear | 0.7078 |
| Polynomial (d=3) | 0.7143 |
| **RBF** | **0.7338** |

### Best Model — RBF SVM (C=1.0, gamma='scale')

| Metric | No Diabetes | Diabetes |
|--------|------------|---------|
| Precision | 0.77 | 0.64 |
| Recall | 0.84 | 0.54 |
| F1 Score | 0.80 | 0.59 |
| **Accuracy** | | **0.7338** |

### Confusion Matrix

| | Predicted No | Predicted Yes |
|--|--|--|
| **Actual No** | 84 | 16 |
| **Actual Yes** | 25 | 29 |


### Key Findings

- **RBF kernel** consistently outperforms Linear and Polynomial
- Model catches **84%** of healthy patients correctly
- Model only catches **54%** of diabetic patients — room for improvement
- **25 missed diabetic patients** highlight the challenge of imbalanced healthcare data
- Multiple C and gamma combinations tied at 73.38% — suggesting a performance ceiling for basic SVM
- Lower C (0.1, 1.0) with `gamma='scale'` is the most reliable and stable configuration

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3 | Programming language |
| Pandas | Data manipulation and cleaning |
| NumPy | Numerical operations |
| Matplotlib | Visualisation |
| Scikit-learn | SVM model, scaling, evaluation |

---

## How to Run

1. Open `diabetes_prediction_svm.ipynb` in Google Colab or Jupyter
2. Run all cells from top to bottom
3. Dataset loads automatically — no manual download needed!

```python
# Dataset loads with just this line
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
df = pd.read_csv(url, names=col_names)
```

---

## Lessons Learned

- Always check for hidden missing values — zeros can be placeholders!
- SVM requires feature scaling — distances dominate without it
- Accuracy is misleading on imbalanced datasets — always check Recall
- RBF kernel is the best default for unknown data structure
- In healthcare ML, missing a positive case costs more than a false alarm
- A performance ceiling exists — advanced techniques needed to break through

---

## Future Improvements

- Use `class_weight='balanced'` in SVC to handle class imbalance
- Apply GridSearchCV with cross-validation for more robust tuning
- Try feature engineering — combining Glucose x BMI as a new feature
- Explore ensemble methods like Random Forest and XGBoost
- Apply SMOTE oversampling to handle class imbalance

---

## Author

Built as part of an AI/ML Practice.

*Demonstrates the full ML workflow on a real healthcare dataset:*
*data cleaning, preprocessing, kernel comparison, evaluation and*
*hyperparameter optimisation.*
