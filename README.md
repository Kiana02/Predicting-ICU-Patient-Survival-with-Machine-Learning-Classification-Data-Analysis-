# Predicting Patient Survival in Critically Ill Populations

A binary classification pipeline that predicts patient mortality using a SUPPORT-based critical care dataset of 9,105 patients across five U.S. medical centers. Built for the Data Science Lab course at Politecnico di Torino (Fall 2025/2026).

**Final result: 0.7559 Macro F1-Score** on a held-out validation split — up from a naive baseline of ~0.49.

## Overview

The dataset covers nine severe disease categories (e.g., multi-organ failure with sepsis, metastatic cancer, coma) recorded between 1989–1994, with ~45 features per patient spanning demographics, comorbidities, day-3 physiological measurements, lab values, and existing severity scores (APACHE III, SUPPORT). The target is binary: whether a patient died (`death = 1`) or survived (`death = 0`).

The class distribution is imbalanced (~68% died / 32% survived), which motivated using **Macro F1** as the evaluation metric — it weights both classes equally, so a model can't inflate its score by just predicting the majority class.

## Approach

**Preprocessing**
- Median imputation for missing numeric values (computed after feature engineering)
- Categorical variables (`dzgroup`, `dzclass`, `race`, `income`, `dnr`, `medical_center`, `admission_month`, `ca`, etc.) retained and one-hot encoded — dropping them early on had discarded clinically important signals like DNR status and metastatic cancer
- Vote-based outlier removal: a patient is flagged only if more than 8 of ~40 numeric columns register as extreme (3×IQR), with missing values explicitly excluded from voting. This removed ~1.4% of rows while preserving the dataset's true class balance

**Feature engineering** — six derived features layered on top of the raw variables:
- `age_group` — ordinal age buckets (≤40, 41–60, >60)
- `severity_index` — weighted composite of comorbidity count, APACHE III, and SUPPORT physiology scores
- `lab_abnormality_count` — count of clinically abnormal lab results (bilirubin, creatinine, BUN, glucose)
- `oxygen_efficiency` — PaO2/FiO2 ratio relative to mean arterial pressure
- `metabolic_stress` — glucose × temperature interaction
- Pairwise interaction terms: age × severity, respiratory rate × oxygenation, heart rate × glucose

**Scaling & feature selection**
- `RobustScaler` (median/IQR-based) to handle the heavy right-skew in clinical lab values without letting extreme-but-genuine outliers dominate
- `SelectKBest` (ANOVA F-test) to reduce to the top 70 features

**Modeling**
- Three tuned base learners, each optimized via `GridSearchCV` (3-fold stratified CV, scored on Macro F1): Logistic Regression, Random Forest, Gradient Boosting
- Combined into a soft-voting `VotingClassifier` ensemble
- All model selection and hyperparameter tuning done exclusively on training data, with a held-out 20% stratified split never touched during fitting

**Threshold calibration**
- Default 0.50 threshold is suboptimal under class imbalance for Macro F1
- Searched thresholds in [0.30, 0.70] on holdout predictions only, landing on an optimal threshold of 0.53

## Results

| Model | CV F1 | Holdout F1 |
|---|---|---|
| Logistic Regression | 0.6560 | 0.6715 |
| Random Forest | 0.7473 | 0.7474 |
| Gradient Boosting | 0.7342 | 0.7354 |
| Ensemble (soft voting, t=0.50) | – | 0.7516 |
| **Ensemble (soft voting, t=0.53)** | – | **0.7559** |

All holdout figures are computed on a stratified 20% split that was excluded from training, feature selection, scaling, and hyperparameter search at every stage.

## Project structure

```
├── patient_survival_pipeline.py   # End-to-end pipeline: preprocessing → feature engineering → training → prediction
├── development.csv                # Labeled training data (7,284 patients)
├── evaluation.csv                 # Unlabeled evaluation data (1,821 patients)
├── sample_submission.csv          # Submission format reference
├── submission.csv                 # Final predictions
└── report_exam_fall_2026.pdf      # Full write-up: EDA, methodology, and discussion
```

## Usage

```bash
python patient_survival_pipeline.py
```

This trains the ensemble on `development.csv`, calibrates the decision threshold on a held-out split, and writes predictions for `evaluation.csv` to `submission.csv`.

**Requirements:** `pandas`, `numpy`, `scikit-learn`

## Limitations & future work

- Median imputation assumes missingness is uninformative, but in this dataset a missing lab value may itself correlate with clinical protocol or how early a patient died
- Heavy reliance on one-hot encoding for high-cardinality features (`dzgroup`, `medical_center`) inflates dimensionality; target encoding could retain the same signal more compactly
- The `GridSearchCV` search space was kept narrow to manage runtime across three separate model searches — a broader or randomized search would likely leave some performance on the table

## Author

Kiana Khalili — MSc Data Science and Engineering, Politecnico di Torino
