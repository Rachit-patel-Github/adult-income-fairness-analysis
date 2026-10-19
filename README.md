# Adult Income Prediction & Fairness Analysis

## Overview

This project investigates the use of machine learning to predict income categories using the **Adult Census Income dataset**. It evaluates predictive performance while examining how model choices, decision thresholds, demographic subgroups, and fairness metrics affect the interpretation and responsible deployment of predictions.

The project was completed as part of the **CSET485 — AI and Society** assignment.

## Objectives

- Preprocess and analyze the Adult Census Income dataset.
- Build and compare linear regression and logistic regression models.
- Evaluate model accuracy and ROC-AUC.
- Investigate subgroup relationships and potential interaction effects.
- Examine the impact of omitting a predictor variable.
- Analyze high-confidence incorrect predictions.
- Compare fairness metrics across different classification thresholds.
- Reflect on the limitations of using a single performance metric to make deployment decisions.

## Dataset

The project uses the Adult Census Income dataset, which contains demographic, educational, employment, and financial attributes to predict whether an individual's annual income exceeds $50,000.

**Dataset summary:**
- Original records: 32,561
- Original columns: 15
- Numerical predictors: 6
- Categorical predictors: 8
- Target classes: `<=50K` and `>50K`

The original dataset contains missing values in attributes including `workclass`, `occupation`, and `native_country`. Preprocessing includes missing-value handling, categorical encoding, and feature scaling.

## Methodology

### 1. Data Preprocessing
- Inspect the dataset and target distribution.
- Handle missing values.
- Separate training and testing data using a stratified split.
- Encode categorical variables and scale numerical features.
- Fit preprocessing transformations using training data to avoid data leakage.

### 2. Model Comparison
- **Linear Regression (OLS):** Convert continuous predictions into binary classifications using a threshold based on the median predicted value.
- **Logistic Regression:** Predict income-class probabilities and classify observations using a threshold of 0.5.

### 3. Subgroup Analysis
Examine the relationship between education and predicted income across demographic subgroups. Evaluate the education-by-gender interaction and interpret the results using coefficients, confidence intervals, and p-values.

### 4. Omitted-Variable Analysis
Remove `education_num` and compare selected logistic regression coefficients between the full and reduced models to investigate model sensitivity.

### 5. Adversarial Error Analysis
Identify high-confidence incorrect predictions and inspect a sample of 50 errors to investigate patterns that may contribute to model failures.

### 6. Fairness–Threshold Analysis
Compare demographic parity, equalized-odds-related ratios, and predictive parity across multiple classification thresholds. Examine how changes in thresholds affect subgroup selection rates and error rates.

## Key Results

| Evaluation | Result |
|---|---:|
| Logistic regression accuracy | 85.10% |
| Logistic regression ROC-AUC | 0.9048 |
| OLS classification accuracy | 70.98% |
| Disagreement between model classifications | 30.05% |
| High-confidence incorrect predictions | 479 |
| High-confidence errors as a proportion of the test set | 4.9% |

The fairness analysis demonstrates that different metrics can favor different thresholds. For example, at a threshold of 0.35, the predictive parity ratio was approximately 1.0104, while the demographic parity ratio was 0.3079. This illustrates why a single metric cannot fully characterize a model's fairness.

## Conclusion

The analysis shows that strong predictive performance does not automatically imply fair or responsible deployment. Model evaluations should consider subgroup outcomes, high-confidence errors, threshold sensitivity, omitted-variable effects, and the consequences of incorrect decisions.

The project supports a **conditional deployment recommendation**: further validation, transparent reporting, human oversight, and ongoing monitoring should be considered before using such predictions in consequential real-world decisions.

## Technologies Used

- Python
- Google Colab / Jupyter Notebook
- Pandas
- NumPy
- Scikit-learn
- Statsmodels
- Matplotlib
- Seaborn

## Repository Contents

A suggested repository structure is:

```text
adult-income-fairness-analysis/
├── README.md
├── notebooks/
│   └── adult_income_fairness_analysis.ipynb
├── data/
│   └── README.md
└── report/
    └── assignment_report.pdf
```

*Include only files that you actually upload. If the dataset is not included, document where it can be obtained.*

## Reproducibility

The experiment uses random seed **297** for the relevant randomized operations. Run the notebook from top to bottom to reproduce the preprocessing, model evaluation, subgroup analysis, error analysis, and fairness comparisons, provided the required dataset and dependencies are available.

## Limitations

- The dataset may not represent the population in which a model would eventually be used.
- Predictive associations do not establish causation.
- Fairness metrics can conflict, and no single threshold necessarily satisfies all fairness objectives.
- High-confidence predictions can still be incorrect.
- The analysis cannot rule out all unobserved confounding factors.

## Disclaimer

This project is for educational and analytical purposes. The model should not be treated as a validated system for making real-world decisions about individuals.

## Author

Created as part of the CSET485 — AI and Society coursework.
