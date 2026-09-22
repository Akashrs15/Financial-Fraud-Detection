# Financial Fraud Detection & Risk Analysis

An end-to-end machine learning project for detecting potentially fraudulent financial transactions using Python, SMOTE, Random Forest, and Power BI.

## Project Overview

The objective of this project is to build a machine learning solution that can identify potentially fraudulent financial transactions while balancing recall and precision.

The project covers the complete workflow from data cleaning and feature engineering to model development, evaluation, threshold optimisation, transaction risk scoring, and preparation of data for Power BI reporting.

The original dataset contains 3,000 transaction records. After data quality checks, 10 records were quarantined and 2,990 valid transactions were retained for analysis. Fraud represented approximately 3.51% of the cleaned dataset.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Imbalanced-learn
- SMOTE
- Logistic Regression
- Random Forest
- Matplotlib
- Joblib
- Power BI

## Project Workflow

### 1. Data Cleaning

The raw transaction data was validated and cleaned before modelling.

Key steps included:

- Identifying invalid transaction amounts
- Detecting duplicate transactions
- Handling missing values
- Standardising data types
- Quarantining invalid records

### 2. Feature Engineering

New features were created to capture customer behaviour and transaction risk patterns.

These included:

- Age Group
- Transaction Hour
- Weekend Indicator
- High Risk Hour Indicator
- Balance Depletion Ratio
- Amount-to-Balance Ratio
- Transaction Risk Score
- Merchant Volatility Index

### 3. Handling Class Imbalance

The cleaned data was divided using a stratified 80/20 train-test split.

Because fraud represented only around 3.5% of the data, SMOTE was used to address class imbalance.

SMOTE was applied only to the training data to prevent data leakage and keep the test data representative of real-world conditions.

### 4. Model Development

Logistic Regression was used as a baseline model.

A Random Forest classifier was then trained and optimised using RandomizedSearchCV with stratified 5-fold cross-validation.

The model optimisation focused particularly on recall because failing to identify fraudulent transactions can have a significant business impact.

### 5. Model Evaluation

The final Random Forest model was evaluated on the untouched test dataset.

The decision threshold was optimised to approximately 0.467 to improve fraud detection while maintaining strong precision.

## Final Model Performance

| Metric | Result |
| --- | ---: |
| Precision | 94.7% |
| Recall | 85.7% |
| ROC-AUC | 0.999 |
| PR-AUC | 0.970 |

The final model achieved 85.7% recall while maintaining 94.7% precision.

## Power BI Preparation

The processed data was transformed into a star-schema structure for Power BI analysis.

The following tables were generated:

- Fact_Transactions
- Dim_Customer
- Dim_Merchant
- Dim_Date

These tables can be used to analyse fraud trends, customer behaviour, merchant patterns, transaction risk, and financial exposure.

## Business Value

This project demonstrates how machine learning and data analytics can support fraud investigation by:

- Identifying potentially fraudulent transactions
- Prioritising high-risk transactions
- Reducing missed fraud cases
- Controlling unnecessary false alerts
- Providing transaction-level risk scores
- Supporting data-driven fraud monitoring

## Future Improvements

- Compare additional models such as XGBoost
- Add SHAP-based model explainability
- Deploy the model using Streamlit or an API
- Add automated model monitoring
- Develop a complete interactive Power BI dashboard

## Author

**Akash Ramesh Sujatha**

MSc Data Science & Analytics
