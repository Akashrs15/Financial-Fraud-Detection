# Financial Fraud Detection & Risk Analysis

This project focuses on detecting potentially fraudulent financial transactions using Python and machine learning. I built an end-to-end fraud detection pipeline covering data cleaning, feature engineering, class imbalance handling, model development, evaluation, threshold optimisation, risk scoring, and preparation of data for Power BI reporting.

## Project Overview

Financial fraud is difficult to detect because fraudulent transactions usually make up only a small percentage of the overall data. In this dataset, fraud represented approximately 3.51% of the cleaned transactions.

The main goal of this project was to build a model that could identify as many fraudulent transactions as possible while keeping false alerts under control.

The original dataset contained 3,000 transaction records. During the data quality checks, 10 records were quarantined because of invalid transaction amounts or duplicate records. This left 2,990 valid transactions for analysis and modelling.

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

I first checked the raw transaction data for quality issues before using it for modelling.

The main steps included:

- Identifying invalid transaction amounts
- Detecting duplicate transactions
- Handling missing values
- Standardising data types
- Quarantining invalid records instead of simply removing them without a record

After cleaning, 2,990 transactions were available for further analysis.

### 2. Feature Engineering

I created additional features to capture transaction behaviour and possible fraud risk patterns.

The engineered features included:

- Age Group
- Transaction Hour
- Weekend Indicator
- High Risk Hour Indicator
- Balance Depletion Ratio
- Amount to Balance Ratio
- Transaction Risk Score
- Merchant Volatility Index

Categorical variables were also encoded so that they could be used by the machine learning models.

### 3. Handling Class Imbalance

Fraud accounted for only around 3.5% of the cleaned dataset, which created a clear class imbalance problem.

I used a stratified 80/20 train-test split so that the fraud distribution was maintained across the training and test datasets.

SMOTE was then applied only to the training data. The test data was kept untouched so that the final evaluation reflected the original fraud distribution and avoided data leakage.

### 4. Model Development

I first trained a Logistic Regression model as a baseline.

I then developed a Random Forest classifier and used RandomizedSearchCV with stratified 5-fold cross-validation to search for better hyperparameters.

The model selection process focused heavily on recall because missing a fraudulent transaction can have a greater business impact than reviewing an additional suspicious transaction.

### 5. Model Evaluation

The final Random Forest model was evaluated on the untouched test dataset.

At the default classification threshold of 0.50, the model achieved high precision but missed more fraudulent transactions than desired.

I therefore tuned the decision threshold to approximately **0.467** to improve fraud detection while still maintaining strong precision.

## Final Model Performance

| Metric | Result |
| --- | ---: |
| Precision | 94.7% |
| Recall | 85.7% |
| ROC-AUC | 0.999 |
| PR-AUC | 0.970 |

At the optimised threshold, the model detected approximately **85.7% of fraudulent transactions** while maintaining **94.7% precision**.

This provided a better balance between detecting fraud and limiting unnecessary false alerts.

## Transaction Risk Scoring

The trained model was used to generate fraud probabilities for individual transactions.

These scores can help identify higher-risk transactions and support the prioritisation of cases that may require further investigation.

## Power BI Preparation

The processed transaction data was prepared in a star-schema structure so that the model outputs could be analysed in Power BI.

The following tables were created:

- Fact_Transactions
- Dim_Customer
- Dim_Merchant
- Dim_Date

This structure supports analysis of areas such as:

- Fraud trends
- High-risk transactions
- Customer behaviour
- Merchant patterns
- Transaction risk
- Potential financial exposure

## Business Value

This project shows how machine learning can support fraud monitoring and investigation.

The solution can help with:

- Identifying potentially fraudulent transactions
- Prioritising high-risk cases for investigation
- Reducing the number of missed fraud cases
- Limiting unnecessary false alerts
- Generating transaction-level fraud probabilities
- Supporting data-driven fraud monitoring and decision-making

## Future Improvements

The project could be extended by:

- Comparing additional models such as XGBoost
- Adding SHAP for model explainability
- Deploying the model through Streamlit or an API
- Adding automated model monitoring
- Building a complete interactive Power BI fraud monitoring dashboard

## Author

**Akash Ramesh Sujatha**

MSc Data Science & Analytics
