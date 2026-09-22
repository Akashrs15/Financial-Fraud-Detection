"""
===============================================================================
FINANCIAL FRAUD DETECTION & RISK ANALYSIS — COMPLETE SOLUTION
===============================================================================
Implements every step of the Project Requirements Document:
  Phase 1 - Data Engineering, Feature Engineering & Class Imbalance Handling
  Phase 2 - Model Development & Evaluation (Random Forest)
  Phase 3 (prep) - Star-schema tables ready to import into Power BI

HOW TO RUN
----------
1. Place Fraud_Transactions_Data.csv in the same folder as this script
   (or edit RAW_PATH below).
2. pip install pandas numpy scikit-learn imbalanced-learn matplotlib joblib
3. python Fraud_Detection_Complete_Solution.py

OUTPUT (written to ./solution_output/)
---------------------------------------
  quarantined_fraud_data.csv          - rows failing data-quality checks (Step 2)
  Fraud_Data_Cleaned_Engineered.csv   - clean, feature-engineered base table
  Fraud_Data_Scored_Full.csv          - base table + model probability/prediction
  fraud_random_forest_model.joblib    - trained, tuned model
  metrics_summary.json                - all evaluation numbers in one place
  confusion_matrix.png / roc_pr_curves.png / feature_importance.png /
  threshold_tuning.png                - evaluation charts
  PowerBI_Star_Schema/                - Fact_Transactions, Dim_Customer,
                                         Dim_Merchant, Dim_Date (ready for
                                         Power Query / Power BI import)
===============================================================================
"""
import os
import json

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, average_precision_score, precision_recall_curve,
    classification_report,
)
from imblearn.over_sampling import SMOTE

# ------------------------------------------------------------------------- #
# CONFIG
# ------------------------------------------------------------------------- #
HERE = os.path.dirname(os.path.abspath(__file__))
RAW_PATH = os.path.join(HERE, "Fraud_Transactions_Data.csv")
OUT_DIR = os.path.join(HERE, "solution_output")
PBI_DIR = os.path.join(OUT_DIR, "PowerBI_Star_Schema")
os.makedirs(PBI_DIR, exist_ok=True)

RANDOM_STATE = 42
TARGET_RECALL = 0.85  # business requirement: catch >= 85% of fraud

pd.set_option("display.width", 140)


# =========================================================================
# PHASE 1 — DATA ENGINEERING, FEATURE ENGINEERING & CLASS IMBALANCE HANDLING
# =========================================================================
def phase1_data_engineering():
    print("=" * 70)
    print("PHASE 1: Data Engineering, Feature Engineering & Class Imbalance")
    print("=" * 70)

    # --- Step 1: Ingestion & type standardization ---------------------- #
    df = pd.read_csv(RAW_PATH, dtype={"Transaction_ID": str, "Customer_ID": str})
    df["Transaction_DateTime"] = pd.to_datetime(df["Transaction_DateTime"], errors="coerce")
    numeric_cols = ["Transaction_Amount", "Account_Balance_Before", "Account_Balance_After",
                     "Distance_From_Home_KM", "Customer_Age", "Customer_Tenure_Months",
                     "Previous_Fraud_Flags"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    print(f"[Step 1] Rows ingested: {len(df):,}")

    # --- Step 2: Heuristic anomaly filtering & deduplication ----------- #
    bad_amount_mask = df["Transaction_Amount"] <= 0
    dup_mask = df.duplicated(subset=["Transaction_ID", "Transaction_DateTime", "Transaction_Amount"],
                              keep="first")
    quarantine_mask = bad_amount_mask | dup_mask

    quarantined = df[quarantine_mask].copy()
    quarantined["Quarantine_Reason"] = np.select(
        [bad_amount_mask[quarantine_mask], dup_mask[quarantine_mask]],
        ["Non-positive Transaction_Amount", "Duplicate transaction record"],
        default="Unknown",
    )
    quarantined.to_csv(os.path.join(OUT_DIR, "quarantined_fraud_data.csv"), index=False)

    df = df[~quarantine_mask].copy()
    print(f"[Step 2] Quarantined: {len(quarantined):,} rows "
          f"(bad amount: {bad_amount_mask.sum()}, duplicates: {dup_mask.sum()}) "
          f"-> clean rows: {len(df):,}")

    df["Distance_From_Home_KM"] = (
        df.groupby("Merchant_Category")["Distance_From_Home_KM"]
        .transform(lambda s: s.fillna(s.median()))
    )
    df["Distance_From_Home_KM"] = df["Distance_From_Home_KM"].fillna(df["Distance_From_Home_KM"].median())

    # --- Step 3: Derived feature engineering (vectorized) --------------- #
    age_bins = [17, 25, 35, 45, 55, 65, 120]
    age_labels = ["18-25", "26-35", "36-45", "46-55", "56-65", "66+"]
    df["Age_Group"] = pd.cut(df["Customer_Age"], bins=age_bins, labels=age_labels)

    df["Transaction_Hour"] = df["Transaction_DateTime"].dt.hour
    df["Is_Weekend"] = df["Transaction_DateTime"].dt.dayofweek.isin([5, 6]).astype(int)
    df["Is_High_Risk_Hour"] = df["Transaction_Hour"].between(0, 5).astype(int)

    raw_depletion = np.where(
        df["Account_Balance_Before"] > 0,
        (df["Account_Balance_Before"] - df["Account_Balance_After"]) / df["Account_Balance_Before"],
        0.0,
    )
    df["Balance_Depletion_Ratio"] = np.clip(raw_depletion, -1, 1)

    df["Amount_to_Balance_Ratio"] = np.where(
        df["Account_Balance_Before"] > 0,
        df["Transaction_Amount"] / df["Account_Balance_Before"],
        0.0,
    )
    print("[Step 3] Engineered: Age_Group, Transaction_Hour, Is_Weekend, "
          "Is_High_Risk_Hour, Balance_Depletion_Ratio, Amount_to_Balance_Ratio")

    # --- Step 4: Advanced feature engineering (risk scoring) ------------ #
    card_present_risky = (df["Card_Present"] == "No") & (
        df["Transaction_Type"].isin(["POS Purchase", "ATM Withdrawal"]))

    df["Transaction_Risk_Score"] = (
        card_present_risky.astype(int) * 30
        + df["Is_High_Risk_Hour"] * 25
        + (df["Distance_From_Home_KM"] > 100).astype(int) * 20
        + df["Previous_Fraud_Flags"].clip(upper=5) * 15
        + (df["Amount_to_Balance_Ratio"] > 0.5).astype(int) * 10
    ).clip(upper=100)

    grp = df.groupby("Merchant_Category")["Transaction_Amount"]
    df["Merchant_Volatility_Index"] = ((grp.transform("std") / grp.transform("mean")) * 100).round(2)
    print("[Step 4] Engineered: Transaction_Risk_Score (0-100), Merchant_Volatility_Index (CV%)")

    df.to_csv(os.path.join(OUT_DIR, "Fraud_Data_Cleaned_Engineered.csv"), index=False)
    print(f"Saved -> Fraud_Data_Cleaned_Engineered.csv ({len(df):,} rows)")

    # --- Step 5: Class imbalance correction with SMOTE (train only) ----- #
    feature_cols = [
        "Transaction_Amount", "Account_Balance_Before", "Account_Balance_After",
        "Distance_From_Home_KM", "Customer_Age", "Customer_Tenure_Months",
        "Previous_Fraud_Flags", "Transaction_Hour", "Is_Weekend", "Is_High_Risk_Hour",
        "Balance_Depletion_Ratio", "Amount_to_Balance_Ratio", "Transaction_Risk_Score",
        "Merchant_Volatility_Index",
    ]
    categorical_cols = ["Transaction_Type", "Merchant_Category", "Card_Present",
                         "Device_Type", "Transaction_Country", "Customer_Gender", "Age_Group"]

    model_df = pd.get_dummies(df[feature_cols + categorical_cols], columns=categorical_cols, drop_first=True)
    X, y = model_df, df["Is_Fraud"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )
    print(f"[Step 5] Split -> train {len(X_train):,} ({y_train.mean()*100:.2f}% fraud), "
          f"test {len(X_test):,} ({y_test.mean()*100:.2f}% fraud)")

    smote = SMOTE(sampling_strategy=0.4, random_state=RANDOM_STATE)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    print(f"[Step 5] After SMOTE (train only) -> {len(X_train_res):,} rows, "
          f"{y_train_res.mean()*100:.2f}% fraud\n")

    return {
        "df": df,
        "X_train_res": X_train_res, "y_train_res": y_train_res,
        "X_test": X_test, "y_test": y_test,
        "feature_names": list(X.columns),
    }


# =========================================================================
# PHASE 2 — MODEL DEVELOPMENT & EVALUATION
# =========================================================================
def phase2_model_development(artifacts):
    print("=" * 70)
    print("PHASE 2: Model Development & Evaluation")
    print("=" * 70)

    X_train_res, y_train_res = artifacts["X_train_res"], artifacts["y_train_res"]
    X_test, y_test = artifacts["X_test"], artifacts["y_test"]
    feature_names = artifacts["feature_names"]

    # --- Step 1: Baseline & candidate model ------------------------------ #
    baseline = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE)
    baseline.fit(X_train_res, y_train_res)
    baseline_pred = baseline.predict(X_test)
    print(f"[Step 1] Baseline (Logistic Regression) -> recall: "
          f"{recall_score(y_test, baseline_pred):.3f}, "
          f"precision: {precision_score(y_test, baseline_pred, zero_division=0):.3f}")

    rf_initial = RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1)
    rf_initial.fit(X_train_res, y_train_res)
    rf_initial_pred = rf_initial.predict(X_test)
    print(f"[Step 1] Candidate (Random Forest, default) -> recall: "
          f"{recall_score(y_test, rf_initial_pred):.3f}, "
          f"precision: {precision_score(y_test, rf_initial_pred, zero_division=0):.3f}")

    # --- Step 2: Hyperparameter tuning ------------------------------------ #
    param_dist = {
        "n_estimators": [200, 300, 400, 600],
        "max_depth": [None, 8, 12, 16, 20],
        "min_samples_leaf": [1, 2, 4, 8],
        "max_features": ["sqrt", "log2"],
        "class_weight": [None, "balanced", "balanced_subsample"],
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    search = RandomizedSearchCV(
        RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
        param_distributions=param_dist, n_iter=20, scoring="recall",
        cv=cv, random_state=RANDOM_STATE, n_jobs=-1,
    )
    search.fit(X_train_res, y_train_res)
    best_rf = search.best_estimator_
    print(f"[Step 2] Best params: {search.best_params_}")

    # --- Step 3: Evaluation on the untouched test set --------------------- #
    y_proba = best_rf.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    cm = confusion_matrix(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_proba)
    pr_auc = average_precision_score(y_test, y_proba)

    print(f"\n[Step 3] Test set @ threshold 0.5")
    print(f"Confusion matrix:\n{cm}")
    print(f"Precision: {precision:.3f} | Recall: {recall:.3f} | F1: {f1:.3f} | "
          f"ROC-AUC: {roc_auc:.3f} | PR-AUC: {pr_auc:.3f}")
    print(classification_report(y_test, y_pred, target_names=["Legit", "Fraud"]))

    _plot_confusion_matrix(cm)
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    prec_curve, rec_curve, pr_thresh = precision_recall_curve(y_test, y_proba)
    _plot_roc_pr(fpr, tpr, roc_auc, rec_curve, prec_curve, pr_auc)

    # --- Step 4: Feature importance --------------------------------------- #
    importances = pd.Series(best_rf.feature_importances_, index=feature_names).sort_values(ascending=False)
    top_n = importances.head(12)
    print(f"\n[Step 4] Top features:\n{top_n}")
    _plot_feature_importance(top_n)

    # --- Step 5: Decision-threshold tuning --------------------------------- #
    valid = rec_curve >= TARGET_RECALL
    if valid[:-1].any():
        candidate_idxs = np.where(valid[:-1])[0]
        chosen_idx = candidate_idxs[np.argmax(pr_thresh[candidate_idxs])]
        chosen_threshold = float(pr_thresh[chosen_idx])
    else:
        chosen_threshold = 0.5

    y_pred_tuned = (y_proba >= chosen_threshold).astype(int)
    cm_tuned = confusion_matrix(y_test, y_pred_tuned)
    precision_tuned = precision_score(y_test, y_pred_tuned, zero_division=0)
    recall_tuned = recall_score(y_test, y_pred_tuned, zero_division=0)
    print(f"\n[Step 5] Tuned threshold for >= {TARGET_RECALL*100:.0f}% recall: {chosen_threshold:.3f}")
    print(f"Confusion matrix (tuned):\n{cm_tuned}")
    print(f"Precision: {precision_tuned:.3f} | Recall: {recall_tuned:.3f}\n")
    _plot_threshold_tuning(pr_thresh, prec_curve, rec_curve, chosen_threshold)

    joblib.dump(best_rf, os.path.join(OUT_DIR, "fraud_random_forest_model.joblib"))

    metrics_summary = {
        "baseline_logreg": {
            "recall": round(recall_score(y_test, baseline_pred), 4),
            "precision": round(precision_score(y_test, baseline_pred, zero_division=0), 4),
        },
        "random_forest_default": {
            "recall": round(recall_score(y_test, rf_initial_pred), 4),
            "precision": round(precision_score(y_test, rf_initial_pred, zero_division=0), 4),
        },
        "best_params": search.best_params_,
        "test_at_threshold_0.50": {
            "confusion_matrix": cm.tolist(), "precision": round(precision, 4),
            "recall": round(recall, 4), "f1": round(f1, 4),
            "roc_auc": round(roc_auc, 4), "pr_auc": round(pr_auc, 4),
        },
        "test_at_tuned_threshold": {
            "threshold": round(chosen_threshold, 4),
            "confusion_matrix": cm_tuned.tolist(),
            "precision": round(precision_tuned, 4), "recall": round(recall_tuned, 4),
        },
        "top_features": top_n.round(4).to_dict(),
    }
    with open(os.path.join(OUT_DIR, "metrics_summary.json"), "w") as f:
        json.dump(metrics_summary, f, indent=2)

    return best_rf, chosen_threshold, feature_names


def _plot_confusion_matrix(cm):
    fig, ax = plt.subplots(figsize=(5, 4.2))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black",
                     fontsize=13, fontweight="bold")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Legit", "Fraud"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Legit", "Fraud"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix (threshold = 0.50)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "confusion_matrix.png"), dpi=150)
    plt.close()


def _plot_roc_pr(fpr, tpr, roc_auc, rec_curve, prec_curve, pr_auc):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    axes[0].plot(fpr, tpr, color="#1F4E79", lw=2, label=f"ROC-AUC = {roc_auc:.3f}")
    axes[0].plot([0, 1], [0, 1], "--", color="gray", lw=1)
    axes[0].set_xlabel("False Positive Rate"); axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curve"); axes[0].legend(loc="lower right")

    axes[1].plot(rec_curve, prec_curve, color="#C0392B", lw=2, label=f"PR-AUC = {pr_auc:.3f}")
    axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall Curve"); axes[1].legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "roc_pr_curves.png"), dpi=150)
    plt.close()


def _plot_feature_importance(top_n):
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    top_n.iloc[::-1].plot(kind="barh", ax=ax, color="#1F4E79")
    ax.set_xlabel("Random Forest Feature Importance")
    ax.set_title("Top Predictive Features — Fraud Detection Model")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "feature_importance.png"), dpi=150)
    plt.close()


def _plot_threshold_tuning(pr_thresh, prec_curve, rec_curve, chosen_threshold):
    fig, ax = plt.subplots(figsize=(7, 4.3))
    ax.plot(pr_thresh, prec_curve[:-1], label="Precision", color="#1F4E79", lw=2)
    ax.plot(pr_thresh, rec_curve[:-1], label="Recall", color="#C0392B", lw=2)
    ax.axvline(chosen_threshold, color="gray", ls="--", lw=1.3,
               label=f"Chosen threshold = {chosen_threshold:.2f}")
    ax.set_xlabel("Decision Threshold"); ax.set_ylabel("Score")
    ax.set_title("Precision & Recall vs. Decision Threshold")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "threshold_tuning.png"), dpi=150)
    plt.close()


# =========================================================================
# PHASE 3 (PREP) — SCORE FULL DATASET & BUILD POWER BI STAR SCHEMA
# =========================================================================
def phase3_star_schema(df, model, threshold, feature_names):
    print("=" * 70)
    print("PHASE 3 PREP: Scoring full dataset & building Power BI star schema")
    print("=" * 70)

    feature_cols = [
        "Transaction_Amount", "Account_Balance_Before", "Account_Balance_After",
        "Distance_From_Home_KM", "Customer_Age", "Customer_Tenure_Months",
        "Previous_Fraud_Flags", "Transaction_Hour", "Is_Weekend", "Is_High_Risk_Hour",
        "Balance_Depletion_Ratio", "Amount_to_Balance_Ratio", "Transaction_Risk_Score",
        "Merchant_Volatility_Index",
    ]
    categorical_cols = ["Transaction_Type", "Merchant_Category", "Card_Present",
                         "Device_Type", "Transaction_Country", "Customer_Gender", "Age_Group"]

    X_full = pd.get_dummies(df[feature_cols + categorical_cols], columns=categorical_cols, drop_first=True)
    X_full = X_full.reindex(columns=feature_names, fill_value=0)

    df["Fraud_Probability"] = model.predict_proba(X_full)[:, 1].round(4)
    df["Predicted_Fraud"] = (df["Fraud_Probability"] >= threshold).astype(int)
    df["Prediction_Outcome"] = np.select(
        [
            (df["Is_Fraud"] == 1) & (df["Predicted_Fraud"] == 1),
            (df["Is_Fraud"] == 1) & (df["Predicted_Fraud"] == 0),
            (df["Is_Fraud"] == 0) & (df["Predicted_Fraud"] == 1),
        ],
        ["True Positive", "False Negative", "False Positive"],
        default="True Negative",
    )
    df["Transaction_Date"] = df["Transaction_DateTime"].dt.date
    df.to_csv(os.path.join(OUT_DIR, "Fraud_Data_Scored_Full.csv"), index=False)
    print(f"Saved -> Fraud_Data_Scored_Full.csv. "
          f"Model-flagged fraud rate: {df['Predicted_Fraud'].mean()*100:.2f}%")

    # Dim_Date - continuous calendar
    full_range = pd.date_range(df["Transaction_DateTime"].dt.date.min(),
                                df["Transaction_DateTime"].dt.date.max(), freq="D")
    dim_date = pd.DataFrame({"Date": full_range})
    dim_date["DateKey"] = dim_date["Date"].dt.strftime("%Y%m%d").astype(int)
    dim_date["Year"] = dim_date["Date"].dt.year
    dim_date["Month"] = dim_date["Date"].dt.month
    dim_date["MonthName"] = dim_date["Date"].dt.strftime("%B")
    dim_date["Week"] = dim_date["Date"].dt.isocalendar().week
    dim_date["DayOfWeek"] = dim_date["Date"].dt.strftime("%A")
    dim_date["Is_Weekend"] = dim_date["Date"].dt.dayofweek.isin([5, 6])
    dim_date = dim_date[["DateKey", "Date", "Year", "Month", "MonthName", "Week", "DayOfWeek", "Is_Weekend"]]
    dim_date.to_csv(os.path.join(PBI_DIR, "Dim_Date.csv"), index=False)

    # Dim_Customer
    dim_customer = (
        df.sort_values("Transaction_DateTime")
        .groupby("Customer_ID")
        .agg(Age_Group=("Age_Group", "first"), Customer_Gender=("Customer_Gender", "first"),
             Customer_Tenure_Months=("Customer_Tenure_Months", "max"),
             Previous_Fraud_Flags=("Previous_Fraud_Flags", "max"),
             Transaction_Country=("Transaction_Country", "first"))
        .reset_index()
    )
    dim_customer.to_csv(os.path.join(PBI_DIR, "Dim_Customer.csv"), index=False)

    # Dim_Merchant
    dim_merchant = (
        df.groupby("Merchant_Category")
        .agg(Merchant_Volatility_Index=("Merchant_Volatility_Index", "first"))
        .reset_index()
    )
    dim_merchant.to_csv(os.path.join(PBI_DIR, "Dim_Merchant.csv"), index=False)

    # Fact_Transactions
    fact_cols = ["Transaction_ID", "Customer_ID", "Merchant_Category", "Transaction_Date",
                 "Transaction_Type", "Device_Type", "Card_Present", "Transaction_Amount",
                 "Account_Balance_Before", "Account_Balance_After", "Distance_From_Home_KM",
                 "Transaction_Risk_Score", "Fraud_Probability", "Is_Fraud",
                 "Predicted_Fraud", "Prediction_Outcome"]
    fact_transactions = df[fact_cols].copy()
    fact_transactions["DateKey"] = pd.to_datetime(fact_transactions["Transaction_Date"]).dt.strftime("%Y%m%d").astype(int)
    fact_transactions = fact_transactions.drop(columns=["Transaction_Date"])
    fact_transactions.to_csv(os.path.join(PBI_DIR, "Fact_Transactions.csv"), index=False)

    print(f"Star schema ready in {PBI_DIR}/  "
          f"(Fact_Transactions, Dim_Customer, Dim_Merchant, Dim_Date)\n")


# =========================================================================
if __name__ == "__main__":
    artifacts = phase1_data_engineering()
    model, threshold, feature_names = phase2_model_development(artifacts)
    phase3_star_schema(artifacts["df"], model, threshold, feature_names)
    print("ALL PHASES COMPLETE. See ./solution_output/ for every deliverable.")
