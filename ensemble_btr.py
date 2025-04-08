"""
ENSEMBLE-BTR: Multi-Algorithm Banking Transaction Risk Classifier Using Bagging and Boosting Techniques

This implementation uses the Credit Card Fraud Detection dataset to demonstrate:
1. Hybrid Ensemble Architecture: RF (bagging) + XGBoost (boosting) with manual ensemble integration
2. Banking-Specific Feature Engineering: Transaction patterns and temporal features
3. Algorithm Optimization: SMOTE for class imbalance and hyperparameter tuning
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import warnings
import time
import os
import joblib

# Suppress warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

print("ENSEMBLE-BTR: Banking Transaction Risk Classifier")
print("Loading and preparing data...")

# Load the Credit Card Fraud Detection dataset
df = pd.read_csv('creditcard.csv')

# Display basic information about the dataset
print(f"\nDataset Shape: {df.shape}")
print(f"Number of Fraudulent Transactions: {df['Class'].sum()}")
print(f"Percentage of Fraudulent Transactions: {df['Class'].mean() * 100:.4f}%")

# Basic data exploration
print("\nData Overview:")
print(df.describe().T)

# Check for missing values
missing_values = df.isnull().sum()
if missing_values.sum() > 0:
    print("\nMissing Values:")
    print(missing_values[missing_values > 0])
else:
    print("\nNo missing values found in the dataset.")

# Feature Engineering
print("\nPerforming feature engineering...")

# 1. Create time-based features
df['Hour'] = df['Time'] / 3600  # Convert seconds to hours
df['Hour'] = df['Hour'] % 24  # Ensure hours are within 0-24 range

# 2. Create amount-based features
df['Amount_Scaled'] = StandardScaler().fit_transform(df[['Amount']])
df['Amount_Log'] = np.log1p(df['Amount'])  # Log transformation to handle skewness

# 3. Create transaction velocity features (simulated)
# Group by hour and calculate transaction frequency
hour_groups = df.groupby(['Hour']).size().reset_index(name='Transactions_Per_Hour')
df = pd.merge(df, hour_groups, on='Hour', how='left')

# 4. Create statistical features based on amount
amount_stats = df.groupby(['Hour'])['Amount'].agg(['mean', 'std', 'max']).reset_index()
amount_stats.columns = ['Hour', 'Hour_Mean_Amount', 'Hour_Std_Amount', 'Hour_Max_Amount']
df = pd.merge(df, amount_stats, on='Hour', how='left')

# 5. Create ratio features
df['Amount_to_Mean_Ratio'] = df['Amount'] / df['Hour_Mean_Amount']
df['Amount_to_Max_Ratio'] = df['Amount'] / df['Hour_Max_Amount']

# Fill NaN values that might have been created
df = df.fillna(0)

# Prepare features and target
print("\nPreparing features and target...")
# Drop original Time column as we've created Hour feature
X = df.drop(['Class', 'Time'], axis=1)
y = df['Class']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

print(f"Training set shape: {X_train.shape}")
print(f"Testing set shape: {X_test.shape}")

# Handle class imbalance using SMOTE
print("\nApplying SMOTE to handle class imbalance...")
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

print(f"Original training set class distribution: {pd.Series(y_train).value_counts()}")
print(f"Resampled training set class distribution: {pd.Series(y_train_resampled).value_counts()}")

# Create directories for outputs
if not os.path.exists('visualizations'):
    os.makedirs('visualizations')
if not os.path.exists('models'):
    os.makedirs('models')

# Model Implementation
print("\nImplementing models...")

# 1. Random Forest (Bagging)
print("\nTraining Random Forest (Bagging) model...")
start_time = time.time()

rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=10,
    min_samples_leaf=4,
    bootstrap=True,
    oob_score=True,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train_resampled, y_train_resampled)
rf_train_time = time.time() - start_time
print(f"Random Forest training time: {rf_train_time:.2f} seconds")

# Make predictions
rf_preds = rf_model.predict(X_test)
rf_probs = rf_model.predict_proba(X_test)[:, 1]

# Evaluate
print("\nRandom Forest (Bagging) Performance:")
print(classification_report(y_test, rf_preds))
fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_probs)
rf_auc = auc(fpr_rf, tpr_rf)
print(f"ROC-AUC: {rf_auc:.4f}")

# 2. XGBoost (Boosting)
print("\nTraining XGBoost (Boosting) model...")
start_time = time.time()

xgb_model = xgb.XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='binary:logistic',
    random_state=42,
    n_jobs=-1
)

xgb_model.fit(X_train_resampled, y_train_resampled)
xgb_train_time = time.time() - start_time
print(f"XGBoost training time: {xgb_train_time:.2f} seconds")

# Make predictions
xgb_preds = xgb_model.predict(X_test)
xgb_probs = xgb_model.predict_proba(X_test)[:, 1]

# Evaluate
print("\nXGBoost (Boosting) Performance:")
print(classification_report(y_test, xgb_preds))
fpr_xgb, tpr_xgb, _ = roc_curve(y_test, xgb_probs)
xgb_auc = auc(fpr_xgb, tpr_xgb)
print(f"ROC-AUC: {xgb_auc:.4f}")

# 3. Manual Ensemble
print("\nTraining Manual Ensemble model...")
start_time = time.time()

# Get predictions from base models
rf_train_probs = rf_model.predict_proba(X_train_resampled)[:, 1]
xgb_train_probs = xgb_model.predict_proba(X_train_resampled)[:, 1]

# Create a new feature matrix with the predictions
X_train_meta = np.column_stack([rf_train_probs, xgb_train_probs])

# Train a meta-learner
meta_learner = LogisticRegression(max_iter=1000)
meta_learner.fit(X_train_meta, y_train_resampled)

# For test data
rf_test_probs = rf_model.predict_proba(X_test)[:, 1]
xgb_test_probs = xgb_model.predict_proba(X_test)[:, 1]
X_test_meta = np.column_stack([rf_test_probs, xgb_test_probs])

# Make predictions
manual_probs = meta_learner.predict_proba(X_test_meta)[:, 1]
manual_preds = meta_learner.predict(X_test_meta)

manual_train_time = time.time() - start_time
print(f"Manual Ensemble training time: {manual_train_time:.2f} seconds")

# Evaluate
print("\nManual Ensemble Performance:")
print(classification_report(y_test, manual_preds))
fpr_manual, tpr_manual, _ = roc_curve(y_test, manual_probs)
manual_auc = auc(fpr_manual, tpr_manual)
print(f"ROC-AUC: {manual_auc:.4f}")

# Model Comparison and Visualization
print("\nGenerating visualizations...")

# 1. ROC Curve Comparison
plt.figure(figsize=(10, 8))
# Plot Random Forest ROC
plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {rf_auc:.3f})')

# Plot XGBoost ROC
plt.plot(fpr_xgb, tpr_xgb, label=f'XGBoost (AUC = {xgb_auc:.3f})')

# Plot Manual Ensemble ROC
plt.plot(fpr_manual, tpr_manual, label=f'Manual Ensemble (AUC = {manual_auc:.3f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison')
plt.legend()
plt.savefig('visualizations/roc_comparison.png')
plt.close()

# 2. Precision-Recall Curve Comparison
plt.figure(figsize=(10, 8))
# Plot Random Forest PR curve
precision_rf, recall_rf, _ = precision_recall_curve(y_test, rf_probs)
plt.plot(recall_rf, precision_rf, label=f'Random Forest')

# Plot XGBoost PR curve
precision_xgb, recall_xgb, _ = precision_recall_curve(y_test, xgb_probs)
plt.plot(recall_xgb, precision_xgb, label=f'XGBoost')

# Plot Manual Ensemble PR curve
precision_manual, recall_manual, _ = precision_recall_curve(y_test, manual_probs)
plt.plot(recall_manual, precision_manual, label=f'Manual Ensemble')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve Comparison')
plt.legend()
plt.savefig('visualizations/pr_comparison.png')
plt.close()

# 3. Confusion Matrix for Manual Ensemble
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, manual_preds)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Manual Ensemble')
plt.savefig('visualizations/confusion_matrix.png')
plt.close()

# 4. Feature Importance Analysis
# Random Forest feature importance
plt.figure(figsize=(12, 8))
feature_importance = rf_model.feature_importances_
sorted_idx = np.argsort(feature_importance)
plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx], align='center')
plt.yticks(range(len(sorted_idx)), np.array(X_train.columns)[sorted_idx])
plt.title('Random Forest Feature Importance')
plt.savefig('visualizations/rf_feature_importance.png')
plt.close()

# XGBoost feature importance
plt.figure(figsize=(12, 8))
xgb.plot_importance(xgb_model, max_num_features=15)
plt.title('XGBoost Feature Importance')
plt.savefig('visualizations/xgb_feature_importance.png')
plt.close()

# 5. Meta-learner coefficients (for Manual Ensemble)
plt.figure(figsize=(8, 6))
coef_names = ['Random Forest', 'XGBoost']
plt.bar(coef_names, meta_learner.coef_[0])
plt.title('Manual Ensemble - Meta-learner Coefficients')
plt.ylabel('Coefficient Value')
plt.savefig('visualizations/meta_learner_coefficients.png')
plt.close()

# 6. Try SHAP values for model interpretability (if available)
try:
    import shap
    
    # Sample a subset of test data for SHAP analysis (for speed)
    X_test_sample = X_test.sample(min(1000, len(X_test)), random_state=42)
    
    # SHAP for XGBoost
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_test_sample)
    
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_test_sample, plot_type="bar", show=False)
    plt.title('SHAP Feature Importance')
    plt.tight_layout()
    plt.savefig('visualizations/shap_summary.png')
    plt.close()
    
    # SHAP decision plot for a few samples
    plt.figure(figsize=(12, 8))
    shap.decision_plot(explainer.expected_value, shap_values[:10], X_test_sample.iloc[:10], show=False)
    plt.title('SHAP Decision Plot')
    plt.tight_layout()
    plt.savefig('visualizations/shap_decision_plot.png')
    plt.close()
except Exception as e:
    print(f"SHAP visualization error: {e}")
    print("Skipping SHAP visualizations.")

# Save models for future use
print("\nSaving models...")
os.makedirs('models', exist_ok=True)
joblib.dump(rf_model, 'models/random_forest_model.pkl')
joblib.dump(xgb_model, 'models/xgboost_model.pkl')
joblib.dump(meta_learner, 'models/meta_learner_model.pkl')

# Save the manual ensemble pipeline as a function
with open('models/manual_ensemble_function.py', 'w') as f:
    f.write("""
def predict_with_manual_ensemble(rf_model, xgb_model, meta_learner, X):
    \"\"\"
    Make predictions using the manual ensemble model.
    
    Parameters:
    -----------
    rf_model : RandomForestClassifier
        Trained Random Forest model
    xgb_model : XGBClassifier
        Trained XGBoost model
    meta_learner : LogisticRegression
        Trained meta-learner model
    X : array-like
        Features to predict on
        
    Returns:
    --------
    predictions : array
        Binary predictions (0 or 1)
    probabilities : array
        Probability of class 1
    \"\"\"
    import numpy as np
    
    # Get base model predictions
    rf_probs = rf_model.predict_proba(X)[:, 1]
    xgb_probs = xgb_model.predict_proba(X)[:, 1]
    
    # Stack predictions
    X_meta = np.column_stack([rf_probs, xgb_probs])
    
    # Get meta-learner predictions
    probabilities = meta_learner.predict_proba(X_meta)[:, 1]
    predictions = meta_learner.predict(X_meta)
    
    return predictions, probabilities
""")

print("\nModel comparison summary:")
print(f"{'Model':<20} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'AUC':<10}")
print("-" * 70)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Random Forest metrics
rf_accuracy = accuracy_score(y_test, rf_preds)
rf_precision = precision_score(y_test, rf_preds)
rf_recall = recall_score(y_test, rf_preds)
rf_f1 = f1_score(y_test, rf_preds)
print(f"{'Random Forest':<20} {rf_accuracy:.4f}     {rf_precision:.4f}     {rf_recall:.4f}     {rf_f1:.4f}     {rf_auc:.4f}")

# XGBoost metrics
xgb_accuracy = accuracy_score(y_test, xgb_preds)
xgb_precision = precision_score(y_test, xgb_preds)
xgb_recall = recall_score(y_test, xgb_preds)
xgb_f1 = f1_score(y_test, xgb_preds)
print(f"{'XGBoost':<20} {xgb_accuracy:.4f}     {xgb_precision:.4f}     {xgb_recall:.4f}     {xgb_f1:.4f}     {xgb_auc:.4f}")

# Manual Ensemble metrics
manual_accuracy = accuracy_score(y_test, manual_preds)
manual_precision = precision_score(y_test, manual_preds)
manual_recall = recall_score(y_test, manual_preds)
manual_f1 = f1_score(y_test, manual_preds)
print(f"{'Manual Ensemble':<20} {manual_accuracy:.4f}     {manual_precision:.4f}     {manual_recall:.4f}     {manual_f1:.4f}     {manual_auc:.4f}")

print("\nENSEMBLE-BTR implementation complete!")
print("Visualizations saved in the 'visualizations' directory.")
print("Models saved in the 'models' directory.")
