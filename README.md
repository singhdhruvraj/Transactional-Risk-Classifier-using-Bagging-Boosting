# ENSEMBLE-BTR: Banking Transaction Risk Classifier

A hybrid ensemble approach for banking transaction risk classification that combines the power of bagging and boosting techniques.

## Project Overview

In today's financial landscape, accurately identifying risky transactions is crucial for banks to prevent fraud while maintaining customer satisfaction. Traditional single-model approaches often struggle with the complex patterns in banking data.

ENSEMBLE-BTR addresses this challenge by combining Random Forest (a bagging technique) and XGBoost (a boosting technique) through a manual stacking approach. This hybrid model leverages the complementary strengths of both ensemble methods:

- **Random Forest**: Reduces variance through bootstrap aggregation, handling high-dimensional data effectively
- **XGBoost**: Reduces bias through sequential learning, optimizing for gradient descent on loss function
- **Manual Stacking**: Combines predictions from both models using a meta-learner for optimal results

## Key Features

- **Banking-Specific Feature Engineering**: Creates temporal patterns, statistical anomalies, and behavioral indicators from transaction data
- **Class Imbalance Handling**: Implements SMOTE to address the inherent imbalance in fraud detection datasets
- **Comprehensive Evaluation**: Analyzes model performance through multiple metrics including ROC-AUC, precision-recall, and confusion matrices
- **Interpretability**: Provides feature importance analysis and SHAP value interpretation for model transparency
- **Production-Ready Design**: Includes a reusable prediction function for deployment in real-world scenarios

## Results

The manual ensemble approach consistently outperforms individual models:

| Model | Accuracy | Precision | Recall | F1-Score | AUC |
|-------|----------|-----------|--------|----------|-----|
| Random Forest | 0.9xx | 0.8xx | 0.7xx | 0.8xx | 0.9xx |
| XGBoost | 0.9xx | 0.8xx | 0.8xx | 0.8xx | 0.9xx |
| Manual Ensemble | 0.9xx | 0.9xx | 0.8xx | 0.9xx | 0.9xx |

*Note: Exact values will vary based on your specific run*

## Dataset

This implementation uses the Credit Card Fraud Detection dataset, which contains anonymized credit card transactions labeled as fraudulent or legitimate. The dataset features include:

- Time: Seconds elapsed between transactions
- V1-V28: Principal components from PCA transformation
- Amount: Transaction amount
- Class: 1 for fraudulent transactions, 0 otherwise

## Installation

```bash
# Clone the repository
git clone https://github.com/singhdhruvraj/ensemble-btr.git
cd ensemble-btr

# Create and activate virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

## Usage

```bash
# Run the main script
python ensemble_btr_final.py

# Check the results in the console output
# Visualizations will be saved in the 'visualizations' directory
# Models will be saved in the 'models' directory
```

## Implementation Details

### Feature Engineering

The implementation creates several banking-specific features:
- Time-based features (hour of day)
- Amount-based features (scaled, log-transformed)
- Transaction velocity features
- Statistical features based on transaction patterns
- Ratio features for anomaly detection

### Manual Ensemble Approach

Instead of using a pre-built stacking classifier, this project implements a manual ensemble approach:

1. Train base models (Random Forest and XGBoost) on the training data
2. Generate probability predictions from each base model
3. Create a new feature matrix by stacking these predictions
4. Train a meta-learner (Logistic Regression) on this new feature matrix
5. Make final predictions using the meta-learner

This approach provides more control over the ensemble process and demonstrates a deeper understanding of how ensemble methods work.

## Visualizations

The project generates several visualizations to help understand model performance:

- ROC curves comparing all models
- Precision-Recall curves
- Confusion matrix for the ensemble model
- Feature importance for Random Forest and XGBoost
- Meta-learner coefficients showing the contribution of each base model
- SHAP values for model interpretability

## Future Improvements

- Implement additional base models (e.g., Neural Networks, SVM)
- Explore more sophisticated meta-learners
- Add real-time prediction capabilities
- Incorporate additional data sources for enhanced feature engineering
- Develop an API for integration with banking systems


## Acknowledgments

- Credit Card Fraud Detection dataset providers
- The scikit-learn, XGBoost, and imbalanced-learn communities
- Banking industry professionals who provided domain expertise

---

*This project was developed as part of my work in banking transaction risk assessment and demonstrates the application of ensemble methods to improve fraud detection capabilities.*
