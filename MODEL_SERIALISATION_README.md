# Model Serialisation — Joblib and Pickle

## Overview
This project demonstrates model serialisation and deserialisation using Python's native `pickle` library and scikit-learn's preferred `joblib` library on a Random Forest classifier trained on the Iris dataset.

## Implementation Details
1. **Dataset:** Iris classification dataset (120 training samples, 30 testing samples, 4 features).
2. **Model:** `RandomForestClassifier(n_estimators=100, random_state=42)`.
3. **Serialisation Methods:**
   - **Pickle:** Serialises the trained estimator into `iris_model.pkl`.
   - **Joblib:** Serialises the trained estimator into `iris_model.joblib`.

## Results & Verification
- **Original Model Accuracy:** `0.9000`
- **Pickle Model Accuracy:** `0.9000` (Predictions match original: `True`)
- **Joblib Model Accuracy:** `0.9000` (Predictions match original: `True`)
- Automated assertions verified model existence, prediction equality, and accuracy metrics.
