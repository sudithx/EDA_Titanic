# Model Serialisation — Joblib and Pickle

## Objective
To save and reload a trained machine-learning model using pickle and joblib.

## Dataset
Iris dataset from scikit-learn.

## Implementation
- Loaded and split the Iris dataset.
- Trained a Random Forest classifier.
- Saved the trained model using pickle and joblib.
- Loaded both serialized models.
- Compared predictions with the original model.
- Ran automated tests to verify correctness.

## Testing
The tests verify that both model files exist and that the loaded models produce the same predictions as the original model.

## Conclusion
Model serialisation allows a trained model to be reused without retraining. Both pickle and joblib were tested.
