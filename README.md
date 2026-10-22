## FastAPI Model Serving Endpoint

### Overview
A machine learning model served through a FastAPI REST API using the Iris dataset and a Random Forest classifier.

### Project Files
- `train_model.py` — trains and saves the model.
- `app.py` — implements the API endpoints.
- `test_app.py` — contains automated tests.
- `requirements.txt` — lists required Python packages.
- `iris_model.joblib` — saved trained model.

### Endpoints
- `GET /` — confirms the API is running.
- `GET /health` — checks model health.
- `POST /predict` — predicts the Iris flower species.

### Run Tests
`pytest -v test_app.py`

### Status
The API and automated tests were successfully tested in Google Colab.Logistic Regression and Classification - AI/ML W3 Assignment
