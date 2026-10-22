
import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

def train_and_save(path="iris_model.joblib"):
    iris = load_iris()
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(iris.data, iris.target)

    joblib.dump({
        "model": model,
        "target_names": iris.target_names.tolist()
    }, path)

    return model

if __name__ == "__main__":
    train_and_save()
    print("Model trained and saved.")
