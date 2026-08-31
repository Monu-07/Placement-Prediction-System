import pandas as pd
import joblib
import os
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

def load_preprocessed_data():
    # Go three levels up from src/ML_Models to reach project root
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    train_path = os.path.join(base_dir, "data", "preprocessed_train.csv")
    test_path = os.path.join(base_dir, "data", "preprocessed_test.csv")

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError(
            f"Preprocessed files not found in {base_dir}/data. "
            "Run preprocess.py first to generate them."
        )

    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    return train_data, test_data

def split_features_target(train_data, test_data):
    x_train = train_data.drop(columns=["PlacementStatus"])
    y_train = train_data["PlacementStatus"]
    x_test = test_data.drop(columns=["PlacementStatus"])
    y_test = test_data["PlacementStatus"]
    return x_train, y_train, x_test, y_test

def create_model():
    return LogisticRegression(max_iter=1000)

def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print("\nAccuracy:")
    print(model.score(X_test, y_test))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

def save_model(model):
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    models_dir = os.path.join(base_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, "logistic_regression.pkl")
    joblib.dump(model, model_path)
    print("\nModel saved successfully at:")
    print(model_path)

def load_model():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    model_path = os.path.join(base_dir, "models", "logistic_regression.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError("Saved model not found. Train and save it first.")
    return joblib.load(model_path)

if __name__ == "__main__":
    train_data, test_data = load_preprocessed_data()
    print("Training data shape:", train_data.shape)
    print("Test data shape:", test_data.shape)

    x_train, y_train, x_test, y_test = split_features_target(train_data, test_data)
    print("\nX_train Shape:", x_train.shape)
    print("y_train Shape:", y_train.shape)
    print("X_test Shape:", x_test.shape)
    print("y_test Shape:", y_test.shape)

    model = create_model()
    print("\nLogistic Regression Model created.")
    model = train_model(model, x_train, y_train)
    print("\nLogistic Regression Model trained.")
    evaluate_model(model, x_test, y_test)
    save_model(model)
