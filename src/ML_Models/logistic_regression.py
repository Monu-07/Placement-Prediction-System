import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os

def run_logistic_regression(csv_file_path=None, target_column=None):
    # 1. Load or Generate Dataset
    if csv_file_path and target_column:
        print(f"Loading data from {csv_file_path}...")
        df = pd.read_csv(csv_file_path)
        X = df.drop(columns=[target_column])
        y = df[target_column]
    else:
        print("Using synthetic dataset...")
        from sklearn.datasets import make_classification
        X, y = make_classification(
            n_samples=1000,
            n_features=5,
            n_informative=3,
            n_redundant=0,
            random_state=42
        )

    # 2. Split into Train and Test Sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3. Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 4. Train the Model
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_scaled, y_train)

    # 5. Evaluate the Model
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)

    print("\n" + "=" * 40)
    print(f"Accuracy: {acc * 100:.2f}%")
    print("=" * 40)
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    # Confusion Matrix Heatmap
    plt.figure(figsize=(6, 4))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Class 0", "Class 1"],
                yticklabels=["Class 0", "Class 1"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()

    # 6. Save Model and Scaler
    model_path = r"C:\Users\asolo\Documents\ML\Placement-Prediction-System\data\logistic_regression_model.pkl"
    scaler_path = r"C:\Users\asolo\Documents\ML\Placement-Prediction-System\data\scaler.pkl"
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    print(f"\nModel saved at: {model_path}")
    print(f"Scaler saved at: {scaler_path}")

    return model, scaler

def load_model_and_scaler():
    model_path = r"C:\Users\asolo\Documents\ML\Placement-Prediction-System\data\logistic_regression_model.pkl"
    scaler_path = r"C:\Users\asolo\Documents\ML\Placement-Prediction-System\data\scaler.pkl"
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        raise FileNotFoundError("Saved model or scaler not found. Train and save them first.")
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

def predict_new_sample(model, scaler, sample_df):
    sample_scaled = scaler.transform(sample_df)
    prediction = model.predict(sample_scaled)
    print("\nPrediction for new sample:", prediction[0])
    return prediction[0]

if __name__ == "__main__":
    # Train and save
    model, scaler = run_logistic_regression()

    # Reload and predict on a new sample
    loaded_model, loaded_scaler = load_model_and_scaler()
    new_sample = pd.DataFrame([[8.5, 1, 0, 0, 1]],
                              columns=["CGPA", "Gender", "Stream", "Hostel", "HistoryOfBacklogs"])
    predict_new_sample(loaded_model, loaded_scaler, new_sample)
