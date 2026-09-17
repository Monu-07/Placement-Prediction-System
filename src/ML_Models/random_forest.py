from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from src.data.load_data import load_data
from src.data.preprocess import (
    split_data,
    identify_features,
    handle_missing_values,
    one_hot_encode_data,
    ordinal_encode_data
)
import os
import joblib
import matplotlib.pyplot as plt
import numpy as np

def create_model():
    # Random Forest with entropy criterion
    return RandomForestClassifier(
        n_estimators=100,
        criterion="entropy",
        max_depth=10,
        random_state=42,
        oob_score=True
    )

def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    print("\nRandom Forest trained successfully!")
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print("\nAccuracy:", accuracy)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nOOB Score:", model.oob_score_)

    return y_pred

def plot_feature_importances(model, feature_names):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(12, 6))
    plt.bar(range(len(importances)), importances[indices], align="center")
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
    plt.title("Random Forest Feature Importances")
    plt.tight_layout()

    save_path = r"C:\Users\asolo\Documents\ML\Placement-Prediction-System (3)\Placement-Prediction-System\App\static\Charts\random_forest_importances.png"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    print(f"\nFeature importance chart saved at: {save_path}")
    plt.show()

def save_model(model):
    # Save trained model into /data folder
    model_path = r"C:\Users\asolo\Documents\ML\Placement-Prediction-System (3)\Placement-Prediction-System\data\random_forest.pkl"
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    print("\nModel saved successfully at:", model_path)

def main():
    df = load_data()
    print("Original Dataset Shape:", df.shape)

    # Split data
    X_train, X_test, y_train, y_test = split_data(
        df,
        target_column="PlacementStatus",
        drop_columns=["StudentID", "Salary Package", "IsAnomaly"]
    )
    print("Training Dataset Shape:", X_train.shape)
    print("Testing Dataset Shape:", X_test.shape)

    # Identify features
    numerical_features, categorical_features = identify_features(X_train)
    print("Numerical features:", numerical_features)
    print("Categorical features:", categorical_features)

    one_hot_features = ['Gender', 'City', 'Stream', 'Specialisation', 'Hostel', 'HistoryOfBacklogs']
    ordinal_features = ['CollegeTier', 'CGPA_Tier']

    # Preprocessing
    X_train, X_test, imputer = handle_missing_values(X_train, X_test, numerical_features)
    print("Missing values handled.")

    X_train, X_test, one_hot_encoder = one_hot_encode_data(X_train, X_test, one_hot_features)
    print("One-hot encoding completed.")

    X_train, X_test, ordinal_encoder = ordinal_encode_data(X_train, X_test, ordinal_features)
    print("Ordinal encoding completed.")

    # Train and evaluate
    model = create_model()
    model = train_model(model, X_train, y_train)
    y_pred = evaluate_model(model, X_test, y_test)

    # Feature importance plot
    plot_feature_importances(model, X_train.columns)

    # Save model
    save_model(model)

if __name__ == "__main__":
    main()
