from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from src.data.load_data import load_data
from src.data.preprocess import (
    split_data,
    identify_features,
    handle_missing_values,
    one_hot_encode_data,
    ordinal_encode_data
)
import matplotlib.pyplot as plt
import os
import joblib

def create_model():
    # Random Tree using random splitter
    return DecisionTreeClassifier(
        criterion="entropy",
        splitter="random",   # ✅ random splitter makes it a Random Tree
        max_depth=5,
        random_state=42
    )

def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    print("Random Tree trained successfully")
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("\nAccuracy:", accuracy)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    return y_pred

def display_tree(model, feature_names):
    plt.figure(figsize=(25, 12))
    plot_tree(
        model,
        feature_names=feature_names,
        class_names=["Not placed", "Placed"],
        filled=True,
        rounded=True,
        fontsize=8,
    )
    plt.title("Random Tree - Placement Prediction")
    plt.show()

def save_model(model):
    # ✅ Save directly into models folder
    models_dir = r"C:\Users\asolo\Documents\ML\Placement-Prediction-System (3)\Placement-Prediction-System\models"
    os.makedirs(models_dir, exist_ok=True)

    model_path = os.path.join(models_dir, "random_forest.pkl")
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

    # Display tree
    display_tree(model, X_train.columns)

    # Save model
    save_model(model)

if __name__ == "__main__":
    main()
