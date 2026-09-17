import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import graphviz
import os

# Custom imports from your project
from src.data.load_data import load_data
from src.data.preprocess import (
    split_data,
    identify_features,
    handle_missing_values,
    one_hot_encode_data,
    ordinal_encode_data,
)

# -----------------------------
# Model Creation
# -----------------------------
def create_model():
    model = DecisionTreeClassifier(
        criterion="entropy",  # ID3-style splitting
        max_depth=5,
        random_state=42,
    )
    return model

# -----------------------------
# Training
# -----------------------------
def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    print("\nDecision Tree trained successfully")
    return model

# -----------------------------
# Evaluation
# -----------------------------
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print("\nAccuracy:", accuracy)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    return y_pred

# -----------------------------
# Visualization (Matplotlib)
# -----------------------------
def display_tree(model, feature_names):
    plt.figure(figsize=(25, 12))
    tree.plot_tree(
        model,
        feature_names=feature_names,
        class_names=["Not placed", "Placed"],
        filled=True,
        rounded=True,
        fontsize=8,
    )
    plt.title("ID3 Decision Tree - Placement Prediction")

    # Save to Charts directory
    save_path = r"C:\Users\asolo\Documents\ML\Placement-Prediction-System (3)\Placement-Prediction-System\App\static\Charts\decision_tree.png"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    print(f"\nDecision tree saved at: {save_path}")
    plt.show()

# -----------------------------
# Visualization (Graphviz)
# -----------------------------
def export_tree_graphviz(model, feature_names, class_names):
    dot_data = export_graphviz(
        model,
        out_file=None,
        feature_names=feature_names,
        class_names=class_names,
        filled=True,
        rounded=True,
        special_characters=True,
    )
    graph = graphviz.Source(dot_data)

    # Save to data directory
    save_path = r"C:\Users\asolo\Documents\ML\Placement-Prediction-System (3)\Placement-Prediction-System\data\decision_tree_graphviz"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    graph.render(save_path, format="png", cleanup=True)
    print(f"\nGraphviz tree saved at: {save_path}.png")

# -----------------------------
# Main Pipeline
# -----------------------------
def main():
    # 1. Load data
    df = load_data()
    print("Original Dataset Shape:", df.shape)

    # 2. Split data
    X_train, X_test, y_train, y_test = split_data(
        df,
        target_column="PlacementStatus",
        drop_columns=["StudentID", "Salary Package", "IsAnomaly"],
    )
    print("Train Shape:", X_train.shape)
    print("Test Shape:", X_test.shape)

    # 3. Identify features
    numerical_features, categorical_features = identify_features(X_train)
    print("Numerical features:", numerical_features)
    print("Categorical features:", categorical_features)

    one_hot_features = ["Gender", "City", "Stream", "Specialisation", "Hostel", "HistoryOfBacklogs"]
    ordinal_features = ["CollegeTier", "CGPA_Tier"]

    # 4. Preprocessing
    X_train, X_test, imputer = handle_missing_values(X_train, X_test, numerical_features)
    print("Missing values handled.")

    X_train, X_test, one_hot_encoder = one_hot_encode_data(X_train, X_test, one_hot_features)
    print("One-hot encoding completed.")

    X_train, X_test, ordinal_encoder = ordinal_encode_data(X_train, X_test, ordinal_features)
    print("Ordinal encoding completed.")

    # 5. Train model
    model = create_model()
    model = train_model(model, X_train, y_train)

    # 6. Evaluate model
    y_pred = evaluate_model(model, X_test, y_test)

    # 7. Visualize tree
    display_tree(model, X_train.columns)
    export_tree_graphviz(model, X_train.columns, ["Not placed", "Placed"])

    # 8. Save model
    model_path = r"C:\Users\asolo\Documents\ML\Placement-Prediction-System (3)\Placement-Prediction-System\data\decision_tree.pkl"
    joblib.dump(model, model_path)
    print(f"\nModel saved at: {model_path}")

if __name__ == "__main__":
    main()
