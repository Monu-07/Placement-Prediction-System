from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from src.data.load_data import load_data
from src.data.preprocess import (
    split_data,
    identify_features,
    handle_missing_values,
    standardize_data,
    one_hot_encode_data,
    ordinal_encode_data
)
import joblib
import os


def create_model():
    model = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )
    return model


def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    print("\nGradient Boosting Model Trained Successfully!")
    return model


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print("\n" + "=" * 40)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print("=" * 40)
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    return y_pred


def main():
    df = load_data()

    print("Original Dataset Shape:")
    print(df.shape)

    X_train, X_test, y_train, y_test = split_data(
        df,
        target_column="PlacementStatus",
        drop_columns=[
            "StudentID",
            "Salary Package",
            "IsAnomaly"
        ]
    )

    print("\nTraining Shape:")
    print(X_train.shape)
    print("\nTesting Shape:")
    print(X_test.shape)

    numerical_features, categorical_features = identify_features(X_train)
    print("\nNumerical Features:", numerical_features)
    print("Categorical Features:", categorical_features)

    one_hot_features = [
        "Gender", "City", "Stream", "Specialisation", "Hostel", "HistoryOfBacklogs"
    ]
    ordinal_features = [
        "CollegeTier", "CGPA_Tier"
    ]

    X_train, X_test, imputer = handle_missing_values(
        X_train, X_test, numerical_features
    )
    print("\nMissing values handled.")

    X_train, X_test, scaler = standardize_data(
        X_train, X_test, numerical_features
    )
    print("Standardization completed.")

    X_train, X_test, one_hot_encoder = one_hot_encode_data(
        X_train, X_test, one_hot_features
    )
    print("One-hot encoding completed.")

    X_train, X_test, ordinal_encoder = ordinal_encode_data(
        X_train, X_test, ordinal_features
    )
    print("Ordinal encoding completed.")

    model = create_model()
    model = train_model(model, X_train, y_train)
    evaluate_model(model, X_test, y_test)

    # Feature importance (top 10)
    importances = sorted(
        zip(X_train.columns, model.feature_importances_),
        key=lambda pair: pair[1],
        reverse=True
    )
    print("\nTop 10 Most Important Features:")
    for feature, importance in importances[:10]:
        print(f"  {feature}: {importance:.4f}")

    # Save the trained model next to the other model artifacts
    data_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "data"
    )
    model_path = os.path.join(data_dir, "gradient_boosting_model.pkl")
    joblib.dump(model, model_path)
    print(f"\nModel saved to {model_path}")

    return model


if __name__ == "__main__":
    main()