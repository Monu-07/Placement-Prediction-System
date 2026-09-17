from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet
)
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from src.data.load_data import load_data   # ✅ fixed import
from src.data.preprocess import (
    split_data,
    identify_features,
    handle_missing_values,
    standardize_data,
    one_hot_encode_data,
    ordinal_encode_data
)

def create_models():
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Lasso Regression": Lasso(alpha=0.01),
        "ElasticNet Regression": ElasticNet(alpha=0.01, l1_ratio=0.5)
    }
    return models

def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model

def predict(model, X_test):
    return model.predict(X_test)

def evaluate(y_test, y_pred):
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, y_pred)
    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2}

def main():
    # Load dataset
    df = load_data()
    print("\nOriginal Dataset Shape:", df.shape)

    # Split data (target = Salary Package, drop StudentID + PlacementStatus)
    X_train, X_test, y_train, y_test = split_data(
        df,
        target_column="Salary Package",
        drop_columns=["StudentID", "PlacementStatus"],
        stratify=False
    )

    # Identify features
    numerical_features, categorical_features = identify_features(X_train)

    # Define feature groups
    one_hot_features = ["Gender", "City", "Stream", "Specialisation", "Hostel", "HistoryOfBacklogs"]
    ordinal_features = ["CollegeTier", "CGPA_Tier"]

    # Preprocessing pipeline
    X_train, X_test, imputer = handle_missing_values(X_train, X_test, numerical_features)
    X_train, X_test, scaler = standardize_data(X_train, X_test, numerical_features)
    X_train, X_test, one_hot_encoder = one_hot_encode_data(X_train, X_test, one_hot_features)
    X_train, X_test, ordinal_encoder = ordinal_encode_data(X_train, X_test, ordinal_features)

    # Train and evaluate models
    models = create_models()
    results_summary = {}

    for name, model in models.items():
        print("\n----------------------------------------")
        print(f"Training {name}...")
        model = train_model(model, X_train, y_train)
        y_pred = predict(model, X_test)
        results = evaluate(y_test, y_pred)
        results_summary[name] = results

        print(f"{name} Results: MAE={results['MAE']:.4f}, "
              f"MSE={results['MSE']:.4f}, RMSE={results['RMSE']:.4f}, "
              f"R²={results['R2']:.4f}")

    # Final comparison summary
    print("\n========================================")
    print("MODEL COMPARISON SUMMARY")
    print("========================================")
    for name, metrics in results_summary.items():
        print(f"{name}: MAE={metrics['MAE']:.4f}, "
              f"MSE={metrics['MSE']:.4f}, RMSE={metrics['RMSE']:.4f}, "
              f"R²={metrics['R2']:.4f}")

if __name__ == "__main__":
    main()
