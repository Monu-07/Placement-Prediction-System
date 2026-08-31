from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso, ElasticNet
)
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score
)
from src.data.load_data import load_data
from src.data.preprocess import (
    split_data, identify_features, handle_missing_values,
    standardize_data, one_hot_encode_data, ordinal_encode_data
)

def predict(model, X_test):
    """Generate predictions using the trained model."""
    y_pred = model.predict(X_test)
    return y_pred

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance with MAE, MSE, RMSE, and R²."""
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, y_pred)

    print("\nModel Evaluation Results:")
    print(f"MAE  : {mae:.4f}")
    print(f"MSE  : {mse:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"R²   : {r2:.4f}")

    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2}

if __name__ == "__main__":
    # Load dataset
    df = load_data()

    # Split data (target = SalaryPackage, drop StudentID + PlacementStatus)
    X_train, X_test, y_train, y_test = split_data(
        df,
        target_column="Salary Package",  # <-- exact name with space
        drop_columns=["StudentID", "PlacementStatus"],
        stratify=False  # regression target is continuous
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

    # Train models
    models = {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=0.01),
        "ElasticNet": ElasticNet(alpha=0.01, l1_ratio=0.5)
    }

    results_summary = {}

    for name, model in models.items():
        print("\n----------------------------------------")
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        results = evaluate_model(model, X_test, y_test)
        results_summary[name] = results

    # Final comparison table
    print("\n========================================")
    print("MODEL COMPARISON SUMMARY")
    print("========================================")
    for name, metrics in results_summary.items():
        print(f"{name}: MAE={metrics['MAE']:.4f}, MSE={metrics['MSE']:.4f}, RMSE={metrics['RMSE']:.4f}, R²={metrics['R2']:.4f}")
