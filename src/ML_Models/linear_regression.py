import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import joblib
import os

def load_dataset():
    # Load dataset
    df = pd.read_csv(
        r"C:\Users\asolo\Documents\ML\Placement-Prediction-System\data\placement_data.csv"
    )
    return df

def create_model():
    return LinearRegression()

def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    print("\nLinear Regression model trained successfully!")
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print(f"\nIntercept (b0): {model.intercept_:.2f}")
    print(f"Slope (b1): {model.coef_[0]:.2f}")
    print(f"\nMean Squared Error (MSE): {mse:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"R² Score: {r2:.2f}")

    return y_pred

def plot_results(X_test, y_test, y_pred):
    plt.scatter(X_test, y_test, label="Actual", color="blue")
    plt.plot(X_test, y_pred, label="Predicted", color="red", linewidth=2)
    plt.xlabel("CGPA")
    plt.ylabel("Salary Package")
    plt.title("Linear Regression: CGPA vs Salary Package")
    plt.legend()
    plt.tight_layout()

    save_path = r"C:\Users\asolo\Documents\ML\Placement-Prediction-System\App\static\Charts\linear_regression.png"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    print(f"\nRegression plot saved at: {save_path}")
    plt.show()

def save_model(model):
    model_path = r"C:\Users\asolo\Documents\ML\Placement-Prediction-System\data\linear_regression.pkl"
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    print(f"\nModel saved successfully at: {model_path}")

def main():
    df = load_dataset()

    # Features and target
    X = df[['CGPA']]
    y = df['Salary Package']

    # Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train and evaluate
    model = create_model()
    model = train_model(model, X_train, y_train)
    y_pred = evaluate_model(model, X_test, y_test)

    # Plot results
    plot_results(X_test, y_test, y_pred)

    # Predict for CGPA = 9.6
    cgpa_value = pd.DataFrame([[9.6]], columns=["CGPA"])
    predicted_salary = model.predict(cgpa_value)
    print(f"\nPredicted Salary Package for CGPA 9.6: {predicted_salary[0]:.2f}")

    # Save model
    save_model(model)

if __name__ == "__main__":
    main()
