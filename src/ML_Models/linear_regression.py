import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv(r"C:\Users\asolo\Documents\ML\Placement-Prediction-System\data\placement_data.csv")

# Features and target
X = df[['CGPA']]                 # Feature
y = df['Salary Package']         # Target column values

# Split data
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
linear_regression = LinearRegression()
linear_regression.fit(x_train, y_train)

# Predictions
y_pred = linear_regression.predict(x_test)

# Evaluation
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print("MSE:", mse)
print("RMSE:", rmse)

cgpa_value = pd.DataFrame([[9.6]], columns=["CGPA"])
predicted_salary = linear_regression.predict(cgpa_value)
print("Predicted Salary Package for CGPA 9.6:", predicted_salary[0])

