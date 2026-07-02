import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Load Dataset
housing = fetch_california_housing()

X = pd.DataFrame(housing.data, columns=housing.feature_names)
y = housing.target

# Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# Feature Scaling
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build Model
svr_model = SVR(
    kernel='rbf',
    C=100,
    epsilon=0.1,
    gamma='scale'
)

# Train Model
svr_model.fit(X_train_scaled, y_train)

# Prediction
y_pred = svr_model.predict(X_test_scaled)

# Evaluation
print("=" * 40)
print("Support Vector Regression Results")
print("=" * 40)

print("R² Score :", round(r2_score(y_test, y_pred), 4))

mse = mean_squared_error(y_test, y_pred)
print("MSE      :", round(mse, 4))
print("RMSE     :", round(np.sqrt(mse), 4))
print("MAE      :", round(mean_absolute_error(y_test, y_pred), 4))

# Compare Actual vs Predicted
comparison = pd.DataFrame({
    "Actual": y_test,
    "Predicted": y_pred
})

print("\nFirst 10 Predictions")
print(comparison.head(10))

# Plot
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("SVR - Actual vs Predicted House Prices")
plt.show()
