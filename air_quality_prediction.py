# air_quality_prediction.py
# Author: Nazarine Wasonga
# Project: AI-SDG13-Air-Quality
# Description: Predict Air Quality Index (AQI) using Linear Regression for SDG 13 - Climate Action

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# -------------------------------
# 1. Load dataset
# -------------------------------
data = pd.read_csv('https://raw.githubusercontent.com/datasets/air-quality/master/data/air-quality.csv')
data = data.dropna()

# Example dataset structure assumption
# You can replace with your own dataset if needed
data = data.rename(columns={
    'value': 'AQI',
    'city': 'City'
})
data = data.head(500)

# Generate sample features for demo
np.random.seed(42)
data['Temperature'] = np.random.uniform(15, 35, len(data))
data['Humidity'] = np.random.uniform(30, 90, len(data))
data['CO2'] = np.random.uniform(300, 800, len(data))
data['PM2.5'] = np.random.uniform(10, 100, len(data))
data['PM10'] = np.random.uniform(20, 150, len(data))

# -------------------------------
# 2. Feature selection
# -------------------------------
features = ['Temperature', 'Humidity', 'CO2', 'PM2.5', 'PM10']
X = data[features]
y = data['AQI']

# -------------------------------
# 3. Train/Test split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -------------------------------
# 4. Model training
# -------------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# -------------------------------
# 5. Prediction
# -------------------------------
y_pred = model.predict(X_test)

# -------------------------------
# 6. Evaluation
# -------------------------------
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("📊 Model Performance:")
print(f"Mean Absolute Error: {mae:.2f}")
print(f"R² Score: {r2:.2f}")

# -------------------------------
# 7. Visualization
# -------------------------------
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.xlabel("Actual AQI")
plt.ylabel("Predicted AQI")
plt.title("Actual vs Predicted Air Quality Index")
plt.grid(True)
plt.savefig("images/results.png")
plt.show()
