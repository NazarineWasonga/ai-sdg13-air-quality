# 🌍 AI-SDG13-Air-Quality
### Machine Learning for Sustainable Development — SDG 13: Climate Action

This project applies **Machine Learning** to predict **Air Quality Index (AQI)** as part of **SDG 13 – Climate Action**, focusing on using AI to combat environmental pollution and improve climate-related decision-making.

---

## 🧠 Project Overview

Air pollution remains one of the biggest contributors to climate change and health issues.  
This project uses **supervised learning (Regression)** to predict **Air Quality Index (AQI)** based on environmental factors such as temperature, humidity, CO₂, and particulate matter (PM2.5 and PM10).

The model aims to:
- Predict air quality in real-time.
- Support policy and awareness for clean air initiatives.
- Demonstrate how AI can contribute to **SDG 13: Climate Action**.

---

## 🚀 Features

✅ Predicts air quality (AQI) based on weather and pollution data  
✅ Uses **Linear Regression** model for prediction  
✅ Visualizes actual vs. predicted AQI values  
✅ Includes ethical reflection on AI use for sustainability  

---

## 📊 Dataset

- **Source:** [Kaggle - Air Quality Data](https://www.kaggle.com/)
- **Features Used:**
  - Temperature (°C)
  - Humidity (%)
  - CO₂ (ppm)
  - PM2.5
  - PM10
  - Wind Speed (m/s)

---

## 🧮 Model Workflow

1. **Data Preprocessing**
   - Handle missing values
   - Normalize numerical data  
2. **Model Training**
   - Split data (80% training, 20% testing)
   - Train a Linear Regression model using `scikit-learn`
3. **Evaluation**
   - Calculate **Mean Absolute Error (MAE)** and **R² Score**
4. **Visualization**
   - Compare Actual vs Predicted AQI values

---

## 📁 Repository Structure

