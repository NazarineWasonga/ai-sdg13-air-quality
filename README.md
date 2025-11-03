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
ai-sdg13-air-quality/
│
├── air_quality_prediction.ipynb # Jupyter notebook version

├── air_quality_prediction.py # Python script version

├── README.md # Documentation

└── images/

├── results.png # Evaluation graph

├── model_output.png # Output visualization


---

## 🧑‍💻 Installation & Usage

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/ai-sdg13-air-quality.git
   cd ai-sdg13-air-quality
   
2. Install dependencies
pip install pandas numpy matplotlib scikit-learn

3. Run the script
python air_quality_prediction.py

4. Or open the notebook
jupyter notebook air_quality_prediction.ipynb

📈 Sample Results
| Metric   | Value |
| -------- | ----- |
| MAE      | 3.27  |
| R² Score | 0.91  |
Visuals:

---
🌱 Ethical Reflection
---

AI can help policymakers monitor and predict pollution levels effectively.
However, bias in data (e.g., missing data from rural areas) may lead to unequal solutions.
The project emphasizes transparency, fairness, and the importance of open environmental data for sustainable action.

💡 SDG Impact — SDG 13: Climate Action
---
Impact Area	Description
Prediction	Helps identify high-risk pollution zones
Awareness	Supports campaigns for cleaner air
Policy	Enables data-driven environmental decisions

---
🧩 Tools & Technologies
---

Python 🐍

Pandas / NumPy

Scikit-learn

Matplotlib

Jupyter Notebook

👏 Author

Nazarine Wasonga
---
AI for Sustainable Development Project — PLP Academy
---


