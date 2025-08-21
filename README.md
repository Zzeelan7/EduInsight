# EduInsight
A real-time data science project that uses machine learning to analyze and predict student performance based on academic records, attendance, quiz scores, and engagement metrics. The goal is to provide actionable insights for early intervention and personalized learning strategies.

# 🎓 Student Performance Prediction (JEE + Mock Tests)

This project uses **Machine Learning** to predict and categorize students into three performance categories:  
- **High Performer**  
- **Average**  
- **At Risk**  

based on their **JEE Main Score, JEE Advanced Score, and Mock Test Score Average**.  

The goal is to help coaching institutes, teachers, and students **identify performance trends** early and provide targeted support to those at risk.

---

## 🚀 Features
- Predicts student performance category (**High Performer, Average, At Risk**)  
- Interactive **Streamlit UI** for input and predictions  
- Model interpretability using **LIME** and **ELI5** (explains why a prediction was made)  
- Simple & easy to extend with more features  

---

## 📂 Dataset
The dataset contains student records with:  
- `jee_main_score`  
- `jee_advanced_score`  
- `mock_test_score_avg`  

(Other fields were dropped to keep the model focused on key predictors.)

---

## 🛠️ Tech Stack
- **Python**  
- **Pandas, NumPy** – Data preprocessing  
- **Scikit-learn** – Machine Learning  
- **LIME, ELI5** – Explainability & Model interpretation  
- **Streamlit** – UI for predictions  

---

## 📊 Model
- Preprocessing: Cleaned dataset to include only required features  
- Labels: Categorized students into 3 classes  
- Algorithm: **Random Forest Classifier** (can be tuned further)  
- Explanation: LIME & ELI5 used to interpret feature impact  

---
