# Customer Churn Prediction

A Machine Learning and Deep Learning project that predicts whether a bank customer is likely to churn (leave the bank) based on customer demographic and account information. The project includes data preprocessing, ANN model training using TensorFlow/Keras, and a Streamlit web application for real-time predictions.

---

## 📌 Project Overview

Customer churn prediction helps businesses identify customers who are likely to stop using their services. By predicting churn in advance, organizations can take proactive measures to improve customer retention.

This project uses:

- Python
- TensorFlow / Keras
- Scikit-learn
- Pandas
- NumPy
- Streamlit

---

## 🚀 Deployment

This project is deployed as a web application using Streamlit Cloud, enabling real-time interaction with the trained machine learning model through a browser interface.

The deployed application hosts the trained customer churn prediction model and performs inference on user-provided inputs to generate predictions instantly.

The deployment environment is configured to ensure compatibility with the machine learning dependencies, including TensorFlow, Scikit-learn, and other required Python libraries.

The application is designed for lightweight, serverless hosting, allowing continuous availability and easy access without requiring local setup.

### 🔗 Live Application

You can access the deployed app here:
👉https://customer-churn-prediction-adxnrovsrccctdewbsa7ra.streamlit.app/

## 📂 Project Structure

```text
Customer-Churn-Prediction/
│
├── artifacts/
│   ├── model.h5
│   ├── encoder.pkl
│   └── scaler.pkl
│
├── dataset/
│   └── Churn_Modelling.csv
│
├── app.py
├── experiments.ipynb
├── prediction.ipynb
├── requirements.txt
├── README.md
└── .gitignore


