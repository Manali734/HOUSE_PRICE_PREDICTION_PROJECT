# House Price Prediction Full Stack MLOps Project

## Project Overview

This project demonstrates the complete Machine Learning Project Lifecycle using modern tools and technologies such as Flask, FastAPI, Jenkins, AutoML, BentoML, MLflow, MySQL, and GitHub.

The application predicts house prices based on various property features and showcases how a machine learning model can be developed, deployed, monitored, and automated using industry-standard MLOps tools.

---

## 🎯 Objectives

* Build a Machine Learning model for House Price Prediction
* Create a Flask-based user interface for prediction
* Create a FastAPI service with Swagger UI testing
* Store prediction records in MySQL Database
* Automate deployment using Jenkins
* Perform automated model selection using FLAML AutoML
* Deploy model as a prediction service using BentoML
* Track experiments using MLflow
* Manage source code using GitHub

---

## 🛠 Technologies Used

| Technology   | Purpose              |
| ------------ | -------------------- |
| Python 3.11  | Programming Language |
| Pandas       | Data Processing      |
| NumPy        | Numerical Operations |
| Scikit-Learn | Machine Learning     |
| Flask        | Web Application      |
| FastAPI      | REST API             |
| MySQL        | Database             |
| Jenkins      | CI/CD Automation     |
| FLAML        | AutoML               |
| BentoML      | Model Serving        |
| MLflow       | Experiment Tracking  |
| GitHub       | Version Control      |

---

## 📂 Project Structure

```text
HOUSE_PRICE_PROJECT
│
├── dataset/
│   └── Housing.csv
│
├── flask_app/
│   ├── app.py
│   ├── model.pkl
│   └── templates/
│       ├── index.html
│       └── result.html
│
├── fastapi_app/
│   ├── main.py
│   └── model.pkl
│
├── automl/
│   └── automl_demo.py
│
├── bentoml_project/
│   ├── service.py
│   └── model.pkl
│
├── mlflow_project/
│   └── mlflow_demo.py
│
├── screenshots/
│
├── train_model.py
│
├── requirements.txt
│
├── Dockerfile
│
└── README.md
```

---

## 📊 Dataset Information

Dataset Name: Housing Dataset

Target Variable:

```text
price
```

Input Features:

```text
area
bedrooms
bathrooms
stories
mainroad
guestroom
basement
hotwaterheating
airconditioning
parking
prefarea
furnishingstatus
```

---

## 🤖 Machine Learning Model

Algorithm Used:

```text
Random Forest Regressor
```

Performance Metric:

```text
Mean Absolute Error (MAE)
```

Model Output:

```text
House Price Prediction
```

Generated Model File:

```text
model.pkl
```

---

## 🌐 Flask Application

Features:

* Modern Bootstrap UI
* User-friendly House Price Prediction Form
* Real-time prediction
* MySQL integration

Run Flask Application:

```bash
cd flask_app

python app.py
```

Access:

```text
http://localhost:5000
```

---

## ⚡ FastAPI Application

Features:

* REST API
* Swagger Documentation
* JSON-based prediction

Run FastAPI:

```bash
cd fastapi_app

uvicorn main:app --reload
```

Access Swagger UI:

```text
http://localhost:8000/docs
```

---

## 🗄 MySQL Database

Database:

```sql
house_price_db
```

Table:

```sql
predictions
```

Stores:

* Input Parameters
* Predicted Price
* Timestamp

---

## 🔄 Jenkins Automation

Purpose:

* Continuous Integration
* Automated Model Training
* Automated FastAPI Deployment

Jenkins Build Steps:

```text
Install Dependencies
        ↓
Train Model
        ↓
Generate model.pkl
        ↓
Deploy FastAPI
```

---

## 🤖 AutoML Using FLAML

Purpose:

* Automatic model selection
* Hyperparameter tuning
* Best model identification

Run:

```bash
cd automl

python automl_demo.py
```

Output:

```text
Best Model
Best Configuration
MAE Score
```

---

## 📦 BentoML Deployment

Purpose:

* Package trained ML model
* Deploy prediction service
* Production-ready model serving

Run:

```bash
cd bentoml_project

python -m bentoml serve service:HousePriceService --reload
```

---

## 📈 MLflow Experiment Tracking

Features:

* Parameter Tracking
* Metric Tracking
* Model Versioning
* Experiment Management

Run:

```bash
cd mlflow_project

python mlflow_demo.py

mlflow ui
```

Dashboard:

```text
http://localhost:5000
```



Full Stack Machine Learning Project using Flask, FastAPI, Jenkins, AutoML, BentoML, MLflow, MySQL, and GitHub.
