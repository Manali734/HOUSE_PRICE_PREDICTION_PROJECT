from flask import Flask, render_template, request

import pandas as pd
import joblib
import mysql.connector

# Flask App
app = Flask(__name__)

# Load Model
model = joblib.load("model.pkl")

# MySQL Connection
db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="root",
    database="house_price_db"
)

cursor = db.cursor()

# Home Page
@app.route('/')

def home():

    return render_template("index.html")

# Prediction
@app.route('/predict', methods=['POST'])

def predict():

    area = int(
        request.form['area']
    )

    bedrooms = int(
        request.form['bedrooms']
    )

    bathrooms = int(
        request.form['bathrooms']
    )

    stories = int(
        request.form['stories']
    )

    mainroad = request.form['mainroad']

    guestroom = request.form['guestroom']

    basement = request.form['basement']

    hotwaterheating = request.form['hotwaterheating']

    airconditioning = request.form['airconditioning']

    parking = int(
        request.form['parking']
    )

    prefarea = request.form['prefarea']

    furnishingstatus = request.form['furnishingstatus']

    # Create DataFrame
    data = pd.DataFrame(
        [{
            "area": area,
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "stories": stories,
            "mainroad": mainroad,
            "guestroom": guestroom,
            "basement": basement,
            "hotwaterheating": hotwaterheating,
            "airconditioning": airconditioning,
            "parking": parking,
            "prefarea": prefarea,
            "furnishingstatus": furnishingstatus
        }]
    )

    # Predict
    prediction = model.predict(data)[0]

    # Save to MySQL
    query = """
    INSERT INTO predictions
    (
        area,
        bedrooms,
        bathrooms,
        stories,
        parking,
        predicted_price
    )

    VALUES (%s,%s,%s,%s,%s,%s)
    """

    values = (
        area,
        bedrooms,
        bathrooms,
        stories,
        parking,
        float(prediction)
    )

    cursor.execute(
        query,
        values
    )

    db.commit()

    return render_template(
    "result.html",
    prediction=round(prediction, 2)
)

# Run App
if __name__ == "__main__":

    app.run(
        debug=True
    )