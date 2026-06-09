from fastapi import FastAPI
from pydantic import BaseModel

import pandas as pd
import joblib

# FastAPI App
app = FastAPI(
    title="House Price Prediction API",
    description="Predict House Price using Machine Learning",
    version="1.0"
)

# Load Model
model = joblib.load("model.pkl")

# Input Schema
class HouseData(BaseModel):

    area: int
    bedrooms: int
    bathrooms: int
    stories: int

    mainroad: str
    guestroom: str
    basement: str
    hotwaterheating: str
    airconditioning: str

    parking: int

    prefarea: str

    furnishingstatus: str


# Home Route
@app.get("/")

def home():

    return {
        "message":
        "House Price Prediction API Running"
    }


# Prediction Route
@app.post("/predict")

def predict(data: HouseData):

    df = pd.DataFrame(
        [data.dict()]
    )

    prediction = model.predict(df)[0]

    return {

        "Predicted Price":
        round(float(prediction), 2)

    }

# Swagger UI
#       ↓
# JSON Input
#       ↓
# Pydantic Validation
#       ↓
# HouseData Object
#       ↓
# DataFrame Creation
#       ↓
# Random Forest Model
#       ↓
# Prediction
#       ↓
# JSON Response