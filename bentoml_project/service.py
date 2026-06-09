import bentoml
import pandas as pd
import joblib

# Load model
model = joblib.load("model.pkl")

@bentoml.service
class HousePriceService:

    @bentoml.api
    def predict(self, input_data: dict):

        df = pd.DataFrame(
            [input_data]
        )

        prediction = model.predict(df)[0]

        return {

            "Predicted Price":
            round(float(prediction), 2)

        }