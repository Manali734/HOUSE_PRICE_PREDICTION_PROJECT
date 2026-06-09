import pandas as pd

import mlflow
import mlflow.sklearn

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Load Dataset
df = pd.read_csv(
    "../dataset/Housing.csv"
)

# Features
X = df.drop(
    "price",
    axis=1
)

# Target
y = df["price"]

# Categorical Columns
cat_cols = X.select_dtypes(
    include="object"
).columns

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        (
            "cat",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            cat_cols
        )
    ],
    remainder="passthrough"
)

# Pipeline
model = Pipeline(
    [
        (
            "preprocessor",
            preprocessor
        ),

        (
            "regressor",
            RandomForestRegressor(
                n_estimators=100,
                random_state=42
            )
        )
    ]
)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Start MLflow Run
with mlflow.start_run():

    # Train
    model.fit(
        X_train,
        y_train
    )

    # Predict
    pred = model.predict(
        X_test
    )

    # MAE
    mae = mean_absolute_error(
        y_test,
        pred
    )

    # Log Parameter
    mlflow.log_param(
        "n_estimators",
        100
    )

    # Log Metric
    mlflow.log_metric(
        "MAE",
        mae
    )

    # Save Model
    mlflow.sklearn.log_model(
        model,
        "house_price_model"
    )

    print("\nMLflow Tracking Completed")

    print("\nMAE:")
    print(mae)


# Housing Dataset
#        ↓
# Machine Learning Model
#        ↓
# Flask UI
#        ↓
# MySQL
#        ↓
# FastAPI
#        ↓
# Swagger
#        ↓
# Jenkins
#        ↓
# FLAML AutoML
#        ↓
# BentoML
#        ↓
# # MLflow Tracking