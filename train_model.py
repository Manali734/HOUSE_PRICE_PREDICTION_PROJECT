import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Load Dataset
df = pd.read_csv("dataset/Housing.csv")

print("\nDataset Shape:")
print(df.shape)

print("\nColumns:")
print(df.columns)

# Features
X = df.drop("price", axis=1)

# Target
y = df["price"]

# Identify categorical columns
cat_cols = X.select_dtypes(include="object").columns

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            cat_cols
        )
    ],
    remainder="passthrough"
)

# Pipeline
model = Pipeline(
    [
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(
            n_estimators=100,
            random_state=42
        ))
    ]
)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train
model.fit(X_train, y_train)

# Predict
pred = model.predict(X_test)

# Accuracy
mae = mean_absolute_error(y_test, pred)

print("\nMean Absolute Error:")
print(mae)

# Save model
joblib.dump(
    model,
    "flask_app/model.pkl"
)

joblib.dump(
    model,
    "fastapi_app/model.pkl"
)

joblib.dump(
    model,
    "bentoml_project/model.pkl"
)

print("\nModel Saved Successfully")

# flow
# Housing.csv
#       ↓
# Data Loaded
#       ↓
# Categorical Encoding
#       ↓
# Random Forest Training
#       ↓
# Model Learned Patterns
#       ↓
# model.pkl Generated
#       ↓
# Saved for Flask
#       ↓
# Saved for FastAPI