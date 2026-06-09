import pandas as pd

from flaml import AutoML

from sklearn.model_selection import train_test_split
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

# Convert categorical columns
X = pd.get_dummies(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# AutoML Object
automl = AutoML()

settings = {

    "time_budget": 60,

    "metric": "mae",

    "task": "regression",

    "log_file_name":
    "housing_automl.log"
}

# Train
automl.fit(
    X_train=X_train,
    y_train=y_train,
    **settings
)

# Predict
predictions = automl.predict(
    X_test
)

# Accuracy
mae = mean_absolute_error(
    y_test,
    predictions
)

print("\n======================")
print("BEST MODEL")
print("======================")

print(
    automl.model
)

print("\n======================")
print("BEST CONFIG")
print("======================")

print(
    automl.best_config
)

print("\n======================")
print("MAE")
print("======================")

print(mae)

# Housing Dataset
#         ↓
# Categorical Encoding
#         ↓
# FLAML
#         ↓
# Tests Multiple Models
#         ↓
# Tunes Parameters
#         ↓
# Selects Best Model
#         ↓
# Reports Best MAE