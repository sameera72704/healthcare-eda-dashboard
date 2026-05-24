import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

# Load dataset
df = pd.read_csv(
    "data/heart_disease_clean.csv"
)

# Encode categories
df["sex"] = df["sex"].map({
    "Male":1,
    "Female":0
})

df["cp"] = df["cp"].map({
    "Typical Angina":1,
    "Atypical Angina":2,
    "Non-anginal Pain":3,
    "Asymptomatic":4
})

# Features
X = df[
    [
        "age",
        "sex",
        "cp",
        "chol",
        "thalach"
    ]
]

# Target
y = df["target"]

# Split
X_train,X_test,y_train,y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train model
model = LogisticRegression()

model.fit(
    X_train,
    y_train
)

# Predict
predictions = model.predict(
    X_test
)

# Accuracy
accuracy = accuracy_score(
    y_test,
    predictions
)

print(
    f"Model Accuracy: {accuracy:.2f}"
)

# Save model
joblib.dump(
    model,
    "model.pkl"
)

print(
    "Model saved successfully."
)