import pandas as pd

df = pd.read_csv("data/heart_disease_raw.csv")

print("Raw shape:", df.shape)
print(df.isnull().sum())

# Fill missing values
df["ca"] = df["ca"].fillna(df["ca"].median())
df["thal"] = df["thal"].fillna(df["thal"].mode()[0])

# Convert target to binary
df["target"] = df["target"].apply(
    lambda x: 1 if x > 0 else 0
)

# Map categories
df["sex"] = df["sex"].map({
    1:"Male",
    0:"Female"
})

df["cp"] = df["cp"].map({
    1:"Typical Angina",
    2:"Atypical Angina",
    3:"Non-anginal Pain",
    4:"Asymptomatic"
})

# Remove duplicates
df = df.drop_duplicates()

print("Cleaned shape:", df.shape)

# Save cleaned data
df.to_csv(
    "data/heart_disease_clean.csv",
    index=False
)

print("Clean file saved successfully.")