import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data/heart_disease_clean.csv")

# Create outputs folder
import os
os.makedirs("outputs", exist_ok=True)

# Age distribution
plt.figure(figsize=(8,5))

sns.histplot(
    data=df,
    x="age",
    hue="target",
    bins=20,
    kde=True
)

plt.title("Age Distribution by Disease")

plt.savefig(
    "outputs/age_distribution.png"
)

# Correlation heatmap
numerical_cols = [
    "age","trestbps","chol",
    "thalach","oldpeak"
]

corr = df[numerical_cols].corr()

plt.figure(figsize=(8,6))

sns.heatmap(
    corr,
    annot=True,
    cmap="coolwarm"
)

plt.savefig(
    "outputs/correlation_heatmap.png"
)

print("EDA completed.")