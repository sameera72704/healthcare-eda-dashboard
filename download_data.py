import pandas as pd

columns = [
    "age","sex","cp","trestbps","chol",
    "fbs","restecg","thalach","exang",
    "oldpeak","slope","ca","thal","target"
]

df = pd.read_csv(
    "data/processed.cleveland.data",
    header=None,
    names=columns,
    na_values="?"
)

df.to_csv(
    "data/heart_disease_raw.csv",
    index=False
)

print(df.head())
print(df.shape)