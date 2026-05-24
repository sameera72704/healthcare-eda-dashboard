import streamlit as st
import pandas as pd
import plotly.express as px
import joblib

st.set_page_config(
    page_title="Healthcare Analysis Dashboard",
    layout="wide"
)

# Load model
model = joblib.load(
    "model.pkl"
)

# Load data
@st.cache_data
def load_data():
    return pd.read_csv(
        "data/heart_disease_clean.csv"
    )

df = load_data()

# Sidebar
st.sidebar.title(
    "Prediction Inputs"
)

age = st.sidebar.slider(
    "Age",
    20,
    90,
    45
)

sex = st.sidebar.selectbox(
    "Sex",
    ["Male","Female"]
)

cp = st.sidebar.selectbox(
    "Chest Pain Type",
    [
        "Typical Angina",
        "Atypical Angina",
        "Non-anginal Pain",
        "Asymptomatic"
    ]
)

chol = st.sidebar.slider(
    "Cholesterol",
    100,
    600,
    220
)

thalach = st.sidebar.slider(
    "Max Heart Rate",
    70,
    220,
    150
)

# Encoding
sex_encoded = 1 if sex=="Male" else 0

cp_mapping = {
    "Typical Angina":1,
    "Atypical Angina":2,
    "Non-anginal Pain":3,
    "Asymptomatic":4
}

cp_encoded = cp_mapping[cp]

# Prediction
if st.sidebar.button(
    "Predict Risk"
):

    prediction = model.predict(
        [[
            age,
            sex_encoded,
            cp_encoded,
            chol,
            thalach
        ]]
    )[0]

    probability = model.predict_proba(
        [[
            age,
            sex_encoded,
            cp_encoded,
            chol,
            thalach
        ]]
    )[0][1]

    if prediction==1:

        st.error(
            f"HIGH RISK — Probability: {probability*100:.1f}%"
        )

    else:

        st.success(
            f"LOW RISK — Probability: {probability*100:.1f}%"
        )

# Dashboard
st.title(
    " Heart Disease Analysis Dashboard"
)

col1,col2,col3 = st.columns(3)

col1.metric(
    "Patients",
    len(df)
)

col2.metric(
    "Disease Positive",
    int(df["target"].sum())
)

col3.metric(
    "Disease Rate",
    f"{df['target'].mean()*100:.1f}%"
)

fig = px.scatter(
    df,
    x="age",
    y="thalach",
    color="target",
    template="plotly_dark"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

if st.checkbox(
    "Show Dataset"
):
    st.dataframe(df)