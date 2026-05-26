import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import joblib

# page setup
st.set_page_config(
    page_title="Heart Disease Dashboard",
    layout="wide"
)

# load files
model = joblib.load("model.pkl")
df = pd.read_csv("data/heart_disease_clean.csv")

# title
st.title("Heart Disease Analytics Dashboard")

# sidebar inputs
st.sidebar.header("Patient Information")

age = st.sidebar.slider("Age", 20, 90, 45)

sex = st.sidebar.selectbox(
    "Sex",
    ["Male", "Female"]
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

# encoding

sex_map = {
    "Male":1,
    "Female":0
}

cp_map = {
    "Typical Angina":1,
    "Atypical Angina":2,
    "Non-anginal Pain":3,
    "Asymptomatic":4
}

# tabs

tab1, tab2 = st.tabs([
    "Prediction",
    "Analytics"
])

# prediction tab

with tab1:

    st.subheader("Heart Disease Prediction")

    if st.button("Predict"):

        input_data = [[
            age,
            sex_map[sex],
            cp_map[cp],
            chol,
            thalach
        ]]

        prediction = model.predict(
            input_data
        )[0]

        probability = model.predict_proba(
            input_data
        )[0][1]

        if prediction == 1:
            st.error(
                f"High Risk ({probability*100:.1f}%)"
            )
        else:
            st.success(
                f"Low Risk ({probability*100:.1f}%)"
            )

        # gauge chart

        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=probability*100,
                title={
                    "text":"Risk Score"
                },
                gauge={
                    "axis":{
                        "range":[0,100]
                    },
                    "steps":[
                        {
                            "range":[0,40],
                            "color":"green"
                        },
                        {
                            "range":[40,70],
                            "color":"orange"
                        },
                        {
                            "range":[70,100],
                            "color":"red"
                        }
                    ]
                }
            )
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.markdown("### Patient Summary")

        st.write(
            f"Age: {age}"
        )

        st.write(
            f"Sex: {sex}"
        )

        st.write(
            f"Chest Pain: {cp}"
        )

# analytics tab

with tab2:

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Patients",
        len(df)
    )

    col2.metric(
        "Disease Cases",
        int(df["target"].sum())
    )

    col3.metric(
        "Disease Rate",
        f"{df['target'].mean()*100:.1f}%"
    )

    scatter = px.scatter(
        df,
        x="age",
        y="thalach",
        color="target",
        title="Age vs Max Heart Rate"
    )

    st.plotly_chart(
        scatter,
        use_container_width=True
    )

    chest_pain_chart = px.bar(
        df.groupby("cp")["target"]
        .mean()
        .reset_index(),
        x="cp",
        y="target",
        title="Disease Rate by Chest Pain"
    )

    st.plotly_chart(
        chest_pain_chart,
        use_container_width=True
    )

    if st.checkbox("Show Dataset"):
        st.dataframe(df)