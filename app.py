import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title="Healthcare Analytics Dashboard",
    page_icon="❤️",
    layout="wide"
)

# ---------- Custom CSS ----------
st.markdown("""
<style>

.main{
background-color:#0E1117;
color:white;
}

[data-testid="metric-container"]{
background:#1E1E2F;
padding:18px;
border-radius:15px;
border:1px solid #333;
box-shadow:0 4px 12px rgba(0,0,0,0.4);
}

h1,h2,h3{
color:#F8F9FA;
}

section[data-testid="stSidebar"]{
background:#151823;
}

</style>
""", unsafe_allow_html=True)

# ---------- Load Data ----------
@st.cache_data
def load_data():
    return pd.read_csv(
        "data/heart_disease_clean.csv"
    )

df = load_data()

# ---------- Sidebar ----------
st.sidebar.title("⚙ Dashboard Filters")

sex_filter = st.sidebar.multiselect(
    "Select Sex",
    df["sex"].unique(),
    default=df["sex"].unique()
)

age_range = st.sidebar.slider(
    "Age Range",
    int(df.age.min()),
    int(df.age.max()),
    (
        int(df.age.min()),
        int(df.age.max())
    )
)

filtered_df = df[
    (df["sex"].isin(sex_filter))
    &
    (df["age"].between(
        age_range[0],
        age_range[1]
    ))
]

# ---------- Header ----------
st.title("❤️ Heart Disease Analytics Dashboard")

st.markdown("""
Explore healthcare insights using **interactive visual analytics**.
""")

# ---------- KPI Cards ----------
col1,col2,col3,col4 = st.columns(4)

col1.metric(
    "Patients",
    len(filtered_df)
)

col2.metric(
    "Disease Positive",
    int(filtered_df["target"].sum())
)

col3.metric(
    "Avg Age",
    round(filtered_df["age"].mean(),1)
)

col4.metric(
    "Disease Rate",
    f"{filtered_df['target'].mean()*100:.1f}%"
)

st.divider()

# ---------- Charts Row ----------
left,right = st.columns(2)

with left:

    fig1 = px.scatter(
        filtered_df,
        x="age",
        y="thalach",
        color=filtered_df["target"].map({
            0:"No Disease",
            1:"Disease"
        }),
        title="Age vs Max Heart Rate",
        template="plotly_dark"
    )

    st.plotly_chart(
        fig1,
        use_container_width=True
    )

with right:

    cp_disease = (
        filtered_df
        .groupby("cp")["target"]
        .mean()
        .reset_index()
    )

    fig2 = px.bar(
        cp_disease,
        x="cp",
        y="target",
        color="target",
        color_continuous_scale="reds",
        title="Disease Rate by Chest Pain Type",
        template="plotly_dark"
    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )

# ---------- Box Plot ----------
fig3 = px.box(
    filtered_df,
    x="sex",
    y="chol",
    color="sex",
    title="Cholesterol Distribution",
    template="plotly_dark"
)

st.plotly_chart(
    fig3,
    use_container_width=True
)

# ---------- Insights ----------
st.subheader("📊 Key Insights")

st.info("""
• Higher age groups show elevated disease prevalence.

• Lower max heart rate correlates with disease presence.

• Chest pain type strongly influences diagnosis probability.
""")

# ---------- Raw Data ----------
if st.checkbox("Show Dataset"):
    st.dataframe(filtered_df)

st.caption(
"Built with Streamlit • Plotly • Pandas"
)