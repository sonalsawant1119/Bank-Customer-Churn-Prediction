import streamlit as st
import pandas as pd
import pickle
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns


# -------------------------------
# Load Model and Files
# -------------------------------

model = pickle.load(
    open("churn_model.pkl", "rb")
)

features = pickle.load(
    open("features.pkl", "rb")
)


importance = pd.read_csv(
    "feature_importance.csv"
)


# -------------------------------
# Page Configuration
# -------------------------------

st.set_page_config(
    page_title="Bank Churn Predictor",
    page_icon="🏦",
    layout="wide"
)


st.title("🏦 Bank Customer Churn Risk Prediction System")

st.write(
    "Machine Learning based customer churn prediction and risk scoring dashboard"
)


# -------------------------------
# Sidebar - What If Analysis
# -------------------------------

st.sidebar.header("🔍 What-If Scenario Simulator")

st.sidebar.write(
    "Change customer behaviour and observe churn risk changes"
)


# -------------------------------
# Customer Inputs
# -------------------------------


st.header("Customer Information")


col1, col2 = st.columns(2)


with col1:

    credit_score = st.number_input(
        "Credit Score",
        300,
        900,
        650
    )


    age = st.number_input(
        "Age",
        18,
        100,
        40
    )


    tenure = st.number_input(
        "Tenure",
        0,
        10,
        5
    )


    balance = st.number_input(
        "Balance",
        value=50000.0
    )


    salary = st.number_input(
        "Estimated Salary",
        value=100000.0
    )



with col2:


    products = st.selectbox(
        "Number of Products",
        [1,2,3,4]
    )


    has_card = st.selectbox(
        "Has Credit Card",
        [0,1]
    )


    active_member = st.selectbox(
        "Is Active Member",
        [0,1]
    )


    geography = st.selectbox(
        "Geography",
        ["France","Germany","Spain"]
    )


    gender = st.selectbox(
        "Gender",
        ["Male","Female"]
    )


# -------------------------------
# What If Inputs
# -------------------------------

scenario_products = st.sidebar.slider(
    "Change Products",
    1,
    4,
    products
)


scenario_active = st.sidebar.selectbox(
    "Change Active Member",
    [0,1],
    index=active_member
)



# -------------------------------
# Prediction Button
# -------------------------------


if st.button("🚀 Predict Churn Risk"):


    # Feature Engineering

    input_data = pd.DataFrame({

        "CreditScore":[credit_score],

        "Age":[age],

        "Tenure":[tenure],

        "Balance":[balance],

        "NumOfProducts":[scenario_products],

        "HasCrCard":[has_card],

        "IsActiveMember":[scenario_active],

        "EstimatedSalary":[salary]

    })


    input_data["BalanceSalaryRatio"] = (
        input_data["Balance"]
        /
        input_data["EstimatedSalary"]
    )


    input_data["EngagementScore"] = (

        input_data["IsActiveMember"]

        +

        input_data["HasCrCard"]

        +

        input_data["NumOfProducts"]

    )


    # Age Group Encoding


    if age < 30:

        age_group = "Young"


    elif age < 50:

        age_group = "Middle"


    else:

        age_group = "Senior"



    input_data["Geography_Germany"] = (
        1 if geography=="Germany" else 0
    )


    input_data["Geography_Spain"] = (
        1 if geography=="Spain" else 0
    )


    input_data["Gender_Male"] = (
        1 if gender=="Male" else 0
    )


    input_data["AgeGroup_Senior"] = (
        1 if age_group=="Senior" else 0
    )


    input_data["AgeGroup_Young"] = (
        1 if age_group=="Young" else 0
    )


    # Arrange columns

    input_data = input_data[features]



    # Prediction

    probability = model.predict_proba(
        input_data
    )[0][1]



    # Risk Category


    if probability < 0.4:

        risk = "Low Risk"


    elif probability < 0.7:

        risk = "Medium Risk"


    else:

        risk = "High Risk"



    # -------------------------------
    # Results
    # -------------------------------


    st.header("Prediction Result")


    st.metric(
        "Churn Probability",
        f"{probability*100:.2f}%"
    )



    # Gauge Chart


    fig = go.Figure(

        go.Indicator(

            mode="gauge+number",

            value=probability*100,

            title={
                "text":"Churn Risk Percentage"
            },

            gauge={

                "axis":{
                    "range":[0,100]
                }

            }

        )

    )


    st.plotly_chart(fig)



    # Risk Display


    if risk=="High Risk":

        st.error(
            f"Customer Risk Level: {risk}"
        )


        st.warning(
            "Recommendation: High churn probability detected. "
            "Start retention campaigns immediately."
        )


    elif risk=="Medium Risk":

        st.warning(
            f"Customer Risk Level: {risk}"
        )


        st.info(
            "Recommendation: Increase engagement "
            "and provide personalized offers."
        )


    else:

        st.success(
            f"Customer Risk Level: {risk}"
        )


        st.success(
            "Recommendation: Maintain relationship "
            "and explore cross-selling opportunities."
        )



    # -------------------------------
    # Feature Importance Dashboard
    # -------------------------------


    st.header(
        "📊 Top Factors Affecting Customer Churn"
    )


    fig2, ax = plt.subplots(
        figsize=(8,5)
    )


    sns.barplot(

        data=importance.head(10),

        x="Importance",

        y="Feature",

        ax=ax

    )


    st.pyplot(fig2)



    # -------------------------------
    # Customer Summary
    # -------------------------------


    st.header(
        "Customer Summary"
    )


    summary = pd.DataFrame({

        "Feature":[
            "Age",
            "Balance",
            "Products",
            "Active Member",
            "Salary"
        ],

        "Value":[
            age,
            balance,
            scenario_products,
            scenario_active,
            salary
        ]

    })


    st.table(summary)