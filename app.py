import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# -------------------------------
# Load Model and Files
# -------------------------------
model = joblib.load("churn_model.pkl")
features = joblib.load("features.pkl")
importance = pd.read_csv("feature_importance.csv")

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="Bank Churn Predictor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------
# Custom CSS — Advanced Design
# -------------------------------
st.markdown("""
<style>
    /* Import fonts */
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@400;500;600&display=swap');

    /* Base */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    }

    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%);
    }

    /* Hero header */
    .hero {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #ec4899 100%);
        padding: 2.5rem 2rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        box-shadow: 0 20px 60px rgba(99, 102, 241, 0.3);
        position: relative;
        overflow: hidden;
    }
    .hero::before {
        content: '';
        position: absolute;
        top: -50%; right: -10%;
        width: 400px; height: 400px;
        background: radial-gradient(circle, rgba(255,255,255,0.15), transparent);
        border-radius: 50%;
    }
    .hero h1 {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2.8rem;
        font-weight: 700;
        color: white;
        margin: 0;
        letter-spacing: -0.02em;
    }
    .hero p {
        color: rgba(255,255,255,0.9);
        font-size: 1.1rem;
        margin-top: 0.5rem;
        margin-bottom: 0;
    }

    /* Section headings */
    h1, h2, h3 {
        font-family: 'Space Grotesk', sans-serif !important;
        color: #f1f5f9 !important;
        letter-spacing: -0.01em;
    }

    /* Card containers */
    .card {
        background: rgba(30, 41, 59, 0.6);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(148, 163, 184, 0.15);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.2);
    }

    /* Inputs */
    .stNumberInput input, .stSelectbox div[data-baseweb="select"] > div {
        background-color: rgba(15, 23, 42, 0.8) !important;
        border: 1px solid rgba(99, 102, 241, 0.3) !important;
        border-radius: 10px !important;
        color: #f1f5f9 !important;
    }
    .stNumberInput input:focus {
        border-color: #6366f1 !important;
        box-shadow: 0 0 0 3px rgba(99,102,241,0.2) !important;
    }
    label, .stSelectbox label, .stNumberInput label {
        color: #cbd5e1 !important;
        font-weight: 500 !important;
        font-size: 0.9rem !important;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        width: 100%;
        transition: all 0.3s ease;
        box-shadow: 0 4px 20px rgba(99, 102, 241, 0.4);
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(99, 102, 241, 0.6);
    }

    /* Metric */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, rgba(99,102,241,0.15), rgba(139,92,246,0.15));
        border: 1px solid rgba(99,102,241,0.3);
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }
    [data-testid="stMetricLabel"] {
        color: #cbd5e1 !important;
        font-weight: 500;
    }
    [data-testid="stMetricValue"] {
        color: #f1f5f9 !important;
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2.5rem !important;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e1b4b 0%, #0f172a 100%);
        border-right: 1px solid rgba(99,102,241,0.2);
    }
    [data-testid="stSidebar"] h2 {
        background: linear-gradient(135deg, #a78bfa, #f0abfc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
    }

    /* Alerts */
    .stAlert {
        border-radius: 12px;
        border: none;
        backdrop-filter: blur(10px);
    }

    /* Tables */
    .stTable, .dataframe {
        background: rgba(30,41,59,0.6) !important;
        border-radius: 12px;
        overflow: hidden;
    }

    /* Divider accent */
    .accent-line {
        height: 3px;
        background: linear-gradient(90deg, #6366f1, #ec4899, transparent);
        border-radius: 2px;
        margin: 2rem 0 1rem 0;
        width: 80px;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------
# Hero Header
# -------------------------------
st.markdown("""
<div class="hero">
    <h1>🏦 Bank Churn Intelligence</h1>
    <p>AI-powered customer retention & risk scoring dashboard</p>
</div>
""", unsafe_allow_html=True)

# -------------------------------
# Sidebar - What If Analysis
# -------------------------------
st.sidebar.header("🔍 What-If Simulator")
st.sidebar.markdown("*Adjust scenarios and observe churn risk shifts in real time.*")
st.sidebar.markdown("---")

# -------------------------------
# Customer Inputs
# -------------------------------
st.markdown("### 👤 Customer Profile")
st.markdown('<div class="accent-line"></div>', unsafe_allow_html=True)

col1, col2 = st.columns(2, gap="large")

with col1:
    credit_score = st.number_input("Credit Score", 300, 900, 650)
    age = st.number_input("Age", 18, 100, 40)
    tenure = st.number_input("Tenure (years)", 0, 10, 5)
    balance = st.number_input("Account Balance", value=50000.0)
    salary = st.number_input("Estimated Salary", value=100000.0)

with col2:
    products = st.selectbox("Number of Products", [1, 2, 3, 4])
    has_card = st.selectbox("Has Credit Card", [0, 1])
    active_member = st.selectbox("Is Active Member", [0, 1])
    geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
    gender = st.selectbox("Gender", ["Male", "Female"])

# -------------------------------
# What If Inputs
# -------------------------------
scenario_products = st.sidebar.slider("Change Products", 1, 4, products)
scenario_active = st.sidebar.selectbox("Change Active Member", [0, 1], index=active_member)

# -------------------------------
# Prediction Button
# -------------------------------
st.markdown("<br>", unsafe_allow_html=True)
if st.button("🚀 Predict Churn Risk"):

    input_data = pd.DataFrame({
        "CreditScore": [credit_score],
        "Age": [age],
        "Tenure": [tenure],
        "Balance": [balance],
        "NumOfProducts": [scenario_products],
        "HasCrCard": [has_card],
        "IsActiveMember": [scenario_active],
        "EstimatedSalary": [salary]
    })

    input_data["BalanceSalaryRatio"] = input_data["Balance"] / input_data["EstimatedSalary"]
    input_data["EngagementScore"] = (
        input_data["IsActiveMember"] + input_data["HasCrCard"] + input_data["NumOfProducts"]
    )

    if age < 30:
        age_group = "Young"
    elif age < 50:
        age_group = "Middle"
    else:
        age_group = "Senior"

    input_data["Geography_Germany"] = 1 if geography == "Germany" else 0
    input_data["Geography_Spain"] = 1 if geography == "Spain" else 0
    input_data["Gender_Male"] = 1 if gender == "Male" else 0
    input_data["AgeGroup_Senior"] = 1 if age_group == "Senior" else 0
    input_data["AgeGroup_Young"] = 1 if age_group == "Young" else 0

    input_data = input_data[features]
    probability = model.predict_proba(input_data)[0][1]

    if probability < 0.4:
        risk = "Low Risk"
    elif probability < 0.7:
        risk = "Medium Risk"
    else:
        risk = "High Risk"

    # -------------------------------
    # Results
    # -------------------------------
    st.markdown("### 📈 Prediction Result")
    st.markdown('<div class="accent-line"></div>', unsafe_allow_html=True)

    res_col1, res_col2 = st.columns([1, 2], gap="large")

    with res_col1:
        st.metric("Churn Probability", f"{probability*100:.2f}%")

    with res_col2:
        # Polished Gauge Chart
        gauge_color = "#10b981" if risk == "Low Risk" else "#f59e0b" if risk == "Medium Risk" else "#ef4444"
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            number={'suffix': "%", 'font': {'size': 40, 'color': '#f1f5f9'}},
            title={"text": "Churn Risk Score", 'font': {'size': 18, 'color': '#cbd5e1'}},
            gauge={
                "axis": {"range": [0, 100], 'tickcolor': '#64748b', 'tickfont': {'color': '#94a3b8'}},
                "bar": {"color": gauge_color, 'thickness': 0.3},
                "bgcolor": "rgba(15,23,42,0.5)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 40], "color": "rgba(16,185,129,0.2)"},
                    {"range": [40, 70], "color": "rgba(245,158,11,0.2)"},
                    {"range": [70, 100], "color": "rgba(239,68,68,0.2)"},
                ],
                "threshold": {
                    "line": {"color": "white", "width": 3},
                    "thickness": 0.75,
                    "value": probability * 100
                }
            }
        ))
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            font={'color': '#f1f5f9'},
            height=320,
            margin=dict(l=20, r=20, t=60, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)

    # Risk Display
    if risk == "High Risk":
        st.error(f"⚠️ Customer Risk Level: **{risk}**")
        st.warning("**Recommendation:** High churn probability detected. Launch retention campaigns immediately.")
    elif risk == "Medium Risk":
        st.warning(f"⚡ Customer Risk Level: **{risk}**")
        st.info("**Recommendation:** Increase engagement and offer personalized incentives.")
    else:
        st.success(f"✅ Customer Risk Level: **{risk}**")
        st.success("**Recommendation:** Maintain relationship and explore cross-selling opportunities.")

    # -------------------------------
    # Feature Importance
    # -------------------------------
    st.markdown("### 📊 Top Drivers of Customer Churn")
    st.markdown('<div class="accent-line"></div>', unsafe_allow_html=True)

    plt.style.use('dark_background')
    fig2, ax = plt.subplots(figsize=(10, 5))
    fig2.patch.set_facecolor('#0f172a')
    ax.set_facecolor('#0f172a')

    sns.barplot(
        data=importance.head(10),
        x="Importance", y="Feature",
        ax=ax,
        palette="mako"
    )
    ax.set_xlabel("Importance", color='#cbd5e1', fontsize=11)
    ax.set_ylabel("Feature", color='#cbd5e1', fontsize=11)
    ax.tick_params(colors='#94a3b8')
    for spine in ax.spines.values():
        spine.set_color('#334155')
    st.pyplot(fig2)

    # -------------------------------
    # Customer Summary
    # -------------------------------
    st.markdown("### 📋 Customer Snapshot")
    st.markdown('<div class="accent-line"></div>', unsafe_allow_html=True)

    summary = pd.DataFrame({
        "Feature": ["Age", "Balance", "Products", "Active Member", "Salary"],
        "Value": [age, balance, scenario_products, scenario_active, salary]
    })
    st.table(summary)
