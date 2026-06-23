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
# Custom CSS — Neon Black Theme
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
        background: #000000 !important;
    }

    .stApp {
        background: #000000 !important;
    }

    /* Hero header */
    .hero {
        background: linear-gradient(135deg, #000000 0%, #0a0a0a 50%, #000000 100%);
        border: 2px solid #00f0ff;
        padding: 2.5rem 2rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        box-shadow: 0 0 30px rgba(0, 240, 255, 0.3), inset 0 0 20px rgba(0, 240, 255, 0.05);
        position: relative;
        overflow: hidden;
    }
    .hero::before {
        content: '';
        position: absolute;
        top: -50%; right: -10%;
        width: 400px; height: 400px;
        background: radial-gradient(circle, rgba(0, 240, 255, 0.15), transparent);
        border-radius: 50%;
    }
    .hero h1 {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2.8rem;
        font-weight: 700;
        color: #00f0ff;
        margin: 0;
        letter-spacing: -0.02em;
        text-shadow: 0 0 20px rgba(0, 240, 255, 0.5);
    }
    .hero p {
        color: #a0a0a0;
        font-size: 1.1rem;
        margin-top: 0.5rem;
        margin-bottom: 0;
    }

    /* Section headings */
    h1, h2, h3 {
        font-family: 'Space Grotesk', sans-serif !important;
        color: #00f0ff !important;
        letter-spacing: -0.01em;
        text-shadow: 0 0 10px rgba(0, 240, 255, 0.3);
    }

    /* Card containers */
    .card {
        background: #0a0a0a;
        border: 1px solid #1a1a1a;
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 0 20px rgba(0, 240, 255, 0.05);
    }

    /* Inputs */
    .stNumberInput input, .stSelectbox div[data-baseweb="select"] > div {
        background-color: #0a0a0a !important;
        border: 1px solid #00f0ff !important;
        border-radius: 10px !important;
        color: #ffffff !important;
        box-shadow: 0 0 8px rgba(0, 240, 255, 0.15);
    }
    .stNumberInput input:focus {
        border-color: #39ff14 !important;
        box-shadow: 0 0 0 3px rgba(57, 255, 20, 0.2) !important;
    }
    label, .stSelectbox label, .stNumberInput label {
        color: #00f0ff !important;
        font-weight: 500 !important;
        font-size: 0.9rem !important;
        text-shadow: 0 0 5px rgba(0, 240, 255, 0.3);
    }

    /* Buttons */
    .stButton > button {
        background: transparent;
        color: #00f0ff;
        border: 2px solid #00f0ff;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        width: 100%;
        transition: all 0.3s ease;
        box-shadow: 0 0 15px rgba(0, 240, 255, 0.3);
        text-shadow: 0 0 10px rgba(0, 240, 255, 0.5);
    }
    .stButton > button:hover {
        background: rgba(0, 240, 255, 0.1);
        transform: translateY(-2px);
        box-shadow: 0 0 30px rgba(0, 240, 255, 0.6);
    }

    /* Metric */
    [data-testid="stMetric"] {
        background: #0a0a0a;
        border: 1px solid #00f0ff;
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 0 20px rgba(0, 240, 255, 0.15);
    }
    [data-testid="stMetricLabel"] {
        color: #a0a0a0 !important;
        font-weight: 500;
    }
    [data-testid="stMetricValue"] {
        color: #39ff14 !important;
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2.5rem !important;
        text-shadow: 0 0 15px rgba(57, 255, 20, 0.5);
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #000000 !important;
        border-right: 1px solid #00f0ff;
        box-shadow: 0 0 20px rgba(0, 240, 255, 0.1);
    }
    [data-testid="stSidebar"] h2 {
        color: #00f0ff !important;
        text-shadow: 0 0 10px rgba(0, 240, 255, 0.5);
        font-weight: 700;
    }
    [data-testid="stSidebar"] p {
        color: #a0a0a0 !important;
    }

    /* Alerts */
    .stAlert {
        border-radius: 12px;
        border: 1px solid !important;
        background: #0a0a0a !important;
    }
    .stAlert [data-baseweb="notification"] {
        background: #0a0a0a !important;
    }

    /* Tables */
    .stTable, .dataframe {
        background: #0a0a0a !important;
        border: 1px solid #00f0ff !important;
        border-radius: 12px;
        overflow: hidden;
    }
    .dataframe th {
        background: #000000 !important;
        color: #00f0ff !important;
        border-bottom: 1px solid #00f0ff !important;
    }
    .dataframe td {
        color: #ffffff !important;
    }
    .dataframe tr:hover {
        background: rgba(0, 240, 255, 0.05) !important;
    }

    /* Slider */
    .stSlider [data-baseweb="slider"] {
        background: #0a0a0a !important;
    }
    .stSlider [data-baseweb="slider"] [role="slider"] {
        border-color: #00f0ff !important;
        background: #00f0ff !important;
        box-shadow: 0 0 10px rgba(0, 240, 255, 0.5) !important;
    }

    /* Divider accent */
    .accent-line {
        height: 3px;
        background: linear-gradient(90deg, #00f0ff, #39ff14, #ff073a, transparent);
        border-radius: 2px;
        margin: 2rem 0 1rem 0;
        width: 120px;
        box-shadow: 0 0 10px rgba(0, 240, 255, 0.4);
    }

    /* Markdown text colors */
    p, li, span {
        color: #e0e0e0 !important;
    }

    /* Selectbox dropdown */
    div[data-baseweb="popover"] ul li {
        background: #0a0a0a !important;
        color: #ffffff !important;
    }
    div[data-baseweb="popover"] ul li:hover {
        background: rgba(0, 240, 255, 0.1) !important;
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
st.markdown('<div class="accent-line"></div>', unsafe_allow_html=True)
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
        # Neon Gauge Chart
        gauge_color = "#39ff14" if risk == "Low Risk" else "#fffc00" if risk == "Medium Risk" else "#ff073a"
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            number={'suffix': "%", 'font': {'size': 40, 'color': '#ffffff'}},
            title={"text": "Churn Risk Score", 'font': {'size': 18, 'color': '#00f0ff'}},
            gauge={
                "axis": {"range": [0, 100], 'tickcolor': '#00f0ff', 'tickfont': {'color': '#00f0ff'}},
                "bar": {"color": gauge_color, 'thickness': 0.3},
                "bgcolor": "#0a0a0a",
                "borderwidth": 2,
                "bordercolor": "#00f0ff",
                "steps": [
                    {"range": [0, 40], "color": "rgba(57, 255, 20, 0.15)"},
                    {"range": [40, 70], "color": "rgba(255, 252, 0, 0.15)"},
                    {"range": [70, 100], "color": "rgba(255, 7, 58, 0.15)"},
                ],
                "threshold": {
                    "line": {"color": "#ffffff", "width": 3},
                    "thickness": 0.75,
                    "value": probability * 100
                }
            }
        ))
        fig.update_layout(
            paper_bgcolor='#000000',
            font={'color': '#ffffff'},
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

    fig2, ax = plt.subplots(figsize=(10, 5))
    fig2.patch.set_facecolor('#000000')
    ax.set_facecolor('#000000')

    # Custom neon palette
    neon_palette = ["#00f0ff", "#39ff14", "#00f0ff", "#39ff14", "#fffc00",
                    "#ff073a", "#00f0ff", "#39ff14", "#fffc00", "#ff073a"]

    sns.barplot(
        data=importance.head(10),
        x="Importance", y="Feature",
        ax=ax,
        palette=neon_palette[:len(importance.head(10))]
    )
    ax.set_xlabel("Importance", color='#00f0ff', fontsize=11)
    ax.set_ylabel("Feature", color='#00f0ff', fontsize=11)
    ax.tick_params(colors='#00f0ff')
    ax.spines['bottom'].set_color('#00f0ff')
    ax.spines['left'].set_color('#00f0ff')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
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
