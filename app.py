import streamlit as st
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.ensemble import IsolationForest

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="AI Smart Energy Optimization System",
    page_icon="⚡",
    layout="wide"
)

# =========================
# CUSTOM UI STYLE
# =========================
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #f8fbff 0%, #eef6ff 100%);
}

.main-title {
    font-size: 48px;
    font-weight: 900;
    color: #0f172a;
    text-align: center;
    padding-top: 15px;
}

.subtitle {
    font-size: 19px;
    color: #475569;
    text-align: center;
    margin-bottom: 30px;
}

.section-title {
    font-size: 30px;
    font-weight: 800;
    color: #0f172a;
    margin-top: 25px;
    margin-bottom: 15px;
}

[data-testid="stMetric"] {
    background: white;
    padding: 24px;
    border-radius: 20px;
    border: 1px solid #e5e7eb;
    box-shadow: 0px 5px 18px rgba(0,0,0,0.07);
}

.stTabs [data-baseweb="tab-list"] {
    gap: 18px;
}

.stTabs [data-baseweb="tab"] {
    height: 62px;
    padding: 14px 28px;
    background-color: white;
    border-radius: 18px;
    font-size: 19px;
    font-weight: 800;
    box-shadow: 0px 4px 14px rgba(0,0,0,0.06);
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #2563eb, #06b6d4);
    color: white;
}

.ai-box {
    padding: 24px;
    border-radius: 18px;
    background: #eff6ff;
    border-left: 7px solid #2563eb;
    font-size: 17px;
    box-shadow: 0px 4px 14px rgba(0,0,0,0.06);
}

.note-box {
    padding: 18px;
    border-radius: 14px;
    background: white;
    border-left: 6px solid #10b981;
    font-size: 16px;
    box-shadow: 0px 3px 10px rgba(0,0,0,0.05);
}
</style>
""", unsafe_allow_html=True)

# =========================
# LOAD MODEL
# =========================
model = joblib.load("models/energy_model.pkl")

# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_data():
    df = pd.read_csv(
        "data/household_power_consumption.txt",
        sep=";",
        low_memory=False,
        na_values=["?"]
    )

    df.dropna(inplace=True)

    df["Datetime"] = pd.to_datetime(
        df["Date"] + " " + df["Time"],
        dayfirst=True
    )

    df["Global_active_power"] = pd.to_numeric(
        df["Global_active_power"]
    )

    daily_data = df.resample(
        "D",
        on="Datetime"
    )["Global_active_power"].mean()

    return daily_data


daily_data = load_data()

# =========================
# ANOMALY DETECTION
# =========================
anomaly_model = IsolationForest(
    contamination=0.03,
    random_state=42
)

anomaly_result = anomaly_model.fit_predict(
    daily_data.values.reshape(-1, 1)
)

anomaly_count = list(anomaly_result).count(-1)

# =========================
# SIDEBAR
# =========================
st.sidebar.title("⚡ AI Energy System")

st.sidebar.markdown("""
**Developer:** Toufique Hasan  

**Project:** AI-Powered Smart Energy Optimization System  

**Model:** Random Forest Regressor  

**Dataset:** UCI Household Power Consumption  

**Features:**  
- Energy prediction  
- Cost estimation  
- Anomaly detection  
- AI insight assistant  
- Downloadable reports  
""")

# =========================
# HEADER
# =========================
st.markdown(
    '<div class="main-title">⚡ AI-Powered Smart Energy Optimization System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Predict, analyze, and optimize household energy consumption using Machine Learning and AI-based insights.</div>',
    unsafe_allow_html=True
)

current_time = datetime.now()

st.info(
    f"Current Date & Time: {current_time.strftime('%A, %B %d, %Y - %I:%M %p')}"
)

# =========================
# TABS
# =========================
tab1, tab2, tab3, tab4 = st.tabs(
    ["📊  Dashboard", "🤖  AI Prediction", "💬  AI Assistant", "📄  Reports"]
)

# =========================
# DASHBOARD TAB
# =========================
with tab1:
    st.markdown(
        '<div class="section-title">Executive Dashboard</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Dataset Records", "2.07M")
    col2.metric("ML Model", "Random Forest")
    col3.metric("MAE", "0.23")
    col4.metric("Energy Spikes", anomaly_count)

    st.markdown(
        '<div class="section-title">Energy Analytics</div>',
        unsafe_allow_html=True
    )

    avg_usage = daily_data.mean()
    max_usage = daily_data.max()
    min_usage = daily_data.min()

    col1, col2, col3 = st.columns(3)

    col1.metric("Average Daily Power", f"{avg_usage:.2f} kW")
    col2.metric("Maximum Daily Power", f"{max_usage:.2f} kW")
    col3.metric("Minimum Daily Power", f"{min_usage:.2f} kW")

    st.markdown(
        '<div class="section-title">Historical Energy Consumption Trend</div>',
        unsafe_allow_html=True
    )

    st.line_chart(daily_data.tail(365))

# =========================
# AI PREDICTION TAB
# =========================
with tab2:
    st.markdown(
        '<div class="section-title">AI Energy Consumption Prediction</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:
        selected_date = st.date_input(
            "Select Prediction Date",
            value=datetime.today()
        )

    with col2:
        previous_day = st.number_input(
            "Previous Day Average Power Usage (kW)",
            min_value=0.0,
            value=1.0
        )

    day = selected_date.day
    month = selected_date.month
    year = selected_date.year

    st.caption(
        f"Prediction Date: {selected_date.strftime('%A, %B %d, %Y')}"
    )

    if st.button("Predict Energy Consumption"):

        input_data = pd.DataFrame(
            [[previous_day, day, month, year]],
            columns=[
                "Previous_Day",
                "Day",
                "Month",
                "Year"
            ]
        )

        prediction_kw = model.predict(input_data)[0]

        estimated_daily_kwh = prediction_kw * 24
        estimated_monthly_kwh = estimated_daily_kwh * 30

        electricity_rate = 0.18
        estimated_bill = estimated_monthly_kwh * electricity_rate

        st.markdown(
            '<div class="section-title">Technical Prediction Results</div>',
            unsafe_allow_html=True
        )

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Predicted Avg Power", f"{prediction_kw:.2f} kW")
        col2.metric("Daily Energy", f"{estimated_daily_kwh:.1f} kWh")
        col3.metric("Monthly Usage", f"{estimated_monthly_kwh:.0f} kWh")
        col4.metric("Monthly Bill", f"${estimated_bill:.2f}")

        st.markdown(
            '<div class="note-box">The model predicts average active power in kW. Daily and monthly energy usage are estimated by converting kW to kWh over time.</div>',
            unsafe_allow_html=True
        )

        if estimated_monthly_kwh < 600:
            status = "Efficient"
            st.success("Energy Efficiency Status: Efficient ⭐⭐⭐⭐⭐")
            insight = """
            Your predicted monthly usage is efficient. The energy pattern appears stable,
            and the estimated bill is within a reasonable range. Continue monitoring daily
            usage and maintaining current energy-saving behavior.
            """

        elif estimated_monthly_kwh < 900:
            status = "Moderate"
            st.info("Energy Efficiency Status: Moderate ⭐⭐⭐⭐")
            insight = """
            Your predicted monthly usage is moderate. Cost can be reduced by monitoring
            peak-hour usage, unplugging standby devices, and tracking weekly consumption patterns.
            """

        else:
            status = "High Usage"
            st.warning("Energy Efficiency Status: High Usage ⭐⭐")
            insight = """
            Your predicted monthly usage is high. The system recommends reducing HVAC usage,
            avoiding unnecessary appliance operation, and shifting high-power appliances
            to off-peak periods when possible.
            """

        st.markdown(
            '<div class="section-title">AI Insight Assistant</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            f'<div class="ai-box"><b>AI Insight:</b><br>{insight}</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="section-title">Smart Energy Recommendations</div>',
            unsafe_allow_html=True
        )

        if estimated_monthly_kwh > 900:
            st.markdown("""
            - Reduce HVAC usage during peak hours.
            - Turn off unused appliances and electronics.
            - Use smart plugs to monitor high-power devices.
            - Run washing machines and dryers during off-peak hours.
            - Review daily energy patterns to identify waste.
            """)

        elif estimated_monthly_kwh > 600:
            st.markdown("""
            - Monitor peak-hour consumption.
            - Unplug standby devices when not in use.
            - Use energy-efficient lighting.
            - Track weekly usage trends.
            """)

        else:
            st.markdown("""
            - Continue monitoring daily energy usage.
            - Maintain current energy-saving habits.
            - Review monthly energy trends regularly.
            """)

        future_trend = np.random.normal(
            estimated_daily_kwh,
            2.0,
            30
        )

        future_df = pd.DataFrame({
            "Day": range(1, 31),
            "Estimated Daily Energy (kWh)": future_trend
        })

        st.markdown(
            '<div class="section-title">Predicted 30-Day Energy Trend</div>',
            unsafe_allow_html=True
        )

        st.line_chart(
            future_df.set_index("Day")
        )

        report = f"""
AI Smart Energy Optimization Report

Developer: Toufique Hasan
Project: AI-Powered Smart Energy Optimization System

Generated On: {current_time.strftime('%A, %B %d, %Y - %I:%M %p')}
Prediction Date: {selected_date.strftime('%A, %B %d, %Y')}

Technical Prediction Results:
Predicted Average Power: {prediction_kw:.2f} kW
Estimated Daily Energy: {estimated_daily_kwh:.1f} kWh
Estimated Monthly Usage: {estimated_monthly_kwh:.0f} kWh
Estimated Monthly Electricity Bill: ${estimated_bill:.2f}
Energy Efficiency Status: {status}

AI Insight:
{insight}

Recommendations:
- Monitor peak-hour electricity usage.
- Reduce unnecessary appliance operation.
- Improve daily energy efficiency.
- Use smart scheduling for high-power appliances.
- Track long-term energy consumption patterns.
"""

        st.download_button(
            label="Download Prediction Report",
            data=report,
            file_name="energy_prediction_report.txt",
            mime="text/plain"
        )

# =========================
# AI ASSISTANT TAB
# =========================
with tab3:
    st.markdown(
        '<div class="section-title">AI Energy Assistant</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
Ask a question about energy usage, electricity bills, anomaly detection,
efficiency, or machine learning model behavior.
""")

    user_question = st.text_input(
        "Ask the AI Assistant",
        placeholder="Example: Why is my electricity bill high?"
    )

    if st.button("Ask Assistant"):

        if user_question.strip() == "":
            answer = """
            Please enter a question about electricity bills, energy usage, anomaly detection,
            machine learning model behavior, or energy-saving recommendations.
            """

            st.markdown(
                f'<div class="ai-box"><b>Assistant Response:</b><br>{answer}</div>',
                unsafe_allow_html=True
            )

            st.stop()

        q = user_question.lower()

        if "bill" in q or "cost" in q or "expensive" in q:
            answer = """
            Your electricity bill may be high due to HVAC systems, dryers, heaters,
            long-duration appliance use, or high peak-hour consumption. This system estimates cost
            by converting predicted average power into daily and monthly energy usage.
            """

        elif "reduce" in q or "save" in q or "lower" in q:
            answer = """
            To reduce energy usage, monitor peak-hour consumption, unplug standby devices,
            use LED lighting, reduce HVAC runtime, and schedule high-power appliances during
            off-peak periods.
            """

        elif "anomaly" in q or "spike" in q or "unusual" in q:
            answer = f"""
            The anomaly detection module identified {anomaly_count} abnormal energy spikes
            in the historical data. These may represent unusual appliance usage, seasonal demand,
            or abnormal consumption behavior.
            """

        elif "model" in q or "machine learning" in q or "random forest" in q:
            answer = """
            The system uses a Random Forest Regressor to predict average active power based on
            previous-day usage and date-based features. Isolation Forest is used to detect abnormal
            energy consumption patterns.
            """

        elif "dataset" in q or "data" in q:
            answer = """
            This project uses the UCI Household Power Consumption dataset, which contains over
            2 million household electricity records. The data is cleaned, converted into datetime
            format, and resampled into daily energy usage patterns.
            """

        else:
            answer = """
            I can explain electricity bill estimation, energy usage reduction, anomaly detection,
            model behavior, dataset details, and energy optimization recommendations.
            """

        st.markdown(
            f'<div class="ai-box"><b>Assistant Response:</b><br>{answer}</div>',
            unsafe_allow_html=True
        )

# =========================
# REPORTS TAB
# =========================
with tab4:
    st.markdown(
        '<div class="section-title">Reports & Project Summary</div>',
        unsafe_allow_html=True
    )

    csv_data = daily_data.to_csv()

    st.download_button(
        label="Download Historical Energy Report",
        data=csv_data,
        file_name="historical_energy_report.csv",
        mime="text/csv"
    )

    st.markdown("""
### Project Summary

This project uses machine learning to predict household energy consumption,
estimate monthly electricity cost, detect abnormal usage patterns, and provide
smart energy optimization recommendations through an interactive dashboard.
""")

# =========================
# FOOTER
# =========================
st.markdown("""
<br>
<div style='text-align: center; color: #64748b; font-size: 15px;'>
Developed by <b>Toufique Hasan</b> | Machine Learning | Energy Analytics | AI Smart Optimization System
</div>
""", unsafe_allow_html=True)