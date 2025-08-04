import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from PIL import Image
import base64

from prophet import Prophet
import prophet

# Fix for cmdstanpy backend
prophet.models.forecaster._validate_inputs = lambda *args, **kwargs: None

# ----------------------------- NumPy Compatibility Check -----------------------------
required_numpy = "1.26"
if not np.__version__.startswith(required_numpy):
    raise ImportError(
        f"❌ Detected NumPy version {np.__version__}. "
        f"This app requires NumPy {required_numpy}.x to work correctly.\n"
        f"➡️ Please install with: pip install numpy==1.26.4"
    )

# ----------------------------- Background Setup -----------------------------
def add_bg_image(image_path):
    try:
        with open(image_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode()
        css = f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        </style>
        """
        st.markdown(css, unsafe_allow_html=True)
    except:
        st.warning("⚠️ Background image not found. Proceeding without it.")

# Set background image (optional: replace with your file name)
add_bg_image("electricity_bg.jpg")

# ----------------------------- Streamlit Setup -----------------------------
st.set_page_config(page_title="Forecasting App", layout="wide")
st.markdown("<h1 style='text-align: center; color: white;'>🔌 Energy Forecasting App</h1>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("PJMW_hourly", type=["csv", "xlsx", "xls"])

if uploaded_file:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        st.success("✅ File uploaded successfully!")

        # Column selection
        ds_col = st.selectbox("📅 Select the Date/Time column", df.columns)
        y_col = st.selectbox("⚡ Select the Energy Usage column", df.select_dtypes(include=np.number).columns)

        # Format data
        df[ds_col] = pd.to_datetime(df[ds_col])
        df = df[[ds_col, y_col]].dropna().rename(columns={ds_col: "ds", y_col: "y"})

        # Frequency selection
        freq_option = st.radio("⏱️ Choose aggregation frequency:", ["Hourly", "Daily", "Monthly"])
        freq_map = {"Hourly": "H", "Daily": "D", "Monthly": "M"}
        freq = freq_map[freq_option]

        df = df.set_index("ds").resample(freq).sum().reset_index()

        # Forecast horizon
        periods = st.number_input("🔮 Forecast how many future periods?", min_value=1, max_value=365, value=30)

        # Prophet model
        model = Prophet()
        model.fit(df)
        future = model.make_future_dataframe(periods=periods, freq=freq)
        forecast = model.predict(future)

        # Display forecasted table
        st.subheader("📋 Forecasted Values")
        forecast_output = forecast[["ds", "yhat"]].tail(periods)
        forecast_output.columns = ["Date", "Forecasted Energy Usage"]
        st.dataframe(forecast_output, use_container_width=True)

        # Plot
        st.subheader("📈 Forecast Chart")
        fig = px.line(forecast, x='ds', y='yhat', title="Forecasted Energy Usage")
        fig.add_scatter(x=df['ds'], y=df['y'], mode='lines', name='Actual Usage')
        fig.update_layout(legend=dict(x=0.01, y=0.99))
        st.plotly_chart(fig, use_container_width=True)

        # Decomposition
        with st.expander("🧩 Forecast Components"):
            st.write(model.plot_components(forecast))

    except Exception as e:
        st.error(f"❌ Something went wrong: {e}")
else:
    st.info("👆 Please upload your time series dataset to begin.")

