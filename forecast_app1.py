#!/usr/bin/env python
# coding: utf-8

# In[27]:


import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
import plotly.express as px
from io import StringIO
from datetime import timedelta


st.set_page_config(page_title="Forecasting App", layout="wide")

st.title("Hourly Energy Consumption Forecast")

uploaded_file = st.file_uploader("Upload your dataset (CSV, Excel, etc.)", type=["csv", "xlsx", "xls"])

if uploaded_file:
    try:
        # Detect file type and read
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith((".xls", ".xlsx")):
            df = pd.read_excel(uploaded_file)
        else:
            st.error("Unsupported file format.")
            st.stop()

        # Column selection
        ds_col = st.selectbox("Select Date/Time column", df.columns)
        y_col = st.selectbox("Select Target column (numeric)", df.select_dtypes(include=np.number).columns)

        # Convert date
        df[ds_col] = pd.to_datetime(df[ds_col])
        df = df[[ds_col, y_col]].dropna().rename(columns={ds_col: "ds", y_col: "y"})

        # Choose aggregation
        freq_option = st.selectbox(
            "Choose aggregation level",
            ["Hourly", "Daily", "Monthly", "Yearly"]
        )

        freq_map = {
            "Hourly": "H",
            "Daily": "D",
            "Monthly": "M",
            "Yearly": "Y"
        }

        freq = freq_map[freq_option]
        df = df.set_index("ds").resample(freq).sum().reset_index()

        # Prophet modeling
        model = Prophet()
        model.fit(df)

        periods = st.slider("Select forecast horizon", min_value=10, max_value=1000, step=10, value=200)
        future = model.make_future_dataframe(periods=periods, freq=freq)
        forecast = model.predict(future)

        # Forecast duration
        last_date = df['ds'].max()
        forecast_end = future['ds'].max()
        duration = forecast_end - last_date
        st.markdown(f"🕒 Forecast extends **{duration.days} days and {duration.seconds // 3600} hours** into the future.")

        # Plotting
        st.subheader("Forecast Plot")
        fig = px.line(forecast, x='ds', y='yhat', title=f"{freq_option} Forecast")
        fig.add_scatter(x=df['ds'], y=df['y'], mode='lines', name='Actual')
        st.plotly_chart(fig, use_container_width=True)

        # Forecast Table (dynamic based on toggle)
        show_details = st.checkbox("Show full Forecast of Energy Consumption")

        forecast_future_only = forecast[forecast['ds'] > df['ds'].max()]
        forecast_full = forecast_future_only[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].rename(columns={
            'ds': 'Date / Time',
            'yhat': 'Forecast Energy Consumption',
            'yhat_lower': 'Lower Confidence Bound',
            'yhat_upper': 'Upper Confidence Bound'
        })

        if show_details:
            # Show last 24 actuals
            st.subheader("Last 24 Observed Hourly Consumptions")
            recent_actual = df[['ds', 'y']].rename(columns={
                'ds': 'Date / Time',
                'y': 'Actual Value'
            }).tail(24)
            st.dataframe(recent_actual)

            # Show full forecast
            st.subheader("All Forecasted Future Consumptions")
            st.dataframe(forecast_full)

            # Download full forecast
            csv = forecast_full.to_csv(index=False).encode()
            st.download_button(
                label="Download Full Forecast CSV",
                data=csv,
                file_name='forecast_full.csv',
                mime='text/csv'
            )
        else:
            # Show only last 24 predictions
            st.subheader("Last 24 Forecasted Hourly Consumptions")
            forecast_preview = forecast_full.tail(24)
            st.dataframe(forecast_preview)

            # Download limited forecast
            csv = forecast_preview.to_csv(index=False).encode()
            st.download_button(
                label="Download",
                data=csv,
                file_name='last_24_predictions.csv',
                mime='text/csv'
            )

        # Show components
        with st.expander("Forecast Components"):
            fig2 = model.plot_components(forecast)
            st.write(fig2)

    except Exception as e:
        st.error(f"Something went wrong: {e}")
else:
    st.info("Upload a time series file to get started.")

# In[ ]:




