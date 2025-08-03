import os
import streamlit as st
import numpy as np
import pandas as pd
import requests
from datetime import datetime, timedelta
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from io import StringIO
import plotly.graph_objs as go
import tensorflow as tf
import random

# Set seeds for reproducibility
seed = 42
np.random.seed(seed)
random.seed(seed)
tf.random.set_seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)

# Load API key securely
API_KEY = st.secrets.get("ALPHAVANTAGE_API_KEY")
if not API_KEY:
    st.error("API key missing. Check your secrets.toml.")
    st.stop()

st.set_page_config(layout="wide")
st.title("💱 Currency Price Prediction (Stable LSTM)")

currency_options = ["USD", "INR", "EUR", "GBP", "JPY"]
base = st.selectbox("Select Base Currency", currency_options)
target = st.selectbox("Select Target Currency", currency_options, index=1 if base == currency_options[0] else 0)
start_date = st.date_input("Prediction Start Date", datetime.now().date())

if base == target:
    st.error("Base and Target currencies must be different.")
else:
    if st.button("Predict"):
        with st.spinner("Fetching and preparing data..."):
            url = f"https://www.alphavantage.co/query?function=FX_DAILY&from_symbol={base}&to_symbol={target}&outputsize=full&apikey={API_KEY}&datatype=csv"
            r = requests.get(url)

            if r.status_code != 200 or "timestamp" not in r.text:
                st.error("API error or invalid response.")
                st.stop()

            df = pd.read_csv(StringIO(r.text))
            if 'timestamp' not in df.columns:
                st.error("Missing timestamp column. API limit may have been reached.")
                st.stop()

            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp')
            start_dt = pd.to_datetime(start_date)
            df = df[df['timestamp'] < start_dt]
            df = df.tail(300)  # use last 300 days

            if df.empty or len(df) < 60:
                st.error("Not enough data available.")
                st.stop()

            prices = df['close'].values.reshape(-1, 1)
            scaler = MinMaxScaler()
            scaled = scaler.fit_transform(prices)

            # Prepare training data
            x, y = [], []
            for i in range(60, len(scaled)):
                x.append(scaled[i - 60:i])
                y.append(scaled[i])

            x = np.array(x)
            y = np.array(y)

        # Model training or loading
        model_path = "currency_model.h5"
        if os.path.exists(model_path):
            model = load_model(model_path)
            st.success("✅ Loaded pre-trained model.")
        else:
            st.info("🔁 Training model for the first time...")
            model = Sequential([
                LSTM(50, return_sequences=True, input_shape=(60, 1)),
                LSTM(50),
                Dense(1)
            ])
            model.compile(optimizer='adam', loss='mean_squared_error')
            model.fit(x, y, epochs=30, batch_size=32, verbose=0)
            model.save(model_path)
            st.success("✅ Model trained and saved!")

        # Predict next day
        last_60 = scaled[-60:]
        input_seq = last_60.reshape(1, 60, 1)
        pred_scaled = model.predict(input_seq, verbose=0)[0][0]
        pred_scaled = np.clip(pred_scaled, 0, 1)
        pred_actual = scaler.inverse_transform([[pred_scaled]])[0][0]

        # Show prediction
        st.markdown(
            f"""
            📈 **Next Day Prediction from {start_date}:**  
            <span style="font-weight:bold; font-size:28px; color:green;">{pred_actual:.4f}</span>
            """,
            unsafe_allow_html=True
        )

        # Plot historical and prediction
        future_date = start_dt + timedelta(days=1)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['close'], mode='lines', name='Historical Data', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=[future_date], y=[pred_actual], mode='markers+text', name='Predicted', marker=dict(color='orange', size=10), text=["Predicted"], textposition="top center"))
        fig.add_shape(type="line", x0=df['timestamp'].iloc[0], y0=prices[-1][0], x1=future_date, y1=prices[-1][0], line=dict(color="red", dash="dot"))

        fig.update_layout(title=f"{base}/{target} - Next Day Forecast", xaxis_title="Date", yaxis_title="Exchange Rate", hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(
            f"🔴 **Last Known Price ({df['timestamp'].iloc[-1].date()}):** "
            f"<span style='font-weight:bold; font-size:20px; color:red;'>{prices[-1][0]:.4f}</span>",
            unsafe_allow_html=True
        )

        st.write("📊 Debug — Predicted Value:", pred_actual)
