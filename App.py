import os
import streamlit as st
import numpy as np
import pandas as pd
import requests
from datetime import datetime, timedelta
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from io import StringIO
import plotly.graph_objs as go
import tensorflow as tf
import random

# 🔒 Set random seed for consistent results
seed = 42
np.random.seed(seed)
random.seed(seed)
tf.random.set_seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)

# Load API key from Streamlit secrets
API_KEY = st.secrets["ALPHAVANTAGE_API_KEY"]

st.set_page_config(layout="wide")
st.title("💱 Custom Currency Price Prediction (LSTM)")

# Currency options
currency_options = ["USD", "INR", "EUR", "GBP", "JPY"]
base = st.selectbox("Select Base Currency", currency_options)
target = st.selectbox("Select Target Currency", currency_options, index=1 if base == currency_options[0] else 0)
start_date = st.date_input("Select Start Date (for next 10-day prediction)", datetime.now().date())

if base == target:
    st.error("Base and Target currencies must be different.")
else:
    if st.button("Predict"):
        with st.spinner("Fetching data and training model..."):
            url = f"https://www.alphavantage.co/query?function=FX_DAILY&from_symbol={base}&to_symbol={target}&outputsize=full&apikey={API_KEY}&datatype=csv"
            r = requests.get(url)

            if r.status_code != 200 or "timestamp" not in r.text:
                st.error("Error fetching data from Alpha Vantage. Try again later.")
                st.stop()

            df = pd.read_csv(StringIO(r.text))

            # Debug: show columns and dates
            st.write("Raw data columns:", df.columns.tolist())
            if 'timestamp' not in df.columns:
                st.error("Missing 'timestamp' column. API may be rate-limited.")
                st.stop()

            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp')

            start_datetime = pd.to_datetime(start_date)
            df = df[df['timestamp'] < start_datetime]
            df = df.tail(60)

            st.write("Prediction start date:", start_datetime)
            if not df.empty:
                st.write("Last available data date:", df['timestamp'].max())

            if df.empty or len(df) < 60:
                st.error("Insufficient historical data available before the selected date.")
                st.stop()

            prices = df['close'].values.reshape(-1, 1)
            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(prices)

            # Prepare LSTM input
            x = [scaled_data[i - 60:i] for i in range(60, len(scaled_data) + 1)]
            x = np.array(x)

            # LSTM model
            model = Sequential([
                LSTM(50, return_sequences=True, input_shape=(x.shape[1], 1)),
                LSTM(50),
                Dense(1)
            ])
            model.compile(optimizer='adam', loss='mean_squared_error')

            y_dummy = scaled_data[60 - 1:]
            model.fit(x, y_dummy, epochs=25, batch_size=32, verbose=0)

            # Forecast 10 future days
            future_prices = []
            input_seq = scaled_data.copy()
            for _ in range(10):
                next_input = input_seq[-60:].reshape(1, 60, 1)
                pred_scaled = model.predict(next_input, verbose=0)[0][0]
                future_prices.append(pred_scaled)
                input_seq = np.append(input_seq, [[pred_scaled]], axis=0)

            # ✅ Clip values before inverse transform
            future_prices = np.clip(future_prices, 0, 1)
            predicted_prices = scaler.inverse_transform(np.array(future_prices).reshape(-1, 1)).flatten()

            # Create prediction DataFrame
            future_dates = [start_datetime + timedelta(days=i) for i in range(1, 11)]
            prediction_df = pd.DataFrame(predicted_prices, index=future_dates, columns=['Predicted'])

            # Display next-day result
            st.markdown(
                f"""
                📈 Next Day Prediction from {start_date}:  
                <span style="font-weight:bold; font-size:28px; color:green;">{predicted_prices[0]:.4f}</span>
                """,
                unsafe_allow_html=True
            )

            # Plot
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df['timestamp'], y=df['close'], mode='lines', name='Past 60 Days', line=dict(color='blue')))
            fig.add_trace(go.Scatter(x=future_dates, y=predicted_prices, mode='lines+markers', name='LSTM Forecast', line=dict(color='orange', dash='dash')))
            last_price = df['close'].iloc[-1]
            fig.add_shape(type="line", x0=df['timestamp'].iloc[0], y0=last_price, x1=future_dates[-1], y1=last_price, line=dict(color="red", width=1, dash="dot"))

            fig.update_layout(title=f"{base}/{target} Forecast (Next 10 Days)", xaxis_title="Date", yaxis_title="Exchange Rate", hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)

            # Display last known price
            st.markdown(
                f"🔴 **Current (Last Known) Price as of {df['timestamp'].iloc[-1].date()}:** "
                f"<span style='font-weight:bold; font-size:20px; color:red;'>{last_price:.4f}</span>",
                unsafe_allow_html=True
            )

            # Show full table
            st.dataframe(prediction_df.reset_index().rename(columns={"index": "Date"}))
