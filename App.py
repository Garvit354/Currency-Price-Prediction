import os
import streamlit as st
import numpy as np
import pandas as pd
import requests
from datetime import datetime, timedelta
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, LSTM, Dense, RepeatVector, TimeDistributed
from sklearn.preprocessing import MinMaxScaler
from io import StringIO
import plotly.graph_objs as go
import tensorflow as tf
import random
import tensorflow.keras.backend as K

# 🧠 Reproducibility
seed = 42
np.random.seed(seed)
random.seed(seed)
tf.random.set_seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
K.clear_session()

# 🔐 API Key
API_KEY = st.secrets.get("ALPHAVANTAGE_API_KEY")
if not API_KEY:
    st.error("API key missing. Check secrets.toml.")
    st.stop()

st.set_page_config(layout="wide")
st.title("💱 Final Diagnostic: Currency 10-Day Forecast (LSTM)")

currency_options = ["USD", "INR", "EUR", "GBP", "JPY"]
base = st.selectbox("Select Base Currency", currency_options)
target = st.selectbox("Select Target Currency", currency_options, index=1 if base == currency_options[0] else 0)
start_date = st.date_input("Prediction Start Date", datetime.now().date())

if base == target:
    st.error("Base and Target currencies must be different.")
else:
    if st.button("Run Final Diagnostic & Predict"):
        with st.spinner("Fetching data..."):
            url = f"https://www.alphavantage.co/query?function=FX_DAILY&from_symbol={base}&to_symbol={target}&outputsize=full&apikey={API_KEY}&datatype=csv"
            r = requests.get(url)

            if r.status_code != 200 or "timestamp" not in r.text:
                st.error("Error fetching data.")
                st.stop()

            df = pd.read_csv(StringIO(r.text))
            if 'timestamp' not in df.columns:
                st.error("Invalid API response.")
                st.stop()

            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp')
            start_dt = pd.to_datetime(start_date)
            df = df[df['timestamp'] < start_dt]
            df = df.tail(300)

            if len(df) < 70:
                st.error("Not enough historical data.")
                st.stop()

            # 📊 Diagnostic: Show training info
            st.write("📅 Training Data:")
            st.write("Date Range:", df['timestamp'].iloc[0], "→", df['timestamp'].iloc[-1])
            st.write("Rows used for training:", len(df))

            prices = df['close'].values.reshape(-1, 1)
            scaler = MinMaxScaler()
            scaled = scaler.fit_transform(prices)

            # Prepare x (last 60) → y (next 10)
            x, y = [], []
            for i in range(60, len(scaled) - 10):
                x.append(scaled[i - 60:i])
                y.append(scaled[i:i + 10])

            x = np.array(x)
            y = np.array(y)

        # Model: load or train
        model_path = "currency_seq2seq_model.h5"
        if os.path.exists(model_path):
            model = load_model(model_path)
            st.success("✅ Loaded saved model")
        else:
            st.info("🔁 Training model from scratch")
            inp = Input(shape=(60, 1))
            enc = LSTM(100, activation='relu')(inp)
            rep = RepeatVector(10)(enc)
            dec = LSTM(100, activation='relu', return_sequences=True)(rep)
            out = TimeDistributed(Dense(1))(dec)

            model = Model(inputs=inp, outputs=out)
            model.compile(optimizer='adam', loss='mse')
            model.fit(x, y, epochs=30, batch_size=32, verbose=0)
            model.save(model_path)
            st.success("✅ Model trained and saved")

        # Predict next 10 days
        last_60 = scaled[-60:].reshape(1, 60, 1)
        pred_scaled = model.predict(last_60, verbose=0)[0].reshape(10)
        pred_scaled = np.clip(pred_scaled, 0, 1)
        pred_actual = scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()

        # 🔍 Diagnostic Outputs
        st.write("🔎 Final 60-day input (scaled):", last_60.flatten())
        st.write("📈 Scaled prediction:", pred_scaled)
        st.write("💰 Predicted prices:", pred_actual)

        # Future dates and DataFrame
        future_dates = [start_dt + timedelta(days=i) for i in range(1, 11)]
        prediction_df = pd.DataFrame(pred_actual, index=future_dates, columns=['Predicted'])

        st.markdown(
            f"""
            🟢 **Next Day Predicted Price from {start_date}:**  
            <span style='font-weight:bold; font-size:28px; color:green;'>{pred_actual[0]:.4f}</span>
            """,
            unsafe_allow_html=True
        )

        # Plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['close'], mode='lines', name='Past 60-300 Days', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=future_dates, y=pred_actual, mode='lines+markers', name='Predicted (10 Days)', line=dict(color='orange', dash='dash')))

        last_price = df['close'].iloc[-1]
        fig.add_shape(type="line", x0=df['timestamp'].iloc[0], y0=last_price, x1=future_dates[-1], y1=last_price, line=dict(color="red", dash="dot"))

        fig.update_layout(title=f"{base}/{target} Prediction (Next 10 Days)", xaxis_title="Date", yaxis_title="Rate", hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(
            f"🔴 **Last Known Price ({df['timestamp'].iloc[-1].date()}):** "
            f"<span style='font-weight:bold; font-size:20px; color:red;'>{last_price:.4f}</span>",
            unsafe_allow_html=True
        )

        st.dataframe(prediction_df.reset_index().rename(columns={"index": "Date"}))
