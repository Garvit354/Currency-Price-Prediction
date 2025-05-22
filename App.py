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

# Load API key from Streamlit secrets
API_KEY = st.secrets["ALPHAVANTAGE_API_KEY"]

st.set_page_config(layout="wide")
st.title("💱 Custom Currency Price Prediction (LSTM)")

# Currency options for dropdowns
currency_options = ["USD", "INR", "EUR", "GBP", "JPY"]

# Dropdowns for currency selection
base = st.selectbox("Select Base Currency", currency_options)
target = st.selectbox("Select Target Currency", currency_options, index=1 if base == currency_options[0] else 0)

# Date picker for start date of prediction
start_date = st.date_input("Select Start Date (for next 10-day prediction)", datetime.now().date())

if base == target:
    st.error("Base and Target currencies must be different.")
else:
    if st.button("Predict"):
        with st.spinner("Fetching data and training model..."):
            # Fetch FX daily data from Alpha Vantage
            url = f"https://www.alphavantage.co/query?function=FX_DAILY&from_symbol={base}&to_symbol={target}&outputsize=full&apikey={API_KEY}&datatype=csv"
            r = requests.get(url)
            if r.status_code != 200:
                st.error("Error fetching data from Alpha Vantage.")
                st.stop()

            # Load CSV data into DataFrame
            df = pd.read_csv(StringIO(r.text))

            # Debug info in Streamlit to check what data is loaded
            st.write("Raw data columns:", df.columns.tolist())  
            st.write(df.head())

            if 'timestamp' not in df.columns:
            st.error("Error: 'timestamp' column not found in the API response. Possibly API limit reached or invalid data returned.")
            st.stop()

            df['timestamp'] = pd.to_datetime(df['timestamp'])

            df = df.sort_values('timestamp')

            # Filter last 60 days before selected start date
            start_datetime = pd.to_datetime(start_date)
            df = df[df['timestamp'] < start_datetime]
            df = df.tail(60)

            if df.empty or len(df) < 60:
                st.error("Insufficient historical data available before the selected date.")
                st.stop()

            prices = df['close'].values.reshape(-1, 1)
            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(prices)

            # Prepare LSTM input sequences
            x = []
            for i in range(60, len(scaled_data) + 1):
                x.append(scaled_data[i - 60:i])

            x = np.array(x)

            # Build LSTM model
            model = Sequential([
                LSTM(50, return_sequences=True, input_shape=(x.shape[1], 1)),
                LSTM(50),
                Dense(1)
            ])
            model.compile(optimizer='adam', loss='mean_squared_error')

            # Create dummy y to fit model shape
            y_dummy = scaled_data[60 - 1:]
            model.fit(x, y_dummy, epochs=10, batch_size=32, verbose=0)

            # Predict next 10 days
            future_prices = []
            input_seq = scaled_data.copy()
            for _ in range(10):
                next_input = input_seq[-60:].reshape(1, 60, 1)
                pred_scaled = model.predict(next_input)[0][0]
                future_prices.append(pred_scaled)
                input_seq = np.append(input_seq, [[pred_scaled]], axis=0)

            # Inverse scale predicted prices
            predicted_prices = scaler.inverse_transform(np.array(future_prices).reshape(-1, 1)).flatten()

            # Create dates for predicted prices
            future_dates = [start_datetime + timedelta(days=i) for i in range(1, 11)]
            prediction_df = pd.DataFrame(predicted_prices, index=future_dates, columns=['Predicted'])

            # Display next day prediction with bold and bigger font
            st.markdown(
                f"""
                📈 Next Day Prediction from {start_date}:  
                <span style="font-weight:bold; font-size:28px; color:green;">{predicted_prices[0]:.4f}</span>
                """,
                unsafe_allow_html=True
            )

            # Plot interactive graph using Plotly
            fig = go.Figure()

            # Past data trace
            fig.add_trace(go.Scatter(
                x=df['timestamp'],
                y=df['close'],
                mode='lines',
                name='Past 60 Days',
                line=dict(color='blue')
            ))

            # Predicted data trace
            fig.add_trace(go.Scatter(
                x=future_dates,
                y=predicted_prices,
                mode='lines+markers',
                name='LSTM Forecast',
                line=dict(color='orange', dash='dash'),
                marker=dict(size=6)
            ))

            # Horizontal line for last known price
            last_price = df['close'].iloc[-1]
            fig.add_shape(
                type="line",
                x0=df['timestamp'].iloc[0],
                y0=last_price,
                x1=future_dates[-1],
                y1=last_price,
                line=dict(color="red", width=1, dash="dot")
            )

            # Update layout for readability and hover behavior
            fig.update_layout(
                title=f"{base}/{target} Forecast (Next 10 Days)",
                xaxis_title="Date",
                yaxis_title="Exchange Rate",
                hovermode="x unified"
            )

            # Show plot in Streamlit
            st.plotly_chart(fig, use_container_width=True)

            # Show current price below the chart with styled text
            st.markdown(
                f"🔴 **Current (Last Known) Price as of {df['timestamp'].iloc[-1].date()}:** "
                f"<span style='font-weight:bold; font-size:20px; color:red;'>{last_price:.4f}</span>",
                unsafe_allow_html=True
            )

            # Display full prediction table
            st.dataframe(prediction_df.reset_index().rename(columns={"index": "Date"}))
