import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from tensorflow.keras.models import load_model
from utils.data_fetcher import fetch_indian_stock, fetch_us_stock
from utils.preprocess import add_technical_indicators, preprocess_for_lstm
from datetime import datetime, timedelta

# Page setup
st.set_page_config(page_title="Pro Stock Predictor", layout="wide")
st.title("📈 Professional Stock Price Predictor")

# Load stock lists
INDIAN_STOCKS = ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'ICICIBANK.NS']  # Add more
US_STOCKS = ['AAPL', 'MSFT', 'AMZN', 'GOOGL']  # Add more

# Sidebar controls
market = st.sidebar.radio("Market", ('Indian', 'US'))
ticker = st.sidebar.selectbox(
    "Select Stock",
    INDIAN_STOCKS if market == 'Indian' else US_STOCKS
)
days_to_predict = st.sidebar.slider("Days to Predict", 1, 30, 7)
show_rsi = st.sidebar.checkbox("Show RSI", True)

@st.cache_resource
def load_lstm_model():
    return load_model('lstm_model.h5')

def create_professional_chart(historical, predictions):
    """Create interactive trading chart with indicators"""
    fig = go.Figure()
    
    # Ensure we have enough historical data
    if len(historical) < 60:
        st.warning("Insufficient historical data for full analysis")
        display_days = len(historical)
    else:
        display_days = 60
    
    # Candlestick for historical data
    fig.add_trace(go.Candlestick(
        x=historical.index[-display_days:],
        open=historical['Open'][-display_days:],
        high=historical['High'][-display_days:],
        low=historical['Low'][-display_days:],
        close=historical['Close'][-display_days:],
        name='Price'
    ))
    
    # Prediction line with bull/bear colors
    if predictions and len(historical) > 0:
        pred_dates = [historical.index[-1] + timedelta(days=i) for i in range(days_to_predict+1)]
        for i in range(1, len(predictions)+1):
            current_close = historical['Close'].iloc[-1] if i==1 else predictions[i-2]
            color = 'green' if predictions[i-1] > current_close else 'red'
            fig.add_trace(go.Scatter(
                x=[pred_dates[i-1], pred_dates[i]],
                y=[current_close, predictions[i-1]],
                line=dict(color=color, width=3),
                name='Prediction',
                showlegend=(i==1)
            ))
    
    # RSI indicator if available
    if show_rsi and 'RSI' in historical.columns:
        fig.add_trace(go.Scatter(
            x=historical.index[-display_days:],
            y=historical['RSI'][-display_days:],
            name='RSI',
            yaxis='y2',
            line=dict(color='purple', width=1)
        )
    
    fig.update_layout(
        title=f'{ticker} Price Prediction',
        yaxis_title='Price',
        xaxis_rangeslider_visible=False,
        hovermode='x unified',
        yaxis2=dict(
            title='RSI',
            overlaying='y',
            side='right',
            range=[0, 100]
        ),
        template='plotly_dark',
        height=600
    )
    return fig

def predict_stock():
    try:
        # Fetch data based on market
        fetch_func = fetch_indian_stock if market == 'Indian' else fetch_us_stock
        data = fetch_func(ticker)
        
        if data is None or data.empty:
            st.error("Failed to fetch data. Try another ticker!")
            return None, None
        
        # Ensure consistent column names
        if 'Adj Close' in data.columns and 'Close' not in data.columns:
            data['Close'] = data['Adj Close']
        
        processed_data = add_technical_indicators(data)
        X, _, scaler = preprocess_for_lstm(processed_data)
        
        if len(X) == 0:
            st.error("Not enough data to make predictions")
            return None, None
            
        # Make predictions
        model = load_lstm_model()
        last_sequence = X[-1].reshape(1, 60, 1)
        predictions = []
        
        for _ in range(days_to_predict):
            pred = model.predict(last_sequence, verbose=0)
            predictions.append(scaler.inverse_transform(pred)[0][0])
            last_sequence = np.roll(last_sequence, -1, axis=1)
            last_sequence[0, -1, 0] = pred[0][0]
        
        return processed_data, predictions
        
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")
        return None, None

# Main app flow
if st.sidebar.button("Predict"):
    with st.spinner("Generating professional analysis..."):
        historical, predictions = predict_stock()
        
        if historical is not None and predictions is not None:
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.plotly_chart(
                    create_professional_chart(historical, predictions), 
                    use_container_width=True
                )
            
            with col2:
                st.subheader("Prediction Summary")
                if not historical.empty:
                    st.metric("Current Price", f"${historical['Close'].iloc[-1]:.2f}" if market == 'US' else f"₹{historical['Close'].iloc[-1]:.2f}")
                    
                    for i, price in enumerate(predictions, 1):
                        current_close = historical['Close'].iloc[-1] if i==1 else predictions[i-2]
                        delta = price - current_close
                        st.metric(
                            f"Day {i} Forecast", 
                            f"${price:.2f}" if market == 'US' else f"₹{price:.2f}",
                            f"{delta:.2f} ({delta/current_close*100:.2f}%)",
                            delta_color="normal"
                        )

# Historical data toggle
if st.sidebar.checkbox("Show Historical Data"):
    fetch_func = fetch_indian_stock if market == 'Indian' else fetch_us_stock
    data = fetch_func(ticker)
    if data is not None:
        st.dataframe(data.tail())
    else:
        st.error("No data available")

# Refresh note
st.sidebar.caption(f"Data as of {datetime.now().strftime('%Y-%m-%d %H:%M')}")