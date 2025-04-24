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
INDIAN_STOCKS = [
    # Nifty 50
    'ADANIENT.NS', 'ADANIPORTS.NS', 'ASIANPAINT.NS', 'AXISBANK.NS', 'BAJAJ-AUTO.NS',
    'BAJFINANCE.NS', 'BAJAJFINSV.NS', 'BPCL.NS', 'BHARTIARTL.NS', 'BRITANNIA.NS',
    'CIPLA.NS', 'COALINDIA.NS', 'DIVISLAB.NS', 'DRREDDY.NS', 'EICHERMOT.NS',
    'GRASIM.NS', 'HCLTECH.NS', 'HDFCBANK.NS', 'HDFCLIFE.NS', 'HEROMOTOCO.NS',
    'HINDALCO.NS', 'HINDUNILVR.NS', 'ICICIBANK.NS', 'INDUSINDBK.NS', 'INFY.NS',
    'ITC.NS', 'JSWSTEEL.NS', 'KOTAKBANK.NS', 'LT.NS', 'M&M.NS',
    'MARUTI.NS', 'NESTLEIND.NS', 'NTPC.NS', 'ONGC.NS', 'POWERGRID.NS',
    'RELIANCE.NS', 'SBILIFE.NS', 'SBIN.NS', 'SUNPHARMA.NS', 'TATACONSUM.NS',
    'TATAMOTORS.NS', 'TATASTEEL.NS', 'TECHM.NS', 'TITAN.NS', 'TCS.NS',
    'ULTRACEMCO.NS', 'UPL.NS', 'WIPRO.NS', 'HINDPETRO.NS', 'BAJAJHLDNG.NS',

    # Nifty Next 50
    'ABB.NS', 'ADANIGREEN.NS', 'ADANITRANS.NS', 'ALKEM.NS', 'AMBUJACEM.NS',
    'AUROPHARMA.NS', 'BANKBARODA.NS', 'BERGEPAINT.NS', 'BIOCON.NS', 'BOSCHLTD.NS',
    'CANBK.NS', 'CHOLAFIN.NS', 'DABUR.NS', 'DLF.NS', 'GAIL.NS',
    'GODREJCP.NS', 'HAVELLS.NS', 'ICICIPRULI.NS', 'IGL.NS', 'INDIGO.NS',
    'L&TFH.NS', 'LICI.NS', 'MCDOWELL-N.NS', 'MOTHERSUMI.NS', 'NMDC.NS',
    'PEL.NS', 'PGHH.NS', 'PIDILITIND.NS', 'PIIND.NS', 'PNB.NS',
    'RECLTD.NS', 'SAIL.NS', 'SHREECEM.NS', 'SIEMENS.NS', 'SRF.NS',
    'TORNTPHARM.NS', 'TRENT.NS', 'TVSMOTOR.NS', 'UBL.NS', 'VEDL.NS',
    'VOLTAS.NS', 'YESBANK.NS', 'ZEEL.NS', 'IDFCFIRSTB.NS', 'INDUSTOWER.NS'
]

US_STOCKS = [
    'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'TSLA', 'NVDA', 'BRK.B', 'UNH', 'JNJ',
    'JPM', 'V', 'PG', 'MA', 'HD', 'XOM', 'LLY', 'CVX', 'ABBV', 'AVGO',
    'PEP', 'KO', 'MRK', 'COST', 'WMT', 'BAC', 'ADBE', 'CRM', 'ACN', 'TMO',
    'INTC', 'NKE', 'LIN', 'ABT', 'NEE', 'MCD', 'DHR', 'WFC', 'TXN', 'UNP',
    'MDT', 'MS', 'QCOM', 'HON', 'PM', 'LOW', 'IBM', 'AMGN', 'SBUX', 'INTU',

    # Nasdaq top
    'AMD', 'PYPL', 'BKNG', 'AMAT', 'CSCO', 'ADP', 'ISRG', 'VRTX', 'GILD', 'ADI',
    'REGN', 'ZM', 'MAR', 'LRCX', 'IDXX', 'ASML', 'PANW', 'CTAS', 'CDNS', 'FISV',
    'SNPS', 'EXC', 'EA', 'ILMN', 'ROST', 'NXPI', 'ORLY', 'KLAC', 'FAST', 'MNST',
    'CHTR', 'PAYX', 'MTCH', 'BIIB', 'WBA', 'ALGN', 'KHC', 'BIDU', 'NTES', 'DOCU',
    'OKTA', 'TEAM', 'CRWD', 'ZS', 'SPLK', 'DDOG', 'MDB', 'PLTR', 'ROKU', 'U'
]

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
    display_days = min(60, len(historical)) if len(historical) > 0 else 0
    if display_days < 60:
        st.warning(f"Insufficient historical data (only {len(historical)} points available)")
    
    # Candlestick for historical data
    if len(historical) > 0:
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
            # Get scalar values for comparison
            current_close = float(historical['Close'].iloc[-1]) if i == 1 else float(predictions[i-2])
            pred_value = float(predictions[i-1])
            
            color = 'green' if pred_value > current_close else 'red'
            fig.add_trace(go.Scatter(
                x=[pred_dates[i-1], pred_dates[i]],
                y=[current_close, pred_value],
                line=dict(color=color, width=3),
                name='Prediction',
                showlegend=(i==1)
            ))
    
    # RSI indicator if available
    if show_rsi and 'RSI' in historical.columns and len(historical) > 0:
        fig.add_trace(go.Scatter(
            x=historical.index[-display_days:],
            y=historical['RSI'][-display_days:],
            name='RSI',
            yaxis='y2',
            line=dict(color='purple', width=1)
        ))
    
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
        ) if show_rsi and 'RSI' in historical.columns else {},
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
                    try:
                        # Get current price as float
                        current_price = float(historical['Close'].iloc[-1])
                        price_str = f"${current_price:.2f}" if market == 'US' else f"₹{current_price:.2f}"
                        st.metric("Current Price", price_str)
                        
                        # Display predictions
                        for i, price in enumerate(predictions, 1):
                            try:
                                # Convert all values to float explicitly
                                pred_price = float(price)
                                current_close_value = float(current_price if i == 1 else predictions[i-2])
                                delta = float(pred_price - current_close_value)
                                
                                # Safe percentage calculation
                                delta_pct = 0.0 if current_close_value == 0 else float((delta / current_close_value) * 100)
                                
                                # Format the delta display
                                delta_display = f"{delta:.2f} ({delta_pct:.2f}%)"
                                
                                st.metric(
                                    f"Day {i} Forecast", 
                                    f"${pred_price:.2f}" if market == 'US' else f"₹{pred_price:.2f}",
                                    delta_display,
                                    delta_color="normal"
                                )
                            except (ValueError, TypeError, IndexError) as e:
                                st.error(f"Error processing prediction {i}: {str(e)}")
                    except Exception as e:
                        st.error(f"Error displaying results: {str(e)}")

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