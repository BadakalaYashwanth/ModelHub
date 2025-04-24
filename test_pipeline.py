from utils.data_fetcher import fetch_indian_stock
from utils.preprocess import add_technical_indicators, preprocess_for_lstm
from models import build_lstm_model
import numpy as np

# Fetch and preprocess data
ticker = "RELIANCE.NS"  # Example stock
stock_data = fetch_indian_stock(ticker, period="5y")
processed_data = add_technical_indicators(stock_data)

# Prepare LSTM data
X, y, scaler = preprocess_for_lstm(processed_data)
print(f"Data shape - X: {X.shape}, y: {y.shape}")  # Should be (n_samples, 60, 1)

# Build and train the model
model = build_lstm_model((60, 1))  # Input shape = (sequence_length, features)
history = model.fit(X, y, epochs=50, batch_size=32, validation_split=0.2)

# Save the trained model
model.save("lstm_model.h5")
print("Model saved as 'lstm_model.h5'")

# Predict the next day's price
last_sequence = X[-1].reshape(1, 60, 1)  # Use the most recent 60 days
predicted_scaled = model.predict(last_sequence)
predicted_price = scaler.inverse_transform(predicted_scaled)[0][0]  # Convert to actual price

print(f"Predicted next day closing price for {ticker}: ₹{predicted_price:.2f}")