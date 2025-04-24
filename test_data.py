from utils.data_fetcher import fetch_indian_stock

# Test Reliance (NSE)
reliance = fetch_indian_stock("RELIANCE.NS", period="1y")
print("\nReliance Data Columns:", reliance.columns)
print("Shape:", reliance.shape)

# Test TCS (NSE)
tcs = fetch_indian_stock("TCS.NS", period="1y")
print("\nTCS Data Columns:", tcs.columns)
print("Shape:", tcs.shape)
