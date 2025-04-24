# Stock Prediction App

A machine learning–powered web application for predicting future stock prices using historical data. Built with Python, Streamlit, and LSTM neural networks, this app fetches financial data, preprocesses it, trains a sequence model, and visualizes forecasts in an interactive dashboard.

## Features

- **Data Fetching**: Retrieves historical stock data from public APIs.
- **Preprocessing**: Cleans and scales data for model training.
- **LSTM Model**: Uses a pre-trained LSTM to forecast stock prices.
- **Interactive Dashboard**: Streamlit UI to select symbols, view past trends, and see predictions.
- **Testing**: Unit tests for data pipeline, model integrity, and end-to-end workflow.

## Tech Stack & Dependencies

- Python 3.10+  
- Streamlit  
- TensorFlow / Keras  
- pandas, NumPy  
- scikit-learn  
- yfinance or equivalent API for data  
- pytest for testing

## Installation

1. **Clone the repository**  
   ```bash
   git clone https://github.com/<your-username>/Stock-Prediction-App.git
   cd Stock-Prediction-App/Stock-Prediction-App
   ```
2. **Create & activate a virtual environment**  
   ```bash
   python -m venv venv
   source venv/bin/activate      # macOS/Linux
   venv\Scripts\activate       # Windows
   ```
3. **Install dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the Streamlit app:  
```bash
streamlit run app.py
```  
- Open the displayed URL in your browser.  
- Choose a stock ticker from the sidebar.  
- View historical price chart and future predictions.

## Project Structure

```
Stock-Prediction-App/
├── app.py               # Main Streamlit application
├── app1.py              # Alternative UI or testing interface
├── models.py            # LSTM model definition and utilities
├── stock_lists.py       # Supported ticker symbols
├── utils/
│   ├── data_fetcher.py  # Functions to fetch data from APIs
│   └── preprocess.py    # Data cleaning and scaling routines
├── data/                # Raw or cached datasets
├── models/              # Trained model files (e.g., lstm_model.h5)
├── test_data.py         # Tests for data fetcher and preprocessing
├── test_models.py       # Tests for model architecture and loading
├── test_pipeline.py     # Integration tests for full workflow
├── requirements.txt     # Project dependencies
└── README.md            # This file
```

## Running Tests

Execute all unit tests with pytest:  
```bash
pytest --maxfail=1 --disable-warnings -q
```

## Contributing

Contributions are welcome!  
1. Fork the repo.  
2. Create a feature branch (`git checkout -b feature/YourFeature`).  
3. Commit your changes (`git commit -m 'Add your feature'`).  
4. Push to the branch (`git push origin feature/YourFeature`).  
5. Open a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

