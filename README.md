# 📈 ModelHub — Professional Stock Price Predictor

> A machine-learning powered web application that forecasts future stock prices using LSTM neural networks. Supports **Indian (NSE)** and **US** markets with an interactive, professional-grade Streamlit dashboard.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-red?logo=streamlit)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12%2B-orange?logo=tensorflow)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 🖼️ Screenshots

| Dashboard | Prediction Chart |
|-----------|-----------------|
| ![Dashboard](https://github.com/user-attachments/assets/95319b22-648d-4ba6-aa69-05746d99443d) | ![Chart](https://github.com/user-attachments/assets/9f391de9-b4f4-4090-9990-78b32b227b78) |

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🌏 **Dual Market Support** | Trade Indian (NSE — Nifty 50 + Nifty Next 50) and US (S&P 500 + Nasdaq) stocks |
| 🤖 **LSTM Predictions** | Pre-trained LSTM model forecasts up to **30 days** ahead |
| 📊 **Candlestick Charts** | Interactive Plotly dark-theme charts with zoom, pan, and hover tooltips |
| 📉 **Technical Indicators** | RSI, SMA20/50, EMA20, Bollinger Bands computed on-the-fly |
| 🔄 **Live Data** | Pulls real-time historical data via `yfinance` at every run |
| 🗂️ **Stock Library** | 200+ symbols — Nifty 100 Indian stocks + top US equities |
| ✅ **Test Suite** | Unit tests for data pipeline, model loading, and end-to-end workflow |

---

## 🏗️ Project Structure

```
ModelHub/
├── app.py               # 🚀 Main Streamlit application (production UI)
├── app1.py              # 🧪 Alternate/slim version for local testing
├── models.py            # 🧠 LSTM model architecture (build_lstm_model)
├── stock_lists.py       # 📋 All supported Indian & US ticker symbols
├── lstm_model.h5        # 💾 Pre-trained LSTM weights
├── history.sqlite       # 🗄️  Local prediction history (SQLite)
├── requirements.txt     # 📦 Python dependencies
│
├── utils/
│   ├── __init__.py      # Package initialiser
│   ├── data_fetcher.py  # 📡 yfinance wrappers — fetch_indian_stock / fetch_us_stock
│   └── preprocess.py    # 🔧 Technical indicators + LSTM sequence builder
│
├── test_data.py         # 🧪 Tests: data fetching correctness
├── test_models.py       # 🧪 Tests: model architecture loading
├── test_pipeline.py     # 🧪 Tests: full train-predict pipeline
│
└── README.md
```

---

## ⚙️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **UI** | [Streamlit](https://streamlit.io/) |
| **Charts** | [Plotly](https://plotly.com/python/) |
| **ML Model** | TensorFlow / Keras — 2-layer LSTM |
| **Data** | [yfinance](https://github.com/ranaroussi/yfinance) |
| **Feature Engineering** | pandas, NumPy, scikit-learn |
| **Testing** | pytest |

---

## 🚀 Quick Start

### 1 — Clone the repository
```bash
git clone https://github.com/BadakalaYashwanth/ModelHub.git
cd ModelHub
```

### 2 — Create & activate a virtual environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python -m venv venv
source venv/bin/activate
```

### 3 — Install dependencies
```bash
pip install -r requirements.txt
```

### 4 — Run the app
```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🕹️ How to Use

1. **Select Market** — choose *Indian* or *US* from the sidebar radio button.
2. **Pick a Stock** — choose any ticker from the dropdown (200+ symbols available).
3. **Set Forecast Horizon** — drag the slider (1 – 30 days).
4. **Toggle RSI** — checkbox to show/hide the RSI overlay on the chart.
5. **Click Predict** — the app fetches live data, runs the LSTM, and renders:
   - A candlestick chart with colour-coded prediction lines (🟢 up / 🔴 down).
   - A sidebar summary with `Day N Forecast` metrics showing price + % change.
6. **Show Historical Data** — checkbox to view the raw OHLCV dataframe.

---

## 🧠 Model Architecture

```
Input  →  LSTM(50, return_sequences=True)
       →  Dropout(0.2)
       →  LSTM(50, return_sequences=False)
       →  Dropout(0.2)
       →  Dense(25)
       →  Dense(1)            ← Predicted next-day close (scaled)
```

- **Sequence length**: 60 trading days
- **Loss function**: Mean Squared Error
- **Optimiser**: Adam
- **Training script**: `test_pipeline.py` (re-trains and saves `lstm_model.h5`)

---

## 🔧 Technical Indicators (utils/preprocess.py)

| Indicator | Window | Description |
|-----------|--------|-------------|
| RSI | 14 | Relative Strength Index — momentum oscillator |
| SMA20 / SMA50 | 20 / 50 | Simple Moving Averages — trend direction |
| EMA20 | 20 | Exponential Moving Average — faster trend signal |
| BB_upper / BB_lower | 20 ± 2σ | Bollinger Bands — volatility envelope |

---

## 🧪 Running Tests

```bash
# Run all tests
pytest --maxfail=1 --disable-warnings -q

# Individual test files
python test_data.py        # Verify data fetching
python test_models.py      # Verify model loads
python test_pipeline.py    # Full train + predict pipeline (re-trains the model)
```

---

## 📦 requirements.txt

```
streamlit>=1.32.0
tensorflow>=2.12.0
keras>=2.12.0
numpy>=1.24.0
pandas>=2.0.0
plotly>=5.18.0
scikit-learn>=1.3.0
yfinance>=0.2.36
requests>=2.31.0
pytest>=7.4.0
```

---

## ☁️ Deploying (Streamlit Community Cloud)

> ⚠️ **Note**: Netlify only hosts static sites and will return a 404 for Python apps. Use Streamlit Community Cloud for free hosting.

1. Push your code to GitHub (done ✅).
2. Go to [share.streamlit.io](https://share.streamlit.io/) and sign in with GitHub.
3. Click **New app** → select this repo → set **Main file path** to `app.py`.
4. Click **Deploy** — the app will be live in ~2 minutes.

---

## 🤝 Contributing

Contributions are welcome!

1. Fork the repo.
2. Create a feature branch: `git checkout -b feature/YourFeature`
3. Commit your changes: `git commit -m 'Add YourFeature'`
4. Push: `git push origin feature/YourFeature`
5. Open a Pull Request.

---

## 📄 License

This project is licensed under the **MIT License**.

---

> Built with ❤️ by [BadakalaYashwanth](https://github.com/BadakalaYashwanth)
