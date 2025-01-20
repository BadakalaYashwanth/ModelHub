# ModelHub
# S&P 500 Stock Value Prediction Dashboard

This project provides a web-based dashboard for visualizing and predicting the stock values of companies listed in the S&P 500 index over time. The dashboard allows users to interactively select countries, companies, and date ranges to explore historical stock data and obtain predictions using multiple machine learning models.

# Features
* Interactive Country and Company Selector: Choose a country and a list of companies within that country from the S&P 500 dataset.
* Date Range Selector: Select the start and end dates to visualize stock data within the chosen period.
* Stock Value Graph: Displays historical stock values over the selected date range.
* Machine Learning Predictions: Provides stock value predictions using three different models:
** Linear Regression
K-Nearest Neighbors (KNN)
Recurrent Neural Network (RNN)


# Technologies Used
Taipy: For building the web-based user interface and orchestrating the machine learning pipeline.
Plotly: For plotting interactive stock value graphs.
Scikit-learn: For implementing Linear Regression and K-Nearest Neighbors algorithms.
TensorFlow: For implementing the Recurrent Neural Network (RNN).
Pandas: For data manipulation and processing.

# Requirements
Python 3.x
Taipy
Scikit-learn
TensorFlow
Plotly
Pandas

# Installation
To install the required dependencies, run the following command:
pip install taipy scikit-learn tensorflow plotly pandas

# File Structure
data/sp500_stocks.csv: Contains historical stock data for S&P 500 companies.
data/sp500_companies.csv: Contains company details like country, symbol, and short name.
images/icons/: Contains icons used for the dashboard interface.
main.py: The main application code that runs the dashboard and machine learning models.

# How It Works
Data Filtering: The stock data is filtered based on the selected date range and company. The filtered data is then displayed as a graph.
Prediction Models: The application uses three prediction models:
Linear Regression: Predicts future stock values based on historical data.
K-Nearest Neighbors (KNN): Predicts future stock values based on the closest data points in the training set.
Recurrent Neural Network (RNN): Uses historical data to predict future stock values with a deep learning approach.
User Interaction: Users can select the country, company, and date range, and the app will update the predictions and graph accordingly.


# Backend Architecture
The backend is powered by Taipy, which handles the orchestration of tasks such as:

Fetching company names based on the selected country.
Building the graph data by filtering stock data based on the selected date range and company.
Generating predictions using the Linear Regression, KNN, and RNN models.
The application uses scikit-learn for Linear Regression and KNN models and TensorFlow for building the RNN model. The data is preprocessed and normalized before being passed into the models for prediction.

# Predictions
Linear Regression (LR): A simple linear model that predicts the stock value based on historical data.
K-Nearest Neighbors (KNN): A non-parametric model that predicts the stock value by finding the most similar past data points.
Recurrent Neural Network (RNN): A deep learning model that captures the temporal nature of stock data for predictions.

# Running the Application
To start the application, run the following command:
taipy run main.py

This will launch a web server, and you can open the dashboard in your browser. The dashboard allows you to interactively select the country, company, and date range to visualize and predict stock values.

# Future Improvements
Integrating more advanced models like LSTM or ARIMA for time series forecasting.
Adding the ability to download the prediction results.
Adding more data sources for a broader set of companies or markets.
# Contributing
Feel free to fork the repository, open issues, or submit pull requests to contribute to this project.
