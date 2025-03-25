Stock Sentiment Analysis and Portfolio Optimization

Overview

This project analyzes financial news sentiment, social media trends, and stock price data to assess their impact on market movements. It integrates sentiment analysis with machine learning models to classify stocks as bullish or bearish and applies predictive modeling to forecast stock price trends. Additionally, it utilizes Modern Portfolio Theory and Monte Carlo simulations to optimize an investment portfolio.

Features

Sentiment Analysis: Utilizes VADER and TextBlob to extract sentiment scores from financial news headlines.

Predictive Modeling:

Classification Models: Implements Linear Discriminant Analysis (LDA), Random Forest, and XGBoost to classify stocks.

Time Series Forecasting: Uses ARIMA to predict stock price trends.

Portfolio Optimization:

Conducts Monte Carlo simulations to optimize stock weight allocations based on risk-return analysis.

Visualizes the efficient frontier for portfolio selection.

Technologies Used

Programming Language: Python

Libraries:

Data Processing: Pandas, NumPy

Sentiment Analysis: VADER SentimentIntensityAnalyzer, TextBlob

Machine Learning: Scikit-learn, XGBoost

Time Series Forecasting: Statsmodels (ARIMA)

Data Visualization: Matplotlib, Seaborn

Dataset

The project uses two datasets:

Financial News Data: Contains daily headlines relevant to stock market movements.

Stock Market Data: Includes stock price movements (Open, Close, High, Low, Volume) over time.

How It Works

Data Preprocessing

Merges financial news data with stock price data on date.

Cleans and processes text for sentiment analysis.

Sentiment Analysis

Computes Subjectivity and Polarity scores using TextBlob.

Generates sentiment intensity scores (Positive, Neutral, Negative, Compound) using VADER.

Stock Classification

Extracts relevant features and labels stocks as bullish or bearish.

Splits data into training and testing sets.

Trains LDA, Random Forest, and XGBoost models.

Evaluates model performance using classification reports.

Stock Price Forecasting

Implements ARIMA for time series forecasting.

Plots predicted stock price trends.

Portfolio Optimization

Simulates 10,000 portfolio allocations using Monte Carlo methods.

Computes expected returns and volatility.

Plots the efficient frontier for optimized portfolio selection.

Installation & Requirements

Install the required Python libraries:

pip install pandas numpy textblob vaderSentiment scikit-learn xgboost statsmodels matplotlib seaborn

Download and prepare the dataset files (dow_jones_industrial_average_news.csv, dow_jones_industrial_average_stock.csv).

Run the script in a Python environment (e.g., Jupyter Notebook, Google Colab, or command line).

Results & Insights

The sentiment analysis helps in identifying market trends based on news headlines.

Machine learning models provide a robust framework for classifying stock movements.

ARIMA forecasting provides insights into potential future stock price trends.

Portfolio optimization enables better decision-making by balancing risk and return.

Future Improvements

Expand dataset to include social media sentiment (e.g., Twitter, Reddit sentiment analysis).

Improve forecasting by incorporating additional technical indicators.

Enhance portfolio optimization by integrating asset correlation and diversification strategies.

License

This project is open-source and available for educational and research purposes.
