# Time Series Portfolio Optimization

Welcome to the **Time Series Portfolio Optimization** project! This project uses time series forecasting models to enhance portfolio management for Guide Me in Finance (GMF) Investments, a financial advisory firm focused on optimizing asset allocation and maximizing returns for clients.

## Project Overview

The goal of this project is to build predictive models that forecast financial trends and guide portfolio adjustments. Using historical data from assets like Tesla (TSLA), Vanguard Total Bond Market ETF (BND), and the S&P 500 ETF (SPY), the project aims to recommend data-backed portfolio adjustments based on predicted trends and risk assessments.

## Repository Structure
```bash
Time_Series_Portfolio_Optimization/
├── data/
│   ├── raw/                   # Raw datasets downloaded from YFinance
│   ├── processed/             # Preprocessed, cleaned datasets
│   └── additional_datasets/   # Any extra datasets used for analysis
├── notebooks/
│   ├── 1_data_preprocessing.ipynb      # Data preprocessing and exploratory analysis
│   ├── 2_model_development.ipynb       # Model training and evaluation
│   ├── 3_forecasting_and_analysis.ipynb # Forecasting and analysis notebook
│   └── 4_portfolio_optimization.ipynb   # Portfolio optimization
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py           # Data cleaning and preprocessing functions
│   ├── model_training.py               # Model training functions (e.g., ARIMA, LSTM)
│   ├── forecasting.py                  # Forecasting functions
│   └── portfolio_optimization.py       # Portfolio optimization functions
├── models/                              # Saved models and scalers
├── config/
│   ├── config.py                       # Configuration file (API keys, file paths)
│   └── parameters.json                 # Model parameters and settings
├── requirements.txt                    # Project dependencies
└── README.md                           # Project overview and instructions
```

## Getting Started
1. Clone the repository:
```bash
git clone https://github.com/Seife1/Time_Series_Portfolio_Optimization.git
```
2. Install Dependency
```bash
pip install -r requirements.txt
```
3. Download the data using YFinance for TSLA, BND, and SPY, and place it in the data/raw/ directory.

4. Run Notebooks to preprocess data, build models, forecast trends, and optimize portfolio allocation.

## Project Steps
1. **Data Preprocessing:** Cleaning and preparing data for analysis, removing anomalies, and filling missing values.
2. **Modeling:** Using ARIMA, SARIMA, and LSTM to predict asset price trends.
3. **Forecasting:** Generating forecasts to analyze future market trends and volatility.
4. **Portfolio Optimization:** Creating an optimized portfolio by adjusting asset allocations based on forecasted returns.

## Key Tools and Libraries
`Python`: Main programming language.

`YFinance`: For downloading historical financial data.

`pandas`, `numpy`: For data manipulation and processing.

`pmdarima`, `statsmodels`: For ARIMA and SARIMA modeling.

`tensorflow/keras`: For building LSTM models.

`Matplotlib`, `Seaborn`: For data visualization.

## Contributing
Feel free to submit issues and pull requests. Contributions that improve forecasting accuracy, add new models, or optimize portfolio allocation strategies are especially welcome.

## License
This project is licensed under the MIT License.