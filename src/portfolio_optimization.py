import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class PortfolioOptimization:
    def load_asset_data(self, file_path):
        data = pd.read_csv(file_path)
        copy_data = data[['index', 'TSLA', 'BND', 'SPY']].copy()
        copy_data.rename(columns={'index': 'Date'}, inplace=True)
        copy_data['Date'] = pd.to_datetime(copy_data['Date'])
        copy_data.set_index('Date', inplace=True)
        
        return copy_data
    
    def calculate_log_returns(self, asset_data):
        log_returns = asset_data.pct_change().apply(lambda x: np.log(1 + x))
        return log_returns
    
    def calculate_variance(self, log_returns):
        variance = log_returns.var()
        return variance 

    def calculate_volatility(self, asset_data, variance):
        volatility = np.sqrt(variance * 252)
        
        # Plotting volatility for each asset
        plt.figure(figsize=(10, 5))
        ax = asset_data.pct_change().apply(lambda x: np.log(1 + x)).std().apply(lambda x: x * np.sqrt(252)).plot(
            kind='bar', label='Annualized Volatility'
        )
        plt.title("Annualized Volatility of Selected Assets")  
        plt.xlabel("Assets")          
        plt.ylabel("Volatility")  
        plt.legend()         
        plt.show()
        
        return volatility

    def plot_heatmap(self, matrix, title):
        plt.figure(figsize=(10, 8))
        sns.heatmap(matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
        plt.title(title)
        plt.show()
        
    def compute_covariance_matrix(self, log_returns):
        cov_matrix = log_returns.cov()
        self.plot_heatmap(cov_matrix, "Covariance Matrix Heatmap for Assets")
        
        return cov_matrix
    
    def compute_correlation_matrix(self, asset_data):
        corr_matrix = asset_data.pct_change().apply(lambda x: np.log(1 + x)).corr()
        self.plot_heatmap(corr_matrix, "Correlation Matrix Heatmap for Assets")
        
        return corr_matrix
        
    def calculate_annualized_returns(self, asset_data):
        avg_annual_return = asset_data.resample('Y').last().pct_change().mean()
        annual_std_dev = asset_data.pct_change().apply(lambda x: np.log(1 + x)).std().apply(lambda x: x * np.sqrt(252))
        
        return avg_annual_return, annual_std_dev
    
    def plot_efficient_frontier(self, asset_data, cov_matrix, avg_annual_return, annual_std_dev):
        assets = pd.concat([avg_annual_return, annual_std_dev], axis=1)
        assets.columns = ['Average Annual Return', 'Annual Volatility']
        
        portfolio_returns = []  # Portfolio return
        portfolio_volatility = []  # Portfolio volatility
        portfolio_weights = []  # Weights for each asset in the portfolio

        num_assets = len(asset_data.columns)
        num_portfolios = 10000

        for _ in range(num_portfolios):
            weights = np.random.random(num_assets)
            weights /= np.sum(weights)
            portfolio_weights.append(weights)
            returns = np.dot(weights, avg_annual_return)
            portfolio_returns.append(returns)
            var = cov_matrix.mul(weights, axis=0).mul(weights, axis=1).sum().sum()
            std_dev = np.sqrt(var) 
            ann_std_dev = std_dev * np.sqrt(250)
            portfolio_volatility.append(ann_std_dev)
            
        portfolio_data = {'Return': portfolio_returns, 'Volatility': portfolio_volatility}

        for i, asset in enumerate(asset_data.columns.tolist()):
            portfolio_data[asset + ' Weight'] = [weight[i] for weight in portfolio_weights]
            
        portfolios_df = pd.DataFrame(portfolio_data)
        
        # Plotting efficient frontier
        ax = portfolios_df.plot.scatter(x='Volatility', y='Return', marker='o', s=10, alpha=0.3, grid=True, figsize=[10, 10], label='Portfolio Options')
        plt.title("Efficient Frontier: Risk vs. Expected Returns")  
        plt.xlabel("Annual Volatility (Risk)")          
        plt.ylabel("Expected Annual Return")  
        plt.legend()         
        plt.show()
        
        return portfolios_df
        
    def identify_min_volatility_portfolio(self, portfolios_df):
        min_volatility_portfolio = portfolios_df.iloc[portfolios_df['Volatility'].idxmin()]
        
        # Plotting the portfolio with minimum volatility
        plt.figure(figsize=[10, 10])
        plt.scatter(portfolios_df['Volatility'], portfolios_df['Return'], marker='o', s=10, alpha=0.3)
        plt.scatter(min_volatility_portfolio['Volatility'], min_volatility_portfolio['Return'], color='red', marker='*', s=500, label='Minimum Volatility Portfolio')
        plt.title("Minimum Volatility Portfolio")
        plt.xlabel("Annual Volatility (Risk)")          
        plt.ylabel("Expected Annual Return")  
        plt.legend()
        
        return min_volatility_portfolio
        
    def identify_optimal_risky_portfolio(self, portfolios_df, min_volatility_portfolio):
        risk_free_rate = 0.01  # Assuming a fixed risk-free rate
        optimal_risky_portfolio = portfolios_df.iloc[((portfolios_df['Return'] - risk_free_rate) / portfolios_df['Volatility']).idxmax()]
        
        # Plotting optimal risky portfolio
        plt.figure(figsize=(10, 10))
        plt.scatter(portfolios_df['Volatility'], portfolios_df['Return'], marker='o', s=10, alpha=0.3)
        plt.scatter(min_volatility_portfolio['Volatility'], min_volatility_portfolio['Return'], color='red', marker='*', s=500, label='Minimum Volatility Portfolio')
        plt.scatter(optimal_risky_portfolio['Volatility'], optimal_risky_portfolio['Return'], color='green', marker='*', s=500, label='Optimal Portfolio')
        plt.title("Optimal Risk-Return Portfolio")
        plt.xlabel("Annual Volatility (Risk)")          
        plt.ylabel("Expected Annual Return")  
        plt.legend()
        
        return optimal_risky_portfolio
