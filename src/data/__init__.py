"""
Data loading and preprocessing utilities.
"""

__all__ = ["load_news_data", "load_stock_data"]

import pandas as pd

def load_news_data(filepath):
    """
    Load and preprocess financial news data.
    
    Args:
        filepath (str): Path to the news data CSV file.
        
    Returns:
        pd.DataFrame: Processed news data.
    """
    # Load the data
    df = pd.read_csv(filepath)
    
    # Convert date column to datetime
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    
    return df

def load_stock_data(ticker, start_date, end_date):
    """
    Load stock price data using yfinance.
    
    Args:
        ticker (str): Stock ticker symbol (e.g., 'AAPL')
        start_date (str): Start date in 'YYYY-MM-DD' format
        end_date (str): End date in 'YYYY-MM-DD' format
        
    Returns:
        pd.DataFrame: Stock price data with OHLCV columns
    """
    try:
        import yfinance as yf
        stock_data = yf.download(ticker, start=start_date, end=end_date)
        return stock_data
    except ImportError:
        raise ImportError("yfinance is required to download stock data. "
                         "Install it with: pip install yfinance")
