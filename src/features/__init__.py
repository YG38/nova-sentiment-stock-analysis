"""
Feature engineering for sentiment analysis and technical indicators.
"""

__all__ = ["calculate_sentiment", "calculate_technical_indicators"]

def calculate_sentiment(text_series):
    """
    Calculate sentiment scores for a series of text data.
    
    Args:
        text_series (pd.Series): Series containing text data
        
    Returns:
        pd.Series: Sentiment scores
    """
    try:
        from textblob import TextBlob
        return text_series.apply(lambda x: TextBlob(str(x)).sentiment.polarity)
    except ImportError:
        raise ImportError("TextBlob is required for sentiment analysis. "
                         "Install it with: pip install textblob")

def calculate_technical_indicators(price_data, window_sizes=[5, 10, 20]):
    """
    Calculate common technical indicators from OHLCV data using pandas-ta.
    
    Args:
        price_data (pd.DataFrame): DataFrame with OHLCV columns
        window_sizes (list): List of window sizes for moving averages
        
    Returns:
        pd.DataFrame: Input DataFrame with added technical indicators
    """
    try:
        import pandas_ta as ta
    except ImportError:
        raise ImportError("pandas_ta is required for technical indicators. "
                         "Install it with: pip install pandas-ta")
    
    # Make a copy to avoid modifying the original
    df = price_data.copy()
    
    # Calculate moving averages
    for window in window_sizes:
        df[f'MA_{window}'] = ta.sma(df['Close'], length=window)
    
    # Calculate RSI
    df['RSI'] = ta.rsi(df['Close'], length=14)
    
    # Calculate MACD
    macd = ta.macd(df['Close'])
    df = pd.concat([df, macd], axis=1)
    
    return df
