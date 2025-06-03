"""Exploratory Data Analysis (EDA) utilities for the stock sentiment analysis project."""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from textblob import TextBlob
from typing import Dict, Tuple, Optional

# Set plotting style
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 12


def load_stock_data(ticker: str) -> Optional[pd.DataFrame]:
    """
    Load stock data for a given ticker.
    
    Args:
        ticker: Stock ticker symbol
        
    Returns:
        DataFrame with stock data or None if file not found
    """
    filepath = f'C:/Users/Yonatan/data/raw/{ticker}_stock_data.csv'
    if os.path.exists(filepath):
        try:
            # Read the CSV file, handling the header properly
            df = pd.read_csv(filepath, header=0)
            
            # Clean up column names
            df.columns = [col.strip() for col in df.columns]
            
            # If the first row contains the ticker symbol, drop it
            if df.iloc[0, 0] == ticker:
                df = df.iloc[1:].reset_index(drop=True)
            
            # Ensure we have the expected columns
            expected_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            if not all(col in df.columns for col in expected_cols):
                # If columns are not in the header, use the first row as header
                df = pd.read_csv(filepath, header=1, names=expected_cols)
            
            # Ensure all numeric columns are properly converted
            for col in expected_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce')
            
            # Add a date index (reverse chronological order)
            df = df.sort_index(ascending=False).reset_index(drop=True)
            df['Date'] = pd.date_range(end=pd.Timestamp.today().normalize(), periods=len(df), freq='D')
            df.set_index('Date', inplace=True)
            
            return df
            
        except Exception as e:
            print(f"Error loading {ticker} data: {str(e)}")
            return None
    return None


def load_and_prepare_news_data(filepath: str) -> pd.DataFrame:
    """
    Load and preprocess news data from a CSV file.
    
    Args:
        filepath: Path to the news data CSV file
        
    Returns:
        Preprocessed DataFrame with news data
    """
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'])
    
    # Extract additional datetime features
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['day_of_week'] = df['date'].dt.day_name()
    df['is_weekend'] = df['date'].dt.dayofweek >= 5
    
    return df


def calculate_sentiment(text: str) -> float:
    """
    Calculate sentiment polarity using TextBlob.
    
    Args:
        text: Input text to analyze
        
    Returns:
        Sentiment score between -1 (negative) and 1 (positive)
    """
    try:
        return TextBlob(str(text)).sentiment.polarity
    except Exception:
        return 0.0


def add_sentiment_analysis(df: pd.DataFrame, text_col: str = 'headline') -> pd.DataFrame:
    """
    Add sentiment analysis columns to a DataFrame.
    
    Args:
        df: Input DataFrame with text data
        text_col: Name of the column containing text to analyze
        
    Returns:
        DataFrame with added sentiment columns
    """
    df = df.copy()
    
    # Calculate sentiment scores
    df['sentiment'] = df[text_col].apply(calculate_sentiment)
    
    # Categorize sentiment
    conditions = [
        (df['sentiment'] > 0.1),
        (df['sentiment'] < -0.1)
    ]
    choices = ['positive', 'negative']
    df['sentiment_category'] = np.select(conditions, choices, default='neutral')
    
    # Add sentiment intensity
    df['sentiment_intensity'] = df['sentiment'].abs()
    
    return df


def plot_sentiment_distribution(df: pd.DataFrame, title: str = 'Sentiment Distribution') -> None:
    """
    Plot the distribution of sentiment categories.
    
    Args:
        df: DataFrame with sentiment_category column
        title: Plot title
    """
    plt.figure(figsize=(10, 5))
    order = ['negative', 'neutral', 'positive']
    ax = sns.countplot(x='sentiment_category', data=df, order=order, palette='viridis')
    
    # Add counts on top of bars
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.0f}', 
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', 
                    xytext=(0, 5), 
                    textcoords='offset points')
    
    plt.title(title, fontsize=14)
    plt.xlabel('Sentiment Category', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.tight_layout()
    plt.show()


def plot_sentiment_over_time(df: pd.DataFrame, time_col: str = 'date', freq: str = 'W') -> None:
    """
    Plot sentiment over time.
    
    Args:
        df: DataFrame with date and sentiment columns
        time_col: Name of the datetime column
        freq: Resampling frequency (e.g., 'D' for daily, 'W' for weekly)
    """
    # Create a copy and set index to datetime
    temp_df = df.set_index(time_col).copy()
    
    # Resample and calculate mean sentiment
    resampled = temp_df['sentiment'].resample(freq).mean()
    
    # Plot
    plt.figure(figsize=(14, 6))
    resampled.plot(color='royalblue', linewidth=2)
    
    # Add horizontal line at y=0
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    
    plt.title(f'Sentiment Over Time ({freq}ly Average)', fontsize=14)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Average Sentiment Score', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_news_volume_over_time(df: pd.DataFrame, time_col: str = 'date', freq: str = 'W') -> None:
    """
    Plot news volume over time.
    
    Args:
        df: DataFrame with datetime column
        time_col: Name of the datetime column
        freq: Resampling frequency
    """
    # Create a copy and set index to datetime
    temp_df = df.set_index(time_col).copy()
    
    # Resample and count articles
    resampled = temp_df.resample(freq).size()
    
    # Plot
    plt.figure(figsize=(14, 6))
    resampled.plot(kind='bar', color='teal', alpha=0.7)
    
    plt.title(f'News Volume Over Time ({freq}ly)', fontsize=14)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Number of Articles', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()


def analyze_stock_data(stock_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """
    Calculate basic statistics for stock data.
    
    Args:
        stock_data: Dictionary of DataFrames with stock data
        
    Returns:
        Dictionary of DataFrames with calculated statistics
    """
    stats = {}
    
    for ticker, df in stock_data.items():
        # Calculate daily returns
        df['daily_return'] = df['Close'].pct_change()
        
        # Calculate volatility (standard deviation of returns)
        volatility = df['daily_return'].std() * np.sqrt(252)  # Annualized
        
        # Calculate moving averages
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        
        # Store basic statistics
        stats[ticker] = pd.DataFrame({
            'Start Date': [df.index.min().strftime('%Y-%m-%d')],
            'End Date': [df.index.max().strftime('%Y-%m-%d')],
            'Start Price': [df['Close'].iloc[0]],
            'End Price': [df['Close'].iloc[-1]],
            'Min Price': [df['Close'].min()],
            'Max Price': [df['Close'].max()],
            'Average Daily Volume': [df['Volume'].mean()],
            'Volatility (Annualized)': [volatility],
            'Total Return': [(df['Close'].iloc[-1] / df['Close'].iloc[0]) - 1]
        })
    
    return stats


def plot_stock_prices(stock_data: Dict[str, pd.DataFrame], normalize: bool = True) -> None:
    """
    Plot stock price time series.
    
    Args:
        stock_data: Dictionary of DataFrames with stock data
        normalize: Whether to normalize prices to start at 1 for comparison
    """
    plt.figure(figsize=(14, 8))
    
    for ticker, df in stock_data.items():
        if normalize:
            # Normalize prices to start at 1
            prices = df['Close'] / df['Close'].iloc[0]
        else:
            prices = df['Close']
            
        plt.plot(prices, label=ticker, linewidth=2)
    
    plt.title('Stock Price Comparison' + (' (Normalized)' if normalize else ''), fontsize=14)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Price' + (' (Normalized)' if normalize else ''), fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_stock_returns(stock_data: Dict[str, pd.DataFrame]) -> None:
    """
    Plot daily returns for stocks.
    
    Args:
        stock_data: Dictionary of DataFrames with stock data
    """
    plt.figure(figsize=(14, 8))
    
    for ticker, df in stock_data.items():
        returns = df['Close'].pct_change()
        plt.plot(returns, label=ticker, alpha=0.7)
    
    plt.title('Daily Returns', fontsize=14)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Daily Return', fontsize=12)
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_volume_distribution(stock_data: Dict[str, pd.DataFrame]) -> None:
    """
    Plot trading volume distribution.
    
    Args:
        stock_data: Dictionary of DataFrames with stock data
    """
    plt.figure(figsize=(14, 6))
    
    volumes = []
    labels = []
    for ticker, df in stock_data.items():
        volumes.append(df['Volume'])
        labels.append(ticker)
    
    plt.boxplot(volumes, labels=labels, patch_artist=True)
    plt.yscale('log')  # Use log scale due to large volume ranges
    plt.title('Trading Volume Distribution (Log Scale)', fontsize=14)
    plt.ylabel('Trading Volume (log scale)', fontsize=12)
    plt.grid(True, alpha=0.3, which='both')
    plt.tight_layout()
    plt.show()


def plot_correlation_heatmap(stock_data: Dict[str, pd.DataFrame]) -> None:
    """
    Plot correlation heatmap of stock returns.
    
    Args:
        stock_data: Dictionary of DataFrames with stock data
    """
    # Create a DataFrame of returns
    returns_df = pd.DataFrame()
    
    for ticker, df in stock_data.items():
        returns_df[ticker] = df['Close'].pct_change()
    
    # Calculate correlation matrix
    corr = returns_df.corr()
    
    # Plot heatmap
    plt.figure(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool))  # Create a mask for the upper triangle
    
    sns.heatmap(
        corr, 
        mask=mask,
        annot=True, 
        cmap='coolwarm', 
        vmin=-1, vmax=1,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8}
    )
    
    plt.title('Correlation of Stock Returns', fontsize=14)
    plt.tight_layout()
    plt.show()


def plot_sentiment_vs_returns(news_df: pd.DataFrame, stock_data: Dict[str, pd.DataFrame]) -> None:
    """
    Plot relationship between sentiment and stock returns.
    
    Args:
        news_df: DataFrame with news and sentiment data
        stock_data: Dictionary of DataFrames with stock data
    """
    # Group news by date and stock, calculate average sentiment
    daily_sentiment = news_df.groupby(['date', 'stock'])['sentiment'].mean().reset_index()
    
    # For each stock, merge with returns
    for ticker in stock_data.keys():
        # Get stock returns
        stock_returns = stock_data[ticker]['Close'].pct_change().reset_index()
        stock_returns.columns = ['date', 'return']
        
        # Get sentiment for this stock
        stock_sentiment = daily_sentiment[daily_sentiment['stock'] == ticker].copy()
        
        if not stock_sentiment.empty:
            # Merge with returns
            merged = pd.merge(
                stock_sentiment,
                stock_returns,
                on='date',
                how='inner'
            )
            
            # Plot
            if not merged.empty:
                plt.figure(figsize=(12, 6))
                plt.scatter(
                    merged['sentiment'],
                    merged['return'],
                    alpha=0.6,
                    edgecolors='w',
                    s=80
                )
                
                # Add regression line
                sns.regplot(
                    x='sentiment',
                    y='return',
                    data=merged,
                    scatter=False,
                    color='red',
                    line_kws={'linestyle': '--'}
                )
                
                plt.title(f'Sentiment vs. Next Day Returns: {ticker}', fontsize=14)
                plt.xlabel('Average Daily Sentiment Score', fontsize=12)
                plt.ylabel('Next Day Return', fontsize=12)
                plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.show()


def analyze_news_impact(news_df: pd.DataFrame, stock_data: Dict[str, pd.DataFrame], 
                       days_after: int = 5) -> Dict[str, pd.DataFrame]:
    """
    Analyze the impact of news sentiment on stock returns.
    
    Args:
        news_df: DataFrame with news and sentiment data
        stock_data: Dictionary of DataFrames with stock data
        days_after: Number of days to analyze after news
        
    Returns:
        Dictionary with analysis results for each stock
    """
    results = {}
    
    for ticker in stock_data.keys():
        # Get stock data
        stock = stock_data[ticker].copy()
        stock['return'] = stock['Close'].pct_change()
        
        # Get news for this stock
        ticker_news = news_df[news_df['stock'] == ticker].copy()
        
        if not ticker_news.empty:
            # For each news item, get future returns
            event_returns = []
            
            for _, row in ticker_news.iterrows():
                date = row['date']
                sentiment = row['sentiment']
                
                # Get returns for the next 'days_after' days
                future_returns = stock[stock.index > date].head(days_after)['return'].values
                
                if len(future_returns) > 0:
                    # Calculate cumulative return
                    cum_return = np.prod(1 + future_returns) - 1
                    
                    event_returns.append({
                        'date': date,
                        'sentiment': sentiment,
                        'cumulative_return': cum_return,
                        'days': len(future_returns)
                    })
            
            # Create DataFrame with results
            if event_returns:
                results[ticker] = pd.DataFrame(event_returns)
    
    return results


def plot_news_impact_analysis(results: Dict[str, pd.DataFrame]) -> None:
    """
    Plot the impact of news sentiment on stock returns.
    
    Args:
        results: Dictionary of DataFrames from analyze_news_impact()
    """
    for ticker, df in results.items():
        if not df.empty:
            # Categorize sentiment
            df['sentiment_category'] = pd.cut(
                df['sentiment'],
                bins=[-np.inf, -0.1, 0.1, np.inf],
                labels=['negative', 'neutral', 'positive']
            )
            
            # Plot
            plt.figure(figsize=(12, 6))
            
            for category in ['negative', 'neutral', 'positive']:
                subset = df[df['sentiment_category'] == category]
                if not subset.empty:
                    plt.scatter(
                        subset['sentiment'],
                        subset['cumulative_return'],
                        label=category.capitalize(),
                        alpha=0.6,
                        s=80
                    )
            
            plt.title(f'News Sentiment vs. {df["days"].iloc[0]}-Day Cumulative Returns: {ticker}', fontsize=14)
            plt.xlabel('Sentiment Score', fontsize=12)
            plt.ylabel('Cumulative Return', fontsize=12)
            plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
