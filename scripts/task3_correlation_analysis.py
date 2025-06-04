"""
Task 3: Correlation Analysis

This script analyzes the correlation between news sentiment and stock price movements.
It combines the analyst ratings data with stock price data to identify relationships.
"""

import os
import pandas as pd
import numpy as np
from textblob import TextBlob
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

def load_and_preprocess_data(news_file, stock_file):
    """Load and preprocess news and stock data."""
    print("Loading and preprocessing data...")
    
    try:
        # Load news data with explicit date parsing
        news_df = pd.read_csv(news_file)
        news_df['date'] = pd.to_datetime(news_df['date'], errors='coerce')
        news_df = news_df.dropna(subset=['date'])
        
        # Load stock data with explicit date parsing
        stock_df = pd.read_csv(stock_file)
        stock_df['Date'] = pd.to_datetime(stock_df['Date'], errors='coerce')
        stock_df = stock_df.rename(columns={'Date': 'date'})
        stock_df = stock_df.dropna(subset=['date'])
        
        # Calculate daily returns
        stock_df['Daily_Return'] = stock_df['Close'].pct_change() * 100
        
        # Extract date from datetime for merging
        news_df['date_only'] = news_df['date'].dt.date
        stock_df['date_only'] = stock_df['date'].dt.date
        
        return news_df, stock_df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None

def calculate_sentiment_scores(news_df):
    """Calculate sentiment scores for news headlines."""
    print("Calculating sentiment scores...")
    
    # Function to get sentiment polarity (-1 to 1)
    def get_sentiment(text):
        try:
            return TextBlob(str(text)).sentiment.polarity
        except:
            return 0.0
    
    # Calculate sentiment for each headline
    news_df['sentiment'] = news_df['headline'].apply(get_sentiment)
    
    # Aggregate sentiment by date
    daily_sentiment = news_df.groupby('date_only')['sentiment'].agg(
        ['mean', 'count', 'std']).reset_index()
    daily_sentiment.columns = ['date_only', 'avg_sentiment', 
                             'news_count', 'sentiment_std']
    
    return news_df, daily_sentiment

def analyze_correlations(daily_sentiment, stock_df, window=5):
    """Analyze correlations between sentiment and stock returns."""
    print("Analyzing correlations...")
    
    # Merge sentiment and stock data
    merged_df = pd.merge(
        daily_sentiment, 
        stock_df[['date_only', 'Close', 'Daily_Return']], 
        on='date_only', 
        how='inner'
    )
    
    # Calculate rolling correlations
    merged_df.set_index('date_only', inplace=True)
    merged_df.sort_index(inplace=True)
    
    # Calculate rolling correlation
    merged_df['rolling_corr'] = merged_df['avg_sentiment'].rolling(window=window).corr(
        merged_df['Daily_Return']
    )
    
    # Calculate lead/lag correlations
    correlations = []
    for lag in range(-5, 6):
        if lag < 0:
            # Negative lag: sentiment leads stock returns
            corr = merged_df['avg_sentiment'].shift(-lag).rolling(window=window).corr(
                merged_df['Daily_Return']
            ).mean()
        elif lag > 0:
            # Positive lag: sentiment lags stock returns
            corr = merged_df['avg_sentiment'].rolling(window=window).corr(
                merged_df['Daily_Return'].shift(lag)
            ).mean()
        else:
            # Same day correlation
            corr = merged_df['avg_sentiment'].rolling(window=window).corr(
                merged_df['Daily_Return']
            ).mean()
        correlations.append((lag, corr))
    
    # Create correlation by lag DataFrame
    lag_corr_df = pd.DataFrame(correlations, columns=['lag_days', 'correlation'])
    
    return merged_df, lag_corr_df

def create_correlation_plots(merged_df, lag_corr_df, ticker='GOOG'):
    """Create interactive plots for correlation analysis."""
    print("Generating correlation plots...")
    
    # Create output directory
    os.makedirs('../output/correlation_analysis', exist_ok=True)
    
    # 1. Time Series Plot
    fig1 = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add stock price
    fig1.add_trace(
        go.Scatter(
            x=merged_df.index, 
            y=merged_df['Close'],
            name='Stock Price',
            line=dict(color='blue')
        ),
        secondary_y=False,
    )
    
    # Add sentiment
    fig1.add_trace(
        go.Scatter(
            x=merged_df.index,
            y=merged_df['avg_sentiment'],
            name='Avg Sentiment',
            line=dict(color='red')
        ),
        secondary_y=True,
    )
    
    # Update layout
    fig1.update_layout(
        title=f"{ticker} Stock Price vs. News Sentiment Over Time",
        xaxis_title="Date",
        yaxis_title="Stock Price ($)",
        yaxis2=dict(title="Average Sentiment", overlaying="y", side="right"),
        legend=dict(x=0.02, y=0.98),
        height=600,
        template='plotly_white'
    )
    
    # 2. Correlation by Lag Plot
    fig2 = go.Figure()
    
    fig2.add_trace(
        go.Bar(
            x=lag_corr_df['lag_days'],
            y=lag_corr_df['correlation'],
            marker_color=np.where(
                lag_corr_df['correlation'] > 0, 
                'green', 
                'red'
            ),
            name='Correlation',
            text=lag_corr_df['correlation'].round(3),
            textposition='auto'
        )
    )
    
    fig2.update_layout(
        title="Correlation Between News Sentiment and Stock Returns by Lag",
        xaxis_title="Lag (Days)",
        yaxis_title="Correlation",
        xaxis=dict(tickvals=list(range(-5, 6))),
        height=500,
        template='plotly_white',
        showlegend=False
    )
    
    # 3. Scatter Plot
    fig3 = go.Figure()
    
    fig3.add_trace(
        go.Scatter(
            x=merged_df['avg_sentiment'],
            y=merged_df['Daily_Return'],
            mode='markers',
            marker=dict(
                size=8,
                color=pd.to_datetime(merged_df.index).astype(np.int64) // 10**9,  # Convert datetime to numeric
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title='Date')
            ),
            name='Daily Data',
            text=pd.to_datetime(merged_df.index).strftime('%Y-%m-%d'),
            hovertemplate=
                'Date: %{text}<br>' +
                'Sentiment: %{x:.3f}<br>' +
                'Return: %{y:.2f}%<br>'
        )
    )
    
    # Add trendline
    z = np.polyfit(merged_df['avg_sentiment'].dropna(), 
                   merged_df['Daily_Return'].dropna(), 1)
    p = np.poly1d(z)
    
    fig3.add_trace(
        go.Scatter(
            x=merged_df['avg_sentiment'],
            y=p(merged_df['avg_sentiment']),
            mode='lines',
            line=dict(color='red', width=2),
            name='Trendline'
        )
    )
    
    fig3.update_layout(
        title=f"{ticker} Daily Returns vs. News Sentiment",
        xaxis_title="Average Daily Sentiment Score",
        yaxis_title="Daily Return (%)",
        height=600,
        template='plotly_white',
        showlegend=True
    )
    
    # Save plots
    fig1.write_html(f"../output/correlation_analysis/{ticker}_sentiment_price_trend.html")
    fig2.write_html(f"../output/correlation_analysis/{ticker}_sentiment_lag_correlation.html")
    fig3.write_html(f"../output/correlation_analysis/{ticker}_sentiment_returns_scatter.html")
    
    print(f"Correlation plots saved to output/correlation_analysis/")
    
    return fig1, fig2, fig3

def generate_correlation_report(merged_df, lag_corr_df, ticker='GOOG'):
    """Generate a markdown report with correlation analysis findings."""
    print("\nGenerating correlation analysis report...")
    
    # Calculate overall correlation
    overall_corr = merged_df['avg_sentiment'].corr(merged_df['Daily_Return'])
    
    # Find strongest lag correlation
    max_corr = lag_corr_df['correlation'].max()
    max_corr_lag = lag_corr_df.loc[lag_corr_df['correlation'].idxmax(), 'lag_days']
    
    # Calculate days with positive sentiment
    positive_days = (merged_df['avg_sentiment'] > 0).mean() * 100
    
    # Calculate average returns by sentiment
    avg_return_positive = merged_df[merged_df['avg_sentiment'] > 0]['Daily_Return'].mean()
    avg_return_negative = merged_df[merged_df['avg_sentiment'] < 0]['Daily_Return'].mean()
    
    # Generate report
    report = f"""# {ticker} News Sentiment and Stock Price Correlation Analysis

## Key Findings

### 1. Overall Correlation
- **Correlation Coefficient**: {overall_corr:.3f}
- **Interpretation**: {"Positive" if overall_corr > 0 else "Negative" if overall_corr < 0 else "No"} correlation between news sentiment and stock returns
- **Strongest Correlation**: {max_corr:.3f} at {max_corr_lag} days lag

### 2. Sentiment Distribution
- **Positive Sentiment Days**: {positive_days:.1f}% of trading days
- **Average Daily Return on Positive Sentiment Days**: {avg_return_positive:.2f}%
- **Average Daily Return on Negative Sentiment Days**: {avg_return_negative:.2f}%

### 3. Lag Analysis
- **Strongest Lead/Lag Relationship**: 
  - Sentiment leads stock returns by {-max_corr_lag} days (if negative)
  - Stock returns lead sentiment by {max_corr_lag} days (if positive)
  - Same-day correlation is {lag_corr_df[lag_corr_df['lag_days'] == 0]['correlation'].values[0]:.3f}

### 4. Key Observations
1. {"Higher" if overall_corr > 0 else "Lower"} sentiment days tend to be followed by {"higher" if overall_corr > 0 else "lower"} stock returns
2. The relationship is {"strong" if abs(overall_corr) > 0.3 else "moderate" if abs(overall_corr) > 0.1 else "weak"}
3. Sentiment appears to be a {"leading" if max_corr_lag < 0 else "lagging" if max_corr_lag > 0 else "coincident"} indicator of stock price movements

## Visualizations
1. **Price vs. Sentiment Trend**: `output/correlation_analysis/{ticker}_sentiment_price_trend.html`
2. **Lag Correlation**: `output/correlation_analysis/{ticker}_sentiment_lag_correlation.html`
3. **Returns vs. Sentiment Scatter**: `output/correlation_analysis/{ticker}_sentiment_returns_scatter.html`

## Recommendations
1. {"Consider" if abs(max_corr) > 0.2 else "Be cautious about"} using news sentiment as a {"leading" if max_corr_lag < 0 else "lagging"} indicator
2. Monitor sentiment trends around key news events
3. Combine sentiment analysis with other technical and fundamental indicators

## Limitations
1. Correlation does not imply causation
2. News sentiment is just one of many factors affecting stock prices
3. The relationship may vary across different market conditions
"""
    
    # Save report
    report_path = f'../output/{ticker}_correlation_analysis_report.md'
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"Correlation analysis report saved to {report_path}")
    return report

def main():
    # File paths
    news_file = os.path.abspath(os.path.join(
        os.path.dirname(__file__), '..', 'raw_analyst_ratings (1).csv'))
    stock_file = os.path.abspath(os.path.join(
        os.path.dirname(__file__), '..', 'yfinance_data (1)', 'yfinance_data', 'GOOG_historical_data.csv'))
    
    # Load and preprocess data
    news_df, stock_df = load_and_preprocess_data(news_file, stock_file)
    if news_df is None or stock_df is None:
        return
    
    # Calculate sentiment scores
    news_df, daily_sentiment = calculate_sentiment_scores(news_df)
    
    # Analyze correlations
    merged_df, lag_corr_df = analyze_correlations(daily_sentiment, stock_df)
    
    # Generate plots
    create_correlation_plots(merged_df, lag_corr_df)
    
    # Generate and print report
    report = generate_correlation_report(merged_df, lag_corr_df)
    
    print("\n=== Correlation Analysis Complete ===")
    print("Results saved to output/correlation_analysis/")
    
    # Print the report to console
    print("\n" + "="*80)
    print("CORRELATION ANALYSIS REPORT")
    print("="*80)
    print(report)

if __name__ == "__main__":
    main()
