"""
Task 2: Quantitative Analysis of Stock Data

This script performs quantitative analysis on stock data using:
- TA-Lib for technical indicators
- PyNance for financial metrics
- Plotly for interactive visualizations
"""

import os
import pandas as pd
import numpy as np
import ta  # Alternative to TA-Lib that's easier to install
import pynance as pn
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# Create output directories
os.makedirs('../output/technical_analysis', exist_ok=True)
os.makedirs('../output/financial_metrics', exist_ok=True)

def load_stock_data(file_path):
    """Load and preprocess stock data."""
    print("Loading stock data...")
    try:
        # Load the data
        df = pd.read_csv(file_path, parse_dates=['Date'])
        
        # Basic preprocessing
        df = df.sort_values('Date')
        df = df.set_index('Date')
        
        print(f"Loaded data from {df.index.min()} to {df.index.max()}")
        print(f"Columns: {', '.join(df.columns)}")
        
        return df
    except Exception as e:
        print(f"Error loading stock data: {e}")
        return None

def calculate_technical_indicators(df):
    """Calculate various technical indicators using the ta library."""
    print("\nCalculating technical indicators...")
    
    # Make a copy to avoid SettingWithCopyWarning
    df_ta = df.copy()
    
    # Initialize indicators
    from ta.trend import SMAIndicator, MACD
    from ta.momentum import RSIIndicator
    from ta.volatility import BollingerBands, AverageTrueRange
    from ta.volume import OnBalanceVolumeIndicator
    
    # 1. Moving Averages
    df_ta['SMA_20'] = SMAIndicator(close=df_ta['Close'], window=20).sma_indicator()
    df_ta['SMA_50'] = SMAIndicator(close=df_ta['Close'], window=50).sma_indicator()
    df_ta['SMA_200'] = SMAIndicator(close=df_ta['Close'], window=200).sma_indicator()
    
    # 2. RSI (Relative Strength Index)
    rsi = RSIIndicator(close=df_ta['Close'], window=14)
    df_ta['RSI_14'] = rsi.rsi()
    
    # 3. MACD (Moving Average Convergence Divergence)
    macd = MACD(close=df_ta['Close'])
    df_ta['MACD'] = macd.macd()
    df_ta['MACD_Signal'] = macd.macd_signal()
    df_ta['MACD_Hist'] = macd.macd_diff()
    
    # 4. Bollinger Bands
    bb = BollingerBands(close=df_ta['Close'], window=20, window_dev=2)
    df_ta['BB_Upper'] = bb.bollinger_hband()
    df_ta['BB_Middle'] = bb.bollinger_mavg()
    df_ta['BB_Lower'] = bb.bollinger_lband()
    
    # 5. ATR (Average True Range)
    atr = AverageTrueRange(high=df_ta['High'], low=df_ta['Low'], 
                          close=df_ta['Close'], window=14)
    df_ta['ATR_14'] = atr.average_true_range()
    
    # 6. OBV (On-Balance Volume)
    obv = OnBalanceVolumeIndicator(close=df_ta['Close'], volume=df_ta['Volume'])
    df_ta['OBV'] = obv.on_balance_volume()
    
    return df_ta

def calculate_financial_metrics(df):
    """Calculate financial metrics using PyNance."""
    print("Calculating financial metrics...")
    
    # Make a copy to avoid SettingWithCopyWarning
    df_fin = df.copy()
    
    # 1. Daily Returns
    df_fin['Daily_Return'] = df_fin['Close'].pct_change() * 100
    
    # 2. Cumulative Returns
    df_fin['Cumulative_Return'] = (1 + df_fin['Daily_Return']/100).cumprod() - 1
    
    # 3. Volatility (20-day rolling)
    df_fin['Volatility_20D'] = df_fin['Daily_Return'].rolling(window=20).std() * np.sqrt(252)
    
    # 4. Moving Average of Volume (20-day)
    df_fin['Volume_MA_20'] = df_fin['Volume'].rolling(window=20).mean()
    
    # 5. Drawdown
    df_fin['Cumulative_Max'] = df_fin['Close'].cummax()
    df_fin['Drawdown'] = (df_fin['Close'] - df_fin['Cumulative_Max']) / df_fin['Cumulative_Max'] * 100
    
    return df_fin

def create_technical_plots(df, ticker='GOOG'):
    """Create interactive technical analysis plots using Plotly."""
    print("\nGenerating technical analysis plots...")
    
    # Create subplots
    fig = make_subplots(rows=4, cols=1, 
                        shared_xaxes=True, 
                        vertical_spacing=0.05,
                        row_heights=[0.5, 0.15, 0.15, 0.2])
    
    # 1. Candlestick with Moving Averages
    fig.add_trace(go.Candlestick(x=df.index,
                               open=df['Open'],
                               high=df['High'],
                               low=df['Low'],
                               close=df['Close'],
                               name='Price'),
                 row=1, col=1)
    
    # Add Moving Averages
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], 
                           line=dict(color='blue', width=1.5), 
                           name='SMA 20'),
                 row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], 
                           line=dict(color='orange', width=1.5), 
                           name='SMA 50'),
                 row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_200'], 
                           line=dict(color='red', width=2), 
                           name='SMA 200'),
                 row=1, col=1)
    
    # Add Bollinger Bands
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_Upper'],
                           line=dict(color='gray', width=1),
                           name='BB Upper',
                           showlegend=False),
                 row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_Lower'],
                           line=dict(color='gray', width=1),
                           name='BB Lower',
                           fill='tonexty',
                           fillcolor='rgba(200,200,200,0.2)',
                           showlegend=False),
                 row=1, col=1)
    
    # 2. Volume
    colors = ['green' if row['Close'] >= row['Open'] else 'red' 
             for _, row in df.iterrows()]
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], 
                        marker_color=colors,
                        name='Volume'),
                 row=2, col=1)
    
    # Add Volume MA
    fig.add_trace(go.Scatter(x=df.index, y=df['Volume_MA_20'],
                           line=dict(color='blue', width=1.5),
                           name='Volume MA 20'),
                 row=2, col=1)
    
    # 3. RSI
    fig.add_trace(go.Scatter(x=df.index, y=df['RSI_14'],
                           line=dict(color='purple', width=1.5),
                           name='RSI 14'),
                 row=3, col=1)
    
    # Add RSI levels
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
    
    # 4. MACD
    fig.add_trace(go.Bar(x=df.index, y=df['MACD_Hist'],
                        marker_color=np.where(df['MACD_Hist'] >= 0, 'green', 'red'),
                        name='MACD Hist'),
                 row=4, col=1)
    
    fig.add_trace(go.Scatter(x=df.index, y=df['MACD'],
                           line=dict(color='blue', width=1.5),
                           name='MACD'),
                 row=4, col=1)
    
    fig.add_trace(go.Scatter(x=df.index, y=df['MACD_Signal'],
                           line=dict(color='orange', width=1.5),
                           name='Signal'),
                 row=4, col=1)
    
    # Update layout
    fig.update_layout(
        title=f'{ticker} Technical Analysis',
        xaxis4_title='Date',
        yaxis_title='Price',
        yaxis2_title='Volume',
        yaxis3_title='RSI',
        yaxis4_title='MACD',
        height=1200,
        showlegend=True,
        template='plotly_white'
    )
    
    # Save the plot
    plot_path = f'../output/technical_analysis/{ticker}_technical_analysis.html'
    fig.write_html(plot_path)
    print(f"Technical analysis plot saved to {plot_path}")
    
    return fig

def create_financial_metrics_plot(df, ticker='GOOG'):
    """Create interactive financial metrics plots."""
    print("Generating financial metrics plots...")
    
    # Create subplots
    fig = make_subplots(rows=2, cols=1, 
                        shared_xaxes=True, 
                        vertical_spacing=0.1,
                        row_heights=[0.6, 0.4])
    
    # 1. Cumulative Returns
    fig.add_trace(go.Scatter(x=df.index, 
                           y=df['Cumulative_Return']*100,
                           line=dict(color='blue', width=2),
                           name='Cumulative Return (%)'),
                 row=1, col=1)
    
    # 2. Drawdown
    fig.add_trace(go.Scatter(x=df.index, 
                           y=df['Drawdown'],
                           line=dict(color='red', width=1.5),
                           name='Drawdown (%)',
                           fill='tozeroy'),
                 row=2, col=1)
    
    # Add zero line for drawdown
    fig.add_hline(y=0, line_dash="dash", line_color="black", row=2, col=1)
    
    # Update layout
    fig.update_layout(
        title=f'{ticker} Financial Metrics',
        xaxis2_title='Date',
        yaxis_title='Cumulative Return (%)',
        yaxis2_title='Drawdown (%)',
        height=800,
        showlegend=True,
        template='plotly_white'
    )
    
    # Save the plot
    plot_path = f'../output/financial_metrics/{ticker}_financial_metrics.html'
    fig.write_html(plot_path)
    print(f"Financial metrics plot saved to {plot_path}")
    
    return fig

def generate_report(df, ticker='GOOG'):
    """Generate a markdown report with key findings."""
    print("\nGenerating analysis report...")
    
    # Calculate key metrics
    last_close = df['Close'].iloc[-1]
    last_vol = df['Volume'].iloc[-1]
    last_rsi = df['RSI_14'].iloc[-1]
    last_atr = df['ATR_14'].iloc[-1]
    
    # Determine trend based on moving averages
    trend = "Bullish" if df['SMA_20'].iloc[-1] > df['SMA_50'].iloc[-1] > df['SMA_200'].iloc[-1] else "Bearish"
    
    # Generate report
    report = f"""# {ticker} Quantitative Analysis Report

## Key Metrics (Latest)
- **Price**: ${last_close:.2f}
- **Volume**: {last_vol:,.0f}
- **RSI (14)**: {last_rsi:.2f}
- **ATR (14)**: {last_atr:.2f}
- **Trend**: {trend}

## Technical Analysis Summary
- **Moving Averages**:
  - 20-day SMA: ${df['SMA_20'].iloc[-1]:.2f}
  - 50-day SMA: ${df['SMA_50'].iloc[-1]:.2f}
  - 200-day SMA: ${df['SMA_200'].iloc[-1]:.2f}
- **Bollinger Bands**:
  - Upper Band: ${df['BB_Upper'].iloc[-1]:.2f}
  - Middle Band: ${df['BB_Middle'].iloc[-1]:.2f}
  - Lower Band: ${df['BB_Lower'].iloc[-1]:.2f}

## Financial Metrics
- **Cumulative Return**: {df['Cumulative_Return'].iloc[-1]*100:.2f}%
- **Max Drawdown**: {df['Drawdown'].min():.2f}%
- **20-day Volatility**: {df['Volatility_20D'].iloc[-1]:.2f} (annualized)

## Visualizations
- Technical Analysis: `output/technical_analysis/{ticker}_technical_analysis.html`
- Financial Metrics: `output/financial_metrics/{ticker}_financial_metrics.html`

## Next Steps
1. Review the interactive plots for detailed analysis
2. Use the technical indicators for trade signals
3. Monitor key support/resistance levels
4. Consider the overall market context
"""
    
    # Save report
    report_path = f'../output/{ticker}_quantitative_analysis_report.md'
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"Analysis report saved to {report_path}")
    return report

def main():
    # File paths
    stock_file = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'yfinance_data (1)', 'yfinance_data', 'GOOG_historical_data.csv'))
    
    # Load and preprocess data
    df = load_stock_data(stock_file)
    if df is None:
        return
    
    # Calculate technical indicators
    df = calculate_technical_indicators(df)
    
    # Calculate financial metrics
    df = calculate_financial_metrics(df)
    
    # Generate plots
    create_technical_plots(df)
    create_financial_metrics_plot(df)
    
    # Generate report
    generate_report(df)
    
    print("\n=== Analysis Complete ===")
    print(f"Results saved to output/technical_analysis/ and output/financial_metrics/")

if __name__ == "__main__":
    main()
