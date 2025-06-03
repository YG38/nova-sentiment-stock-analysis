"""
Task 1: Exploratory Data Analysis of Analyst Ratings

This script performs exploratory data analysis on the analyst ratings dataset to understand:
1. Basic statistics and data quality
2. Text analysis of headlines and summaries
3. Time series analysis of publication frequency
4. Publisher analysis
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from collections import Counter
import re

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 200)

# Create output directory for plots
os.makedirs('../output/eda_plots', exist_ok=True)

def load_data(file_path, sample_size=100000):
    """Load the dataset with error handling and sampling."""
    try:
        # Read only the first sample_size rows for initial analysis
        df = pd.read_csv(file_path, nrows=sample_size, parse_dates=['date'], low_memory=False)
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def basic_data_analysis(df):
    """Perform basic data quality and statistical analysis."""
    print("\n=== Basic Data Analysis ===")
    print(f"Data shape: {df.shape}")
    
    # Basic info
    print("\nData Types and Missing Values:")
    print(df.info())
    
    # Basic statistics
    print("\nBasic Statistics:")
    print(df.describe(include='all').T)
    
    # Check for duplicates
    print(f"\nNumber of duplicate rows: {df.duplicated().sum()}")
    
    # Check for missing values
    print("\nMissing values per column:")
    print(df.isnull().sum())

def analyze_text(df, text_column='headline'):
    """Analyze text data including length and word frequency."""
    print(f"\n=== Text Analysis ({text_column}) ===")
    
    # Calculate text length
    df[f'{text_column}_length'] = df[text_column].str.len()
    
    # Plot length distribution
    plt.figure(figsize=(12, 6))
    sns.histplot(df[f'{text_column}_length'].dropna(), bins=50, kde=True)
    plt.title(f'Distribution of {text_column.capitalize()} Lengths')
    plt.xlabel(f'{text_column.capitalize()} Length (characters)')
    plt.ylabel('Frequency')
    plt.savefig(f'../output/eda_plots/{text_column}_length_distribution.png')
    plt.close()
    
    # Top words analysis
    def get_top_words(text_series, n=20):
        words = ' '.join(text_series.dropna().astype(str)).lower().split()
        # Remove common stop words and single characters
        stop_words = set(nltk.corpus.stopwords.words('english'))
        words = [re.sub(r'[^\w\s]', '', word) for word in words]  # Remove punctuation
        words = [word for word in words if word not in stop_words and len(word) > 2]
        return pd.Series(Counter(words)).nlargest(n)
    
    top_words = get_top_words(df[text_column])
    
    # Plot top words
    plt.figure(figsize=(12, 8))
    sns.barplot(x=top_words.values, y=top_words.index, palette='viridis')
    plt.title(f'Top 20 Most Common Words in {text_column.capitalize()}')
    plt.xlabel('Frequency')
    plt.tight_layout()
    plt.savefig(f'../output/eda_plots/{text_column}_top_words.png')
    plt.close()

def analyze_time_series(df, date_column='date'):
    """Analyze time series patterns in the data."""
    print("\n=== Time Series Analysis ===")
    
    # Ensure date is datetime
    df[date_column] = pd.to_datetime(df[date_column])
    
    # Set date as index for resampling
    df.set_index(date_column, inplace=True)
    
    # Resample by day and count articles
    daily_counts = df.resample('D').size()
    
    # Plot publication frequency over time
    plt.figure(figsize=(14, 6))
    daily_counts.plot()
    plt.title('Number of Articles Published Daily')
    plt.xlabel('Date')
    plt.ylabel('Number of Articles')
    plt.grid(True)
    plt.savefig('../output/eda_plots/daily_publication_frequency.png')
    plt.close()
    
    # Analyze by day of week
    df['day_of_week'] = df.index.day_name()
    
    # Count by day of week
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    day_counts = df['day_of_week'].value_counts().reindex(day_order)
    
    # Plot by day of week
    plt.figure(figsize=(10, 6))
    sns.barplot(x=day_counts.index, y=day_counts.values, order=day_order)
    plt.title('Number of Articles by Day of Week')
    plt.xlabel('Day of Week')
    plt.ylabel('Number of Articles')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('../output/eda_plots/publication_by_weekday.png')
    plt.close()
    
    # Reset index for further analysis
    df.reset_index(inplace=True)

def analyze_publishers(df, publisher_column='publisher'):
    """Analyze publisher information and patterns."""
    print("\n=== Publisher Analysis ===")
    
    # Count articles per publisher
    publisher_counts = df[publisher_column].value_counts().reset_index()
    publisher_counts.columns = ['publisher', 'article_count']
    
    print(f"\nTotal unique publishers: {len(publisher_counts)}")
    print("\nTop 10 publishers by article count:")
    print(publisher_counts.head(10).to_string(index=False))
    
    # Plot top publishers
    plt.figure(figsize=(12, 6))
    sns.barplot(
        x='article_count', 
        y='publisher', 
        data=publisher_counts.head(20),
        palette='viridis'
    )
    plt.title('Top 20 Publishers by Number of Articles')
    plt.xlabel('Number of Articles')
    plt.ylabel('Publisher')
    plt.tight_layout()
    plt.savefig('../output/eda_plots/top_publishers.png')
    plt.close()
    
    # Extract domains if publisher is email
    if df[publisher_column].str.contains('@').any():
        print("\nPublisher email domains found. Extracting domains...")
        df['publisher_domain'] = df[publisher_column].str.extract(r'@([\w.-]+)')
        domain_counts = df['publisher_domain'].value_counts().reset_index()
        domain_counts.columns = ['domain', 'count']
        
        print("\nTop 10 publisher domains:")
        print(domain_counts.head(10).to_string(index=False))
        
        # Plot top domains
        plt.figure(figsize=(12, 6))
        sns.barplot(
            x='count', 
            y='domain', 
            data=domain_counts.head(20),
            palette='viridis'
        )
        plt.title('Top 20 Publisher Domains')
        plt.xlabel('Number of Articles')
        plt.ylabel('Domain')
        plt.tight_layout()
        plt.savefig('../output/eda_plots/top_publisher_domains.png')
        plt.close()

def main():
    # Download required NLTK data
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
    
    # File path
    file_path = '../raw_analyst_ratings (1).csv'
    
    # Load data
    print("Loading data...")
    df = load_data(file_path)
    
    if df is not None:
        # Perform analyses
        basic_data_analysis(df)
        analyze_text(df, 'headline')
        analyze_text(df, 'summary')
        analyze_time_series(df)
        analyze_publishers(df)
        
        print("\n=== Analysis Complete ===")
        print(f"Plots saved to: {os.path.abspath('../output/eda_plots/')}")

if __name__ == "__main__":
    main()
