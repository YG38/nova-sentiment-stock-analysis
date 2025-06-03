# Analyst Ratings EDA
# This script performs exploratory data analysis on the raw analyst ratings data.

# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
import os

# Download required NLTK data
nltk.download('vader_lexicon')
nltk.download('punkt')
nltk.download('stopwords')

# Set style for plots
sns.set(style='whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

# Create output directory for plots
os.makedirs('output/plots', exist_ok=True)

def load_data():
    """Load and return the analyst ratings data."""
    print("Loading data...")
    # Get the absolute path to the project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    file_path = os.path.join(project_root, 'raw_analyst_ratings (1).csv')
    print(f"Loading data from: {file_path}")
    df = pd.read_csv(file_path, encoding='latin1')
    print(f"Loaded data shape: {df.shape}")
    return df

def clean_data(df):
    """Clean the dataset."""
    print("\nCleaning data...")
    # Remove duplicates
    initial_rows = len(df)
    df = df.drop_duplicates()
    print(f"Removed {initial_rows - len(df)} duplicate rows")
    
    # Handle missing values
    print("\nMissing values per column:")
    print(df.isnull().sum())
    
    # Add any additional cleaning steps here
    return df

def analyze_descriptive_stats(df):
    """Generate descriptive statistics."""
    print("\nGenerating descriptive statistics...")
    
    # Basic statistics for numerical columns
    print("\nDescriptive statistics for numerical columns:")
    print(df.describe())
    
    # Count of unique values in categorical columns
    print("\nCount of unique values in categorical columns:")
    for column in df.select_dtypes(include=['object']).columns:
        print(f"\n{column}:")
        print(df[column].value_counts().head())
        print(f"Total unique values: {df[column].nunique()}")

def analyze_time_series(df, date_column=None):
    """Analyze time series data if a date column exists."""
    if date_column and date_column in df.columns:
        print(f"\nAnalyzing time series for column: {date_column}")
        try:
            df[date_column] = pd.to_datetime(df[date_column])
            df = df.sort_values(date_column)
            
            # Resample by day and count
            df.set_index(date_column, inplace=True)
            daily_counts = df.resample('D').size()
            
            # Plot time series
            plt.figure(figsize=(14, 6))
            daily_counts.plot()
            plt.title('Number of Articles Over Time')
            plt.xlabel('Date')
            plt.ylabel('Number of Articles')
            plt.tight_layout()
            plt.savefig('output/plots/time_series.png')
            plt.close()
            
            # Reset index for further analysis
            df = df.reset_index()
            return df
            
        except Exception as e:
            print(f"Error in time series analysis: {e}")
    return df

def analyze_text(df, text_column='text'):
    """Analyze text data if a text column exists."""
    if text_column in df.columns:
        print(f"\nAnalyzing text in column: {text_column}")
        
        # Analyze text length
        df['text_length'] = df[text_column].astype(str).apply(len)
        
        # Plot distribution of text lengths
        plt.figure(figsize=(12, 6))
        sns.histplot(df['text_length'], bins=50)
        plt.title('Distribution of Text Lengths')
        plt.xlabel('Text Length')
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.savefig('output/plots/text_length_distribution.png')
        plt.close()
        
        # Sentiment Analysis
        print("Performing sentiment analysis...")
        sia = SentimentIntensityAnalyzer()
        df['sentiment'] = df[text_column].apply(lambda x: sia.polarity_scores(str(x))['compound'])
        
        # Plot sentiment distribution
        plt.figure(figsize=(12, 6))
        sns.histplot(df['sentiment'], bins=50, kde=True)
        plt.title('Distribution of Sentiment Scores')
        plt.xlabel('Sentiment Score')
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.savefig('output/plots/sentiment_distribution.png')
        plt.close()
        
        return df
    return df

def analyze_publishers(df, publisher_column='publisher'):
    """Analyze publisher data if a publisher column exists."""
    if publisher_column in df.columns:
        print(f"\nAnalyzing publishers in column: {publisher_column}")
        
        # Top publishers by article count
        publisher_counts = df[publisher_column].value_counts().head(20)
        
        # Plot top publishers
        plt.figure(figsize=(12, 8))
        sns.barplot(y=publisher_counts.index, x=publisher_counts.values, palette='viridis')
        plt.title('Top 20 Publishers by Number of Articles')
        plt.xlabel('Number of Articles')
        plt.ylabel('Publisher')
        plt.tight_layout()
        plt.savefig('output/plots/top_publishers.png')
        plt.close()
        
        return df
    return df

def analyze_correlations(df):
    """Analyze correlations between numerical variables."""
    print("\nAnalyzing correlations...")
    # Select numerical columns for correlation analysis
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    if len(numerical_cols) > 1:  # Need at least 2 numerical columns
        plt.figure(figsize=(10, 8))
        sns.heatmap(df[numerical_cols].corr(), annot=True, cmap='coolwarm', center=0)
        plt.title('Correlation Matrix of Numerical Variables')
        plt.tight_layout()
        plt.savefig('output/plots/correlation_matrix.png')
        plt.close()

def save_processed_data(df, output_file='../processed_analyst_ratings.csv'):
    """Save the processed data to a CSV file."""
    print(f"\nSaving processed data to {output_file}")
    df.to_csv(output_file, index=False)

def main():
    """Main function to run the EDA."""
    try:
        # Load the data
        df = load_data()
        
        # Clean the data
        df = clean_data(df)
        
        # Analyze descriptive statistics
        analyze_descriptive_stats(df)
        
        # Try to identify date and text columns automatically
        date_columns = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
        text_columns = [col for col in df.columns if 'text' in col.lower() or 'content' in col.lower() or 'headline' in col.lower()]
        
        # Analyze time series if date column is found
        if date_columns:
            df = analyze_time_series(df, date_columns[0])
        else:
            print("\nNo date column found for time series analysis")
        
        # Analyze text if text column is found
        if text_columns:
            df = analyze_text(df, text_columns[0])
        else:
            print("\nNo text column found for text analysis")
        
        # Analyze publishers if publisher column exists
        publisher_columns = [col for col in df.columns if 'publisher' in col.lower() or 'source' in col.lower()]
        if publisher_columns:
            df = analyze_publishers(df, publisher_columns[0])
        
        # Analyze correlations
        analyze_correlations(df)
        
        # Save processed data
        save_processed_data(df)
        
        print("\nEDA completed successfully! Check the 'output/plots' directory for visualizations.")
        
    except Exception as e:
        print(f"\nAn error occurred during EDA: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
