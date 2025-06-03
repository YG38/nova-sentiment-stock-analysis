"""
Tests for the EDA script.
"""
import os
import sys
import pytest
import pandas as pd

# Add the parent directory to the path so we can import the script
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the functions we want to test
from scripts.task1_eda import load_data, basic_data_analysis

# Test data path
TEST_DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'raw_analyst_ratings (1).csv')

def test_load_data():
    """Test that data loads correctly."""
    # Test loading a small sample of data
    df = load_data(TEST_DATA_PATH, sample_size=100)
    
    # Check that we got a DataFrame back
    assert isinstance(df, pd.DataFrame)
    
    # Check that we have some data
    assert not df.empty
    assert len(df) > 0
    
    # Check that required columns exist
    required_columns = ['headline', 'date', 'publisher']
    for col in required_columns:
        assert col in df.columns

def test_basic_data_analysis(capsys):
    """Test that basic data analysis runs without errors."""
    # Load a small sample of data
    df = load_data(TEST_DATA_PATH, sample_size=100)
    
    # Run the analysis (should not raise exceptions)
    basic_data_analysis(df)
    
    # Capture the output
    captured = capsys.readouterr()
    
    # Check that we got some output
    assert "Data shape:" in captured.out
    assert "Data Types and Missing Values:" in captured.out

# This test requires a bit more setup, so we'll mark it as a slow test
@pytest.mark.slow
def test_analyze_text(tmp_path):
    """Test text analysis functions."""
    # This test is more complex and requires mocking or test data
    # We'll just test that the function exists and can be imported
    from scripts.task1_eda import analyze_text
    assert callable(analyze_text)

# Run tests with: python -m pytest tests/test_eda.py -v
