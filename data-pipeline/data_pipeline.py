# !pip install pyarrow
# !pip install fastparquet
# !pip install polars

import glob
import pickle
import datetime
import os
import pandas as pd
import polars as pl
from pathlib import Path
from typing import List, Tuple, Dict, Any


#const
REPO_DIR = Path(__file__).parent.parent
DATA_DIR = REPO_DIR / "data"
PRICE_DIR = DATA_DIR / "01_price"
NEWS_DIR = DATA_DIR / "02_news"
SUMMARY_DIR = DATA_DIR / "03_summary"
FILING_DIR = DATA_DIR / "04_financial_filings"
OUTPUT_DIR = DATA_DIR / "05_model_input_log"

# Ensure directories exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FILING_DIR.mkdir(parents=True, exist_ok=True)


def read_price_data(price_dir: Path, tickers: List[str]) -> Dict[datetime.date, Dict[str, Dict[str, float]]]:
    """Reads price data from parquet files and formats it into a nested dictionary."""
    combined_dict = {}
    
    for ticker in tickers:
        price_file = price_dir / f"{ticker}_price.parquet"
        if not price_file.exists():
            print(f"Warning: Price file for {ticker} not found at {price_file}")
            continue
            
        print(f"Reading price data for {ticker}")
        # Read price data using polars
        df = pl.read_parquet(price_file)
        # Convert to pandas for compatibility with the rest of the code
        pdf = df.to_pandas()
        # Rename columns to match expected format
        pdf = pdf.rename(columns={"time": "date"})
        # Ensure date is in datetime.date format
        pdf["date"] = pd.to_datetime(pdf["date"]).dt.date
        
        # Create a dictionary for each date with price structure
        for date, row in pdf.iterrows():
            if date not in combined_dict:
                combined_dict[row["date"]] = {"price": {}}
            combined_dict[row["date"]]["price"][ticker] = row["close"]
    
    # Save the combined dictionary
    pkl_filename = price_dir / "price.pkl"
    with open(pkl_filename, "wb") as file:
        pickle.dump(combined_dict, file)
    print(f"Price data saved to: {pkl_filename}")
    
    return combined_dict

def read_news_summary(summary_dir: Path, tickers: List[str]) -> Dict[datetime.date, Dict[str, Dict[str, List[str]]]]:
    """Reads news summary data from parquet files and formats it into a nested dictionary."""
    combined_dict = {}
    
    for ticker in tickers:
        summary_file = summary_dir / f"{ticker}_summary.parquet"
        if not summary_file.exists():
            print(f"Warning: Summary file for {ticker} not found at {summary_file}")
            continue
            
        print(f"Reading news summaries for {ticker}")
        # Read summary data using polars
        df = pl.read_parquet(summary_file)
        # Convert to pandas for compatibility with the rest of the code
        pdf = df.to_pandas()
        # Ensure date is in datetime.date format
        pdf["date"] = pd.to_datetime(pdf["date"]).dt.date
        
        # Group summaries by date
        for date, group in pdf.groupby("date"):
            if date not in combined_dict:
                combined_dict[date] = {"news": {}}
            # Store list of summaries for this ticker and date
            combined_dict[date]["news"][ticker] = group["summary"].tolist()
    
    # Save the combined dictionary
    pkl_filename = summary_dir / "news.pkl"
    with open(pkl_filename, "wb") as file:
        pickle.dump(combined_dict, file)
    print(f"News data saved to: {pkl_filename}")
    
    return combined_dict

def read_filing_data(data_dir: Path, tickers: List[str], price_dates: set = None) -> Tuple[Dict, Dict]:
    """
    Reads financial filing data for quarterly and annual reports.
    
    Args:
        data_dir: Path to the directory containing financial filings
        tickers: List of stock tickers
        price_dates: Optional set of dates for which price data exists
        
    Returns:
        Tuple of (quarterly_filings, annual_filings) dictionaries
    """
    filing_dir = data_dir / "04_financial_filings"
    filing_q = {}
    filing_k = {}
    
    for ticker in tickers:
        # Check for quarterly filing data
        q_file = filing_dir / f"{ticker}_filing_q.pkl"
        if q_file.exists():
            print(f"Reading quarterly filing data for {ticker}")
            with open(q_file, 'rb') as f:
                ticker_q = pickle.load(f)
                # Merge into main dictionary, filtering by price dates if provided
                for date, data in ticker_q.items():
                    if price_dates is None or date in price_dates:
                        filing_q[date] = data
                    # else we skip this date since it doesn't have price data
            print(f"Loaded {len(filing_q)} quarterly filings")
        else:
            print(f"Warning: Quarterly filing data for {ticker} not found at {q_file}")
            
        # Check for annual filing data
        k_file = filing_dir / f"{ticker}_filing_k.pkl"
        if k_file.exists():
            print(f"Reading annual filing data for {ticker}")
            with open(k_file, 'rb') as f:
                ticker_k = pickle.load(f)
                # Merge into main dictionary, filtering by price dates if provided
                for date, data in ticker_k.items():
                    if price_dates is None or date in price_dates:
                        filing_k[date] = data
                    # else we skip this date since it doesn't have price data
            print(f"Loaded {len(filing_k)} annual filings")
        else:
            print(f"Warning: Annual filing data for {ticker} not found at {k_file}")
    
    return filing_q, filing_k
    
if __name__ == '__main__':

    # Define parameters
    START_DATE = '2020-01-01'
    END_DATE = '2024-05-25'
    tickers = ['FPT']  
    col_name = 'summary'    
    
    # Read data from parquet files instead of downloading
    print("Reading price data...")
    price = read_price_data(PRICE_DIR, tickers)
    
    print("Reading news summary data...")
    news = read_news_summary(SUMMARY_DIR, tickers)
    
    print("Reading financial filing data...")
    # Pass the set of dates with price data to filter filings
    price_dates = set(price.keys())
    q, k = read_filing_data(DATA_DIR, tickers, price_dates)
    
    # Update dictionaries to ensure all dates are represented in all datasets
    all_dates = set(price.keys()) | set(news.keys()) | set(q.keys()) | set(k.keys())
    print(f"Total unique dates across all datasets: {len(all_dates)}")
    
    for date in all_dates:
        if date not in price:
            print(f"Warning: Missing price data for {date}")
        if date not in news:
            news[date] = {'news': {ticker: [] for ticker in tickers}}
        else:
            # Ensure all tickers have news entries, even if empty
            for ticker in tickers:
                if ticker not in news[date]['news']:
                    news[date]['news'][ticker] = []
                    
    # Combining data
    env_data = {}
    for key in sorted(price.keys()):
        # Make sure all dictionaries have this key
        news_data = news.get(key, {'news': {ticker: [] for ticker in tickers}})
        q_data = q.get(key, {'filing_q': {}})
        k_data = k.get(key, {'filing_k': {}})
        
        # Add data to env_data
        env_data[key] = (price[key], news_data, q_data, k_data)
    
    # Save the combined data
    output_path = OUTPUT_DIR / "env_data.pkl"
    with open(output_path, 'wb') as file:
        pickle.dump(env_data, file)
    
    print(f'Environment data saved to: {output_path}')
    print(f'Total trading days: {len(env_data)}')
    print(f'Date range: {min(env_data.keys())} to {max(env_data.keys())}')
    
    
# Uncomment function below to test reading and viewing the env_data file

def test_env_data(file_path: Path):
    """
    Tests reading the env_data.pkl file and displays its structure and sample content.
    
    Args:
        file_path: Path to the env_data.pkl file
    """
    if not file_path.exists():
        print(f"Error: File {file_path} not found")
        return
        
    print(f"\n{'='*50}")
    print(f"TESTING ENV_DATA FILE: {file_path}")
    print(f"{'='*50}")
    
    # Load the pickle file
    with open(file_path, 'rb') as file:
        env_data = pickle.load(file)
    
    # Display basic info
    total_days = len(env_data)
    print(f"\nTotal trading days: {total_days}")
    
    # Get date range
    dates = sorted(list(env_data.keys()))
    print(f"Date range: {dates[0]} to {dates[-1]}")
    
    # Sample data for first and last day
    sample_dates = [dates[-3], dates[-4]]
    
    for date in sample_dates:
        print(f"\n{'-'*50}")
        print(f"SAMPLE DATA FOR: {date}")
        print(f"{'-'*50}")
        
        # Extract components
        price_data, news_data, filing_q_data, filing_k_data = env_data[date]
        
        # Display price data
        print("\nPRICE DATA:")
        for ticker, price in price_data['price'].items():
            print(f"  {ticker}: {price}")
        
        # Display news data
        print("\nNEWS DATA:")
        for ticker, news_list in news_data['news'].items():
            news_count = len(news_list)
            print(f"  {ticker}: {news_count} news items")
            
            # Show all news items
            if news_count > 0:
                print(f"  All news for {ticker} on {date}:")
                for i, news_item in enumerate(news_list, 1):
                    print(f"  [News {i}]: {news_item}")
                    print(f"  {'-'*30}")
            else:
                print(f"  No news for {ticker} on this date")
        
        # Display filing data status
        print("\nFILING DATA (Q):")
        print(f"  Has filing_q data: {'Yes' if filing_q_data['filing_q'] else 'No'}")
        
        print("\nFILING DATA (K):")
        print(f"  Has filing_k data: {'Yes' if filing_k_data['filing_k'] else 'No'}")

# To run the test function, uncomment this line:
#test_env_data(OUTPUT_DIR / "env_data.pkl")
