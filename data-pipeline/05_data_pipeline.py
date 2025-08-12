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
SUMMARY_DIR = DATA_DIR / "03_summary"
FILING_DIR = DATA_DIR / "04_financial_filings"
OUTPUT_DIR = DATA_DIR / "05_env_data"

# Ensure directories exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


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
                print(f"  Found {len(ticker_q)} quarterly filings")

                if ticker_q and price_dates: # Đảm bảo không rỗng
                    sample_filing_date = list(ticker_q.keys())[0]
                    sample_price_date = list(price_dates)[0]
                    print("\n--- DEBUGGING DATES ---")
                    print(f"Sample Filing Date: {sample_filing_date} (type: {type(sample_filing_date)})")
                    print(f"Sample Price Date:  {sample_price_date} (type: {type(sample_price_date)})")
                    print("--- END DEBUGGING ---\n")             

                # Merge into main dictionary, filtering by price dates if provided
                count = 0
                for date, data in ticker_q.items():
                    if price_dates is None or date in price_dates:
                        filing_q[date] = data
                        count += 1
                print(f"  Added {count} quarterly filings after price date filtering")
        else:
            print(f"Warning: Quarterly filing data for {ticker} not found at {q_file}")
            
        # Check for annual filing data
        k_file = filing_dir / f"{ticker}_filing_k.pkl"
        if k_file.exists():
            print(f"Reading annual filing data for {ticker}")
            with open(k_file, 'rb') as f:
                ticker_k = pickle.load(f)
                print(f"  Found {len(ticker_k)} annual filings")
                
                # Merge into main dictionary, filtering by price dates if provided
                count = 0
                for date, data in ticker_k.items():
                    if price_dates is None or date in price_dates:
                        filing_k[date] = data
                        count += 1
                print(f"  Added {count} annual filings after price date filtering")
        else:
            print(f"Warning: Annual filing data for {ticker} not found at {k_file}")
    
    return filing_q, filing_k
    
def main():
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
    print(f"Price dates: {len(price.keys())}, News dates: {len(news.keys())}, Q filing dates: {len(q.keys())}, K filing dates: {len(k.keys())}")
    
    # Check for filing dates that match with price dates
    q_price_overlap = set(q.keys()) & set(price.keys())
    k_price_overlap = set(k.keys()) & set(price.keys())
    print(f"Q filing dates that match price dates: {len(q_price_overlap)}")
    print(f"K filing dates that match price dates: {len(k_price_overlap)}")
    
    if len(q_price_overlap) == 0 and len(q.keys()) > 0:
        print("\nWARNING: No quarterly filing dates match price dates!")
        print("Sample filing date:", list(q.keys())[0])
        print("Sample price date:", list(price.keys())[0])
        
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
    filing_q_added = 0
    filing_k_added = 0
    
    for key in sorted(price.keys()):
        # Make sure all dictionaries have this key
        news_data = news.get(key, {'news': {ticker: [] for ticker in tickers}})
        
        # Get filing data if available for this date
        q_data = q.get(key, {'filing_q': {}})
        if key in q:
            filing_q_added += 1
            
        k_data = k.get(key, {'filing_k': {}})
        if key in k:
            filing_k_added += 1
        
        # Add data to env_data
        env_data[key] = (price[key], news_data, q_data, k_data)
    
    print(f"Added filing_q data for {filing_q_added} dates")
    print(f"Added filing_k data for {filing_k_added} dates")
    
    # Save the combined data
    output_path = OUTPUT_DIR / "env_data.pkl"
    with open(output_path, 'wb') as file:
        pickle.dump(env_data, file)
    
    print(f'Environment data saved to: {output_path}')
    print(f'Total trading days: {len(env_data)}')
    print(f'Date range: {min(env_data.keys())} to {max(env_data.keys())}')

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
    
    # Count dates with filing data
    q_count = sum(1 for date in dates if env_data[date][2]['filing_q'])
    k_count = sum(1 for date in dates if env_data[date][3]['filing_k'])
    print(f"\nDates with quarterly filing data: {q_count} out of {total_days}")
    print(f"Dates with annual filing data: {k_count} out of {total_days}")
    
    # Find dates with filing data
    dates_with_q = [date for date in dates if env_data[date][2]['filing_q']]
    dates_with_k = [date for date in dates if env_data[date][3]['filing_k']]
    
    if dates_with_q:
        print("\nSample dates with quarterly filing data:")
        for date in dates_with_q[:3]:
            print(f"  {date}")
    else:
        print("\nNo dates with quarterly filing data found")
        
    if dates_with_k:
        print("\nSample dates with annual filing data:")
        for date in dates_with_k[:3]:
            print(f"  {date}")
    else:
        print("\nNo dates with annual filing data found")
    
    # Sample data for first and last day
    sample_dates = [dates[-3], dates[-250]]
    
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
        if filing_q_data['filing_q']:
            print(f"  Has filing_q data: Yes")
            for ticker in filing_q_data['filing_q']:
                print(f"  Ticker: {ticker}")
                print(f"  GPM: {filing_q_data['filing_q'][ticker]['ratios']['gross_profit_margin_pct']}")
                print(f"  Year: {filing_q_data['filing_q'][ticker]['year']}\nQuarter: {filing_q_data['filing_q'][ticker]['quarter']}")


        else:
            print(f"  Has filing_q data: No")
        
        print("\nFILING DATA (K):")
        if filing_k_data['filing_k']:
            print(f"  Has filing_k data: Yes")
            for ticker in filing_k_data['filing_k']:
                print(f"  Ticker: {ticker}")
                print(f"  GPM: {filing_q_data['filing_q'][ticker]['ratios']['gross_profit_margin_pct']}")
                print(f"  Year: {filing_q_data['filing_q'][ticker]['year']}\nQuarter: {filing_q_data['filing_q'][ticker]['quarter']}")

        else:
            print(f"  Has filing_k data: No")


if __name__ == '__main__':
    main()
    
    # To run the test function after generating the env_data.pkl file
    test_env_data(OUTPUT_DIR / "env_data.pkl")
   
'''
# Cấu trúc của một entry trong env_data
{
    datetime.date(2024, 8, 12): (
        # [0] Dữ liệu giá
        {'price': {'FPT': 135000.0}},

        # [1] Dữ liệu tin tức
        {'news': {'FPT': ['Tóm tắt tin 1...', 'Tóm tắt tin 2...']}},

        # [2] Dữ liệu báo cáo tài chính Quý
        {'filing_q': {'FPT': {'year': 2024, 'quarter': 2, 'revenue': ..., 'ratios': {...}}}},

        # [3] Dữ liệu báo cáo tài chính Năm 
        {'filing_k': {'FPT': {'year': 2023, 'quarter': None, 'revenue': ..., 'ratios': {...}}}}
    )
}
'''
