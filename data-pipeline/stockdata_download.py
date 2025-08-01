#dependencies
import requests
import json
from pathlib import Path
from bs4 import BeautifulSoup
from vnstock import Quote
import polars as pl
from datetime import datetime

#const
REPO_DIR = Path(__file__).parent.parent
DATA_DIR = REPO_DIR / "data"
CUR_DIR = REPO_DIR / "data-pipeline"
SYMBOLS = ["FPT"]
START_DATE = '2020-01-01'
END_DATE = '2024-05-25'

def download_stock_data(symbol, start_date, end_date):
    """
    Download stock price data for a single symbol.
    
    Args:
        symbol (str): Stock symbol to download
        start_date (str): Start date in 'YYYY-MM-DD' format
        end_date (str): End date in 'YYYY-MM-DD' format
    
    Returns:
        pl.DataFrame: DataFrame containing OHLCV data for the symbol
    """
    print(f"Downloading {symbol} stock data...")
    quote = Quote(symbol=symbol, source='VCI')
    
    # Get OHLCV Data
    df_pandas = quote.history(start=start_date, end=end_date)
    
    if df_pandas.empty:
        raise ValueError(f"No data found for {symbol}")
        
    # Convert to Polars and add symbol column
    df_polars = pl.from_pandas(df_pandas)

    # Sort by time
    df_polars = df_polars.sort("time")
    
    print(f"Downloaded {len(df_polars)} records for {symbol}")
    return df_polars

def test_parquet_data():
    """
    Test function to preview downloaded parquet data.
    Uncomment in main to use.
    """
    # Change this path to test different symbols
    test_file = DATA_DIR / "FPT_price.parquet"
    
    try:
        df = pl.read_parquet(test_file)
        print(f"\n--- Testing parquet file: {test_file} ---")
        print(f"Shape: {df.shape}")
        print(f"Columns: {df.columns}")
        print(f"Date range: {df['time'].min()} to {df['time'].max()}")
        print("\nFirst 5 rows:")
        print(df.head())
        print("\nLast 5 rows:")
        print(df.tail())
    except Exception as e:
        print(f"Error reading parquet file: {e}")

def main():
    print(f"Starting stock data download for {len(SYMBOLS)} symbols")
    print(f"Date range: {START_DATE} to {END_DATE}")
    
    successful_downloads = 0
    failed_downloads = 0
    
    for symbol in SYMBOLS:
        try:
            # Download data for single symbol
            df = download_stock_data(symbol, START_DATE, END_DATE)
            
            # Save to separate parquet file
            output_file = DATA_DIR / f"{symbol}_price.parquet"
            df.write_parquet(output_file)
            
            print(f"Successfully saved {len(df)} records to {output_file}")
            print(f"Date range: {df['time'].min()} to {df['time'].max()}")
            successful_downloads += 1
            
        except Exception as e:
            print(f"Failed to download {symbol}: {e}")
            failed_downloads += 1
    
    print(f"\nDownload summary:")
    print(f"Successful: {successful_downloads}")
    print(f"Failed: {failed_downloads}")
    print(f"Total: {len(SYMBOLS)}")

if __name__ == "__main__":
    # Uncomment to test reading parquet data
    test_parquet_data()
    #main()

