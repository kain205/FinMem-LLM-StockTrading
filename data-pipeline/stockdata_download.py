#dependencies
import requests
import json
from pathlib import Path
from bs4 import BeautifulSoup
from vnstock import Quote

import polars as pl

#const
REPO_DIR = Path(__file__).parent.parent
DATA_DIR = REPO_DIR / "data"
CUR_DIR = REPO_DIR / "data-pipeline"
SYMBOLS = ["FPT"]
START_DATE = '2020-01-01'
END_DATE = '2024-05-25'

def main():
    try:
        for sym in SYMBOLS:
            print(f"Scraping {sym} stock data")
            quote = Quote(symbol=sym, source='VCI')
            #OHLCV Data
            df = quote.history(start= START_DATE, end= END_DATE)
            df_polars = pl.from_pandas(df)
            df_polars.write_parquet(DATA_DIR / "price.parquet")
            print("Saved")
        
    except Exception as e:
        print(f"Failed to fetch data: {e}")

if __name__ == "__main__":
    main()

