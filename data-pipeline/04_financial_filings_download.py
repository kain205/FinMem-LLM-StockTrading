import polars as pl
import pandas as pd
import pickle
import datetime
from pathlib import Path
from vnstock import Finance


#const
REPO_DIR = Path(__file__).parent.parent
DATA_DIR = REPO_DIR / "data"
PRICE_DIR = DATA_DIR / "01_price"
FILING_DIR = DATA_DIR / "04_financial_filings"
FILING_DIR.mkdir(parents=True, exist_ok=True)
START_DATE = '2020-01-01'
END_DATE = '2024-05-25'

def map_quarter_to_date(year, quarter_length):
    """
    Map quarterly report to its corresponding date.
    
    Args:
        year (int): Year of the report
        quarter_length (int): Quarter length (1=Q1, 2=Q2, 3=Q3, 4=Q4)
        
    Returns:
        datetime.date: End date of the quarter
    """
    if quarter_length == 1:  # Q1
        return datetime.date(year, 3, 31)
    elif quarter_length == 2:  # Q2
        return datetime.date(year, 6, 30)
    elif quarter_length == 3:  # Q3
        return datetime.date(year, 9, 30)
    elif quarter_length == 4:  # Q4
        return datetime.date(year, 12, 31)
    else:
        raise ValueError(f"Invalid quarter length: {quarter_length}")

def process_financial_statements(symbol, lang='en', start_date=None, end_date=None):
    """
    Process financial statements for a ticker and format them for the data pipeline.
    
    Args:
        symbol (str): Stock symbol (e.g., 'FPT')
        lang (str): Language for reports ('en' or 'vn')
        start_date (datetime.date, optional): Start date for filtering reports
        end_date (datetime.date, optional): End date for filtering reports
        
    Returns:
        tuple: (filing_q, filing_k) dictionaries for data pipeline
    """
    print(f"Processing financial statements for {symbol}...")
    if start_date and end_date:
        print(f"Filtering reports between {start_date} and {end_date}")
    
    # Initialize dictionaries
    filing_q = {}
    filing_k = {}  # Annual data will be derived from Q4 reports
    
    try:
        # Initialize Finance object
        finance = Finance(symbol=symbol, source='VCI')
        
        # Get quarterly financial statements
        income_statement = finance.income_statement(period='quarterly', lang=lang)
        balance_sheet = finance.balance_sheet(period='quarterly', lang=lang)
        cash_flow = finance.cash_flow(period='quarterly', lang=lang)
        
        # Print debug info
        print(f"Income statement columns: {income_statement.columns.tolist()}")
        print(f"Balance sheet columns: {balance_sheet.columns.tolist()}")
        print(f"Cash flow columns: {cash_flow.columns.tolist()}")
        
        # Process each report
        for index, row in income_statement.iterrows():
            year = row['yearReport']
            quarter = row['lengthReport']
            
            # Map to a date
            report_date = map_quarter_to_date(year, quarter)
            
            # Skip if report date is outside the requested date range
            if (start_date and report_date < start_date) or (end_date and report_date > end_date):
                continue
                
            # Get corresponding data from other statements
            bs_row = balance_sheet[(balance_sheet['yearReport'] == year) & 
                                (balance_sheet['lengthReport'] == quarter)]
            cf_row = cash_flow[(cash_flow['yearReport'] == year) & 
                            (cash_flow['lengthReport'] == quarter)]
            
            if bs_row.empty or cf_row.empty:
                print(f"Warning: Missing data for {year} Q{quarter}")
                continue
            
            # Extract key financial indicators
            key_indicators = {
                # Basic info
                'year': year,
                'quarter': quarter,
                
                # Income statement
                'revenue': row['Revenue (Bn. VND)'] if 'Revenue (Bn. VND)' in row else None,
                'net_profit': row['Net Profit For the Year'] if 'Net Profit For the Year' in row else None,
                'gross_profit': row['Gross Profit'] if 'Gross Profit' in row else None,
                'operating_profit': row['Operating Profit/Loss'] if 'Operating Profit/Loss' in row else None,
                
                # Balance sheet
                'total_assets': bs_row['TOTAL ASSETS (Bn. VND)'].values[0] if 'TOTAL ASSETS (Bn. VND)' in bs_row.columns else None,
                'total_equity': bs_row["OWNER'S EQUITY(Bn.VND)"].values[0] if "OWNER'S EQUITY(Bn.VND)" in bs_row.columns else None,
                'total_liabilities': bs_row['LIABILITIES (Bn. VND)'].values[0] if 'LIABILITIES (Bn. VND)' in bs_row.columns else None,
                'cash_equivalents': bs_row['Cash and cash equivalents (Bn. VND)'].values[0] if 'Cash and cash equivalents (Bn. VND)' in bs_row.columns else None,
                
                # Cash flow
                'operating_cash_flow': cf_row['Net cash inflows/outflows from operating activities'].values[0] if 'Net cash inflows/outflows from operating activities' in cf_row.columns else None,
                'investing_cash_flow': cf_row['Net Cash Flows from Investing Activities'].values[0] if 'Net Cash Flows from Investing Activities' in cf_row.columns else None,
                'financing_cash_flow': cf_row['Cash flows from financial activities'].values[0] if 'Cash flows from financial activities' in cf_row.columns else None,
                
                # Financial ratios will be added later
            }
            
            # Add to quarterly filings dictionary
            filing_q[report_date] = {
                'filing_q': {
                    symbol: key_indicators
                }
            }
            
            # If this is Q4, also add to annual filings
            if quarter == 4:
                filing_k[report_date] = {
                    'filing_k': {
                        symbol: key_indicators
                    }
                }
        
        print(f"Processed {len(filing_q)} quarterly reports and {len(filing_k)} annual reports")
    except Exception as e:
        print(f"Error processing financial statements: {e}")
    
    return filing_q, filing_k

def save_financial_data(filing_q, filing_k, symbol, output_dir=FILING_DIR):
    """
    Save processed financial data to pickle files
    
    Args:
        filing_q (dict): Dictionary of quarterly filing data
        filing_k (dict): Dictionary of annual filing data
        symbol (str): Stock symbol
        output_dir (Path): Directory to save output files
    """
    # Save quarterly data
    q_file = output_dir / f"{symbol}_filing_q.pkl"
    with open(q_file, 'wb') as f:
        pickle.dump(filing_q, f)
    print(f"Quarterly filing data saved to: {q_file}")
    
    # Save annual data
    k_file = output_dir / f"{symbol}_filing_k.pkl"
    with open(k_file, 'wb') as f:
        pickle.dump(filing_k, f)
    print(f"Annual filing data saved to: {k_file}")

def main():
    symbol = "FPT"
    
    # Get price data to determine date range
    price_data = pl.read_parquet(PRICE_DIR / f"{symbol}_price.parquet")
    start_date_str = price_data["time"].min()
    end_date_str = price_data["time"].max()
    print(f"Price data range: {start_date_str} to {end_date_str}")
    
    # Convert to datetime.date objects
    start_date = pd.to_datetime(start_date_str).date()
    end_date = pd.to_datetime(end_date_str).date()
    
    # Process financial statements within the price data date range
    filing_q, filing_k = process_financial_statements(
        symbol, 
        start_date=start_date,
        end_date=end_date
    )
    
    # Save financial data
    save_financial_data(filing_q, filing_k, symbol)
    
    # Print sample data for verification
    if filing_q:
        sample_date = list(filing_q.keys())[0]
        print(f"\nSample quarterly filing data for {sample_date}:")
        print(filing_q[sample_date])
    
    print("\nFinancial data processing complete!")

if __name__ == "__main__":
    main()
