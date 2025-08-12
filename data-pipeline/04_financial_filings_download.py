from pathlib import Path
import datetime
from typing import Dict, Tuple, Optional
import pandas as pd
import polars as pl
from vnstock import Finance
import pickle

REPO_DIR = Path(__file__).parent.parent
DATA_DIR = REPO_DIR / "data"
PRICE_DIR = DATA_DIR / "01_price"
FILING_DIR = DATA_DIR / "04_financial_filings"
FILING_DIR.mkdir(parents=True, exist_ok=True)

START_DATE_STR: Optional[str] = None
END_DATE_STR: Optional[str] = None
AVAILABILITY_LAG_DAYS: int = 20


def _get_period_end_date(year: int, period: int = 4) -> datetime.date:
    """Get end date for quarter or year."""
    month = period * 3 if period <= 4 else 12
    return datetime.date(year, month, 30)

def _get_availability_date(year: int, period: int = 4) -> datetime.date:
    """Get availability date for a report."""
    end_date = _get_period_end_date(year, period)
    return end_date + datetime.timedelta(days=AVAILABILITY_LAG_DAYS)

def _safe_get(data, key: str) -> Optional[float]:
    """Extract value safely from DataFrame or Series."""
    try:
        if isinstance(data, pd.DataFrame):
            return data[key].values[0] if not data.empty and key in data.columns else None
        return data[key] if key in data else None
    except Exception:
        return None

def _safe_get_multi_index(df: pd.DataFrame, main_key: str, sub_key: str) -> Optional[float]:
    """Get value from multi-index DataFrame."""
    try:
        return df[(main_key, sub_key)].values[0] if (main_key, sub_key) in df.columns else None
    except Exception:
        return None


def _get_finance_data(finance: Finance, period: str, lang: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
    """Get all financial data for a period."""
    income_statement = finance.income_statement(period=period, lang=lang)
    balance_sheet = finance.balance_sheet(period=period, lang=lang)
    cash_flow = finance.cash_flow(period=period, lang=lang)
    
    try:
        ratio_df = finance.ratio(period=period, lang=lang)
    except:
        try:
            ratio_df = finance.ratio(period=period, lang="")
        except:
            ratio_df = None
            
    return income_statement, balance_sheet, cash_flow, ratio_df

def _get_aligned_rows(year: int, quarter: Optional[int], df: pd.DataFrame) -> pd.DataFrame:
    """Get rows aligned by year and optional quarter."""
    filter_cond = (df["yearReport"] == year)
    if quarter is not None:
        filter_cond &= (df["lengthReport"] == quarter)
    return df[filter_cond]

def _get_aligned_rows_multi(year: int, quarter: Optional[int], df: pd.DataFrame) -> pd.DataFrame:
    """Get rows aligned by year and optional quarter."""
    quarter = 5 if quarter is None else quarter
    year_cond = (df[("Meta", "yearReport")] == year)
    length_cond = (df[("Meta", "lengthReport")] == quarter) 
    return df[year_cond & length_cond]


def _build_indicators(row: pd.DataFrame, bs_row: pd.DataFrame, cf_row: pd.DataFrame, 
                     ratio_row: Optional[pd.DataFrame], year: int, quarter: Optional[int], 
                     report_date: datetime.date, report_type: str, symbol: str) -> Dict:
    """Build financial indicators dictionary."""
    indicators = {
        "year": year,
        "quarter": quarter,
        "report_date": str(report_date),
        "report_type": report_type,
        
        # Income Statement
        "revenue_yoy_pct": _safe_get(row, "Revenue YoY (%)"),
        "attribute_to_parent_yoy_pct": _safe_get(row, "Attribute to parent company YoY (%)"),
        "revenue": _safe_get(row, "Revenue (Bn. VND)"),
        "gross_profit": _safe_get(row, "Gross Profit"),
        "operating_profit": _safe_get(row, "Operating Profit/Loss"),
        "attribute_to_parent": _safe_get(row, "Attribute to parent company (Bn. VND)"),
        "interest_expenses": _safe_get(row, "Interest Expenses"),
        
        # Balance Sheet
        "cash_equivalents": _safe_get(bs_row, "Cash and cash equivalents (Bn. VND)"),
        "accounts_receivable": _safe_get(bs_row, "Accounts receivable (Bn. VND)"),
        "net_inventories": _safe_get(bs_row, "Net Inventories"),
        "total_assets": _safe_get(bs_row, "TOTAL ASSETS (Bn. VND)"),
        "short_term_borrowings": _safe_get(bs_row, "Short-term borrowings (Bn. VND)"),
        "long_term_borrowings": _safe_get(bs_row, "Long-term borrowings (Bn. VND)"),
        "total_liabilities": _safe_get(bs_row, "LIABILITIES (Bn. VND)"),
        "total_equity": _safe_get(bs_row, "OWNER'S EQUITY(Bn.VND)"),
        
        # Cash Flow
        "operating_cash_flow": _safe_get(cf_row, "Net cash inflows/outflows from operating activities"),
        "investing_cash_flow": _safe_get(cf_row, "Net Cash Flows from Investing Activities"),
        "financing_cash_flow": _safe_get(cf_row, "Cash flows from financial activities"),
        "purchase_fixed_assets": _safe_get(cf_row, "Purchase of fixed assets"),
        "dividends_paid": _safe_get(cf_row, "Dividends paid"),
        
        "ratios": {}
    }
    
    # Add ratios if available
    if isinstance(ratio_row, pd.DataFrame) and not ratio_row.empty:
        ratios = {
            "borrowings_to_equity": _safe_get_multi_index(ratio_row, "Chỉ tiêu cơ cấu nguồn vốn", "(ST+LT borrowings)/Equity"),
            "debt_to_equity": _safe_get_multi_index(ratio_row, "Chỉ tiêu cơ cấu nguồn vốn", "Debt/Equity"),
            "asset_turnover": _safe_get_multi_index(ratio_row, "Chỉ tiêu hiệu quả hoạt động", "Asset Turnover"),
            "days_sales_outstanding": _safe_get_multi_index(ratio_row, "Chỉ tiêu hiệu quả hoạt động", "Days Sales Outstanding"),
            "days_inventory_outstanding": _safe_get_multi_index(ratio_row, "Chỉ tiêu hiệu quả hoạt động", "Days Inventory Outstanding"),
            "days_payable_outstanding": _safe_get_multi_index(ratio_row, "Chỉ tiêu hiệu quả hoạt động", "Days Payable Outstanding"),
            "cash_cycle": _safe_get_multi_index(ratio_row, "Chỉ tiêu hiệu quả hoạt động", "Cash Cycle"),
            "inventory_turnover": _safe_get_multi_index(ratio_row, "Chỉ tiêu hiệu quả hoạt động", "Inventory Turnover"),
            "ebit_margin_pct": _safe_get_multi_index(ratio_row, "Chỉ tiêu khả năng sinh lợi", "EBIT Margin (%)"),
            "gross_profit_margin_pct": _safe_get_multi_index(ratio_row, "Chỉ tiêu khả năng sinh lợi", "Gross Profit Margin (%)"),
            "net_profit_margin_pct": _safe_get_multi_index(ratio_row, "Chỉ tiêu khả năng sinh lợi", "Net Profit Margin (%)"),
            "roe_pct": _safe_get_multi_index(ratio_row, "Chỉ tiêu khả năng sinh lợi", "ROE (%)"),
            "roic_pct": _safe_get_multi_index(ratio_row, "Chỉ tiêu khả năng sinh lợi", "ROIC (%)"),
            "roa_pct": _safe_get_multi_index(ratio_row, "Chỉ tiêu khả năng sinh lợi", "ROA (%)"),
            "dividend_yield_pct": _safe_get_multi_index(ratio_row, "Chỉ tiêu khả năng sinh lợi", "Dividend yield (%)"),
            "current_ratio": _safe_get_multi_index(ratio_row, "Chỉ tiêu thanh khoản", "Current Ratio"),
            "cash_ratio": _safe_get_multi_index(ratio_row, "Chỉ tiêu thanh khoản", "Cash Ratio"),
            "quick_ratio": _safe_get_multi_index(ratio_row, "Chỉ tiêu thanh khoản", "Quick Ratio"),
            "interest_coverage": _safe_get_multi_index(ratio_row, "Chỉ tiêu thanh khoản", "Interest Coverage"),
            "financial_leverage": _safe_get_multi_index(ratio_row, "Chỉ tiêu thanh khoản", "Financial Leverage"),
            "market_capital": _safe_get_multi_index(ratio_row, "Chỉ tiêu định giá", "Market Capital (Bn. VND)"),
            "pe": _safe_get_multi_index(ratio_row, "Chỉ tiêu định giá", "P/E"),
            "pb": _safe_get_multi_index(ratio_row, "Chỉ tiêu định giá", "P/B"),
            "ps": _safe_get_multi_index(ratio_row, "Chỉ tiêu định giá", "P/S"),
            "pcf": _safe_get_multi_index(ratio_row, "Chỉ tiêu định giá", "P/Cash Flow"),
            "eps": _safe_get_multi_index(ratio_row, "Chỉ tiêu định giá", "EPS (VND)"),
            "bvps": _safe_get_multi_index(ratio_row, "Chỉ tiêu định giá", "BVPS (VND)"),
            "ev_ebitda": _safe_get_multi_index(ratio_row, "Chỉ tiêu định giá", "EV/EBITDA"),
        }
        indicators["ratios"] = {k: v for k, v in ratios.items() if v is not None}
    
    return indicators

def process_statements(symbol: str, period: str = "quarter", lang: str = "en",
                      start_date: Optional[datetime.date] = None,
                      end_date: Optional[datetime.date] = None) -> Dict[datetime.date, Dict]:
    """Process financial statements for a given period type."""
    print(f"Processing {period} statements for {symbol}...")
    reports = {}
    
    try:
        finance = Finance(symbol=symbol, source="VCI")
        income_statement, balance_sheet, cash_flow, ratio_df = _get_finance_data(finance, period, lang)
        
        for _, row in income_statement.iterrows():
            year = int(row["yearReport"])
            quarter = int(row["lengthReport"]) if period == "quarter" else None
            
            report_date = _get_availability_date(year, quarter if period == "quarter" else 4)
                
            bs_row = _get_aligned_rows(year, quarter, balance_sheet)
            cf_row = _get_aligned_rows(year, quarter, cash_flow)
            ratio_row = _get_aligned_rows_multi(year, quarter, ratio_df) 
            indicators = _build_indicators(row, bs_row, cf_row, ratio_row, year, quarter, 
                                        report_date, period, symbol)
            filing_type = "filing_q" if period == "quarter" else "filing_k"
            reports[report_date] = {filing_type: {symbol: indicators}}
            
    except Exception as e:
        print(f"Error processing {period} statements: {e}")
        
    return reports


def broadcast_filings_to_all_days(reports: Dict[datetime.date, Dict],
                              start_date: datetime.date, 
                              end_date: datetime.date,
                              is_quarterly: bool = True) -> Dict[datetime.date, Dict]:
    """Broadcast filings to all days between start_date and end_date."""
    availability = {}
    filing_type = "filing_q" if is_quarterly else "filing_k"
    
    # Map report dates to availability dates
    for rdate, payload in reports.items():
        avail = rdate
        availability[avail] = payload

    if not availability:
        return {}
    
    s = pd.Series(availability)
    s.index = pd.to_datetime(s.index)

    date_range = pd.date_range(start = start_date, end = end_date, freq= "D")
    combined_index = s.index.union(date_range)
    daily_filing_series = s.reindex(combined_index).ffill()
    
    final_series = daily_filing_series.loc[date_range]
    final_series = final_series.apply(lambda x: x if pd.notna(x) else {filing_type: {}})
    return final_series.to_dict()

def save_financial_data(filing_q: Dict[datetime.date, Dict],
                     filing_k: Dict[datetime.date, Dict],
                     symbol: str,
                     output_dir: Path = FILING_DIR) -> None:
    """Save financial data to pickle files and print summary."""
    # Save files
    with open(output_dir / f"{symbol}_filing_q.pkl", "wb") as f:
        pickle.dump(filing_q, f)
    with open(output_dir / f"{symbol}_filing_k.pkl", "wb") as f:
        pickle.dump(filing_k, f)

    # Count non-empty filings
    q_count = sum(1 for data in filing_q.values() if data.get("filing_q"))
    k_count = sum(1 for data in filing_k.values() if data.get("filing_k"))
    
    print(f"\nSaved {q_count}/{len(filing_q)} quarterly and {k_count}/{len(filing_k)} annual reports")
    print(f"Data saved to: {output_dir}/{symbol}_filing_[q,k].pkl")

    # Print sample data
    for date in sorted(filing_q.keys()):
        if filing_q[date].get("filing_q"):
            data = filing_q[date]["filing_q"][symbol]
            print(f"\nSample quarterly filing for {date}:")
            print(f"  Year: {data.get('year')}, Quarter: {data.get('quarter')}")
            print(f"  Revenue: {data.get('revenue')} Bn. VND")
            break
            
    for date in sorted(filing_k.keys(), reverse= True):
        if filing_k[date].get("filing_k"):
            data = filing_k[date]["filing_k"][symbol]
            print(f"\nSample annual filing for {date}:")
            print(f"  Year: {data.get('year')}")
            print(f"  Revenue: {data.get('gross_profit_margin_pct')} Bn. VND")
            break


def main(symbol: str = "FPT", lang: str = "en") -> None:
    """Download and process financial filing data."""
    # Get date range
    if START_DATE_STR and END_DATE_STR:
        start_date = pd.to_datetime(START_DATE_STR).date()
        end_date = pd.to_datetime(END_DATE_STR).date()
        print(f"Using date range: {start_date} to {end_date}")
    else:
        price_path = PRICE_DIR / f"{symbol}_price.parquet"
        if not price_path.exists():
            raise FileNotFoundError(f"Price parquet not found: {price_path}")
        price_data = pl.read_parquet(price_path)
        start_date = pd.to_datetime(price_data["time"].min()).date()
        end_date = pd.to_datetime(price_data["time"].max()).date()
        print(f"Using price data range: {start_date} to {end_date}")

    # Process and broadcast reports
    print("\nProcessing statements...")
    filing_q = process_statements(symbol, "quarter", lang, start_date, end_date)
    filing_k = process_statements(symbol, "year", lang, start_date, end_date)
    q_count = sum(1 for data in filing_q.values() if data.get("filing_q"))
    k_count = sum(1 for data in filing_k.values() if data.get("filing_k"))
    
    print(f"\nProcessing {q_count}/{len(filing_q)} quarterly and {k_count}/{len(filing_k)} annual reports")

    print("\nBroadcasting to daily records...")
    daily_filing_q = broadcast_filings_to_all_days(filing_q, start_date, end_date, True)
    daily_filing_k = broadcast_filings_to_all_days(filing_k, start_date, end_date, False)

    save_financial_data(daily_filing_q, daily_filing_k, symbol)


def analyze_filing_data(symbol: str):
    """Analyze the content and structure of filing data files.
    
    Args:
        symbol (str): Stock symbol (e.g., 'FPT')
    """
    q_file = FILING_DIR / f"{symbol}_filing_q.pkl"
    k_file = FILING_DIR / f"{symbol}_filing_k.pkl"
    
    if not (q_file.exists() and k_file.exists()):
        print(f"Financial data files not found for {symbol}")
        return
        
    with open(q_file, "rb") as f:
        filing_q = pickle.load(f)
    with open(k_file, "rb") as f:
        filing_k = pickle.load(f)
    
    print(f"\nFinancial Filing Analysis for {symbol}")
    print("=" * 80)
    
    # 1. Date Range Analysis
    q_dates = sorted(filing_q.keys())
    k_dates = sorted(filing_k.keys())
    
    print("\n1. Date Range Information:")
    print("-" * 80)
    print(f"Quarterly Filing Date Range: {q_dates[0]} to {q_dates[-1]}")
    print(f"Annual Filing Date Range: {k_dates[0]} to {k_dates[-1]}")
    print(f"Total Days with Data:")
    print(f"  - Quarterly: {len(q_dates):,} days")
    print(f"  - Annual: {len(k_dates):,} days")
    
    # 2. Data Availability Analysis
    def count_reports(data_dict, filing_type):
        return sum(1 for d in data_dict.values() if d.get(filing_type) and d[filing_type].get(symbol))
    
    q_reports = count_reports(filing_q, "filing_q")
    k_reports = count_reports(filing_k, "filing_k")
    
    print("\n2. Data Availability:")
    print("-" * 80)
    print(f"Days with Quarterly Reports: {q_reports:,}/{len(q_dates):,} ({q_reports/len(q_dates)*100:.1f}%)")
    print(f"Days with Annual Reports: {k_reports:,}/{len(k_dates):,} ({k_reports/len(k_dates)*100:.1f}%)")
    
    # 3. Report Frequency Analysis
    def get_unique_reports(data_dict, filing_type):
        unique_reports = set()
        for data in data_dict.values():
            if data.get(filing_type) and data[filing_type].get(symbol):
                report = data[filing_type][symbol]
                key = (report.get('year'), report.get('quarter'))
                unique_reports.add(key)
        return sorted(unique_reports)
    
    q_unique = get_unique_reports(filing_q, "filing_q")
    k_unique = get_unique_reports(filing_k, "filing_k")
    
    print("\n3. Unique Reports:")
    print("-" * 80)
    print(f"Total Unique Quarterly Reports: {len(q_unique)}")
    print("Most Recent Quarterly Reports:")
    for year, quarter in q_unique[-5:]:
        print(f"  - Year: {year}, Quarter: {quarter}")
        
    print(f"\nTotal Unique Annual Reports: {len(k_unique)}")
    print("Most Recent Annual Reports:")
    for year, _ in k_unique[-5:]:
        print(f"  - Year: {year}")
    
    # 4. Data Structure Analysis
    def get_sample_data(data_dict, filing_type):
        for data in data_dict.values():
            if data.get(filing_type) and data[filing_type].get(symbol):
                return data[filing_type][symbol]
        return None
    
    print("\n4. Data Structure:")
    print("-" * 80)
    
    sample_q = get_sample_data(filing_q, "filing_q")
    sample_k = get_sample_data(filing_k, "filing_k")
    
    if sample_q:
        print("\nQuarterly Reports Structure:")
        print("Main indicators:")
        for key in sorted(k for k in sample_q.keys() if k != "ratios"):
            value = sample_q[key]
            value_type = type(value).__name__
            print(f"  - {key:<25} ({value_type})")
        
        if "ratios" in sample_q:
            print("\n  Financial Ratios Available:")
            for key in sorted(sample_q["ratios"].keys()):
                value = sample_q["ratios"][key]
                value_type = type(value).__name__
                print(f"    - {key:<23} ({value_type})")
    
    # 5. Data Continuity Analysis
    def analyze_gaps(dates):
        if not dates:
            return []
        gaps = []
        for i in range(1, len(dates)):
            gap = (dates[i] - dates[i-1]).days
            if gap > 1:  # More than 1 day between reports
                gaps.append((dates[i-1], dates[i], gap))
        return gaps
    
    print("\n5. Data Continuity Analysis:")
    print("-" * 80)
    
    # Get dates where actual reports exist (not just propagated data)
    q_report_dates = [date for date in q_dates if filing_q[date].get("filing_q")]
    k_report_dates = [date for date in k_dates if filing_k[date].get("filing_k")]
    
    q_gaps = analyze_gaps(q_report_dates)
    k_gaps = analyze_gaps(k_report_dates)
    
    print("\nLargest gaps between consecutive reports:")
    print("\nQuarterly Reports:")
    for prev, next, days in sorted(q_gaps, key=lambda x: x[2], reverse=True)[:3]:
        print(f"  {prev} to {next} ({days} days)")
    
    print("\nAnnual Reports:")
    for prev, next, days in sorted(k_gaps, key=lambda x: x[2], reverse=True)[:3]:
        print(f"  {prev} to {next} ({days} days)")

def display_filing_sample(symbol: str):
    """Display a sample entry from both quarterly and annual filing files.
    
    Args:
        symbol (str): Stock symbol (e.g., 'FPT')
    """
    # Load the financial data
    q_file = FILING_DIR / f"{symbol}_filing_q.pkl"
    k_file = FILING_DIR / f"{symbol}_filing_k.pkl"
    
    if not (q_file.exists() and k_file.exists()):
        print(f"Financial data files not found for {symbol}")
        return
        
    with open(q_file, "rb") as f:
        filing_q = pickle.load(f)
    with open(k_file, "rb") as f:
        filing_k = pickle.load(f)
    
    print(f"\nFinancial Filing Sample Data for {symbol}")
    print("=" * 80)
    
    # Get one sample from each file
    def get_sample_data(data_dict, filing_type):
        for date, data in data_dict.items():
            if data.get(filing_type) and data[filing_type].get(symbol):
                return date, data[filing_type][symbol]
        return None, None
    
    q_date, q_data = get_sample_data(filing_q, "filing_q")
    k_date, k_data = get_sample_data(filing_k, "filing_k")
    
    # Display quarterly data sample
    print("\nQuarterly Filing Sample:")
    print("-" * 80)
    if q_date and q_data:
        print(f"Sample date: {q_date}")
        print("\nMain indicators:")
        for i, (key, value) in enumerate(sorted(q_data.items())):
            if key != "ratios":
                print(f"[{i:2d}] {key:<25}: {value}")
        
        if "ratios" in q_data:
            print("\nFinancial Ratios:")
            for i, (key, value) in enumerate(sorted(q_data["ratios"].items())):
                print(f"[{i:2d}] {key:<25}: {value}")
    else:
        print("No quarterly data found")
    
    # Display annual data sample
    print("\nAnnual Filing Sample:")
    print("-" * 80)
    if k_date and k_data:
        print(f"Sample date: {k_date}")
        print("\nMain indicators:")
        for i, (key, value) in enumerate(sorted(k_data.items())):
            if key != "ratios":
                print(f"[{i:2d}] {key:<25}: {value}")
        
        if "ratios" in k_data:
            print("\nFinancial Ratios:")
            for i, (key, value) in enumerate(sorted(k_data["ratios"].items())):
                print(f"[{i:2d}] {key:<25}: {value}")
    else:
        print("No annual data found")
    
    # File info
    print("\nFile Information:")
    print("-" * 80)
    
    # Display all available keys with their indices
    if q_data:
        print("\nQuarterly Data Keys:")
        for i, key in enumerate(sorted(q_data.keys())):
            if key != "ratios":
                value = q_data[key]
                print(f"  [{i}] {key}: {value}")
        
        if "ratios" in q_data:
            print("\nQuarterly Ratios Keys:")
            for i, key in enumerate(sorted(q_data["ratios"].keys())):
                value = q_data["ratios"][key]
                print(f"  [{i}] {key}: {value}")
    
    if k_data:
        print("\nAnnual Data Keys:")
        for i, key in enumerate(sorted(k_data.keys())):
            if key != "ratios":
                value = k_data[key]
                print(f"  [{i}] {key}: {value}")
        
        if "ratios" in k_data:
            print("\nAnnual Ratios Keys:")
            for i, key in enumerate(sorted(k_data["ratios"].keys())):
                value = k_data["ratios"][key]
                print(f"  [{i}] {key}: {value}")




if __name__ == "__main__":
    main()
    # Uncomment to test
    #display_filing_sample("FPT")