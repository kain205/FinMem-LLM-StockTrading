import pickle
from tqdm import tqdm
import datetime
from transformers import BertTokenizer, BertForSequenceClassification
#from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import torch
import pandas as pd
import numpy as np
import os
from pathlib import Path

# Constants - consistent with data-pipeline structure
REPO_DIR = Path(__file__).parent.parent
DATA_DIR = REPO_DIR / "data"
PRICE_DIR = DATA_DIR / "01_price"
NEWS_DIR = DATA_DIR / "02_news"
SUMMARY_DIR = DATA_DIR / "03_summary"
FILING_DIR = DATA_DIR / "04_financial_filings"
INPUT_DIR = DATA_DIR / "05_model_input_log"
OUTPUT_DIR = DATA_DIR / "06_sentiment_analysis"

# Ensure directories exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def subset_symbol_dict(input_dir, cur_symbol):
    """
    Extract data for a specific symbol from the environment data.
    
    Args:
        input_dir: Path to the env_data.pkl file
        cur_symbol: Symbol to extract data for
        
    Returns:
        tuple: (new_dict, ticker_dict_byDate)
    """
    new_dict = {}
    with open(input_dir, "rb") as f:
        data = pickle.load(f)
    
    # Debug information
    print(f"Total dates in env_data: {len(data)}")
    sample_date = list(data.keys())[0]
    print(f"Sample date structure: {sample_date}, type: {type(sample_date)}")
    
    # Extract data for the symbol
    new_dict = {}
    ticker_dict_byDate = {}
    
    for k, v in tqdm(data.items()):
        # Unpack the tuple (price, news, filing_q, filing_k)
        cur_price = v[0]['price']  # price
        cur_news = v[1]['news']    # news
        cur_filing_q = v[2]['filing_q']  # form q
        cur_filing_k = v[3]['filing_k']  # form k
        # print('Date: ---------', k)
        # print('Available tickers: ---------',cur_news.keys())

        new_price = {}
        new_filing_k = {}
        new_filing_q = {}
        new_news = {}
        if cur_symbol in list(cur_price.keys()):
            new_price[cur_symbol] = cur_price[cur_symbol]
        if cur_symbol in list(cur_filing_k.keys()):
            new_filing_k[cur_symbol] = cur_filing_k[cur_symbol]
        if cur_symbol in list(cur_filing_q.keys()):
            new_filing_q[cur_symbol] = cur_filing_q[cur_symbol]
        if cur_symbol in list(cur_news.keys()):
            new_news[cur_symbol] = cur_news[cur_symbol]
        else:
            continue

        new_dict[k] = {
            "price": new_price,
            "filing_k": new_filing_k,
            "filing_q": new_filing_q,
            "news": new_news
        }
        ticker_dict_byDate[k] = list(new_dict[k]["price"].keys())
        # print("On date: ", k, "ticker list: ---", ticker_dict_byDate[k])

    return new_dict, ticker_dict_byDate

#### finBERT
# Load the tokenizer and model
tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')
model = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone')

# Function to analyze sentiment
def sentiment_score(text):
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
    outputs = model(**inputs)
    scores = torch.nn.functional.softmax(outputs.logits, dim=-1)
    return scores.tolist()[0]

def assign_finBERT_scores(new_dict, cur_symbol):
    """
    Assign FinBERT sentiment scores to news items.
    
    Args:
        new_dict: Dictionary containing news data
        cur_symbol: Symbol to process
    """
    # Count total news items to process
    total_news = sum(
        len(new_dict[date]['news'].get(cur_symbol, [])) 
        for date in new_dict 
        if cur_symbol in new_dict[date]["news"]
    )
    print(f"Total news items to analyze: {total_news}")
    
    # Create progress bar for dates
    for i_date in tqdm(new_dict, desc="Processing dates"):
        # Skip if symbol not in news
        if cur_symbol not in new_dict[i_date].get("news", {}):
            continue
            
        # Skip if no news for this symbol
        news_items = new_dict[i_date]["news"][cur_symbol]
        if not news_items:
            continue
        
        # Process each news item
        analyzed_news = []
        for news_text in news_items:
            try:
                # Apply FinBERT sentiment analysis
                scores = sentiment_score(news_text)
                
                # Format scores into sentences
                pos_score = scores[2]
                neu_score = scores[1]
                neg_score = scores[0]
                
                sentiment_info = (
                    f"The positive score for this news is {pos_score}. "
                    f"The neutral score for this news is {neu_score}. "
                    f"The negative score for this news is {neg_score}."
                )
                
                # Combine original news with sentiment information
                analyzed_news.append(f"{news_text} {sentiment_info}")
                
            except Exception as e:
                print(f"Error analyzing news: {str(e)[:100]}")
                # Keep original if analysis fails
                analyzed_news.append(news_text)
        
        # Update the news items with sentiment analysis
        new_dict[i_date]["news"][cur_symbol] = analyzed_news


### vader
# VADER sentiment analyzer
# Uncomment the following line to use VADER:
#from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
#analyzer = SentimentIntensityAnalyzer()

def assign_vader_scores(new_dict, cur_symbol):
    """
    Assign VADER sentiment scores to news items.
    Note: Requires the SentimentIntensityAnalyzer import to be uncommented.
    
    Args:
        new_dict: Dictionary containing news data
        cur_symbol: Symbol to process
    """
    # Check if VADER analyzer is imported and available
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        analyzer = SentimentIntensityAnalyzer()
    except (ImportError, ModuleNotFoundError):
        print("VADER analyzer not available. Please install with: pip install vaderSentiment")
        return

    print(f"Processing news with VADER for {cur_symbol}...")
    
    # Count total news items to process
    total_news = sum(
        len(new_dict[date]['news'].get(cur_symbol, [])) 
        for date in new_dict 
        if cur_symbol in new_dict[date]["news"]
    )
    print(f"Total news items to analyze: {total_news}")
    
    # Create progress bar for dates
    for i_date in tqdm(new_dict, desc="Processing dates"):
        # Skip if symbol not in news
        if cur_symbol not in new_dict[i_date].get("news", {}):
            continue
            
        # Skip if no news for this symbol
        news_items = new_dict[i_date]["news"][cur_symbol]
        if not news_items:
            continue
        
        # Process each news item
        analyzed_news = []
        for news_text in news_items:
            try:
                # Apply VADER sentiment analysis
                scores = analyzer.polarity_scores(news_text)
                
                # Format scores into sentences
                pos_score = scores["pos"]
                neu_score = scores["neu"]
                neg_score = scores["neg"]
                
                sentiment_info = (
                    f"The positive score for this news is {pos_score}. "
                    f"The neutral score for this news is {neu_score}. "
                    f"The negative score for this news is {neg_score}."
                )
                
                # Combine original news with sentiment information
                analyzed_news.append(f"{news_text} {sentiment_info}")
                
            except Exception as e:
                print(f"Error analyzing news: {str(e)[:100]}")
                # Keep original if analysis fails
                analyzed_news.append(news_text)
        
        # Update the news items with sentiment analysis
        new_dict[i_date]["news"][cur_symbol] = analyzed_news



def export_sub_symbol(cur_symbol_lst, senti_model_type):
    """
    Process and export sentiment analysis for given symbols.
    
    Args:
        cur_symbol_lst: List of symbols to process
        senti_model_type: Type of sentiment model to use ('FinBERT' or 'Vader')
    """
    print(f'Processing {len(cur_symbol_lst)} tickers: {cur_symbol_lst}')
    
    for cur_symbol in cur_symbol_lst:
        print(f"\n{'='*50}")
        print(f"Processing {cur_symbol} with {senti_model_type}")
        print(f"{'='*50}")
        
        try:
            # Extract data for the symbol
            new_dict, ticker_dict_byDate = subset_symbol_dict(input_file, cur_symbol)
            
            # Check if we have any data
            if not new_dict:
                print(f"No data found for {cur_symbol}. Skipping...")
                continue
                
            # Count news items
            news_count = sum(len(new_dict[date]['news'].get(cur_symbol, [])) for date in new_dict)
            print(f"Found {news_count} news items for {cur_symbol} across {len(new_dict)} dates")
            
            # Find a sample date with news
            sample_dates = [date for date in new_dict if date in new_dict and 
                          cur_symbol in new_dict[date]['news'] and 
                          len(new_dict[date]['news'][cur_symbol]) > 0]
            
            sample_date = sample_dates[0] if sample_dates else None
            
            # Apply sentiment analysis
            if senti_model_type == 'FinBERT':
                print(f"Applying FinBERT sentiment analysis to {news_count} news items...")
                assign_finBERT_scores(new_dict, cur_symbol)
                
                # Show sample results
                if sample_date:
                    print(f"\n----- Sample FinBERT Analysis for {cur_symbol} on {sample_date} -----")
                    if cur_symbol in new_dict[sample_date]['news'] and new_dict[sample_date]['news'][cur_symbol]:
                        news_text = new_dict[sample_date]['news'][cur_symbol][0]
                        # Split by score indicators to display separately
                        parts = news_text.split("The positive score for this news is")
                        original_news = parts[0].strip()
                        scores_part = "The positive score for this news is" + parts[1]
                        
                        print(f"Original news: {original_news}")
                        print(f"Sentiment scores: {scores_part}")
                    else:
                        print("No news found for sample date")
            else: 
                print(f"Applying VADER sentiment analysis to {news_count} news items...")
                assign_vader_scores(new_dict, cur_symbol)
                
                # Show sample results
                if sample_date:
                    print(f"\n----- Sample VADER Analysis for {cur_symbol} on {sample_date} -----")
                    if cur_symbol in new_dict[sample_date]['news'] and new_dict[sample_date]['news'][cur_symbol]:
                        news_text = new_dict[sample_date]['news'][cur_symbol][0]
                        # Split by score indicators to display separately
                        parts = news_text.split("The positive score for this news is")
                        original_news = parts[0].strip()
                        scores_part = "The positive score for this news is" + parts[1]
                        
                        print(f"Original news: {original_news}")
                        print(f"Sentiment scores: {scores_part}")
                    else:
                        print("No news found for sample date")
        
            # Save to output directory with clear naming
            out_file = OUTPUT_DIR / f"sentiment_{cur_symbol}_{senti_model_type.lower()}.pkl"
            with open(out_file, "wb") as f:
                pickle.dump(new_dict, f)
            print(f'✓ Saved sentiment analysis for {cur_symbol} using {senti_model_type} to {out_file}')
            
        except Exception as e:
            print(f"Error processing {cur_symbol}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    
cur_symbol_lst = ['FPT']  
input_file = INPUT_DIR / "env_data.pkl"

def main():
    # Check if input file exists
    if not input_file.exists():
        print(f"Error: Input file {input_file} not found")
        print(f"Make sure to run data_pipeline.py first to generate env_data.pkl")
        exit(1)
        
    print(f"\n{'='*60}")
    print(f"SENTIMENT ANALYSIS FOR VIETNAMESE STOCKS")
    print(f"{'='*60}")
    print(f"Input file: {input_file}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Symbols to process: {cur_symbol_lst}")
    
    # Run sentiment analysis with FinBERT
    print("\nRunning FinBERT sentiment analysis...")
    export_sub_symbol(cur_symbol_lst, senti_model_type='FinBERT')
    
    print("\nSentiment analysis complete!")
    print(f"Results saved to: {OUTPUT_DIR}")

def view_sample_sentiment(symbol='FPT', model_type='finbert'):
    """
    View a sample of the sentiment analysis data to check structure and scores.
    
    Args:
        symbol: Symbol to view data for (default: 'FPT')
        model_type: Type of sentiment model used ('finbert' or 'vader')
    """
    # Construct the file path
    file_path = OUTPUT_DIR / f"sentiment_{symbol}_{model_type}.pkl"
    
    if not file_path.exists():
        print(f"Error: File {file_path} not found")
        print(f"Run sentiment analysis for {symbol} with {model_type} first")
        return
    
    print(f"\n{'='*60}")
    print(f"SAMPLE SENTIMENT ANALYSIS DATA FOR {symbol}")
    print(f"{'='*60}")
    print(f"File: {file_path}")
    
    # Load the data
    with open(file_path, "rb") as f:
        sentiment_data = pickle.load(f)
    
    # Find a date with news
    dates_with_news = [date for date in sentiment_data 
                      if symbol in sentiment_data[date]['news'] 
                      and sentiment_data[date]['news'][symbol]]
    
    if not dates_with_news:
        print(f"No news found for {symbol}")
        return
    
    # Get the first date with news
    sample_date = dates_with_news[0]
    print(f"\nSample date: {sample_date}")
    
    # Show overall structure
    print("\nData structure:")
    for key in sentiment_data[sample_date]:
        if key == 'news':
            news_count = len(sentiment_data[sample_date]['news'].get(symbol, []))
            print(f"  - news: {news_count} items")
        else:
            print(f"  - {key}: {'Present' if symbol in sentiment_data[sample_date][key] else 'Not present'}")
    
    # Show a sample news item with sentiment
    if symbol in sentiment_data[sample_date]['news'] and sentiment_data[sample_date]['news'][symbol]:
        print("\nSample news item with sentiment scores:")
        news_item = sentiment_data[sample_date]['news'][symbol][0]
        
        # Try to separate the original news from sentiment scores
        if "The positive score for this news is" in news_item:
            parts = news_item.split("The positive score for this news is")
            original_news = parts[0].strip()
            scores = "The positive score for this news is" + parts[1]
            
            print(f"\nOriginal news:\n{original_news[:500]}..." if len(original_news) > 500 else original_news)
            print(f"\nSentiment scores:\n{scores}")
        else:
            print(f"\nNews with sentiment:\n{news_item[:500]}..." if len(news_item) > 500 else news_item)


if __name__ == "__main__":
    #main()
        
    # Uncomment to view a sample of the sentiment analysis data
    view_sample_sentiment(symbol='FPT', model_type='finbert')

# Uncomment to also run VADER sentiment analysis
# print("\nRunning VADER sentiment analysis...")
# export_sub_symbol(cur_symbol_lst, senti_model_type='Vader')
