#dependencies
import requests
import json
import polars as pl
from pathlib import Path
from bs4 import BeautifulSoup
from datetime import datetime, timedelta

#const
REPO_DIR = Path(__file__).parent.parent
DATA_DIR = REPO_DIR / "data"
CUR_DIR = REPO_DIR / "data-pipeline"
SOURCE = "https://cafef.vn"

def str_to_date(date_str):
    return datetime.strptime(date_str, "%d/%m/%Y %H:%M")

def round_to_trade_date(cur_date):
    """
    Round news datetime to Vietnam stock market trading hours (HNX).

    Args:
        cur_date (datetime): Original news datetime

    Returns:
        datetime: Rounded datetime to trading hours
    """
    # Define trading hours
    morning_start = cur_date.replace(hour=9, minute=0, second=0, microsecond=0)
    morning_end = cur_date.replace(hour=11, minute=30, second=0, microsecond=0)
    afternoon_start = cur_date.replace(hour=13, minute=0, second=0, microsecond=0)
    afternoon_end = cur_date.replace(hour=14, minute=30, second=0, microsecond=0)
    
    # Check if it's weekend (Saturday=5, Sunday=6)
    if cur_date.weekday() >= 5:  # Weekend
        # Round to next Monday 9:00 AM
        days_ahead = 7 - cur_date.weekday()  # Days until next Monday
        next_monday = cur_date + timedelta(days=days_ahead)
        return next_monday.replace(hour=9, minute=0, second=0, microsecond=0)
    
    # Weekday logic
    if cur_date.time() < morning_start.time():
        # Before 9:00 AM - round to 9:00 AM same day
        return morning_start
    elif morning_start.time() <= cur_date.time() <= morning_end.time():
        # During morning session - keep original time
        return cur_date
    elif morning_end.time() < cur_date.time() < afternoon_start.time():
        # During lunch break - round to 1:00 PM same day
        return afternoon_start
    elif afternoon_start.time() <= cur_date.time() <= afternoon_end.time():
        # During afternoon session - keep original time
        return cur_date
    else:
        # After 2:30 PM - round to 9:00 AM next trading day
        next_day = cur_date + timedelta(days=1)
        # If next day is weekend, go to Monday
        while next_day.weekday() >= 5:
            next_day += timedelta(days=1)
        return next_day.replace(hour=9, minute=0, second=0, microsecond=0) 

def scrap_page_content(link):
    try:
        response = requests.get(link)
        soup = BeautifulSoup(response.text, "html.parser")
        html_tag = soup.html
        #print(html_tag.attrs)
        if "xmlns" in html_tag.attrs:
            return None # Skip company disclosure news        
        news_item = soup.find("div", attrs= {"class":"left_cate totalcontentdetail"})
        if not news_item:
            print("Could not find news content")
        subheading = news_item.h2.text.strip()
        content = news_item.find("div", attrs = {"class":"detail-content afcbc-body"})
        full_text = " ".join([p.text.strip() for p in content.find_all("p")])
        complete_content = f"{subheading} {full_text}" if subheading else full_text
        return complete_content
    except Exception as e:
        print(f"Error scraping content from {link}: {e}")
        return None

def scrap_news(origin_url, params, START_DATE, END_DATE):
    """
    Scrape news articles from a given source within a specified date range.
    
    Args:
        origin_url (str): Base URL to scrape news from
        params (dict): Query parameters for the request
        START_DATE (datetime): Start date for news filtering
        END_DATE (datetime): End date for news filtering
    
    Returns:
        pl.DataFrame: DataFrame containing news titles, links, contents, and timestamps
    """

    titles = []
    links = []
    contents = []
    timestamps = []
    
    index = 1
    while True:
        params["PageIndex"] = index
        response = requests.get(url = origin_url, params = params) 
        
        if response.status_code == 200:
            print(f"START SCRAPING. SOURCE: {response.url}")
            #print(response.text)
            soup = BeautifulSoup(response.text, "html.parser")
            news_items = soup.find_all('li')
            if not news_items:
                print("Error: Blank page")
                break

            batch_start_date = str_to_date(news_items[-1].span.text)
            batch_end_date = str_to_date(news_items[0].span.text)

            if batch_start_date > END_DATE:
                print(f"TRUE. {batch_start_date} > {END_DATE}SKIP")
                index +=1
                continue
            if batch_end_date < START_DATE:
                print(f"TRUE. {batch_end_date} < {START_DATE}\nEND THE SEARCH")
                break
            for item in news_items:
                cur_date = str_to_date(item.span.text)
                if START_DATE <= cur_date <= END_DATE:
                    a_tag = item.a
                    title = a_tag.text 
                    single_link = SOURCE + a_tag['href']
                    content = scrap_page_content(single_link)
                    if content is None:
                        continue
                    
                    # Round to trading hours
                    trade_date = round_to_trade_date(cur_date)
                    # Append data
                    titles.append(title)
                    links.append(single_link)
                    contents.append(content)
                    timestamps.append(trade_date)  
            index += 1

        else:
            print(f"Failed to fetch data. Status code: {response.status_code}")
            break

    # Create Polars Dataframe
    df = pl.DataFrame({
        'title': titles,
        'link': links,
        "content": contents,
        "date": timestamps        
    })
    return df

def main():
    # Price data
    price_data = pl.read_parquet(DATA_DIR / "price.parquet")
    START_DATE = price_data["time"].min()
    END_DATE = price_data["time"].max()
    START_DATE = str_to_date("27/05/2025 10:20")
    END_DATE = str_to_date("30/05/2025 08:45")

    # Read source config
    with open("sources_config.json") as f:
        data_sources = json.load(f)
    cafef_source = data_sources[0]        
    url = cafef_source['url']
    source = cafef_source['source']    
    params = cafef_source['params']
    
    try:
        print(f"Start scraping from {source}")
        df = scrap_news(origin_url = url, params = params, START_DATE = START_DATE, END_DATE = END_DATE)
        df.write_parquet(DATA_DIR / "news.parquet")
        print("Saved news data successfully")
        print(f"Found {len(df)} news articles from {START_DATE} to {END_DATE}")
        print(f"Example:\n{df.head()}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    # Uncomment to test the rounding function
    # test_round_to_trade_date()
    main()