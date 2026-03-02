import os
import argparse
from datetime import datetime

import requests
from dotenv import load_dotenv

# Load API key from alpkey.env
load_dotenv(dotenv_path='alpkey.env')
API_KEY = os.getenv('STOCKNEWS_API_KEY')

def ymd_to_mmddyyyy(date_str):
    # Convert 'YYYY-MM-DD' to 'MMDDYYYY'
    dt = datetime.strptime(date_str, '%Y-%m-%d')
    return dt.strftime('%m%d%Y')

def fetch_stock_news(ticker, date, items=3, start_time=None, end_time=None):
    mmddyyyy = ymd_to_mmddyyyy(date)
    date_param = f'{mmddyyyy}-{mmddyyyy}'
    url = f'https://stocknewsapi.com/api/v1?tickers={ticker}&items={items}&date={date_param}'
    if start_time and end_time:
        # Ensure time is in HHMMSS format
        url += f'&time={start_time}-{end_time}'
    url += f'&token={API_KEY}'
    print(f"Requesting URL: {url}")  # Debug: show the request URL
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        articles = data.get('data', [])
        if not articles:
            print(f"No news found for {ticker} on {date} between {start_time} and {end_time}.")
            print(f"Full API response: {data}")
        else:
            for article in articles:
                print(f"{article['date']} - {article['title']}")
    else:
        print(f"Error: {response.status_code} - {response.text}")

def parse_args():
    parser = argparse.ArgumentParser(description="Fetch StockNewsAPI articles for a specific ticker/date")
    parser.add_argument('--ticker', required=True, help='Ticker symbol (e.g. AMZN)')
    parser.add_argument('--date', required=True, help='Date in YYYY-MM-DD')
    parser.add_argument('--items', type=int, default=100, help='Number of articles to fetch (default: 3)')
    parser.add_argument('--start-time', help='Optional start time (HHMMSS, e.g. 130000)')
    parser.add_argument('--end-time', help='Optional end time (HHMMSS, e.g. 150000)')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    fetch_stock_news(
        ticker=args.ticker,
        date=args.date,
        items=args.items,
        start_time=args.start_time,
        end_time=args.end_time,
    )
