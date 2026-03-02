import os
import requests
from dotenv import load_dotenv
from datetime import datetime, time
import pytz

load_dotenv()
API_KEY = os.getenv("APCA_API_KEY_ID")
API_SECRET = os.getenv("APCA_API_SECRET_KEY")

url = "https://data.alpaca.markets/v1beta1/news"
headers = {
    "APCA-API-KEY-ID": API_KEY,
    "APCA-API-SECRET-KEY": API_SECRET
}
params = {
    "limit": 100  # Increase to ensure you get all articles in the window
}

try:
    response = requests.get(url, headers=headers, params=params)
    response.raise_for_status()
    news = response.json()
except Exception as e:
    print("ERROR: Exception occurred during API request.")
    print(e)
    exit(1)

ny_tz = pytz.timezone("America/New_York")
today_ny = datetime.now(ny_tz).date()
start_time = ny_tz.localize(datetime.combine(today_ny, time(13, 0)))
end_time = ny_tz.localize(datetime.combine(today_ny, time(14, 0)))

found = False
print("Articles published between 1 pm and 2 pm NY time:")
for i, article in enumerate(news["news"], 1):
    utc_dt = datetime.fromisoformat(article['created_at'].replace("Z", "+00:00"))
    ny_dt = utc_dt.astimezone(ny_tz)
    if start_time <= ny_dt < end_time:
        found = True
        print(f"{i}. {ny_dt.strftime('%Y-%m-%d %H:%M:%S %Z')} - {article['headline']}")

if not found:
    print("No news articles found in that time window.")