import os
import requests
from dotenv import load_dotenv
from datetime import datetime, time as _time
import pytz

load_dotenv("c:/Users/david/OneDrive/Documents/DS - Coding - Python/Stocks/Alpaca/alpkey.env")
API_KEY = os.getenv("APCA_API_KEY_ID")
API_SECRET = os.getenv("APCA_API_SECRET_KEY")

url = "https://data.alpaca.markets/v1beta1/news"
headers = {
    "APCA-API-KEY-ID": API_KEY,
    "APCA-API-SECRET-KEY": API_SECRET
}

# CONFIG: set NY time window and optional symbol(s)
START_HOUR = 8   # start hour in NY time
END_HOUR = 15    # end hour in NY time
SYMBOLS = ["EA"]    # server-side filter as array<string>, or [] to omit
FILTER_SYMBOL = "EA"  # client-side filter (None to skip)

ny_tz = pytz.timezone("America/New_York")
utc_tz = pytz.utc

now_ny = datetime.now(ny_tz)
today_ny = now_ny.date()

start_ny = ny_tz.localize(datetime.combine(today_ny, _time(START_HOUR, 0)))
end_ny = ny_tz.localize(datetime.combine(today_ny, _time(END_HOUR, 0)))

start_utc = start_ny.astimezone(utc_tz).isoformat().replace("+00:00", "Z")
end_utc = end_ny.astimezone(utc_tz).isoformat().replace("+00:00", "Z")

requested_limit = 100
MAX_LIMIT = 50
limit = min(requested_limit, MAX_LIMIT)

params = {
    "start": start_utc,
    "end": end_utc,
    "limit": limit,
}
if SYMBOLS:
    params["symbols"] = SYMBOLS  # sent as array-style params

response = requests.get(url, headers=headers, params=params, timeout=15)
response.raise_for_status()
news = response.json()

articles = news.get("news", []) if isinstance(news, dict) else (news if isinstance(news, list) else [])
if not articles:
    print("No news articles returned by the API for that time window.")
    exit(0)

# Optional client-side symbol filtering and time filtering, remove duplicates, sort by created_at desc
seen_ids = set()
clean = []
for a in articles:
    aid = a.get("id")
    if aid in seen_ids:
        continue
    seen_ids.add(aid)
    created = a.get("created_at")
    if not created:
        continue
    try:
        utc_dt = datetime.fromisoformat(created.replace("Z", "+00:00"))
    except Exception:
        continue
    ny_dt = utc_dt.astimezone(ny_tz)
    if not (start_ny <= ny_dt < end_ny):
        continue
    if FILTER_SYMBOL:
        symbols = {s.upper() for s in a.get("symbols", [])}
        if FILTER_SYMBOL.upper() not in symbols:
            continue
    clean.append((ny_dt, a))

if not clean:
    sym_txt = FILTER_SYMBOL or "any"
    print(f"No articles mentioning {sym_txt} between {START_HOUR}:00 and {END_HOUR}:00 NY time.")
    exit(0)

# sort newest first and print concise results
clean.sort(key=lambda x: x[0], reverse=True)
for i, (ny_dt, a) in enumerate(clean, 1):
    print(f"{i}. {ny_dt.strftime('%Y-%m-%d %H:%M:%S %Z')} - {a.get('headline')} - symbols: {a.get('symbols')}")