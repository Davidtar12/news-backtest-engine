# Simple script: fetch recent articles and filter by symbol (no time filtering)
import os
import requests
from dotenv import load_dotenv
from datetime import datetime, time as _time, timedelta
import pytz

load_dotenv("c:/Users/david/OneDrive/Documents/DS - Coding - Python/Stocks/Alpaca/alpkey.env")
API_KEY = os.getenv("APCA_API_KEY_ID")
API_SECRET = os.getenv("APCA_API_SECRET_KEY")

url = "https://data.alpaca.markets/v1beta1/news"
headers = {"APCA-API-KEY-ID": API_KEY, "APCA-API-SECRET-KEY": API_SECRET}

# CONFIG
START_HOUR = 8   # start hour in NY time
END_HOUR = 15    # end hour in NY time
SYMBOLS = ["YAAS"]    # server-side filter as array<string>, or [] to omit
FILTER_SYMBOL = "YAAS"  # client-side filter (None to skip)
STRICT_SYMBOL_ONLY = False  # True: accept articles only when symbols == [FILTER_SYMBOL]

ny_tz = pytz.timezone("America/New_York")
utc_tz = pytz.utc

today_ny = datetime.now(ny_tz).date()
yesterday_ny = today_ny - timedelta(days=1)

requested_limit = 100
MAX_LIMIT = 50
limit = min(requested_limit, MAX_LIMIT)

def make_params_for_day(day_date):
    start_ny = ny_tz.localize(datetime.combine(day_date, _time(START_HOUR, 0)))
    end_ny = ny_tz.localize(datetime.combine(day_date, _time(END_HOUR, 0)))
    start_utc = start_ny.astimezone(utc_tz).isoformat().replace("+00:00", "Z")
    end_utc = end_ny.astimezone(utc_tz).isoformat().replace("+00:00", "Z")
    p = {"start": start_utc, "end": end_utc, "limit": limit}
    if SYMBOLS:
        p["symbols"] = SYMBOLS
    return p, start_ny, end_ny

def matches_symbol_strict(a: dict, sym: str) -> bool:
    if not sym:
        return True
    syms = [s.upper() for s in (a.get("symbols") or []) if s]
    if STRICT_SYMBOL_ONLY:
        return len(syms) == 1 and syms[0] == sym.upper()
    return sym.upper() in syms

# collect articles for yesterday and today
collected = {}
for day in (yesterday_ny, today_ny):
    params, day_start_ny, day_end_ny = make_params_for_day(day)
    r = requests.get(url, headers=headers, params=params, timeout=15)
    r.raise_for_status()
    data = r.json()
    day_articles = data.get("news", []) if isinstance(data, dict) else (data if isinstance(data, list) else [])
    for a in day_articles:
        aid = a.get("id")
        if not aid:
            continue
        # dedupe by id, keep latest (overwrite if newer)
        existing = collected.get(aid)
        if existing is None:
            collected[aid] = (a, day_start_ny, day_end_ny)
        else:
            collected[aid] = (a, day_start_ny, day_end_ny)

# filter by time window and symbol, then sort
results = []
for a, day_start_ny, day_end_ny in collected.values():
    created = a.get("created_at")
    if not created:
        continue
    try:
        utc_dt = datetime.fromisoformat(created.replace("Z", "+00:00"))
    except Exception:
        continue
    ny_dt = utc_dt.astimezone(ny_tz)
    # ensure it falls into the day-specific window
    if not (day_start_ny <= ny_dt < day_end_ny):
        continue
    if FILTER_SYMBOL and not matches_symbol_strict(a, FILTER_SYMBOL):
        continue
    results.append((ny_dt, a))

if not results:
    sym_txt = FILTER_SYMBOL or "any"
    print(f"No articles mentioning {sym_txt} for yesterday and today in the configured windows.")
    exit(0)

results.sort(key=lambda x: x[0], reverse=True)
for i, (ny_dt, a) in enumerate(results, 1):
    print(f"{i}. {ny_dt.strftime('%Y-%m-%d %H:%M:%S %Z')} - {a.get('headline')} - symbols: {a.get('symbols')}")