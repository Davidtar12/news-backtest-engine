import os
import requests
from dotenv import load_dotenv
from datetime import datetime, time as _time, timedelta
import pytz
import time

load_dotenv()
POLYGON_KEY = os.getenv("POLYGON_API_KEY")

# CONFIG
START_HOUR = 8
END_HOUR = 15
SYMBOLS = ["EA"]           # not all Polygon endpoints accept many tickers; we'll use FILTER_SYMBOL for single-ticker requests
FILTER_SYMBOL = "EA"
STRICT_SYMBOL_ONLY = False
NY_TZ = pytz.timezone("America/New_York")
UTC = pytz.utc

BASE_URL = "https://api.polygon.io/v2/reference/news"
# reduce page size to avoid hitting limits
REQUEST_LIMIT = 10  # was 100

def make_params_for_day(day_date, ticker=None, limit=100):
    start_ny = NY_TZ.localize(datetime.combine(day_date, _time(START_HOUR, 0)))
    end_ny = NY_TZ.localize(datetime.combine(day_date, _time(END_HOUR, 0)))
    # Polygon expects published_utc in RFC3339 (UTC)
    start_utc = start_ny.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    end_utc = end_ny.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    params = {
        "limit": limit,
        "sort": "published_utc",
        "order": "desc",
        "published_utc.gte": start_utc,
        "published_utc.lte": end_utc,
        "apiKey": POLYGON_KEY,
    }
    if ticker:
        params["ticker"] = ticker
    return params, start_ny, end_ny

def parse_published_utc(s):
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(NY_TZ)
    except Exception:
        return None

def fetch_page(url, params=None, max_retries=5):
    backoff = 1
    for attempt in range(max_retries):
        try:
            # if params is None, requests will use the full next_url as given
            r = requests.get(url, params=params, timeout=15)
            if r.status_code == 429:
                # Respect Retry-After header when provided
                ra = r.headers.get("Retry-After")
                wait = int(ra) if ra and ra.isdigit() else backoff
                time.sleep(wait)
                backoff = min(backoff * 2, 60)
                continue
            r.raise_for_status()
            return r.json()
        except requests.exceptions.RequestException as e:
            # on transient network errors or 5xx, backoff and retry
            if attempt + 1 == max_retries:
                raise
            time.sleep(backoff)
            backoff = min(backoff * 2, 60)
    raise RuntimeError("Max retries exceeded for fetch_page")

def main():
    if not POLYGON_KEY:
        print("No Polygon API key found in env (POLYGON_API_KEY).")
        return

    today_ny = datetime.now(NY_TZ).date()
    yesterday_ny = today_ny - timedelta(days=1)

    collected = []

    for day in (yesterday_ny, today_ny):
        params, day_start_ny, day_end_ny = make_params_for_day(day, ticker=FILTER_SYMBOL, limit=REQUEST_LIMIT)
        payload = fetch_page(BASE_URL, params)
        results = payload.get("results", []) if isinstance(payload, dict) else []
        # follow next_url pages if present (stop if too many pages)
        next_url = payload.get("next_url")
        pages = 0
        while next_url and pages < 5:  # safe guard
            try:
                more = fetch_page(next_url, {})
                more_results = more.get("results", []) if isinstance(more, dict) else []
                results.extend(more_results)
                next_url = more.get("next_url")
                pages += 1
            except Exception:
                break

        for a in results:
            pub_dt = parse_published_utc(a.get("published_utc") or a.get("publishedAt") or a.get("published"))
            if not pub_dt:
                continue
            # ensure it falls within that day's NY window
            if not (day_start_ny <= pub_dt < day_end_ny):
                continue
            # basic symbol check (Polygon returns tickers array)
            if FILTER_SYMBOL:
                tickers = a.get("tickers") or []
                if STRICT_SYMBOL_ONLY:
                    if not (isinstance(tickers, list) and tickers == [FILTER_SYMBOL]):
                        continue
                else:
                    if FILTER_SYMBOL not in tickers and f"${FILTER_SYMBOL}" not in (a.get("title") or ""):
                        # fallback to title/content scan
                        if FILTER_SYMBOL not in (a.get("title") or "") and FILTER_SYMBOL not in (a.get("description") or ""):
                            continue
            collected.append((pub_dt, a))

    if not collected:
        print(f"No articles mentioning {FILTER_SYMBOL} for yesterday and today in the configured windows.")
        return

    # dedupe by id and keep newest
    seen = {}
    for pub, a in collected:
        key = a.get("id") or a.get("url") or (a.get("title") or "")[:120]
        prev = seen.get(key)
        if not prev or pub > prev[0]:
            seen[key] = (pub, a)
    deduped = list(seen.values())

    deduped.sort(key=lambda x: x[0], reverse=True)

    for i, (pub, a) in enumerate(deduped, 1):
        title = a.get("title") or a.get("headline") or a.get("description") or ""
        tickers = a.get("tickers") or []
        sentiment = None
        # polygon insights may include sentiment in 'insights'
        insights = a.get("insights") or []
        if isinstance(insights, list) and insights:
            try:
                s = insights[0].get("sentiment")
                sentiment = s
            except Exception:
                sentiment = None
        pub_s = pub.strftime("%Y-%m-%d %H:%M:%S %Z")
        tickers_s = ",".join(tickers) if tickers else ""
        line = f"{i}. {pub_s} | {title}"
        if tickers_s:
            line += f" | tickers: {tickers_s}"
        if sentiment:
            line += f" | sentiment: {sentiment}"
        print(line)

if __name__ == "__main__":
    main()