import os
import re
import requests
from dotenv import load_dotenv
from datetime import datetime
import pytz
import textwrap
import statistics

load_dotenv()
TIINGO_KEY = os.getenv("TIINGO_API_KEY")

# CONFIG
SYMBOLS = ["YAAS"]
FILTER_SYMBOL = "YAAS"
NY_TZ = pytz.timezone("America/New_York")
BASE_URL = "https://api.tiingo.com/tiingo/news"

def make_params(symbols, limit=50):
    p = {"tickers": ",".join(symbols)} if symbols else {}
    p["token"] = TIINGO_KEY
    p["limit"] = limit
    return p

def parse_published(dt_str):
    if not dt_str:
        return None
    try:
        if dt_str.endswith("Z"):
            return datetime.fromisoformat(dt_str.replace("Z", "+00:00")).astimezone(NY_TZ)
        return datetime.fromisoformat(dt_str).astimezone(NY_TZ)
    except Exception:
        # try common variants
        try:
            return datetime.strptime(dt_str, "%Y-%m-%dT%H:%M:%S").astimezone(NY_TZ)
        except Exception:
            return None

def fetch_tiingo(symbols):
    params = make_params(symbols)
    r = requests.get(BASE_URL, params=params, timeout=15)
    r.raise_for_status()
    return r.json() or []

def _short(s, mx=100):
    if not s:
        return ""
    t = re.sub(r"<[^>]+>", "", str(s))
    t = re.sub(r"\s+", " ", t).strip()
    return textwrap.shorten(t, width=mx, placeholder="…")

def _summarize_tickers(raw, max_items=3):
    if not raw:
        return ""
    if isinstance(raw, str):
        parts = [p.strip() for p in raw.split(",") if p.strip()]
    else:
        parts = []
        for t in raw:
            if isinstance(t, dict):
                parts.append(t.get("ticker") or t.get("symbol") or str(t))
            else:
                parts.append(str(t))
    shown = parts[:max_items]
    more = len(parts) - len(shown)
    return ",".join(shown) + (f"+{more}" if more > 0 else "")

def _article_sentiment(a):
    # Tiingo sometimes includes 'sentiment' or per-ticker sentiment in 'tickers'
    v = a.get("sentiment") or a.get("sentiment_score") or a.get("sentimentScore")
    if v is not None:
        try:
            return float(v)
        except Exception:
            pass
    scores = []
    for t in (a.get("tickers") or []):
        if isinstance(t, dict):
            for k in ("sentiment", "sentiment_score", "score"):
                if k in t and t[k] is not None:
                    try:
                        scores.append(float(t[k]))
                    except Exception:
                        pass
    return statistics.mean(scores) if scores else 0.0

def dedupe_by_id(items):
    seen = {}
    for pub, a in items:
        key = a.get("id") or a.get("uuid") or a.get("url") or (a.get("title") or "")[:120]
        prev = seen.get(key)
        if not prev or pub > prev[0]:
            seen[key] = (pub, a)
    return list(seen.values())

def matches_symbol_simple(article, sym):
    sym_up = sym.upper()
    for key in ("tickers", "symbols", "entities"):
        val = article.get(key)
        if isinstance(val, (list, tuple)):
            for x in val:
                if isinstance(x, dict):
                    if str(x.get("ticker") or x.get("symbol") or "").upper() == sym_up:
                        return True
                else:
                    if str(x).upper() == sym_up:
                        return True
        elif isinstance(val, str):
            if sym_up in [s.strip().upper() for s in val.split(",") if s.strip()]:
                return True
    pat = re.compile(rf'(?<!\w)\${re.escape(sym)}(?!\w)|(?<!\w){re.escape(sym)}(?!\w)', re.IGNORECASE)
    for f in ("title", "description", "summary", "url"):
        if pat.search(str(article.get(f) or "")):
            return True
    return False

def main():
    if not TIINGO_KEY:
        print("No Tiingo API key found in env (TIINGO_API_KEY).")
        return

    raw = fetch_tiingo(SYMBOLS)
    if not raw:
        print("No articles returned from Tiingo.")
        return

    parsed = []
    for a in raw:
        pub = parse_published(a.get("publishedDate") or a.get("published_at") or a.get("published"))
        if not pub:
            pub = parse_published(a.get("createdAt") or a.get("created_at"))
        if not pub:
            continue
        if FILTER_SYMBOL and not matches_symbol_simple(a, FILTER_SYMBOL):
            continue
        parsed.append((pub, a))

    if not parsed:
        print(f"No articles mentioning {FILTER_SYMBOL} in Tiingo results.")
        return

    parsed = dedupe_by_id(parsed)
    parsed.sort(key=lambda x: (x[0], _article_sentiment(x[1])), reverse=True)

    for i, (pub, a) in enumerate(parsed, 1):
        title = a.get("title") or a.get("headline") or a.get("description") or ""
        tickers_s = _summarize_tickers(a.get("tickers") or a.get("tickers") or a.get("symbols") or [], max_items=3)
        sent = _article_sentiment(a)
        print(f"{i}. {pub.strftime('%Y-%m-%d %H:%M:%S %Z')} | {_short(title, mx=120)}"
              + (f" | tickers: {tickers_s}" if tickers_s else "")
              + (f" | sentiment: {sent:.3f}" if sent is not None else ""))

if __name__ == "__main__":
    main()
# filepath: c:\Users\david\OneDrive\Documents\DS - Coding - Python\Stocks\Alpaca\news\tiingo_ticker.py
import os
import re
import requests
from dotenv import load_dotenv
from datetime import datetime
import pytz
import textwrap
import statistics

load_dotenv()
INGO_KEY = os.getenv("TIINGO_API_KEY") or os.getenv("TIINGO_KEY")

# CONFIG
SYMBOLS = ["YAAS"]
FILTER_SYMBOL = "YAAS"
NY_TZ = pytz.timezone("America/New_York")
BASE_URL = "https://api.tiingo.com/tiingo/news"

def make_params(symbols, limit=50):
    p = {"tickers": ",".join(symbols)} if symbols else {}
    p["token"] = TIINGO_KEY
    p["limit"] = limit
    return p

def parse_published(dt_str):
    if not dt_str:
        return None
    try:
        if dt_str.endswith("Z"):
            return datetime.fromisoformat(dt_str.replace("Z", "+00:00")).astimezone(NY_TZ)
        return datetime.fromisoformat(dt_str).astimezone(NY_TZ)
    except Exception:
        # try common variants
        try:
            return datetime.strptime(dt_str, "%Y-%m-%dT%H:%M:%S").astimezone(NY_TZ)
        except Exception:
            return None

def fetch_tiingo(symbols):
    params = make_params(symbols)
    r = requests.get(BASE_URL, params=params, timeout=15)
    r.raise_for_status()
    return r.json() or []

def _short(s, mx=100):
    if not s:
        return ""
    t = re.sub(r"<[^>]+>", "", str(s))
    t = re.sub(r"\s+", " ", t).strip()
    return textwrap.shorten(t, width=mx, placeholder="…")

def _summarize_tickers(raw, max_items=3):
    if not raw:
        return ""
    if isinstance(raw, str):
        parts = [p.strip() for p in raw.split(",") if p.strip()]
    else:
        parts = []
        for t in raw:
            if isinstance(t, dict):
                parts.append(t.get("ticker") or t.get("symbol") or str(t))
            else:
                parts.append(str(t))
    shown = parts[:max_items]
    more = len(parts) - len(shown)
    return ",".join(shown) + (f"+{more}" if more > 0 else "")

def _article_sentiment(a):
    # Tiingo sometimes includes 'sentiment' or per-ticker sentiment in 'tickers'
    v = a.get("sentiment") or a.get("sentiment_score") or a.get("sentimentScore")
    if v is not None:
        try:
            return float(v)
        except Exception:
            pass
    scores = []
    for t in (a.get("tickers") or []):
        if isinstance(t, dict):
            for k in ("sentiment", "sentiment_score", "score"):
                if k in t and t[k] is not None:
                    try:
                        scores.append(float(t[k]))
                    except Exception:
                        pass
    return statistics.mean(scores) if scores else 0.0

def dedupe_by_id(items):
    seen = {}
    for pub, a in items:
        key = a.get("id") or a.get("uuid") or a.get("url") or (a.get("title") or "")[:120]
        prev = seen.get(key)
        if not prev or pub > prev[0]:
            seen[key] = (pub, a)
    return list(seen.values())

def matches_symbol_simple(article, sym):
    sym_up = sym.upper()
    for key in ("tickers", "symbols", "entities"):
        val = article.get(key)
        if isinstance(val, (list, tuple)):
            for x in val:
                if isinstance(x, dict):
                    if str(x.get("ticker") or x.get("symbol") or "").upper() == sym_up:
                        return True
                else:
                    if str(x).upper() == sym_up:
                        return True
        elif isinstance(val, str):
            if sym_up in [s.strip().upper() for s in val.split(",") if s.strip()]:
                return True
    pat = re.compile(rf'(?<!\w)\${re.escape(sym)}(?!\w)|(?<!\w){re.escape(sym)}(?!\w)', re.IGNORECASE)
    for f in ("title", "description", "summary", "url"):
        if pat.search(str(article.get(f) or "")):
            return True
    return False

def main():
    if not TIINGO_KEY:
        print("No Tiingo API key found in env (TIINGO_API_KEY).")
        return

    raw = fetch_tiingo(SYMBOLS)
    if not raw:
        print("No articles returned from Tiingo.")
        return

    parsed = []
    for a in raw:
        pub = parse_published(a.get("publishedDate") or a.get("published_at") or a.get("published"))
        if not pub:
            pub = parse_published(a.get("createdAt") or a.get("created_at"))
        if not pub:
            continue
        if FILTER_SYMBOL and not matches_symbol_simple(a, FILTER_SYMBOL):
            continue
        parsed.append((pub, a))

    if not parsed:
        print(f"No articles mentioning {FILTER_SYMBOL} in Tiingo results.")
        return

    parsed = dedupe_by_id(parsed)
    parsed.sort(key=lambda x: (x[0], _article_sentiment(x[1])), reverse=True)

    for i, (pub, a) in enumerate(parsed, 1):
        title = a.get("title") or a.get("headline") or a.get("description") or ""
        tickers_s = _summarize_tickers(a.get("tickers") or a.get("tickers") or a.get("symbols") or [], max_items=3)
        sent = _article_sentiment(a)
        print(f"{i}. {pub.strftime('%Y-%m-%d %H:%M:%S %Z')} | {_short(title, mx=120)}"
              + (f" | tickers: {tickers_s}" if tickers_s else "")
              + (f" | sentiment: {sent:.3f}" if sent is not None else ""))

if __name__ == "__main__":
    main()