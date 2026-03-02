import os
import re
import requests
from dotenv import load_dotenv
from datetime import datetime
import pytz

load_dotenv()
MARKETAUX_API_KEY = os.getenv("MARKETAUX_API_KEY")

# CONFIG
SYMBOLS = ["YAAS"]           # server-side symbol(s)
FILTER_SYMBOL = "YAAS"       # client-side exact ticker filter (None to skip)
NY_TZ = pytz.timezone("America/New_York")

BASE_URL = "https://api.marketaux.com/v1/news/all"

def make_params(symbols, limit=50, language="en", filter_entities=True):
    params = {
        "api_token": MARKETAUX_API_KEY,
        "symbols": ",".join(symbols) if symbols else None,
        "limit": limit,
        "language": language,
        "filter_entities": "true" if filter_entities else "false",
        "sort": "published_at",
        "order": "desc",
    }
    # remove None values
    return {k: v for k, v in params.items() if v is not None}

def matches_symbol(article: dict, sym: str) -> bool:
    if not sym:
        return True
    sym_up = sym.upper()
    # 1) check explicit tickers / tickers-like fields
    for key in ("tickers", "symbols", "entities"):
        val = article.get(key)
        if isinstance(val, (list, tuple)):
            if any((str(x).upper() == sym_up) for x in val if x):
                return True
        elif isinstance(val, str):
            if sym_up in [s.strip().upper() for s in val.split(",") if s.strip()]:
                return True
    # 2) fallback: search title/description/url for $TICKER or word-boundary ticker
    pattern = re.compile(rf'(?<!\w)\${re.escape(sym)}(?!\w)|(?<!\w){re.escape(sym)}(?!\w)', re.IGNORECASE)
    for field in ("title", "headline", "description", "summary", "content", "url"):
        if pattern.search(str(article.get(field) or "")):
            return True
    return False

def parse_published(dt_str: str):
    if not dt_str:
        return None
    try:
        # MarketAux returns ISO timestamps; ensure tz-aware
        if dt_str.endswith("Z"):
            return datetime.fromisoformat(dt_str.replace("Z", "+00:00")).astimezone(NY_TZ)
        return datetime.fromisoformat(dt_str).astimezone(NY_TZ)
    except Exception:
        return None

def fetch_marketaux(symbols):
    params = make_params(symbols)
    resp = requests.get(BASE_URL, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    # MarketAux: primary list is usually in 'data'
    return data.get("data") or data.get("news") or (data if isinstance(data, list) else [])

def main():
    if not MARKETAUX_API_KEY:
        print("No MarketAux API key found in env (MARKETAUX_API_KEY).")
        return

    articles = fetch_marketaux(SYMBOLS)
    if not articles:
        print("No articles returned from MarketAux.")
        return

    cleaned = []
    for a in articles:
        pub = parse_published(a.get("published_at") or a.get("publishedAt") or a.get("published"))
        if not pub:
            # try other common fields
            pub = parse_published(a.get("created_at") or a.get("createdAt"))
        if not pub:
            continue
        if FILTER_SYMBOL and not matches_symbol(a, FILTER_SYMBOL):
            continue
        cleaned.append((pub, a))

    if not cleaned:
        print(f"No articles mentioning {FILTER_SYMBOL} in MarketAux results.")
        return

    # sort newest first and print
    # ---- replace the previous single-line printing block with this ----
    # choose source list (parsed/cleaned)
    items = cleaned if "cleaned" in locals() else []

    def _short_text(s, max_chars=100):
        if not s:
            return ""
        t = re.sub(r"<[^>]+>", "", str(s))
        t = re.sub(r"\s+", " ", t).strip()
        return (t[: max_chars - 1] + "…") if len(t) > max_chars else t

    def _summarize_tickers(raw, max_items=3):
        if not raw:
            return ""
        items = []
        if isinstance(raw, str):
            parts = [p.strip() for p in raw.split(",") if p.strip()]
            items = parts
        else:
            for t in raw:
                if isinstance(t, dict):
                    items.append(t.get("symbol") or t.get("ticker") or str(t))
                else:
                    items.append(str(t))
        shown = items[:max_items]
        more = len(items) - len(shown)
        return ",".join(shown) + (f"+{more}" if more > 0 else "")

    def _article_sentiment(a):
        # try top-level fields first, then ticker highlights
        for key in ("sentiment_score", "sentiment", "sentimentScore"):
            v = a.get(key)
            if v is not None:
                try:
                    return float(v)
                except Exception:
                    pass
        # check tickers/entities for sentiment
        for key in ("tickers", "symbols", "entities", "highlights"):
            for t in (a.get(key) or []):
                if isinstance(t, dict):
                    for k in ("sentiment_score", "sentiment", "score"):
                        v = t.get(k)
                        if v is not None:
                            try:
                                return float(v)
                            except Exception:
                                pass
        return 0.0

    # dedupe & sort if not already done
    if items and isinstance(items[0], tuple) and (len(items[0]) == 2):
        deduped = dedupe_by_id(items) if "dedupe_by_id" in globals() else items
        deduped.sort(key=lambda x: (x[0], _article_sentiment(x[1])), reverse=True)
    else:
        deduped = items

    # print one compact line per article (keeps all results)
    for i, entry in enumerate(deduped, 1):
        pub, a = entry if isinstance(entry, tuple) and len(entry) == 2 else (None, entry)
        ts = pub.strftime("%Y-%m-%d %H:%M:%S %Z") if pub else ""
        title = a.get("title") or a.get("headline") or a.get("summary") or ""
        title_s = _short_text(title, max_chars=100)
        tickers_s = _summarize_tickers(a.get("tickers") or a.get("symbols") or a.get("entities") or [], max_items=3)
        sent = _article_sentiment(a)
        sent_s = f"{sent:.3f}" if sent is not None else ""
        line = f"{i}. {ts} | {title_s}"
        if tickers_s:
            line += f" | tickers: {tickers_s}"
        if sent_s:
            line += f" | sentiment: {sent_s}"
        print(line)

if __name__ == "__main__":
    main()
# filepath: c:\Users\david\OneDrive\Documents\DS - Coding - Python\Stocks\Alpaca\news\marketaux_symbol.py


