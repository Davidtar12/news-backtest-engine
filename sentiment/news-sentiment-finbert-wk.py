import os
import requests
from dotenv import load_dotenv
from datetime import datetime, time as _time, timedelta
import pytz
import re

load_dotenv("c:/Users/david/OneDrive/Documents/DS - Coding - Python/Stocks/Alpaca/alpkey.env")
API_KEY = os.getenv("APCA_API_KEY_ID")
API_SECRET = os.getenv("APCA_API_SECRET_KEY")

url = "https://data.alpaca.markets/v1beta1/news"
headers = {"APCA-API-KEY-ID": API_KEY, "APCA-API-SECRET-KEY": API_SECRET}

# CONFIG
START_HOUR = 8   # start hour in NY time
END_HOUR = 15    # end hour in NY time
SYMBOLS = ["AAPL"]    # server-side filter as array<string>, or [] to omit
FILTER_SYMBOL = "AAPL"  # client-side filter (None to skip)
STRICT_SYMBOL_ONLY = False  # True: accept articles only when symbols == [FILTER_SYMBOL]
MODEL_NAME = "ProsusAI/finbert"  # prefer FinBERT

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


# collect articles for yesterday only
collected = {}
day = yesterday_ny
params, day_start_ny, day_end_ny = make_params_for_day(day)
r = requests.get(url, headers=headers, params=params, timeout=15)
r.raise_for_status()

data = r.json()
day_articles = data.get("news", []) if isinstance(data, dict) else (data if isinstance(data, list) else [])

for a in day_articles:
    aid = a.get("id")
    if not aid:
        continue
    collected[aid] = (a, day_start_ny, day_end_ny)

# filter by time window and symbol
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
    if not (day_start_ny <= ny_dt < day_end_ny):
        continue
    if FILTER_SYMBOL and not matches_symbol_strict(a, FILTER_SYMBOL):
        continue
    results.append((ny_dt, a))

if not results:
    sym_txt = FILTER_SYMBOL or "any"
    print(f"No articles mentioning {sym_txt} for yesterday in the configured window.")
    exit(0)

results.sort(key=lambda x: x[0], reverse=True)

# --- sentiment analyzer selection (no debug prints) ---
analyzer_name = None


def get_sentiment(texts):
    return [{"positive": 0.0, "negative": 0.0, "neutral": 1.0} for _ in texts]


# 1) try local sentiment.py
try:
    from sentiment import FinBertSentiment  # optional local helper
    fb = FinBertSentiment()
    analyzer_name = "FinBert (local)"

    def get_sentiment(texts):
        return fb.analyze_texts(texts)
except Exception:
    # 2) try transformers pipeline with FinBERT preference
    try:
        from transformers import pipeline
        import torch as _torch
        device = 0 if _torch.cuda.is_available() else -1
        try:
            hf_pipe = pipeline("sentiment-analysis", model=MODEL_NAME, device=device)
            analyzer_name = f"HuggingFace pipeline ({MODEL_NAME})"
        except Exception:
            hf_pipe = pipeline("sentiment-analysis", device=device)
            analyzer_name = "HuggingFace pipeline (default)"

        def get_sentiment(texts):
            out = []
            for t in texts:
                try:
                    r = hf_pipe(t[:512], truncation=True)
                except Exception:
                    out.append({"positive": 0.0, "negative": 0.0, "neutral": 1.0})
                    continue
                lbl = r[0]["label"].lower()
                score = float(r[0]["score"])
                if lbl.startswith("pos"):
                    out.append({"positive": score, "negative": 1 - score, "neutral": 0.0})
                elif lbl.startswith("neg"):
                    out.append({"positive": 1 - score, "negative": score, "neutral": 0.0})
                else:
                    out.append({"positive": 0.0, "negative": 0.0, "neutral": score})
            return out

    except Exception:
        # 3) fallback: keyword-based sentiment
        POS_WORDS = {"good", "great", "positive", "gain", "up", "beat", "raise", "soar", "improve", "strong"}
        NEG_WORDS = {"bad", "weak", "negative", "drop", "down", "miss", "cut", "fall", "plunge", "slump"}

        def get_sentiment(texts):
            out = []
            for t in texts:
                txt = (t or "").lower()
                pos = sum(len(re.findall(rf'\b{re.escape(w)}\b', txt)) for w in POS_WORDS)
                neg = sum(len(re.findall(rf'\b{re.escape(w)}\b', txt)) for w in NEG_WORDS)
                total = pos + neg
                if total == 0:
                    out.append({"positive": 0.0, "negative": 0.0, "neutral": 1.0})
                else:
                    pos_score = pos / total
                    neg_score = neg / total
                    out.append({"positive": pos_score, "negative": neg_score, "neutral": max(0.0, 1.0 - (pos_score + neg_score))})
            return out

# prepare texts (headline + summary + content) and run sentiment in batch
def article_text(a):
    parts = []
    for f in ("headline", "summary", "content"):
        v = a.get(f)
        if v:
            parts.append(v)
    return " — ".join(parts).strip()


texts = [article_text(a) for _, a in results]
sent_scores = get_sentiment(texts)

# print with sentiment label and score
for i, ((ny_dt, a), sc) in enumerate(zip(results, sent_scores), 1):
    try:
        top_label, top_score = max(sc.items(), key=lambda x: x[1])
    except Exception:
        top_label, top_score = "neutral", 0.0
    print(f"{i}. {ny_dt.strftime('%Y-%m-%d %H:%M:%S %Z')} | {top_label.upper()} {top_score:.3f} | {a.get('headline')} | symbols: {a.get('symbols')}")