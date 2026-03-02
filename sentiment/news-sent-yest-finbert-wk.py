import os
import requests
from dotenv import load_dotenv
from datetime import datetime, time as _time, timedelta
import pytz

load_dotenv()
API_KEY = os.getenv("APCA_API_KEY_ID")
API_SECRET = os.getenv("APCA_API_SECRET_KEY")

url = "https://data.alpaca.markets/v1beta1/news"
headers = {"APCA-API-KEY-ID": API_KEY, "APCA-API-SECRET-KEY": API_SECRET}

# CONFIG
START_HOUR = 8   # start hour in NY time
END_HOUR = 15    # end hour in NY time
SYMBOLS = ["CRNX"]    # server-side filter as array<string>, or [] to omit
FILTER_SYMBOL = "CRNX"  # client-side filter (None to skip)
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
print(f"DEBUG: Requesting day={day} params={params}")
try:
    r = requests.get(url, headers=headers, params=params, timeout=15)
    print(f"DEBUG: HTTP {r.status_code} for day={day}")
    print(f"DEBUG: response preview: {r.text[:400]}")
    r.raise_for_status()
except Exception as e:
    print(f"ERROR: request failed for day={day}: {e}")
    raise

data = r.json()
day_articles = data.get("news", []) if isinstance(data, dict) else (data if isinstance(data, list) else [])
print(f"DEBUG: day={day} returned {len(day_articles)} articles, next_page_token={data.get('next_page_token') if isinstance(data, dict) else None}")

for a in day_articles:
    aid = a.get("id")
    created = a.get("created_at")
    syms = a.get("symbols")
    print(f"DEBUG: got article id={aid} created_at={created} symbols={syms}")
    if not aid:
        print("DEBUG:  - skipping article with no id")
        continue
    collected[aid] = (a, day_start_ny, day_end_ny)

print(f"DEBUG: total unique collected articles: {len(collected)}")

# filter by time window and symbol
results = []
for a, day_start_ny, day_end_ny in collected.values():
    created = a.get("created_at")
    if not created:
        print("DEBUG: skipping article with no created_at")
        continue
    try:
        utc_dt = datetime.fromisoformat(created.replace("Z", "+00:00"))
    except Exception as ex:
        print("DEBUG: datetime parse error for created_at:", created, ex)
        continue
    ny_dt = utc_dt.astimezone(ny_tz)
    if not (day_start_ny <= ny_dt < day_end_ny):
        print(f"DEBUG: article {a.get('id')} at {ny_dt} outside window {day_start_ny} - {day_end_ny}")
        continue
    if FILTER_SYMBOL and not matches_symbol_strict(a, FILTER_SYMBOL):
        print(f"DEBUG: article {a.get('id')} does not match FILTER_SYMBOL={FILTER_SYMBOL}; symbols={a.get('symbols')}")
        continue
    results.append((ny_dt, a))

print(f"DEBUG: matched results count: {len(results)}")
if not results:
    sym_txt = FILTER_SYMBOL or "any"
    print(f"No articles mentioning {sym_txt} for yesterday in the configured window.")
    exit(0)

results.sort(key=lambda x: x[0], reverse=True)

# --- sentiment analyzer selection with debugging ---
print("DEBUG: initializing sentiment analyzer...")
analyzer_name = None

def get_sentiment(texts):
    return [{"positive": 0.0, "negative": 0.0, "neutral": 1.0} for _ in texts]

# 1) try local sentiment.py
try:
    print("DEBUG: trying import from local sentiment.py")
    from sentiment import FinBertSentiment  # optional local helper
    fb = FinBertSentiment()
    analyzer_name = "FinBert (local)"
    def get_sentiment(texts):
        return fb.analyze_texts(texts)
except Exception as e_local:
    print("DEBUG: local sentiment import failed:", repr(e_local))
    # 2) try transformers pipeline with FinBERT preference
    try:
        print(f"DEBUG: trying HuggingFace transformers pipeline with model={MODEL_NAME}")
        from transformers import pipeline
        import torch as _torch
        device = 0 if _torch.cuda.is_available() else -1
        try:
            hf_pipe = pipeline("sentiment-analysis", model=MODEL_NAME, device=device)
            analyzer_name = f"HuggingFace pipeline ({MODEL_NAME})"
        except Exception as load_err:
            print("DEBUG: failed loading FinBERT:", repr(load_err))
            print("DEBUG: falling back to default HF sentiment model")
            hf_pipe = pipeline("sentiment-analysis", device=device)
            analyzer_name = "HuggingFace pipeline (default)"

        def get_sentiment(texts):
            out = []
            for t in texts:
                try:
                    r = hf_pipe(t[:512], truncation=True)
                except Exception as e_pipe:
                    print("DEBUG: transformers pipeline call failed for text:", repr(e_pipe))
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

    except Exception as e_hf:
        print("DEBUG: transformers import failed:", repr(e_hf))
        # 3) fallback: keyword-based sentiment
        analyzer_name = "keyword-fallback"
        POS_WORDS = {"good", "great", "positive", "gain", "up", "beat", "raise", "soar", "improve", "strong"}
        NEG_WORDS = {"bad", "weak", "negative", "drop", "down", "miss", "cut", "fall", "plunge", "slump"}
        import re
        def get_sentiment(texts):
            out = []
            for t in texts:
                txt = (t or "").lower()
                # word boundary matches
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

print(f"DEBUG: sentiment analyzer selected: {analyzer_name}")

# prepare texts (headline + summary) and run sentiment in batch
def article_text(a):
    parts = []
    for f in ("headline", "summary", "content"):
        v = a.get(f)
        if v:
            parts.append(v)
    return " — ".join(parts).strip()

texts = [article_text(a) for _, a in results]
print(f"DEBUG: running sentiment on {len(texts)} texts")
sent_scores = get_sentiment(texts)

# print with sentiment label and score
for i, ((ny_dt, a), sc) in enumerate(zip(results, sent_scores), 1):
    try:
        top_label, top_score = max(sc.items(), key=lambda x: x[1])
    except Exception:
        top_label, top_score = "neutral", 0.0
    print(f"{i}. {ny_dt.strftime('%Y-%m-%d %H:%M:%S %Z')} | {top_label.upper()} {top_score:.3f} | {a.get('headline')} | symbols: {a.get('symbols')}")