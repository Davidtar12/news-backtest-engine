#!/usr/bin/env python3
"""Fetch Amazon (AMZN) news for a specific date (default: today) and optionally query /stat for numeric sentiment.

Usage examples:
  python amzn_today.py                 # use API key from alpkey.env, date=today
  python amzn_today.py --date 2025-10-07 --stat
  python amzn_today.py --token YOURTOKEN --output-json amzn_today.json
"""

import os
import sys
import json
import argparse
from datetime import datetime, date
from email.utils import parsedate_to_datetime

import requests
from dotenv import load_dotenv

# load API key from alpkey.env if present
load_dotenv(dotenv_path='alpkey.env')
API_KEY = os.getenv('STOCKNEWS_API_KEY')


def parse_args(argv=None):
    p = argparse.ArgumentParser(description='Fetch AMZN news for a given date and optional stat score')
    p.add_argument('--date', '-d', help='Date to filter (YYYY-MM-DD). Defaults to today', default=None)
    p.add_argument('--items', '-n', type=int, default=50, help='Items per page when requesting news')
    p.add_argument('--token', help='API token (overrides alpkey.env)')
    p.add_argument('--stat', action='store_true', help='Also query /stat endpoint for numeric sentiment')
    p.add_argument('--stat-range', help='Range to pass to /stat (e.g. today, last7days, last30days). Defaults to today', default='today')
    p.add_argument('--output-json', help='Write filtered articles to JSON file')
    p.add_argument('--verbose', '-v', action='store_true')
    return p.parse_args(argv)


def build_normal_url(ticker: str, items: int, token: str) -> str:
    base = 'https://stocknewsapi.com/api/v1'
    return f"{base}?tickers={ticker}&items={items}&token={token}"


def build_stat_url(ticker: str, date_param: str, token: str) -> str:
    base = 'https://stocknewsapi.com/api/v1/stat'
    return f"{base}?&tickers={ticker}&date={date_param}&page=1&token={token}&cache=false"


def fetch_json(url: str, verbose: bool = False, timeout: int = 15):
    if verbose:
        print('GET', url)
    r = requests.get(url, timeout=timeout)
    try:
        r.raise_for_status()
    except Exception as e:
        if verbose:
            print('HTTP error', e, file=sys.stderr)
            print('Response body:', r.text[:1000], file=sys.stderr)
        raise
    return r.json()


def parse_article_date(dstr: str):
    if not dstr:
        return None
    try:
        dt = parsedate_to_datetime(dstr)
        # normalize to date
        return dt.date()
    except Exception:
        # try ISO parse
        try:
            return datetime.fromisoformat(dstr).date()
        except Exception:
            return None


def main(argv=None):
    args = parse_args(argv)
    token = args.token or API_KEY
    if not token:
        print('ERROR: Missing API token. Set STOCKNEWS_API_KEY in alpkey.env or pass --token', file=sys.stderr)
        sys.exit(1)

    target_date = None
    if args.date:
        try:
            target_date = datetime.strptime(args.date, '%Y-%m-%d').date()
        except ValueError:
            print('ERROR: date must be YYYY-MM-DD', file=sys.stderr)
            sys.exit(2)
    else:
        target_date = date.today()

    ticker = 'AMZN'
    normal_url = build_normal_url(ticker, args.items, token)

    try:
        data = fetch_json(normal_url, verbose=args.verbose)
    except Exception as e:
        print('ERROR: failed to fetch normal articles:', e, file=sys.stderr)
        sys.exit(3)

    articles = []
    raw_list = data.get('data') if isinstance(data, dict) else (data if isinstance(data, list) else [])
    for item in raw_list or []:
        # item expected dict
        if not isinstance(item, dict):
            continue
        adate = parse_article_date(item.get('date') or item.get('published_at') or item.get('time'))
        if adate == target_date:
            # normalize minimal fields
            articles.append({
                'title': item.get('title'),
                'date': item.get('date'),
                'source': item.get('source_name') or item.get('source'),
                'sentiment': item.get('sentiment'),
                'sentiment_score': item.get('sentiment_score'),
                'news_url': item.get('news_url') or item.get('url'),
                'raw': item,
            })

    print(f'Found {len(articles)} AMZN articles for {target_date.isoformat()}')
    for a in articles:
        title = a.get('title') or '<no title>'
        sent = a.get('sentiment') or '?'
        score = a.get('sentiment_score')
        url = a.get('news_url') or ''
        print(f"{a.get('date')} | {title} | {sent} | {score} | {url}")

    # optionally call /stat to get numeric sentiment for the ticker/date range
    if args.stat:
        # /stat expects a range like last1days/last7days; avoid single-date (422)
        date_param = args.stat_range or 'last1days'
        stat_url = build_stat_url(ticker, date_param, token)
        try:
            stat = fetch_json(stat_url, verbose=args.verbose)
            # The /stat response can be shaped as {"data": {"YYYY-MM-DD": {"TICKER": {...}}}, "total": {...}}
            score = None
            if isinstance(stat, dict):
                data_blob = stat.get('data') or {}
                # prefer the exact date key (e.g. '2025-10-07') when present
                date_key = None
                # if user passed 'today' as stat-range, use target_date
                try:
                    date_key = target_date.isoformat()
                except Exception:
                    pass
                if date_key and isinstance(data_blob, dict) and date_key in data_blob:
                    day_entry = data_blob.get(date_key, {})
                    if isinstance(day_entry, dict):
                        tentry = day_entry.get(ticker) or day_entry.get(ticker.upper()) or day_entry.get(ticker.lower())
                        if isinstance(tentry, dict):
                            score = tentry.get('sentiment_score') or tentry.get('Sentiment Score') or tentry.get('score')
                # fallback: some responses include a per-ticker 'total' or top-level mapping
                if score is None:
                    total = stat.get('total') or {}
                    if isinstance(total, dict):
                        ttotal = total.get(ticker) or total.get(ticker.upper())
                        if isinstance(ttotal, dict):
                            score = ttotal.get('Sentiment Score') or ttotal.get('sentiment_score') or ttotal.get('score')

            # coerce numeric if possible
            try:
                if score is not None:
                    score = float(score)
            except Exception:
                pass

            print('STAT score for', ticker, date_param, '->', score)

            # merge into each article where sentiment_score is missing
            if score is not None and articles:
                for a in articles:
                    if a.get('sentiment_score') is None:
                        a['sentiment_score'] = score

        except requests.exceptions.HTTPError as e:
            print('STAT endpoint error:', e, file=sys.stderr)
        except Exception as e:
            print('STAT fetch failed:', e, file=sys.stderr)

    if args.output_json:
        out = {'date': target_date.isoformat(), 'ticker': ticker, 'articles': articles}
        try:
            with open(args.output_json, 'w', encoding='utf8') as f:
                json.dump(out, f, indent=2, ensure_ascii=False)
            print('Wrote', args.output_json)
        except Exception as e:
            print('Failed to write output file:', e, file=sys.stderr)


if __name__ == '__main__':
    main()
