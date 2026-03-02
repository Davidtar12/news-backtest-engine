import os
import sys
import json
import argparse
from typing import List, Dict, Any

import requests
from dotenv import load_dotenv


# Load API key
load_dotenv(dotenv_path='alpkey.env')
API_KEY = os.getenv('STOCKNEWS_API_KEY')


def fail(msg: str, code: int = 1):
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(code)


def parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Fetch alerts or trending headlines and merge results')
    p.add_argument('--tickers', '-t', required=True, help='Comma-separated tickers (e.g. TSLA or AAPL,MSFT)')
    p.add_argument('--page', '-p', type=int, default=1, help='Page number')
    p.add_argument('--items', '-n', type=int, default=100, help='Items per page')
    p.add_argument('--mode', choices=['alerts', 'trending', 'normal', 'both', 'all'], default='all', help='Which endpoint(s) to query')
    p.add_argument('--output-json', help='Write merged JSON to file')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--verbose', action='store_true')
    p.add_argument('--full-text', action='store_true', help='Include/print full article text when available')
    return p.parse_args(argv)


def build_alerts_url(tickers: str, page: int, items: int) -> str:
    base = 'https://stocknewsapi.com/api/v1/alerts'
    return f"{base}?page={page}&items={items}&category=ticker&tickers={tickers}&token={API_KEY}"


def build_trending_url(ticker: str, page: int) -> str:
    base = 'https://stocknewsapi.com/api/v1/trending-headlines'
    return f"{base}?page={page}&ticker={ticker}&token={API_KEY}"


def fetch(url: str, verbose: bool = False) -> Dict[str, Any]:
    r = requests.get(url, timeout=30)
    if r.status_code != 200:
        if verbose:
            print('DEBUG STATUS', r.status_code)
            for k, v in r.headers.items():
                print(f'  {k}: {v}')
            print('DEBUG BODY:', r.text[:2000])
        fail(f'API request failed {r.status_code}: {r.text[:300]}')
    try:
        return r.json()
    except ValueError:
        fail('Invalid JSON response')


def merge_and_dedupe(list_of_articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # Preserve and merge origin tags across duplicates
    seen: Dict[str, int] = {}
    out: List[Dict[str, Any]] = []
    for a in list_of_articles:
        key = a.get('news_url') or a.get('title')
        if not key:
            key = json.dumps(a, sort_keys=True)
        # determine origins for this item
        origins = set()
        if isinstance(a, dict):
            if '_origins' in a and isinstance(a['_origins'], (list, set)):
                origins.update(a['_origins'])
            if '_origin' in a and a['_origin']:
                origins.add(a['_origin'])
        if key in seen:
            idx = seen[key]
            existing = out[idx]
            ex_origins = set(existing.get('_origins', []))
            ex_origins.update(origins)
            existing['_origins'] = sorted(ex_origins)
        else:
            # copy item shallowly and set _origins
            item_copy = dict(a) if isinstance(a, dict) else {'raw': a}
            item_copy['_origins'] = sorted(origins) if origins else []
            seen[key] = len(out)
            out.append(item_copy)
    return out


def normalize_article(a: Dict[str, Any], retrieved_at: str) -> Dict[str, Any]:
    # Map likely fields to a consistent shape
    return {
        'title': a.get('title') or a.get('headline') or '',
        'date': a.get('date') or a.get('published_at') or a.get('time') or None,
        'retrieved_at': retrieved_at,
        'source': a.get('source_name') or a.get('source') or a.get('publisher') or None,
        'sentiment': a.get('sentiment'),
        'sentiment_score': a.get('sentiment_score'),
        'text': a.get('text') or a.get('content') or a.get('description') or None,
        'news_url': a.get('news_url') or a.get('url') or None,
        'origins': a.get('_origins') or ([a.get('_origin')] if a.get('_origin') else []),
        'raw': a,
    }


def main(argv: List[str]):
    if not API_KEY:
        fail('Missing STOCKNEWS_API_KEY in alpkey.env')
    args = parse_args(argv)
    tickers = args.tickers
    page = args.page
    items = args.items
    all_articles: List[Dict[str, Any]] = []
    if args.dry_run:
        if args.mode in ('alerts', 'both'):
            print('Alerts URL:', build_alerts_url(tickers, page, items))
        if args.mode in ('trending', 'both'):
            # show one example for first ticker
            first = tickers.split(',')[0]
            print('Trending URL:', build_trending_url(first, page))
        return

    if args.mode in ('alerts', 'both'):
        url = build_alerts_url(tickers, page, items)
        if args.verbose:
            print('Requesting alerts ->', url)
        data = fetch(url, verbose=args.verbose)
        for item in data.get('data', []):
            if isinstance(item, dict):
                item.setdefault('_origin', 'alerts')
            all_articles.append(item)

    # normal endpoint: base /api/v1 (general news endpoint)
    if args.mode in ('normal', 'both', 'all'):
        normal_base = 'https://stocknewsapi.com/api/v1'
        # the normal endpoint accepts tickers and items
        normal_url = f"{normal_base}?tickers={tickers}&items={items}&token={API_KEY}"
        if args.verbose:
            print('Requesting normal ->', normal_url)
        data = fetch(normal_url, verbose=args.verbose)
        for item in data.get('data', []):
            if isinstance(item, dict):
                item.setdefault('_origin', 'normal')
            all_articles.append(item)

    if args.mode in ('trending', 'both', 'all'):
        # trending endpoint accepts single ticker param; query each ticker individually
        for t in tickers.split(','):
            t = t.strip()
            url = build_trending_url(t, page)
            if args.verbose:
                print('Requesting trending ->', url)
            data = fetch(url, verbose=args.verbose)
            # trading endpoint might return top-level 'data' or 'articles'
            if isinstance(data, dict) and 'data' in data:
                all_articles.extend(data.get('data', []))
            else:
                # attempt to treat whole response as a list
                if isinstance(data, list):
                    all_articles.extend(data)

    merged = merge_and_dedupe(all_articles)
    retrieved_at = __import__('datetime').datetime.utcnow().isoformat() + 'Z'
    normalized = [normalize_article(a, retrieved_at) for a in merged]
    print(f'Merged articles: {len(normalized)}')
    if args.output_json:
        with open(args.output_json, 'w', encoding='utf-8') as f:
            json.dump({'articles': normalized}, f, indent=2)
        print('Wrote', args.output_json)
    else:
        # print brief list
        for a in normalized:
            date = a.get('date') or '?'
            title = a.get('title') or '<no title>'
            sent = a.get('sentiment') or '?'
            url = a.get('news_url') or ''
            origins = a.get('origins') or []
            # short origin tags
            tags = ''.join(['A' if 'alerts' in o else ('T' if 'trending' in o else 'N') for o in origins]) or 'N'
            print(f"[{tags}] {date} | {title} | {sent} | {url}")
            if args.full_text and a.get('text'):
                print('--- Full text ---')
                print(a.get('text')[:4000])
                print('--- End text ---')

    # Additional processing for sentiment scores (optional)
    # Use the configured API key and call the /stat endpoint, but don't crash on 403 or network errors.
    stat_url = f"https://stocknewsapi.com/api/v1/stat?&tickers={tickers}&date=last30days&page=1&token={API_KEY}"
    stat = None
    ticker_score_map = {}
    try:
        if args.verbose:
            print('Requesting stats ->', stat_url)
        r = requests.get(stat_url, timeout=10)
        r.raise_for_status()
        stat = r.json()
    except requests.exceptions.HTTPError as e:
        # Common case: subscription plan doesn't allow date/stat data (403). Warn and continue.
        print(f'WARNING: stat endpoint request failed: {e} (status={getattr(e.response, "status_code", "?")})', file=sys.stderr)
        if args.verbose and getattr(e, 'response', None) is not None:
            try:
                print('STAT RESPONSE BODY:', e.response.text[:1000], file=sys.stderr)
            except Exception:
                pass
        stat = None
    except Exception as e:
        print(f'WARNING: could not fetch stat data: {e}', file=sys.stderr)
        stat = None

    if stat:
        # Example: stat response likely contains items with 'ticker' and 'score' or similar
        for item in stat.get('data', []):
            if not isinstance(item, dict):
                # skip unexpected shapes
                continue
            t = item.get('ticker') or item.get('symbol') or item.get('tickers')
            score = item.get('score') or item.get('sentiment_score') or item.get('sentiment')
            if t and score is not None:
                try:
                    # coerce numeric if it's a string
                    ticker_score_map[t.upper()] = float(score)
                except Exception:
                    ticker_score_map[t.upper()] = score

    # Merge into saved articles file if present. Use the same output filename as input when possible.
    infile = args.output_json or 'kuke_all.json'
    try:
        with open(infile, 'r', encoding='utf8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f'INFO: input file for merging scores not found: {infile}. Skipping score merge.', file=sys.stderr)
        data = None

    if data and ticker_score_map:
        for art in data.get('articles', []):
            if art.get('sentiment_score') is None:
                tickers_in_art = art.get('raw', {}).get('tickers', []) or []
                # prefer the first matching ticker; could be averaged if desired
                for t in tickers_in_art:
                    s = ticker_score_map.get(t.upper())
                    if s is not None:
                        art['sentiment_score'] = s
                        break

        out_file = infile.replace('.json', '_with_scores.json')
        try:
            with open(out_file, 'w', encoding='utf8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print('Wrote', out_file)
        except Exception as e:
            print(f'WARNING: failed to write merged scores file: {e}', file=sys.stderr)
    elif data and not ticker_score_map:
        print('INFO: no ticker stat scores available to merge (stat endpoint returned no data)', file=sys.stderr)


if __name__ == '__main__':
    main(sys.argv[1:])
