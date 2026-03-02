import os
import sys
import json
import argparse
from datetime import datetime
from typing import List, Dict, Any

import requests
from dotenv import load_dotenv


# ----------------------------
# Environment / API Key Load
# ----------------------------
load_dotenv(dotenv_path='alpkey.env')  # expects STOCKNEWS_API_KEY=...
API_KEY = os.getenv('STOCKNEWS_API_KEY')


# ----------------------------
# Helpers
# ----------------------------
def fail(msg: str, code: int = 1):
	print(f"ERROR: {msg}", file=sys.stderr)
	sys.exit(code)


def to_mmddyyyy(date_str: str) -> str:
	try:
		return datetime.strptime(date_str, '%Y-%m-%d').strftime('%m%d%Y')
	except ValueError:
		fail(f"Invalid date format '{date_str}'. Use YYYY-MM-DD.")


def build_date_param(single_day: str) -> str:
	mmddyyyy = to_mmddyyyy(single_day)
	return f"{mmddyyyy}-{mmddyyyy}"  # StockNewsAPI single-day range


VALID_SENTIMENT = {"positive", "neutral", "negative"}


def validate_sentiments(s: List[str]) -> List[str]:
	out = []
	for val in s:
		v = val.lower().strip()
		if v not in VALID_SENTIMENT:
			fail(f"Unsupported sentiment '{val}'. Allowed: {', '.join(sorted(VALID_SENTIMENT))}")
		out.append(v)
	return out


def parse_args(argv: List[str]) -> argparse.Namespace:
	p = argparse.ArgumentParser(
		description="Fetch StockNewsAPI articles for a ticker on a specific date filtered by sentiment."
	)
	p.add_argument('--ticker', '-t', required=True, help='Single ticker symbol (e.g. AAPL). For multiple, comma-separate (AAPL,MSFT).')
	p.add_argument('--date', '-d', required=True, help='Target date in YYYY-MM-DD (single day).')
	p.add_argument('--sentiment', '-s', nargs='+', default=['positive'], help='Sentiment(s) to include: positive neutral negative (space separated).')
	p.add_argument('--items', '-n', type=int, default=50, help='Max number of items (API default upper bound).')
	p.add_argument('--start-time', help='Optional start time HHMMSS (24h). Example 130000.')
	p.add_argument('--end-time', help='Optional end time HHMMSS (24h). Example 150000.')
	p.add_argument('--output-json', help='Optional path to write full JSON response.')
	p.add_argument('--min-sentiment-score', type=float, default=None, help='Filter: keep only articles with absolute sentiment score >= value.')
	p.add_argument('--dry-run', action='store_true', help='Print URL and exit without calling API.')
	p.add_argument('--verbose', action='store_true', help='Print response headers and body on error.')
	p.add_argument('--show-score', action='store_true', help='Print numeric sentiment_score alongside each headline')
	return p.parse_args(argv)


def build_url(tickers: str, date_str: str, sentiments: List[str], items: int, start_time: str = None, end_time: str = None) -> str:
	base = 'https://stocknewsapi.com/api/v1'
	date_param = build_date_param(date_str)
	url = f"{base}?tickers={tickers}&items={items}&date={date_param}"
	if start_time and end_time:
		url += f"&time={start_time}-{end_time}"
	# StockNewsAPI sentiment filter (one sentiment per request). If multiple sentiments requested, we'll merge client-side.
	if len(sentiments) == 1:
		url += f"&sentiment={sentiments[0]}"
	url += f"&token={API_KEY}"
	return url


def fetch(url: str, verbose: bool = False) -> Dict[str, Any]:
	r = requests.get(url, timeout=30)
	if r.status_code != 200:
		# Provide more diagnostics when requested
		if verbose:
			print(f"DEBUG: status={r.status_code}")
			print("DEBUG: response headers:")
			for k, v in r.headers.items():
				print(f"  {k}: {v}")
			print("DEBUG: response body:\n", r.text[:4000])
		fail(f"API request failed {r.status_code}: {r.text[:300]}")
	try:
		return r.json()
	except ValueError:
		fail("Response was not valid JSON")


def filter_and_merge(sentiments: List[str], base_url: str, multi: bool, min_score: float, verbose: bool = False) -> List[Dict[str, Any]]:
	articles: List[Dict[str, Any]] = []
	if multi:
		for s in sentiments:
			url = base_url.replace('&token=', f"&sentiment={s}&token=") if '&sentiment=' not in base_url else base_url
			data = fetch(url, verbose=verbose)
			for item in data.get('data', []):
				# mark origin so callers know which endpoint returned this piece
				if isinstance(item, dict):
					item.setdefault('_origin', 'normal')
				articles.append(item)
	else:
		try:
			data = fetch(base_url, verbose=verbose)
			for item in data.get('data', []):
				if isinstance(item, dict):
					item.setdefault('_origin', 'normal')
				articles.append(item)
		except SystemExit as e:
			# Detect subscription restriction error (friendly fallback):
			# Some plans don't allow historical date queries and return 403 with a specific message.
			# If that occurs, retry once without the date parameter and continue.
			msg = str(e)
			# We don't parse e reliably; instead, attempt a diagnostic GET to inspect response
			# by making a minimal request without date.
			fallback_url = base_url
			# remove '&date=' fragment if present
			if 'date=' in fallback_url:
				parts = fallback_url.split('&')
				parts = [p for p in parts if 'date=' not in p]
				fallback_url = '&'.join(parts)
			# ensure token remains
			if 'token=' not in fallback_url:
				fallback_url += f"&token={API_KEY}"
			print("\nWARNING: Date not available under current subscription. Retrying without date filter...")
			data = fetch(fallback_url, verbose=verbose)
			for item in data.get('data', []):
				if isinstance(item, dict):
					item.setdefault('_origin', 'normal')
				articles.append(item)
	# de-duplicate by "news_url" if present
	seen = set()
	deduped = []
	for a in articles:
		key = a.get('news_url') or a.get('title')
		if key in seen:
			continue
		seen.add(key)
		deduped.append(a)
	if min_score is not None:
		filtered = []
		for a in deduped:
			score = a.get('sentiment_score')
			try:
				if score is not None and abs(float(score)) >= min_score:
					filtered.append(a)
			except (TypeError, ValueError):
				pass
		deduped = filtered
	return deduped


def summarize(articles: List[Dict[str, Any]]) -> Dict[str, Any]:
	sentiments_count = {s: 0 for s in VALID_SENTIMENT}
	sources = set()
	times = []
	for a in articles:
		s = (a.get('sentiment') or '').lower()
		if s in sentiments_count:
			sentiments_count[s] += 1
		src = a.get('source_name') or a.get('source')
		if src:
			sources.add(src)
		# articles often have 'date' like '2025-09-26 14:32:01'
		dt_str = a.get('date')
		if dt_str:
			times.append(dt_str)
	times_sorted = sorted(times)
	return {
		'total_articles': len(articles),
		'sentiment_breakdown': {k: v for k, v in sentiments_count.items() if v > 0},
		'sources': sorted(sources),
		'time_span': {'start': times_sorted[0] if times_sorted else None, 'end': times_sorted[-1] if times_sorted else None},
	}


def main(argv: List[str]):
	if not API_KEY:
		fail("Missing STOCKNEWS_API_KEY in alpkey.env")
	args = parse_args(argv)
	sentiments = validate_sentiments(args.sentiment)
	multi = len(sentiments) > 1
	url = build_url(args.ticker, args.date, sentiments, args.items, args.start_time, args.end_time)
	print(f"Base request URL: {url}")
	if args.dry_run:
		print("Dry run mode - exiting before API call.")
		return
	articles = filter_and_merge(sentiments, url, multi, args.min_sentiment_score, verbose=args.verbose)
	if not articles:
		print("No articles matched criteria.")
		return
	summary = summarize(articles)
	print("\nSummary:")
	print(json.dumps(summary, indent=2))
	print("\nArticles:")
	for a in articles:
		origin = a.get('_origin', 'normal')
		sent = a.get('sentiment', '?')
		score = a.get('sentiment_score')
		line = f"[{sent[:7]:7}] ({origin}) {a.get('date','?')} | {a.get('title','<no title>')}"
		if args.show_score:
			line += f" | score={score}"
		print(line)
	if args.output_json:
		# annotate articles with 'origin' and optionally include score
		payload = {'summary': summary, 'articles': articles}
		with open(args.output_json, 'w', encoding='utf-8') as f:
			json.dump(payload, f, indent=2)
		print(f"\nWrote JSON to {args.output_json}")


if __name__ == '__main__':
	main(sys.argv[1:])

