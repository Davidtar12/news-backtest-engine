#!/usr/bin/env python3
"""Keyword / trigger driven StockNewsAPI monitor.

Features (initial version):
  - Fetch news from StockNewsAPI base endpoint (/api/v1) with optional filters:
	  * date range (MMDDYYYY-MMDDYYYY) via --date-range (YYYY-MM-DD:YYYY-MM-DD input)
	  * exchange filter (--exchange NYSE,NASDAQ)
	  * tickers (optional; if omitted, pulls general news for each exchange)
	  * sentiment filter (comma-list)
	  * topics: --topic (AND list), --topic-or (OR list), --topic-exclude
	  * keyword search: --search (AND list), --search-or (OR list)
  - Apply trigger pattern detection with severity scores and actions.
  - Print only items meeting severity threshold (configurable) OR all with --print-all.
  - Continuous loop mode (--loop-seconds) with de-duplication by news_url.
  - Graceful handling of 403 date-plan limitation (retry without date range once).

Future (not yet): integrate per-article local sentiment scoring, persistence, technical analysis.
"""

import os
import sys
import re
import json
import time
import argparse
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple, Set

import requests
from dotenv import load_dotenv

# Load .env from the script directory so running the script from another cwd still finds it.
script_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(script_dir, 'alpkey.env')
if os.path.exists(env_path):
	load_dotenv(env_path)
else:
	# fallback to default search locations
	load_dotenv('alpkey.env')
API_KEY = os.getenv('STOCKNEWS_API_KEY')


# ---------------------------- Trigger Definitions ----------------------------
# Each trigger: name, patterns (regex, case-insensitive), severity, action, optional boost rules
TRIGGERS: List[Dict[str, Any]] = [
	{
		'name': 'deal_definitive',
		'patterns': [r'\bdefinitive agreement\b', r'\bsigned agreement\b', r'has agreed to acquire'],
		'severity': 100,
		'action': 'Auto-exec / notify desk'
	},
	{
		'name': 'deal_closed',
		'patterns': [r'\bclosed the transaction\b', r'\bclosing today\b', r'\bconsummation\b'],
		'severity': 95,
		'action': 'Auto-exec / rebalance'
	},
	{
		'name': 'multibillion',
		'patterns': [
			r'\bmultibillion\b',
			r'multi-billion',
			r'enterprise value',
			r'\$[0-9,]+(\.[0-9]+)?\s*(?:billion|bn|B).*(?:deals?|contracts?|partnerships?|pacts?|agreements?)',
			r'(?:deals?|contracts?|partnerships?|pacts?|agreements?).*\$[0-9,]+(\.[0-9]+)?\s*(?:billion|bn|B)',
			r'\bbillions?\s+(?:\w+\s+){0,5}(?:deals?|contracts?|partnerships?|pacts?|agreements?)\b',
			r'\bmultibillion[\s-]+(?:\w+[\s-]+){0,5}(?:deals?|contracts?|partnerships?|pacts?|agreements?)\b',
			r'\$[0-9,]+\s*(?:billion|bn|B)\s+(?:cloud|data|computing|chip|AI)\s+(?:deals?|contracts?|pacts?)',
			r'(?:roughly|approximately|about|near)\s+\$[0-9,]+\s*(?:billion|bn|B)\s+deals?'
		],
		'severity': 100,
		'action': 'Promote (boost +10 if multi-billion value found)'
	},
	{
		'name': 'massive_deal',
		'patterns': [r'massive deal', r'huge deal', r'major transaction'],
		'severity': 90,
		'action': 'Promote → human review'
	},
	{
		'name': 'loi_exclusivity',
		'patterns': [r'letter of intent', r'\bLOI\b', r'exclusivity period', r'exclusive negotiations'],
		'severity': 85,
		'action': 'Prep sizing / desk alert'
	},
	{
		'name': 'merger_acq_general',
		'patterns': [
			r'merger announced',
			r'acquisition announced',
			r'merger agreement',
			r'announces?\s+.*acquisition',
			# Common acquisition wordings
			r'\bto\s+be\s+acquired\s+by\b',
			r'\bto\s+acquire\b',
			r'\bacquired\s+by\b',
			r'\bwill\s+acquire\b',
			r'\bagrees?\s+to\s+acquire\b',
			r'\bto\s+buy\b',
			r'\bto\s+purchase\b',
			r'\btakeover\b',
			r'\bbuyout\b'
		],
		'severity': 88,
		'action': 'Promote high priority'
	},
	# Explicit acquisition with a stated deal value (captures "for up to $5.2 Billion", etc.)
	{
		'name': 'acq_with_value',
		'patterns': [
			r'(?:to\s+be\s+acquired\s+by|to\s+acquire|acquires?|acquired\s+by)[^\n]*?(?:for|worth|valued\s+at)\s+(?:up\s+to\s+)?\$[0-9,.]+(?:\.[0-9]+)?\s*(?:billion|bn|B|million|m|M)'
		],
		'severity': 97,
		'action': 'Acquisition with stated value'
	},
	{
		'name': 'fda_approval',
		'patterns': [
			r'announces?\s+fda\s+approval',
			r'fda\s+approves',
			r'gets?\s+fda\s+approval'
		],
		'severity': 92,
		'action': 'Regulatory alert / notify healthcare desk'
	},
	{
		'name': 'deadline_extension',
		'patterns': [
			r'deadline',
			r'extends?\s+.*(ban|deal)',
			r'delay(?:s|ed)\s+.*(ban|deal)'
		],
		'severity': 82,
		'action': 'Regulatory timeline monitoring'
	},
	{
		'name': 'controlling_stake',
		'patterns': [
			r'controlling\s+stake',
			r'controlling\s+interest',
			r'(control|stake).*(?:consortium|investors?)',
			r'\d{1,3}%\s+stake',
			r'(?:acquisition|acquires?)\s+(?:of\s+)?controlling\s+(?:stake|interest)',
			r'strategic\s+acquisition\s+of\s+controlling'
		],
		'severity': 94,
		'action': 'Ownership structure change alert'
	},
	{
		'name': 'cloud_ai_deal',
		'patterns': [
			r'cloud\s+(?:computing\s+)?(?:deals?|contracts?)\b',
			r'\bAI\b.*(?:deals?|contracts?|partnerships?)\b',
			r'data\s+center\s+(?:deals?|contracts?)\b',
			r'computing\s+(?:deal|contract)\b',
			r'partners?\s+with\s+(?:OpenAI|Anthropic|Nvidia|AMD|IBM|Oracle|Microsoft)'
		],
		'severity': 86,
		'action': 'AI / cloud strategic deal'
	},
	{
		'name': 'contract_award',
		'patterns': [
			r'\b(wins?|awarded|signs?|lands?)\s+.*contract\b',
			r'\bcontract\s+win\b',
			r'secures?\s+.*contract'
		],
		'severity': 85,
		'action': 'Contract award / win alert'
	},
	{
		'name': 'deal_chatter',
		'patterns': [
			r'deal\s+involvement',
			r'talk\s+of\s+a\s+deal',
			r'deal\s+near',
			r'a\s+.*deal\s+near',
			r'join\s+.*deal',
			r'work\s+on\s+deal',
			r'help\s+keep.*(?:under|in)\s+(?:new\s+)?deal',
			r'(?:key|major)\s+role.*in.*deal',
			r'play.*role.*in.*deal',
			r'to\s+(?:go|take)\s+private',
			r'near.*deal\s+to\s+go\s+private'
		],
		'severity': 83,
		'action': 'Deal progress / chatter watch'
	},
	{
		'name': 'public_private_partnership',
		'patterns': [
			r'public[-\s]*private\s+partnership',
			r'partnership\s+with\s+the\s+department\s+of\s+defense',
			r'department\s+of\s+defense\s+partnership',
			r'(?:U\.?S\.?|United States)\s+(?:government|DoD)\s+partnership',
			r'public[-\s]*private.*(?:department\s+of\s+defense|DoD)'
		],
		'severity': 92,
		'action': 'Government partnership alert'
	},
	{
		'name': 'government_investment',
		'patterns': [
			r'strategic\s+investment\s+by\s+(?:the\s+)?U\.?S\.?\s+(?:federal\s+)?government',
			r'(?:Department|Dept)\s+of\s+Defense\s+(?:investment|funding|stake)',
			r'(?:U\.?S\.?|United States)\s+government\s+to\s+take\s+\d{1,3}%\s+stake',
			r'government\s+(?:takes?|taking|to\s+take)\s+\d{1,3}%\s+stake'
		],
		'severity': 93,
		'action': 'Government investment / ownership change'
	},
	{
		'name': 'ai_enterprise_partnership',
		'patterns': [
			r'partners?\s+to\s+advance\s+(?:enterprise\s+)?(?:AI|artificial intelligence)',
			r'\bAI\b\s+(?:strategic\s+)?partnership',
			r'(?:OpenAI|Anthropic|IBM|AMD|Nvidia|Oracle)\s+and\s+(?:OpenAI|Anthropic|IBM|AMD|Nvidia|Oracle).*partners?',
			r'partners?\s+with\s+(?:OpenAI|Anthropic|IBM|AMD|Nvidia|Oracle)'
		],
		'severity': 94,
		'action': 'AI enterprise partnership alert'
	},
	{
		'name': 'strategic_partnership',
		'patterns': [
			r'strategic partnership',
			r'joint venture',
			r'strategic alliance',
			r'partners?\s+with\s+',
			r'partners?\s+in\s+(?:bid|drive|effort)',
			r'partners?\s+to\s+'
		],
		'severity': 80,
		'action': 'Desk review'
	},
	{
		'name': 'stake_purchase',
		'patterns': [r'acquires? \d{1,3}% stake', r'\d{1,3}%\s+stake', r'buys stake', r'increases stake', r'majority stake'],
		'severity': 78,
		'action': 'Desk review (promote if >20%)'
	},
	{
		'name': 'tender_hostile_proxy',
		'patterns': [r'tender offer', r'unsolicited offer', r'hostile bid', r'proxy fight', r'\bin talks\b'],
		'severity': 85,
		'action': 'Hedge + urgent desk alert'
	},
]

# Quick compiled patterns stored lazily
for trig in TRIGGERS:
	trig['compiled'] = [re.compile(p, re.IGNORECASE) for p in trig['patterns']]


def parse_args(argv: List[str]) -> argparse.Namespace:
	p = argparse.ArgumentParser(description='Continuous keyword / trigger monitor for StockNewsAPI')
	p.add_argument('--tickers', '-t', help='Comma-separated tickers to filter (optional)')
	p.add_argument('--exchange', help='Comma-separated exchanges (e.g. NYSE,NASDAQ)')
	p.add_argument('--date-range', default='2025-09-25:2025-09-26',
				   help='YYYY-MM-DD:YYYY-MM-DD (converted to mmddyyyy-mmddyyyy)')
	p.add_argument('--items', type=int, default=50, help='Items per page')
	p.add_argument('--pages', type=int, default=1, help='Max pages to pull per request (paginate)')
	p.add_argument('--sentiment', help='Comma list sentiment filter (positive,negative,neutral)')
	# Topic / search filters
	p.add_argument('--topic', help='Topic AND list, comma separated')
	p.add_argument('--topic-or', help='Topic OR list, comma separated')
	p.add_argument('--topic-exclude', help='Topic exclude list')
	p.add_argument('--source-exclude', help='Comma-separated sources to exclude (e.g. CNBC)')
	p.add_argument('--search', help='Keyword AND list (API &search=) use + or space replaced automatically')
	p.add_argument('--search-or', help='Keyword OR list (API &searchOR=)')
	p.add_argument('--severity-threshold', type=int, default=80, help='Print only articles with max severity >= this (unless --print-all)')
	p.add_argument('--print-all', action='store_true', help='Print every article line even if below threshold')
	p.add_argument('--no-triggers', action='store_true', help='Disable trigger keyword matching (all severities = 0)')
	p.add_argument('--show-matches', action='store_true', help='Append matched trigger keyword snippets to each printed line')
	p.add_argument('--test-title', help='Test a single headline string against triggers and exit')
	p.add_argument('--loop-seconds', type=int, default=0, help='If >0, poll in a loop with this sleep interval (seconds)')
	p.add_argument('--max-runtime', type=int, help='Optional max runtime in seconds for loop')
	p.add_argument('--output-jsonl', help='Append JSON lines with full matched articles to this file')
	p.add_argument('--verbose', '-v', action='store_true')
	p.add_argument('--debug-triggers', action='store_true', help='Enable detailed trigger evaluation debugging')
	return p.parse_args(argv)


def to_api_date(range_str: str) -> Optional[str]:
	try:
		start_s, end_s = range_str.split(':', 1)
		sdt = datetime.strptime(start_s, '%Y-%m-%d')
		edt = datetime.strptime(end_s, '%Y-%m-%d')
		return f"{sdt.strftime('%m%d%Y')}-{edt.strftime('%m%d%Y')}"
	except Exception:
		return None


def parse_article_datetime(date_str: Optional[str]) -> Optional[datetime]:
	if not date_str:
		return None
	try:
		return datetime.strptime(date_str.strip(), '%a, %d %b %Y %H:%M:%S %z')
	except Exception:
		return None


def build_base_url(**params) -> str:
	base = 'https://stocknewsapi.com/api/v1'
	# Filter out None/empty
	qp = []
	for k, v in params.items():
		if v is None:
			continue
		qp.append(f"{k}={v}")
	qp.append(f"token={API_KEY}")
	return base + '?' + '&'.join(qp)


def fetch(url: str, verbose: bool = False) -> Dict[str, Any]:
	r = requests.get(url, timeout=30)
	if r.status_code == 403 and 'date is not available' in r.text.lower():
		raise PermissionError('Subscription plan does not allow this date range')
	if r.status_code != 200:
		if verbose:
			print(f"DEBUG {r.status_code} {url}\n{r.text[:500]}", file=sys.stderr)
		raise RuntimeError(f"API error {r.status_code}")
	try:
		return r.json()
	except ValueError:
		raise RuntimeError('Invalid JSON response')


def sentiment_allowed(article: Dict[str, Any], allowed: Optional[Set[str]]) -> bool:
	if not allowed:
		return True
	s = (article.get('sentiment') or '').lower()
	return s in allowed


def evaluate_triggers(title: str, debug: bool = False) -> Tuple[int, List[Dict[str, Any]]]:
	"""Evaluate triggers using headline text only."""
	if debug:
		print(f"\n--- DEBUG: Evaluating Title ---\n'{title}'")

	max_sev = 0
	matched: List[Dict[str, Any]] = []
	if not title:
		return 0, matched
	for trig in TRIGGERS:
		if debug:
			print(f"  Testing trigger '{trig['name']}'...")
		title_hits: List[str] = []
		patterns_hit: List[str] = []
		for i, cp in enumerate(trig['compiled']):
			if debug:
				print(f"    - Pattern {i+1}: {cp.pattern}")
			t_matches = [m.group(0) for m in cp.finditer(title)]
			if t_matches:
				if debug:
					print(f"      ==> HIT: {t_matches}")
				title_hits.extend(t_matches)
				patterns_hit.append(cp.pattern)
		if patterns_hit:
			sev = trig['severity']
			if trig['name'] == 'multibillion' and re.search(r'\$[0-9,]+(\.[0-9]+)?\s*billion', title, re.IGNORECASE):
				sev = min(100, sev + 10)
			if trig['name'] == 'stake_purchase':
				m = re.findall(r'(\d{1,3})%\s+stake', title, re.IGNORECASE)
				if m:
					try:
						pct_max = max(int(x) for x in m)
						if pct_max >= 20:
							sev = min(100, sev + 10)
					except Exception:
						pass
			examples = list(dict.fromkeys(title_hits))[:5]
			matched.append({
				'trigger': trig['name'],
				'patterns': patterns_hit,
				'examples': examples,
				'location': 'title',
				'severity': sev,
				'action': trig['action']
			})
			if sev > max_sev:
				max_sev = sev
	if debug:
		print(f"--- END DEBUG: Max Severity Found: {max_sev} ---")
	return max_sev, matched


def process_articles(raw_articles: List[Dict[str, Any]], args: argparse.Namespace, seen: Set[str]) -> List[Dict[str, Any]]:
	out = []
	allowed_sent = set(x.strip().lower() for x in args.sentiment.split(',')) if args.sentiment else None
	for a in raw_articles:
		if not isinstance(a, dict):
			continue
		url = a.get('news_url') or a.get('url') or a.get('link')
		if not url:
			continue
		# source name / url (left here for later filtering if needed)
		source_name = (a.get('source_name') or a.get('source') or '')
		# Client-side source exclusion
		if args.source_exclude:
			try:
				excluded = [s.strip().lower() for s in args.source_exclude.split(',') if s.strip()]
				if any(ex in (source_name or '').lower() for ex in excluded):
					continue
			except Exception:
				pass
		if url in seen:
			continue
		if not sentiment_allowed(a, allowed_sent):
			continue
		title = a.get('title') or ''
		txt = a.get('text') or a.get('content') or a.get('description') or ''
		if args.no_triggers:
			max_sev, matches = 0, []
		else:
			max_sev, matches = evaluate_triggers(title, debug=args.debug_triggers)
		record = {
			'title': title,
			'date': a.get('date'),
			'source': a.get('source_name') or a.get('source'),
			'sentiment': a.get('sentiment'),
			'tickers': a.get('tickers') or a.get('symbols'),
			'news_url': url,
			'max_severity': max_sev,
			'trigger_matches': matches,
			'raw': a
		}
		out.append(record)
		seen.add(url)
		# If triggers disabled, severity threshold has no effect unless print_all missing; default print all.
		if args.no_triggers:
			should_print = True  # always print when triggers disabled
		else:
			should_print = args.print_all or max_sev >= args.severity_threshold
		if should_print:
			sev_tag = f"SEV{max_sev}" if max_sev else "SEV0"
			match_snip = ''
			if args.show_matches and matches:
				example_pairs = []  # (location, example)
				for m in matches:
					loc = (m.get('location') or 'title').lower()
					for ex in m.get('examples') or []:
						example_pairs.append((loc, ex))
				title_examples = [(loc, ex) for loc, ex in example_pairs if loc == 'title']
				if title_examples:
					display_pairs = title_examples
				else:
					display_pairs = example_pairs
				# Deduplicate while preserving order
				seen = set()
				uniq_pairs = []
				for loc, ex in display_pairs:
					key = (loc, ex)
					if key in seen:
						continue
					seen.add(key)
					uniq_pairs.append((loc, ex))
				uniq_pairs = uniq_pairs[:5]
				match_snip = ' | KW=' + ';'.join(f"[{loc}] {repr(ex)}" for loc, ex in uniq_pairs)
			print(f"{a.get('date')} | {sev_tag} | {title[:140]} | {a.get('sentiment')} | {url}{match_snip}")
			if matches and args.verbose:
				for m in matches:
					print(f"  -> {m['trigger']} severity={m['severity']} patterns={','.join(m['patterns'])}")
	return out


def build_requests(args: argparse.Namespace) -> List[str]:
	date_param = to_api_date(args.date_range)
	tickers = (args.tickers or '').strip()
	exchanges = [e.strip() for e in args.exchange.split(',')] if args.exchange else []
	if not exchanges:
		exchanges = [None]  # no exchange filter
	urls = []
	for ex in exchanges:
		base_params = {
			'items': args.items,
			'page': 1,
		}
		if date_param:
			base_params['date'] = date_param
		if tickers:
			base_params['tickers'] = tickers
		if ex:
			base_params['exchange'] = ex
		# topic/search parameters
		if args.topic:
			base_params['topic'] = args.topic
		if args.topic_or:
			base_params['topicOR'] = args.topic_or
		if args.topic_exclude:
			base_params['topicexclude'] = args.topic_exclude
		if args.source_exclude:
			base_params['sourceexclude'] = args.source_exclude
		if args.search:
			base_params['search'] = args.search.replace(' ', '+')
		if args.search_or:
			base_params['searchOR'] = args.search_or.replace(' ', '+')
		# Build per-page URLs
		for pg in range(1, args.pages + 1):
			base_params['page'] = pg
			urls.append(build_base_url(**base_params))
	return urls


def main(argv: List[str]):
	if not API_KEY:
		print('ERROR: Missing STOCKNEWS_API_KEY in alpkey.env', file=sys.stderr)
		sys.exit(1)
	args = parse_args(argv)
	# Quick test mode: evaluate a single headline string and exit
	if args.test_title:
		if args.debug_triggers:
			print("TEST MODE: Evaluating single title")
		sev, matches = evaluate_triggers(args.test_title, debug=args.debug_triggers)
		print(f"Title: {args.test_title}\nMax severity: {sev}")
		if matches:
			print('Matches:')
			for m in matches:
				print(f" - trigger={m['trigger']} severity={m['severity']} examples={m.get('examples')}")
		else:
			print('No triggers matched.')
		return
	seen: Set[str] = set()
	start_time = time.time()
	iteration = 0
	start_dt: Optional[datetime] = None
	if args.date_range:
		try:
			start_part = args.date_range.split(':', 1)[0]
			start_dt = datetime.strptime(start_part, '%Y-%m-%d')
		except Exception:
			start_dt = None
	while True:
		iteration += 1
		try:
			urls = build_requests(args)
			oldest_seen: Optional[datetime] = None
			total_returned = 0
			for u in urls:
				if args.verbose:
					print('Request ->', u)
				try:
					data = fetch(u, verbose=args.verbose)
				except PermissionError as pe:
					# fallback: retry without date once
					if args.verbose:
						print('Date restricted, retry without date param:', pe)
					if 'date=' in u:
						no_date = re.sub(r'&date=[^&]+', '', u)
						try:
							data = fetch(no_date, verbose=args.verbose)
						except Exception as e2:
							if args.verbose:
								print('Failed fallback:', e2)
							continue
					else:
						continue
				except Exception as e:
					if args.verbose:
						print('Fetch error:', e)
					continue
				raw_articles = data.get('data', []) if isinstance(data, dict) else []
				total_returned += len(raw_articles)
				for art in raw_articles:
					dt = parse_article_datetime(art.get('date'))
					if dt is not None and (oldest_seen is None or dt < oldest_seen):
						oldest_seen = dt
				processed = process_articles(raw_articles, args, seen)
				if processed and args.output_jsonl:
					try:
						with open(args.output_jsonl, 'a', encoding='utf-8') as f:
							for rec in processed:
								f.write(json.dumps(rec) + '\n')
					except Exception as e:
						print('WARNING: cannot write output_jsonl:', e, file=sys.stderr)
			if start_dt and oldest_seen and oldest_seen.date() > start_dt.date():
				print(
					f"NOTE: Retrieved {total_returned} articles but oldest headline fetched is {oldest_seen.strftime('%Y-%m-%d')}. "
					f"Requested date range starts {start_dt.strftime('%Y-%m-%d')} — increase --pages/--items or narrow the date range to capture earlier news.",
					file=sys.stderr
				)
		except KeyboardInterrupt:
			print('Interrupted by user.')
			break
		if args.loop_seconds <= 0:
			break
		if args.max_runtime and (time.time() - start_time) >= args.max_runtime:
			print('Max runtime reached; exiting.')
			break
		if args.verbose:
			print(f'--- Sleep {args.loop_seconds}s (iteration {iteration}) ---')
		time.sleep(args.loop_seconds)


if __name__ == '__main__':
	main(sys.argv[1:])

