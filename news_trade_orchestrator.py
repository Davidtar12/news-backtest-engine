#!/usr/bin/env python3
"""News-to-trade orchestrator integrating an inline BREAKING/REPETITION classifier plus
flexible universe sources, MT5 extended-hours logic, multi-horizon exits & summary statistics.

Classifier Provenance & Design:
    Embedded adaptation of 'keyword-algo-paircooldown.py' eliminates subprocess & stdout parsing
    fragility; constructs BreakingEvent objects directly. Maintains cooldown semantics,
    partner/entity detection, cluster grouping, sentiment filtering & price-move suppression.

Cooldown Key Strategy:
    Base key: TICKER__CLUSTER
    Partner refinement (ai_enterprise_partnership): TICKER__CLUSTER__with:PARTNER
    Group key (unless disabled): TICKER__group:GROUP (gov_involvement, ai_enterprise, mna, regulatory)
    Entity keys (unless disabled): TICKER__entity_DoD, TICKER__entity_US_GOV
    Most recent timestamp among candidate keys governs BREAKING vs REPETITION within cooldown.

State & Resets:
    Persistent JSON (ISO datetimes) at --classifier-state-file. --ignore-classifier-state keeps
    ephemeral state only. Wildcard removal via --classifier-reset-pairs e.g. NVDA:ai_*.

Filters & Suppression:
    Price-move verbs removed unless --classifier-no-ignore-price-moves.
    Sentiment narrowing via --classifier-sentiment positive,neutral,negative.
    Strict ticker mode (--classifier-strict-ticker) requires API tickers; otherwise title pattern fallback.
    Direction suppression (--classifier-suppress-direction forward|both) influences handling of
    older articles inside cooldown.

External Mode & Diagnostics:
    --use-external-classifier calls the original script. --debug-event-stats prints per-key decisions.
    --debug-parse-failures reports lines from external output that failed structured parsing.

Extended Hours Policy (when --extended-hours-mode):
    RTH window 09:30–16:00 America/New_York. Outside RTH retained only if ticker supports extended
    AND pepperstone broker membership present (from MT5 cache). Output columns include session (RTH|EXT).

Examples:
    MT5 cache universe (no IB trade simulation):
        python .\news_trade_orchestrator.py \
            --universe-from-mt5-cache .\cache\mt5_universe_cache.json \
            --months 1 --items 40 --pages 3 --skip-ib \
            --exit-horizons 5,10,30 --extended-hours-mode \
            --output events.csv --summary-output summary.csv

    Explicit tickers with trade simulation:
        python .\news_trade_orchestrator.py \
            --tickers AVGO,AMD \
            --months 2 --items 50 --pages 4 \
            --exit-horizons 5,10,15,30 --lookahead-min 30 --pre-window-sec 900 \
            --output events_sim.csv --summary-output summary.csv
"""
from __future__ import annotations
import argparse
import csv
import fnmatch
import json
import os
import re
import subprocess
import sys
import time
from datetime import timedelta
from dataclasses import dataclass  # retained for potential future use
from datetime import datetime, time as dt_time
from pathlib import Path
from typing import List, Optional, Dict, Set
from zoneinfo import ZoneInfo

import pandas as pd  # type: ignore
try:
    import requests  # type: ignore
except Exception:  # pragma: no cover - requests should already be available from external classifier usage
    requests = None  # type: ignore

# Attempt to load API key from local alpkey.env (same logic as external classifier) if not already set.
def _ensure_stocknews_api_key_loaded_lazy(base_dir: Path):
    if os.getenv('STOCKNEWS_API_KEY'):
        return
    candidate_paths = [
        base_dir / 'alpkey.env',
        base_dir.parent / 'alpkey.env',
    ]
    for p in candidate_paths:
        try:
            if not p.exists():
                continue
            for line in p.read_text(encoding='utf-8').splitlines():
                if not line or line.strip().startswith('#'):
                    continue
                if '=' not in line:
                    continue
                k, v = line.split('=', 1)
                if k.strip() == 'STOCKNEWS_API_KEY' and not os.getenv('STOCKNEWS_API_KEY'):
                    os.environ['STOCKNEWS_API_KEY'] = v.strip()
                    return
        except Exception:
            continue

HIST_TS_FILENAME = 'hist-time-sales.py'
THIS_DIR = Path(__file__).resolve().parent
HIST_TS_PATH = THIS_DIR / HIST_TS_FILENAME
CLASSIFIER_PATH = THIS_DIR / 'keyword-algo-paircooldown.py'
INDICES_DIR = THIS_DIR / 'indices'
NY_TZ = ZoneInfo('America/New_York')

_ensure_stocknews_api_key_loaded_lazy(THIS_DIR)

if not HIST_TS_PATH.exists():
    print(f"WARN: Missing {HIST_TS_FILENAME} in {THIS_DIR}", file=sys.stderr)
if not CLASSIFIER_PATH.exists():
    print(f"WARN: Missing keyword-algo-paircooldown.py in {THIS_DIR}", file=sys.stderr)


def load_hist_module():
    import importlib.util
    spec = importlib.util.spec_from_file_location('hist_time_sales', HIST_TS_PATH)
    mod = importlib.util.module_from_spec(spec)  # type: ignore
    assert spec and spec.loader
    spec.loader.exec_module(mod)  # type: ignore
    return mod


# Pattern captures:
#   kind | dt | ticker | cluster | [optional first_seen=...] | source | title | [optional url]
BREAKING_OR_REP_RE = re.compile(
    r'^(?P<kind>BREAKING|REPETITION) \| '
    r'(?P<dt>[^|]+) \| '
    r'(?P<ticker>[A-Z]{1,6}) \| '
    r'(?P<cluster>[^|]+) \| '
    r'(?:(first_seen=(?P<first_seen>[^|]+)) \| )?'
    r'(?P<source>[^|]*) \| '
    r'(?P<title>[^|]+?)'
    r'(?: \| (?P<url>\S+))?'
    r'$'
)
DATE_PARSE_FMT = '%Y-%m-%d %H:%M:%S %z'


class BreakingEvent:
    __slots__ = (
        'ticker','cluster','dt','source','title','kind','url','first_seen','staleness_min','dedup_key'
    )
    def __init__(
        self,
        ticker: str,
        cluster: str,
        dt: datetime,
        source: str,
        title: str,
        kind: str,
        url: Optional[str] = None,
        first_seen: Optional[datetime] = None,
        staleness_min: Optional[float] = None,
        dedup_key: Optional[str] = None,
    ):
        self.ticker = ticker
        self.cluster = cluster
        self.dt = dt
        self.source = source
        self.title = title
        self.kind = kind
        self.url = url
        self.first_seen = first_seen
        self.staleness_min = staleness_min
        self.dedup_key = dedup_key
    def __repr__(self) -> str:  # helpful for debugging
        return (f"BreakingEvent(ticker={self.ticker!r}, cluster={self.cluster!r}, dt={self.dt!s}, kind={self.kind})")


def parse_breaking_lines(raw: str) -> List[BreakingEvent]:
    """Parse classifier stdout capturing both BREAKING and REPETITION lines.

    Handles line wrapping introduced by some shells (where long lines are hard-wrapped,
    inserting newlines mid-record). Strategy:
      - Start a new buffer when a line begins with BREAKING | or REPETITION |
      - Concatenate subsequent non-starting lines (space-separated) until we have at
        least 6 pipe separators (expected parts) or encounter a new start line.
      - Apply regex allowing either BREAKING or REPETITION.
    """
    events: List[BreakingEvent] = []
    buffer: Optional[str] = None

    def _parse_dt(dt_raw: str) -> Optional[datetime]:
        dt_norm = dt_raw.replace('\n', ' ').replace('  ', ' ').strip()
        try:
            return datetime.strptime(dt_norm, DATE_PARSE_FMT)
        except Exception:
            return None

    failed_buffers: List[str] = []

    def try_flush(buf: Optional[str]):
        if not buf:
            return
        text = ' '.join(part.strip() for part in buf.splitlines())
        m = BREAKING_OR_REP_RE.match(text.strip())
        if m:
            dt = _parse_dt(m.group('dt').strip())
            if not dt:
                failed_buffers.append(text)
                return
            first_seen_dt: Optional[datetime] = None
            fs_token = m.group('first_seen') if 'first_seen' in m.groupdict() else None
            if fs_token:
                fs_norm = fs_token.replace('T', ' ').strip()
                fs_norm = fs_norm.replace('\n', ' ').replace('  ', ' ')
                parsed = None
                for fmt in (DATE_PARSE_FMT, '%Y-%m-%d %H:%M:%S%z', '%Y-%m-%d %H:%M:%S'):
                    try:
                        parsed = datetime.strptime(fs_norm, fmt)
                        break
                    except Exception:
                        continue
                if parsed:
                    first_seen_dt = parsed if parsed.tzinfo else dt.tzinfo and parsed.replace(tzinfo=dt.tzinfo) or None
            staleness_min: Optional[float] = None
            if first_seen_dt and dt >= first_seen_dt:
                try:
                    staleness_min = (dt - first_seen_dt).total_seconds() / 60.0
                except Exception:
                    staleness_min = None
            events.append(BreakingEvent(
                ticker=m.group('ticker').upper(),
                cluster=m.group('cluster').strip(),
                dt=dt,
                source=m.group('source').strip(),
                title=m.group('title').strip(),
                kind=m.group('kind'),
                first_seen=first_seen_dt,
                url=m.group('url') if m.group('url') else None,
                staleness_min=staleness_min,
            ))
            return
        # Fallback manual split parser (more tolerant to extra pipes / line wraps)
        raw_line = text.strip()
        if not (raw_line.startswith('BREAKING |') or raw_line.startswith('REPETITION |')):
            return
        # Split on ' | ' preserving possible pipes inside title by rejoining later
        parts = [p for p in [seg.strip() for seg in raw_line.split(' | ')] if p != '']
        if len(parts) < 6:
            failed_buffers.append(text)
            return
        kind = parts[0]
        dt_part = parts[1]
        ticker = parts[2]
        cluster = parts[3]
        idx = 4
        first_seen_dt = None
        if parts[idx].startswith('first_seen='):
            fs_val = parts[idx].split('=',1)[1]
            for fmt in (DATE_PARSE_FMT, '%Y-%m-%dT%H:%M:%S%z', '%Y-%m-%d %H:%M:%S%z', '%Y-%m-%dT%H:%M:%S', '%Y-%m-%d %H:%M:%S'):
                try:
                    parsed = datetime.strptime(fs_val.replace('T',' '), fmt)
                    if not parsed.tzinfo:
                        # assume same tz as event later
                        first_seen_dt = parsed
                    else:
                        first_seen_dt = parsed
                    break
                except Exception:
                    continue
            idx += 1
        if len(parts) <= idx+1:
            failed_buffers.append(text)
            return
        source = parts[idx]
        remaining = parts[idx+1:]
        url = None
        if remaining and remaining[-1].startswith('http'):
            url = remaining[-1]
            title_parts = remaining[:-1]
        else:
            title_parts = remaining
        title = ' | '.join(title_parts).strip()
        dt = _parse_dt(dt_part)
        if not dt:
            failed_buffers.append(text)
            return
        if first_seen_dt and first_seen_dt.tzinfo is None and dt.tzinfo:
            first_seen_dt = first_seen_dt.replace(tzinfo=dt.tzinfo)
        staleness_min = None
        if first_seen_dt and dt >= first_seen_dt:
            staleness_min = (dt - first_seen_dt).total_seconds()/60.0
        events.append(BreakingEvent(
            ticker=ticker.upper(),
            cluster=cluster,
            dt=dt,
            source=source,
            title=title,
            kind=kind,
            first_seen=first_seen_dt,
            url=url,
            staleness_min=staleness_min,
        ))

    for raw_line in raw.splitlines():
        line = raw_line.rstrip('\r\n')
        if line.startswith('BREAKING |') or line.startswith('REPETITION |'):
            # flush previous buffer
            try_flush(buffer)
            buffer = line
        else:
            if buffer is not None:
                buffer += ' ' + line
    # Flush last
    try_flush(buffer)
    return events


# ---------------- Inline classifier logic (embedded from keyword-algo-paircooldown) -----------------
# Provenance: The following code block is a direct adaptation of the core logic from the
# 'keyword-algo-paircooldown.py' script. It has been integrated into this orchestrator to
# eliminate subprocess overhead, remove parsing inconsistencies, and enable a more reliable
# in-memory simulation pipeline. This version focuses on generating BREAKING and REPETITION
# events with high fidelity to the original script's output.
#
# NOTE: This is a trimmed integration focusing on BREAKING/REPETITION generation parity. For full
# feature parity (partner/entity linking, sentiment filters, record-move suppression) ensure the
# upstream classifier file is kept in sync. This embedded version avoids subprocess overhead and
# enables direct construction of BreakingEvent objects.

# Default clusters (copied from external script). Extend here if external updated.
INLINE_DEFAULT_CLUSTERS: Dict[str, List[str]] = {
    "ai_enterprise_partnership": [
        r"\bAI\b\s+(?:strategic\s+)?partnership",
        r"(?:OpenAI|Anthropic|IBM|AMD|Nvidia|Oracle|Microsoft).*\bpartners?\b",
        r"\bpartners?\s+(?:with\s+)?(?:OpenAI|Anthropic|IBM|AMD|Nvidia|Oracle|Microsoft)",
        r"(?:OpenAI|Anthropic|IBM|AMD|Nvidia|Oracle|Microsoft|Meta).*(?:sign|announce).*(?:computing|cloud|AI)\s+(?:deal|contract)",
        r"\bcloud\b\s+(?:computing\s+)?(?:deals?|contracts?|partnerships?)",
        r"data\s+center\s+(?:deals?|contracts?)",
        r"(?:computing|cloud|AI)\s+(?:deal|contract).*(?:OpenAI|Anthropic|IBM|AMD|Nvidia|Oracle|Microsoft)",
    ],
    "government_investment": [
        r"strategic\s+investment\s+by\s+(?:the\s+)?U\.?S\.?\s+(?:federal\s+)?government",
        r"(?:Department|Dept)\s+of\s+Defense\s+(?:investment|funding|stake)",
        r"(?:U\.?S\.?|United States)\s+government\s+to\s+take\s+\d{1,3}%\s+stake",
        r"government\s+(?:takes?|taking|to\s+take)\s+\d{1,3}%\s+stake",
        r"(?:U\.?S\.?|United States)\s+(?:administration|White\s+House)\s+(?:takes?|taking|to\s+take)\s+\d{1,3}%\s+stake",
        r"(?:Biden|Trump|Obama|Bush|Clinton|Reagan|Harris)\s+(?:administration|admin)\s+(?:takes?|taking|to\s+take)\s+\d{1,3}%\s+stake",
        r"(?:U\.?S\.?|United States)\s+(?:administration|White\s+House)\s+(?:announces?|plans?|to\s+invest|investment)\b",
        r"(?:Biden|Trump|Obama|Bush|Clinton|Reagan|Harris)\s+(?:administration|admin)\s+(?:announces?|plans?|to\s+invest|investment)\b",
        r"(?:U\.?S\.?|United States)\s+(?:government|administration|admin|White\s+House).*?(?:acquire(?:s|d)?|to\s+acquire).*?(?:stake|stakes)\b",
        r"(?:Biden|Trump|Obama|Bush|Clinton|Reagan|Harris)\s+(?:administration|admin).*?(?:acquire(?:s|d)?|to\s+acquire).*?(?:stake|stakes)\b",
        r"(?:U\.?S\.?|United States)\s+(?:government|administration|admin|White\s+House).*?(?:takes?|to\s+take).*?stake\b",
        r"(?:Biden|Trump|Obama|Bush|Clinton|Reagan|Harris)\s+(?:administration|admin).*?(?:takes?|to\s+take).*?stake\b",
    ],
    "public_private_partnership": [
        r"public[-\s]*private\s+partnership",
        r"partnership\s+with\s+the\s+department\s+of\s+defense",
        r"department\s+of\s+defense\s+partnership",
        r"public[-\s]*private.*(?:department\s+of\s+defense|DoD)",
    ],
    "merger_acq": [
        r"merger\s+announced",
        r"acquisition\s+announced",
        r"merger\s+agreement",
        r"announces?\s+.*acquisition",
        # Exclude government acquisitions - use negative lookbehind to prevent matching
        # government/administration/admin + acquires/acquire/acquired
        r"(?<!government\s)(?<!administration\s)(?<!admin\s)(?:\bto\s+acquire\b|\bacquires?\b|\bacquired\s+by\b)",
        r"\bwill\s+acquire\b|\bagrees?\s+to\s+acquire\b",
        r"\btakeover\b|\bbuyout\b",
        r"\bnear\b.*\bdeal\s+to\s+go\s+private\b",
        r"(?:to\s+be\s+acquired\s+by|to\s+acquire|acquires?|acquired\s+by)[^\n]*?(?:for|worth|valued\s+at)\s+(?:up\s+to\s+)?\$[0-9,.]+\s*(?:billion|bn|B|million|m|M)",
    ],
    "fda_approval": [
        r"announces?\s+fda\s+approval",
        r"fda\s+approves",
        r"gets?\s+fda\s+approval",
    ],
    "contract_award": [
        r"\b(wins?|awarded|signs?|lands?)\s+.*contract\b",
        r"\bcontract\s+win\b",
        r"secures?\s+.*contract",
    ],
}

INLINE_DATE_FMT_STOCKNEWS = '%a, %d %b %Y %H:%M:%S %z'

def _inline_parse_article_dt(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    try:
        return datetime.strptime(s.strip(), INLINE_DATE_FMT_STOCKNEWS)
    except Exception:
        return None

def _inline_api_date_range(months: int) -> str:
    end = datetime.now()
    start = end - timedelta(days=30 * max(1, months))
    return f"{start.strftime('%m%d%Y')}-{end.strftime('%m%d%Y')}"

def _inline_build_url(params: Dict[str, str], token: Optional[str]) -> str:
    parts = [f"{k}={v}" for k,v in params.items() if v is not None]
    if token:
        parts.append(f"token={token}")
    return 'https://stocknewsapi.com/api/v1?' + '&'.join(parts)

def _inline_compile_clusters() -> Dict[str, List[re.Pattern]]:
    return {k: [re.compile(p, re.IGNORECASE) for p in pats] for k, pats in INLINE_DEFAULT_CLUSTERS.items()}

def _inline_match_clusters(title: str, clusters_re: Dict[str, List[re.Pattern]]) -> List[str]:
    hits: List[str] = []
    for name, pats in clusters_re.items():
        for cp in pats:
            if cp.search(title):
                hits.append(name)
                break
    return hits

def _inline_should_exclude_title(title: str) -> bool:
    """Return True if title contains excluded words like 'soar' and conjugations."""
    exclude_pattern = re.compile(r'\b(?:soar|soars|soared|soaring)\b', re.IGNORECASE)
    return bool(exclude_pattern.search(title))

# AI Partner detection patterns (for entity-aware repetition tracking)
INLINE_AI_PARTNER_PATTERNS: Dict[str, List[re.Pattern]] = {
    "OpenAI": [re.compile(r"\bOpenAI\b", re.IGNORECASE)],
    "IBM": [re.compile(r"\bIBM\b", re.IGNORECASE)],
    "Microsoft": [re.compile(r"\bMicrosoft\b", re.IGNORECASE), re.compile(r"\bAzure\b", re.IGNORECASE)],
    "Oracle": [re.compile(r"\bOracle\b", re.IGNORECASE)],
    "Nvidia": [re.compile(r"\bNVIDIA\b", re.IGNORECASE), re.compile(r"\bNvidia\b", re.IGNORECASE)],
    "AMD": [re.compile(r"\bAMD\b", re.IGNORECASE), re.compile(r"\bAdvanced\s+Micro\s+Devices\b", re.IGNORECASE)],
    "Anthropic": [re.compile(r"\bAnthropic\b", re.IGNORECASE)],
    "Broadcom": [re.compile(r"\bBroadcom\b", re.IGNORECASE), re.compile(r"\bAVGO\b", re.IGNORECASE)],
}


INLINE_CURATED_PARTNER_NAMES: Set[str] = {
    # AI model labs, infra, tooling
    'OpenAI','Anthropic','xAI','Cohere','Databricks','Mistral AI','Stability AI','Midjourney','Perplexity',
    'Inflection AI','Adept AI','Character.AI','Scale AI','Hugging Face','Together AI','Runway','Reka AI',
    'AI21 Labs','Weights & Biases','Pinecone','Sierra','Anysphere','Applied Intuition','Cerebras Systems',
    'VAST Data','CoreWeave','SambaNova','Amplitude','Cleanlab',
    # Space, mobility, frontier tech
    'SpaceX','Waymo','Cruise','Neuralink','Figure','Anduril','Skydio','Relativity Space','Shield AI',
    # Fintech, payments, enterprise SaaS
    'Stripe','Checkout.com','Revolut','Brex','Ramp','Deel','Personio','Bolt','Notion','Airtable','Tanium',
    'Talkdesk','Gusto','Navan','TripActions','Digital Currency Group','Ripple','KuCoin','Dunamu','Bilt Rewards',
    'Celonis','Tipalti','N26','Octopus Energy Group','Ping An Healthcare Management',
    # Consumer, commerce, social, entertainment
    'ByteDance','TikTok','Shein','Epic Games','Fanatics','Canva','JUUL Labs','Gopuff','OpenSea','Miro','Xiaohongshu',
    'Yuanfudao','Trendyol','Discord','Faire','Grammarly','Infinite Reality','Oura','Hopper','Niantic','HeyTea',
    # Health, bio, quantum
    'Devoted Health','Grail','Biosplice Therapeutics','Helsing','Colossal Biosciences','Quantinuum',
    # Energy, EV, semiconductor, crypto mining
    'Northvolt','PhonePe','Lalamove','Tata Passenger Electric Mobility','Mahindra Electric','Yangtze Memory Technologies',
    'ChangXin Memory Technologies','Bitmain','YMTC','CXMT','DJI','SenseTime','Megvii',
}

INLINE_CLUSTER_GROUPS: Dict[str, str] = {
    'public_private_partnership': 'gov_involvement',
    'government_investment': 'gov_involvement',
    'contract_award': 'gov_involvement',
    'merger_acq': 'mna',
    'fda_approval': 'regulatory',
    'ai_enterprise_partnership': 'ai_enterprise',
}

INLINE_ENTITY_PATTERNS: Dict[str, List[re.Pattern]] = {
    'entity_DoD': [
        re.compile(r"\bDepartment\s+of\s+Defense\b", re.IGNORECASE),
        re.compile(r"\bDept\.?\s+of\s+Defense\b", re.IGNORECASE),
        re.compile(r"\bDoD\b", re.IGNORECASE),
        re.compile(r"\bPentagon\b", re.IGNORECASE),
        re.compile(r"\bDefense\s+Department\b", re.IGNORECASE),
    ],
    'entity_US_GOV': [
        re.compile(r"\bU\.?S\.?\s+government\b", re.IGNORECASE),
        re.compile(r"\bUnited\s+States\s+government\b", re.IGNORECASE),
        re.compile(r"\bU\.?S\.?\s+(?:administration|admin)\b", re.IGNORECASE),
        re.compile(r"\bUnited\s+States\s+(?:administration|admin)\b", re.IGNORECASE),
        re.compile(r"\bWhite\s+House\b", re.IGNORECASE),
        re.compile(r"\b(?:Biden|Trump|Obama|Bush|Clinton|Reagan|Harris)\s+(?:administration|admin)\b", re.IGNORECASE),
    ],
}

INLINE_DOD_RE = re.compile(r"\b(?:Department\s+of\s+Defense|Dept\.?\s+of\s+Defense|DoD|Pentagon|Defense\s+Department)\b", re.IGNORECASE)

INLINE_PARTNER_RELATION_RE = re.compile(r"\b(deal|partnership|agreement|tie[-\s]?up|collaboration|pact|alliance)\b", re.IGNORECASE)

# Deal value extraction for same-deal detection (e.g., "$300 billion", "$50B")
INLINE_DEAL_VALUE_RE = re.compile(r'\$\s*([0-9,.]+)\s*(billion|bn|B|million|mn|M|trillion|tn|T)\b', re.IGNORECASE)


def _inline_extract_deal_signature(title: str) -> Optional[str]:
    """Extract deal signature from title for same-deal detection.
    
    Returns normalized deal value string like '300B' for "$300 billion" deals.
    This helps detect multiple articles about the SAME deal on the same day.
    
    EXCLUDES revenue/sales projections - only looks for actual deal amounts.
    """
    # Find all dollar amounts in the title
    for match in INLINE_DEAL_VALUE_RE.finditer(title):
        # Get the context before and after the match to check for revenue/sales keywords
        start_pos = match.start()
        end_pos = match.end()
        context_start = max(0, start_pos - 50)  # Look 50 chars before
        context_end = min(len(title), end_pos + 100)  # Look 100 chars after
        context_before = title[context_start:start_pos].lower()
        context_after = title[end_pos:context_end].lower()
        
        # Skip if this amount is associated with revenue/sales projections or speculation
        projection_keywords = [
            'revenue', 'sales', 'generate', 'worth', 'valuation',
            'market cap', 'market value', 'expected to generate',
            'could supercharge', 'could generate', 'could reach',
            'projecting', 'projected', 'analyst says', 'analyst predicts',
            'earnings boost', 'earnings impact'
        ]
        
        if any(keyword in context_before or keyword in context_after for keyword in projection_keywords):
            continue
        
        # This looks like an actual deal amount
        value_str = match.group(1).replace(',', '')
        unit = match.group(2)[0].upper()  # B/M/T
        try:
            value = float(value_str)
            # Normalize to billions for comparison
            if unit == 'M':
                value /= 1000
            elif unit == 'T':
                value *= 1000
            # Return normalized signature (first valid match)
            if value >= 1:
                return f"{int(value)}B"
            else:
                return f"{int(value * 1000)}M"
        except (ValueError, ZeroDivisionError):
            continue
    
    return None


def _inline_build_generic_partner_patterns(partner_names: Set[str]) -> Dict[str, List[re.Pattern]]:
    patterns: Dict[str, List[re.Pattern]] = {}
    for name in sorted(partner_names, key=lambda s: (-len(s), s.lower())):
        safe = re.escape(name.strip())
        if not safe:
            continue
        patterns[name] = [re.compile(rf"\b{safe}\b", re.IGNORECASE)]
    return patterns


def _inline_load_partner_map_from_csv(path: Path, existing: Optional[Set[str]] = None) -> Set[str]:
    results: Set[str] = set(existing or set())
    try:
        with path.open('r', encoding='utf-8') as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                return results
            name_field = None
            for candidate in ('name', 'partner', 'company', 'counterparty'):
                if candidate in (f.lower() for f in reader.fieldnames):
                    idx = [f.lower() for f in reader.fieldnames].index(candidate)
                    name_field = reader.fieldnames[idx]
                    break
            if name_field is None:
                name_field = reader.fieldnames[0]
            for row in reader:
                raw = row.get(name_field) if row else None
                if raw and raw.strip():
                    results.add(raw.strip())
    except Exception:
        return results
    return results


def _inline_load_partner_map_from_json(path: Path, existing: Optional[Set[str]] = None) -> Set[str]:
    results: Set[str] = set(existing or set())
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
    except Exception:
        return results
    if isinstance(data, dict):
        for names in data.values():
            if isinstance(names, list):
                for name in names:
                    if isinstance(name, str) and name.strip():
                        results.add(name.strip())
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                names = item.get('names')
                if isinstance(names, list):
                    for name in names:
                        if isinstance(name, str) and name.strip():
                            results.add(name.strip())
    return results


def _inline_load_index_constituents(code: str) -> List[str]:
    fname = None
    lc = code.strip().lower()
    if lc == 'sp500':
        fname = INDICES_DIR / 'sp500_constituents.csv'
    elif lc == 'nasdaq100':
        fname = INDICES_DIR / 'nasdaq100_constituents.csv'
    if not fname or not fname.exists():
        return []
    out: List[str] = []
    try:
        with fname.open('r', encoding='utf-8') as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                return []
            name_field = None
            for candidate in ('security', 'name', 'company'):
                if candidate in (f.lower() for f in reader.fieldnames):
                    idx = [f.lower() for f in reader.fieldnames].index(candidate)
                    name_field = reader.fieldnames[idx]
                    break
            if name_field is None:
                name_field = reader.fieldnames[0]
            for row in reader:
                raw = row.get(name_field) if row else None
                if raw and raw.strip():
                    out.append(raw.strip())
    except Exception:
        return []
    return out


def _inline_load_private_company_names(code: str) -> List[str]:
    mapping = {
        'private100': INDICES_DIR / 'private100_companies.csv',
        'cloud100': INDICES_DIR / 'cloud100_companies.csv',
        'ai100': INDICES_DIR / 'ai100_companies.csv',
    }
    fname = mapping.get(code.strip().lower())
    if not fname or not fname.exists():
        return []
    names: List[str] = []
    try:
        with fname.open('r', encoding='utf-8') as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                return []
            lower_fields = [f.lower() for f in reader.fieldnames]
            name_field = None
            for candidate in ('name', 'company', 'companyname', 'company_name'):
                if candidate in lower_fields:
                    idx = lower_fields.index(candidate)
                    name_field = reader.fieldnames[idx]
                    break
            if name_field is None:
                name_field = reader.fieldnames[0]
            for row in reader:
                raw = row.get(name_field) if row else None
                if raw and raw.strip():
                    names.append(raw.strip())
    except Exception:
        return []
    return names


def _inline_prepare_partner_patterns(
    partner_files: Optional[List[str]],
    partner_from_index: Optional[List[str]],
    partner_from_private: Optional[List[str]],
) -> Dict[str, List[re.Pattern]]:
    names: Set[str] = set(INLINE_CURATED_PARTNER_NAMES)
    names.update(INLINE_AI_PARTNER_PATTERNS.keys())
    # Load names from index shortcuts
    for code in partner_from_index or []:
        for name in _inline_load_index_constituents(code):
            names.add(name)
    # Load from private company shortcuts
    for code in partner_from_private or []:
        for name in _inline_load_private_company_names(code):
            names.add(name)
    # Load from explicit files
    for raw_path in partner_files or []:
        path = Path(raw_path)
        if not path.is_absolute():
            path = THIS_DIR / raw_path
        if not path.exists():
            continue
        if path.suffix.lower() == '.csv':
            names = _inline_load_partner_map_from_csv(path, names)
        elif path.suffix.lower() == '.json':
            names = _inline_load_partner_map_from_json(path, names)
    return _inline_build_generic_partner_patterns(names)


def _inline_detect_generic_partner(
    title: str,
    primary_ticker: str,
    partner_patterns: Dict[str, List[re.Pattern]],
) -> Optional[str]:
    """Detect partner company in title, prioritizing earliest match position."""
    if not title:
        return None
    primary_upper = (primary_ticker or '').upper()
    
    # Build exclusion list: ticker itself + known company names for common tickers
    ticker_company_map = {
        'ORCL': 'ORACLE',
        'MSFT': 'MICROSOFT',
        'GOOGL': 'GOOGLE', 'GOOG': 'GOOGLE',
        'AMZN': 'AMAZON',
        'META': 'META', 'FB': 'META',
        'AAPL': 'APPLE',
        'NVDA': 'NVIDIA',
        'TSLA': 'TESLA',
        'AMD': 'AMD',
        'INTC': 'INTEL',
        'AVGO': 'BROADCOM',
        'IBM': 'IBM',
    }
    excluded_names = {primary_upper}
    if primary_upper in ticker_company_map:
        excluded_names.add(ticker_company_map[primary_upper])
    
    # Find all matches with their positions
    matches: List[Tuple[int, str]] = []  # (position, partner_name)
    for partner_name, patterns in partner_patterns.items():
        partner_upper = partner_name.upper()
        # Skip if this partner is the primary company
        if partner_upper in excluded_names:
            continue
        for compiled in patterns:
            match = compiled.search(title)
            if match:
                matches.append((match.start(), partner_name))
                break  # Only record first match per partner
    
    if not matches:
        return None
    
    # Return the partner that appears earliest in the title
    matches.sort(key=lambda x: x[0])
    return matches[0][1]

def _inline_detect_ai_partner(title: str, primary_ticker: str) -> Optional[str]:
    """Detect AI partner company in title.
    
    Prioritizes partners mentioned near deal keywords like 'deal', 'partnership', etc.
    Falls back to earliest match if no deal keywords found.
    """
    if not title:
        return None
    pt = (primary_ticker or "").upper()
    
    # Build exclusion list: ticker + company name
    ticker_company_map = {
        'ORCL': 'ORACLE', 'MSFT': 'MICROSOFT', 'GOOGL': 'GOOGLE', 'GOOG': 'GOOGLE',
        'AMZN': 'AMAZON', 'META': 'META', 'FB': 'META', 'AAPL': 'APPLE',
        'NVDA': 'NVIDIA', 'TSLA': 'TESLA', 'AMD': 'AMD', 'INTC': 'INTEL',
        'AVGO': 'BROADCOM', 'IBM': 'IBM',
    }
    excluded_names = {pt}
    if pt in ticker_company_map:
        excluded_names.add(ticker_company_map[pt])
    
    # Find deal keywords in title
    deal_keywords_re = re.compile(
        r'\b(deal|partnership|partner|collaboration|agreement|contract|signs?|announces?)\b',
        re.IGNORECASE
    )
    deal_keyword_positions = [m.start() for m in deal_keywords_re.finditer(title)]
    
    # Find all partner matches with their positions
    matches: List[Tuple[int, str]] = []  # (position, partner_name)
    for name, pats in INLINE_AI_PARTNER_PATTERNS.items():
        if name.upper() in excluded_names:
            continue
        for p in pats:
            match = p.search(title)
            if match:
                matches.append((match.start(), name))
                break  # Only record first match per partner
    
    if not matches:
        return None
    
    # If we found deal keywords, prioritize partners near them
    if deal_keyword_positions:
        # Calculate distance from each partner to nearest deal keyword
        partner_scores = []
        for pos, partner in matches:
            min_distance = min(abs(pos - kw_pos) for kw_pos in deal_keyword_positions)
            partner_scores.append((min_distance, pos, partner))
        
        # Sort by: 1) closest to deal keyword, 2) earliest in title
        partner_scores.sort(key=lambda x: (x[0], x[1]))
        return partner_scores[0][2]
    
    # No deal keywords - use earliest match
    matches.sort(key=lambda x: x[0])
    return matches[0][1]

def classify_inline(
    tickers: List[str], months: int, items: int, pages: int,
    cooldown_days: int = 14,
    breaking_mode: bool = True,
    state_file: Optional[Path] = None,
    ignore_state: bool = False,
    verbose: bool = False,
    session: Optional["requests.sessions.Session"] = None,  # type: ignore[type-arg]
    api_key: Optional[str] = None,
    sentiment_filter: Optional[Set[str]] = None,
    strict_ticker_match: bool = False,
    no_cluster_grouping: bool = False,
    no_entity_linking: bool = False,
    no_ignore_price_moves: bool = False,
    partner_files: Optional[List[str]] = None,
    partner_from_index: Optional[List[str]] = None,
    partner_from_private: Optional[List[str]] = None,
    reset_pairs: Optional[List[str]] = None,
    suppress_direction: str = 'forward',
) -> List[BreakingEvent]:
    """Fetch articles & classify into BREAKING / REPETITION with partner/entity awareness."""
    if not tickers:
        return []
    if api_key is None:
        api_key = os.getenv('STOCKNEWS_API_KEY')
    if not api_key:
        print('WARN: STOCKNEWS_API_KEY missing for inline classifier; returning empty set', file=sys.stderr)
        return []
    sess = session or requests.Session() if requests else None  # type: ignore
    clusters_re = _inline_compile_clusters()
    sentiment_filter_set = {s.strip().lower() for s in sentiment_filter} if sentiment_filter else None
    partner_patterns = _inline_prepare_partner_patterns(partner_files, partner_from_index, partner_from_private)
    for name, pats in INLINE_AI_PARTNER_PATTERNS.items():
        partner_patterns.setdefault(name, list(pats))

    ignore_price_moves = not no_ignore_price_moves
    price_move_re = None
    record_move_re = None
    record_exception_re = None
    generic_fluff_re = None
    if ignore_price_moves:
        verbs = (
            r"soar(?:s|ed|ing)?|surge(?:s|d|ing)?|jump(?:s|ed|ing)?|spike(?:s|d|ing)?|"
            r"rocket(?:s|ed|ing)?|skyrocket(?:s|ed|ing)?|rally(?:ies|ed|ing)?|pop(?:s|ped|ping)?|"
            r"leap(?:s|ed|ing)?|rip(?:s|ped|ping)?|moon(?:s|ed|ing)?|plunge(?:s|d|ing)?|tumble(?:s|d|ing)?|"
            r"crater(?:s|ed|ing)?|tank(?:s|ed|ing)?|climb(?:s|ed|ing)?|gain(?:s|ed|ing)?|"
            r"rise(?:s|rising)?|slide(?:s|d|ing)?|slid|fall(?:s|ing)?|fell|"
            r"advance(?:s|d|ing)?|decline(?:s|d|ing)?|sink(?:s|ing)?|dip(?:s|ped|ping)?|slump(?:s|ed|ing)?|"
            r"rebound(?:s|ed|ing)?|retreat(?:s|ed|ing)?|plummet(?:s|ed|ing)?"
        )
        ctx = r"(?:stock(?:s)?|share(?:s)?|price|stock\s+price|share\s+price)"
        pattern_1 = rf"\b(?:{ctx}\s+)?(?:{verbs})\b"
        pattern_2 = rf"\b(?:{ctx}\s+)?(?:up|down)\s+\d{{1,3}}%\b"
        pattern_3 = r"\b(?:stock(?:s)?|share(?:s)?)\s+(?:is|are|was|were)\s+(?:rising|falling|higher|lower|up|down)\b"
        pattern_4 = r"\bedge(?:s|d|ing)?\s+(?:higher|lower)\b"
        price_move_re = re.compile("|".join([pattern_1, pattern_2, pattern_3, pattern_4]), re.IGNORECASE)
        record_move_re = re.compile(r"\b(?:hits?|reach(?:es|ed)?|touches?|at)\s+(?:fresh\s+)?(?:record|all[-\s]?time)\s+(?:high|low)\b", re.IGNORECASE)
        record_exception_re = re.compile(r"\b(revenue|estimate(?:s)?|earnings)\b", re.IGNORECASE)
        # Exclude generic fluff headlines that don't announce specific deals
        generic_fluff_re = re.compile(
            r"\b\d+\s+Reasons?\s+Why\b|"  # "3 Reasons Why..."
            r"\bLaunches\s+New\s+Customer\s+and\s+Partner\s+Offerings\b|"  # Generic product launches
            r"\bHere'?s\s+Why\b|"  # "Here's Why..."
            r"\bWhat\s+(?:You|Investors)\s+(?:Need|Should)\s+(?:to\s+)?Know\b|"  # "What You Need to Know"
            r"\bShould\s+You\s+(?:Buy|Sell)\b|"  # Investment advice clickbait (Buy/Sell)
            r"\bBetter\s+Buy\b|"  # "Better Buy" comparison articles
            r"\bWhy\s+I'?m\s+(?:Buying|Selling)\b|"  # Opinion pieces
            r"\bstocks?\s+bounce\b|"  # "stocks bounce", "stock bounces"
            r"\bJim\s+Cramer\b|"  # Jim Cramer opinion pieces
            r"\boverreacting\b|"  # "overreacting" opinion/analysis
            r"\brisk\b|"  # "risk" - risk analysis articles
            r"\b(?:NVIDIA|NVDA|Intel|INTC|Broadcom|AVGO)\s+CEO\b|"  # Competitor CEO commentary
            r"\b(?:NVIDIA|NVDA|Intel|INTC|Broadcom|AVGO)'?s\s+(?:CEO|Huang|Gelsinger|Tan)\b|"  # "Nvidia's Huang says...", "Intel's CEO..."
            r"\b(?:Empire|Plot\s+Twist|Takes?\s+(?:a\s+)?Hit|Took\s+(?:a\s+)?Hit)\b|"  # Sensationalist language
            r"\b(?:Challenges?\s+Loom|Funding\s+Challenges?)\b",  # Commentary about deal challenges
            re.IGNORECASE
        )

    params_base = {
        'tickers': ','.join(tickers),
        'date': _inline_api_date_range(months),
        'items': str(int(items)),
    }
    all_articles: List[Dict[str, object]] = []
    for page in range(1, pages + 1):
        qp = dict(params_base)
        qp['page'] = str(page)
        url = _inline_build_url(qp, api_key)
        try:
            if not sess:  # requests missing
                break
            r = sess.get(url, timeout=30)
            if r.status_code != 200:
                if verbose:
                    print(f"INLINE API {r.status_code}: {url}", file=sys.stderr)
                break
            data = r.json()
            arts = data.get('data', []) if isinstance(data, dict) else []
            if not arts:
                break
            all_articles.extend(arts)
        except Exception as e:  # pragma: no cover
            print(f"WARN: inline fetch failed page {page}: {e}", file=sys.stderr)
            break
        time.sleep(0.3)  # light pacing
    
    # Sort oldest-to-newest (chronological order)
    def _dt(a: Dict[str, object]) -> float:
        dt = _inline_parse_article_dt(a.get('date')) if isinstance(a, dict) else None
        return dt.timestamp() if dt else 0.0
    all_articles.sort(key=_dt)  # ascending = oldest first
    
    # Load state file (persistent cooldown tracking)
    state: Dict[str, str] = {}
    if state_file and (not ignore_state) and state_file.exists():
        try:
            state = json.loads(state_file.read_text(encoding='utf-8'))
        except Exception:
            state = {}

    if reset_pairs and state and (not ignore_state):
        for spec in reset_pairs:
            if not spec or ':' not in spec:
                continue
            ticker_part, pattern_part = spec.split(':', 1)
            prefix = f"{ticker_part.strip().upper()}__"
            suffix_pattern = (pattern_part or '').strip()
            if not prefix.strip() or not suffix_pattern:
                continue
            for key in list(state.keys()):
                if key.startswith(prefix):
                    suffix = key[len(prefix):]
                    if fnmatch.fnmatch(suffix, suffix_pattern):
                        state.pop(key, None)
    
    cooldown = timedelta(days=max(1, cooldown_days))
    run_last_seen: Dict[str, datetime] = {}  # for ignore_state mode
    out: List[BreakingEvent] = []
    requested_set = {t.upper() for t in tickers}
    suppress_dir = (suppress_direction or 'forward').lower()
    
    for art in all_articles:
        if not isinstance(art, dict):
            continue
        title = (art.get('title') or '').strip()
        if not title:
            continue
        
        if sentiment_filter_set:
            sent_value = (art.get('sentiment') or '').strip().lower()
            if sent_value not in sentiment_filter_set:
                continue

        # Exclude generic fluff headlines
        if ignore_price_moves and generic_fluff_re and generic_fluff_re.search(title):
            if verbose:
                print(f"[DEBUG] EXCLUDED generic-fluff: \"{title}\"", file=sys.stderr, flush=True)
            continue

        # Exclude titles with "soar" and conjugations
        if ignore_price_moves:
            if price_move_re and price_move_re.search(title):
                if verbose:
                    print(f"[DEBUG] EXCLUDED price-move: \"{title}\"", file=sys.stderr, flush=True)
                continue
            if record_move_re and record_move_re.search(title):
                if not (record_exception_re and record_exception_re.search(title)):
                    if verbose:
                        print(f"[DEBUG] EXCLUDED record-high: \"{title}\"", file=sys.stderr, flush=True)
                    continue
            if price_move_re is None and _inline_should_exclude_title(title):
                if verbose:
                    print(f"[DEBUG] EXCLUDED (contains 'soar'): \"{title}\"", file=sys.stderr, flush=True)
                continue
        
        art_dt = _inline_parse_article_dt(art.get('date')) or datetime.now().astimezone()
        
        # Determine tickers intersection
        raw_tickers = art.get('tickers') or art.get('symbols') or art.get('tickers_list') or []
        art_tickers: List[str] = []
        if isinstance(raw_tickers, list):
            for entry in raw_tickers:
                if isinstance(entry, str):
                    sym = entry.strip().upper()
                    if sym and sym in requested_set:
                        art_tickers.append(sym)
        if not art_tickers and not strict_ticker_match:
            guesses = [w for w in re.findall(r'\b[A-Z]{1,5}\b', title) if w in requested_set]
            art_tickers = guesses
        if not art_tickers:
            continue
        
        clusters_hit = _inline_match_clusters(title, clusters_re)
        if not clusters_hit:
            # Fallback: detect ai partnership if partner + relation keyword present
            if INLINE_PARTNER_RELATION_RE.search(title):
                for tk in art_tickers:
                    partner_guess = _inline_detect_generic_partner(title, tk, partner_patterns) or _inline_detect_ai_partner(title, tk)
                    if partner_guess:
                        clusters_hit.append('ai_enterprise_partnership')
                        break
        if not clusters_hit:
            continue
        
        source = art.get('source_name') or art.get('source') or ''
        
        # Skip sources known for clickbait/opinion/analysis content
        if source and any(excluded.lower() in source.lower() for excluded in ['Motley Fool', 'Seeking Alpha', 'Zacks Investment Research']):
            if verbose:
                print(f"[DEBUG] EXCLUDED source: \"{title[:80]}...\" (Source: {source})", file=sys.stderr, flush=True)
            continue
        
        url_a = art.get('news_url') or art.get('url') or None

        entity_tokens: Set[str] = set()
        if not no_entity_linking:
            if INLINE_DOD_RE.search(title):
                entity_tokens.add('entity_DoD')
            for entity_key, patterns in INLINE_ENTITY_PATTERNS.items():
                if entity_key == 'entity_DoD':  # already handled above
                    continue
                if any(p.search(title) for p in patterns):
                    entity_tokens.add(entity_key)
        
        for tk in art_tickers:
            for cl in clusters_hit:
                # Base key
                key = f"{tk}__{cl}"
                
                # For AI partnerships, refine by partner entity
                partner_key: Optional[str] = None
                deal_signature: Optional[str] = None
                detected_partner: Optional[str] = None
                if cl == 'ai_enterprise_partnership':
                    detected_partner = _inline_detect_generic_partner(title, tk, partner_patterns) or _inline_detect_ai_partner(title, tk)
                    if detected_partner:
                        # Extract deal value signature if present (e.g., "300B" from "$300 billion")
                        deal_signature = _inline_extract_deal_signature(title)
                        if verbose and deal_signature:
                            print(f"[DEBUG] Deal signature detected: {deal_signature} in \"{title[:80]}...\"", file=sys.stderr, flush=True)
                        if deal_signature:
                            # Include deal value in key for same-deal detection
                            partner_key = f"{tk}__{cl}__with:{detected_partner.upper()}__deal:{deal_signature}"
                        else:
                            # No specific deal value, use partner-only key
                            partner_key = f"{tk}__{cl}__with:{detected_partner.upper()}"
                
                # Primary key to use
                primary_key = partner_key or key
                
                # Check last seen timestamp
                last_seen: Optional[datetime] = None
                last_seen_iso: Optional[str] = None
                candidate_keys = [primary_key]
                
                # For deal-specific keys, also check the non-deal partner key
                # This prevents articles without deal values from being BREAKING
                # when a deal-specific article already exists (e.g., "$300B" article exists,
                # then generic "OpenAI-Oracle deal" article appears - should be REPETITION)
                if detected_partner and deal_signature:
                    # Add the non-deal partner key to candidate keys
                    non_deal_partner_key = f"{tk}__{cl}__with:{detected_partner.upper()}"
                    if non_deal_partner_key not in candidate_keys:
                        candidate_keys.append(non_deal_partner_key)
                
                if not no_cluster_grouping and not partner_key:
                    group_name = INLINE_CLUSTER_GROUPS.get(cl)
                    if group_name:
                        candidate_keys.append(f"{tk}__group:{group_name}")
                if entity_tokens and not no_entity_linking:
                    candidate_keys.extend([f"{tk}__{token}" for token in entity_tokens])

                if ignore_state:
                    for ckey in candidate_keys:
                        ts = run_last_seen.get(ckey)
                        if ts and (last_seen is None or ts > last_seen):
                            last_seen = ts
                else:
                    for ckey in candidate_keys:
                        iso = state.get(ckey)
                        if not iso:
                            continue
                        try:
                            ts = datetime.fromisoformat(iso)
                        except Exception:
                            continue
                        if ts and (last_seen is None or ts > last_seen):
                            last_seen = ts
                            last_seen_iso = iso
                
                # Determine if cooldown has elapsed
                allow_breaking = False
                delta = None
                if last_seen is None:
                    allow_breaking = True  # First occurrence
                else:
                    try:
                        delta = art_dt - last_seen
                    except Exception:
                        delta = None
                    if delta is not None and delta.total_seconds() < 0:
                        # Article is OLDER than last_seen (backwards in time)
                        # This happens when state has newer articles from previous runs
                        # For partner-specific keys, allow BREAKING for older articles too
                        # (they represent distinct partnership announcements even if discovered later)
                        if partner_key:
                            allow_breaking = True  # Partner-specific events always BREAKING
                        else:
                            allow_breaking = (suppress_dir == 'both')  # Generic events respect suppress_direction
                    else:
                        # Article is NEWER than or same time as last_seen
                        # For same-day events, apply full-day cooldown (24 hours) regardless of partner
                        # This prevents multiple articles about the SAME deal on the same day from all being BREAKING
                        # Example: "OpenAI-Oracle $300B deal" at 14:09, 14:22, 15:00, 15:12, 16:10 (all same day, same deal)
                        same_day_cooldown = timedelta(hours=24)
                        # Normalize to same timezone before date comparison to avoid aware/naive mismatch
                        art_date = art_dt.astimezone(NY_TZ).date() if art_dt.tzinfo else art_dt.date()
                        last_seen_date = last_seen.astimezone(NY_TZ).date() if last_seen.tzinfo else last_seen.date()
                        is_same_day = (art_date == last_seen_date)
                        effective_cooldown = same_day_cooldown if is_same_day else cooldown
                        allow_breaking = bool(delta and delta.total_seconds() >= 0 and delta >= effective_cooldown)
                
                if verbose:
                    same_day_cooldown = timedelta(hours=24)
                    # Normalize to same timezone before date comparison
                    if last_seen:
                        art_date = art_dt.astimezone(NY_TZ).date() if art_dt.tzinfo else art_dt.date()
                        last_seen_date = last_seen.astimezone(NY_TZ).date() if last_seen.tzinfo else last_seen.date()
                        is_same_day = (art_date == last_seen_date)
                        effective_cooldown = same_day_cooldown if is_same_day else cooldown
                    else:
                        is_same_day = False
                        effective_cooldown = cooldown
                    print(f"\n[DEBUG] {tk}|{cl} ({primary_key})", file=sys.stderr, flush=True)
                    print(f"  - Article:    {art_dt.isoformat()}", file=sys.stderr, flush=True)
                    print(f"  - Last Seen:  {last_seen.isoformat() if last_seen else 'Never'}", file=sys.stderr, flush=True)
                    if last_seen:
                        try:
                            dbg_delta = art_dt - last_seen
                            same_day = art_dt.date() == last_seen.date()
                            print(f"  - Delta:      {dbg_delta} (Same Day: {same_day}, Effective Cooldown: {effective_cooldown})", file=sys.stderr, flush=True)
                        except Exception:
                            pass
                    print(f"  - Decision:   {'BREAKING' if allow_breaking else 'REPETITION'}", file=sys.stderr, flush=True)
                
                if allow_breaking:
                    # BREAKING: update state
                    if ignore_state:
                        # For in-memory state, update with EARLIEST timestamp (handles backward discovery)
                        for ckey in candidate_keys:
                            existing = run_last_seen.get(ckey)
                            if existing is None or art_dt < existing:
                                run_last_seen[ckey] = art_dt
                        # Also update non-deal partner key if this is a deal-specific event
                        if detected_partner and deal_signature:
                            non_deal_key = f"{tk}__{cl}__with:{detected_partner.upper()}"
                            existing = run_last_seen.get(non_deal_key)
                            if existing is None or art_dt < existing:
                                run_last_seen[non_deal_key] = art_dt
                    else:
                        # For persistent state, update with EARLIEST timestamp
                        # This ensures that if we discover an older article later (e.g., Sept 10 after Sept 15),
                        # we correctly record Sept 10 as the first occurrence
                        iso_value = art_dt.isoformat()
                        for ckey in candidate_keys:
                            existing_iso = state.get(ckey)
                            if existing_iso:
                                try:
                                    existing_dt = datetime.fromisoformat(existing_iso)
                                    # Only update if current article is EARLIER
                                    if art_dt < existing_dt:
                                        state[ckey] = iso_value
                                except Exception:
                                    state[ckey] = iso_value
                            else:
                                state[ckey] = iso_value
                        # Also update non-deal partner key if this is a deal-specific event
                        if detected_partner and deal_signature:
                            non_deal_key = f"{tk}__{cl}__with:{detected_partner.upper()}"
                            existing_iso = state.get(non_deal_key)
                            if existing_iso:
                                try:
                                    existing_dt = datetime.fromisoformat(existing_iso)
                                    if art_dt < existing_dt:
                                        state[non_deal_key] = iso_value
                                except Exception:
                                    state[non_deal_key] = iso_value
                            else:
                                state[non_deal_key] = iso_value
                    
                    event = BreakingEvent(
                        ticker=tk.upper(),
                        cluster=cl,
                        dt=art_dt,
                        source=str(source),
                        title=title,
                        kind='BREAKING',
                        url=url_a,
                        first_seen=art_dt,
                        staleness_min=None,
                        dedup_key=primary_key,
                    )
                    out.append(event)
                else:
                    # REPETITION: don't update state (keep original timestamp)
                    staleness = None
                    if delta and delta.total_seconds() >= 0:
                        staleness = delta.total_seconds() / 60.0
                    event = BreakingEvent(
                        ticker=tk.upper(),
                        cluster=cl,
                        dt=art_dt,
                        source=str(source),
                        title=title,
                        kind='REPETITION',
                        url=url_a,
                        first_seen=last_seen,
                        staleness_min=staleness,
                        dedup_key=primary_key,
                    )
                    out.append(event)
                    if ignore_state and last_seen:
                        for ckey in candidate_keys:
                            run_last_seen.setdefault(ckey, last_seen)
    
    # Persist state file
    if state_file and (not ignore_state):
        try:
            state_file.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding='utf-8')
        except Exception as e:  # pragma: no cover
            print(f"WARN: could not write inline classifier state: {e}", file=sys.stderr)
    
    return out


def run_external_classifier(tickers: List[str], months: int, items: int, pages: int, extra_args: List[str]) -> List[BreakingEvent]:
    if not CLASSIFIER_PATH.exists() or not tickers:
        return []
    base_cmd = [sys.executable, str(CLASSIFIER_PATH), '--tickers', ','.join(tickers), '--months', str(months), '--items', str(items), '--pages', str(pages), '--breaking-mode']
    base_cmd.extend(extra_args)
    proc = subprocess.run(base_cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"Classifier exited with code {proc.returncode}", file=sys.stderr)
        if proc.stderr:
            print(proc.stderr[:500], file=sys.stderr)
    return parse_breaking_lines(proc.stdout)


def simulate_trade_for_event(mod, event: BreakingEvent, lookahead_min: int, pre_window_sec: int, what: str, use_rth: int, pause_sec: float, ports: List[int], client_id: int, horizons: List[int]) -> Dict[str, object]:
    end_dt = event.dt + pd.Timedelta(minutes=lookahead_min)
    window_sec = pre_window_sec + (lookahead_min * 60)
    dt_str = end_dt.isoformat()

    IB = getattr(mod, 'IB')
    Stock = getattr(mod, 'Stock')
    parse_times_to_intervals = getattr(mod, 'parse_times_to_intervals')
    fetch_ticks = getattr(mod, 'fetch_ticks')
    connect_any = getattr(mod, 'connect_any')

    ib = IB()
    port = connect_any(ib, ports, client_id)
    if not port:
        raise RuntimeError('Could not connect to IB on provided ports')
    try:
        intervals = parse_times_to_intervals([dt_str], window_sec, 'America/New_York')
        contract = Stock(event.ticker, 'SMART', 'USD')
        df = fetch_ticks(ib=ib, contract=contract, merged=intervals, what=what, use_rth=use_rth, target_tz='America/New_York', pause_sec=pause_sec, max_requests=0)
    finally:
        if ib.isConnected():
            ib.disconnect()
    if df.empty:
        base = {
            'ticker': event.ticker,
            'cluster': event.cluster,
            'event_time': event.dt.isoformat(),
            'title': event.title,
            'source': event.source,
            'entry_time': None,
            'entry_price': None,
            'exit_time': None,
            'exit_price': None,
            'pct_return': None,
            'rows': 0,
            'pre_price': None,
        }
        for h in horizons:
            base[f'ret_{h}m'] = None
        return base

    df['time_dt'] = pd.to_datetime(df['time'])
    
    # Adaptive entry: first trade at event+1min AND volume still active
    # Simple volume check for tick data (no vol_z needed)
    entry_target = event.dt + pd.Timedelta(minutes=1)
    
    # Calculate volume baseline from first 2 minutes after event (initial surge)
    initial_window = (df['time_dt'] >= event.dt) & (df['time_dt'] < event.dt + pd.Timedelta(seconds=120))
    initial_ticks = df[initial_window]
    avg_size_initial = initial_ticks['size'].mean() if 'size' in initial_ticks and len(initial_ticks) > 0 else 0
    
    # Filter future data after entry target
    mask_future = (df['time_dt'] >= entry_target) & (df['time_dt'] <= end_dt)
    future_df_all = df[mask_future].copy().sort_values('time_dt')
    
    # For each potential entry point, check if volume is still active
    # Use rolling 30-tick average volume vs initial average
    entry_candidates = []
    for idx in range(len(future_df_all)):
        current_time = future_df_all.iloc[idx]['time_dt']
        # Get recent 30 ticks before this point
        recent_window = (df['time_dt'] >= current_time - pd.Timedelta(seconds=30)) & (df['time_dt'] <= current_time)
        recent_ticks = df[recent_window]
        avg_size_recent = recent_ticks['size'].mean() if 'size' in recent_ticks and len(recent_ticks) > 0 else 0
        
        # Volume ratio check: must be >20% of initial surge
        volume_ratio = avg_size_recent / avg_size_initial if avg_size_initial > 0 else 0
        
        if volume_ratio > 0.20:  # Volume still active
            entry_candidates.append(idx)
            break  # Take first valid entry
    
    # If we found valid entry candidates, use the first one
    if entry_candidates:
        future_df = future_df_all.iloc[entry_candidates[0]:].copy()
        first_entry_idx = entry_candidates[0]
        entry_time_check = future_df_all.iloc[first_entry_idx]['time_dt']
        # Calculate volume ratio at entry for diagnostics
        recent_window = (df['time_dt'] >= entry_time_check - pd.Timedelta(seconds=30)) & (df['time_dt'] <= entry_time_check)
        recent_ticks = df[recent_window]
        avg_size_recent = recent_ticks['size'].mean() if 'size' in recent_ticks and len(recent_ticks) > 0 else 0
        entry_volume_ratio = avg_size_recent / avg_size_initial if avg_size_initial > 0 else 0
    else:
        future_df = pd.DataFrame()  # No valid entry
        entry_volume_ratio = 0
    
    base_df = df[df['time_dt'] < event.dt]
    pre_price = base_df['price'].iloc[-1] if 'price' in base_df and not base_df.empty else None
    if future_df.empty:
        base = {
            'ticker': event.ticker,
            'cluster': event.cluster,
            'event_time': event.dt.isoformat(),
            'title': event.title,
            'source': event.source,
            'entry_time': None,
            'entry_price': None,
            'exit_time': None,
            'exit_price': None,
            'pct_return': None,
            'rows': len(df),
            'pre_price': pre_price,
            'entry_reject_reason': f'No volume (ratio={entry_volume_ratio:.2f}, need >0.20)',
        }
        for h in horizons:
            base[f'ret_{h}m'] = None
        return base

    entry_row = future_df.iloc[0]
    entry_price = entry_row.get('price')
    entry_time = entry_row['time_dt']

    # Pattern classification: measure first 60s after entry
    window_60s = (df['time_dt'] >= entry_time) & (df['time_dt'] < entry_time + pd.Timedelta(seconds=60))
    ticks_60s = df[window_60s]
    
    # Price change in first 60s
    price_at_60s = ticks_60s['price'].iloc[-1] if len(ticks_60s) > 0 else entry_price
    price_change_60s = (price_at_60s - entry_price) / entry_price if entry_price else 0
    
    # Volume metrics: compare first 60s to initial 2-min baseline
    volume_60s = ticks_60s['size'].sum() if 'size' in ticks_60s else 0
    volume_ratio = volume_60s / (avg_size_initial * 60) if avg_size_initial > 0 else 0  # Normalize to 60 ticks

    # Adaptive strategy selection
    if price_change_60s > 0.025 and volume_ratio > 5:
        minimum_hold = 30
        maximum_hold = 120
        strategy = "Volume Exhaustion"
        pattern_type = "EXPLOSIVE_SPIKE"
    elif price_change_60s > 0.005 and volume_ratio > 1.5:
        minimum_hold = 300
        maximum_hold = None  # No max, trailing stop
        strategy = "Trailing Stop"
        pattern_type = "SUSTAINED_TREND"
    else:
        minimum_hold = 180
        maximum_hold = 180
        strategy = "Time-based exit"
        pattern_type = "WEAK_MOMENTUM"

    # Exit logic placeholder: actual strategy (A,B,C,D,E) should be implemented here
    # For now, use last row in future_df as exit
    exit_row = future_df.iloc[-1]
    exit_price = exit_row.get('price')
    pct_return = None
    if entry_price and exit_price and entry_price != 0:
        pct_return = (exit_price - entry_price) / entry_price * 100.0

    def ret_at(minutes: int):
        # Exit horizons measured from entry time (news + 1 min + horizon)
        target = entry_time + pd.Timedelta(minutes=minutes)
        sub = future_df[future_df['time_dt'] >= target]
        if sub.empty or not entry_price:
            return None
        p = sub.iloc[0].get('price')
        if not p or entry_price == 0:
            return None
        return (p - entry_price) / entry_price * 100.0

    row = {
        'ticker': event.ticker,
        'cluster': event.cluster,
        'event_time': event.dt.isoformat(),
        'title': event.title,
        'source': event.source,
        'entry_time': entry_row['time'],
        'entry_price': entry_price,
        'entry_volume_ratio': entry_volume_ratio,  # Volume health at entry
        'exit_time': exit_row['time'],
        'exit_price': exit_price,
        'pct_return': pct_return,
        'rows': len(future_df),
        'pre_price': pre_price,
        'pattern_type': pattern_type,
        'strategy': strategy,
        'minimum_hold': minimum_hold,
        'maximum_hold': maximum_hold,
        'price_change_60s': price_change_60s * 100,  # Convert to percentage
        'volume_ratio_60s': volume_ratio,
        'avg_size_initial': avg_size_initial,  # Diagnostics
    }
    for h in horizons:
        row[f'ret_{h}m'] = ret_at(h)
    return row


from typing import Tuple  # added for proper tuple return annotation

def load_mt5_cache(path: Path, broker_filter: Optional[Set[str]]) -> Tuple[List[str], Set[str], Dict[str, Dict[str, object]]]:
    if not path.exists():
        raise FileNotFoundError(f'MT5 cache not found: {path}')
    data = json.loads(path.read_text(encoding='utf-8'))
    # Backward compatibility (old format: tickers + extended_hours) vs new possible richer formats
    by_ticker: Dict[str, Dict[str, object]] = data.get('combined', {}).get('by_ticker') or {}
    if not by_ticker:
        tickers = data.get('tickers', [])
        ext = set(data.get('extended_hours', []))
        by_ticker = {t: {'brokers': ['unknown'], 'extended': (t in ext)} for t in tickers}
    else:
        # Build ext set from structure if not top-level
        ext = {t for t, meta in by_ticker.items() if meta.get('extended')}
    # Apply broker filter
    if broker_filter:
        filtered = {}
        for t, meta in by_ticker.items():
            brokers = set(meta.get('brokers', []))
            if not brokers or brokers & broker_filter:
                filtered[t] = meta
        by_ticker = filtered
    tickers_final = sorted(by_ticker.keys())
    ext_final = {t for t in tickers_final if by_ticker[t].get('extended')}
    return tickers_final, ext_final, by_ticker


def is_rth(ts: datetime) -> bool:
    local = ts.astimezone(NY_TZ)
    if local.weekday() >= 5:  # weekend
        return False
    return dt_time(9,30) <= local.time() <= dt_time(16,0)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description='Fetch BREAKING news and simulate multi-horizon reaction trades (IBKR) with optional MT5 extended-hours universe logic.')
    ap.add_argument('--tickers', help='Comma list of tickers (optional if universe specified)')
    ap.add_argument('--universe-from-index', help='Comma list: sp500,nasdaq100')
    ap.add_argument('--universe-from-mt5-cache', help='Path to mt5_universe_cache.json to derive universe')
    ap.add_argument('--mt5-cache-brokers', default='admirals,pepperstone', help='Filter MT5 cache brokers (comma, default both); ignored if cache lacks broker info')
    ap.add_argument('--extended-hours-mode', action='store_true', help='Enable extended-hours eligibility logic based on MT5 cache (Pepperstone extended only)')
    ap.add_argument('--months', type=int, default=3)
    ap.add_argument('--items', type=int, default=50)
    ap.add_argument('--pages', type=int, default=5)
    ap.add_argument('--lookahead-min', type=int, default=30)
    ap.add_argument('--pre-window-sec', type=int, default=900)
    ap.add_argument('--what', default='TRADES')
    ap.add_argument('--use-rth', type=int, choices=[0,1], default=0)
    ap.add_argument('--pause-sec', type=float, default=2.5)
    ap.add_argument('--ports', default='7497,7496,4002,4001')
    ap.add_argument('--client-id', type=int, default=8801)
    ap.add_argument('--output', default='news_trade_events.csv')
    ap.add_argument('--summary-output', help='Write aggregated horizon summary CSV')
    ap.add_argument('--exit-horizons', default='5,10,15', help='Comma minute horizons')
    ap.add_argument('--price-threshold', type=float, default=100.0)
    ap.add_argument('--chunk-size', type=int, default=5)
    ap.add_argument('--extra-classifier-args', help='Extra args forwarded to classifier script')
    ap.add_argument('--skip-ib', action='store_true')
    ap.add_argument('--include-repetition', action='store_true', help='If set, include REPETITION events; default is to keep only BREAKING events')
    ap.add_argument('--use-mt5-prices', action='store_true', help='Fetch MT5 M1 bars to simulate returns (useful with --skip-ib or as fallback)')
    ap.add_argument('--mt5-pre-window-sec', type=int, default=900, help='Pre-event window seconds for MT5 bar fetching (default 900)')
    ap.add_argument('--no-ib-batch-per-ticker', action='store_true', help='Disable per-ticker batched IB fetch (revert to one fetch per event)')
    ap.add_argument('--fallback-repetition-if-none', action='store_true', help='If no BREAKING events parsed, fallback to using REPETITION events instead of exiting')
    ap.add_argument('--debug-event-stats', action='store_true', help='Print counts of BREAKING/REPETITION before and after filtering')
    ap.add_argument('--classifier-chronological-breaking', action='store_true', help='Mark earliest article per cluster as BREAKING regardless of state (based on publication time, not when you saw it)')
    ap.add_argument('--classifier-path', help='Override path to keyword-algo-paircooldown.py (will NOT be modified)')
    # Inline classifier specific flags
    ap.add_argument('--use-external-classifier', action='store_true', help='Force using external classifier script via subprocess instead of inline logic')
    ap.add_argument('--cooldown-days', type=int, default=14, help='Cooldown days for (ticker,cluster) pairs in inline mode')
    ap.add_argument('--classifier-state-file', default=str(THIS_DIR / 'inline_pair_cooldown_state.json'), help='State file path for inline classifier cooldown persistence')
    ap.add_argument('--ignore-classifier-state', action='store_true', help='Do not load/save state for inline classifier (in-run only)')
    ap.add_argument('--max-staleness-min', type=float, help='If set, drop repetition events whose staleness (minutes) exceeds this value')
    ap.add_argument('--classifier-sentiment', help='Comma list of sentiments to include (positive,negative,neutral) when using inline classifier')
    ap.add_argument('--classifier-strict-ticker', action='store_true', help='Inline classifier: require API-provided tickers (no title fallback)')
    ap.add_argument('--classifier-no-cluster-grouping', action='store_true', help='Inline classifier: disable grouping of clusters under higher-level themes')
    ap.add_argument('--classifier-no-entity-linking', action='store_true', help='Inline classifier: disable entity-linking keys (e.g., DoD detection)')
    ap.add_argument('--classifier-no-ignore-price-moves', action='store_true', help='Inline classifier: do not filter price-move headlines (e.g., stock jumps)')
    ap.add_argument('--classifier-partner-files', help='Inline classifier: comma CSV/JSON files describing ticker->partner names (same format as external script)')
    ap.add_argument('--classifier-partner-from-index', help='Inline classifier: comma index shortcodes (sp500,nasdaq100) for partner names')
    ap.add_argument('--classifier-partner-from-private', help='Inline classifier: comma private-company shortcodes (private100,cloud100,ai100)')
    ap.add_argument('--classifier-reset-pairs', help='Inline classifier: comma list of TICKER:pattern wildcards to delete from state (e.g., NVDA:ai_*). Ignored with --ignore-classifier-state')
    ap.add_argument('--classifier-suppress-direction', choices=['forward','both'], default='forward', help="Inline classifier: 'forward' ignores older repeats; 'both' suppresses older and newer within cooldown")
    ap.add_argument('--debug-parse-failures', action='store_true', help='External classifier mode: show count/sample of lines that failed parsing')
    args = ap.parse_args(argv)

    tickers: List[str] = []
    ticker_meta: Dict[str, Dict[str, object]] = {}
    extended_set: Set[str] = set()

    if args.universe_from_mt5_cache:
        broker_filter = {b.strip().lower() for b in args.mt5_cache_brokers.split(',') if b.strip()}
        try:
            mt5_tickers, extended_set, ticker_meta = load_mt5_cache(Path(args.universe_from_mt5_cache), broker_filter)
            tickers.extend(mt5_tickers)
        except Exception as e:
            print(f"WARN: Failed loading MT5 cache: {e}", file=sys.stderr)
    if args.tickers:
        tickers.extend([t.strip().upper() for t in args.tickers.split(',') if t.strip()])
    if args.universe_from_index:
        for code in [c.strip().lower() for c in args.universe_from_index.split(',') if c.strip()]:
            fname = None
            if code == 'sp500':
                fname = INDICES_DIR / 'sp500_constituents.csv'
            elif code == 'nasdaq100':
                fname = INDICES_DIR / 'nasdaq100_constituents.csv'
            if fname and fname.exists():
                try:
                    import csv as _csv
                    with fname.open('r', encoding='utf-8') as f:
                        rdr = _csv.DictReader(f)
                        for row in rdr:
                            sym = (row.get('Symbol') or row.get('Ticker') or '').strip().upper()
                            if sym:
                                tickers.append(sym)
                except Exception:
                    pass
    # Deduplicate preserving order
    seen_order = {}
    for t in tickers:
        if t not in seen_order:
            seen_order[t] = True
    tickers = list(seen_order.keys())
    if not tickers:
        print('ERROR: No tickers resolved (use --tickers or --universe-from-index).', file=sys.stderr)
        return 2

    try:
        horizons = [int(x) for x in args.exit_horizons.split(',') if x.strip()]
    except Exception:
        print('ERROR: invalid --exit-horizons', file=sys.stderr)
        return 2
    horizons = [h for h in horizons if h > 0]
    if not horizons:
        print('ERROR: no positive horizons', file=sys.stderr)
        return 2

    extra_args: List[str] = []
    if args.extra_classifier_args:
        extra_args.extend(args.extra_classifier_args.split())

    # Allow overriding classifier path safely
    global CLASSIFIER_PATH  # pragma: no mutate
    if args.classifier_path:
        override = Path(args.classifier_path)
        if override.exists():
            CLASSIFIER_PATH = override
        else:
            print(f"WARN: --classifier-path not found: {override}; using default {CLASSIFIER_PATH}", file=sys.stderr)

    all_events: List[BreakingEvent] = []
    # Helper to fetch events (BREAKING+REPETITION) using chosen mode
    def fetch_events(include_repetition_only: bool = False) -> List[BreakingEvent]:
        out: List[BreakingEvent] = []
        if args.use_external_classifier:
            for i in range(0, len(tickers), max(1, args.chunk_size)):
                batch = tickers[i:i+args.chunk_size]
                evs = run_external_classifier(batch, args.months, args.items, args.pages, extra_args)
                if args.debug_parse_failures:
                    # Re-run external classifier to inspect raw output and identify parse failures.
                    raw_cmd = [sys.executable, str(CLASSIFIER_PATH), '--tickers', ','.join(batch), '--months', str(args.months), '--items', str(args.items), '--pages', str(args.pages), '--breaking-mode']
                    raw_cmd.extend(extra_args)
                    proc2 = subprocess.run(raw_cmd, capture_output=True, text=True)
                    raw_lines = [l for l in proc2.stdout.splitlines() if l.strip()]
                    candidate_lines = [l for l in raw_lines if l.startswith('BREAKING |') or l.startswith('REPETITION |')]
                    # Very loose matching: consider a line parsed if any event title fragment appears in it.
                    parsed_titles = {e.title for e in evs if e.title}
                    failed = []
                    for l in candidate_lines:
                        if not any(t in l for t in parsed_titles):
                            failed.append(l)
                    if failed:
                        sample = failed[:5]
                        print(f"[DEBUG] External parse failures: {len(failed)} (sample up to 5)\n  " + "\n  ".join(sample), file=sys.stderr)
                if include_repetition_only:
                    evs = [e for e in evs if e.kind == 'REPETITION']
                out.extend(evs)
        else:
            state_path = Path(args.classifier_state_file)
            sentiment_filter = {s.strip().lower() for s in args.classifier_sentiment.split(',') if s.strip()} if args.classifier_sentiment else None
            partner_files = [s.strip() for s in args.classifier_partner_files.split(',') if s.strip()] if args.classifier_partner_files else None
            partner_from_index = [s.strip() for s in args.classifier_partner_from_index.split(',') if s.strip()] if args.classifier_partner_from_index else None
            partner_from_private = [s.strip() for s in args.classifier_partner_from_private.split(',') if s.strip()] if args.classifier_partner_from_private else None
            reset_pairs = [s.strip() for s in args.classifier_reset_pairs.split(',') if s.strip()] if args.classifier_reset_pairs else None
            for i in range(0, len(tickers), max(1, args.chunk_size)):
                batch = tickers[i:i+args.chunk_size]
                inline_events = classify_inline(
                    tickers=batch,
                    months=args.months,
                    items=args.items,
                    pages=args.pages,
                    cooldown_days=args.cooldown_days,
                    breaking_mode=True,
                    state_file=state_path,
                    ignore_state=args.ignore_classifier_state,
                    verbose=args.debug_event_stats,
                    sentiment_filter=sentiment_filter,
                    strict_ticker_match=args.classifier_strict_ticker,
                    no_cluster_grouping=args.classifier_no_cluster_grouping,
                    no_entity_linking=args.classifier_no_entity_linking,
                    no_ignore_price_moves=args.classifier_no_ignore_price_moves,
                    partner_files=partner_files,
                    partner_from_index=partner_from_index,
                    partner_from_private=partner_from_private,
                    reset_pairs=reset_pairs,
                    suppress_direction=args.classifier_suppress_direction,
                )
                if include_repetition_only:
                    inline_events = [e for e in inline_events if e.kind == 'REPETITION']
                out.extend(inline_events)
        return out

    all_events = fetch_events(include_repetition_only=False)

    # Optional staleness filter
    if args.max_staleness_min is not None:
        before = len(all_events)
        all_events = [e for e in all_events if (e.staleness_min is None) or (e.staleness_min <= args.max_staleness_min)]
        if args.debug_event_stats:
            print(f"Staleness filter removed {before - len(all_events)} events (> {args.max_staleness_min} min)")

    # Diagnostics: count kinds before filtering
    if args.debug_event_stats:
        from collections import Counter
        kind_counts = Counter(e.kind for e in all_events)
        print(f"Event kind counts before filtering: {dict(kind_counts)}")

    # Optionally reclassify based on chronological order (earliest per cluster = BREAKING)
    if args.classifier_chronological_breaking and all_events:
        from collections import defaultdict
        import re
        
        cluster_map = defaultdict(list)
        
        for ev in all_events:
            # Use dedup_key (includes partner) if available, otherwise fall back to ticker__cluster
            key = ev.dedup_key if hasattr(ev, 'dedup_key') and ev.dedup_key else f"{ev.ticker}__{ev.cluster}"
            cluster_map[key].append(ev)
        
        # Each unique key gets its own cluster in chronological mode
        # (deal signatures keep events separate - different deals are different events)
        merged_clusters = {}
        for key, events in cluster_map.items():
            merged_clusters[key] = events
        
        reclassified = 0
        for key, events in merged_clusters.items():
            if len(events) >= 1:
                # Sort by timestamp
                events_sorted = sorted(events, key=lambda x: x.dt)
                first_article_time = events_sorted[0].dt
                
                # Mark first as BREAKING, rest as REPETITION
                if events_sorted[0].kind != 'BREAKING':
                    events_sorted[0].kind = 'BREAKING'
                    reclassified += 1
                # Set first_seen to the earliest article time for the first article
                events_sorted[0].first_seen = first_article_time
                events_sorted[0].staleness_min = 0.0
                
                # Process remaining events
                for ev in events_sorted[1:]:
                    if ev.kind != 'REPETITION':
                        ev.kind = 'REPETITION'
                        reclassified += 1
                    # Always update first_seen and staleness to reference the first article
                    ev.first_seen = first_article_time
                    delta = (ev.dt - first_article_time).total_seconds() / 60
                    ev.staleness_min = delta
        
        if reclassified > 0:
            print(f"[Chronological mode] Reclassified {reclassified} events based on publication time", file=sys.stderr, flush=True)

    # Filter out repetition unless explicitly requested
    if not args.include_repetition:
        pre_ct = len(all_events)
        rep_ct = sum(1 for e in all_events if e.kind == 'REPETITION')
        brk_ct = sum(1 for e in all_events if e.kind == 'BREAKING')
        
        # Always show both kinds for verification
        print(f"\n=== EVENT CLASSIFICATION RESULTS ===", file=sys.stderr, flush=True)
        print(f"Total events found: {pre_ct}", file=sys.stderr, flush=True)
        print(f"  - BREAKING: {brk_ct}", file=sys.stderr, flush=True)
        print(f"  - REPETITION: {rep_ct}", file=sys.stderr, flush=True)
        
        # Always show BREAKING events
        print(f"\nBREAKING events (sorted by time):", file=sys.stderr, flush=True)
        breaking_sorted = sorted([ev for ev in all_events if ev.kind == 'BREAKING'], key=lambda x: x.dt)
        for e in breaking_sorted:
            print(f"  {e.ticker} | {e.cluster} | {e.dt.isoformat()} | {e.source} | {e.title[:80]}", file=sys.stderr, flush=True)
        
        if args.debug_event_stats:
            print(f"\nREPETITION events (sorted by time):", file=sys.stderr, flush=True)
            repetition_sorted = sorted([ev for ev in all_events if ev.kind == 'REPETITION'], key=lambda x: x.dt)
            for e in repetition_sorted:
                fs = e.first_seen.isoformat() if e.first_seen else 'N/A'
                stale = f"{e.staleness_min:.1f}min" if e.staleness_min is not None else 'N/A'
                print(f"  {e.ticker} | {e.cluster} | {e.dt.isoformat()} | staleness={stale} | {e.source} | {e.title[:50]}", file=sys.stderr, flush=True)
        
        all_events = [e for e in all_events if e.kind == 'BREAKING']
        post_ct = len(all_events)
        print(f"\nAfter filtering (BREAKING only): {post_ct} events\n", file=sys.stderr, flush=True)
        if pre_ct > 0 and post_ct == 0:
            msg = 'No BREAKING events after filtering out REPETITION.'
            if args.fallback_repetition_if_none and rep_ct > 0:
                print(msg + ' Falling back to REPETITION events due to --fallback-repetition-if-none.')
                # reuse original list (we still have rep events in variable 'pre_events'?)
                # We need to re-run classification or keep original list; store earlier.
            else:
                print(msg)
            if args.fallback_repetition_if_none and rep_ct > 0:
                # Re-run classification explicitly asking to keep only repetition events
                all_events = fetch_events(include_repetition_only=True)
                if args.debug_event_stats:
                    print(f"Fallback repetitions count: {len(all_events)}")

    if not all_events:
        print('No BREAKING events found.')
        Path(args.output).write_text('')
        if args.summary_output:
            Path(args.summary_output).write_text('')
        return 0

    rows: List[Dict[str, object]] = []
    # Lazy MT5 import & init if requested
    mt5 = None
    mt5_broker = None  # Track which broker we're connected to
    if args.use_mt5_prices:
        try:
            import MetaTrader5 as _mt5  # type: ignore
            import time
            
            # Try simple initialization first (for already logged-in terminal)
            if _mt5.initialize():
                mt5 = _mt5
            else:
                # If simple init fails, try with credentials (will start terminal)
                from dotenv import load_dotenv
                load_dotenv()
                
                LOGIN = int(os.getenv('ADMIRAL_MT5_LOGIN', '0'))
                PASSWORD = os.getenv('ADMIRAL_PASSWORD')
                SERVER = os.getenv('ADMIRAL_MT5_SERVER', 'AdmiralsSC-Demo')
                PATH = os.getenv('ADMIRAL_MT5_PATH', r'C:\Program Files\Admirals SC MT5 Terminal\terminal64.exe')
                
                if PASSWORD:
                    if _mt5.initialize(path=PATH, login=LOGIN, password=PASSWORD, server=SERVER):
                        mt5 = _mt5
                    else:
                        print(f'WARN: MT5 initialize failed. Error: {_mt5.last_error()}', file=sys.stderr)
            
            if mt5:
                # Detect which broker we're connected to
                account_info = mt5.account_info()
                if account_info:
                    if 'Admirals' in account_info.server or account_info.login == int(os.getenv('ADMIRAL_MT5_LOGIN', '0')):
                        mt5_broker = 'admirals'
                        print(f'✅ MT5 connected to Admirals (Account: {account_info.login})', file=sys.stderr)
                    elif 'Pepperstone' in account_info.server or account_info.login == 61418548:
                        mt5_broker = 'pepperstone'
                        print(f'✅ MT5 connected to Pepperstone (Account: {account_info.login})', file=sys.stderr)
                    else:
                        mt5_broker = 'unknown'
                        print(f'⚠️  MT5 connected to unknown broker: {account_info.server} (Account: {account_info.login})', file=sys.stderr)
        except Exception as e:
            print(f'WARN: MT5 import/init failed: {e}', file=sys.stderr)

    def simulate_with_mt5(ev: BreakingEvent, horizons: List[int]) -> Dict[str, object]:
        if not mt5:
            base = {
                'ticker': ev.ticker,
                'cluster': ev.cluster,
                'event_time': ev.dt.isoformat(),
                'title': ev.title,
                'source': ev.source,
                'error': 'mt5_unavailable',
                'price_source': 'none',
            }
            for h in horizons:
                base[f'ret_{h}m'] = None
            return base
        try:
            # Determine MT5 symbol format based on connected broker
            # Admirals: #TICKER-T or #TICKER.US-T
            # Pepperstone: TICKER.US or TICKER.US-24 (extended hours only)
            
            if mt5_broker == 'admirals':
                possible_symbols = [
                    f"#{ev.ticker}-T",           # Standard: #AAPL-T
                    f"#{ev.ticker}.US-T",        # With .US: #SOFI.US-T
                ]
            elif mt5_broker == 'pepperstone':
                possible_symbols = [
                    f"{ev.ticker}.US",           # Regular hours: AAPL.US
                    f"{ev.ticker}.US-24",        # Extended hours: AAPL.US-24
                ]
            else:
                # Unknown broker - try all formats
                possible_symbols = [
                    f"#{ev.ticker}-T",
                    f"#{ev.ticker}.US-T",
                    f"{ev.ticker}.US",
                    f"{ev.ticker}.US-24",
                ]
            
            mt5_symbol = None
            for sym in possible_symbols:
                if mt5.symbol_select(sym, True):
                    mt5_symbol = sym
                    break
            
            if not mt5_symbol:
                raise RuntimeError(f'symbol_not_found:tried_{"|".join(possible_symbols)}')
            
            from_dt = (ev.dt - pd.Timedelta(seconds=args.mt5_pre_window_sec)).astimezone(NY_TZ)
            to_dt = (ev.dt + pd.Timedelta(minutes=max(horizons) + 1)).astimezone(NY_TZ)
            # MetaTrader5 expects naive datetimes in platform local tz; we pass naive NY times
            from_naive = datetime(from_dt.year, from_dt.month, from_dt.day, from_dt.hour, from_dt.minute, from_dt.second)
            to_naive = datetime(to_dt.year, to_dt.month, to_dt.day, to_dt.hour, to_dt.minute, to_dt.second)
            rates = mt5.copy_rates_range(mt5_symbol, mt5.TIMEFRAME_M1, from_naive, to_naive)
            if rates is None or len(rates) == 0:
                raise RuntimeError(f'no_mt5_bars_for_{mt5_symbol}')
            import numpy as _np  # type: ignore
            df = pd.DataFrame(rates)
            df['time_dt'] = pd.to_datetime(df['time'], unit='s').dt.tz_localize(NY_TZ)
            # Entry bar: first bar whose time >= event time + 1 minute
            entry_target = (ev.dt + pd.Timedelta(minutes=1)).astimezone(NY_TZ)
            entry_bar = df[df['time_dt'] >= entry_target].sort_values('time_dt').head(1)
            entry_price = float(entry_bar['close'].iloc[0]) if not entry_bar.empty else None
            row: Dict[str, object] = {
                'ticker': ev.ticker,
                'cluster': ev.cluster,
                'event_time': ev.dt.isoformat(),
                'title': ev.title,
                'source': ev.source,
                'entry_time': entry_bar['time_dt'].iloc[0].isoformat() if not entry_bar.empty else None,
                'entry_price': entry_price,
                'price_source': 'mt5_m1',
                'mt5_symbol': mt5_symbol,  # Track which symbol was used
            }
            for h in horizons:
                # Exit horizons measured from entry time (news + 1 min + horizon)
                target_ts = ev.dt + pd.Timedelta(minutes=1 + h)
                target_bar = df[df['time_dt'] >= target_ts.astimezone(NY_TZ)].sort_values('time_dt').head(1)
                if entry_price is None or target_bar.empty:
                    row[f'ret_{h}m'] = None
                else:
                    px = float(target_bar['close'].iloc[0])
                    row[f'ret_{h}m'] = (px - entry_price) / entry_price * 100.0 if entry_price else None
            return row
        except Exception as e:
            base = {
                'ticker': ev.ticker,
                'cluster': ev.cluster,
                'event_time': ev.dt.isoformat(),
                'title': ev.title,
                'source': ev.source,
                'error': f'mt5_fail:{e}',
                'price_source': 'none',
            }
            for h in horizons:
                base[f'ret_{h}m'] = None
            return base

    if args.skip_ib and not args.use_mt5_prices:
        for ev in all_events:
            session = 'RTH' if is_rth(ev.dt) else 'EXT'
            brokers = ticker_meta.get(ev.ticker, {}).get('brokers', []) if ticker_meta else []
            external_eligible = (session == 'EXT' and args.extended_hours_mode and ev.ticker in extended_set and ('pepperstone' in [b.lower() for b in brokers] if brokers else True))
            external_skipped = (session == 'EXT' and args.extended_hours_mode and not external_eligible)
            if external_skipped:
                # Still record event but no returns
                row = {
                    'ticker': ev.ticker,
                    'cluster': ev.cluster,
                    'event_time': ev.dt.isoformat(),
                    'title': ev.title,
                    'source': ev.source,
                    'kind': ev.kind,
                    'session': session,
                    'external_eligible': external_eligible,
                    'external_skipped': external_skipped,
                    'platform': 'none',
                }
                for h in horizons:
                    row[f'ret_{h}m'] = None
                rows.append(row)
                continue
            row = {
                'ticker': ev.ticker,
                'cluster': ev.cluster,
                'event_time': ev.dt.isoformat(),
                'first_seen': ev.first_seen.isoformat() if ev.first_seen else None,
                'staleness_min': ev.staleness_min,
                'title': ev.title,
                'source': ev.source,
                'url': ev.url,
                'kind': ev.kind,
                'session': session,
                'external_eligible': external_eligible,
                'external_skipped': external_skipped,
                'platform': 'none',
            }
            for h in horizons:
                row[f'ret_{h}m'] = None
            rows.append(row)
    elif args.skip_ib and args.use_mt5_prices:
        for ev in all_events:
            session = 'RTH' if is_rth(ev.dt) else 'EXT'
            brokers = ticker_meta.get(ev.ticker, {}).get('brokers', []) if ticker_meta else []
            external_eligible = (session == 'EXT' and args.extended_hours_mode and ev.ticker in extended_set and ('pepperstone' in [b.lower() for b in brokers] if brokers else True))
            external_skipped = (session == 'EXT' and args.extended_hours_mode and not external_eligible)
            if external_skipped:
                base = {
                    'ticker': ev.ticker,
                    'cluster': ev.cluster,
                    'event_time': ev.dt.isoformat(),
                    'first_seen': ev.first_seen.isoformat() if ev.first_seen else None,
                    'staleness_min': ev.staleness_min,
                    'title': ev.title,
                    'source': ev.source,
                    'url': ev.url,
                    'kind': ev.kind,
                    'session': session,
                    'external_eligible': external_eligible,
                    'external_skipped': external_skipped,
                    'platform': 'mt5',
                }
                for h in horizons:
                    base[f'ret_{h}m'] = None
                rows.append(base)
                continue
            mt5_row = simulate_with_mt5(ev, horizons)
            mt5_row['kind'] = ev.kind
            mt5_row['session'] = session
            mt5_row['external_eligible'] = external_eligible
            mt5_row['external_skipped'] = external_skipped
            mt5_row['platform'] = 'mt5'
            mt5_row['first_seen'] = ev.first_seen.isoformat() if ev.first_seen else None
            mt5_row['staleness_min'] = ev.staleness_min
            mt5_row['url'] = ev.url or mt5_row.get('url')
            rows.append(mt5_row)
    else:
        mod = load_hist_module()
        ports = [int(p) for p in args.ports.split(',') if p.strip()]

        if args.no_ib_batch_per_ticker:
            # Legacy per-event fetching (may hit pacing sooner)
            for ev in all_events:
                session = 'RTH' if is_rth(ev.dt) else 'EXT'
                brokers = ticker_meta.get(ev.ticker, {}).get('brokers', []) if ticker_meta else []
                external_eligible = (session == 'EXT' and args.extended_hours_mode and ev.ticker in extended_set and ('pepperstone' in [b.lower() for b in brokers] if brokers else True))
                external_skipped = (session == 'EXT' and args.extended_hours_mode and not external_eligible)
                if external_skipped:
                    sim_row = {
                        'ticker': ev.ticker,
                        'cluster': ev.cluster,
                        'event_time': ev.dt.isoformat(),
                        'first_seen': ev.first_seen.isoformat() if ev.first_seen else None,
                        'staleness_min': ev.staleness_min,
                        'title': ev.title,
                        'source': ev.source,
                        'url': ev.url,
                        'kind': ev.kind,
                        'session': session,
                        'external_eligible': external_eligible,
                        'external_skipped': external_skipped,
                        'platform': 'ib',
                    }
                    for h in horizons:
                        sim_row[f'ret_{h}m'] = None
                    rows.append(sim_row)
                    continue
                try:
                    sim_row = simulate_trade_for_event(mod, ev, args.lookahead_min, args.pre_window_sec, args.what, args.use_rth, args.pause_sec, ports, args.client_id, horizons)
                except Exception as e:
                    sim_row = {
                        'ticker': ev.ticker,
                        'cluster': ev.cluster,
                        'event_time': ev.dt.isoformat(),
                        'first_seen': ev.first_seen.isoformat() if ev.first_seen else None,
                        'staleness_min': ev.staleness_min,
                        'title': ev.title,
                        'source': ev.source,
                        'url': ev.url,
                        'kind': ev.kind,
                        'error': str(e),
                        'price_source': 'ib_error',
                        'platform': 'ib',
                    }
                    for h in horizons:
                        sim_row[f'ret_{h}m'] = None
                else:
                    sim_row['price_source'] = 'ib'
                    sim_row['platform'] = 'ib'
                sim_row['session'] = session
                sim_row['external_eligible'] = external_eligible
                sim_row['external_skipped'] = external_skipped
                sim_row['kind'] = ev.kind
                if 'first_seen' not in sim_row:
                    sim_row['first_seen'] = ev.first_seen.isoformat() if ev.first_seen else None
                if 'staleness_min' not in sim_row:
                    sim_row['staleness_min'] = ev.staleness_min
                if 'url' not in sim_row:
                    sim_row['url'] = ev.url
                rows.append(sim_row)
        else:
            # Batched per ticker: single fetch covers all event windows for that ticker
            IB = getattr(mod, 'IB')
            Stock = getattr(mod, 'Stock')
            parse_times_to_intervals = getattr(mod, 'parse_times_to_intervals')
            fetch_ticks = getattr(mod, 'fetch_ticks')
            fetch_ticks_fast = getattr(mod, 'fetch_ticks_fast', None)  # Try to get faster version
            connect_any = getattr(mod, 'connect_any')

            ib = IB()
            port = connect_any(ib, ports, args.client_id)
            if not port:
                print('ERROR: Could not connect to IB for batched mode', file=sys.stderr)
                return 3
            try:
                # Group events by ticker
                events_by_ticker: Dict[str, List[BreakingEvent]] = {}
                for ev in all_events:
                    events_by_ticker.setdefault(ev.ticker, []).append(ev)
                window_sec = args.pre_window_sec + (args.lookahead_min * 60)
                for ticker, ev_list in events_by_ticker.items():
                    ev_list.sort(key=lambda e: e.dt)
                    end_times = [(e.dt + pd.Timedelta(minutes=args.lookahead_min)).isoformat() for e in ev_list]
                    try:
                        intervals = parse_times_to_intervals(end_times, window_sec, 'America/New_York')
                        contract = Stock(ticker, 'SMART', 'USD')
                        # Use faster version with reduced pacing if available
                        fetch_fn = fetch_ticks_fast if fetch_ticks_fast else fetch_ticks
                        df = fetch_fn(ib=ib, contract=contract, merged=intervals, what=args.what, use_rth=args.use_rth, target_tz='America/New_York', pause_sec=args.pause_sec, max_requests=0)
                        if not df.empty:
                            df['time_dt'] = pd.to_datetime(df['time'])
                        else:
                            df['time_dt'] = []  # maintain column existence
                    except Exception as fe:
                        # Record error rows for all events of this ticker
                        for ev in ev_list:
                            session = 'RTH' if is_rth(ev.dt) else 'EXT'
                            brokers = ticker_meta.get(ev.ticker, {}).get('brokers', []) if ticker_meta else []
                            external_eligible = (session == 'EXT' and args.extended_hours_mode and ev.ticker in extended_set and ('pepperstone' in [b.lower() for b in brokers] if brokers else True))
                            external_skipped = (session == 'EXT' and args.extended_hours_mode and not external_eligible)
                            sim_row = {
                                'ticker': ev.ticker,
                                'cluster': ev.cluster,
                                'event_time': ev.dt.isoformat(),
                                'first_seen': ev.first_seen.isoformat() if ev.first_seen else None,
                                'staleness_min': ev.staleness_min,
                                'title': ev.title,
                                'source': ev.source,
                                'url': ev.url,
                                'kind': ev.kind,
                                'error': f'ib_batch_fail:{fe}',
                                'platform': 'ib',
                                'price_source': 'ib_error',
                                'session': session,
                                'external_eligible': external_eligible,
                                'external_skipped': external_skipped,
                            }
                            for h in horizons:
                                sim_row[f'ret_{h}m'] = None
                            rows.append(sim_row)
                        continue

                    # For each event derive returns from shared df
                    for ev in ev_list:
                        session = 'RTH' if is_rth(ev.dt) else 'EXT'
                        brokers = ticker_meta.get(ev.ticker, {}).get('brokers', []) if ticker_meta else []
                        external_eligible = (session == 'EXT' and args.extended_hours_mode and ev.ticker in extended_set and ('pepperstone' in [b.lower() for b in brokers] if brokers else True))
                        external_skipped = (session == 'EXT' and args.extended_hours_mode and not external_eligible)
                        if external_skipped:
                            sim_row = {
                                'ticker': ev.ticker,
                                'cluster': ev.cluster,
                                'event_time': ev.dt.isoformat(),
                                'first_seen': ev.first_seen.isoformat() if ev.first_seen else None,
                                'staleness_min': ev.staleness_min,
                                'title': ev.title,
                                'source': ev.source,
                                'url': ev.url,
                                'kind': ev.kind,
                                'session': session,
                                'external_eligible': external_eligible,
                                'external_skipped': external_skipped,
                                'platform': 'ib',
                            }
                            for h in horizons:
                                sim_row[f'ret_{h}m'] = None
                            rows.append(sim_row)
                            continue
                        end_dt = ev.dt + pd.Timedelta(minutes=args.lookahead_min)
                        future_df = df[(df['time_dt'] >= ev.dt) & (df['time_dt'] <= end_dt)].copy().sort_values('time_dt') if not df.empty else pd.DataFrame()
                        base_df = df[df['time_dt'] < ev.dt] if not df.empty else pd.DataFrame()
                        pre_price = base_df['price'].iloc[-1] if ('price' in base_df and not base_df.empty) else None
                        if future_df.empty:
                            sim_row = {
                                'ticker': ev.ticker,
                                'cluster': ev.cluster,
                                'event_time': ev.dt.isoformat(),
                                'first_seen': ev.first_seen.isoformat() if ev.first_seen else None,
                                'staleness_min': ev.staleness_min,
                                'title': ev.title,
                                'source': ev.source,
                                'url': ev.url,
                                'kind': ev.kind,
                                'session': session,
                                'external_eligible': external_eligible,
                                'external_skipped': external_skipped,
                                'platform': 'ib',
                                'pre_price': pre_price,
                            }
                            for h in horizons:
                                sim_row[f'ret_{h}m'] = None
                            rows.append(sim_row)
                            continue
                        entry_row = future_df.iloc[0]
                        exit_row = future_df.iloc[-1]
                        entry_price = entry_row.get('price')
                        exit_price = exit_row.get('price')
                        pct_return = None
                        if entry_price and exit_price and entry_price != 0:
                            pct_return = (exit_price - entry_price) / entry_price * 100.0

                        def ret_at(minutes: int):
                            target = ev.dt + pd.Timedelta(minutes=minutes)
                            sub = future_df[future_df['time_dt'] >= target]
                            if sub.empty or not entry_price:
                                return None
                            p = sub.iloc[0].get('price')
                            if not p or entry_price == 0:
                                return None
                            return (p - entry_price) / entry_price * 100.0

                        sim_row = {
                            'ticker': ev.ticker,
                            'cluster': ev.cluster,
                            'event_time': ev.dt.isoformat(),
                            'first_seen': ev.first_seen.isoformat() if ev.first_seen else None,
                            'staleness_min': ev.staleness_min,
                            'title': ev.title,
                            'source': ev.source,
                            'url': ev.url,
                            'kind': ev.kind,
                            'entry_time': entry_row['time'],
                            'entry_price': entry_price,
                            'exit_time': exit_row['time'],
                            'exit_price': exit_price,
                            'pct_return': pct_return,
                            'rows': len(future_df),
                            'pre_price': pre_price,
                            'session': session,
                            'external_eligible': external_eligible,
                            'external_skipped': external_skipped,
                            'platform': 'ib',
                            'price_source': 'ib',
                        }
                        for h in horizons:
                            sim_row[f'ret_{h}m'] = ret_at(h)
                        rows.append(sim_row)
            finally:
                if ib.isConnected():
                    ib.disconnect()

    # Price bucket tagging
    threshold = float(args.price_threshold)
    for r in rows:
        price_candidate = r.get('entry_price') or r.get('pre_price')
        try:
            pval = float(price_candidate) if price_candidate is not None else None
        except Exception:
            pval = None
        if pval is None:
            r['price_bucket'] = None
        else:
            r['price_bucket'] = 'smallcap' if pval < threshold else 'medlargecap'

    out_path = Path(args.output)
    # Determine CSV columns
    cols: List[str] = []
    for r in rows:
        for k in r.keys():
            if k not in cols:
                cols.append(k)
    with out_path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    msg = f"Wrote {len(rows)} rows to {out_path}"
    print(msg)
    print(msg, file=sys.stderr, flush=True)

    if args.summary_output:
        df = pd.DataFrame(rows)
        horizon_cols = [c for c in df.columns if c.startswith('ret_')]
        summaries: List[Dict[str, object]] = []
        if not horizon_cols or df.empty:
            # Write empty summary if no returns data
            pd.DataFrame([]).to_csv(args.summary_output, index=False)
            print(f"Wrote empty summary to {args.summary_output}")
            return 0


if __name__ == '__main__':
    sys.exit(main())
