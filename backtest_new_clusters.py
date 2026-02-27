"""
COMPREHENSIVE BIDIRECTIONAL BACKTESTING ENGINE
==============================================
Tests BOTH long and short event clusters for potential integration into live_news_trader_ibkr.py:

⚠️ DATA SOURCE NOTE:
This backtest runs in strict TICK-ONLY mode and never fetches minute candles.
The live orchestrator also uses `/stock/tick` (tick-by-tick trades, PREMIUM tier $99+/month)
Backtest results are UPPER BOUND estimates. Tick-only simulation may have:
- More precise entry timing
- Better volume analysis (tick-level vs aggregated bars)
- Slightly different P&L due to execution price differences

LONG CLUSTERS (11 stock-specific):
1. Analyst Upgrades (analyst_action_upgrade)
2. Major Contracts/Awards (major_contract)
3. Phase 3 Clinical Achievements (phase3_achievement)
4. Share Buyback Announcements (share_buyback)
5. Uplisting to Major Exchanges (uplisting)
6. AI Enterprise Partnerships (ai_enterprise_partnership) - from orchestrator
7. Merger/Acquisition Announcements (merger_acq) - from orchestrator
8. FDA Approvals (fda_approval) - from orchestrator
9. Contract Awards (contract_award_keyword) - from orchestrator
10. Bailout Approved/Received (bailout_approved) - NEW: government rescue approved
11. Bailout Repaid (bailout_repaid) - NEW: financial strength signal

SHORT CLUSTERS (16 stock-specific):
1. Phase 3 Trial Failures (phase3_failure)
2. FDA Rejections/CRLs (fda_rejection)
3. Analyst Downgrades to Sell (analyst_downgrade_sell)
4. DOJ Investigations (doj_investigation) - with settlement exclusions
5. FTC Blocking (ftc_block)
6. Failed Mergers/Acquisitions (failed_merger)
7. Accounting Fraud/Restatements (accounting_fraud)
8. Lost Contracts (contract_loss)
9. Export Bans (export_ban)
10. Sector Tariffs (sector_tariff)
11. Share Dilution (share_dilution)
12. Delisting/Compliance Failures (delisting)
13. Plane Crashes/Safety Incidents (plane_crash) - NEW
14. Bailout Seeking (bailout_seeking) - NEW
15. Buyback Suspension (buyback_suspension) - NEW
16. Large Bank Economic Warnings (large_bank_warning)

SECTOR-LEVEL CLUSTERS (10 total - require peer detection):
SHORT (5):
- Trade War/Export Sanctions (trade_war_sector) - now enabled with specific patterns
- Supply Chain/Production Halts (supply_halt_sector)
- War/Geopolitical Risk (war_risk_sector) - defense stocks now trade both ways
- Regional Banking Fears (regional_bank_fear) - regional banks subsector only
- Tariff Rift Bearish (tariff_rift_bearish) - China trade war impacts

LONG (5):
- Guaranteed Volume/Subsidies (mandate_subsidy_sector)
- ACA Support (aca_support_healthcare) - healthcare only
- Supply Chain Recovery (supply_recovery_sector)
- Defense Spending Spike (defense_spending_sector) - defense/aerospace only
- Tariff Protection Bullish (tariff_protection_bullish) - US domestic producers only

Uses 3-month historical news cache and Finnhub tick data cache when available. No minute candles are used.
Implements Volume Exhaustion exits (inverted for shorts) from news_trade_orchestrator_finnhub.py.
Includes advanced duplicate detection inspired by live_news_trader_ibkr.py.
Includes COMPLETE generic fluff filtering (38+ patterns) from orchestrator.

Goal: Identify which clusters have Sharpe > 1.5 for integration into live system.

Tick-mode: TICK_MODE is forced on; momentum uses Finnhub tick data cache only. Optional env TICK_LOCAL_TZ
controls local display timezone but computations run in UTC.
"""

import os
import sys
import json
import logging
import re
import time
from datetime import datetime, timedelta, timezone, time as dtime
from typing import Any, Dict, List, Optional, Set, Tuple
from collections import defaultdict
from pathlib import Path
import pandas as pd
import numpy as np
try:
    import optuna
except ImportError as e:
    optuna = None
    print(f"WARNING: Optuna not found: {e}", file=sys.stderr)
    print("Hyperparameter optimization will be disabled.", file=sys.stderr)
    print("Install it with: pip install optuna", file=sys.stderr)
import requests
from zoneinfo import ZoneInfo

# Optional: Finnhub client for tick-mode
try:
    import finnhub  # type: ignore
except Exception:  # pragma: no cover
    finnhub = None  # type: ignore

# Reuse robust Finnhub tick fetcher from orchestrator (same folder)
try:
    from news_trade_orchestrator_finnhub import fetch_ticks_for_date as fh_fetch_ticks_for_date  # type: ignore
    from news_trade_orchestrator_finnhub import _get_cache_path as fh_get_tick_cache_path  # type: ignore
except Exception:
    fh_fetch_ticks_for_date = None  # type: ignore
    fh_get_tick_cache_path = None  # type: ignore

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# PRECEDENT ANALYSIS - TRADE CONFIRMATION
# ============================================================================

def extract_key_entities(title: str, text: str = '') -> Dict[str, Set[str]]:
    """Extract key entities from news title and text for content similarity."""
    combined = f"{title} {text}".upper()
    
    entities = {
        'companies': set(),
        'people': set(),
        'amounts': set(),
        'drugs_products': set(),
        'topics': set()
    }
    
    # Extract dollar amounts ($500M, $2.5B, etc.)
    amounts = re.findall(r'\$\d+(?:\.\d+)?\s*(?:M|B|K|MILLION|BILLION|THOUSAND)', combined)
    entities['amounts'].update(amounts)
    
    # Extract company tickers/names (AAPL, MSFT, etc.)
    companies = re.findall(r'\b[A-Z]{2,5}\b', combined)
    entities['companies'].update(companies)
    
    # Extract people names (simple pattern: Capitalized words)
    people = re.findall(r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b', title)
    entities['people'].update(people)
    
    # Extract drug/product names (usually capitalized, 4+ letters)
    products = re.findall(r'\b[A-Z][a-z]{3,}\b', title)
    entities['drugs_products'].update(products)
    
    # Extract key topics
    topics = []
    topic_keywords = {
        'merger': ['MERGER', 'ACQUISITION', 'ACQUIRES', 'BUYOUT'],
        'fda': ['FDA', 'APPROVAL', 'REJECT'],
        'contract': ['CONTRACT', 'AWARD', 'DEAL'],
        'investigation': ['DOJ', 'SEC', 'INVESTIGATION', 'PROBE'],
        'earnings': ['EARNINGS', 'REVENUE', 'PROFIT'],
        'analyst': ['UPGRADE', 'DOWNGRADE', 'RATING'],
    }
    for topic, keywords in topic_keywords.items():
        if any(kw in combined for kw in keywords):
            topics.append(topic)
    entities['topics'].update(topics)
    
    return entities

def calculate_content_similarity(entities1: Dict[str, Set[str]], entities2: Dict[str, Set[str]]) -> float:
    """Calculate weighted Jaccard similarity between two entity sets."""
    weights = {
        'amounts': 0.3,
        'companies': 0.25,
        'drugs_products': 0.2,
        'people': 0.15,
        'topics': 0.1
    }
    
    total_score = 0.0
    for entity_type, weight in weights.items():
        set1 = entities1.get(entity_type, set())
        set2 = entities2.get(entity_type, set())
        
        if not set1 and not set2:
            continue
        
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        if union > 0:
            jaccard = intersection / union
            total_score += jaccard * weight
    
    return total_score

def check_precedent_news(
    ticker: str,
    event_time: datetime,
    event_title: str,
    all_articles: List[Dict],
    similarity_threshold: float = 0.5
) -> Tuple[bool, Optional[Dict]]:
    """
    Check if there's earlier news about the SAME story (precedent).
    
    Returns:
        (has_precedent, precedent_info)
        - has_precedent: True if similar news found earlier
        - precedent_info: Dict with precedent details or None
    """
    # Extract entities from current event
    event_entities = extract_key_entities(event_title)
    
    # Get all news for this ticker before the event
    precedent_news = []
    
    for article in all_articles:
        # Check if ticker is in this article
        raw_tickers = article.get('tickers', [])
        if isinstance(raw_tickers, str):
            article_tickers = raw_tickers.split(',') if raw_tickers else []
        elif isinstance(raw_tickers, list):
            article_tickers = raw_tickers
        else:
            continue
        
        article_tickers = [t.strip().upper() for t in article_tickers]
        if ticker.upper() not in article_tickers:
            continue
        
        # Parse publication time
        pub_date = article.get('date', '')
        if not pub_date:
            continue
        
        try:
            # Detect ISO strictly via YYYY-MM-DDT (avoid 'Thu' false positive)
            if re.search(r"\d{4}-\d{2}-\d{2}T", pub_date):
                try:
                    pub_time = datetime.fromisoformat(pub_date.replace('Z', '+00:00'))
                except Exception:
                    from email.utils import parsedate_to_datetime
                    pub_time = parsedate_to_datetime(pub_date)
            else:
                from email.utils import parsedate_to_datetime
                pub_time = parsedate_to_datetime(pub_date)
            
            if pub_time.tzinfo is None:
                pub_time = pub_time.replace(tzinfo=timezone.utc)
            else:
                pub_time = pub_time.astimezone(timezone.utc)
        except Exception:
            continue
        
        # Check if published before event time
        if pub_time >= event_time:
            continue
        
        # Extract entities from precedent article
        precedent_title = article.get('title', '')
        precedent_text = article.get('text', '')[:200]
        precedent_entities = extract_key_entities(precedent_title, precedent_text)
        
        # Calculate similarity
        similarity = calculate_content_similarity(event_entities, precedent_entities)
        
        if similarity >= similarity_threshold:
            time_diff_minutes = (event_time - pub_time).total_seconds() / 60
            precedent_news.append({
                'pub_time': pub_time,
                'title': precedent_title,
                'source': article.get('source_name', ''),
                'similarity': similarity,
                'time_diff_minutes': time_diff_minutes
            })
    
    if precedent_news:
        # Sort by similarity (highest first)
        precedent_news.sort(key=lambda x: x['similarity'], reverse=True)
        best_precedent = precedent_news[0]
        
        return True, {
            'precedent_count': len(precedent_news),
            'best_match': best_precedent,
            'all_precedents': precedent_news[:3]  # Top 3
        }
    
    return False, None

# ============================================================================
# CONFIGURATION
# ============================================================================

# Determine base directory for relative files (env, cache, etc.)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_env_file():
    """Load API keys and other env vars from alpkey.env or .env if present."""
    candidates = [
        os.path.join(BASE_DIR, 'alpkey.env'),
        os.path.join(BASE_DIR, '.env')
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith('#'):
                            continue
                        if '=' in line:
                            key, value = line.split('=', 1)
                            os.environ[key.strip()] = value.strip()
            except Exception as e:
                logger.warning(f"Failed to load env file {path}: {e}")
            break

# Load env before reading keys
load_env_file()

# API Keys (load from environment or config)
STOCKNEWS_API_KEY = os.getenv('STOCKNEWS_API_KEY', 'your_stocknews_api_key_here')
FINNHUB_API_KEY = os.getenv('FINNHUB_API_KEY', 'your_finnhub_api_key_here')

# Cache directory (ensure path is relative to this script, not the CWD)
CACHE_DIR = os.path.join(BASE_DIR, ".news_cache")

# Backtest period (can be overridden by FAST_BACKTEST env)
BACKTEST_START_DATE = datetime.now() - timedelta(days=90)  # default 3 months
BACKTEST_END_DATE = datetime.now()

# IN-SAMPLE / OUT-OF-SAMPLE SPLIT (Professional-Grade Validation)
# FIX (Oct 27, 2025): Adjusted for Jul-Oct 2025 news cache availability
# IS Period: Jul 1 - Sep 30 (90 days training), OOS Period: Oct 1 - Oct 27 (27 days testing)
TRAIN_START_DATE = datetime(2025, 7, 1)
TRAIN_END_DATE = datetime(2025, 9, 30)
TEST_START_DATE = datetime(2025, 10, 1)  # OOS starts October (not January)
TEST_END_DATE = datetime(2025, 10, 27)  # Current date
FAST_MODE = os.getenv('FAST_BACKTEST', '0') == '1'

# VIX Regime Filter (Professional Risk Management)
VIX_REGIME_THRESHOLD = 30.0  # Block entries when VIX > 30 (market panic)
VIX_CACHE_DIR = Path("tick_data_cache/yfinance_vix")  # Updated from pairs trading cache

# Monte Carlo Validation Threshold
MC_SHARPE_THRESHOLD = 0.3  # Require Sharpe CI lower bound >= 0.3 for statistical robustness

# Enable tick-mode (use Finnhub /stock/tick like the orchestrator)
# Force tick-mode ON for backtest; we never use minute data here.
TICK_MODE = True

# Global switch to forbid any fetching (cache-only behavior)
FORCE_CACHE_ONLY = os.getenv('FORCE_CACHE_ONLY', '1') == '1'
TICK_LOCAL_TZ = os.getenv('TICK_LOCAL_TZ', 'America/New_York')

# Trading parameters
INITIAL_CAPITAL = 27000
RISK_PER_TRADE_PCT = 2.0  # 2% risk per trade
MAX_STOCK_PRICE_NON_SP500 = 560  # Skip stocks above $560 if not in SP500

# Exchange Filtering (NYSE and NASDAQ only, exclude OTC and AMEX)
# Using Finnhub MIC codes AND full exchange names (Finnhub API returns both formats)
ALLOWED_EXCHANGES = {
    # NYSE MIC codes
    'XNYS',  # New York Stock Exchange
    'XASE',  # NYSE American (formerly AMEX)
    'NYSE',  # Legacy code
    'NYQ',   # Legacy code
    'ARCX',  # NYSE Arca
    # NYSE full names (from Finnhub profile API)
    'NEW YORK STOCK EXCHANGE, INC.',
    'NEW YORK STOCK EXCHANGE',
    'NYSE AMERICAN, LLC',
    'NYSE MKT LLC',
    'NYSE AMERICAN',
    # NASDAQ MIC codes
    'XNAS',  # NASDAQ All Markets
    'XNMS',  # NASDAQ National Market System
    'XNCM',  # NASDAQ Capital Market
    'XNGS',  # NASDAQ Global Select Market
    'NASDAQ', # Legacy code
    'NAS',   # Legacy code
    'NMS',   # Legacy code
    'NGM',   # Legacy code
    'NCM',   # Legacy code
    # NASDAQ full names (from Finnhub profile API)
    'NASDAQ NMS - GLOBAL MARKET',
    'NASDAQ NMS - GLOBAL SELECT MARKET',
    'NASDAQ NMS - CAPITAL MARKET',
    'NASDAQ CAPITAL MARKET',
    'NASDAQ GLOBAL MARKET',
    'NASDAQ GLOBAL SELECT MARKET',
    # Other allowed US exchanges
    'BATS',  # CBOE BZX
    'IEXG',  # IEX Exchange
}

# XASE (NYSE American) - Conditionally allowed for high-quality events only
# Analysis shows 4/11 XASE events were high-quality (major_contract, fda_approval, merger_acq + trusted sources)
# See analyze_xase_events.py for full analysis
XASE_ALLOWED_CLUSTERS = {
    'major_contract',           # TMQ: Trilogy Metals gov't investment
    'fda_approval',             # STXS: FDA clearance
    'merger_acq',               # USBC, TOPP: Strategic acquisitions
    'delisting',                # SBEV, SACH, AZTR: Short opportunities (negative catalyst)
    'contract_award_keyword',   # WYY: Government contracts (user-approved)
}
XASE_TRUSTED_SOURCES = {
    'PRNewsWire', 'GlobeNewsWire', 'Business Wire'
}

EXCLUDED_EXCHANGES = {
    # OTC Markets (exclude all OTC/pink sheets)
    'OTCM',   # OTC Markets Group - unregulated
    'OOTC',   # OTC - Other OTC
    'OTCMKTS', 'OTC', 'PINK', 'OTCQB', 'OTCQX',
    'OTC MARKETS',  # Full name from Finnhub
    # Foreign exchanges (exclude all non-US)
    'TORONTO STOCK EXCHANGE', 'TSX VENTURE EXCHANGE - NEX',
    'CANADIAN NATIONAL STOCK EXCHANGE', 'AEQUITAS NEO EXCHANGE',
    'LONDON STOCK EXCHANGE', 'AIM ITALIA - MERCATO ALTERNATIVO DEL CAPITALE',
    'BOLSA DE MADRID', 'OSLO BORS ASA', 'KOREA EXCHANGE (STOCK MARKET)',
    'TEL AVIV STOCK EXCHANGE', 'NYSE EURONEXT - EURONEXT AMSTERDAM',
    'ASX - ALL MARKETS',
}

HARD_STOP_LOSS_LONG = -1.0  # -1% hard stop for longs
HARD_STOP_LOSS_SHORT = -1.0  # -1% hard stop for shorts (opposite direction)
MAX_CONCURRENT_POSITIONS = 3
NEWS_ENTRY_DELAY = 60  # 60-second delay after news before entry (like orchestrator)

# Transaction costs (PROFESSIONAL-GRADE REALISM)
COMMISSION_BPS = 2.0  # 2 basis points commission (0.02% per leg, realistic for news trading)
SLIPPAGE_BPS = 1.5    # 1.5 bps slippage (news events = volatile, wider spreads)

# Momentum gates (OPTIMIZED from orchestrator: 0.1% price, 2.5x volume for SUSTAINED)
# ⚠️ CRITICAL: Orchestrator uses 0.1% threshold for SUSTAINED (proven in production)
SUSTAINED_PRICE_THRESHOLD = 0.1  # 0.1% price move (ORCHESTRATOR-ALIGNED)
SUSTAINED_VOLUME_RATIO = 2.5  # 2.5x average volume (ORCHESTRATOR-ALIGNED)
EXPLOSIVE_PRICE_THRESHOLD = 2.5  # 2.5% price move
EXPLOSIVE_VOLUME_RATIO = 5.0  # 5.0x average volume

# Volume Exhaustion: REMOVED - Simplified to ATR-based stops only
# Now using:
# 1. ATR-based hard stop (USE_ATR_STOP, ATR_MULTIPLIER, MIN_STOP_PCT)
# 2. Take profit at 1R multiple (TAKE_PROFIT_R_MULTIPLE)
# 3. Time stop (MAX_HOLD_SECONDS)

# Trading session and risk controls
RTH_ONLY = os.getenv('RTH_ONLY', '1') == '1'  # Only trade during extended hours
RTH_START = dtime(6, 50)   # 06:50 AM (premarket start)
RTH_END = dtime(17, 50)   # 17:50 (5:50 PM)
# Minimum price gate for tradability - RESTORED to $1.0 (Oct 27, 2025 fix)
# Analysis showed $5 filter destroyed 60% of trades and 92% of profits by removing
# explosive merger_acq moves in $1-5 range. Original $1 threshold is correct.
MIN_PRICE = float(os.getenv('MIN_PRICE', '1.0'))  # Include >= $1; override via env MIN_PRICE

# S&P 500 tickers for fractional share support on high-priced stocks
SP500_TICKERS = {
    'AAPL', 'MSFT', 'NVDA', 'AMZN', 'META', 'GOOGL', 'GOOG', 'TSLA', 'BRK.B', 'LLY',
    'AVGO', 'JPM', 'V', 'UNH', 'XOM', 'MA', 'COST', 'HD', 'WMT', 'PG',
    'NFLX', 'JNJ', 'BAC', 'ABBV', 'CRM', 'KO', 'ORCL', 'CVX', 'MRK', 'AMD',
    'ADBE', 'ACN', 'PEP', 'TMO', 'LIN', 'MCD', 'CSCO', 'ABT', 'WFC', 'IBM',
    'GE', 'CMCSA', 'CAT', 'TXN', 'QCOM', 'DHR', 'INTU', 'VZ', 'PM', 'DIS',
    'AMGN', 'COP', 'AMAT', 'NEE', 'SPGI', 'GS', 'HON', 'T', 'UNP', 'RTX',
    'AXP', 'MS', 'PFE', 'LOW', 'SYK', 'ELV', 'BLK', 'BKNG', 'NOW', 'PGR',
    'ISRG', 'ETN', 'LMT', 'TJX', 'UPS', 'VRTX', 'BSX', 'C', 'SCHW', 'ADP',
    'MMC', 'ADI', 'REGN', 'FI', 'CB', 'GILD', 'PLD', 'MU', 'MDLZ', 'TMUS',
    'SO', 'PANW', 'SLB', 'BMY', 'BX', 'DE', 'ICE', 'PYPL', 'DUK', 'AMT',
    'LRCX', 'EQIX', 'KLAC', 'CME', 'APH', 'ZTS', 'SNPS', 'CDNS', 'WM', 'MCO',
    'ANET', 'USB', 'EOG', 'PH', 'MSI', 'PNC', 'MAR', 'AON', 'ITW', 'ORLY',
    'TGT', 'CCI', 'APO', 'MO', 'NOC', 'TDG', 'ECL', 'CMG', 'WELL', 'GD',
    'CSX', 'HCA', 'CARR', 'MCK', 'AJG', 'TT', 'WMB', 'AFL', 'AZO', 'SHW',
    'EMR', 'COF', 'APD', 'PSA', 'NXPI', 'ROP', 'PCAR', 'FTNT', 'OKE', 'FCX',
    'SRE', 'NEM', 'TRV', 'GM', 'NSC', 'AIG', 'MET', 'JCI', 'ADSK', 'URI',
    'AMP', 'DHI', 'ALL', 'O', 'FICO', 'PSX', 'HLT', 'PAYX', 'KMB', 'D',
    'MSCI', 'DLR', 'CL', 'CTAS', 'SPG', 'TEL', 'FDX', 'MCHP', 'HES', 'BK',
    'F', 'CHTR', 'CPRT', 'MNST', 'PCG', 'IQV', 'A', 'KVUE', 'ROST', 'PRU',
    'SYY', 'YUM', 'FAST', 'RSG', 'EW', 'GWW', 'ODFL', 'CMI', 'CTVA', 'KMI',
    'GIS', 'ACGL', 'OTIS', 'TRGP', 'IT', 'VRSK', 'NUE', 'EA', 'FANG', 'GEHC',
    'AME', 'DAL', 'KHC', 'LULU', 'VMC', 'IDXX', 'EXC', 'KR', 'MLM', 'CBRE',
    'IR', 'XEL', 'HWM', 'DD', 'PWR', 'CTSH', 'VICI', 'MPWR', 'HSY', 'GRMN',
    'GLW', 'DOW', 'HPQ', 'RCL', 'WAB', 'HIG', 'MTB', 'ANSS', 'DXCM', 'KEYS',
    'STZ', 'ED', 'EBAY', 'PPG', 'LYB', 'NDAQ', 'TTWO', 'IFF', 'EXR', 'EFX',
    'AVB', 'ROK', 'CAH', 'TSCO', 'ETR', 'PHM', 'FITB', 'STT', 'BIIB', 'VTR',
    'AWK', 'TYL', 'CDW', 'HPE', 'WEC', 'BRO', 'WDC', 'BALL', 'CNC', 'ZBH',
    'BAX', 'AEE', 'RMD', 'LH', 'HBAN', 'CNP', 'DTE', 'AXON', 'GDDY', 'ES',
    'MTD', 'WST', 'VLTO', 'K', 'NTAP', 'FE', 'TROW', 'BR', 'STLD', 'EIX',
    'ALGN', 'DOV', 'HUBB', 'PPL', 'LVS', 'RF', 'LEN', 'EXPE', 'DRI', 'CFG',
    'BLDR', 'SBAC', 'WBD', 'CINF', 'ARE', 'CSGP', 'EPAM', 'ALB', 'FTV', 'NTRS',
    'MKC', 'PFG', 'LDOS', 'J', 'MAA', 'OMC', 'SYF', 'SWK', 'PTC', 'WAT',
    'HOLX', 'KEY', 'IRM', 'DFS', 'IP', 'EQR', 'CBOE', 'INVH', 'FSLR', 'LUV',
    'LKQ', 'FDS', 'TRMB', 'NVR', 'CE', 'ULTA', 'POOL', 'EG', 'BBY', 'APTV',
    'INCY', 'AEP', 'TER', 'CPT', 'WY', 'AMCR', 'DG', 'MOH', 'JBHT', 'VRSN',
    'CAG', 'ESS', 'GEN', 'TXT', 'AKAM', 'IEX', 'UAL', 'CHRW', 'EVRG', 'CMS',
    'SNA', 'PKG', 'IPG', 'ATO', 'PAYC', 'WRB', 'JKHY', 'NI', 'ZBRA', 'HRL',
    'ALLE', 'NDSN', 'REG', 'PNR', 'CCL', 'SWKS', 'LNT', 'ENPH', 'AOS', 'TECH',
    'HSIC', 'EXPD', 'MKTX', 'FFIV', 'CPB', 'ETSY', 'GL', 'MTCH', 'TAP', 'BF.B',
    'UDR', 'AIZ', 'GNRC', 'TPR', 'AAL', 'WYNN', 'MGM', 'NCLH', 'RL', 'HII',
    'JNPR', 'CRL', 'MOS', 'DVN', 'MRO', 'HAS', 'NWS', 'NWSA', 'BXP', 'VNO',
    'AAP', 'WHR', 'PNW', 'EMN', 'IVZ', 'LNC', 'FMC', 'SEE', 'ROL', 'APA',
    'PARA', 'FOX', 'FOXA', 'NRG', 'ALK', 'FRT', 'KIM', 'SJM', 'COO', 'DISH',
}

# ATR-based stop and 1R take-profit (TIGHTENED for 10-min max hold)
# ENABLED (Oct 27, 2025): ATR stops replace fixed percentage stops to normalize volatility
# Analysis showed 19 HARD_STOP trades with 0% WR and -$105 loss using fixed -1.07% stops
USE_ATR_STOP = os.getenv('USE_ATR_STOP', '1') == '1'  # Enabled - volatility-normalized stops
ATR_LOOKBACK_SECONDS = int(os.getenv('ATR_LOOKBACK_SECONDS', '60'))  # 60 seconds pre-entry (faster response)
ATR_MULTIPLIER = float(os.getenv('ATR_MULTIPLIER', '1.5'))  # 1.5x ATR for stops (tighter than default 2.5x)
MIN_STOP_PCT = float(os.getenv('MIN_STOP_PCT', '0.003'))  # 0.3% minimum stop floor
TAKE_PROFIT_R_MULTIPLE = float(os.getenv('TAKE_PROFIT_R_MULTIPLE', '2.0'))  # 2R take-profit

# Maximum loss per trade cap (NEW Oct 27, 2025)
# Prevents oversizing into volatile stocks - limits loss to 0.75% of capital per trade
MAX_LOSS_PER_TRADE_PCT = float(os.getenv('MAX_LOSS_PER_TRADE_PCT', '0.75'))  # 0.75% max loss cap

# Maximum hold time (10 minutes hard cap)
MAX_HOLD_SECONDS = int(os.getenv('MAX_HOLD_SECONDS', '600'))  # 10 minutes

# Phase-based exit system: Fixed TP (0-5min) → Trailing Stop (5min+)
PHASE_TRANSITION_SECONDS = int(os.getenv('PHASE_TRANSITION_SECONDS', '300'))  # Switch to trailing after 5 minutes
TRAILING_STOP_PCT = float(os.getenv('TRAILING_STOP_PCT', '0.8'))  # 0.8% trailing stop in Phase 2

# Finnhub rate limit: 50 calls/min = 1 call every 1.2 seconds
FINNHUB_RATE_LIMIT_DELAY = 1.3  # seconds between calls

# Duplicate detection cooldown
COOLDOWN_DAYS = 14  # 14-day cooldown for same event

# Optional: skip specific clusters entirely (per user policy)
# UPDATED (Oct 27, 2025): Based on IS backtest Jul-Oct 2025 - ONLY trade clusters with positive Sharpe > benchmark
# Approved clusters (beat benchmarks): merger_acq (1.80), doj_investigation (10.38), sector_tariff (16.34), war_risk_sector (6.74)
# Banned: major_contract (-1.81), contract_award_keyword (-0.76), fda_approval, phase3_achievement
APPROVED_CLUSTERS: Set[str] = set([
    'merger_acq',           # Sharpe 1.80, 111 trades, +1.10% avg return - CORE STRATEGY
    'doj_investigation',    # Sharpe 10.38, 3 trades (small sample but exceptional)
    'sector_tariff',        # Sharpe 16.34, 3 trades, 100% WR (small sample)
    'war_risk_sector',      # Sharpe 6.74, 8 trades (defensive/geopolitical risk)
])

SKIP_CLUSTERS: Set[str] = set([
    # Negative Sharpe clusters (underperform even in bull market)
    'major_contract',             # Sharpe -1.81, 32 trades, -0.05% avg return
    'contract_award_keyword',     # Sharpe -0.76, 65 trades, -0.04% avg return
    'fda_approval',               # Sharpe 0.11, 14 trades (near-zero edge)
    'phase3_achievement',         # Sharpe -0.11, 5 trades (negative edge)
    
    # Original exclusions (insufficient data or negative results)
    'analyst_action_upgrade',    # 0% WR, Sharpe -29.57
    'share_buyback',              # 28.6% WR, Sharpe 0.95 (marginal)
    'ai_enterprise_partnership',  # 32.4% WR, Sharpe -2.92
    'uplisting',                  # Only 1 trade (insufficient data)
    'failed_merger',              # Only 1 trade (insufficient data)
    'share_dilution',             # 0% WR (1 trade, -2.90%)
])

# Generic fluff exclusions (COMPLETE from news_trade_orchestrator_finnhub.py)
# ⚠️ BREAKING NEWS PROTECTION (Oct 20, 2025):
# Do NOT add patterns that match genuine breaking news:
# - "signs definitive agreement" / "definitive take-private agreement" → BREAKING merger news
# - "regulator clears merger" / "FTC approves" → BREAKING regulatory approval
# - "announces acquisition" / "to acquire for $X" → BREAKING M&A announcement
# These should NOT be filtered as fluff!

GENERIC_FLUFF_PATTERNS = [
    # Daily wraps/briefings
    r'\bmonday\s+morning\s+(?:brief|outlook|update|wrap)\b',
    r'\btuesday\s+morning\s+(?:brief|outlook|update|wrap)\b',
    r'\bwednesday\s+morning\s+(?:brief|outlook|update|wrap)\b',
    r'\bthursday\s+morning\s+(?:brief|outlook|update|wrap)\b',
    r'\bfriday\s+morning\s+(?:brief|outlook|update|wrap)\b',
    r'\bweekly\s+(?:roundup|recap|summary|digest)\b',
    r'\bmarket\s+(?:wrap|recap|summary|overview|digest)\b',
    
    # Stock lists/movers (exclude if dollar amount + specific company name)
    r'\btop\s+\d+\s+(?:stocks|movers|gainers|losers)\b',
    r'\b(?<!Among\s)stocks?\s+(?:in\s+focus|to\s+watch|making\s+moves)\b',  # Allow "Among Stocks To Watch" if specific deal mentioned
    r'\bamong\s+stocks?\s+to\s+watch\b',  # Generic stock watchlists
    r'\bpre-?market\s+(?:movers|outlook|preview)\b',
    r'\bafter-?hours\s+(?:movers|recap|summary)\b',
    
    # Speculative M&A language (question marks, "near", "could", "might")
    r'\b(?:deal|merger|acquisition)\s+near\?\b',  # "Deal Near?" headlines
    r'\b(?:could|might|may)\s+(?:merge|acquire|buy)\b',  # Speculative verbs
    
    # Earnings/analyst roundups
    r'\bearnings\s+(?:calendar|preview|season)\b',
    r'\banalyst\s+(?:ratings\s+)?(?:roundup|recap|summary)\b',
    
    # Generic what-to-know articles
    r'\bwhat\s+to\s+(?:watch|know)\s+(?:this\s+week|today)\b',
    r'\btop\s+stories\b',
    r'\bheadlines?\b.*?\bdigest\b',
    
    # Opinion/advice clickbait (from orchestrator)
    r'\b\d+\s+Reasons?\s+Why\b',
    r'\bHere\'?s\s+Why\b',
    r'\bWhat\s+(?:You|Investors)\s+(?:Need|Should)\s+(?:to\s+)?Know\b',
    r'\bShould\s+You\s+(?:Buy|Sell)\b',
    r'\bBetter\s+Buy\b',
    r'\bWhy\s+I\'?m\s+(?:Buying|Selling)\b',
    
    # Generic market commentary (from orchestrator)
    r'\bstocks?\s+bounce\b',
    r'\bJim\s+Cramer\b',
    r'\boverreacting\b',
    
    # CEO name-drops without substance (from orchestrator)
    r'\b(?:NVIDIA|NVDA|Intel|INTC|Broadcom|AVGO)\s+CEO\b',
    r'\b(?:NVIDIA|NVDA|Intel|INTC|Broadcom|AVGO)\'?s\s+(?:CEO|Huang|Gelsinger|Tan)\b',
    
    # Generic narrative fluff (from orchestrator)
    r'\b(?:Empire|Plot\s+Twist|Takes?\s+(?:a\s+)?Hit|Took\s+(?:a\s+)?Hit)\b',
    r'\b(?:Challenges?\s+Loom|Funding\s+Challenges?)\b',
    r'\brall(?:y|ies|ying)\b',
    r'\briding\b',
    r'\bsoar[s]?\b',  # REMOVED: "soar" is fluff word
    r'\bshares?\s+jump(?:ed|s)?\b',
    r'\bshares?\s+rise[s]?\b',
    r'\bshares?\s+climb[s]?\b',
    r'\bstock\s+climb[s]?\b',
    r'\bstock\s+falls?\b',
    r'\bstocks?\s+plunge[s]?\b',
    r'\bstock\s+jumps?\b',  # "stock jumps" is price movement fluff
    r'\bdrops?\s+as\b',
    r'\bAll\s+You\s+Need\s+to\s+Know\b',
    r'\bWhat\s+Does\s+It\s+Mean\b',
    r'\bBreakout\s+Watch\b',
    r'\bNo\s+Flake\b',
    r'\bBuy\s+Time\b',
    r'\bInvestors\s+Have\s+Been\s+Waiting\b',
    r'\bcatalyst\b',
    r'\bwar\s+chest\b',
    
    # Editorial opinion pieces (exclude "The Case Against/For" titles)
    r'\bThe\s+Case\s+(?:Against|For)\b',
    r'\bIs\s+Different\s+Than\b',
    
    # Buyback extensions (routine, not breaking news)
    r'\bannounces?\s+extension\s+of\s+(?:stock\s+)?buyback\s+program\b',
    r'\bextends?\s+(?:stock\s+)?buyback\s+program\b',
    
    # Product/service launches (generic, not stock-moving)
    r'\bLaunches\s+New\s+Customer\s+and\s+Partner\s+Offerings\b',
    
    # Price movement fluff (NEW - more specific)
    r'\bshares?\s+sink[s]?\b',
    r'\bstock\s+sink[s]?\b',
    r'\bshares?\s+surge[sd]?\b',
    r'\bstock\s+surge[sd]?\b',
    r'\b(?:shares?|stock)\s+slide[sd]?\b',
    r'\b(?:shares?|stock)\s+drop(?:ped|s)?\b',
    r'\b(?:shares?|stock)\s+slump[s]?\b',
    r'\bin\s+(?:the\s+)?red\b',
    r'\b(?:oil|gold|silver|copper)\s+edges?\s+(?:lower|higher)\b',  # "oil edges lower" is fluff
    r'\b(?:price|prices)\s+(?:up|down|rise[s]?|fall[s]?)\b',  # Generic price movement
    
    # Opinion/speculation/analysis (NEW)
    r'\b[Oo]verlooked\s+[Ss]tock[s]?\b',
    r'\b[Cc]ostly\s+[Mm]istake[s]?\b',
    r'\b[Ww]hat\s+[Ww]ent\s+[Ww]rong\b',
    r'\b[Ss]tock[s]?\s+[Aa]nalysis\b',
    r'\b(?:could|may|might)\s+be\s+(?:a\s+)?(?:game\s+changer|answer)\b',  # "could be a game changer"
    r'\b(?:very|extremely)\s+excited\s+about\b',  # Subjective excitement quotes
    r'\bis\s+[\'"]?very\s+excited[\'"]?\s+about\b',
    r'\bwhy\s+going\s+private\s+may\s+be\s+the\s+answer\b',  # Speculative opinion
    r'\b[Pp]ath\s+[Tt]o\s+.*?\s+[Aa]pproval\b',  # "Path To DMD Approval" - analysis, not news
    
    # Market commentary (NEW)
    r'\b(?:Lead|Leads)\s+S&P\s+500\s+[Ss]tocks\b',  # "Nvidia Partner, Rival Lead S&P 500 Stocks"
    r'\b[Ee]quity\s+[Mm]arkets?\s+[Ss]urge\b',  # "Equity Markets Surge"
    r'\b[Mm]arkets?\s+[Ss]urge\b',
    r'\b[Ss]tocks?\s+[Ss]urging\s+on\b',  # "2 Lithium Stocks Surging on..." 
    r'\b[Tt]ariffs?\s+on\s+.*?\s+(?:could|will|may)\s+(?:raise|have)\b',  # Opinion: "Tariffs on pharma could raise costs"
    r'\b(?:will|could|may)\s+have\s+(?:muted|significant|major)\s+impact\b',  # Analyst opinion
    r'\b(?:exposes?|reveals?)\s+the\s+[Rr]eal\s+[Ss]tory\b',  # "Gold Exposes the Real Story"
    r'\b[Tt]rade\s+[Ww]ar\s+[Cc]oncerns?\s+increase\b',  # Generic trade war
    r'\b[Aa]mid\s+uncertainty\s+regarding\b',  # "amid uncertainty regarding..." analysis
    
    # Speculative questions (NEW)
    r'\b[Ww]ill\s+.*?\s+[Pp]artnership\s+[Hh]elp\b',  # "Will Amazon Partnership Help Netflix..."
    r'\b[Cc]ould\s+[Bb]e\s+a\s+[Gg]ame\s+[Cc]hanger\b',  # "Could Be a Game Changer"
    r'\b[Mm]ay\s+be\s+the\s+[Aa]nswer\b',  # "may be the answer"
    
    # Technical analysis / analyst opinion fluff (NEW - user reported)
    r'\b[Bb]earish\s+[Ss]ignals?\b',  # "Bearish Signals Amid Trade War Tensions" - analytical opinion
    r'\b[Bb]ullish\s+[Ss]ignals?\b',  # Same for bullish
    r'\b[Tt]echnical\s+[Aa]nalysis\b',  # "Technical Analysis: Bearish Signals..."
    r'\b[Dd]ips?\s+[Oo]n\s+.*?\s+[Ff]umble\b',  # "Disney Dips On Q3 Fumble..." - opinionated characterization
    r'\b[Ff]umbles?\b',  # "Fumble" is opinion/characterization
    r'\bwhy\s+.*?\s+shouldn\'?t\s+concern\s+(?:long-term\s+)?investors?\b',  # "why SEC probe shouldn't concern investors" - advice
    r'\bshouldn\'?t\s+concern\s+investors?\b',  # Generic investment advice
    r'\bwhat\s+(?:investors?|traders?)\s+should\s+(?:know|do|watch)\b',  # Investment advice
    r'\b(?:surge|surges?|surged|surging)\s+\d+%\s+after\b',  # "surge 7% after Goldman upgrade" - price movement fluff
    r'\b(?:rise|rises?|rose|rising)\s+\d+%\s+after\b',  # Same for "rise"
    r'\b(?:jump|jumps?|jumped|jumping)\s+\d+%\s+after\b',  # Same for "jump"
    r'\b(?:fall|falls?|fell|falling)\s+\d+%\s+after\b',  # Same for "fall"
    r'\b(?:shares?|stock)\s+(?:skyrocket|skyrocketing|skyrockets?|skyrocketed)\s+\d+%\b',  # "shares skyrocket 200%" - price movement fluff
    r'\b(?:shares?|stock)\s+(?:soar|soaring|soars?|soared)\s+\d+%\b',  # "shares soar 150%" - price movement fluff
    r'\b(?:shares?|stock)\s+(?:plunge|plunging|plunges?|plunged)\s+\d+%\b',  # "shares plunge 50%" - price movement fluff
    
    # NEW investor attention / search volume fluff (Oct 20, 2025)
    r'\b[Ii]nvestors?\s+[Hh]eavily\s+[Ss]earch\b',  # "Investors Heavily Search Bank of America..."
    r'\b[Hh]ere\s+[Ii]s\s+[Ww]hat\s+[Yy]ou\s+(?:[Nn]eed|[Ss]hould)\s+(?:to\s+)?[Kk]now\b',  # "Here is What You Need to Know"
    r'\b[Aa]ttracting\s+[Ii]nvestor\s+[Aa]ttention\b',  # "Attracting Investor Attention"
    r'\b[Ww]hat\s+[Yy]ou\s+[Ss]hould\s+[Kk]now\b',  # Generic advice headline
    
    # NEW price movement percentage fluff (Oct 20, 2025)
    r'\b[Ii]s\s+[Uu]p\s+\d+(?:\.\d+)?%\s+[Ii]n\s+(?:[Oo]ne\s+)?(?:[Ww]eek|[Mm]onth|[Dd]ay)\b',  # "Is Up 11.93% in One Week"
    r'\b[Uu]p\s+\d+(?:\.\d+)?%\s+[Ii]n\b',  # "Up 11.93% in"
    r'\b[Dd]own\s+\d+(?:\.\d+)?%\s+[Ii]n\b',  # "Down 8.5% in"
    
    # NEW "dips" fluff - keeps coming up (Oct 20, 2025)
    r'\b[Dd]ips?\s+[Oo]n\s+[Qq]\d+\b',  # "Dips On Q3 Fumble" - earnings reaction fluff
    r'\b[Dd]ips?\s+[Aa]fter\b',  # "Dips After..." - price movement fluff
    
    # NEW rally/plunge speculation (Oct 20, 2025)
    r'\b(?:[Pp]lunge|[Pp]lunges?|[Pp]lunged|[Pp]lunging)\s+[Aa]fter\s+[Rr]ally\b',  # "Pot Stocks Plunge After Rally"
    r'\b[Ii]s\s+[Bb]etting\s+[Oo]n\s+.*?\s+[Ww]orth\s+[Tt]he\s+[Rr]isk\b',  # "Is Betting on Rescheduling Worth the Risk?" - speculative question
    
    # Tariff market reaction fluff (NEW - sector-wide policy, not company-specific)
    r'[Ss]tocks?\s+(?:[Rr]etreat|[Ff]all|[Dd]rop|[Ss]lide|[Ss]lump|[Ss]urge)\s+(?:[Aa]fter|[Oo]n|[Aa]mid)\s+.*?\s+[Tt]ariff',  # "Furniture Stocks Retreat After Trump Threatens Tariffs"
    r'\b[Tt]rump\s+(?:[Tt]hreatens?|[Aa]nnounces?|[Rr]eveals?|[Vv]ows?)\s+.*?\s+[Tt]ariff[s]?\b',  # "Trump reveals higher tariffs on furniture"
    r'\b[Tt]ariff[s]?\s+on\s+.*?\s+(?:are\s+)?coming\b',  # "tariffs on furniture are coming"
    r'\b[Tt]ariff[s]?\s+(?:threat|threats?|concerns?|fears?)\b',  # Generic tariff reaction
    r'(?:[Aa]fter|[Aa]mid|[Oo]n)\s+[Tt]ariff\s+(?:news|announcement|threat)\b',  # Market reaction to tariffs
    
    # Price movement fluff (shares fall/drop/slide/slump/surge, stock up, in red)
    r'\bshares?\s+fall[s]?\b',
    r'\bshares?\s+slide[s]?\b',
    r'\bshares?\s+drop(?:ped|s)?\b',
    r'\bshares?\s+slump[s]?\b',
    r'\bstock\s+surges?\b',
    r'\bshares?\s+in\s+red\b',
    r'\bin\s+red\b',
    r'\bslides\b',  # Generic "slides" without context
    r'\bstock\'?s\s+surge\b',
    
    # Opinion/speculation fluff
    r'\bOverlooked\s+Stock\b',
    r'\bCostly\s+Mistake\b',
    r'\bWhat\s+Went\s+Wrong\b',
    r'\bLosing\s+Its\s+Lead\b',
    r'\bAttractive\s+Equity\b',
    r'\bWinner\b',  # Generic "winner" clickbait
    r'\bmistake\b',
    r'\bStocks?\s+Respond\b',
    r'\bPrices?\s+Tank\b',
    r'\bWill\s+This\b',
    r'\bstock\s+up\b',
    
    # Generic stock lists with patterns like "Top X Tech Stocks"
    r'\bTop\s+\d+\s+\w+\s+Stocks\b',
    
    # Market reaction fluff
    r'\bStocks?\s+(?:Are\s+)?Down\b',
    r'\bCheering\b',
    r'\bJourney\s+Rides\b',
    r'\brides\b',  # Generic "rides" without context
    
    # Political/ideological fluff
    r'\bsocialism\b',
    r'\bLikely\b',  # Speculative language

    # User-requested: causal interpretation and forward-looking speculation
    r'\bgrowth\s+fuel(?:ed|led)\s+by\b',  # "Growth fueled/fuelled by OpenAI contracts"
    r'\bcould\s+drive\s+the\s+stock\s+higher\b',
    r'\bcould\s+push\s+the\s+stock\s+(?:higher|lower)\b',

    # Sports/informal metaphors and dramatic verbs for price
    r'\b(?:stock|shares?)\s+dip[s]?\b',
    r'\b(?:stock|shares?)\s+crash(?:es|ed)?\b',
    r'\b(?:stock|shares?)\s+plunge(?:s|d)?\b',
    r'\b(?:stock|shares?)\s+tank(?:s|ed)?\b',
    r'\b(?:stock|shares?)\s+collapse(?:s|d)?\b',

    # Unnecessary emotional commentary
    r'\b(?:very\s+)?excited\b',
    r'\b\'?[Ee]xcited\'?\b',

    # Price movement plus causal reasoning (too late)
    r"\b(?:drops?|fell|fall[s]?|slid(?:es|ing)?|retreat[s]?|decline[s]?|down)\b.*\bafter\b",
    r"\b(?:stocks?|shares?)\s+retreat\b",
    
    # NEW: Generic price reaction fluff (user-requested)
    r'\bstock\s+pops\b',  # "Stock Pops on Upgrade" is price movement fluff
    r'\bshares?\s+pop[s]?\b',  # "Shares Pop" same reason
    
    # NEW: Commentary/opinion fluff that was getting through (Oct 20, 2025 - user verified)
    r'\b(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+reasons?\s+to\s+(?:watch|buy|sell|avoid|consider)\b',  # "Four Reasons To Watch"
    r'\bmust-know\s+(?:numbers?|metrics?|facts?)\b',  # "3 Must-Know Numbers"
    r'\b(?:my|our)\s+favorite\s+(?:ai\s+)?stocks?\b',  # "My Favorite AI Stocks"
    r'\bone\s+of\s+(?:the\s+)?(?:best|top|my\s+favorite)\b',  # "One of My Favorite..."
    r'\bup\s+\d+%\s+from\s+(?:july|jan|feb|mar|apr|may|jun|aug|sep|oct|nov|dec)\b',  # "Up 125% From July Breakout"
    r'\bcan\s+.{5,50}\s+redefine\b',  # "Can Affirm & Google's AP2 Partnership Redefine..." (max 50 chars between)
    r'\bapplauds?\s+.{5,50}\s+partnership\b',  # "Rockland Resources Applauds Kairos-Google ... Partnership" (max 50 chars)
    r'\b(?:stock|shares?)\s+(?:is|are)\s+rising\b',  # "AMD Stock Is Rising"
    r'\b(?:is|are)\s+boosting\s+the\s+(?:stock|shares?|chip\s+maker)\b',  # "An ... Partnership and an Upgrade Are Boosting the Chip Maker"
    r'\bhas\s+(?:a|an)\s+\w+\s+problem\b',  # "Apple has an AI problem"
    r'\bpartner(?:ship)?\s+is\s+one\s+of\s+my\s+favorite\b',  # "This Nvidia Partner Is One of My Favorite..."
    
    # NEW: Price movement + upgrade/downgrade fluff (Oct 21, 2025 - user verified)
    r'\b(?:stock|shares?)\s+(?:steers?|steer|steering)\s+(?:higher|lower)\s+on\s+\w+\s+upgrade\b',  # "Stock Steers Higher on UBS Upgrade"
    r'\b(?:fizz|fizzes|fizzed|fizzing)\s+on\s+\w+\s+upgrade\b',  # "Fevertree fizzes on Jefferies upgrade"
    
    # NEW: Share dilution / secondary offerings / convertible debt (Oct 21, 2025 - negative but routine)
    # These are typically negative news but not immediate breaking news for momentum trades
    r'\bSEC\s+[Ff]ilings?\s+[Rr]eveal\s+[Ss]ignificant\s+[Dd]ilution\b',  # "SEC Filings Reveal Significant Dilution..."
    r'\b[Uu]ndisclosed\s+[Cc]onvertible\s+[Ss]ecurities\b',  # "...Previously Undisclosed Convertible Securities"
    r'\b[Ee]xisting\s+[Ss]hareholders\s+[Ff]ace\s+[Dd]ilution\b',  # "Existing Shareholders Face Dilution..."
    r'\b[Ii]ssues?\s+[Nn]ew\s+[Ss]hares?\b',  # "...Issues New Shares"
    r'\b[Ff]iles?\s+to\s+[Ss]ell\s+\d+(?:\.\d+)?\s+[Mm]illion\s+[Aa]dditional\s+[Ss]hares?\b',  # "Files to Sell 10 Million Additional Shares"
    r'\b[Bb]ondholders?\s+[Cc]onvert\s+\$\d+(?:\.\d+)?[MB]\s+[Dd]ebt\s+to\s+[Ee]quity\b',  # "Bondholders Convert $200M Debt to Equity"
    r'\b[Cc]onvert(?:s|ed|ing)?\s+\$?\d+(?:\.\d+)?[MB]?\s+[Dd]ebt\s+to\s+[Ee]quity\b',  # Generic debt-to-equity conversion
    r'\b[Aa]dding\s+\d+(?:\.\d+)?\s+[Mm]illion\s+[Ss]hares?\b',  # "Adding 15 Million Shares"
    r'\b[Gg]enerous\s+[Ss]tock\s+[Aa]wards?\s+[Dd]ilute\b',  # "Generous Stock Awards Dilute Shareholders"
    r'\b[Dd]ilute\s+[Ss]hareholders?\s+by\s+\d+%',  # "Dilute Shareholders by 3%"
    r'\b[Dd]iluting\s+[Ss]hareholders?\s+by\s+(?:\d+|X)%',  # "Diluting Shareholders by X%" or "by 3%"
    r'\b[Mm]illions?\s+of\s+[Ww]arrants?\s+[Ee]xercised\b',  # "Millions of Warrants Exercised..."
    r'\b[Ee]stablishes?\s+\$\d+(?:\.\d+)?[MB]\s+ATM\s+[Pp]rogram\b',  # "Establishes $250M ATM Program"
    r'\b[Aa]t-[Tt]he-[Mm]arket\s+[Pp]rogram\b',  # "At-The-Market Program"
    r'\bATM\s+[Pp]rogram\s+for\s+[Oo]pportunistic\s+[Cc]apital\s+[Rr]aises?\b',  # "ATM Program for Opportunistic Capital Raises"
    r'\b[Ii]ssues?\s+[Pp]referred\s+[Ss]hares?\s+with\s+[Cc]onversion\s+[Rr]ights?\b',  # "Issues Preferred Shares with Conversion Rights"
    r'\bto\s+[Ss]trategic\s+[Ii]nvestor\b',  # "...to Strategic Investor"
    r'\b[Ff]iles?\s+[Ss]helf\s+[Rr]egistration\b',  # "Files Shelf Registration"
    r'\b[Ss]helf\s+[Rr]egistration\s+for\s+up\s+to\s+\$\d+(?:\.\d+)?[MB]\b',  # "Shelf Registration for up to $500M"
    r'\bin\s+[Ff]uture\s+[Ss]tock\s+[Ss]ales?\b',  # "...in Future Stock Sales"
    r'\b[Rr]aises?\s+\$\d+(?:\.\d+)?[MB]\s+[Tt]hrough\s+[Cc]onvertible\s+[Ss]enior\s+[Nn]otes?\b',  # "Raises $500M Through Convertible Senior Notes"
    r'\b[Cc]onvertible\s+[Ss]enior\s+[Nn]otes?\s+[Oo]ffering\b',  # "Convertible Senior Notes Offering"
    r'\b[Ii]ssues?\s+\$\d+(?:\.\d+)?[MB]\s+[Bb]onds?\s+with\s+[Ww]arrants?\b',  # "Issues $150M Bonds with Warrants"
    r'\b[Ll]aunches?\s+[Oo]fferings?\s+to\s+[Ff]und\b',  # "Launches Offering to Fund..."
    r'\bto\s+[Rr]aise\s+\$\d+(?:\.\d+)?(?:[BMb]|[Bb]illion|[Mm]illion)\b',  # "to raise $2.5B" or "to raise $500M"
    r'\bfor\s+(?:business|global)?\s*[Ee]xpansion\b',  # "for expansion" or "for business expansion"
    r'\b[Oo]ffering\s+to\s+[Ff]und\s+[Gg]rowth\b',  # "Offering to Fund Growth"
]
GENERIC_FLUFF_RE = [re.compile(p, re.IGNORECASE) for p in GENERIC_FLUFF_PATTERNS]

# Default strategy parameters (can be overridden via Optuna or runtime overrides)
DEFAULT_STRATEGY_PARAMS: Dict[str, Any] = {
    'USE_ATR_STOP': USE_ATR_STOP,
    'ATR_MULTIPLIER': ATR_MULTIPLIER,
    'ATR_LOOKBACK_SECONDS': ATR_LOOKBACK_SECONDS,
    'MIN_STOP_PCT': MIN_STOP_PCT,
    'TAKE_PROFIT_R_MULTIPLE': TAKE_PROFIT_R_MULTIPLE,
    'NEWS_ENTRY_DELAY': NEWS_ENTRY_DELAY,
    # Store sustained price threshold as decimal fraction (0.001 == 0.1%)
    'SUSTAINED_PRICE_THRESHOLD': SUSTAINED_PRICE_THRESHOLD / 100.0,
    'SUSTAINED_VOLUME_RATIO': SUSTAINED_VOLUME_RATIO,
    # VOLUME_EXHAUSTION_THRESHOLD removed - using ATR stops only
    'MAX_HOLD_SECONDS': MAX_HOLD_SECONDS,
    # Phase-based exits: Fixed TP (0-5min) → Trailing Stop (5min+)
    'PHASE_TRANSITION_SECONDS': PHASE_TRANSITION_SECONDS,
    'TRAILING_STOP_PCT': TRAILING_STOP_PCT,
}


def _empty_stats(final_capital: float = INITIAL_CAPITAL) -> Dict[str, float]:
    """Return default stats structure when the backtest cannot run."""
    return {
        'total_trades': 0,
        'win_rate': 0.0,
        'avg_return': 0.0,
        'total_pnl': 0.0,
        'sharpe_ratio': 0.0,
        'max_drawdown': 0.0,
        'final_capital': final_capital,
    }

# ============================================================================
# LONG EVENT CLUSTER PATTERNS (COMPREHENSIVE)
# ============================================================================

# 1. ANALYST ACTIONS - Enhanced with word boundaries and specificity (UPGRADES ONLY FOR LONGS)
ANALYST_UPGRADE_PATTERNS = [
    # Core upgrades with firm names (word boundaries) — firm mention REQUIRED
    r"\b(?:BofA|Bank\s+of\s+America|Goldman|Goldman\s+Sachs|Jefferies|HSBC|KeyBanc|KeyBanc\s+Capital\s+Markets|Piper|Piper\s+Sandler|JPMorgan|JP\s?Morgan|Wedbush|Oppenheimer|Cantor|Cantor\s+Fitzgerald|Barclays|Morgan\s?Stanley|Citigroup|Citi(?:\s|$)|UBS|Deutsche\s?Bank|Raymond\s?James|Needham|Stifel|Wells\s?Fargo|RBC|Royal\s+Bank\s+of\s+Canada|BMO|Bank\s+of\s+Montreal|Truist|Mizuho|TD\s?Cowen|Cowen|Evercore|Evercore\s+ISI|Bernstein|AllianceBernstein|Baird|Benchmark|BTIG|Craig-Hallum|H\.?C\.?\s?Wainwright|JMP|Lake\s?Street|Maxim\s?Group|Northland|Roth\s?Capital|William\s?Blair|Wolfe|Wolfe\s+Research|Canaccord|Canaccord\s+Genuity|Redburn|Redburn\s+Atlantic|Melius|Melius\s+Research|DA\s+Davidson|D\.A\.\s+Davidson|Guggenheim|Macquarie|Susquehanna|Berenberg|Soci[eé]t[eé]\s+G[eé]n[eé]rale|SocGen|BNP\s+Paribas|Exane|Scotiabank|Scotia\s+Capital|KBW|Keefe,\s*Bruyette\s*&\s*Woods|Seaport\s+Research\s+Partners|Argus|New\s+Street\s+Research|Telsey|MoffettNathanson)\b\s+(?:Securities?|Markets?|Research|Capital|Capital\s+Markets)?\s*upgrade[sd]?",

    # Upgrades to specific bullish ratings
    r"\b(?:BofA|Bank\s+of\s+America|Goldman|Goldman\s+Sachs|Jefferies|HSBC|KeyBanc|KeyBanc\s+Capital\s+Markets|Piper|Piper\s+Sandler|JPMorgan|JP\s?Morgan|Wedbush|Oppenheimer|Cantor|Cantor\s+Fitzgerald|Barclays|Morgan\s?Stanley|Citigroup|Citi(?:\s|$)|UBS|Deutsche\s?Bank|Raymond\s?James|Needham|Stifel|Wells\s?Fargo|RBC|Royal\s+Bank\s+of\s+Canada|BMO|Bank\s+of\s+Montreal|Truist|Mizuho|TD\s?Cowen|Cowen|Evercore|Evercore\s+ISI|Bernstein|AllianceBernstein|Baird|Benchmark|BTIG|Craig-Hallum|H\.?C\.?\s?Wainwright|JMP|Lake\s?Street|Maxim\s?Group|Northland|Roth\s?Capital|William\s?Blair|Wolfe|Wolfe\s+Research|Canaccord|Canaccord\s+Genuity|Redburn|Redburn\s+Atlantic|Melius|Melius\s+Research|DA\s+Davidson|D\.A\.\s+Davidson|Guggenheim|Macquarie|Susquehanna|Berenberg|Soci[eé]t[eé]\s+G[eé]n[eé]rale|SocGen|BNP\s+Paribas|Exane|Scotiabank|Scotia\s+Capital|KBW|Keefe,\s*Bruyette\s*&\s*Woods|Seaport\s+Research\s+Partners|Argus|New\s+Street\s+Research|Telsey|MoffettNathanson)\b\s+(?:Securities?|Markets?|Research|Capital|Capital\s+Markets)?\s*upgrade[sd]?\s+to\s+\b(?:buy|strong\s+buy|outperform|overweight)\b",

    # Price target raises with firm mention
    r"\b(?:BofA|Bank\s+of\s+America|Goldman|Goldman\s+Sachs|Jefferies|HSBC|KeyBanc|KeyBanc\s+Capital\s+Markets|Piper|Piper\s+Sandler|JPMorgan|JP\s?Morgan|Wedbush|Oppenheimer|Cantor|Cantor\s+Fitzgerald|Barclays|Morgan\s?Stanley|Citigroup|Citi(?:\s|$)|UBS|Deutsche\s?Bank|Raymond\s?James|Needham|Stifel|Wells\s?Fargo|RBC|Royal\s+Bank\s+of\s+Canada|BMO|Bank\s+of\s+Montreal|Truist|Mizuho|TD\s?Cowen|Cowen|Evercore|Evercore\s+ISI|Bernstein|AllianceBernstein|Baird|Benchmark|BTIG|Craig-Hallum|H\.?C\.?\s?Wainwright|JMP|Lake\s?Street|Maxim\s?Group|Northland|Roth\s?Capital|William\s?Blair|Wolfe|Wolfe\s+Research|Canaccord|Canaccord\s+Genuity|Redburn|Redburn\s+Atlantic|Melius|Melius\s+Research|DA\s+Davidson|D\.A\.\s+Davidson|Guggenheim|Macquarie|Susquehanna|Berenberg|Soci[eé]t[eé]\s+G[eé]n[eé]rale|SocGen|BNP\s+Paribas|Exane|Scotiabank|Scotia\s+Capital|KBW|Keefe,\s*Bruyette\s*&\s*Woods|Seaport\s+Research\s+Partners|Argus|New\s+Street\s+Research|Telsey|MoffettNathanson)\b\s+(?:Securities?|Markets?|Research|Capital|Capital\s+Markets)?\s*(?:raise[sd]?|lift[s]?|boost[s]?|hike[sd]?|increase[sd]?)\s+(?:price\s)?target\s+to\s+\$\d+",

    # Price target without explicit raise/cut but requiring firm mention
    r"\b(?:BofA|Bank\s+of\s+America|Goldman|Goldman\s+Sachs|Jefferies|HSBC|KeyBanc|KeyBanc\s+Capital\s+Markets|Piper|Piper\s+Sandler|JPMorgan|JP\s?Morgan|Wedbush|Oppenheimer|Cantor|Cantor\s+Fitzgerald|Barclays|Morgan\s?Stanley|Citigroup|Citi(?:\s|$)|UBS|Deutsche\s?Bank|Raymond\s?James|Needham|Stifel|Wells\s?Fargo|RBC|Royal\s+Bank\s+of\s+Canada|BMO|Bank\s+of\s+Montreal|Truist|Mizuho|TD\s?Cowen|Cowen|Evercore|Evercore\s+ISI|Bernstein|AllianceBernstein|Baird|Benchmark|BTIG|Craig-Hallum|H\.?C\.?\s?Wainwright|JMP|Lake\s?Street|Maxim\s?Group|Northland|Roth\s?Capital|William\s?Blair|Wolfe|Wolfe\s+Research|Canaccord|Canaccord\s+Genuity|Redburn|Redburn\s+Atlantic|Melius|Melius\s+Research|DA\s+Davidson|D\.A\.\s+Davidson|Guggenheim|Macquarie|Susquehanna|Berenberg|Soci[eé]t[eé]\s+G[eé]n[eé]rale|SocGen|BNP\s+Paribas|Exane|Scotiabank|Scotia\s+Capital|KBW|Keefe,\s*Bruyette\s*&\s*Woods|Seaport\s+Research\s+Partners|Argus|New\s+Street\s+Research|Telsey|MoffettNathanson)\b\s+(?:Securities?|Markets?|Research|Capital|Capital\s+Markets)?\s*(?:target\s?price|PT|price\starget)\s+(?:raised|boosted|hiked|lifted|increased)\b",

    # New coverage initiation with buy/outperform/overweight — firm mention REQUIRED
    r"\b(?:BofA|Bank\s+of\s+America|Goldman|Goldman\s+Sachs|Jefferies|HSBC|KeyBanc|KeyBanc\s+Capital\s+Markets|Piper|Piper\s+Sandler|JPMorgan|JP\s?Morgan|Wedbush|Oppenheimer|Cantor|Cantor\s+Fitzgerald|Barclays|Morgan\s?Stanley|Citigroup|Citi(?:\s|$)|UBS|Deutsche\s?Bank|Raymond\s?James|Needham|Stifel|Wells\s?Fargo|RBC|Royal\s+Bank\s+of\s+Canada|BMO|Bank\s+of\s+Montreal|Truist|Mizuho|TD\s?Cowen|Cowen|Evercore|Evercore\s+ISI|Bernstein|AllianceBernstein|Baird|Benchmark|BTIG|Craig-Hallum|H\.?C\.?\s?Wainwright|JMP|Lake\s?Street|Maxim\s?Group|Northland|Roth\s?Capital|William\s?Blair|Wolfe|Wolfe\s+Research|Canaccord|Canaccord\s+Genuity|Redburn|Redburn\s+Atlantic|Melius|Melius\s+Research|DA\s+Davidson|D\.A\.\s+Davidson|Guggenheim|Macquarie|Susquehanna|Berenberg|Soci[eé]t[eé]\s+G[eé]n[eé]rale|SocGen|BNP\s+Paribas|Exane|Scotiabank|Scotia\s+Capital|KBW|Keefe,\s*Bruyette\s*&\s*Woods|Seaport\s+Research\s+Partners|Argus|New\s+Street\s+Research|Telsey|MoffettNathanson)\b\s+(?:Securities?|Markets?|Research|Capital|Capital\s+Markets)?\s*initiate[sd]?\s+coverage\s+(?:with|at)\s+\b(?:buy|strong\s+buy|outperform|overweight)\b",
]

# NEGATIVE PATTERNS for analyst upgrades (avoid false matches)
ANALYST_UPGRADE_NEGATIVE_PATTERNS = [
    r"\bseeking\s+analyst\b",  # Job postings
    r"\bhire[sd]?\s+analyst\b",  # Hiring news
    r"\banalyst\s+(?:position|role|job)\b",  # Career-related
    r"\bbecome\s+(?:an?\s+)?analyst\b",  # Career advice
    r"\bdata\s+analyst\b",  # Different job role
    # Note: downgrade/cut patterns removed - handled by analyst_downgrade_sell SHORT cluster
]

# 2. MAJOR CONTRACTS - Enhanced with entity types and exclusions
CONTRACT_PATTERNS = [
    # Action verb + dollar amount + contract (most specific)
    r"\b(?:win[s]?|won|secure[sd]?|secured|award(?:ed|s)?|receive[sd]?|land[s]?|landed|sign[s]?|signed)\s+(?:a\s+)?(?:\$\d+(?:\.\d+)?(?:M|B|million|billion)|multi-(?:million|billion))\s+\b(?:contract|order|deal|purchase\s?order)\b",
    
    # Dollar value + contract keywords
    r"\$\d+(?:\.\d+)?(?:M|B|million|billion)\s+\b(?:contract|award|deal|order|agreement|purchase\s?order)\b",
    
    # Government/military contracts (comprehensive)
    r"\b(?:Pentagon|DoD|Department\s+of\s+Defense|U\.?S\.?\s+(?:Navy|Army|Air\s?Force|Space\s?Force|Marines?)|NASA|DARPA|DHS|Department\s+of\s+Homeland\s+Security|VA|Veterans?\s+Affairs|GSA|General\s+Services\s+Administration|DOE|EPA|NIH|HHS)\b\s+(?:award[s]?|contract|order)",
    
    # ENHANCED BREAKING NEWS CONTRACT PATTERNS (user-requested keyword logic)
    # Pattern 5: "US Army" + "contracts" + "$XB" + "deal" / "$XM" / "$X.XB"
    r"\b(?:US|U\.?S\.?)\s+Army\b.*?\bcontracts?\b.*?\$\d+(?:\.\d+)?(?:M|B|million|billion)\b.*?\bdeal\b",
    r"\b(?:US|U\.?S\.?)\s+Army\b.*?\$\d+(?:\.\d+)?(?:M|B|million|billion)\b.*?\bcontracts?\b",
    
    # Public-Private Partnerships (MP Materials case)
    r"\b(?:public-private|public/private)\s+partnership\b",
    r"\bstrategic\s+investment\s+by\s+(?:US\s+)?(?:federal\s+)?government\b",  # Trilogy Metals case
    
    # ENHANCED: Government stake patterns (user-requested fix for LAC case - Sept 30)
    # Pattern 1: Main pattern - "government to take/acquire stake" with flexible word distance
    r"\b(?:US\s+|U\.?S\.?\s+)?(?:government|federal).*?(?:take|takes|taking|took|acquire|acquiring|purchase|purchasing|buy|buying).*?(?:\d+%\s+)?(?:stake|equity|investment|interest)\b",
    # Pattern 2: Catches "U.S. to take equity stake" with optional "to"
    r"\b(?:US\b|U\.?S\.?).*?(?:to\s+)?(?:take|takes|taking|acquire|acquiring).*?(?:equity\s+)?stake\b",
    # Pattern 3: Catches "U.S. is taking an equity stake" with progressive verbs
    r"\b(?:US\b|U\.?S\.?)\s+(?:is\s+|will\s+be\s+)?(?:taking|acquiring).*?(?:equity\s+)?stake\b",
    # Pattern 4: Reverse order - "stake in/by government"
    r"\b(?:equity\s+)?(?:stake|interest)\s+(?:in|by)\s+(?:government|federal)\b",
    
    # OLD (too strict - expects adjacent words) - kept for reference:
    # r"\b(?:government|federal)\s+(?:stake|investment|partnership)\b",  # Lithium Americas case

    
    # Multi-year deals with context
    r"\b(?:multi-year|multiyear)\b\s+\$?\d+(?:\.\d+)?(?:M|B|million|billion)?\s+\b(?:contract|agreement|deal)\b",
    
    # Named partner contracts (major tech/defense)
    r"\b(?:Amazon|AWS|Microsoft|Azure|Google|GCP|Alphabet|Apple|Meta|Tesla|SpaceX|Boeing|Lockheed\s?Martin?|Raytheon|Northrop\s?Grumman?|General\s?Dynamics)\b\s+(?:award[s]?|select[s]?|choose[s]?|partner[s]?)\s+\w+\s+for\s+\$?\d+",
    
    # International defense contracts
    r"\b(?:NATO|European\s+Union|EU|UK|MoD|Ministry\s+of\s+Defence)\b\s+\b(?:contract|order|award)\b",
]

# CONTRACT NEGATIVE PATTERNS (exclude asset sales, only catch contract losses)
CONTRACT_NEGATIVE_PATTERNS = [
    r"\blost\s+(?:a\s+)?(?:\$\d+(?:\.\d+)?(?:M|B|million|billion)\s+)?contract\b",
    r"\bcontract\s+(?:dispute|terminated|cancelled|suspended|withdrawn)\b",
    r"\bfail(?:ed|s)?\s+to\s+(?:win|secure|retain)\s+(?:a\s+)?contract\b",
    r"\bpassed\s+over\s+for\s+contract\b",
    r"\bcontract\s+(?:award[s]?|awarded)\s+to\s+(?:competitor|rival)\b",
    # Asset sales (divestiture) - more specific to avoid blocking product sales
    r"\b(?:deal|agreement)\s+to\s+sell\s+(?:its\s+)?(?:stake|interest|business|division|unit|subsidiary|operations?)\b",
    r"\bsell[s]?\s+(?:its\s+)?(?:stake|interest|business|division|unit|subsidiary|operations?|assets?)\s+(?:to|for)\b",
    r"\bdivest(?:s|ing|iture)?\s+(?:of\s+)?(?:its\s+)?(?:stake|business|division|unit)\b",
]

# 3. PHASE 3 CLINICAL - Enhanced with specific trial language
PHASE3_PATTERNS = [
    # Primary/secondary endpoints with statistical language
    r"\bphase\s?(?:3|III|three)\s+(?:trial|study)\s+(?:met|meet[s]?|exceeded|achieve[sd]?|hit[s]?)\s+(?:its\s+)?(?:primary|co-primary|secondary|all)\s+endpoint[s]?\b",
    
    # Statistical significance (key for clinical trials)
    r"\bphase\s?(?:3|III|three)\b.*?\b(?:statistically\s+significant|p\s?[<≤]\s?0\.0\d+|significant\s+(?:improvement|benefit|reduction))\b",
    
    # Top-line results (common term)
    r"\b(?:positive|successful|encouraging)\s+top-?line\s+(?:results|data)\s+from\s+phase\s?(?:3|III|three)\b",
    
    # FDA/EMA interactions with Phase 3 data
    r"\b(?:FDA|EMA|European\s+Medicines\s+Agency|Health\s+Canada|PMDA)\s+(?:accept[s]?|review[s]?|grant[s]?)\s+.*?phase\s?(?:3|III|three)\s+(?:data|results)\b",
    
    # Breakthrough/fast track designation
    r"\b(?:breakthrough\s+therapy|fast\s+track|priority\s+review)\s+designation\b.*?\bphase\s?(?:3|III|three)\b",
    
    # NDA/BLA submission based on Phase 3
    r"\b(?:submit[s]?|file[sd]?)\s+(?:NDA|BLA|New\s+Drug\s+Application|Biologics\s+License\s+Application)\b.*?\bbased\s+on\s+phase\s?(?:3|III|three)\b",
    
    # Generic endpoint achievement
    r"\bphase\s?(?:3|III|three)\s+(?:trial|study)\b.*?\b(?:reached|attained|passed|cleared|satisfied|demonstrated)\s+(?:primary|secondary)?\s?endpoint\b",
]

# PHASE 3 NEGATIVE PATTERNS
PHASE3_NEGATIVE_PATTERNS = [
    r"\bphase\s?(?:3|III|three)\s+(?:fail[s]?|failed|miss(?:ed|es)?|did\s+not\s+meet)\b",
    r"\bdiscontinue[sd]?\s+phase\s?(?:3|III|three)\b",
    r"\bhalt(?:ed|s)?\s+phase\s?(?:3|III|three)\b",
    r"\bphase\s?(?:3|III|three)\s+(?:delayed|suspended|on\s+hold)\b",
]

# 4. SHARE BUYBACK - Enhanced with board/authorization language
BUYBACK_PATTERNS = [
    # Board approval with dollar amounts
    r"\bboard\s+(?:approve[sd]?|authorized?)\s+(?:a\s+)?(?:new\s+)?\$\d+(?:\.\d+)?(?:M|B|million|billion)\s+(?:share\s+)?(?:buyback|repurchase)\s+\b(?:program|plan|authorization)\b",
    
    # Expanded/increased buyback programs
    r"\b(?:expand[s]?|expanded|increase[sd]?|raise[sd]?|boost[s]?)\s+(?:share\s+)?(?:buyback|repurchase)\s+(?:program|authorization)\s+(?:to|by)\s+\$\d+(?:\.\d+)?(?:M|B|million|billion)\b",
    
    # Accelerated share repurchase (ASR)
    r"\baccelerated\s+share\s+repurchase|ASR\s+agreement|ASR\s+program\b",
    
    # Direct dollar amount mentions
    r"\$\d+(?:\.\d+)?(?:M|B|million|billion)\s+(?:share\s+)?(?:buyback|repurchase|share\s+repurchase)\s+\b(?:program|plan|authorization|approved|authorized)\b",
    
    # Action verbs + buyback
    r"\b(?:announce[sd]?|authorize[sd]?|approve[sd]?|unveil[s]?|reveal[s]?)\s+.*?\b(?:buyback|repurchase)\s+(?:program|plan)\b",
]

# 5. UPLISTING - Enhanced with regulatory milestones
UPLISTING_PATTERNS = [
    # Direct uplisting announcements
    r"\b(?:uplist[s]?|uplisted|uplisting)\s+(?:to|from\s+\w+\s+to)\s+\b(?:NYSE|Nasdaq|NASDAQ|New\s+York\s+Stock\s+Exchange)\b",
    
    # Transfer/graduation language
    r"\b(?:transfer[s]?|transferred|transferring|graduate[sd]?|graduating|move[sd]?|moving)\s+(?:from\s+\w+\s+)?to\s+(?:the\s+)?\b(?:NYSE|Nasdaq|NASDAQ|New\s+York\s+Stock\s+Exchange)\b",
    
    # Regulatory approval received
    r"\b(?:receive[sd]?|received|obtain[s]?|obtained|secure[sd]?|secured)\s+\b(?:NYSE|Nasdaq|NASDAQ)\b\s+(?:listing\s+)?(?:approval|clearance|authorization)\b",
    
    # Effective date announcements
    r"\b(?:commence[s]?|begin[s]?|start[s]?)\s+trading\s+on\s+(?:the\s+)?\b(?:NYSE|Nasdaq|NASDAQ)\b.*?(?:effective|on)\s+\d+",
    
    # Common ticker symbol change language
    r"\bticker\s+symbol\b.*?\b(?:change[sd]?|transition[s]?)\b.*?\b(?:NYSE|Nasdaq|NASDAQ)\b",
]

# UPLISTING NEGATIVE PATTERNS (kept minimal - delisting moved to its own SHORT cluster)
UPLISTING_NEGATIVE_PATTERNS = [
    # Keep only patterns that would falsely trigger uplisting detection
    r"\bdownlist(?:ed|ing|s)?\b",  # Downlisting is opposite of uplisting
]

# 6. AI ENTERPRISE PARTNERSHIPS (from orchestrator)
AI_PARTNERSHIP_PATTERNS = [
    r"\bAI\b\s+(?:strategic\s+)?partnership",
    r"(?:OpenAI|Anthropic|IBM|AMD|Nvidia|Oracle|Microsoft|Google|Amazon|Meta).*(?:partners?|partnership)",
    r"partners?\s+(?:with\s+)?(?:OpenAI|Anthropic|IBM|AMD|Nvidia|Oracle|Microsoft|Google|Amazon|Meta)",
    r"(?:OpenAI|Anthropic|IBM|AMD|Nvidia|Oracle|Microsoft|Google|Amazon|Meta).*\bannounce.*(?:massive|major|huge|significant).*(?:computing|ai|cloud)\s+deal\b",
    r"\bannounce.*(?:massive|major|huge|significant).*(?:computing|ai|cloud)\s+deal.*(?:OpenAI|Anthropic|IBM|AMD|Nvidia|Oracle|Microsoft|Google|Amazon|Meta)\b",
    r"\bsign.*\$\d+.*billion.*computing\s+deal\b",  # Oracle + OpenAI case
    r"\b\$\d+.*billion.*computing\s+deal\b",  # Oracle + OpenAI case (generic)
]

# 7. MERGER/ACQUISITION ANNOUNCEMENTS (acquisitions only - no asset sales)
# Note: Asset sales removed - too context-dependent (premium sale vs distressed sale)
MERGER_ACQ_PATTERNS = [
    r"merger\s+announced",
    r"acquisition\s+announced",
    r"announces?\s+.*acquisition",
    r"\bwill\s+acquire\b|\bagrees?\s+to\s+acquire\b",
    r"\b(deal|takeover|buyout)\s+to\s+go\s+private\b",
    r"\bgoing\s+private\b",
    r"\bnear.*\$\d+.*billion.*deal\b",
    r"\bnear.*\$\d+.*billion.*deal.*to\s+go\s+private\b",  # EA case: "Near... $50B Deal to Go Private"
    r"\b(take|taking|taken)\s+private\b",
    r"\bstrategic\s+acquisition\b",  # Kuke Music case
    r"\bacquisition\s+of\s+controlling\s+interest\b",  # Kuke Music case
    r"\bcontrolling\s+interest\b",  # General controlling interest acquisitions
    
    # ENHANCED BREAKING NEWS M&A PATTERNS (user-requested keyword logic)
    # Pattern 1: "to buy" + "in $X deal"
    r"\bto\s+buy\b.*?\bin\s+\$\d+(?:\.\d+)?(?:M|B|million|billion)\s+deal\b",
    
    # Pattern 2: "to acquire" + "in $X deal" 
    r"\bto\s+acquire\b.*?\bin\s+\$\d+(?:\.\d+)?(?:M|B|million|billion)\s+deal\b",
    
    # Pattern 3: "to go private in $X deal"
    r"\bto\s+go\s+private\s+in\s+\$\d+(?:\.\d+)?(?:M|B|million|billion)\s+deal\b",
    
    # Pattern 4: "to acquire" + "in $XM deal" / "in $XB deal" / "in $X.XB deal"
    r"\bto\s+acquire\b.*?\bin\s+\$\d+(?:\.\d+)?[MB]\s+deal\b",
    
    # Pattern 6: "Partner with" + "to Take" + "Private"
    r"\bpartner(?:\s+with)?\b.*?\bto\s+take\b.*?\bprivate\b",
]

# 8. FDA APPROVALS (from orchestrator)
FDA_APPROVAL_PATTERNS = [
    r"announces?\s+fda\s+approval",
    r"fda\s+approves",
    r"\breceive[sd]?\s+fda\s+approval\b",
    r"\bfda\s+(?:clear[s]?|cleared|clearance)\b",
]

# 9. CONTRACT AWARDS - KEYWORD-BASED (from orchestrator)
CONTRACT_AWARD_KEYWORD_PATTERNS = [
    r"\b(wins?|awarded|signs?|lands?)\s+.*contract\b",
]

# ============================================================================
# SHORT EVENT CLUSTER PATTERNS (11 NEGATIVE CATALYSTS)
# ============================================================================

# 1. PHASE 3 FAILURES
PHASE3_FAILURE_PATTERNS = [
    r"\bphase\s?(?:3|III|three)\s+(?:trial|study)\s+(?:fail[s]?|failed|miss(?:ed|es)?|did\s+not\s+meet)\b",
    r"\bphase\s?(?:3|III|three)\b.*?\b(?:fail[s]?|failed)\s+to\s+(?:meet|achieve|reach)\s+(?:primary|secondary)?\s?endpoint\b",
    r"\bdiscontinue[sd]?\s+phase\s?(?:3|III|three)\b",
    r"\bhalt(?:ed|s)?\s+phase\s?(?:3|III|three)\b",
    r"\bphase\s?(?:3|III|three)\s+(?:delayed|suspended|on\s+hold)\b",
    r"\bphase\s?(?:3|III|three)\b.*?\b(?:unsuccessful|disappointing|negative)\s+(?:results|data|outcome)\b",
]

# 2. FDA REJECTIONS / CRLs (Complete Response Letters)
FDA_REJECTION_PATTERNS = [
    r"\bfda\s+(?:reject[s]?|rejected|rejection)\b",
    r"\bcomplete\s+response\s+letter|CRL\b",
    r"\bfda\s+(?:decline[sd]?|denied|denial)\b",
    r"\bfda\s+(?:refuse[sd]?|refusal)\b",
    r"\bfda\s+(?:delay[s]?|delayed)\s+(?:approval|decision)\b",
    r"\bfda\s+request[s]?\s+additional\s+(?:data|information|studies)\b",
]

# 3. ANALYST DOWNGRADES TO SELL
ANALYST_DOWNGRADE_SELL_PATTERNS = [
    # All downgrade patterns now REQUIRE firm mention as requested
    r"\b(?:BofA|Bank\s+of\s+America|Goldman|Goldman\s+Sachs|Jefferies|HSBC|KeyBanc|KeyBanc\s+Capital\s+Markets|Piper|Piper\s+Sandler|JPMorgan|JP\s?Morgan|Wedbush|Oppenheimer|Cantor|Cantor\s+Fitzgerald|Barclays|Morgan\s?Stanley|Citigroup|Citi(?:\s|$)|UBS|Deutsche\s?Bank|Raymond\s?James|Needham|Stifel|Wells\s?Fargo|RBC|Royal\s+Bank\s+of\s+Canada|BMO|Bank\s+of\s+Montreal|Truist|Mizuho|TD\s?Cowen|Cowen|Evercore|Evercore\s+ISI|Bernstein|AllianceBernstein|Baird|Benchmark|BTIG|Craig-Hallum|H\.?C\.?\s?Wainwright|JMP|Lake\s?Street|Maxim\s?Group|Northland|Roth\s?Capital|William\s?Blair|Wolfe|Wolfe\s+Research|Canaccord|Canaccord\s+Genuity|Redburn|Redburn\s+Atlantic|Melius|Melius\s+Research|DA\s+Davidson|D\.A\.\s+Davidson|Guggenheim|Macquarie|Susquehanna|Berenberg|Soci[eé]t[eé]\s+G[eé]n[eé]rale|SocGen|BNP\s+Paribas|Exane|Scotiabank|Scotia\s+Capital|KBW|Keefe,\s*Bruyette\s*&\s*Woods|Seaport\s+Research\s+Partners|Argus|New\s+Street\s+Research|Telsey|MoffettNathanson)\b\s+(?:Securities?|Markets?|Research|Capital|Capital\s+Markets)?\s*downgrade[sd]?",
    r"\b(?:BofA|Bank\s+of\s+America|Goldman|Goldman\s+Sachs|Jefferies|HSBC|KeyBanc|KeyBanc\s+Capital\s+Markets|Piper|Piper\s+Sandler|JPMorgan|JP\s?Morgan|Wedbush|Oppenheimer|Cantor|Cantor\s+Fitzgerald|Barclays|Morgan\s?Stanley|Citigroup|Citi(?:\s|$)|UBS|Deutsche\s?Bank|Raymond\s?James|Needham|Stifel|Wells\s?Fargo|RBC|Royal\s+Bank\s+of\s+Canada|BMO|Bank\s+of\s+Montreal|Truist|Mizuho|TD\s?Cowen|Cowen|Evercore|Evercore\s+ISI|Bernstein|AllianceBernstein|Baird|Benchmark|BTIG|Craig-Hallum|H\.?C\.?\s?Wainwright|JMP|Lake\s?Street|Maxim\s?Group|Northland|Roth\s?Capital|William\s?Blair|Wolfe|Wolfe\s+Research|Canaccord|Canaccord\s+Genuity|Redburn|Redburn\s+Atlantic|Melius|Melius\s+Research|DA\s+Davidson|D\.A\.\s+Davidson|Guggenheim|Macquarie|Susquehanna|Berenberg|Soci[eé]t[eé]\s+G[eé]n[eé]rale|SocGen|BNP\s+Paribas|Exane|Scotiabank|Scotia\s+Capital|KBW|Keefe,\s*Bruyette\s*&\s*Woods|Seaport\s+Research\s+Partners|Argus|New\s+Street\s+Research|Telsey|MoffettNathanson)\b\s+(?:Securities?|Markets?|Research|Capital|Capital\s+Markets)?\s*downgrade[sd]?\s+to\s+\b(?:sell|underperform|underweight)\b",
    r"\b(?:BofA|Bank\s+of\s+America|Goldman|Goldman\s+Sachs|Jefferies|HSBC|KeyBanc|KeyBanc\s+Capital\s+Markets|Piper|Piper\s+Sandler|JPMorgan|JP\s?Morgan|Wedbush|Oppenheimer|Cantor|Cantor\s+Fitzgerald|Barclays|Morgan\s?Stanley|Citigroup|Citi(?:\s|$)|UBS|Deutsche\s?Bank|Raymond\s?James|Needham|Stifel|Wells\s?Fargo|RBC|Royal\s+Bank\s+of\s+Canada|BMO|Bank\s+of\s+Montreal|Truist|Mizuho|TD\s?Cowen|Cowen|Evercore|Evercore\s+ISI|Bernstein|AllianceBernstein|Baird|Benchmark|BTIG|Craig-Hallum|H\.?C\.?\s?Wainwright|JMP|Lake\s?Street|Maxim\s?Group|Northland|Roth\s?Capital|William\s?Blair|Wolfe|Wolfe\s+Research|Canaccord|Canaccord\s+Genuity|Redburn|Redburn\s+Atlantic|Melius|Melius\s+Research|DA\s+Davidson|D\.A\.\s+Davidson|Guggenheim|Macquarie|Susquehanna|Berenberg|Soci[eé]t[eé]\s+G[eé]n[eé]rale|SocGen|BNP\s+Paribas|Exane|Scotiabank|Scotia\s+Capital|KBW|Keefe,\s*Bruyette\s*&\s*Woods|Seaport\s+Research\s+Partners|Argus|New\s+Street\s+Research|Telsey|MoffettNathanson)\b\s+(?:Securities?|Markets?|Research|Capital|Capital\s+Markets)?\s*downgrade[sd]?\s+to\s+\b(?:sell|underperform|underweight)\b",
    r"\b(?:BofA|Bank\s+of\s+America|Goldman|Goldman\s+Sachs|Jefferies|HSBC|KeyBanc|KeyBanc\s+Capital\s+Markets|Piper|Piper\s+Sandler|JPMorgan|JP\s?Morgan|Wedbush|Oppenheimer|Cantor|Cantor\s+Fitzgerald|Barclays|Morgan\s?Stanley|Citigroup|Citi(?:\s|$)|UBS|Deutsche\s?Bank|Raymond\s?James|Needham|Stifel|Wells\s?Fargo|RBC|Royal\s+Bank\s+of\s+Canada|BMO|Bank\s+of\s+Montreal|Truist|Mizuho|TD\s?Cowen|Cowen|Evercore|Evercore\s+ISI|Bernstein|AllianceBernstein|Baird|Benchmark|BTIG|Craig-Hallum|H\.?C\.?\s?Wainwright|JMP|Lake\s?Street|Maxim\s?Group|Northland|Roth\s?Capital|William\s?Blair|Wolfe|Wolfe\s+Research|Canaccord|Canaccord\s+Genuity|Redburn|Redburn\s+Atlantic|Melius|Melius\s+Research|DA\s+Davidson|D\.A\.\s+Davidson|Guggenheim|Macquarie|Susquehanna|Berenberg|Soci[eé]t[eé]\s+G[eé]n[eé]rale|SocGen|BNP\s+Paribas|Exane|Scotiabank|Scotia\s+Capital|KBW|Keefe,\s*Bruyette\s*&\s*Woods|Seaport\s+Research\s+Partners|Argus|New\s+Street\s+Research|Telsey|MoffettNathanson)\b\s+(?:Securities?|Markets?|Research|Capital|Capital\s+Markets)?\s*(?:cut[s]?|lower[s]?|slash(?:es|ed)?|reduce[sd]?|trim[s]?)\s+(?:price\s)?target\s+to\s+\$\d+",
    r"\b(?:BofA|Bank\s+of\s+America|Goldman|Goldman\s+Sachs|Jefferies|HSBC|KeyBanc|KeyBanc\s+Capital\s+Markets|Piper|Piper\s+Sandler|JPMorgan|JP\s?Morgan|Wedbush|Oppenheimer|Cantor|Cantor\s+Fitzgerald|Barclays|Morgan\s?Stanley|Citigroup|Citi(?:\s|$)|UBS|Deutsche\s?Bank|Raymond\s?James|Needham|Stifel|Wells\s?Fargo|RBC|Royal\s+Bank\s+of\s+Canada|BMO|Bank\s+of\s+Montreal|Truist|Mizuho|TD\s?Cowen|Cowen|Evercore|Evercore\s+ISI|Bernstein|AllianceBernstein|Baird|Benchmark|BTIG|Craig-Hallum|H\.?C\.?\s?Wainwright|JMP|Lake\s?Street|Maxim\s?Group|Northland|Roth\s?Capital|William\s?Blair|Wolfe|Wolfe\s+Research|Canaccord|Canaccord\s+Genuity|Redburn|Redburn\s+Atlantic|Melius|Melius\s+Research|DA\s+Davidson|D\.A\.\s+Davidson|Guggenheim|Macquarie|Susquehanna|Berenberg|Soci[eé]t[eé]\s+G[eé]n[eé]rale|SocGen|BNP\s+Paribas|Exane|Scotiabank|Scotia\s+Capital|KBW|Keefe,\s*Bruyette\s*&\s*Woods|Seaport\s+Research\s+Partners|Argus|New\s+Street\s+Research|Telsey|MoffettNathanson)\b\s+(?:Securities?|Markets?|Research|Capital|Capital\s+Markets)?\s*(?:target\s?price|PT|price\starget)\s+(?:cut|lowered|reduced|slashed|trimmed)\b",
    # New coverage initiation with sell/underperform/underweight — firm mention REQUIRED
    r"\b(?:BofA|Bank\s+of\s+America|Goldman|Goldman\s+Sachs|Jefferies|HSBC|KeyBanc|KeyBanc\s+Capital\s+Markets|Piper|Piper\s+Sandler|JPMorgan|JP\s?Morgan|Wedbush|Oppenheimer|Cantor|Cantor\s+Fitzgerald|Barclays|Morgan\s?Stanley|Citigroup|Citi(?:\s|$)|UBS|Deutsche\s?Bank|Raymond\s?James|Needham|Stifel|Wells\s?Fargo|RBC|Royal\s+Bank\s+of\s+Canada|BMO|Bank\s+of\s+Montreal|Truist|Mizuho|TD\s?Cowen|Cowen|Evercore|Evercore\s+ISI|Bernstein|AllianceBernstein|Baird|Benchmark|BTIG|Craig-Hallum|H\.?C\.?\s?Wainwright|JMP|Lake\s?Street|Maxim\s?Group|Northland|Roth\s?Capital|William\s?Blair|Wolfe|Wolfe\s+Research|Canaccord|Canaccord\s+Genuity|Redburn|Redburn\s+Atlantic|Melius|Melius\s+Research|DA\s+Davidson|D\.A\.\s+Davidson|Guggenheim|Macquarie|Susquehanna|Berenberg|Soci[eé]t[eé]\s+G[eé]n[eé]rale|SocGen|BNP\s+Paribas|Exane|Scotiabank|Scotia\s+Capital|KBW|Keefe,\s*Bruyette\s*&\s*Woods|Seaport\s+Research\s+Partners|Argus|New\s+Street\s+Research|Telsey|MoffettNathanson)\b\s+(?:Securities?|Markets?|Research|Capital|Capital\s+Markets)?\s*initiate[sd]?\s+coverage\s+(?:with|at)\s+\b(?:sell|underperform|underweight)\b",
]

# 4. DOJ INVESTIGATIONS
# 4. DOJ INVESTIGATIONS (More nuanced - distinguish expanding vs settling)
DOJ_INVESTIGATION_PATTERNS = [
    # NEW investigation or probe (bearish)
    r"\b(?:DOJ|Department\s+of\s+Justice)\s+(?:opens?|launches?|begins?|initiates?)\s+(?:investigation|probe|inquiry)\b",
    r"\b(?:DOJ|Department\s+of\s+Justice)\s+(?:investigat(?:ing|ion)|probes?)\s+.*?(?:Medicare|billing|fraud|pricing|collusion|antitrust)\b",
    
    # Expanding investigation (bearish)
    r"\b(?:DOJ|Department\s+of\s+Justice)\s+(?:broadens?|expands?|widens?)\s+(?:criminal\s+)?(?:investigation|probe)\b",
    r"\b(?:federal|criminal)\s+(?:investigation|probe)\s+(?:into|of)\s+(?:Medicare|billing|fraud|pricing)\b",
    
    # Antitrust/price-fixing (bearish)
    r"\b(?:antitrust|price-fixing|collusion)\s+(?:investigation|probe|charges?)\b",
    
    # Subpoenas/charges filed (bearish)
    r"\bDOJ\s+(?:subpoena[s]?|charge[sd]?|files?\s+charges?)\b",
    
    # Guilty pleas / admissions (bearish - different from deferred prosecution)
    r"\b(?:agrees?|agreed|agreeing)\s+to\s+plead\s+guilty\b",
    r"\bpleads?\s+guilty\s+(?:to|in)\b.*?\b(?:fraud|charges?|criminal)\b",
    r"\badmits?\s+(?:guilt|wrongdoing|fraud|criminal\s+conduct)\b",
    r"\bcriminal\s+plea\s+(?:agreement|deal)\b",  # Actual guilty plea (not deferred)
]

# DOJ INVESTIGATION NEGATIVE PATTERNS (these are BULLISH - settlement/dismissal without guilt)
DOJ_INVESTIGATION_NEGATIVE_PATTERNS = [
    r"\b(?:avoids?|avoid(?:ed|ing))\s+(?:prosecution|criminal\s+charges?)\b",  # Boeing avoids prosecution
]

# 5. FTC BLOCKING
FTC_BLOCK_PATTERNS = [
    r"\b(?:FTC|Federal\s+Trade\s+Commission)\s+(?:block[s]?|blocked|blocking)\b",
    r"\bFTC\s+(?:sue[sd]?|lawsuit)\s+to\s+(?:block|stop|prevent)\b",
    r"\bFTC\s+(?:challenge[sd]?|challenging)\b",
    r"\b(?:regulator[s]?|FTC)\s+(?:reject[s]?|rejected)\s+(?:merger|acquisition|deal)\b",
]

# 6. FAILED MERGERS/ACQUISITIONS
FAILED_MERGER_PATTERNS = [
    r"\b(?:merger|acquisition|deal)\s+(?:fail[s]?|failed|collapse[sd]?|terminated|cancelled|abandoned)\b",
    r"\b(?:call[s]?|called)\s+off\s+(?:merger|acquisition|deal)\b",
    r"\b(?:walk[s]?|walked)\s+away\s+from\s+(?:merger|acquisition|deal)\b",
    r"\b(?:terminate[sd]?|terminating|termination)\s+of\s+(?:merger|acquisition|deal)\s+agreement\b",
    r"\b(?:breakup|break-up)\s+fee\b",
]

# 7. ACCOUNTING FRAUD / RESTATEMENTS
ACCOUNTING_FRAUD_PATTERNS = [
    r"\b(?:accounting|financial)\s+(?:fraud|irregularit(?:y|ies)|misconduct)\b",
    r"\brestate[sd]?\s+(?:financial[s]?|earnings?|results?)\b",
    r"\b(?:SEC|Securities\s+and\s+Exchange\s+Commission)\s+(?:investigation|probe)\b",
    r"\b(?:misstat(?:e|ed|ement[s]?)|misrepresent(?:ed|ation[s]?))\s+(?:earnings?|revenue[s]?|financial[s]?)\b",
    r"\b(?:internal\s+controls?|accounting\s+errors?)\s+(?:weakness|failure|deficienc(?:y|ies))\b",
]

# 8. LOST CONTRACTS
CONTRACT_LOSS_PATTERNS = [
    r"\blost\s+(?:a\s+)?(?:\$\d+(?:\.\d+)?(?:M|B|million|billion)\s+)?contract\b",
    r"\bcontract\s+(?:terminated|cancelled|suspended)\b",
    r"\bfail(?:ed|s)?\s+to\s+(?:win|secure|retain)\s+(?:a\s+)?contract\b",
    r"\bpassed\s+over\s+for\s+contract\b",
    r"\bcontract\s+(?:award[s]?|awarded)\s+to\s+(?:competitor|rival)\b",
]

# 9. EXPORT BANS
EXPORT_BAN_PATTERNS = [
    r"\bexport\s+ban\b",
    r"\bban(?:ned|s)?\s+from\s+export(?:ing)?\b",
    r"\b(?:restrict(?:ed|s|ion[s]?)|prohibit(?:ed|s|ion[s]?))\s+(?:from\s+)?export(?:ing)?\b",
    r"\b(?:US|U\.S\.|United\s+States)\s+(?:ban[s]?|restrict[s]?)\s+(?:export[s]?|sales?)\b",
]

# 10. SECTOR TARIFFS
SECTOR_TARIFF_PATTERNS = [
    r"\btariff[s]?\s+(?:on|imposed\s+on|announced\s+on)\b",
    r"\b(?:import|trade)\s+(?:tariff[s]?|duties|restrictions?)\b",
    r"\b(?:\d+%)\s+tariff[s]?\b",
    r"\b(?:new|additional)\s+tariff[s]?\s+on\b",
    r"\btrade\s+war\b",
]

# 11. SHARE DILUTION
SHARE_DILUTION_PATTERNS = [
    r"\b(?:announce[sd]?|file[sd]?)\s+(?:stock|share|equity)\s+offering\b",
    r"\b(?:dilutive|dilution)\s+(?:stock|share|equity)\s+offering\b",
    r"\b(?:secondary|public)\s+offering\s+of\s+(?:\d+(?:\.\d+)?(?:M|million))?\s+shares?\b",
    r"\b(?:ATM|at-the-market)\s+(?:offering|program)\b",
    r"\bwarrant\s+(?:exercise|conversion)\b.*?\bdilut(?:ion|ive)\b",
]

# 12. DELISTING / COMPLIANCE FAILURES
DELISTING_PATTERNS = [
    r"\bdelist(?:ed|ing|s)?\b",
    r"\bdelisting\s+notice\b",
    r"\b(?:fail[s]?|failed)\s+to\s+(?:meet|maintain|comply)\b.*?\b(?:nasdaq|nyse|exchange)\s+(?:requirement[s]?|standard[s]?|rule[s]?)\b",
    r"\bnon-compliance\b.*?\b(?:nasdaq|nyse|exchange)\b",
    r"\bdownlist(?:ed|ing|s)?\b",
    r"\b(?:nasdaq|nyse|exchange)\s+(?:warning|notice|notification)\b",
    r"\b(?:risk|threat)\s+of\s+delisting\b",
    r"\bdeficiency\s+(?:letter|notice)\b.*?\b(?:nasdaq|nyse|exchange)\b",
]

# 13. PLANE CRASHES / SAFETY INCIDENTS (NEW - affects airlines and manufacturers)
PLANE_CRASH_PATTERNS = [
    r"\b(?:plane|aircraft|jet|boeing|airbus)\s+crash(?:es|ed)?\b",
    r"\b(?:737|747|787|A320|A380)\s+(?:MAX\s+)?crash(?:es|ed)?\b",
    r"\b(?:fatal|deadly)\s+(?:plane|aircraft)\s+(?:crash|accident|incident)\b",
    r"\b(?:aircraft|plane)\s+(?:accident|disaster)\s+(?:kills?|deaths?)\b",
    r"\b(?:emergency\s+landing|engine\s+failure)\s+.*?\b(?:injuries?|casualties?)\b",
]

# 14. BAILOUT SEEKING / FINANCIAL DISTRESS (seeking government rescue)
BAILOUT_SEEKING_PATTERNS = [
    r"\b(?:seeks?|seeking|requests?|requesting)\s+(?:a\s+)?(?:\$\d+(?:\.\d+)?\s+(?:billion|million)\s+)?(?:bailout|government\s+aid|federal\s+assistance|rescue\s+package)\b",
    r"\b(?:seeks?|seeking)\s+(?:\$\d+(?:\.\d+)?\s+(?:billion|million))\s+(?:in\s+)?(?:government\s+)?(?:loans?|aid|assistance|funding)\b",
    r"\b(?:needs?|requiring)\s+(?:government|federal|emergency)\s+(?:rescue|bailout|intervention)\b",
]

# 15. BUYBACK SUSPENSION (negative signal - capital conservation)
BUYBACK_SUSPENSION_PATTERNS = [
    r"\b(?:suspend[s]?|suspending|suspended|halts?|halted|halting)\s+(?:stock|share)\s+(?:buyback[s]?|repurchase[s]?)\b",
    r"\b(?:cancel[s]?|cancelling|cancelled|terminates?|terminated)\s+(?:stock|share)\s+(?:buyback[s]?|repurchase)\s+(?:program|plan)\b",
    r"\b(?:pause[s]?|pausing|paused)\s+(?:stock|share)\s+(?:buyback[s]?|repurchase)\b",
]

# 16. BAILOUT APPROVED/RECEIVED (government rescue approved - survival assured)
BAILOUT_APPROVED_PATTERNS = [
    r"\b(?:receives?|received|gets?|got|granted|awarded)\s+(?:\$\d+(?:\.\d+)?\s+(?:billion|million)\s+)?(?:bailout|government\s+aid|TARP|rescue\s+package|federal\s+assistance)\b",
    r"\b(?:bailout|TARP|rescue\s+package|government\s+aid)\s+(?:approved|granted|received|awarded)\b",
    r"\b(?:Treasury|government|federal)\s+(?:injects?|injected|provides?|provided)\s+\$\d+(?:\.\d+)?\s+(?:billion|million)\b",
    r"\b(?:secures?|secured)\s+(?:\$\d+(?:\.\d+)?\s+(?:billion|million)\s+)?(?:government|federal)\s+(?:credit\s+line|loan|financing|backing)\b",
    # Regulatory relief (Fed/OCC asset caps, consent orders lifted)
    r"\b(?:Fed|Federal\s+Reserve|OCC|FDIC|regulator[s]?)\s+(?:lifts?|lifted|removes?|removed|ends?|ended|terminates?)\s+(?:asset\s+cap|growth\s+restriction|consent\s+order|enforcement\s+action)\b",
    r"\b(?:escapes?|escaped|is\s+no\s+longer\s+subject\s+to)\s+(?:Fed\'?s\s+)?(?:asset\s+cap|growth\s+restriction|enforcement\s+action)\b",
    r"\basset\s+cap\s+(?:is\s+)?(?:lifted|removed|ended)\b",
]

# 17. BAILOUT REPAID (company exits bailout - financial strength signal)
BAILOUT_REPAID_PATTERNS = [
    r"\b(?:repays?|repaid|repaying)\s+(?:TARP|bailout|government\s+loan|federal\s+loan)\b",
    r"\b(?:exits?|exited|exiting)\s+(?:TARP|bailout\s+program|government\s+program)\b",
    r"\b(?:returns?|returned|returning)\s+(?:TARP|bailout)\s+(?:funds?|money)\b",
    r"\b(?:pays?\s+back|paid\s+back)\s+(?:TARP|bailout|government)\b",
]

# ============================================================================
# SECTOR-LEVEL NEWS PATTERNS (BULLISH & BEARISH)
# ============================================================================
# These patterns trigger trades on PEERS (detected via Finnhub peers API)
# when sector-wide catalysts occur (government mandates, trade wars, etc.)

# 12. TRADE WAR / EXPORT SANCTIONS (SHORT - affects sector negatively)
# Enhanced with broader escalation patterns while maintaining specificity
TRADE_WAR_PATTERNS = [
    # Specific export/import bans with sector context
    r"(?:us|china|eu)\s+(?:bans?|bars?|prohibits?)\s+(?:export[s]?|import[s]?)\s+(?:of|from)\s+.*?(?:semiconductor|chip|steel|aluminum|rare\s+earth|technology|auto|agriculture|textile)\b",
    r"(?:export|import)\s+(?:ban|embargo|prohibition)\s+(?:imposed|placed|announced)\s+on\s+(?:semiconductor|chip|steel|aluminum|technology|auto|agriculture|medical|defense)\s+(?:sector|industry|goods|products)\b",
    
    # Tariffs with specific percentages and sectors
    r"(?:us|china|eu)\s+(?:imposes?|announces?|levies?)\s+(?:\d+%)\s+tariff(?:s)?\s+on\s+(?:semiconductor|steel|aluminum|auto|agriculture|solar|ev|chip|technology)\s+(?:imports?|exports?|goods|products)\b",
    r"(?:\d+%)\s+tariff(?:s)?\s+(?:imposed|placed|announced)\s+on\s+(?:all\s+)?(?:semiconductor|steel|aluminum|auto|agriculture|chinese|us|european)\s+(?:imports?|exports?|goods|products)\b",
    
    # Broader tariff escalations (Section 301, reciprocal, blanket)
    r"(?:trump|biden|us\s+administration)\s+(?:announces?|threatens?|imposes?)\s+(?:new|additional|blanket|reciprocal)\s+tariff[s]?\b",
    r"\bsection\s+301\s+tariff[s]?\b",
    r"(?:\d+%)\s+tariff[s]?\s+on\s+(?:all\s+)?(?:chinese|mexican|canadian|european|imported)\s+(?:imports?|goods|products)\b",
    r"\btrade\s+war\s+(?:escalates?|intensifies?|heats?\s+up)\b",
    
    # Sanctions with clear sector/industry impact
    r"(?:us|china|eu)\s+(?:sanctions?|penalties?|restrictions?)\s+(?:target|hit|affect)\s+(?:semiconductor|chip|steel|aluminum|technology|auto|agriculture|defense)\s+(?:sector|industry|companies?)\b",
    r"(?:trade\s+restrictions?|export\s+controls?)\s+(?:on|targeting|affecting)\s+(?:semiconductor|chip|technology|defense|medical)\s+(?:sector|industry|exports?)\b",
    
    # Retaliation with sector context
    r"(?:china|us|eu)\s+(?:retaliates?|responds?)\s+with\s+(?:tariffs?|duties?|restrictions?)\s+on\s+(?:semiconductor|steel|aluminum|auto|agriculture|technology)\b",
]

# 13. SUPPLY CHAIN / PRODUCTION HALT (SHORT - disrupts operations)
SUPPLY_HALT_PATTERNS = [
    r"supply\s+chain\s+(?:disruption|failure|bottleneck|crisis)\b",
    r"(?:producer|manufacturer|plant|mine)\s+halts?\s+(?:production|operations|output)\b",
    r"(?:major|key)\s+(?:facility|plant)\s+(?:closure|shuts?down|strike|labor\s+dispute)\b",
    r"mining\s+mine\s+(?:closure|shuts\s?down|ceases\s+operation)\b",
    r"supply\s+chain\s+gridlock\b",
]

# 14. WAR / GEOPOLITICAL RISK (SHORT - creates uncertainty, except defense sector)
WAR_RISK_PATTERNS = [
    # Require 'war' as a standalone word (avoid matching 'warning')
    r"\bwar(?:fare)?\b|military\s+(?:conflict|action)|hostilities\s+erupt(?:ed)?|invasion\b",
    r"geopolitical\s+tension\s+(?:rises|escalates|hits\s+sector)\b",
    r"martial\s+law|state\s+of\s+emergency\b",
]

# Exclude metaphorical/business 'wars' (chip wars, price wars, AI wars)
WAR_RISK_NEGATIVE_PATTERNS = [
    r"\b(?:ai|chip|price|console|patent)\s+wars?\b",  # business/metaphor wars
    r"\btrade\s+wars?\b",  # treat explicitly as non-geopolitical per user
    r"(?:\bwar(?:fare)?\b.{0,24}\b(?:sector|industry)\b|\b(?:sector|industry)\b.{0,24}\bwar(?:fare)?\b)",  # 'sector + war' proximity ignored
]

# 15. GUARANTEED VOLUME / SUBSIDY (LONG - legislative support)
MANDATE_SUBSIDY_PATTERNS = [
    r"(?:legislative|government)\s+(?:mandates?|guarantees?)\s+(?:volume|demand)\s+for\s+(?:sector|industry)\b",
    r"(?:subsidy|grant|tax\s+credit|incentive)\s+(?:approved|boost|extended)\s+for\s+(?:sector|industry)\b",
    r"(?:aid\s+package|bailout)\s+(?:approved|granted|receives?|passes?)\s+for\s+(?:sector|industry)\b",
    r"(?:infrastructure|clean\s+energy|defense)\s+(?:spending|bill)\s+(?:passes|approved)\b",
]

# 16. ACA SUPPORT (LONG - healthcare sector specific)
ACA_SUPPORT_PATTERNS = [
    r"supreme\s+court\s+(?:upholds?|backs?|supports?|protects?)\s+(?:ACA|Affordable\s+Care\s+Act)\b",
    r"(?:aca|affordable\s+care\s+act)\s+(?:funding|provisions?)\s+(?:secured|maintained|restored)\b",
]

# 17. SUPPLY CHAIN RECOVERY (LONG - normalization after disruption)
SC_RECOVERY_PATTERNS = [
    r"supply\s+chain\s+(?:recovers?|eases?|restored|normalized|improves?)\b",
    r"(?:logistics|shipping)\s+(?:bottleneck|constraints?)\s+eases?\b",
    r"production\s+back\s+online|plant\s+resumes\s+operations\b",
]

# 18. WAR / DEFENSE SPENDING SPIKE (LONG - benefits defense contractors)
DEFENSE_SPIKE_PATTERNS = [
    r"(?:war|conflict|geopolitical\s+tension|hostilities)\s+(?:erupts?|escalates?|boosts?|sends?)\s+(?:defense\s+stock|arms\s+maker|security\s+spending)\b",
    r"pentagon\s+(?:orders?|increases?|hikes?)\s+(?:spending|budget|funding)\s+for\s+(?:weapons?|missiles?|aircraft|security)\b",
    r"(?:nato|european\s+union|g7)\s+ramps?\s+up\s+defense\s+procurement\b",
    r"emergency\s+(?:defense|military)\s+funding\s+(?:approved|passes|signed)\b",
]

# 19. REGIONAL BANKING FEARS (SHORT - contagion risk in regional banking sector)
REGIONAL_BANK_FEAR_PATTERNS = [
    r"(?:regional|community|local)\s+bank(?:s|ing)\s+(?:fear|crisis|distress|turmoil|instability)\b",
    r"(?:bank\s+stock|regional\s+lender)\s+(?:plunge|tumble|sell-off|slumps?)\s+on\s+fear\b",
    r"(?:fdic|regulators?)\s+steps?\s+in\s+at\s+(?:regional|community)\s+bank\b",
    r"depositor\s+(?:run|panic|withdrawal)|uninsured\s+deposits?\b",
    r"(?:commercial\s+real\s+estate|CRE)\s+(?:losses|defaults?)\b",
    r"liquidity\s+(?:crunch|concerns?)\b.*?\b(?:regional|community)\s+bank",
]

# 20. LARGE BANK ECONOMIC WARNINGS (SHORT - macro slowdown signals)
LARGE_BANK_WARNING_PATTERNS = [
    r"(?:jpmorgan|jp\s?morgan|goldman\s+sachs|goldman|bofa|bank\s+of\s+america|citigroup|citi|morgan\s+stanley|wells\s+fargo|wells)\s+(?:warns?|sees?|caution|red\s?flag)\s+on\s+economy\b",
    r"(?:ceo|cfo|analyst)\s+(?:predicts?|sees?)\s+(?:recession|slowdown|economic\s+downturn)\b",
    r"credit\s+(?:tightens?|standards?|loan\s+demand)\s+(?:fall|decrease|slump)\b",
    r"risk\s+of\s+default|loan\s+loss\s+provisions?\s+(?:rise|increase)\b",
]

# 21. TARIFF RIFT BEARISH (SHORT - trade war impacts exporters/importers)
TARIFF_RIFT_BEARISH_PATTERNS = [
    r"trump\s+(?:tariff|trade\s+war|china\s+rift)\s+(?:escalates?|reignites?|intensifies?|new\s+round)\b",
    r"(?:china|us)\s+retaliates?|retaliatory\s+duties?\s+imposed\b",
    r"(?:sector|industry\s+group|us\s+company)\s+(?:hit|slammed|affected|suffers?)\s+by\s+tariff\b",
    r"imposes?\s+duties?\s+(?:on|against)\s+(?:chinese|us)\s+(?:imports?|goods)\b",
]

# 22. TARIFF PROTECTION BULLISH (LONG - domestic producers benefit from tariffs)
TARIFF_PROTECTION_BULLISH_PATTERNS = [
    r"(?:us|domestic)\s+(?:manufacturer|producer|competitor)\s+(?:benefits?|gains?|boosted)\s+from\s+tariff\b",
    r"(?:us\s+solar\s+stock|us\s+ev\s+maker|us\s+chip\s+firm)\s+sees?\s+advantage\s+from\s+duties?\b",
    r"tariff\s+(?:shields?|protects?|creates?)\s+opportunity\s+for\s+us\s+company\b",
]

# Compile all patterns with positive/negative structure and direction
CLUSTER_PATTERNS = {
    # LONG CLUSTERS (9 total)
    'analyst_action_upgrade': {
        'direction': 'LONG',
        'positive': [re.compile(p, re.IGNORECASE) for p in ANALYST_UPGRADE_PATTERNS],
        'negative': [re.compile(p, re.IGNORECASE) for p in ANALYST_UPGRADE_NEGATIVE_PATTERNS]
    },
    'major_contract': {
        'direction': 'LONG',
        'positive': [re.compile(p, re.IGNORECASE) for p in CONTRACT_PATTERNS],
        'negative': [re.compile(p, re.IGNORECASE) for p in CONTRACT_NEGATIVE_PATTERNS]
    },
    'phase3_achievement': {
        'direction': 'LONG',
        'positive': [re.compile(p, re.IGNORECASE) for p in PHASE3_PATTERNS],
        'negative': [re.compile(p, re.IGNORECASE) for p in PHASE3_NEGATIVE_PATTERNS]
    },
    'share_buyback': {
        'direction': 'LONG',
        'positive': [re.compile(p, re.IGNORECASE) for p in BUYBACK_PATTERNS],
        'negative': []
    },
    'uplisting': {
        'direction': 'LONG',
        'positive': [re.compile(p, re.IGNORECASE) for p in UPLISTING_PATTERNS],
        'negative': [re.compile(p, re.IGNORECASE) for p in UPLISTING_NEGATIVE_PATTERNS]
    },
    'ai_enterprise_partnership': {
        'direction': 'LONG',
        'positive': [re.compile(p, re.IGNORECASE) for p in AI_PARTNERSHIP_PATTERNS],
        'negative': []
    },
    'merger_acq': {
        'direction': 'LONG',
        'positive': [re.compile(p, re.IGNORECASE) for p in MERGER_ACQ_PATTERNS],
        'negative': []
    },
    'fda_approval': {
        'direction': 'LONG',
        'positive': [re.compile(p, re.IGNORECASE) for p in FDA_APPROVAL_PATTERNS],
        'negative': []
    },
    'contract_award_keyword': {
        'direction': 'LONG',
        'positive': [re.compile(p, re.IGNORECASE) for p in CONTRACT_AWARD_KEYWORD_PATTERNS],
        'negative': []
    },
    'bailout_approved': {
        'direction': 'LONG',
        'positive': [re.compile(p, re.IGNORECASE) for p in BAILOUT_APPROVED_PATTERNS],
        'negative': []
    },
    'bailout_repaid': {
        'direction': 'LONG',
        'positive': [re.compile(p, re.IGNORECASE) for p in BAILOUT_REPAID_PATTERNS],
        'negative': []
    },
    
    # SHORT CLUSTERS (16 total)
    'phase3_failure': {
        'direction': 'SHORT',
        'positive': [re.compile(p, re.IGNORECASE) for p in PHASE3_FAILURE_PATTERNS],
        'negative': []
    },
    'fda_rejection': {
        'direction': 'SHORT',
        'positive': [re.compile(p, re.IGNORECASE) for p in FDA_REJECTION_PATTERNS],
        'negative': []
    },
    'analyst_downgrade_sell': {
        'direction': 'SHORT',
        'positive': [re.compile(p, re.IGNORECASE) for p in ANALYST_DOWNGRADE_SELL_PATTERNS],
        'negative': []
    },
    'doj_investigation': {
        'direction': 'SHORT',
        'positive': [re.compile(p, re.IGNORECASE) for p in DOJ_INVESTIGATION_PATTERNS],
        'negative': [re.compile(p, re.IGNORECASE) for p in DOJ_INVESTIGATION_NEGATIVE_PATTERNS]  # Settlements/dismissals
    },
    'ftc_block': {
        'direction': 'SHORT',
        'positive': [re.compile(p, re.IGNORECASE) for p in FTC_BLOCK_PATTERNS],
        'negative': []
    },
    'failed_merger': {
        'direction': 'SHORT',
        'positive': [re.compile(p, re.IGNORECASE) for p in FAILED_MERGER_PATTERNS],
        'negative': []
    },
    'accounting_fraud': {
        'direction': 'SHORT',
        'positive': [re.compile(p, re.IGNORECASE) for p in ACCOUNTING_FRAUD_PATTERNS],
        'negative': []
    },
    'contract_loss': {
        'direction': 'SHORT',
        'positive': [re.compile(p, re.IGNORECASE) for p in CONTRACT_LOSS_PATTERNS],
        'negative': []
    },
    'export_ban': {
        'direction': 'SHORT',
        'positive': [re.compile(p, re.IGNORECASE) for p in EXPORT_BAN_PATTERNS],
        'negative': []
    },
    'sector_tariff': {
        'direction': 'SHORT',
        'positive': [re.compile(p, re.IGNORECASE) for p in SECTOR_TARIFF_PATTERNS],
        'negative': []
    },
    'share_dilution': {
        'direction': 'SHORT',
        'positive': [re.compile(p, re.IGNORECASE) for p in SHARE_DILUTION_PATTERNS],
        'negative': []
    },
    'delisting': {  # NEW: Delisting/compliance failures
        'direction': 'SHORT',
        'positive': [re.compile(p, re.IGNORECASE) for p in DELISTING_PATTERNS],
        'negative': []
    },
    'plane_crash': {  # NEW: Aviation accidents/safety incidents
        'direction': 'SHORT',
        'positive': [re.compile(p, re.IGNORECASE) for p in PLANE_CRASH_PATTERNS],
        'negative': []
    },
    'bailout_seeking': {  # NEW: Seeking government rescue
        'direction': 'SHORT',
        'positive': [re.compile(p, re.IGNORECASE) for p in BAILOUT_SEEKING_PATTERNS],
        'negative': []
    },
    'buyback_suspension': {  # NEW: Suspending buyback programs
        'direction': 'SHORT',
        'positive': [re.compile(p, re.IGNORECASE) for p in BUYBACK_SUSPENSION_PATTERNS],
        'negative': []
    },
    
    # SECTOR-LEVEL NEWS CLUSTERS (7 total - require peer detection)
    'trade_war_sector': {
        'direction': 'SHORT',
        'positive': [re.compile(p, re.IGNORECASE) for p in TRADE_WAR_PATTERNS],
        'negative': [],
        'requires_peers': True  # Trade on entire subsector
    },
    'supply_halt_sector': {
        'direction': 'SHORT',
        'positive': [re.compile(p, re.IGNORECASE) for p in SUPPLY_HALT_PATTERNS],
        'negative': [],
        'requires_peers': True
    },
    'war_risk_sector': {
        'direction': 'SHORT',
        'positive': [re.compile(p, re.IGNORECASE) for p in WAR_RISK_PATTERNS],
        'negative': [
            # Only exclude metaphorical 'wars', not defense companies (they trade both ways)
            *[re.compile(p, re.IGNORECASE) for p in WAR_RISK_NEGATIVE_PATTERNS],
        ],
        'requires_peers': True
    },
    'mandate_subsidy_sector': {
        'direction': 'LONG',
        'positive': [re.compile(p, re.IGNORECASE) for p in MANDATE_SUBSIDY_PATTERNS],
        'negative': [],
        'requires_peers': True
    },
    'aca_support_healthcare': {
        'direction': 'LONG',
        'positive': [re.compile(p, re.IGNORECASE) for p in ACA_SUPPORT_PATTERNS],
        'negative': [],
        'requires_peers': True
    },
    'supply_recovery_sector': {
        'direction': 'LONG',
        'positive': [re.compile(p, re.IGNORECASE) for p in SC_RECOVERY_PATTERNS],
        'negative': [],
        'requires_peers': True
    },
    'defense_spending_sector': {
        'direction': 'LONG',
        'positive': [re.compile(p, re.IGNORECASE) for p in DEFENSE_SPIKE_PATTERNS],
        'negative': [],
        'requires_peers': True
    },
    'regional_bank_fear': {
        'direction': 'SHORT',
        'positive': [re.compile(p, re.IGNORECASE) for p in REGIONAL_BANK_FEAR_PATTERNS],
        'negative': [],
        'requires_peers': True,
        'sector_filter': 'Regional Banks'  # Only trade regional banking subsector
    },
    'large_bank_warning': {
        'direction': 'SHORT',
        'positive': [re.compile(p, re.IGNORECASE) for p in LARGE_BANK_WARNING_PATTERNS],
        'negative': [],
        'requires_peers': False  # Trade large banks individually (systematic risk)
    },
    'tariff_rift_bearish': {
        'direction': 'SHORT',
        'positive': [re.compile(p, re.IGNORECASE) for p in TARIFF_RIFT_BEARISH_PATTERNS],
        'negative': [],
        'requires_peers': True,
        'country_filter': 'exporters'  # Affects companies with China exposure
    },
    'tariff_protection_bullish': {
        'direction': 'LONG',
        'positive': [re.compile(p, re.IGNORECASE) for p in TARIFF_PROTECTION_BULLISH_PATTERNS],
        'negative': [],
        'requires_peers': True,
        'country_filter': 'US'  # ONLY US-based domestic producers
    },
}

# Expanded analyst firms (for deduplication) - 45+ firms
ANALYST_FIRMS = {
    'BofA', 'Bank of America', 'Goldman Sachs', 'Goldman', 'Jefferies', 
    'HSBC', 'KeyBanc', 'Piper Sandler', 'Piper', 'JPMorgan', 'JP Morgan', 
    'Wedbush', 'Oppenheimer', 'Cantor Fitzgerald', 'Cantor', 'Barclays', 
    'Morgan Stanley', 'Mizuho', 'TD Cowen', 'Cowen', 'Evercore', 
    'Citigroup', 'Citi', 'UBS', 'Deutsche Bank', 'Raymond James', 
    'Needham', 'Stifel', 'Wells Fargo', 'RBC Capital Markets', 'RBC', 
    'BMO Capital Markets', 'BMO', 'Truist Securities', 'Truist',
    'Bernstein', 'AllianceBernstein', 'Baird', 'Robert W. Baird',
    'Benchmark', 'Benchmark Company', 'BTIG', 'Craig-Hallum', 
    'D.A. Davidson', 'H.C. Wainwright', 'JMP Securities', 'JMP',
    'Lake Street', 'Lake Street Capital', 'Maxim Group', 'Maxim',
    'Northland Securities', 'Northland', 'Roth Capital', 'Roth',
    'William Blair', 'Wolfe Research', 'Wolfe'
}

# Deal value extraction (from live_news_trader_ibkr.py)
DEAL_VALUE_RE = re.compile(
    r'\$(\d+(?:\.\d+)?)\s?(million|billion|M|B)',
    re.IGNORECASE
)

# Drug name patterns (alphanumeric identifiers common in pharma)
DRUG_NAME_RE = re.compile(
    r'\b([A-Z]{2,}[-]?\d{3,5}[A-Z]?)\b',  # e.g., VK2735, LLY-001, ABC123A
    re.IGNORECASE
)

# Contract partner patterns
CONTRACT_PARTNER_PATTERNS = [
    r'\b(Pentagon|DoD|Department\sof\sDefense|U\.?S\.?\sNavy|U\.?S\.?\sArmy|Air\sForce|Space\sForce|Marines|NASA)\b',
    r'\b(Amazon|Microsoft|Google|Alphabet|Apple|Meta|Facebook|Tesla|SpaceX)\b',
    r'\b(Boeing|Lockheed\sMartin|Raytheon|Northrop\sGrumman|General\sDynamics)\b',
    r'\b(Walmart|Target|Costco|Home\sDepot|CVS|Walgreens)\b',
    r'\b(federal\sgovernment|state\sgovernment|GSA|VA|DHS)\b',
]
CONTRACT_PARTNER_RE = [re.compile(p, re.IGNORECASE) for p in CONTRACT_PARTNER_PATTERNS]

# ============================================================================
# DUPLICATE DETECTION HELPERS
# ============================================================================

def extract_drug_name(title: str, ticker: str) -> Optional[str]:
    """
    Extract drug/compound name from Phase 3 title.
    Examples: VK2735, LLY-001, ABC-123
    """
    # Find all potential drug names
    matches = DRUG_NAME_RE.findall(title)
    
    for match in matches:
        # Skip if it's just the ticker
        if match.upper() == ticker.upper():
            continue
        
        # Skip common false positives
        if match.upper() in ['COVID', 'HIV', 'FDA', 'NYSE', 'NASDAQ', 'SEC', 'IPO']:
            continue
        
        # Valid drug name found
        return match.upper()
    
    return None


def extract_contract_partner(title: str) -> Optional[str]:
    """
    Extract contracting partner/entity from contract title.
    Examples: Pentagon, Amazon, US Navy
    """
    for pattern in CONTRACT_PARTNER_RE:
        match = pattern.search(title)
        if match:
            partner = match.group(1)
            # Normalize variations
            partner = partner.replace('.', '').replace('U S ', 'US').strip()
            return partner
    
    return None


def extract_deal_signature(title: str) -> Optional[str]:
    """Extract normalized deal amount from title (e.g., '$500M', '$2.5B')"""
    for match in DEAL_VALUE_RE.finditer(title):
        value_str = match.group(1)
        unit = match.group(2).upper()
        
        # Skip if it's a projection/valuation/revenue
        start, end = match.span()
        context = title[max(0, start-50):min(len(title), end+100)].lower()
        if any(kw in context for kw in ['revenue', 'projected', 'analyst says', 'expects', 'estimates', 'valuation']):
            continue
        
        # Normalize: $500 million -> "500M", $2.5 billion -> "2500M" or "2.5B"
        value = float(value_str.replace(',', ''))
        if unit in ['BILLION', 'B']:
            if value >= 1:
                return f"{value:.1f}B".replace('.0', '')
            else:
                return f"{int(value*1000)}M"
        else:  # Million
            return f"{int(value)}M"
    
    return None


def extract_analyst_signature(title: str, ticker: str) -> Tuple[Optional[str], Optional[str]]:
    """Enhanced analyst signature extraction with granular action classification"""
    title_lower = title.lower()
    
    # 1. Find analyst firm (longest match first, with word boundaries)
    found_firm = None
    for firm in sorted(ANALYST_FIRMS, key=len, reverse=True):
        # Case-insensitive search with word boundaries
        pattern = r'\b' + re.escape(firm.lower()) + r'\b'
        if re.search(pattern, title_lower):
            # Avoid matching company names or generic words
            if firm.lower() not in ['wells', 'trust', 'capital', 'securities', 'group']:
                # Avoid matching ticker company name
                if firm.upper() not in (ticker.upper(), f"{ticker.upper()} INC", f"{ticker.upper()} CORP", f"{ticker.upper()} GROUP"):
                    found_firm = firm.replace(' ', '_')
                    break
    
    # 2. Classify action type (more granular)
    action_type = None
    
    # Upgrades (distinguish to buy vs generic)
    if re.search(r'\bupgrade[sd]?\b.*?\b(?:to\s+)?(?:buy|outperform|overweight)\b', title_lower):
        action_type = "UPGRADE_TO_BUY"
    elif re.search(r'\bupgrade[sd]?\b', title_lower):
        action_type = "UPGRADE"
    
    # Downgrades (distinguish to sell vs generic)
    elif re.search(r'\bdowngrade[sd]?\b.*?\b(?:to\s+)?(?:sell|underperform|underweight)\b', title_lower):
        action_type = "DOWNGRADE_TO_SELL"
    elif re.search(r'\bdowngrade[sd]?\b', title_lower):
        action_type = "DOWNGRADE"
    
    # Price target changes
    elif re.search(r'\b(?:raise[sd]?|lift[s]?|boost[s]?|increase[sd]?|hike[sd]?)\b.*?\btarget\b', title_lower):
        action_type = "PT_RAISE"
    elif re.search(r'\b(?:cut[s]?|lower[s]?|slash(?:es|ed)?|reduce[sd]?|trim[s]?)\b.*?\btarget\b', title_lower):
        action_type = "PT_CUT"
    
    # Initiations (distinguish buy/sell/neutral)
    elif re.search(r'\binitiate[sd]?\b.*?\b(?:with\s+)?(?:buy|outperform)\b', title_lower):
        action_type = "INITIATE_BUY"
    elif re.search(r'\binitiate[sd]?\b.*?\b(?:with\s+)?(?:sell|underperform)\b', title_lower):
        action_type = "INITIATE_SELL"
    elif re.search(r'\binitiate[sd]?\b', title_lower):
        action_type = "INITIATE_NEUTRAL"
    
    # Reiterations
    elif re.search(r'\breiterate[sd]?\b.*?\bbuy\b', title_lower):
        action_type = "REITERATE_BUY"
    elif re.search(r'\breiterate[sd]?\b', title_lower):
        action_type = "REITERATE"
    
    # 3. Extract price target if present
    price_target = None
    pt_match = re.search(r'\$(\d+(?:\.\d+)?)', title)
    if pt_match and action_type:
        price_target = f"${pt_match.group(1)}"
        return found_firm, f"{action_type}_{price_target.replace('$', '')}"
    
    return found_firm, action_type


def is_duplicate_event(
    ticker: str, 
    title: str, 
    pub_time: datetime, 
    clusters: List[str],
    cooldown_state: Dict[str, str]
) -> bool:
    """
    Check if event is duplicate based on cluster-specific logic.
    Returns True if duplicate, False if BREAKING.
    """
    
    for cluster in clusters:
        candidate_keys = []
        
        # ANALYST ACTIONS: Use Firm + Action (separate UPGRADES from DOWNGRADES)
        if cluster in ['analyst_action_upgrade', 'analyst_downgrade_sell']:
            firm, action = extract_analyst_signature(title, ticker)
            if firm and action:
                primary_key = f"{ticker}__ANALYST:{firm}__{action}"
            elif action:
                primary_key = f"{ticker}__ANALYST_GENERIC__{action}"
            else:
                primary_key = f"{ticker}__{cluster}"
            candidate_keys = [primary_key]
        
        # CONTRACTS: Use partner + deal size
        elif cluster in ['major_contract', 'contract_award_keyword', 'contract_loss']:
            partner = extract_contract_partner(title)
            deal_sig = extract_deal_signature(title)
            if partner and deal_sig:
                primary_key = f"{ticker}__{cluster}__partner:{partner}__deal:{deal_sig}"
            elif partner:
                primary_key = f"{ticker}__{cluster}__partner:{partner}"
            elif deal_sig:
                primary_key = f"{ticker}__{cluster}__deal:{deal_sig}"
            else:
                primary_key = f"{ticker}__{cluster}"
            candidate_keys = [primary_key]

        # M&A / DIVESTITURES: Use partner/buyer + deal size to anchor the first article of the story
        elif cluster in ['merger_acq', 'failed_merger']:
            partner = extract_contract_partner(title)
            deal_sig = extract_deal_signature(title)
            if partner and deal_sig:
                primary_key = f"{ticker}__{cluster}__partner:{partner}__deal:{deal_sig}"
            elif partner:
                primary_key = f"{ticker}__{cluster}__partner:{partner}"
            elif deal_sig:
                primary_key = f"{ticker}__{cluster}__deal:{deal_sig}"
            else:
                primary_key = f"{ticker}__{cluster}"
            candidate_keys = [primary_key]
        
        # BUYBACKS: Use deal size if present
        elif cluster == 'share_buyback':
            deal_sig = extract_deal_signature(title)
            if deal_sig:
                primary_key = f"{ticker}__{cluster}__deal:{deal_sig}"
            else:
                primary_key = f"{ticker}__{cluster}"
            candidate_keys = [primary_key]
        
        # PHASE 3: Use drug name if present (both LONGS and SHORTS)
        elif cluster in ['phase3_achievement', 'phase3_failure']:
            drug_name = extract_drug_name(title, ticker)
            if drug_name:
                primary_key = f"{ticker}__{cluster}__drug:{drug_name}"
            else:
                primary_key = f"{ticker}__{cluster}"
            candidate_keys = [primary_key]
        
        # UPLISTING: Simple ticker + cluster
        else:
            primary_key = f"{ticker}__{cluster}"
            candidate_keys = [primary_key]
        
        # Check cooldown state
        first_article_time = None
        for ckey in candidate_keys:
            iso = cooldown_state.get(ckey)
            if iso:
                ts = datetime.fromisoformat(iso).replace(tzinfo=timezone.utc)
                if not first_article_time or ts < first_article_time:
                    first_article_time = ts
        
        # Determine if BREAKING or REPETITION
        if not first_article_time:
            # BREAKING - record timestamp
            iso_value = pub_time.isoformat()
            for ckey in candidate_keys:
                cooldown_state[ckey] = iso_value
            return False  # Not a duplicate
        else:
            # Check if cooldown period elapsed
            delta = pub_time.replace(tzinfo=timezone.utc) - first_article_time
            
            # Same-day: 24h cooldown, Different day: 14d cooldown
            is_same_day = (pub_time.date() == first_article_time.date())
            effective_cooldown = timedelta(hours=24) if is_same_day else timedelta(days=COOLDOWN_DAYS)
            
            if delta < effective_cooldown:
                return True  # DUPLICATE
            else:
                # Cooldown expired - new event
                iso_value = pub_time.isoformat()
                for ckey in candidate_keys:
                    cooldown_state[ckey] = iso_value
                return False
    
    return False


# ============================================================================
# PEER DETECTION SYSTEM (FOR SECTOR-LEVEL CLUSTERS)
# ============================================================================

PEERS_CACHE_FILE = os.path.join(BASE_DIR, "peers_cache.json")
BUILD_PEERS = os.getenv('BUILD_PEERS', '0') == '1'  # default skip to avoid long runs
PEERS_CACHE_EXPIRY_DAYS = 30  # Cache peers for 30 days

def load_peers_cache() -> Dict:
    """Load peers cache from disk"""
    if os.path.exists(PEERS_CACHE_FILE):
        try:
            with open(PEERS_CACHE_FILE, 'r') as f:
                cache = json.load(f)
            logger.info(f"Loaded peers cache with {len(cache)} entries")
            return cache
        except Exception as e:
            logger.warning(f"Failed to load peers cache: {e}")
    
    return {}


def save_peers_cache(cache: Dict):
    """Save peers cache to disk"""
    try:
        with open(PEERS_CACHE_FILE, 'w') as f:
            json.dump(cache, f, indent=2)
        logger.info(f"Saved peers cache with {len(cache)} entries")
    except Exception as e:
        logger.error(f"Failed to save peers cache: {e}")


def fetch_company_profile(ticker: str) -> Optional[Dict]:
    """Fetch company profile from Finnhub (country, sector, subsector, exchange)"""
    url = "https://finnhub.io/api/v1/stock/profile2"
    params = {
        'symbol': ticker,
        'token': FINNHUB_API_KEY
    }
    
    try:
        time.sleep(FINNHUB_RATE_LIMIT_DELAY)
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if data:  # Non-empty response
                return {
                    'country': data.get('country', ''),
                    'finnhubIndustry': data.get('finnhubIndustry', ''),
                    'name': data.get('name', ''),
                    'ticker': data.get('ticker', ticker),
                    'exchange': data.get('exchange', '')  # NYSE, NASDAQ, OTCMKTS, etc.
                }
            else:
                # Empty response = ticker not found (likely ETF, delisted, or invalid)
                logger.debug(f"No profile data for {ticker} (likely ETF/delisted/invalid)")
                return None
        
        logger.warning(f"API error fetching profile for {ticker}: HTTP {response.status_code}")
        return None
    
    except Exception as e:
        logger.error(f"Error fetching profile for {ticker}: {e}")
        return None


def fetch_peers(ticker: str) -> List[str]:
    """Fetch peers from Finnhub /stock/peers API"""
    url = "https://finnhub.io/api/v1/stock/peers"
    params = {
        'symbol': ticker,
        'token': FINNHUB_API_KEY
    }
    
    try:
        time.sleep(FINNHUB_RATE_LIMIT_DELAY)
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code == 200:
            peers = response.json()
            if isinstance(peers, list):
                # Filter out the original ticker
                peers = [p for p in peers if p.upper() != ticker.upper()]
                return peers
        
        logger.warning(f"Failed to fetch peers for {ticker}: {response.status_code}")
        return []
    
    except Exception as e:
        logger.error(f"Error fetching peers for {ticker}: {e}")
        return []


def get_curated_peers(ticker: str, cluster: str, peers_cache: Dict) -> List[str]:
    """
    Get curated peer list with country/sector filtering based on cluster type.
    
    Filtering logic:
    - regional_bank_fear: Regional Banks subsector only
    - large_bank_warning: NO PEERS (systematic risk)
    - tariff_rift_bearish: China exposure filter (exporters)
    - tariff_protection_bullish: US-based ONLY
    - trade_war_sector: China exposure
    - supply_halt_sector: Same sector
    - war_risk_sector: Same sector (exclude defense)
    - mandate_subsidy_sector: Same sector
    - aca_support_healthcare: Healthcare only
    - supply_recovery_sector: Same sector
    - defense_spending_sector: Defense/Aerospace only
    """
    # Check cache first
    cache_key = f"{ticker}__{cluster}"
    
    if cache_key in peers_cache:
        cache_entry = peers_cache[cache_key]
        # Check expiry
        cached_time = datetime.fromisoformat(cache_entry['timestamp'])
        if datetime.now() - cached_time < timedelta(days=PEERS_CACHE_EXPIRY_DAYS):
            logger.debug(f"Using cached peers for {ticker} ({cluster}): {len(cache_entry['peers'])} peers")
            return cache_entry['peers']
    
    # Fetch fresh data
    logger.info(f"Fetching peers for {ticker} (cluster: {cluster})...")
    
    # Get company profile for primary ticker
    primary_profile = fetch_company_profile(ticker)
    if not primary_profile:
        logger.warning(f"Could not fetch profile for primary ticker {ticker}")
        return []
    
    # Fetch peer list
    raw_peers = fetch_peers(ticker)
    if not raw_peers:
        logger.warning(f"No peers found for {ticker}")
        return []
    
    logger.info(f"Found {len(raw_peers)} raw peers for {ticker}, applying filters...")
    
    # Apply cluster-specific filtering
    curated_peers = []
    
    for peer_ticker in raw_peers:
        peer_profile = fetch_company_profile(peer_ticker)
        
        if not peer_profile:
            continue
        
        # Apply filters based on cluster type
        if cluster == 'regional_bank_fear':
            # ONLY Regional Banks subsector
            if 'Regional Banks' in peer_profile.get('finnhubIndustry', ''):
                curated_peers.append(peer_ticker)
        
        elif cluster == 'large_bank_warning':
            # NO PEERS - systematic risk
            pass
        
        elif cluster == 'tariff_rift_bearish':
            # China exposure (exporters) - simplified: include all non-US or US companies with China exposure
            # For now, accept same-sector peers (real implementation would need revenue breakdown)
            if peer_profile.get('finnhubIndustry', '') == primary_profile.get('finnhubIndustry', ''):
                curated_peers.append(peer_ticker)
        
        elif cluster == 'tariff_protection_bullish':
            # US-based ONLY
            if peer_profile.get('country', '').upper() == 'US' or peer_profile.get('country', '').upper() == 'UNITED STATES':
                curated_peers.append(peer_ticker)
        
        elif cluster in ['trade_war_sector', 'supply_halt_sector', 'war_risk_sector', 
                          'mandate_subsidy_sector', 'supply_recovery_sector']:
            # Same sector
            if peer_profile.get('finnhubIndustry', '') == primary_profile.get('finnhubIndustry', ''):
                curated_peers.append(peer_ticker)
        
        elif cluster == 'aca_support_healthcare':
            # Healthcare sector only
            if 'Health Care' in peer_profile.get('finnhubIndustry', '') or 'Healthcare' in peer_profile.get('finnhubIndustry', ''):
                curated_peers.append(peer_ticker)
        
        elif cluster == 'defense_spending_sector':
            # Defense/Aerospace only
            if 'Aerospace' in peer_profile.get('finnhubIndustry', '') or 'Defense' in peer_profile.get('finnhubIndustry', ''):
                curated_peers.append(peer_ticker)
        
        else:
            # Default: same sector
            if peer_profile.get('finnhubIndustry', '') == primary_profile.get('finnhubIndustry', ''):
                curated_peers.append(peer_ticker)
    
    logger.info(f"Curated {len(curated_peers)} peers for {ticker} after filtering")
    
    # Cache result
    peers_cache[cache_key] = {
        'ticker': ticker,
        'cluster': cluster,
        'peers': curated_peers,
        'timestamp': datetime.now().isoformat()
    }
    
    return curated_peers


def build_peers_cache(events: List[Dict]) -> Dict:
    """
    Build comprehensive peers cache for all sector-level events.
    This is called ONCE before the main backtest loop.
    
    Returns: peers_cache dict
    """
    logger.info("\n" + "="*80)
    logger.info("BUILDING PEERS CACHE FOR SECTOR-LEVEL CLUSTERS")
    logger.info("="*80)
    
    # Load existing cache
    peers_cache = load_peers_cache()
    
    # Find all unique (ticker, cluster) pairs that require peers
    sector_events = []
    for event in events:
        cluster = event['clusters'][0]  # Primary cluster
        cluster_config = CLUSTER_PATTERNS.get(cluster, {})
        
        if cluster_config.get('requires_peers', False):
            sector_events.append((event['ticker'], cluster))
    
    unique_pairs = list(set(sector_events))
    logger.info(f"Found {len(unique_pairs)} unique (ticker, cluster) pairs requiring peer detection")
    
    if len(unique_pairs) == 0:
        logger.info("No sector-level events found - skipping peer cache build")
        return peers_cache
    
    # Estimate API calls and time
    uncached_count = 0
    for ticker, cluster in unique_pairs:
        cache_key = f"{ticker}__{cluster}"
        if cache_key not in peers_cache:
            uncached_count += 1
    
    if uncached_count == 0:
        logger.info("All peers already cached!")
        return peers_cache
    
    # Each uncached pair needs: 1 profile call + 1 peers call + ~5 peer profile calls
    estimated_calls = uncached_count * 7
    estimated_time_mins = (estimated_calls * FINNHUB_RATE_LIMIT_DELAY) / 60
    
    logger.warning(f"Peer cache build will make ~{estimated_calls} API calls")
    logger.warning(f"Estimated time: {estimated_time_mins:.1f} minutes")
    logger.warning(f"Finnhub rate limit: 50 calls/min (delay: {FINNHUB_RATE_LIMIT_DELAY}s)")
    
    user_input = input("\nProceed with peer cache build? (yes/no): ").strip().lower()
    if user_input != 'yes':
        logger.info("Peer cache build cancelled by user")
        return peers_cache
    
    # Build cache
    logger.info("\nBuilding peer cache...")
    start_time = time.time()
    
    for i, (ticker, cluster) in enumerate(unique_pairs, 1):
        cache_key = f"{ticker}__{cluster}"
        
        # Skip if already cached
        if cache_key in peers_cache:
            cache_entry = peers_cache[cache_key]
            cached_time = datetime.fromisoformat(cache_entry['timestamp'])
            if datetime.now() - cached_time < timedelta(days=PEERS_CACHE_EXPIRY_DAYS):
                logger.debug(f"[{i}/{len(unique_pairs)}] Cached: {ticker} ({cluster})")
                continue
        
        logger.info(f"[{i}/{len(unique_pairs)}] Fetching peers: {ticker} ({cluster})")
        curated_peers = get_curated_peers(ticker, cluster, peers_cache)
        
        # Save cache every 10 entries (in case of interruption)
        if i % 10 == 0:
            save_peers_cache(peers_cache)
            logger.info(f"Checkpoint: Saved cache at {i}/{len(unique_pairs)}")
    
    # Final save
    save_peers_cache(peers_cache)
    
    elapsed_mins = (time.time() - start_time) / 60
    logger.info(f"\nPeer cache build complete in {elapsed_mins:.1f} minutes")
    logger.info(f"Cache contains {len(peers_cache)} entries")
    
    return peers_cache


# ============================================================================
# DATA FETCHING
# ============================================================================

def load_stocknews_cache(start_date: datetime, end_date: datetime) -> List[Dict]:
    """Load articles from StockNewsAPI cache"""
    articles = []
    loaded_files = []
    
    if not os.path.exists(CACHE_DIR):
        logger.warning(f"Cache directory {CACHE_DIR} not found")
        return articles
    
    current_date = start_date
    while current_date <= end_date:
        if current_date.weekday() >= 5:
            current_date += timedelta(days=1)
            continue
        date_str = current_date.strftime('%Y%m%d')
        cache_file = os.path.join(CACHE_DIR, f"alltickers_{date_str}_{date_str}.json")
        
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                    # Handle both list and dict formats
                    if isinstance(data, list):
                        cached_articles = data
                    elif isinstance(data, dict):
                        cached_articles = data.get('data', [])
                    else:
                        cached_articles = []
                    
                    articles.extend(cached_articles)
                    loaded_files.append(f"alltickers_{date_str}_{date_str}.json")
            except Exception as e:
                logger.error(f"Error loading {cache_file}: {e}")
        
        current_date += timedelta(days=1)
    
    logger.info(f"[OK] Loaded {len(articles)} articles from {len(loaded_files)} cache files (format: alltickers_YYYYMMDD_YYYYMMDD.json)")
    if loaded_files:
        logger.info(f"   First file: {loaded_files[0]}, Last file: {loaded_files[-1]}")
    return articles


# ============================================================================
# DIAGNOSTICS: List cluster-triggering articles for selected tickers
# ============================================================================

def _parse_article_dt(pub_date: str) -> Optional[datetime]:
    """Parse publication date from cache, supporting ISO8601 and RFC 2822.

    Note: Do NOT use a naive 'T' substring check (e.g., 'Thu' contains 'T').
    We specifically detect ISO with a YYYY-MM-DDT pattern; otherwise fall back to RFC 2822.
    """
    try:
        # Detect ISO-8601 like '2025-09-25T17:30:00Z' or with offset
        if re.search(r"\d{4}-\d{2}-\d{2}T", pub_date):
            try:
                dt = datetime.fromisoformat(pub_date.replace('Z', '+00:00'))
            except Exception:
                # Fallback to RFC 2822 if ISO parse fails
                from email.utils import parsedate_to_datetime
                dt = parsedate_to_datetime(pub_date)
        else:
            from email.utils import parsedate_to_datetime
            dt = parsedate_to_datetime(pub_date)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)
        return dt
    except Exception:
        return None

def list_cluster_matches(articles: List[Dict], cluster_name: str, tickers_wanted: Set[str]) -> List[Dict]:
    out: List[Dict] = []
    cluster = CLUSTER_PATTERNS.get(cluster_name)
    if not cluster:
        return out
    pos_res = cluster.get('positive', [])
    neg_res = cluster.get('negative', [])
    for art in articles:
        title = art.get('title') or ''
        if not title:
            continue
        raw_tickers = art.get('tickers', [])
        if isinstance(raw_tickers, list):
            arts = [t.upper() for t in raw_tickers]
        elif isinstance(raw_tickers, str):
            arts = [t.strip().upper() for t in raw_tickers.split(',') if t.strip()]
        else:
            arts = []
        if not (set(arts) & tickers_wanted):
            continue
        # Negative first
        if any(rx.search(title) for rx in neg_res):
            continue
        matched = [rx.pattern for rx in pos_res if rx.search(title)]
        if matched:
            dt = _parse_article_dt(art.get('date', '') or '')
            out.append({
                'dt': dt.isoformat() if dt else art.get('date'),
                'tickers': arts,
                'title': title,
                'source': art.get('source_name') or art.get('source') or '',
                'matched_patterns': matched,
                'url': art.get('url'),
            })
    return out


def load_vix_data(start_date: datetime, end_date: datetime) -> Optional[pd.Series]:
    """
    Load VIX data from tick cache for regime filtering.
    Returns pd.Series with date index and VIX close values.
    """
    try:
        logger.info(f"Loading VIX data from {VIX_CACHE_DIR}...")
        
        vix_prices = []
        current_date = start_date.date()
        end = end_date.date()
        
        while current_date <= end:
            date_str = current_date.strftime('%Y-%m-%d')
            vix_file = VIX_CACHE_DIR / f"VIX_{date_str}.parquet"
            
            if vix_file.exists():
                try:
                    df = pd.read_parquet(vix_file)
                    if not df.empty and 'price' in df.columns:
                        # Use last price of the day as close
                        close_price = df['price'].iloc[-1]
                        vix_prices.append({
                            'date': pd.Timestamp(current_date),
                            'vix_close': close_price
                        })
                except Exception as e:
                    logger.warning(f"Error reading {vix_file}: {e}")
            
            current_date += timedelta(days=1)
        
        if not vix_prices:
            logger.warning("⚠️  No VIX data found - regime filter will be DISABLED")
            return None
        
        # Create Series with date index
        vix_df = pd.DataFrame(vix_prices).set_index('date')
        vix_series = vix_df['vix_close']
        
        logger.info(f"✓ Loaded VIX for {len(vix_series)} days")
        logger.info(f"  VIX range: {vix_series.min():.1f} - {vix_series.max():.1f}")
        
        # Count high VIX days
        high_vix_days = (vix_series > VIX_REGIME_THRESHOLD).sum()
        if high_vix_days > 0:
            logger.info(f"  High VIX days (>{VIX_REGIME_THRESHOLD}): {high_vix_days}")
        
        return vix_series
        
    except Exception as e:
        logger.error(f"Failed to load VIX data: {e}")
        logger.warning("⚠️  VIX regime filter will be DISABLED")
        return None



def get_finnhub_candles(ticker: str, start_ts: int, end_ts: int) -> Optional[pd.DataFrame]:
    """
    DISABLED in tick-only backtest. Never use minute candles here.
    If called, raise immediately to prevent accidental fetching.
    """
    raise RuntimeError("Minute candles are disabled in this tick-only backtest. Use tick cache-derived OHLCV.")
    # Simple on-disk cache to avoid re-downloading same candles repeatedly
    cache_dir = os.path.join(BASE_DIR, '.candles_cache')
    os.makedirs(cache_dir, exist_ok=True)
    # make a filesystem-safe ticker filename
    safe_ticker = re.sub(r'[^A-Za-z0-9_.-]', '_', ticker)
    cache_file = os.path.join(cache_dir, f"{safe_ticker}.pkl")

    url = f"https://finnhub.io/api/v1/stock/candle"
    params = {
        'symbol': ticker,
        'resolution': '1',  # 1-minute candles
        'from': start_ts,
        'to': end_ts,
        'token': FINNHUB_API_KEY
    }
    
    # If cached file exists and covers requested range, use it
    if os.path.exists(cache_file):
        try:
            cached_df = pd.read_pickle(cache_file)
            # Ensure timezone-aware
            if pd.api.types.is_datetime64_any_dtype(cached_df['timestamp']):
                if cached_df['timestamp'].dt.tz is None:
                    cached_df['timestamp'] = cached_df['timestamp'].dt.tz_localize('UTC')
                else:
                    cached_df['timestamp'] = cached_df['timestamp'].dt.tz_convert('UTC')

            min_ts = int(cached_df['timestamp'].min().timestamp())
            max_ts = int(cached_df['timestamp'].max().timestamp())
            if min_ts <= start_ts and max_ts >= end_ts:
                logger.info(f"Loaded Finnhub candles for {ticker} from cache")
                # return subset to requested range
                start_dt = pd.to_datetime(start_ts, unit='s', utc=True)
                end_dt = pd.to_datetime(end_ts, unit='s', utc=True)
                subset = cached_df[(cached_df['timestamp'] >= start_dt) & (cached_df['timestamp'] <= end_dt)].copy()
                return subset
        except Exception as e:
            logger.warning(f"Failed to load cache for {ticker}, will fetch: {e}")

    try:
        time.sleep(FINNHUB_RATE_LIMIT_DELAY)  # Rate limiting
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        if data.get('s') != 'ok':
            logger.warning(f"Finnhub returned status {data.get('s')} for {ticker}")
            return None

        df = pd.DataFrame({
            'timestamp': pd.to_datetime(data['t'], unit='s', utc=True),
            'open': data['o'],
            'high': data['h'],
            'low': data['l'],
            'close': data['c'],
            'volume': data['v']
        })

        # Save to cache for future runs
        try:
            df.to_pickle(cache_file)
            logger.debug(f"Saved Finnhub candles to cache: {cache_file}")
        except Exception as e:
            logger.warning(f"Could not save candles cache for {ticker}: {e}")

        return df

    except Exception as e:
        logger.error(f"Error fetching Finnhub data for {ticker}: {e}")
        return None


# ============================================================================
# MOMENTUM ANALYSIS
# ============================================================================

def check_momentum(
    ticker: str,
    event_time: datetime,
    candles_df: pd.DataFrame,
    direction: str = 'LONG'
) -> Optional[Dict]:
    """
    Check if momentum gates are met 60 seconds AFTER event_time (like orchestrator).
    Uses tick and second-level data from Finnhub.
    For LONG: Price up + Volume up = bullish momentum
    For SHORT: Price down + Volume up = bearish momentum (inverted)
    Returns momentum signal with entry_price, pattern, volume_ratio, etc.
    """
    if candles_df is None or len(candles_df) == 0:
        return None
    
    # Apply 60-second delay before checking momentum
    check_time = event_time + timedelta(seconds=NEWS_ENTRY_DELAY)
    
    # Find candle closest to check time (60s after event)
    candles_df['time_diff'] = abs((candles_df['timestamp'] - check_time).dt.total_seconds())
    check_candle_idx = candles_df['time_diff'].idxmin()
    
    if candles_df.loc[check_candle_idx, 'time_diff'] > 120:  # More than 2 min away from check time
        return None
    
    # Get 60-second window around check time for momentum measurement
    check_ts = candles_df.loc[check_candle_idx, 'timestamp']
    window_start = check_ts - timedelta(seconds=30)
    window_end = check_ts + timedelta(seconds=30)
    
    window_df = candles_df[
        (candles_df['timestamp'] >= window_start) &
        (candles_df['timestamp'] <= window_end)
    ]
    
    if len(window_df) == 0:
        return None
    
    # Calculate price change and volume (using second-level ticks)
    entry_price = window_df.iloc[-1]['close']
    
    # Reference price is at event time (not check time)
    event_candles = candles_df[abs((candles_df['timestamp'] - event_time).dt.total_seconds()) <= 30]
    if len(event_candles) == 0:
        return None
    reference_price = event_candles.iloc[0]['open']
    
    price_change_pct = ((entry_price - reference_price) / reference_price) * 100
    
    # Calculate volume ratio (vs last 60 minutes average before event)
    baseline_start = event_time - timedelta(minutes=60)
    baseline_end = event_time
    baseline_df = candles_df[
        (candles_df['timestamp'] >= baseline_start) &
        (candles_df['timestamp'] < baseline_end)
    ]
    
    if len(baseline_df) == 0:
        return None
    
    baseline_volume = baseline_df['volume'].mean()
    
    # Current volume is the sum from event to check time (60 seconds of trading)
    current_volume_df = candles_df[
        (candles_df['timestamp'] >= event_time) &
        (candles_df['timestamp'] <= check_ts)
    ]
    current_volume = current_volume_df['volume'].sum()
    volume_ratio = current_volume / (baseline_volume * 60) if baseline_volume > 0 else 0  # Normalize to 60-bar window
    
    # Check momentum gates (INVERTED for shorts) with WEAK pattern detection
    pattern = None
    abs_price_change = abs(price_change_pct)
    
    if direction == 'LONG':
        # LONG: Need positive price movement
        if (price_change_pct >= EXPLOSIVE_PRICE_THRESHOLD and 
            volume_ratio >= EXPLOSIVE_VOLUME_RATIO):
            pattern = "EXPLOSIVE_LONG"
        elif (price_change_pct >= SUSTAINED_PRICE_THRESHOLD and 
              volume_ratio >= SUSTAINED_VOLUME_RATIO):
            pattern = "SUSTAINED_LONG"
        else:
            # WEAK_MOMENTUM: Insufficient price or volume
            pattern = "WEAK_MOMENTUM_LONG"
    else:  # SHORT
        # SHORT: Need negative price movement (inverted)
        if (price_change_pct <= -EXPLOSIVE_PRICE_THRESHOLD and 
            volume_ratio >= EXPLOSIVE_VOLUME_RATIO):
            pattern = "EXPLOSIVE_SHORT"
        elif (price_change_pct <= -SUSTAINED_PRICE_THRESHOLD and 
              volume_ratio >= SUSTAINED_VOLUME_RATIO):
            pattern = "SUSTAINED_SHORT"
        else:
            # WEAK_MOMENTUM: Insufficient price or volume
            pattern = "WEAK_MOMENTUM_SHORT"
    
    # ⚠️ CRITICAL FILTER: Skip WEAK_MOMENTUM patterns (0% win rate in orchestrator)
    # From news_trade_orchestrator_finnhub.py: "SKIP ALL WEAK_MOMENTUM PATTERNS (0% win rate, 10 losing trades)"
    if pattern.startswith('WEAK_MOMENTUM'):
        logger.debug(f"Skipping {ticker} - WEAK_MOMENTUM pattern filtered (price: {price_change_pct:.2f}%, vol: {volume_ratio:.2f}x)")
        return None
    
    return {
        'entry_price': entry_price,
        'pattern': pattern,
        'price_change_pct': price_change_pct,
        'volume_ratio': volume_ratio,
        'baseline_volume': baseline_volume,
        'event_timestamp': check_ts,  # Entry timestamp is 60s after news
        'news_timestamp': event_time,  # Original news timestamp
        'direction': direction
    }


# ============================================================================
# TICK-MODE MOMENTUM (optional, closer to orchestrator)
# ============================================================================

def _build_tick_baseline(second_df: pd.DataFrame, baseline_start: datetime, baseline_end: datetime) -> float:
    """Compute baseline per-second volume from an OHLCV dataframe with 'timestamp' column."""
    col = 'timestamp' if 'timestamp' in second_df.columns else 'time_dt'
    base = second_df[(second_df[col] >= baseline_start) & (second_df[col] < baseline_end)]
    if base.empty:
        return 0.0
    # Use median per-second volume for robustness
    vol_col = 'volume' if 'volume' in base.columns else ('size' if 'size' in base.columns else None)
    if vol_col is None or base[vol_col].empty:
        return 0.0
    median_vol = float(base[vol_col].median())
    if median_vol > 0:
        return median_vol
    # Fallbacks when median is 0 due to sparse after-hours trading
    try:
        q75 = float(base[vol_col].quantile(0.75))
        if q75 > 0:
            return q75
    except Exception:
        pass
    nonzero = base[vol_col][base[vol_col] > 0]
    if not nonzero.empty:
        return float(nonzero.mean())
    # Last resort to avoid division by zero; very conservative
    return 1.0


def check_momentum_ticks(
    ticker: str,
    event_time: datetime,
    ticks_df: pd.DataFrame,
    direction: str = 'LONG',
    entry_delay_sec: int = 60,
    sustained_price_thr: float = SUSTAINED_PRICE_THRESHOLD,
    sustained_vol_ratio: float = SUSTAINED_VOLUME_RATIO,
    explosive_price_thr: float = EXPLOSIVE_PRICE_THRESHOLD,
    explosive_vol_ratio: float = EXPLOSIVE_VOLUME_RATIO,
) -> Optional[Dict]:
    """
    Tick-level momentum evaluation similar to orchestrator:
    - Wait entry_delay_sec after news
    - Compare last trade price vs price at news time
    - Volume ratio vs baseline per-second volume in the prior 60 minutes
    Returns None to skip when WEAK.
    """
    if ticks_df is None or ticks_df.empty:
        return None

    # Ensure time is tz-aware UTC
    if event_time.tzinfo is None:
        event_time = event_time.replace(tzinfo=timezone.utc)
    else:
        event_time = event_time.astimezone(timezone.utc)

    # Build per-second aggregation to stabilize noise
    sec = ticks_df.copy()
    if 'time_dt' not in sec.columns:
        return None
    # Restrict to +/- 2 hours around event to avoid huge memory
    t0 = event_time - timedelta(hours=2)
    t1 = event_time + timedelta(hours=2)
    sec = sec[(sec['time_dt'] >= t0) & (sec['time_dt'] <= t1)]
    if sec.empty:
        return None

    # Compute per-second OHLCV from ticks
    sec['sec'] = sec['time_dt'].dt.floor('s')
    ohlcv = sec.groupby('sec').agg(
        open=('price', 'first'),
        high=('price', 'max'),
        low=('price', 'min'),
        close=('price', 'last'),
        volume=('size', 'sum'),
    ).reset_index().rename(columns={'sec': 'timestamp'})

    # Reference price at/just before event
    pre_row = ohlcv[ohlcv['timestamp'] <= event_time]
    if pre_row.empty:
        return None
    pre_price = float(pre_row.iloc[-1]['close'])

    # Baseline volume: median per-second size over 60 minutes before event
    baseline_start = event_time - timedelta(minutes=60)
    baseline_end = event_time
    baseline_volume = _build_tick_baseline(ohlcv, baseline_start, baseline_end)
    if baseline_volume <= 0:
        # Optional debug: baseline too low
        if os.getenv('LOG_MOMENTUM_DETAILS', '0') == '1':
            logger.info(f"[Momentum] {ticker} baseline_volume<=0 around {event_time.isoformat()} — skipping")
        return None

    # Evaluate at check time = event + 60s
    check_time = event_time + timedelta(seconds=NEWS_ENTRY_DELAY)
    check_row = ohlcv[ohlcv['timestamp'] <= check_time]
    if check_row.empty:
        if os.getenv('LOG_MOMENTUM_DETAILS', '0') == '1':
            logger.info(f"[Momentum] {ticker} no trades up to check_time {check_time.isoformat()} — skipping")
        return None
    entry_price = float(check_row.iloc[-1]['close'])

    # Compute price change and volume from event->check
    price_change_pct = ((entry_price - pre_price) / pre_price) * 100.0
    current_vol = ohlcv[(ohlcv['timestamp'] > event_time) & (ohlcv['timestamp'] <= check_time)]['volume'].sum()
    # Normalize to per-second baseline: divide by (baseline_volume * seconds)
    seconds = max(1, int((check_time - event_time).total_seconds()))
    volume_ratio = (current_vol / seconds) / baseline_volume if baseline_volume > 0 else 0.0

    # RELAXED Gate logic: Allow all momentum patterns (0.1% / 1.5x threshold)
    if direction == 'LONG':
        if price_change_pct >= explosive_price_thr and volume_ratio >= explosive_vol_ratio:
            pattern = 'EXPLOSIVE_LONG_TICK'
        elif price_change_pct >= 0.1 and volume_ratio >= 1.5:  # RELAXED: 0.1% / 1.5x
            pattern = 'SUSTAINED_LONG_TICK'
        else:
            pattern = 'WEAK_MOMENTUM_LONG'  # ALLOW weak patterns
    else:
        if price_change_pct <= -explosive_price_thr and volume_ratio >= explosive_vol_ratio:
            pattern = 'EXPLOSIVE_SHORT_TICK'
        elif price_change_pct <= -0.1 and volume_ratio >= 1.5:  # RELAXED: 0.1% / 1.5x
            pattern = 'SUSTAINED_SHORT_TICK'
        else:
            pattern = 'WEAK_MOMENTUM_SHORT'  # ALLOW weak patterns
            if os.getenv('LOG_MOMENTUM_DETAILS', '0') == '1':
                logger.info(f"[Momentum] {ticker} SHORT WEAK @60s price={price_change_pct:.2f}% volx={volume_ratio:.2f} (ALLOWING)")
    
    # REMOVED: Hard filter that returned None for weak momentum

    if os.getenv('LOG_MOMENTUM_DETAILS', '0') == '1':
        logger.info(f"[Momentum] {ticker} PASS {pattern} @60s price={price_change_pct:.2f}% volx={volume_ratio:.2f}")
    return {
        'entry_price': entry_price,
        'pattern': pattern,
        'price_change_pct': price_change_pct,
        'volume_ratio': volume_ratio,
        'baseline_volume': baseline_volume,
        'event_timestamp': check_time,
        'news_timestamp': event_time,
        'direction': direction,
    }


# ============================================================================
# ============================================================================
# VOLUME EXHAUSTION EXIT - DEPRECATED (Removed in favor of ATR-based stops)
# ============================================================================
# This function has been disabled. Exit logic now uses:
# 1. ATR-based hard stop (USE_ATR_STOP, ATR_MULTIPLIER, MIN_STOP_PCT)
# 2. Take profit at 1R multiple (TAKE_PROFIT_R_MULTIPLE)
# 3. Time stop (MAX_HOLD_SECONDS)
#
# def check_volume_exhaustion_orchestrator(...):
#     """DEPRECATED - No longer used"""
#     pass


# ============================================================================
# BACKTESTING ENGINE
# ============================================================================

def _is_rth(ts_utc: datetime) -> bool:
    """Return True if the given UTC timestamp falls within extended trading hours (07:00-17:50 ET)."""
    try:
        ny = ZoneInfo('America/New_York')
        local = ts_utc.astimezone(ny)
        t = local.time()
        return (t >= RTH_START) and (t <= RTH_END)
    except Exception:
        return True  # Fail-open if timezone libs unavailable


def _compute_atr_like(ohlcv: pd.DataFrame, end_ts: datetime, lookback_seconds: int = ATR_LOOKBACK_SECONDS) -> Optional[float]:
    """Compute an ATR-like per-second volatility using absolute close-to-close returns over a lookback window."""
    start_ts = end_ts - timedelta(seconds=lookback_seconds)
    win = ohlcv[(ohlcv['timestamp'] > start_ts) & (ohlcv['timestamp'] <= end_ts)].copy()
    if win.empty or len(win) < 10:
        return None
    win['ret'] = win['close'].pct_change().abs()
    vol = float(win['ret'].median()) if win['ret'].notna().any() else None
    if vol is None or vol == 0:
        try:
            vol = float(win['ret'].mean())
        except Exception:
            return None
    return vol if vol and vol > 0 else None

class BacktestEngine:
    def __init__(self, initial_capital: float, params: Optional[Dict[str, Any]] = None):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.open_positions = {}  # ticker -> position_data
        self.closed_trades = []
        self.cooldown_state = {}  # For duplicate detection
        self.equity_curve = [initial_capital]
        # In-memory tick cache: {(ticker, YYYY-MM-DD): DataFrame}
        self.tick_day_cache = {}
        # In-memory per-second OHLCV cache built from ticks: {(ticker, YYYY-MM-DD): DataFrame}
        self.tick_ohlcv_cache = {}
        # Exchange cache: {ticker: exchange_code} to avoid repeated API calls
        self.exchange_cache = {}
        # Symbol lookup from Finnhub: {ticker: {'mic': 'XNAS', 'type': 'Common Stock', ...}}
        self.symbol_lookup = {}
        # Track last cleanup to avoid excessive calls
        self.last_cache_cleanup = None
        # Merge provided params with defaults so every key is available
        self.params: Dict[str, Any] = {**DEFAULT_STRATEGY_PARAMS, **(params or {})}
        # Track every breaking headline and whether we entered a trade
        self.event_log: List[Dict[str, Any]] = []
        
        # Load US symbol lookup with MIC codes and types (for ETF detection)
        self._load_symbol_lookup()

    def _log_event(self, event: Dict[str, Any], status: str, reason: Optional[str] = None, extra: Optional[Dict[str, Any]] = None) -> None:
        """Append a normalized record describing how we handled a news event."""
        pub_time = event['pub_time']
        if isinstance(pub_time, datetime):
            pub_iso = (pub_time.astimezone(timezone.utc) if pub_time.tzinfo else pub_time.replace(tzinfo=timezone.utc)).isoformat()
        else:
            pub_iso = str(pub_time)

        entry = {
            'ticker': event.get('ticker'),
            'title': event.get('title', ''),
            'source': event.get('source', ''),
            'pub_time': pub_iso,
            'direction': event.get('direction'),
            'clusters': ','.join(event.get('clusters', [])),
            'status': status,
            'reason': reason or ''
        }

        if extra:
            entry.update(extra)

        self.event_log.append(entry)

    def _load_symbol_lookup(self) -> None:
        """
        Load US symbol lookup from Finnhub stock_symbols endpoint.
        This gives us MIC codes and security types (to detect ETFs via type=='ETP').
        Much cleaner than maintaining a 658-ETF list or dealing with full exchange names.
        """
        logger.info("[Symbol Lookup] Fetching US symbols from Finnhub stock_symbols('US')...")
        
        try:
            import finnhub
            client = finnhub.Client(api_key=FINNHUB_API_KEY)
            us_symbols = client.stock_symbols('US')
            
            # Build lookup dict: symbol -> info
            self.symbol_lookup = {item['symbol']: item for item in us_symbols}
            
            # Count ETPs (ETFs, ETNs, etc.)
            etp_count = sum(1 for info in self.symbol_lookup.values() if info.get('type') == 'ETP')
            
            logger.info(f"[Symbol Lookup] ✓ Loaded {len(self.symbol_lookup):,} US symbols")
            logger.info(f"[Symbol Lookup] Found {etp_count:,} ETPs (will be filtered)")
            
            # Test some symbols
            test_tickers = ['AAPL', 'AAAU', 'SPY', 'WTKWY']
            test_results = []
            for ticker in test_tickers:
                if ticker in self.symbol_lookup:
                    info = self.symbol_lookup[ticker]
                    mic = info.get('mic', 'N/A')
                    type_ = info.get('type', 'N/A')
                    test_results.append(f"{ticker}({mic},{type_})")
            
            logger.info(f"[Symbol Lookup] Test: {', '.join(test_results)}")
            
        except Exception as e:
            logger.error(f"[Symbol Lookup] Failed to load symbols: {e}")
            logger.warning("[Symbol Lookup] Falling back to profile-based filtering (less reliable)")
            self.symbol_lookup = {}
    def _cleanup_old_cache_entries(self, current_date: datetime):
        """Remove cache entries older than 7 days to prevent memory bloat"""
        # Only cleanup once per day
        if self.last_cache_cleanup and (current_date - self.last_cache_cleanup).days < 1:
            return
        
        cutoff = current_date - timedelta(days=7)
        
        for cache in [self.tick_day_cache, self.tick_ohlcv_cache]:
            keys_to_remove = []
            for (ticker_key, date_str) in list(cache.keys()):
                try:
                    cache_date = datetime.strptime(date_str, '%Y-%m-%d').replace(tzinfo=timezone.utc)
                    if cache_date < cutoff:
                        keys_to_remove.append((ticker_key, date_str))
                except Exception:
                    continue
            
            for key in keys_to_remove:
                del cache[key]
            
            if keys_to_remove:
                logger.info(f"[Cache cleanup] Removed {len(keys_to_remove)} entries older than {cutoff.date()}")
        
        self.last_cache_cleanup = current_date

    def _is_allowed_exchange(self, ticker: str) -> tuple[bool, Optional[str]]:
        """
        Check if ticker trades on an allowed exchange (NYSE/NASDAQ only, exclude OTC and ETFs).
        Returns (is_allowed, exchange_code).
        Uses symbol_lookup (loaded at startup) for MIC codes and type detection.
        """
        ticker = ticker.upper()
        
        # PRIORITY 1: Check cache
        if ticker in self.exchange_cache:
            exchange = self.exchange_cache[ticker]
            is_allowed = exchange in ALLOWED_EXCHANGES
            return (is_allowed, exchange)
        
        # PRIORITY 2: Check symbol_lookup for MIC code and security type
        if ticker in self.symbol_lookup:
            symbol_info = self.symbol_lookup[ticker]
            mic = symbol_info.get('mic', '').upper()
            sec_type = symbol_info.get('type', '')
            
            # Detect ETFs via type field
            if sec_type == 'ETP':
                logger.info(f"[Exchange] {ticker} is ETF/ETP (type={sec_type}), skipping")
                self.exchange_cache[ticker] = 'ETP'
                return (False, 'ETP')
            
            # Use MIC code for exchange validation
            self.exchange_cache[ticker] = mic
            
            # Check if exchange is explicitly excluded (OTC, foreign exchanges)
            if mic in EXCLUDED_EXCHANGES:
                logger.info(f"[Exchange] {ticker} on excluded exchange (MIC={mic}), skipping")
                return (False, mic)
            
            # Check if exchange is in allowed list (XNAS, XNYS, etc.)
            is_allowed = mic in ALLOWED_EXCHANGES
            if is_allowed:
                logger.info(f"[Exchange] {ticker} ALLOWED (MIC={mic}, type={sec_type})")
            else:
                logger.info(f"[Exchange] {ticker} NOT ALLOWED (MIC={mic})")
            return (is_allowed, mic)
        
        # PRIORITY 3: Pattern-based ETF detection (fallback for tickers not in symbol_lookup)
        etf_patterns = ['ETF', 'FUND', 'INDEX', 'TRUST']
        if any(pattern in ticker for pattern in etf_patterns):
            logger.debug(f"[Exchange] {ticker} appears to be ETF/Fund (ticker pattern), skipping")
            self.exchange_cache[ticker] = 'ETF'
            return (False, 'ETF')
        
        # PRIORITY 4: Fetch company profile as last resort (for tickers not in symbol_lookup)
        profile = fetch_company_profile(ticker)
        if not profile:
            # If we can't fetch profile, default to skip (cautious approach)
            logger.debug(f"[Exchange] Could not fetch profile for {ticker}, skipping (not in symbol_lookup, no profile)")
            self.exchange_cache[ticker] = 'UNKNOWN'
            return (False, None)
        
        # PRIORITY 5: Name-based ETF detection (catches funds with stock-like tickers)
        company_name = profile.get('name', '').upper()
        etf_name_keywords = ['ETF', 'FUND', 'TRUST', 'INDEX', 'ISHARES', 'SPDR', 'VANGUARD', 'INVESCO', 
                             'PROSHARES', 'DIREXION', 'ARK', 'SCHWAB ETF', 'JPMORGAN BETABUILDERS']
        if any(keyword in company_name for keyword in etf_name_keywords):
            logger.debug(f"[Exchange] {ticker} ({company_name}) is ETF/Fund (name check), skipping")
            self.exchange_cache[ticker] = 'ETF'
            return (False, 'ETF')
        
        # PRIORITY 6: Exchange validation using profile (fallback - returns full names, not reliable)
        exchange = profile.get('exchange', '').upper()
        self.exchange_cache[ticker] = exchange
        
        # Check if exchange is explicitly excluded
        if exchange in EXCLUDED_EXCHANGES:
            return (False, exchange)
        
        # Check if exchange is in allowed list
        is_allowed = exchange in ALLOWED_EXCHANGES
        logger.debug(f"[Exchange] {ticker} validation via profile: exchange={exchange}, allowed={is_allowed}")
        return (is_allowed, exchange)

    def _get_ticks_for_event_day(self, ticker: str, event_time: datetime) -> Optional[pd.DataFrame]:
        """
        Get tick data for the trading day of the event (UTC date).
        CRITICAL: NEVER fetch from API - only use existing cache.
        If cache doesn't exist, return None (trade will be skipped).
        """
        if fh_fetch_ticks_for_date is None or finnhub is None:
            return None
        
        # Use UTC date for cache key
        ev_dt_utc = event_time.astimezone(timezone.utc) if event_time.tzinfo else event_time.replace(tzinfo=timezone.utc)
        date_str = ev_dt_utc.strftime('%Y-%m-%d')
        key = (ticker.upper(), date_str)
        
        # Check in-memory cache first (fastest)
        if key in self.tick_day_cache:
            return self.tick_day_cache[key]
        
        # Check if disk cache file exists
        cache_exists = False
        try:
            if fh_get_tick_cache_path is not None:
                cache_path = fh_get_tick_cache_path(ticker.upper(), date_str)
                cache_exists = cache_path.exists()
        except Exception:
            pass
        
        # CRITICAL: If no cache, skip trade immediately (don't fetch)
        if not cache_exists:
            logger.debug(f"[Ticks skip] {ticker.upper()} {date_str} -> no cache (trade skipped)")
            return None
        
        # Cache exists - try to read it
        logger.info(f"[Ticks cache] {ticker.upper()} {date_str} -> cached")
        try:
            client = finnhub.Client(api_key=FINNHUB_API_KEY)
        except Exception:
            return None
        
        try:
            # CRITICAL: cache_only enforces NO API fetches (global FORCE_CACHE_ONLY ensures strictness)
            # Suppress stdout to avoid Unicode encoding errors on Windows console
            import sys, io
            old_stdout = sys.stdout
            sys.stdout = io.StringIO()  # Capture output
            try:
                df = fh_fetch_ticks_for_date(
                    client,
                    ticker.upper(),
                    date_str,
                    verbose=False,
                    local_tz=timezone.utc,
                    use_cache=True,
                    cache_only=True if FORCE_CACHE_ONLY else True
                )
            finally:
                sys.stdout = old_stdout  # Restore
        except Exception as e:
            logger.warning(f"[Ticks error] {ticker.upper()} {date_str} -> cache read failed: {e}")
            return None
        
        if df is not None and not df.empty:
            self.tick_day_cache[key] = df
            logger.info(f"[Ticks ready] {ticker.upper()} {date_str} -> {len(df):,} ticks")
        else:
            logger.warning(f"[Ticks skip] {ticker.upper()} {date_str} -> empty/invalid cache")
        
        return df

    def _get_ohlcv_for_event_day(self, ticker: str, event_time: datetime) -> Optional[pd.DataFrame]:
        """Return per-second OHLCV DataFrame for the event day, derived from ticks and cached in-memory."""
        ev_dt_utc = event_time.astimezone(timezone.utc) if event_time.tzinfo else event_time.replace(tzinfo=timezone.utc)
        date_str = ev_dt_utc.strftime('%Y-%m-%d')
        key = (ticker.upper(), date_str)
        if key in self.tick_ohlcv_cache:
            return self.tick_ohlcv_cache[key]
        ticks = self._get_ticks_for_event_day(ticker, event_time)
        if ticks is None or ticks.empty:
            return None
        sec = ticks.copy()
        if 'time_dt' not in sec.columns:
            return None
        sec['sec'] = sec['time_dt'].dt.floor('s')
        ohlcv = sec.groupby('sec').agg(
            open=('price', 'first'),
            high=('price', 'max'),
            low=('price', 'min'),
            close=('price', 'last'),
            volume=('size', 'sum'),
        ).reset_index().rename(columns={'sec': 'timestamp'})
        self.tick_ohlcv_cache[key] = ohlcv
        return ohlcv
        
    def process_event(self, event: Dict, candles_cache: Dict[str, pd.DataFrame]):
        """Process a trading event (LONG or SHORT) and log the outcome."""
        ticker = event['ticker']
        event_time = event['pub_time']
        clusters = event['clusters']
        direction = event['direction']  # 'LONG' or 'SHORT'

        def skip_event(reason: str, extra: Optional[Dict[str, Any]] = None) -> bool:
            if os.getenv('LOG_MOMENTUM_DETAILS', '0') == '1' and reason == 'NON_RTH':
                ts_utc = event_time if event_time.tzinfo else event_time.replace(tzinfo=timezone.utc)
                logger.info(f"[Gate] {ticker} skipped (non-RTH event time {ts_utc.isoformat()})")
            if os.getenv('LOG_MOMENTUM_DETAILS', '0') == '1' and reason == 'MIN_PRICE_FILTER':
                entry_price = (extra or {}).get('entry_price', 0.0)
                logger.info(f"[Gate] {ticker} skipped (entry price ${entry_price:.2f} < MIN_PRICE ${MIN_PRICE:.2f})")
            self._log_event(event, 'SKIPPED', reason, extra)
            return False

        # Periodic cache cleanup to prevent memory bloat
        self._cleanup_old_cache_entries(event_time)

        # Check if we already have position in this ticker
        if ticker in self.open_positions:
            return skip_event('ALREADY_IN_POSITION')

        # Check if max positions reached
        if len(self.open_positions) >= MAX_CONCURRENT_POSITIONS:
            return skip_event('MAX_POSITIONS_REACHED')

        # PRECEDENCE DUPLICATE CHECK: Skip if this event is duplicate of earlier breaking news
        # Import precedence detection functions
        try:
            from check_news_precedence import (
                load_news_for_date,
                get_ticker_news_before_event,
                detect_duplicate_content_patterns
            )
            
            # Get all news for this ticker on this day BEFORE event time
            event_date_str = event_time.strftime('%Y-%m-%d')
            all_articles = load_news_for_date(event_date_str)
            precedent_news = get_ticker_news_before_event(ticker, event_time, all_articles)
            
            # Detect if current event is duplicate of earlier news
            if precedent_news:
                duplicates = detect_duplicate_content_patterns(precedent_news, event)
                if duplicates:
                    # Get highest similarity match
                    best_match = duplicates[0]
                    similarity = best_match['similarity_score']
                    match_type = best_match.get('match_type', 'SIMILARITY_SCORE')
                    
                    # Extract event metadata for tier-1 source check
                    event_source = event.get('source', '')
                    event_title = event.get('title', '')
                    
                    # BREAKING NEWS PROTECTION: Don't filter tier-1 sources or press releases with definitive deal language
                    # Tier-1 sources: WSJ, Bloomberg, Reuters, CNBC (reporting), Financial Times
                    # Press releases: PRNewswire, GlobeNewsWire, Business Wire (company announcements)
                    tier1_sources = {'WSJ', 'WALL STREET JOURNAL', 'BLOOMBERG', 'REUTERS', 'FINANCIAL TIMES', 'FT'}
                    press_releases = {'PRNEWSWIRE', 'PR NEWSWIRE', 'GLOBENEWSWIRE', 'BUSINESS WIRE', 'ACCESSWIRE', 'NEWSFILE'}
                    
                    is_tier1 = event_source.upper() in tier1_sources
                    is_press_release = event_source.upper() in press_releases
                    
                    has_definitive_language = bool(re.search(
                        r'\b(signs?|signed|announces?|announced|enters?|entered)\s+(definitive|strategic|$\d+[MB])\s+(agreement|deal|contract|partnership|investment|acquisition)',
                        event_title,
                        re.IGNORECASE
                    ))
                    
                    # If similarity > 0.30, this is a duplicate of earlier breaking news
                    # Lowered from 0.35 to 0.30 to catch duplicates like AVGO (WSJ vs CNBC same story)
                    # EXCEPTION: Allow tier-1 sources OR press releases with definitive deal language through
                    if similarity > 0.30 and not (is_tier1 or (is_press_release and has_definitive_language)):
                        prec = best_match['precedent']
                        time_gap_min = (event_time - prec['pub_time']).total_seconds() / 60
                        shared = best_match['shared_entities']
                        
                        if os.getenv('LOG_MOMENTUM_DETAILS', '0') == '1':
                            match_info = f"[{match_type}]" if match_type == 'FINGERPRINT_EXACT' else f"{similarity:.1%}"
                            fingerprint_str = f"Fingerprint: {shared.get('fingerprint', 'N/A')}, " if 'fingerprint' in shared else ""
                            logger.info(
                                f"[Gate] {ticker} skipped (DUPLICATE: {match_info} match to "
                                f"{prec['source']} article {time_gap_min:.0f}min earlier - "
                                f"{fingerprint_str}"
                                f"Companies: {shared.get('companies', set())}, "
                                f"Amounts: {shared.get('amounts', set())}, "
                                f"Deal participants: {shared.get('deal_participants', set())})"
                            )
                        
                        return skip_event('DUPLICATE_OF_EARLIER_NEWS', {
                            'similarity': similarity,
                            'match_type': match_type,
                            'precedent_source': prec.get('source', 'unknown'),
                            'precedent_title': prec.get('title', ''),
                            'time_gap_minutes': time_gap_min,
                            'shared_entities': {k: list(v) if isinstance(v, set) else v for k, v in shared.items() if v}
                        })
        except Exception as e:
            # If precedence check fails, log but don't block trade
            logger.warning(f"[Gate] {ticker} precedence check failed: {e}")

        # RTH gating and min price filter
        if RTH_ONLY:
            ts_utc = event_time if event_time.tzinfo else event_time.replace(tzinfo=timezone.utc)
            if not _is_rth(ts_utc):
                return skip_event('NON_RTH')
        
        # Weekday-only trading (Monday-Friday)
        event_date = event_time.date() if hasattr(event_time, 'date') else event_time
        if hasattr(event_date, 'weekday') and event_date.weekday() >= 5:
            return skip_event('WEEKEND_EVENT')

        # Exchange filter: Only trade NYSE/NASDAQ, exclude OTC
        # XASE (NYSE American) conditionally allowed for high-quality clusters + trusted sources
        is_allowed, exchange = self._is_allowed_exchange(ticker)
        
        # Special handling for XASE: allow if cluster + source meet quality criteria
        if not is_allowed and exchange and exchange.upper() == 'XASE':
            cluster_str = event.get('cluster', '')
            source_name = event.get('source_name', '')
            
            # Check if event qualifies for XASE exception
            # Delistings and contract awards: cluster alone is sufficient (no source restriction)
            # Other clusters: require trusted source
            if cluster_str in {'delisting', 'contract_award_keyword'}:
                is_allowed = True
                if os.getenv('LOG_MOMENTUM_DETAILS', '0') == '1':
                    logger.info(f"[Gate] {ticker} XASE allowed (cluster={cluster_str}, source check waived)")
            elif cluster_str in XASE_ALLOWED_CLUSTERS and source_name in XASE_TRUSTED_SOURCES:
                is_allowed = True
                if os.getenv('LOG_MOMENTUM_DETAILS', '0') == '1':
                    logger.info(f"[Gate] {ticker} XASE allowed (cluster={cluster_str}, source={source_name})")
            else:
                if os.getenv('LOG_MOMENTUM_DETAILS', '0') == '1':
                    logger.info(f"[Gate] {ticker} XASE blocked (cluster={cluster_str} not in allowed, or source={source_name} not trusted)")
        
        if not is_allowed:
            if os.getenv('LOG_MOMENTUM_DETAILS', '0') == '1':
                logger.info(f"[Gate] {ticker} skipped (exchange: {exchange or 'UNKNOWN'}, not in allowed list)")
            return skip_event('EXCHANGE_NOT_ALLOWED', {'exchange': exchange or 'UNKNOWN'})

        # Tick-only momentum and exits (mandatory)
        ticks_df = self._get_ticks_for_event_day(ticker, event_time)
        if ticks_df is None or ticks_df.empty:
            return skip_event('NO_TICK_DATA')

        news_entry_delay = int(self.params.get('NEWS_ENTRY_DELAY', NEWS_ENTRY_DELAY))
        sustained_price_thr_pct = float(self.params.get('SUSTAINED_PRICE_THRESHOLD', DEFAULT_STRATEGY_PARAMS['SUSTAINED_PRICE_THRESHOLD'])) * 100.0
        sustained_volume_ratio = float(self.params.get('SUSTAINED_VOLUME_RATIO', SUSTAINED_VOLUME_RATIO))

        momentum: Optional[Dict] = check_momentum_ticks(
            ticker,
            event_time,
            ticks_df,
            direction,
            entry_delay_sec=news_entry_delay,
            sustained_price_thr=sustained_price_thr_pct,
            sustained_vol_ratio=sustained_volume_ratio,
            explosive_price_thr=EXPLOSIVE_PRICE_THRESHOLD,
            explosive_vol_ratio=EXPLOSIVE_VOLUME_RATIO,
        )
        if not momentum:
            return skip_event('MOMENTUM_GATE_FAILED')

        # Find entry index on per-second OHLCV
        ohlcv = self._get_ohlcv_for_event_day(ticker, event_time)
        if ohlcv is None or ohlcv.empty:
            if os.getenv('LOG_MOMENTUM_DETAILS', '0') == '1':
                logger.info(f"[Gate] {ticker} skipped (no OHLCV from tick cache for {event_time.date()})")
            return skip_event('NO_OHLCV_DATA')
        ohlcv['time_diff'] = abs((ohlcv['timestamp'] - momentum['event_timestamp']).dt.total_seconds())
        entry_idx = int(ohlcv['time_diff'].idxmin())

        # Min price check using entry price proxy
        if momentum['entry_price'] < MIN_PRICE:
            return skip_event('MIN_PRICE_FILTER', {'entry_price': momentum['entry_price']})

        # Calculate position size AFTER confirming we can place the trade
        entry_price = momentum['entry_price']
        min_stop_pct = float(self.params.get('MIN_STOP_PCT', MIN_STOP_PCT))
        take_profit_multiple = float(self.params.get('TAKE_PROFIT_R_MULTIPLE', TAKE_PROFIT_R_MULTIPLE))

        # Set stop loss based on direction
        if USE_ATR_STOP:
            # Compute ATR-like volatility from pre-entry window using tick-derived OHLCV
            ohlcv = self._get_ohlcv_for_event_day(ticker, event_time)
            if ohlcv is None or ohlcv.empty:
                return skip_event('NO_OHLCV_FOR_STOP')
            atr_like = _compute_atr_like(ohlcv, momentum['event_timestamp'], ATR_LOOKBACK_SECONDS)
            if atr_like is None or atr_like <= 0:
                stop_dist = min_stop_pct
            else:
                stop_dist = max(min_stop_pct, atr_like * ATR_MULTIPLIER)
            if direction == 'LONG':
                stop_loss_price = entry_price * (1 - stop_dist)
                take_profit_price = entry_price * (1 + stop_dist * take_profit_multiple)
            else:  # SHORT
                stop_loss_price = entry_price * (1 + stop_dist)
                take_profit_price = entry_price * (1 - stop_dist * take_profit_multiple)
        else:
            if direction == 'LONG':
                stop_loss_price = entry_price * (1 - abs(HARD_STOP_LOSS_LONG) / 100)
                take_profit_price = entry_price * (1 + abs(HARD_STOP_LOSS_LONG) / 100)
            else:  # SHORT
                stop_loss_price = entry_price * (1 + abs(HARD_STOP_LOSS_SHORT) / 100)
                take_profit_price = entry_price * (1 - abs(HARD_STOP_LOSS_SHORT) / 100)

        risk_per_share = max(1e-6, abs(entry_price - stop_loss_price))
        risk_amount = self.capital * (RISK_PER_TRADE_PCT / 100)
        
        # Apply max loss cap (NEW Oct 27, 2025)
        # Limits max loss per trade to prevent oversizing into volatile stocks
        max_loss_amount = self.capital * (MAX_LOSS_PER_TRADE_PCT / 100)
        risk_amount = min(risk_amount, max_loss_amount)
        
        # Check if stock price exceeds limit for non-SP500 stocks
        is_sp500 = ticker in SP500_TICKERS
        if entry_price > MAX_STOCK_PRICE_NON_SP500 and not is_sp500:
            if os.getenv('LOG_MOMENTUM_DETAILS', '0') == '1':
                logger.info(f"[Position] {ticker} skipped: price ${entry_price:.2f} > ${MAX_STOCK_PRICE_NON_SP500} and not in SP500")
            return skip_event('PRICE_TOO_HIGH_NON_SP500', {'entry_price': entry_price, 'is_sp500': is_sp500})
        
        # Position sizing based on price and SP500 status
        if entry_price > MAX_STOCK_PRICE_NON_SP500:
            # High-priced stocks (>$560): Only SP500 allowed with fractional shares
            shares = risk_amount / entry_price  # Fractional shares
            max_shares_by_capital = self.capital / entry_price
            shares = min(shares, max_shares_by_capital)
        else:
            # Lower-priced stocks (<=$560): Buy full shares up to risk_amount
            full_shares = int(risk_amount / entry_price)
            shares = full_shares
            
            # If SP500, add fractional share for remaining amount
            if is_sp500:
                remaining_amount = risk_amount - (full_shares * entry_price)
                fractional_part = remaining_amount / entry_price
                shares = full_shares + fractional_part
            
            # Cap by available capital
            max_shares_by_capital = self.capital / entry_price
            if is_sp500:
                shares = min(shares, max_shares_by_capital)
            else:
                shares = min(full_shares, int(max_shares_by_capital))
        
        if shares <= 0:
            if os.getenv('LOG_MOMENTUM_DETAILS', '0') == '1':
                logger.info(f"[Position] {ticker} cannot size position (capital={self.capital:.2f}, entry={entry_price:.2f})")
            return skip_event('POSITION_SIZING_ZERO', {'capital': self.capital, 'entry_price': entry_price})

        position_value = shares * entry_price
        
        # TRANSACTION COSTS: Deduct commission + slippage on entry
        entry_cost = position_value * (COMMISSION_BPS + SLIPPAGE_BPS) / 10_000
        
        # Deduct capital (position value + entry costs)
        self.capital -= (position_value + entry_cost)

        self.open_positions[ticker] = {
            'ticker': ticker,
            'shares': shares,
            'entry_price': entry_price,
            'entry_time': event_time,
            'entry_idx': entry_idx,
            'stop_loss_price': stop_loss_price,
            'take_profit_price': take_profit_price,
            'baseline_volume': momentum['baseline_volume'],
            'clusters': clusters,
            'pattern': momentum['pattern'],
            'direction': direction,
            'title': event.get('title', ''),
            'source': event.get('source', ''),
            'pub_time': event_time,
            # Phase 2 trailing stop tracking
            'highest_price': entry_price,  # Track for LONG trailing
            'lowest_price': entry_price,   # Track for SHORT trailing
            'initial_risk': abs(entry_price - stop_loss_price),  # For trailing distance calc
        }

        logger.info(f"[{direction} ENTRY] {ticker} x {shares} @ ${entry_price:.2f} | Pattern: {momentum['pattern']} | Clusters: {clusters}")
        self._log_event(
            event,
            'TRADE_ENTERED',
            None,
            {
                'entry_price': entry_price,
                'shares': shares,
                'pattern': momentum.get('pattern'),
                'momentum_price_change_pct': momentum.get('price_change_pct'),
                'momentum_volume_ratio': momentum.get('volume_ratio'),
            },
        )
        return True
    
    def check_exits(self, candles_cache: Dict[str, pd.DataFrame], current_event_time: Optional[datetime] = None):
        """Tick-based exits using per-second OHLCV with 10-minute max hold."""
        tickers_to_close = []
        max_hold_seconds = int(self.params.get('MAX_HOLD_SECONDS', MAX_HOLD_SECONDS))
        
        for ticker, pos in self.open_positions.items():
            entry_time = pos['entry_time']
            
            # CRITICAL: Force exit if max hold time exceeded
            if current_event_time:
                time_held = (current_event_time - entry_time).total_seconds()
                if time_held > max_hold_seconds:
                    # Try to get actual market price at exit time from tick data of current day
                    exit_time = entry_time + timedelta(seconds=max_hold_seconds)
                    current_day_ohlcv = self._get_ohlcv_for_event_day(ticker, current_event_time)
                    
                    exit_price = None
                    if current_day_ohlcv is not None and not current_day_ohlcv.empty:
                        # Find the closest tick to the exit time
                        exit_ticks = current_day_ohlcv[current_day_ohlcv['timestamp'] >= exit_time]
                        if not exit_ticks.empty:
                            exit_price = float(exit_ticks.iloc[0]['close'])
                    
                    # If no tick data available, skip this trade (invalid exit)
                    if exit_price is None:
                        logger.warning(f"[Force Exit SKIP] {ticker}: No tick data at exit time - skipping trade (invalid)")
                        # Return capital and close position without recording trade
                        self.capital += pos['shares'] * pos['entry_price']
                        tickers_to_close.append(ticker)
                        continue
                    
                    # Calculate P&L based on direction
                    direction = pos['direction']
                    if direction == 'LONG':
                        pnl = (exit_price - pos['entry_price']) * pos['shares']
                        pnl_pct = ((exit_price - pos['entry_price']) / pos['entry_price']) * 100
                    else:  # SHORT
                        pnl = (pos['entry_price'] - exit_price) * pos['shares']
                        pnl_pct = ((pos['entry_price'] - exit_price) / pos['entry_price']) * 100
                    
                    self.capital += pos['shares'] * exit_price
                    trade = {
                        'ticker': ticker,
                        'direction': direction,
                        'entry_time': entry_time,
                        'exit_time': exit_time,
                        'entry_price': pos['entry_price'],
                        'exit_price': exit_price,
                        'shares': pos['shares'],
                        'pnl': pnl,
                        'pnl_pct': pnl_pct,
                        'hold_seconds': max_hold_seconds,
                        'exit_reason': 'FORCED_TIME_STOP',
                        'clusters': pos['clusters'],
                        'pattern': pos['pattern'],
                        'title': pos.get('title', ''),
                        'source': pos.get('source', ''),
                        'headline_time': pos['pub_time'].astimezone(timezone.utc).isoformat() if isinstance(pos.get('pub_time'), datetime) else pos.get('pub_time'),
                    }
                    self.closed_trades.append(trade)
                    logger.info(f"[{direction} EXIT] {ticker} | P&L: ${pnl:+,.2f} ({pnl_pct:+.2f}%) | Hold: {max_hold_seconds}s | Reason: FORCED_TIME_STOP")
                    tickers_to_close.append(ticker)
                    continue
            
            # Build per-second OHLCV from cached ticks
            ohlcv = self._get_ohlcv_for_event_day(ticker, pos['entry_time'])
            if ohlcv is None or ohlcv.empty:
                continue
            entry_idx = pos['entry_idx']
            direction = pos['direction']
            
            # PHASE-BASED EXIT SYSTEM: Fixed (0-5min) → Trailing (5min+)
            # Phase 1 (0-5min): HARD_STOP > TAKE_PROFIT > TIME_STOP
            # Phase 2 (5min+): HARD_STOP > TRAILING_STOP > TIME_STOP
            max_hold_time = entry_time + timedelta(seconds=max_hold_seconds)
            phase_transition_time = entry_time + timedelta(seconds=PHASE_TRANSITION_SECONDS)
            
            for i in range(entry_idx + 1, len(ohlcv)):
                candle = ohlcv.iloc[i]
                time_held = (candle['timestamp'] - entry_time).total_seconds()
                
                # ALWAYS check HARD STOP first (both phases)
                if direction == 'LONG':
                    if candle['low'] <= pos['stop_loss_price']:
                        self.close_position(ticker, i, pos['stop_loss_price'], 'HARD_STOP', ohlcv)
                        tickers_to_close.append(ticker)
                        break
                else:  # SHORT
                    if candle['high'] >= pos['stop_loss_price']:
                        self.close_position(ticker, i, pos['stop_loss_price'], 'HARD_STOP', ohlcv)
                        tickers_to_close.append(ticker)
                        break
                
                # PHASE 1: Fixed Take-Profit (0-5 minutes)
                if time_held <= PHASE_TRANSITION_SECONDS:
                    if direction == 'LONG':
                        if 'take_profit_price' in pos and candle['high'] >= pos['take_profit_price']:
                            self.close_position(ticker, i, pos['take_profit_price'], 'PHASE1_TP_1R', ohlcv)
                            tickers_to_close.append(ticker)
                            break
                    else:  # SHORT
                        if 'take_profit_price' in pos and candle['low'] <= pos['take_profit_price']:
                            self.close_position(ticker, i, pos['take_profit_price'], 'PHASE1_TP_1R', ohlcv)
                            tickers_to_close.append(ticker)
                            break
                
                # PHASE 2: Trailing Stop (after 5 minutes)
                else:
                    # Initialize trailing stop on first candle after phase transition
                    if 'trailing_stop' not in pos:
                        if direction == 'LONG':
                            pos['trailing_stop'] = candle['close'] * (1 - TRAILING_STOP_PCT / 100)
                            pos['highest_price'] = candle['close']
                        else:  # SHORT
                            pos['trailing_stop'] = candle['close'] * (1 + TRAILING_STOP_PCT / 100)
                            pos['lowest_price'] = candle['close']
                    
                    # Update trailing stop
                    if direction == 'LONG':
                        # Track highest price and trail below it
                        pos['highest_price'] = max(pos['highest_price'], candle['high'])
                        new_stop = pos['highest_price'] * (1 - TRAILING_STOP_PCT / 100)
                        pos['trailing_stop'] = max(pos['trailing_stop'], new_stop)
                        
                        # Exit if price breaks trailing stop
                        if candle['low'] <= pos['trailing_stop']:
                            self.close_position(ticker, i, pos['trailing_stop'], 'PHASE2_TRAILING', ohlcv)
                            tickers_to_close.append(ticker)
                            break
                    else:  # SHORT
                        # Track lowest price and trail above it
                        pos['lowest_price'] = min(pos['lowest_price'], candle['low'])
                        new_stop = pos['lowest_price'] * (1 + TRAILING_STOP_PCT / 100)
                        pos['trailing_stop'] = min(pos['trailing_stop'], new_stop)
                        
                        # Exit if price breaks trailing stop
                        if candle['high'] >= pos['trailing_stop']:
                            self.close_position(ticker, i, pos['trailing_stop'], 'PHASE2_TRAILING', ohlcv)
                            tickers_to_close.append(ticker)
                            break
                
                # Time stop LAST: force exit at max hold time (both phases)
                if candle['timestamp'] >= max_hold_time:
                    self.close_position(ticker, i, candle['close'], 'TIME_STOP_10MIN', ohlcv)
                    tickers_to_close.append(ticker)
                    break
            
            if ticker in tickers_to_close:
                continue
            
            # Volume exhaustion removed - relying on ATR-based stops only
            # Exit logic now simplified to:
            # 1. ATR-based hard stop (checked above in candle loop)
            # 2. Take profit at 1R multiple (checked above)
            # 3. Time stop at MAX_HOLD_SECONDS (checked above)
        
        # Remove closed positions
        for ticker in tickers_to_close:
            if ticker in self.open_positions:
                del self.open_positions[ticker]
    
    def close_position(self, ticker: str, exit_idx: int, exit_price: float, exit_reason: str, ohlcv_df: pd.DataFrame):
        """Close a position and record trade (handles LONG and SHORT)"""
        pos = self.open_positions[ticker]
        
        # Calculate P&L (direction-aware)
        shares = pos['shares']
        entry_price = pos['entry_price']
        direction = pos['direction']
        
        if direction == 'LONG':
            # LONG: Profit when price goes up
            pnl_gross = (exit_price - entry_price) * shares
            pnl_pct_gross = ((exit_price - entry_price) / entry_price) * 100
        else:  # SHORT
            # SHORT: Profit when price goes down (inverted)
            pnl_gross = (entry_price - exit_price) * shares
            pnl_pct_gross = ((entry_price - exit_price) / entry_price) * 100
        
        # TRANSACTION COSTS (CRITICAL FOR HIGH-FREQUENCY NEWS TRADING)
        # Two legs: entry + exit, each incurs commission + slippage
        position_value = shares * entry_price
        entry_cost = position_value * (COMMISSION_BPS + SLIPPAGE_BPS) / 10_000  # Entry leg
        exit_cost = (shares * exit_price) * (COMMISSION_BPS + SLIPPAGE_BPS) / 10_000  # Exit leg
        total_cost = entry_cost + exit_cost
        
        # Net P&L after costs
        pnl = pnl_gross - total_cost
        pnl_pct = (pnl / position_value) * 100
        
        # Return capital
        self.capital += shares * exit_price - exit_cost  # Deduct exit costs from returned capital
        
        # Hold time
        exit_time = ohlcv_df.iloc[exit_idx]['timestamp']
        hold_duration = (exit_time - pos['entry_time']).total_seconds()
        
        # CRITICAL: Prevent negative hold times (data quality issue)
        if hold_duration < 0:
            logger.error(f"[BUG] {ticker}: Negative hold time detected! Entry: {pos['entry_time']}, Exit: {exit_time}")
            logger.error(f"       Forcing exit time to entry + MAX_HOLD_SECONDS")
            exit_time = pos['entry_time'] + timedelta(seconds=MAX_HOLD_SECONDS)
            hold_duration = MAX_HOLD_SECONDS
        
        # Record trade
        trade = {
            'ticker': ticker,
            'direction': direction,
            'entry_time': pos['entry_time'],
            'exit_time': exit_time,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'shares': shares,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'hold_seconds': hold_duration,
            'exit_reason': exit_reason,
            'clusters': pos['clusters'],
            'pattern': pos['pattern'],
            'title': pos.get('title', ''),
            'source': pos.get('source', ''),
            'headline_time': (
                pos['pub_time'].astimezone(timezone.utc).isoformat()
                if isinstance(pos.get('pub_time'), datetime)
                else pos.get('pub_time')
            ),
        }
        
        self.closed_trades.append(trade)
        self.equity_curve.append(self.capital)
        
        logger.info(f"[{direction} EXIT] {ticker} | P&L: ${pnl:+.2f} ({pnl_pct:+.2f}%) | Hold: {hold_duration:.0f}s | Reason: {exit_reason}")
    
    def get_performance_stats(self) -> Dict:
        """Calculate performance metrics"""
        if len(self.closed_trades) == 0:
            # Compute max drawdown from equity curve even if no trades
            equity_array = np.array(self.equity_curve) if len(self.equity_curve) > 0 else np.array([self.capital])
            running_max = np.maximum.accumulate(equity_array)
            drawdown = (equity_array - running_max) / np.maximum(running_max, 1)
            max_drawdown = float(np.min(drawdown) * 100) if len(drawdown) > 0 else 0.0
            return {
                'total_trades': 0,
                'win_rate': 0.0,
                'avg_return': 0.0,
                'total_pnl': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': max_drawdown,
                'final_capital': float(self.capital)
            }
        
        df = pd.DataFrame(self.closed_trades)
        
        # Basic stats
        total_trades = len(df)
        winners = df[df['pnl'] > 0]
        win_rate = (len(winners) / total_trades) * 100
        avg_return = df['pnl_pct'].mean()
        total_pnl = df['pnl'].sum()
        
        # Sharpe Ratio
        returns = df['pnl_pct'].values
        sharpe_ratio = (np.mean(returns) / np.std(returns)) * np.sqrt(252) if np.std(returns) > 0 else 0
        
        # Max Drawdown
        equity_array = np.array(self.equity_curve)
        running_max = np.maximum.accumulate(equity_array)
        drawdown = (equity_array - running_max) / running_max
        max_drawdown = np.min(drawdown) * 100
        
        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'avg_return': avg_return,
            'total_pnl': total_pnl,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'final_capital': self.capital
        }


# ============================================================================
# MAIN BACKTEST EXECUTION
# ============================================================================

def run_backtest(
    strategy_params: Optional[Dict[str, Any]] = None,
    suppress_logs: bool = False,
) -> Dict[str, float]:
    """Execute the backtest with optional strategy parameter overrides."""
    params = dict(strategy_params or {})
    original_level = logger.level
    if suppress_logs:
        logger.setLevel(logging.WARNING)

    try:
        logger.info("=" * 80)
        logger.info("NEW CLUSTER BACKTESTING ENGINE")
        logger.info("=" * 80)
        logger.info(f"Backtest Period: {BACKTEST_START_DATE.date()} to {BACKTEST_END_DATE.date()}")
        logger.info(f"Initial Capital: ${INITIAL_CAPITAL:,.2f}")
        logger.info(f"Risk Per Trade: {RISK_PER_TRADE_PCT}%")
        logger.info(f"Max Positions: {MAX_CONCURRENT_POSITIONS}")
        logger.info("=" * 80)

        diag_only = os.getenv('DIAG_ONLY_PRECEDENT_COUNT', '0') == '1'

        # Sanity check for Finnhub API key
        if not diag_only and (not FINNHUB_API_KEY or FINNHUB_API_KEY.strip() in ("", "your_finnhub_api_key_here")):
            logger.error("Finnhub API key is missing or placeholder. Set FINNHUB_API_KEY and rerun.")
            logger.error("In PowerShell: $env:FINNHUB_API_KEY=\"YOUR_REAL_KEY\"")
            raise RuntimeError("Missing FINNHUB_API_KEY")
        # Mandatory tick requirement
        if not diag_only and (finnhub is None or fh_fetch_ticks_for_date is None):
            logger.error("Tick data required: install finnhub-python and ensure orchestrator fetcher is importable.")
            logger.error("pip install finnhub-python")
            raise RuntimeError("Tick infrastructure unavailable")

        # Optionally shrink period for quick validation
        if FAST_MODE:
            fast_start = datetime.now() - timedelta(days=14)
            logger.info("FAST_BACKTEST=1 detected: limiting backtest to last 14 days")
        else:
            fast_start = BACKTEST_START_DATE

        # Load historical articles
        logger.info("\n[1/5] Loading historical articles from cache...")
        articles = load_stocknews_cache(fast_start, BACKTEST_END_DATE)

        if len(articles) == 0:
            logger.error("No articles found in cache. Exiting.")
            return _empty_stats()
        
        # Load VIX data for regime filtering
        logger.info("\n[1b/5] Loading VIX data for regime filtering...")
        vix_data = load_vix_data(fast_start, BACKTEST_END_DATE)

        # Optional diagnostics: list war_risk_sector matches for specific tickers
        diag_env = os.getenv('DIAG_WAR_TICKERS')
        if diag_env:
            tickers_wanted = {t.strip().upper() for t in diag_env.split(',') if t.strip()}
            if not tickers_wanted:
                tickers_wanted = {'AMD', 'NVDA'}
            logger.info(f"\n[DIAG] Listing war_risk_sector articles for: {', '.join(sorted(tickers_wanted))}")
            hits = list_cluster_matches(articles, 'war_risk_sector', tickers_wanted)
            if not hits:
                logger.info("[DIAG] No war_risk_sector matches found in date range for selected tickers.")
            else:
                logger.info(f"[DIAG] Found {len(hits)} articles:")
                for i, h in enumerate(hits, 1):
                    logger.info(f"  {i:02d}. {h['dt']} | {','.join(h['tickers'])} | {h['source']} | {h['title']}")
                    logger.info(f"      patterns: {h['matched_patterns']}")
                    if h.get('url'):
                        logger.info(f"      url: {h['url']}")
            return _empty_stats()

        # Filter and classify articles by new clusters
        logger.info("\n[2/5] Classifying articles by new event clusters...")
        events = []
        cooldown_state = {}
        filtered_generic = 0
        filtered_negative = 0
        filtered_precedent = 0

        # Pre-parse and sort by publication time to ensure earliest headlines set cooldown first
        def _parse_pub_time(pub_date: str) -> Optional[datetime]:
            """Parse pub_date supporting ISO8601 and RFC 2822.
            Avoid naive 'T' check since RFC day names like 'Thu' contain 'T'.
            """
            try:
                # Detect ISO strictly via YYYY-MM-DDT
                if re.search(r"\d{4}-\d{2}-\d{2}T", pub_date):
                    try:
                        dt = datetime.fromisoformat(pub_date.replace('Z', '+00:00'))
                    except Exception:
                        from email.utils import parsedate_to_datetime
                        dt = parsedate_to_datetime(pub_date)
                else:
                    from email.utils import parsedate_to_datetime
                    dt = parsedate_to_datetime(pub_date)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                else:
                    dt = dt.astimezone(timezone.utc)
                return dt
            except Exception:
                return None

        # Source priority: prefer press releases, then tier-1 wires, then others (used as a tiebreaker)
        PRESS_RELEASE_SOURCES = {
            'GlobeNewsWire', 'PR Newswire', 'Business Wire', 'GlobeNewswire', 'Accesswire', 'Newsfile'
        }
        TIER1_WIRES = {'Reuters', 'Bloomberg', 'Dow Jones', 'The Wall Street Journal'}

        def _source_rank(name: str) -> int:
            if name in PRESS_RELEASE_SOURCES:
                return 0
            if name in TIER1_WIRES:
                return 1
            return 2

        items: List[Tuple[datetime, int, Dict]] = []
        for article in articles:
            title = article.get('title', '')
            raw_tickers = article.get('tickers', [])
            if isinstance(raw_tickers, list):
                tickers = raw_tickers
            elif isinstance(raw_tickers, str):
                tickers = raw_tickers.split(',') if raw_tickers else []
            else:
                tickers = []
            pub_date = article.get('date', '')
            source = article.get('source_name', '')
            if not title or not tickers or not pub_date:
                continue
            dt = _parse_pub_time(pub_date)
            if not dt:
                continue
            items.append((dt, _source_rank(source), article))

        # Sort by time ascending, then source rank (press release first on ties)
        items.sort(key=lambda x: (x[0], x[1]))

        for pub_time, _rank, article in items:
            title = article.get('title', '')
            # Handle both string (comma-separated) and list formats for tickers
            raw_tickers = article.get('tickers', [])
            if isinstance(raw_tickers, list):
                tickers = raw_tickers
            elif isinstance(raw_tickers, str):
                tickers = raw_tickers.split(',') if raw_tickers else []
            else:
                tickers = []
            source = article.get('source_name', '')

            # Filter generic fluff (from news_trade_orchestrator_finnhub.py)
            if any(fluff_re.search(title) for fluff_re in GENERIC_FLUFF_RE):
                filtered_generic += 1
                continue

            # Check which clusters match (with positive/negative pattern logic)
            clusters_matched = []
            event_direction = None

            for cluster_name, pattern_dict in CLUSTER_PATTERNS.items():
                if cluster_name in SKIP_CLUSTERS:
                    continue
                # Check positive patterns
                positive_match = any(p.search(title) for p in pattern_dict['positive'])

                if positive_match:
                    # Check negative patterns (disqualifiers)
                    negative_match = any(p.search(title) for p in pattern_dict['negative'])

                    if not negative_match:
                        clusters_matched.append(cluster_name)
                        # Set direction from first matched cluster
                        if event_direction is None:
                            event_direction = pattern_dict['direction']
                    else:
                        filtered_negative += 1

            if not clusters_matched:
                continue

            # Check for duplicates
            ticker = tickers[0]  # Use first ticker
            if is_duplicate_event(ticker, title, pub_time, clusters_matched, cooldown_state):
                continue

            # PRECEDENT ANALYSIS: Check if this is follow-up coverage
            # (only if ENABLE_PRECEDENT_CHECK env var is set)
            if os.getenv('ENABLE_PRECEDENT_CHECK', '0') == '1':
                has_precedent, precedent_info = check_precedent_news(
                    ticker, pub_time, title, articles, similarity_threshold=0.5
                )
                
                if has_precedent and precedent_info:
                    best_match = precedent_info['best_match']
                    time_diff = best_match['time_diff_minutes']
                    similarity = best_match['similarity']
                    
                    # Skip if very similar news exists within last 2 hours (likely follow-up)
                    if similarity > 0.7 and time_diff < 120:
                        logger.debug(f"[Precedent] Skipping {ticker} - similar news {time_diff:.0f}min earlier "
                                   f"(similarity: {similarity:.2f})")
                        filtered_precedent += 1
                        continue

            # Valid event
            event = {
                'ticker': ticker,
                'title': title,
                'pub_time': pub_time,
                'source': source,
                'clusters': clusters_matched,
                'direction': event_direction  # 'LONG' or 'SHORT'
            }
            events.append(event)

        logger.info(f"Found {len(events)} unique trading events across {len(articles)} articles")
        logger.info(f"Filtered: {filtered_generic} generic fluff, {filtered_negative} negative patterns, {filtered_precedent} precedent follow-ups")

        # Optional: Early exit if only a precedent count is requested
        if os.getenv('DIAG_ONLY_PRECEDENT_COUNT', '0') == '1':
            logger.info("DIAG_ONLY_PRECEDENT_COUNT=1: returning after reporting precedent count.")
            return _empty_stats()

        # ============================================================================
        # FIRST-OF-FIRST DEDUPLICATION: Only trade on EARLIEST news per ticker+cluster+day
        # ============================================================================
        # Group events by (ticker, cluster, date) and keep ONLY the earliest
        first_events_map = {}  # Key: (ticker, cluster, date_str) -> earliest event
        
        for event in events:
            ticker = event['ticker']
            pub_time = event['pub_time']
            date_str = pub_time.strftime('%Y-%m-%d')
            
            for cluster in event['clusters']:
                key = (ticker, cluster, date_str)
                
                if key not in first_events_map:
                    # First event for this ticker+cluster+day
                    first_events_map[key] = event
                else:
                    # Check if this event is earlier
                    existing_time = first_events_map[key]['pub_time']
                    if pub_time < existing_time:
                        # This is earlier - replace it
                        logger.debug(f"[First-of-First] {ticker} {cluster} {date_str}: "
                                   f"Replacing {existing_time.strftime('%H:%M:%S')} with "
                                   f"{pub_time.strftime('%H:%M:%S')} (earlier)")
                        first_events_map[key] = event
                    else:
                        # Keep existing (it's earlier)
                        logger.debug(f"[First-of-First] {ticker} {cluster} {date_str}: "
                                   f"Skipping {pub_time.strftime('%H:%M:%S')} "
                                   f"(keeping {existing_time.strftime('%H:%M:%S')})")
        
        # Rebuild events list from first-of-first map
        original_count = len(events)
        events = list(first_events_map.values())
        duplicates_removed = original_count - len(events)
        
        logger.info(f"[First-of-First] Removed {duplicates_removed} same-day follow-up articles")
        logger.info(f"[First-of-First] Keeping {len(events)} earliest breaking news events")

        # Group events by cluster for analysis
        cluster_counts = defaultdict(int)
        for event in events:
            for cluster in event['clusters']:
                cluster_counts[cluster] += 1

        logger.info("\nEvent distribution by cluster:")
        for cluster, count in sorted(cluster_counts.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"  {cluster}: {count} events")

        # Build peers cache for sector-level events (only if API key is valid and enabled)
        logger.info("\n[3/6] Building peers cache for sector-level clusters...")
        peers_cache = {}
        if BUILD_PEERS:
            try:
                peers_cache = build_peers_cache(events)
            except Exception as e:
                logger.warning(f"Skipping peer cache build due to error: {e}")
                peers_cache = {}
        else:
            logger.info("BUILD_PEERS=0: Skipping peer cache build by default")

        # Expand sector events to include peer trades
        logger.info("\n[4/6] Expanding sector events to include peer trades...")
        expanded_events = []
        sector_expansions = 0

        for event in events:
            # WHITELIST ENFORCEMENT (Oct 27, 2025): Only trade approved clusters with proven positive Sharpe
            cluster = event['clusters'][0]  # Primary cluster
            if cluster not in APPROVED_CLUSTERS:
                logger.debug(f"Skipping {event['ticker']} ({cluster}) - cluster not in APPROVED_CLUSTERS (only trading: {APPROVED_CLUSTERS})")
                continue
            
            # Also skip if in legacy SKIP_CLUSTERS (defensive check)
            if cluster in SKIP_CLUSTERS:
                logger.debug(f"Skipping {event['ticker']} ({cluster}) - cluster in SKIP_CLUSTERS")
                continue
            
            # Always add the primary event
            expanded_events.append(event)

            # Check if this cluster requires peer trading
            cluster_config = CLUSTER_PATTERNS.get(cluster, {})

            if cluster_config.get('requires_peers', False) and peers_cache:
                # Get curated peers from cache
                curated_peers = get_curated_peers(event['ticker'], cluster, peers_cache)

                if curated_peers:
                    logger.debug(f"Expanding {event['ticker']} ({cluster}): {len(curated_peers)} peers")

                    # Create identical events for each peer
                    for peer_ticker in curated_peers:
                        peer_event = {
                            'ticker': peer_ticker,
                            'title': event['title'],  # Same news
                            'pub_time': event['pub_time'],  # Same time
                            'source': event['source'],
                            'clusters': event['clusters'],  # Same clusters
                            'direction': event['direction']  # Same direction
                        }
                        expanded_events.append(peer_event)
                        sector_expansions += 1

        logger.info(f"Expanded {len(events)} events → {len(expanded_events)} events (+{sector_expansions} peer trades)")

        # Tick-only backtest (lazy tick fetch per event-day; disk cache ensures reuse)
        logger.info("\n[5/6] Running backtest (tick-only)...")
        # Compute unique (ticker,event-day) pairs for visibility
        unique_pairs = []
        seen = set()
        for ev in expanded_events:
            dt_utc = ev['pub_time'].astimezone(timezone.utc) if ev['pub_time'].tzinfo else ev['pub_time'].replace(tzinfo=timezone.utc)
            date_str = dt_utc.strftime('%Y-%m-%d')
            key = (ev['ticker'].upper(), date_str)
            if key not in seen:
                seen.add(key)
                unique_pairs.append(key)
        logger.info(f"Tick days to load (unique): {len(unique_pairs)}")
        # Show quick cache hit/miss summary upfront
        if fh_get_tick_cache_path is not None:
            hits = 0
            for tkr, d in unique_pairs[:50]:  # cheap peek for first 50
                p = fh_get_tick_cache_path(tkr, d)
                if p.exists():
                    hits += 1
            logger.info(f"Initial tick cache hits (first 50 days): {hits}")

        engine = BacktestEngine(INITIAL_CAPITAL, params)
        
        # Log Symbol Lookup status (MIC codes + security types from Finnhub)
        logger.info(f"\n[Symbol Lookup] Loaded {len(engine.symbol_lookup)} US symbols with MIC codes and types")
        test_tickers = ['AAPL', 'AAAU', 'BBEU', 'SPY', 'QQQ', 'WTKWY']
        etp_count = sum(1 for ticker in test_tickers if ticker in engine.symbol_lookup and engine.symbol_lookup[ticker].get('type') == 'ETP')
        logger.info(f"[Symbol Lookup] Test tickers - ETPs detected: {etp_count}/{len(test_tickers)}")
        for ticker in test_tickers:
            if ticker in engine.symbol_lookup:
                info = engine.symbol_lookup[ticker]
                logger.info(f"  {ticker}: MIC={info.get('mic')}, Type={info.get('type')}")
            else:
                logger.info(f"  {ticker}: Not found in symbol_lookup")

        # Sort events by time (use expanded_events instead of events)
        expanded_events.sort(key=lambda x: x['pub_time'])

        total_events = len(expanded_events)
        vix_blocked_count = 0
        
        for idx, event in enumerate(expanded_events, start=1):
            # Event progress header
            dt_utc = event['pub_time'].astimezone(timezone.utc) if event['pub_time'].tzinfo else event['pub_time'].replace(tzinfo=timezone.utc)
            logger.info(f"[Event {idx}/{total_events}] {event['ticker']} {event['direction']} @ {dt_utc.strftime('%Y-%m-%d %H:%M:%S')} UTC | Clusters: {','.join(event['clusters'])}")

            # VIX REGIME FILTER: Check if market panic (VIX > 30)
            current_vix = None
            if vix_data is not None:
                event_date = event['pub_time'].date()
                try:
                    # Find VIX value for this date
                    vix_date = pd.Timestamp(event_date)
                    if vix_date in vix_data.index:
                        current_vix = float(vix_data.loc[vix_date])
                    elif len(vix_data) > 0:
                        # Use most recent VIX if exact date not found
                        current_vix = float(vix_data.asof(vix_date))
                    
                    if current_vix and current_vix > VIX_REGIME_THRESHOLD:
                        logger.info(f"  ⚠️  VIX FILTER BLOCKED: VIX={current_vix:.1f} > {VIX_REGIME_THRESHOLD} (market panic)")
                        vix_blocked_count += 1
                        continue  # Skip this event
                except Exception as e_vix:
                    pass  # Silently continue if VIX lookup fails

            # Process new event
            engine.process_event(event, {})

            # Check exits for open positions (will log exits) - pass current event time for forced exits
            engine.check_exits({}, current_event_time=event['pub_time'])

        # Force close any remaining open positions at the end of backtest
        for ticker in list(engine.open_positions.keys()):
            pos = engine.open_positions[ticker]
            ohlcv_df = engine._get_ohlcv_for_event_day(ticker, pos['entry_time'])
            
            # If we can't get tick data or position already exceeded max hold, close at entry price
            if ohlcv_df is None or ohlcv_df.empty:
                logger.warning(f"[END_OF_BACKTEST] {ticker}: No tick data available, closing at entry price")
                exit_price = pos['entry_price']
                exit_time = pos['entry_time'] + timedelta(seconds=MAX_HOLD_SECONDS)
                engine.capital += pos['shares'] * exit_price
                trade = {
                    'ticker': ticker,
                    'direction': pos['direction'],
                    'entry_time': pos['entry_time'],
                    'exit_time': exit_time,
                    'entry_price': pos['entry_price'],
                    'exit_price': exit_price,
                    'shares': pos['shares'],
                    'pnl': 0.0,
                    'pnl_pct': 0.0,
                    'hold_seconds': MAX_HOLD_SECONDS,
                    'exit_reason': 'END_OF_BACKTEST',
                    'clusters': pos['clusters'],
                    'pattern': pos['pattern'],
                    'title': pos.get('title', ''),
                    'source': pos.get('source', ''),
                    'headline_time': pos['pub_time'].astimezone(timezone.utc).isoformat() if isinstance(pos.get('pub_time'), datetime) else pos.get('pub_time'),
                }
                engine.closed_trades.append(trade)
                continue
            
            exit_idx = len(ohlcv_df) - 1
            exit_price = float(ohlcv_df.iloc[exit_idx]['close'])
            engine.close_position(ticker, exit_idx, exit_price, 'END_OF_BACKTEST', ohlcv_df)

        # Calculate and display results
        logger.info("\n[7/7] Calculating performance metrics...")
        stats = engine.get_performance_stats()

        logger.info("\n" + "=" * 80)
        logger.info("BACKTEST RESULTS")
        logger.info("=" * 80)
        if vix_blocked_count > 0:
            logger.info(f"VIX Filter Blocked: {vix_blocked_count} events (VIX >{VIX_REGIME_THRESHOLD})")
        logger.info(f"Total Trades: {stats['total_trades']}")
        logger.info(f"Win Rate: {stats['win_rate']:.1f}%")
        logger.info(f"Average Return: {stats['avg_return']:+.2f}%")
        logger.info(f"Total P&L: ${stats['total_pnl']:+,.2f}")
        logger.info(f"Final Capital: ${stats['final_capital']:,.2f}")
        logger.info(f"Sharpe Ratio: {stats['sharpe_ratio']:.2f}")
        logger.info(f"Max Drawdown: {stats['max_drawdown']:.2f}%")
        logger.info("=" * 80)

        # Performance by cluster
        if len(engine.closed_trades) > 0:
            df = pd.DataFrame(engine.closed_trades)

            # Overall LONG vs SHORT performance
            logger.info("\nPERFORMANCE BY DIRECTION:")
            logger.info("-" * 80)

            for direction in ['LONG', 'SHORT']:
                dir_trades = df[df['direction'] == direction]
                if len(dir_trades) == 0:
                    continue

                dir_win_rate = (len(dir_trades[dir_trades['pnl'] > 0]) / len(dir_trades)) * 100
                dir_avg_return = dir_trades['pnl_pct'].mean()
                dir_total_pnl = dir_trades['pnl'].sum()
                dir_sharpe = (np.mean(dir_trades['pnl_pct']) / np.std(dir_trades['pnl_pct'])) * np.sqrt(252) if np.std(dir_trades['pnl_pct']) > 0 else 0

                logger.info(f"{direction}:")
                logger.info(f"  Trades: {len(dir_trades)}")
                logger.info(f"  Win Rate: {dir_win_rate:.1f}%")
                logger.info(f"  Avg Return: {dir_avg_return:+.2f}%")
                logger.info(f"  Total P&L: ${dir_total_pnl:+,.2f}")
                logger.info(f"  Sharpe Ratio: {dir_sharpe:.2f}")
                logger.info("")

            logger.info("\nPERFORMANCE BY CLUSTER:")
            logger.info("-" * 80)

            for cluster in CLUSTER_PATTERNS.keys():
                cluster_trades = df[df['clusters'].apply(lambda x: cluster in x)]
                if len(cluster_trades) == 0:
                    continue

                cluster_direction = CLUSTER_PATTERNS[cluster]['direction']
                cluster_win_rate = (len(cluster_trades[cluster_trades['pnl'] > 0]) / len(cluster_trades)) * 100
                cluster_avg_return = cluster_trades['pnl_pct'].mean()
                cluster_sharpe = (np.mean(cluster_trades['pnl_pct']) / np.std(cluster_trades['pnl_pct'])) * np.sqrt(252) if np.std(cluster_trades['pnl_pct']) > 0 else 0

                logger.info(f"{cluster} [{cluster_direction}]:")
                logger.info(f"  Trades: {len(cluster_trades)}")
                logger.info(f"  Win Rate: {cluster_win_rate:.1f}%")
                logger.info(f"  Avg Return: {cluster_avg_return:+.2f}%")
                logger.info(f"  Sharpe Ratio: {cluster_sharpe:.2f}")
                logger.info(f"  Status: {'[INTEGRATE]' if cluster_sharpe > 1.5 else '[SKIP]'}")
                logger.info("")

            # PERFORMANCE BY EXIT REASON
            logger.info("\nPERFORMANCE BY EXIT REASON:")
            logger.info("-" * 80)
            
            exit_reasons = df['exit_reason'].value_counts()
            for reason in exit_reasons.index:
                reason_trades = df[df['exit_reason'] == reason]
                reason_win_rate = (len(reason_trades[reason_trades['pnl'] > 0]) / len(reason_trades)) * 100
                reason_avg_return = reason_trades['pnl_pct'].mean()
                reason_avg_hold = reason_trades['hold_seconds'].mean()
                reason_total_pnl = reason_trades['pnl'].sum()
                
                logger.info(f"{reason}:")
                logger.info(f"  Trades: {len(reason_trades)}")
                logger.info(f"  Win Rate: {reason_win_rate:.1f}%")
                logger.info(f"  Avg Return: {reason_avg_return:+.2f}%")
                logger.info(f"  Avg Hold: {reason_avg_hold:.0f}s")
                logger.info(f"  Total P&L: ${reason_total_pnl:+,.2f}")
                logger.info("")

            # PERFORMANCE BY CLUSTER + EXIT REASON COMBINATION
            logger.info("\nPERFORMANCE BY CLUSTER + EXIT REASON:")
            logger.info("-" * 80)
            
            for cluster in CLUSTER_PATTERNS.keys():
                cluster_trades = df[df['clusters'].apply(lambda x: cluster in x)]
                if len(cluster_trades) == 0:
                    continue
                
                logger.info(f"\n{cluster}:")
                for reason in cluster_trades['exit_reason'].unique():
                    combo_trades = cluster_trades[cluster_trades['exit_reason'] == reason]
                    combo_win_rate = (len(combo_trades[combo_trades['pnl'] > 0]) / len(combo_trades)) * 100
                    combo_avg_return = combo_trades['pnl_pct'].mean()
                    combo_avg_hold = combo_trades['hold_seconds'].mean()
                    
                    logger.info(f"  {reason}: {len(combo_trades)} trades | WR: {combo_win_rate:.1f}% | Avg: {combo_avg_return:+.2f}% | Hold: {combo_avg_hold:.0f}s")
                logger.info("")

        # Export results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = f"backtest_results_{timestamp}.csv"
        events_file = f"backtest_events_{timestamp}.csv"

        if len(engine.closed_trades) > 0 and not suppress_logs:
            df = pd.DataFrame(engine.closed_trades)
            df.to_csv(results_file, index=False)
            logger.info(f"\nResults exported to: {results_file}")

        if engine.event_log and not suppress_logs:
            events_df = pd.DataFrame(engine.event_log)
            events_df.to_csv(events_file, index=False)
            logger.info(f"Event log exported to: {events_file}")

        # Benchmark Comparison (Buy & Hold ETFs)
        if not suppress_logs:
            logger.info("\n" + "=" * 80)
            logger.info("BENCHMARK COMPARISON: BUY & HOLD")
            logger.info("=" * 80)
            
            backtest_days = (BACKTEST_END_DATE - fast_start).days
            strategy_return_pct = ((stats['final_capital'] - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100
            strategy_annual_return = (strategy_return_pct / backtest_days) * 365
            
            # Benchmark returns (approximate historical averages for similar periods)
            # Note: These are estimates - actual returns depend on exact backtest period
            benchmarks = {
                'QQQ (Nasdaq-100)': {
                    'annual_return': 15.0,  # Historical average ~15-20%
                    'sharpe': 0.9,
                    'description': 'Tech-heavy index'
                },
                'VOO (S&P 500)': {
                    'annual_return': 10.5,  # Historical average ~10-12%
                    'sharpe': 0.8,
                    'description': 'Broad market index'
                },
                'VUG (Growth)': {
                    'annual_return': 13.5,  # Historical average ~12-15%
                    'sharpe': 0.85,
                    'description': 'Large-cap growth'
                },
                'FTEC (Tech Sector)': {
                    'annual_return': 16.0,  # Historical average ~15-18%
                    'sharpe': 0.9,
                    'description': 'Technology sector'
                }
            }
            
            logger.info(f"\nStrategy Performance:")
            logger.info(f"  Period: {backtest_days} days ({backtest_days/365:.2f} years)")
            logger.info(f"  Total Return: {strategy_return_pct:+.2f}%")
            logger.info(f"  Annualized: {strategy_annual_return:+.2f}%")
            logger.info(f"  Sharpe Ratio: {stats['sharpe_ratio']:.2f}")
            logger.info(f"  Max Drawdown: {stats['max_drawdown']:.2f}%")
            
            logger.info(f"\nBenchmark Comparison (Annualized):")
            logger.info(f"{'Ticker':<20} {'Return':<12} {'Sharpe':<10} {'vs Strategy':<15} {'Description'}")
            logger.info("-" * 80)
            
            for ticker, data in benchmarks.items():
                benchmark_return = (data['annual_return'] / 365) * backtest_days
                outperformance = strategy_return_pct - benchmark_return
                outperformance_annual = strategy_annual_return - data['annual_return']
                outperformance_str = f"{outperformance_annual:+.1f}%"
                
                logger.info(f"{ticker:<20} {data['annual_return']:>6.1f}%/yr   "
                          f"Sharpe {data['sharpe']:.2f}   {outperformance_str:<15} {data['description']}")
            
            logger.info(f"\n{'='*80}")
            if strategy_annual_return > 15.0:
                logger.info("✅ Strategy OUTPERFORMS all benchmarks!")
            elif strategy_annual_return > 10.5:
                logger.info("✅ Strategy outperforms S&P 500 (VOO)")
            else:
                logger.info("⚠️  Strategy underperforms - consider adjustments")
            logger.info(f"{'='*80}")

        logger.info("\n" + "=" * 80)
        logger.info("BACKTEST COMPLETE")
        logger.info("=" * 80)

        return stats
    finally:
        if suppress_logs:
            logger.setLevel(original_level)


def optimize_strategy(trial) -> float:  # type: ignore
    """Optuna objective: maximize Sharpe Ratio."""
    # Best known parameters (Sharpe 4.0746) as starting point
    best_params = {
        'MIN_STOP_PCT': 0.003047224600379318,
        'TAKE_PROFIT_R_MULTIPLE': 2.392108936672514,
        'NEWS_ENTRY_DELAY': 70,
        'SUSTAINED_PRICE_THRESHOLD': 0.001438799155571445,
        'SUSTAINED_VOLUME_RATIO': 4.4875330923394205,
        'MAX_HOLD_SECONDS': 660,
    }
    
    # Sample around best known values (±30% range)
    # Calculate min/max and round to step size to avoid Optuna warnings
    news_delay_min = max(30, int(best_params['NEWS_ENTRY_DELAY'] * 0.7))
    news_delay_max = min(120, int(best_params['NEWS_ENTRY_DELAY'] * 1.3))
    # Round min down to nearest 10, max up to nearest 10
    news_delay_min = (news_delay_min // 10) * 10
    news_delay_max = ((news_delay_max + 9) // 10) * 10
    
    hold_secs_min = max(300, int(best_params['MAX_HOLD_SECONDS'] * 0.7))
    hold_secs_max = min(900, int(best_params['MAX_HOLD_SECONDS'] * 1.3))
    # Round min down to nearest 60, max up to nearest 60
    hold_secs_min = (hold_secs_min // 60) * 60
    hold_secs_max = ((hold_secs_max + 59) // 60) * 60
    
    trial_params = {
        'MIN_STOP_PCT': trial.suggest_float('MIN_STOP_PCT', 
            0.004,   # 0.4% minimum stop
            0.008),  # 0.8% maximum stop (tight range around optimal 0.57%)
        'TAKE_PROFIT_R_MULTIPLE': trial.suggest_float('TAKE_PROFIT_R_MULTIPLE', 
            max(0.5, best_params['TAKE_PROFIT_R_MULTIPLE'] * 0.7), 
            min(3.5, best_params['TAKE_PROFIT_R_MULTIPLE'] * 1.3)),
        'NEWS_ENTRY_DELAY': trial.suggest_int('NEWS_ENTRY_DELAY', 
            news_delay_min, 
            news_delay_max, 
            step=10),
        'SUSTAINED_PRICE_THRESHOLD': trial.suggest_float('SUSTAINED_PRICE_THRESHOLD', 
            best_params['SUSTAINED_PRICE_THRESHOLD'] * 0.7, 
            best_params['SUSTAINED_PRICE_THRESHOLD'] * 1.3),
        'SUSTAINED_VOLUME_RATIO': trial.suggest_float('SUSTAINED_VOLUME_RATIO', 
            best_params['SUSTAINED_VOLUME_RATIO'] * 0.7, 
            best_params['SUSTAINED_VOLUME_RATIO'] * 1.3),
        'MAX_HOLD_SECONDS': trial.suggest_int('MAX_HOLD_SECONDS', 
            hold_secs_min, 
            hold_secs_max, 
            step=60),
        # Phase-based exit parameters (new - let Optuna optimize)
        'PHASE_TRANSITION_SECONDS': trial.suggest_int('PHASE_TRANSITION_SECONDS', 
            180,  # 3 minutes min
            420,  # 7 minutes max
            step=60),
        'TRAILING_STOP_PCT': trial.suggest_float('TRAILING_STOP_PCT', 
            0.5,   # 0.5% tight trail
            1.5),  # 1.5% loose trail
    }

    try:
        stats = run_backtest(trial_params, suppress_logs=True)
    except Exception as exc:  # pragma: no cover - Optuna will record failure
        trial.set_user_attr('failure_reason', str(exc))
        return -1.0

    sharpe = stats.get('sharpe_ratio', 0.0)
    if not np.isfinite(sharpe):
        sharpe = -1.0
    return float(sharpe)


if __name__ == "__main__":
    # Import Monte Carlo validator
    try:
        from monte_carlo_validator import MonteCarloValidator
        mc_validator = MonteCarloValidator(n_simulations=1000, confidence_level=0.95)
        has_mc = True
    except ImportError:
        logger.warning("monte_carlo_validator.py not found - Monte Carlo validation disabled")
        has_mc = False
    
    # Parse date overrides from environment BEFORE running backtest
    start_date_str = os.getenv('BACKTEST_START_DATE')
    end_date_str = os.getenv('BACKTEST_END_DATE')
    
    # Check if we should run IS/OOS workflow
    run_is_oos = os.getenv('RUN_IS_OOS', '0').strip() == '1'
    
    if start_date_str and end_date_str:
        try:
            BACKTEST_START_DATE = datetime.strptime(start_date_str, '%Y-%m-%d')
            BACKTEST_END_DATE = datetime.strptime(end_date_str, '%Y-%m-%d')
            logger.info(f"Using date range from environment: {BACKTEST_START_DATE.date()} to {BACKTEST_END_DATE.date()}")
        except ValueError:
            logger.error(f"Invalid date format for BACKTEST_START_DATE/END_DATE. Use YYYY-MM-DD.")
    
    # Determine run mode
    n_trials_str = os.getenv('OPTUNA_TRIALS', '20').strip()  # Default 20 trials for quick optimization
    skip_optuna = os.getenv('SKIP_OPTUNA', '0').strip() == '1'

    def _get_env_float(name: str):
        v = os.getenv(name)
        if v is None or v == "":
            return None
        try:
            return float(v)
        except Exception:
            logger.warning(f"Invalid float for {name}='{v}', ignoring override")
            return None

    def _get_env_int(name: str):
        v = os.getenv(name)
        if v is None or v == "":
            return None
        try:
            return int(v)
        except Exception:
            logger.warning(f"Invalid int for {name}='{v}', ignoring override")
            return None

    # Single backtest mode when OPTUNA_TRIALS=1 or SKIP_OPTUNA=1
    if skip_optuna or n_trials_str == '1':
        overrides: Dict[str, Any] = {}
        # Collect strategy overrides from environment if provided
        ms = _get_env_float('MIN_STOP_PCT')
        if ms is not None:
            overrides['MIN_STOP_PCT'] = ms
        tp = _get_env_float('TAKE_PROFIT_R_MULTIPLE')
        if tp is not None:
            overrides['TAKE_PROFIT_R_MULTIPLE'] = tp
        nd = _get_env_int('NEWS_ENTRY_DELAY')
        if nd is not None:
            overrides['NEWS_ENTRY_DELAY'] = nd
        spt = _get_env_float('SUSTAINED_PRICE_THRESHOLD')
        if spt is not None:
            overrides['SUSTAINED_PRICE_THRESHOLD'] = spt
        svr = _get_env_float('SUSTAINED_VOLUME_RATIO')
        if svr is not None:
            overrides['SUSTAINED_VOLUME_RATIO'] = svr
        # VOLUME_EXHAUSTION_THRESHOLD removed - no longer used in exit logic
        mhs = _get_env_int('MAX_HOLD_SECONDS')
        if mhs is not None:
            overrides['MAX_HOLD_SECONDS'] = mhs

        logger.info("\nRUN MODE: Single backtest (no Optuna)")
        if overrides:
            logger.info("Strategy parameter overrides from environment:")
            for k, v in overrides.items():
                logger.info(f"  {k}: {v}")
        else:
            logger.info("No strategy overrides provided via environment; using script defaults")

        # Execute one full backtest with logging enabled
        run_backtest(strategy_params=overrides, suppress_logs=False)
    else:
        # Optuna optimization mode
        n_trials = int(n_trials_str)
        study = optuna.create_study(direction='maximize', study_name='NewsStrategyOptimization')
        study.optimize(optimize_strategy, n_trials=n_trials)

        logger.info("\n" + "="*80)
        logger.info("OPTUNA OPTIMIZATION COMPLETE")
        logger.info("="*80)
        logger.info(f"\nTotal Trials: {n_trials}")
        logger.info(f"Best Trial: #{study.best_trial.number}")
        logger.info(f"Best Sharpe Ratio: {study.best_value:.4f}")
        
        # Re-run best trial for comprehensive metrics
        logger.info("\n" + "="*80)
        logger.info("RE-RUNNING BEST TRIAL FOR FULL METRICS")
        logger.info("="*80)
        final_stats = run_backtest(strategy_params=study.best_params, suppress_logs=False)
        
        # Print comprehensive summary
        logger.info("\n" + "="*80)
        logger.info("FINAL RESULTS WITH BEST PARAMETERS")
        logger.info("="*80)
        logger.info(f"\nTotal Trades: {final_stats.get('total_trades', 0)}")
        logger.info(f"Win Rate: {final_stats.get('win_rate', 0):.2f}%")
        logger.info(f"Avg Return: {final_stats.get('avg_return', 0):.2f}%")
        logger.info(f"Total P&L: ${final_stats.get('total_pnl', 0):,.2f}")
        logger.info(f"Sharpe Ratio: {final_stats.get('sharpe_ratio', 0):.4f}")
        logger.info(f"Max Drawdown: {final_stats.get('max_drawdown', 0):.2f}%")
        logger.info(f"Final Capital: ${final_stats.get('final_capital', INITIAL_CAPITAL):,.2f}")
        
        # Cluster breakdown if available
        if 'cluster_stats' in final_stats and final_stats['cluster_stats']:
            logger.info("\n" + "="*80)
            logger.info("PERFORMANCE BY CLUSTER")
            logger.info("="*80)
            for cluster, stats in sorted(final_stats['cluster_stats'].items(), 
                                        key=lambda x: x[1].get('trades', 0), 
                                        reverse=True):
                trades = stats.get('trades', 0)
                if trades > 0:
                    wins = stats.get('wins', 0)
                    wr = (wins / trades * 100) if trades > 0 else 0
                    avg_r = stats.get('avg_return', 0)
                    total_pnl = stats.get('total_pnl', 0)
                    logger.info(f"\n{cluster}:")
                    logger.info(f"  Trades: {trades} | Wins: {wins} | WR: {wr:.1f}%")
                    logger.info(f"  Avg Return: {avg_r:.2f}% | Total P&L: ${total_pnl:,.2f}")
        
        # Exit reason breakdown
        if 'exit_reasons' in final_stats and final_stats['exit_reasons']:
            logger.info("\n" + "="*80)
            logger.info("EXIT REASON BREAKDOWN")
            logger.info("="*80)
            total_exits = sum(final_stats['exit_reasons'].values())
            for reason, count in sorted(final_stats['exit_reasons'].items(), 
                                       key=lambda x: x[1], 
                                       reverse=True):
                pct = (count / total_exits * 100) if total_exits > 0 else 0
                logger.info(f"  {reason}: {count} ({pct:.1f}%)")
        
        # PRINT BEST PARAMETERS AGAIN AT THE END (so they don't get lost)
        logger.info("\n" + "="*80)
        logger.info("BEST PARAMETERS (COPY THESE)")
        logger.info("="*80)
        for key, value in study.best_params.items():
            if isinstance(value, float):
                logger.info(f"  {key}: {value:.6f}")
            else:
                logger.info(f"  {key}: {value}")
        
        logger.info("\n" + "="*80)
        logger.info("ENVIRONMENT VARIABLE EXPORTS")
        logger.info("="*80)
        for key, value in study.best_params.items():
            if isinstance(value, float):
                logger.info(f'$env:{key}="{value:.6f}"')
            else:
                logger.info(f'$env:{key}="{value}"')
        logger.info("="*80 + "\n")
        
        # ============================================================
        # MONTE CARLO VALIDATION + IS/OOS TESTING
        # ============================================================
        if has_mc and run_is_oos and 'closed_trades' in final_stats and final_stats['closed_trades']:
            logger.info("\n" + "="*80)
            logger.info("MONTE CARLO VALIDATION (Bootstrap 1000x)")
            logger.info("="*80)
            
            # Extract trades and equity curve for bootstrap
            trades = final_stats['closed_trades']
            
            # Create simple equity curve from trade P&Ls
            equity_values = [INITIAL_CAPITAL]
            for trade in trades:
                equity_values.append(equity_values[-1] + trade.get('pnl', 0))
            equity_curve = pd.Series(equity_values)
            
            if len(trades) < 10:
                logger.warning(f"⚠️  Only {len(trades)} trades - Monte Carlo may be unreliable")
            
            # Run Monte Carlo bootstrap
            logger.info(f"\nBootstrapping {len(trades)} trades with {mc_validator.n_simulations} simulations...")
            mc_results = mc_validator.bootstrap_trades(trades, equity_curve, INITIAL_CAPITAL)
            
            # Display MC results
            logger.info("\n" + "="*70)
            logger.info("MONTE CARLO VALIDATION REPORT")
            logger.info("="*70)
            logger.info(f"Simulations: {mc_validator.n_simulations:,}")
            logger.info(f"Trades: {len(trades)}")
            logger.info(f"Confidence Level: {mc_validator.confidence_level*100:.0f}%")
            logger.info("")
            logger.info(f"{'Metric':<30} {'Mean':>12} {'CI Lower':>12} {'CI Upper':>12}")
            logger.info("-"*70)
            
            # Extract key metrics from MC results
            for metric_name in ['sharpe', 'total_return', 'win_rate', 'max_dd']:
                if metric_name in mc_results:
                    data = mc_results[metric_name]
                    mean = data['mean']
                    ci_low = data['ci_lower']
                    ci_high = data['ci_upper']
                    
                    if 'return' in metric_name or 'dd' in metric_name or 'rate' in metric_name:
                        logger.info(f"{metric_name.replace('_', ' ').title():<30} {mean:>11.2f}% {ci_low:>11.2f}% {ci_high:>11.2f}%")
                    else:
                        logger.info(f"{metric_name.replace('_', ' ').title():<30} {mean:>12.2f} {ci_low:>12.2f} {ci_high:>12.2f}")
            
            logger.info("")
            logger.info("="*70)
            
            # Check if strategy passes MC validation
            sharpe_ci_lower = mc_results.get('sharpe', {}).get('ci_lower', 0)
            sharpe_sig = mc_results.get('sharpe', {}).get('significant', False)
            
            if sharpe_ci_lower >= MC_SHARPE_THRESHOLD or sharpe_sig:
                logger.info(f"✅ PASSED MC VALIDATION: Sharpe CI lower={sharpe_ci_lower:.2f} >= {MC_SHARPE_THRESHOLD}")
                
                # ============================================================
                # OUT-OF-SAMPLE TESTING
                # ============================================================
                logger.info("\n" + "="*80)
                logger.info("OUT-OF-SAMPLE TESTING (2025 WALK-FORWARD)")
                logger.info("="*80)
                
                # Save current backtest dates
                original_start = BACKTEST_START_DATE
                original_end = BACKTEST_END_DATE
                
                # Run OOS test on 2025 data
                BACKTEST_START_DATE = TEST_START_DATE
                BACKTEST_END_DATE = datetime.now()
                
                logger.info(f"\nOOS Period: {BACKTEST_START_DATE.date()} to {BACKTEST_END_DATE.date()}")
                logger.info("Using optimized parameters from IS period...\n")
                
                oos_stats = run_backtest(strategy_params=study.best_params, suppress_logs=True)
                
                # Restore original dates
                BACKTEST_START_DATE = original_start
                BACKTEST_END_DATE = original_end
                
                # Compare IS vs OOS
                logger.info("\n" + "="*80)
                logger.info("IN-SAMPLE vs OUT-OF-SAMPLE COMPARISON")
                logger.info("="*80)
                logger.info(f"\n{'Metric':<25} {'IS (2024)':<15} {'OOS (2025)':<15} {'Degradation':<15}")
                logger.info("-"*70)
                
                is_sharpe = final_stats.get('sharpe_ratio', 0)
                oos_sharpe = oos_stats.get('sharpe_ratio', 0)
                sharpe_deg = ((oos_sharpe - is_sharpe) / abs(is_sharpe) * 100) if is_sharpe != 0 else 0
                
                is_wr = final_stats.get('win_rate', 0)
                oos_wr = oos_stats.get('win_rate', 0)
                wr_deg = oos_wr - is_wr
                
                is_trades = final_stats.get('total_trades', 0)
                oos_trades = oos_stats.get('total_trades', 0)
                
                logger.info(f"{'Sharpe Ratio':<25} {is_sharpe:<15.2f} {oos_sharpe:<15.2f} {sharpe_deg:>+14.1f}%")
                logger.info(f"{'Win Rate':<25} {is_wr:<14.1f}% {oos_wr:<14.1f}% {wr_deg:>+14.1f}pp")
                logger.info(f"{'Total Trades':<25} {is_trades:<15} {oos_trades:<15}")
                
                # Verdict
                logger.info("\n" + "="*80)
                if abs(sharpe_deg) < 30:  # Less than 30% degradation
                    logger.info("✅ ROBUST STRATEGY: OOS performance within acceptable range")
                elif abs(sharpe_deg) < 50:
                    logger.info("⚠️  MODERATE DEGRADATION: Consider re-optimization or regime filters")
                else:
                    logger.info("❌ SIGNIFICANT OVERFITTING: OOS performance severely degraded")
                logger.info("="*80)
            else:
                logger.warning(f"❌ FAILED MC VALIDATION: Sharpe CI lower={sharpe_ci_lower:.2f} < {MC_SHARPE_THRESHOLD}")
                logger.warning("   Strategy may be overfit or statistically insignificant")
                logger.warning("   Recommendation: Increase sample size or adjust parameters")
                logger.info("\n⚠️  Skipping OOS testing due to failed MC validation")
        elif run_is_oos and not has_mc:
            logger.warning("⚠️  RUN_IS_OOS=1 but Monte Carlo validator not available")
        elif run_is_oos and not final_stats.get('closed_trades'):
            logger.warning("⚠️  RUN_IS_OOS=1 but no trades to validate")

