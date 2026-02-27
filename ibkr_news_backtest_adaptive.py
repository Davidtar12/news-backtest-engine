#!/usr/bin/env python3
"""
IBKR News Backtest with Adaptive Exit Strategies
=================================================

Backtest breaking news trades on IBKR with:
- Entry: First trade ≥ (news time + 60s) AND vol_z > 1.5 (matches halt-resume logic)
- Adaptive exit strategies based on initial 60s pattern classification
- Multiple return snapshots (30s, 1m, 2m, 3m, 5m, 10m, 15m)
- Passive vs Active PnL tracking
- IBKR pacing compliance (8 req/sec, following hist-time-sales.py)

Entry Logic (matches halt-resume):
-----------------------------------
- Wait 60 seconds after news time
- Entry on FIRST bar where vol_z > 1.5
- Volume baseline: Initial 2-minute average (first 120 bars)

Pattern Classification (first 60s after entry):
------------------------------------------------
EXPLOSIVE_SPIKE: price >+2.5%, volume >5x baseline
  - Min hold: 30s, Max hold: 120s
  - Exit: Volume exhaustion (vol_z <0.5, obv_slope <-50, price_slope <0)

SUSTAINED_TREND: price >+0.5%, volume >1.5x baseline  
  - Min hold: 300s (5min), Max hold: None
  - Exit: 5% trailing stop from peak

WEAK_MOMENTUM: else
  - Min hold: 180s, Max hold: 180s
  - Exit: Time-based (3 minutes)

PnL Tracking:
-------------
- Active PnL: Entry → Exit (actual strategy performance)
- Passive PnL: News → Exit (buy-and-hold from news time)
- Strategy Alpha: Active PnL - Passive Drift (news→entry)

Usage:
------
python ibkr_news_backtest_adaptive.py --start 2024-09-14 --end 2024-10-14 --min-quality 0.45 --max-trades 100
"""

import argparse
import sys
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import numpy as np
import pandas as pd
from ib_insync import IB, Stock

# Import from existing modules
sys.path.insert(0, str(Path(__file__).parent))
try:
    from news_classifier_dbscan import NewsClassifier, cached_classify
    from news_trade_orchestrator import load_cache, save_cache
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)


# ============================================================================
# IBKR Pacing (from hist-time-sales.py)
# ============================================================================

class IBKRPacer:
    """Enforce IBKR pacing: <=8 requests per 1s window, <=60 in 10min.
    
    Following hist-time-sales.py pacing logic but with 8 req/sec instead of 5
    for better throughput while staying safe.
    """
    
    def __init__(self, max_per_10min: int = 60, max_per_1s: int = 8, hard_cap: int = 0):
        self.window10 = 600.0
        self.window1 = 1.0
        self.max10 = max_per_10min
        self.max1 = max_per_1s
        self.hard_cap = hard_cap
        self.times: deque[float] = deque()
        self.total = 0
    
    def wait_or_abort(self) -> bool:
        """Wait if needed to respect pacing limits. Returns False if hard cap reached."""
        if self.hard_cap and self.total >= self.hard_cap:
            return False
        
        now = time.time()
        
        # Clean 10-min window
        while self.times and now - self.times[0] > self.window10:
            self.times.popleft()
        
        # Enforce 1s rule
        count1 = sum(1 for t in self.times if now - t <= self.window1)
        if count1 >= self.max1:
            oldest1 = min(t for t in self.times if now - t <= self.window1)
            sleep_for = (oldest1 + self.window1) - now + 0.01
            if sleep_for > 0:
                time.sleep(sleep_for)
            now = time.time()
        
        # Enforce 10-min window
        if len(self.times) >= self.max10:
            oldest10 = self.times[0]
            sleep_for = (oldest10 + self.window10) - now + 0.01
            if sleep_for > 0:
                time.sleep(sleep_for)
        
        return True
    
    def record(self) -> None:
        """Record a request."""
        self.times.append(time.time())
        self.total += 1


# ============================================================================
# Pattern Classification & Exit Logic
# ============================================================================

@dataclass
class TradePattern:
    """Classification of trade pattern based on initial 60s momentum."""
    pattern_type: str  # EXPLOSIVE_SPIKE, SUSTAINED_TREND, WEAK_MOMENTUM
    price_change_60s: float  # % price change in first 60s
    volume_ratio: float  # Volume ratio to baseline
    min_hold_sec: int  # Minimum hold time
    max_hold_sec: Optional[int]  # Maximum hold time (None = unlimited)
    exit_strategy: str  # Description of exit strategy


def classify_pattern(df: pd.DataFrame, entry_time: pd.Timestamp, entry_price: float) -> TradePattern:
    """Classify trade pattern based on first 60 seconds after entry.
    
    Volume baseline: Initial 2-minute average after news (first 120 bars).
    
    Returns:
        TradePattern with classification and exit parameters
    """
    # Get first 60 seconds after entry
    first_60s = df[(df.index >= entry_time) & 
                   (df.index < entry_time + pd.Timedelta(seconds=60))]
    
    if len(first_60s) == 0:
        return TradePattern(
            pattern_type="WEAK_MOMENTUM",
            price_change_60s=0.0,
            volume_ratio=0.0,
            min_hold_sec=180,
            max_hold_sec=180,
            exit_strategy="Time Exit (no data)"
        )
    
    # Price change in first 60s
    price_60s = first_60s['close'].iloc[-1]
    price_change_pct = ((price_60s - entry_price) / entry_price) * 100
    
    # Volume baseline: initial 2-minute average (first 120 bars)
    initial_120s = df.iloc[:min(120, len(df))]
    avg_volume_initial = initial_120s['volume'].mean()
    
    # Volume in first 60s after entry
    volume_60s = first_60s['volume'].mean()
    volume_ratio = volume_60s / avg_volume_initial if avg_volume_initial > 0 else 0
    
    # Pattern classification
    if price_change_pct > 2.5 and volume_ratio > 5.0:
        return TradePattern(
            pattern_type="EXPLOSIVE_SPIKE",
            price_change_60s=price_change_pct,
            volume_ratio=volume_ratio,
            min_hold_sec=30,
            max_hold_sec=120,
            exit_strategy="Volume Exhaustion"
        )
    elif price_change_pct > 0.5 and volume_ratio > 1.5:
        return TradePattern(
            pattern_type="SUSTAINED_TREND",
            price_change_60s=price_change_pct,
            volume_ratio=volume_ratio,
            min_hold_sec=300,
            max_hold_sec=None,
            exit_strategy="Trailing Stop 5%"
        )
    else:
        return TradePattern(
            pattern_type="WEAK_MOMENTUM",
            price_change_60s=price_change_pct,
            volume_ratio=volume_ratio,
            min_hold_sec=180,
            max_hold_sec=180,
            exit_strategy="Time Exit"
        )


def compute_features_adaptive(df: pd.DataFrame) -> pd.DataFrame:
    """Compute features for adaptive exit strategies.
    
    Features:
    - OBV (On-Balance Volume)
    - OBV slope (9-bar rolling)
    - Volume z-score (60-bar window)
    - Price slope (5-bar rolling)
    - ATR (Average True Range from 1-second bars)
    """
    close = df['close'].astype(float)
    volume = df['volume'].astype(float)
    high = df['high'].astype(float)
    low = df['low'].astype(float)
    
    # OBV
    obv = pd.Series(0.0, index=df.index)
    for i in range(1, len(df)):
        if close.iloc[i] > close.iloc[i-1]:
            obv.iloc[i] = obv.iloc[i-1] + volume.iloc[i]
        elif close.iloc[i] < close.iloc[i-1]:
            obv.iloc[i] = obv.iloc[i-1] - volume.iloc[i]
        else:
            obv.iloc[i] = obv.iloc[i-1]
    
    # OBV slope (rolling least-squares)
    def rolling_slope(series: pd.Series, window: int) -> pd.Series:
        if window <= 1:
            return pd.Series(np.nan, index=series.index)
        
        x = np.arange(window, dtype=float)
        x_mean = x.mean()
        denom = np.sum((x - x_mean) ** 2)
        
        def _slope(y: np.ndarray) -> float:
            if np.any(np.isnan(y)) or denom == 0:
                return np.nan
            y_mean = y.mean()
            num = np.sum((x - x_mean) * (y - y_mean))
            return num / denom
        
        return series.rolling(window=window, min_periods=window).apply(_slope, raw=True)
    
    obv_slope = rolling_slope(obv, 9)
    
    # Volume z-score
    def rolling_zscore(series: pd.Series, window: int) -> pd.Series:
        if window <= 1:
            return pd.Series(np.nan, index=series.index)
        rol = series.rolling(window=window, min_periods=window)
        mean = rol.mean()
        std = rol.std(ddof=0)
        z = (series - mean) / std.replace(0, np.nan)
        return z.fillna(0.0)
    
    vol_z = rolling_zscore(volume, 60)
    
    # Price slope
    price_slope = rolling_slope(close, 5)
    
    # ATR (Average True Range)
    tr = pd.Series(0.0, index=df.index)
    for i in range(1, len(df)):
        tr.iloc[i] = max(
            high.iloc[i] - low.iloc[i],
            abs(high.iloc[i] - close.iloc[i-1]),
            abs(low.iloc[i] - close.iloc[i-1])
        )
    atr = tr.rolling(window=14, min_periods=1).mean()
    
    # Add to dataframe
    df['obv'] = obv
    df['obv_slope'] = obv_slope
    df['vol_z'] = vol_z
    df['price_slope'] = price_slope
    df['atr'] = atr
    
    return df


def find_exit_adaptive(df: pd.DataFrame, entry_idx: pd.Timestamp, entry_price: float, 
                       pattern: TradePattern) -> Tuple[pd.Timestamp, float, str]:
    """Find exit point based on pattern-specific strategy.
    
    Args:
        df: DataFrame with features
        entry_idx: Entry timestamp
        entry_price: Entry price
        pattern: TradePattern classification
    
    Returns:
        (exit_time, exit_price, exit_reason)
    """
    post = df.loc[df.index > entry_idx].copy()
    
    if len(post) == 0:
        return df.index[-1], df['close'].iloc[-1], "End of data"
    
    # Apply minimum hold time
    min_hold_end = entry_idx + pd.Timedelta(seconds=pattern.min_hold_sec)
    post = post[post.index >= min_hold_end]
    
    if len(post) == 0:
        return df.index[-1], df['close'].iloc[-1], f"End of data (min hold {pattern.min_hold_sec}s)"
    
    # Pattern-specific exit logic
    if pattern.pattern_type == "EXPLOSIVE_SPIKE":
        # Volume exhaustion: vol_z <0.5, obv_slope <-50, price_slope <0
        exit_mask = (
            (post['vol_z'] < 0.5) &
            (post['obv_slope'] < -50) &
            (post['price_slope'] < 0)
        )
        
        if exit_mask.any():
            exit_idx = post.index[exit_mask].min()
            return exit_idx, post.loc[exit_idx, 'close'], "Volume Exhaustion"
        
        # Max hold time
        if pattern.max_hold_sec is not None:
            max_hold_end = entry_idx + pd.Timedelta(seconds=pattern.max_hold_sec)
            if post.index.max() >= max_hold_end:
                closest_idx = post.index[post.index >= max_hold_end].min()
                return closest_idx, post.loc[closest_idx, 'close'], f"Max Hold {pattern.max_hold_sec}s"
    
    elif pattern.pattern_type == "SUSTAINED_TREND":
        # Trailing stop: 5% drop from peak
        running_peak = post['close'].expanding().max()
        trailing_dd = (post['close'] - running_peak) / running_peak
        hit_trailing = trailing_dd <= -0.05
        
        if hit_trailing.any():
            exit_idx = post.index[hit_trailing].min()
            return exit_idx, post.loc[exit_idx, 'close'], "Trailing Stop 5%"
    
    elif pattern.pattern_type == "WEAK_MOMENTUM":
        # Time-based exit
        if pattern.max_hold_sec is not None:
            max_hold_end = entry_idx + pd.Timedelta(seconds=pattern.max_hold_sec)
            if post.index.max() >= max_hold_end:
                closest_idx = post.index[post.index >= max_hold_end].min()
                return closest_idx, post.loc[closest_idx, 'close'], f"Time Exit {pattern.max_hold_sec}s"
    
    # Fallback: end of data
    return post.index[-1], post['close'].iloc[-1], "End of data"


def calculate_snapshots(df: pd.DataFrame, entry_idx: pd.Timestamp, entry_price: float,
                       exit_idx: pd.Timestamp, exit_price: float) -> Dict[str, float]:
    """Calculate returns at fixed time intervals.
    
    Snapshots: 30s, 1m, 2m, 3m, 5m, 10m, 15m
    Records even if exit occurred earlier.
    
    Returns:
        Dictionary with snapshot returns and actual exit metrics
    """
    snapshot_times = [30, 60, 120, 180, 300, 600, 900]  # seconds
    snapshots = {}
    
    for sec in snapshot_times:
        snapshot_time = entry_idx + pd.Timedelta(seconds=sec)
        
        # Find closest bar at or after snapshot time
        future_bars = df[df.index >= snapshot_time]
        if len(future_bars) > 0:
            snapshot_idx = future_bars.index[0]
            snapshot_price = df.loc[snapshot_idx, 'close']
            snapshot_ret = ((snapshot_price - entry_price) / entry_price) * 100
        else:
            # No data at snapshot time
            snapshot_ret = np.nan
        
        # Format: ret_30s, ret_1m, etc.
        if sec < 60:
            label = f"ret_{sec}s"
        else:
            label = f"ret_{sec//60}m"
        
        snapshots[label] = snapshot_ret
    
    # Actual exit metrics
    hold_time_sec = (exit_idx - entry_idx).total_seconds()
    exit_ret = ((exit_price - entry_price) / entry_price) * 100
    
    snapshots['actual_exit_ret'] = exit_ret
    snapshots['actual_hold_sec'] = hold_time_sec
    
    # Peak price and max drawdown
    trade_window = df[(df.index >= entry_idx) & (df.index <= exit_idx)]
    if len(trade_window) > 0:
        peak_price = trade_window['close'].max()
        peak_ret = ((peak_price - entry_price) / entry_price) * 100
        
        # Drawdown from entry
        min_price = trade_window['close'].min()
        max_dd = ((min_price - entry_price) / entry_price) * 100
    else:
        peak_ret = exit_ret
        max_dd = 0.0
    
    snapshots['peak_ret'] = peak_ret
    snapshots['max_dd'] = max_dd
    
    return snapshots


# ============================================================================
# IBKR Data Fetching
# ============================================================================

def connect_ibkr(ports: List[int], client_id: int = 22001) -> Optional[IB]:
    """Connect to IBKR TWS/Gateway."""
    ib = IB()
    for port in ports:
        try:
            ib.connect("127.0.0.1", port, clientId=client_id, timeout=3)
            if ib.isConnected():
                print(f"✓ Connected to IBKR on port {port}")
                return ib
        except Exception as e:
            continue
    return None


def fetch_1s_bars(ib: IB, ticker: str, exchange: str, start_dt: pd.Timestamp, 
                  end_dt: pd.Timestamp, pacer: IBKRPacer) -> Optional[pd.DataFrame]:
    """Fetch 1-second bars from IBKR.
    
    Args:
        ib: IB connection
        ticker: Stock symbol
        exchange: Exchange (NASDAQ, NYSE, etc.)
        start_dt: Start time (timezone-aware)
        end_dt: End time (timezone-aware)
        pacer: IBKR pacing controller
    
    Returns:
        DataFrame with columns: time, open, high, low, close, volume
        Or None if fetch fails
    """
    if not pacer.wait_or_abort():
        print(f"⚠ IBKR pacing hard cap reached")
        return None
    
    contract = Stock(ticker, exchange, "USD")
    
    # Convert to Eastern for IBKR API
    start_et = start_dt.tz_convert("America/New_York")
    end_et = end_dt.tz_convert("America/New_York")
    
    duration = int((end_et - start_et).total_seconds())
    
    try:
        bars = ib.reqHistoricalData(
            contract,
            endDateTime=end_et.strftime("%Y%m%d %H:%M:%S US/Eastern"),
            durationStr=f"{duration} S",
            barSizeSetting="1 secs",
            whatToShow="TRADES",
            useRTH=False,  # Include extended hours
            formatDate=1
        )
        
        pacer.record()
        
        if not bars:
            return None
        
        # Convert to DataFrame
        df = pd.DataFrame([{
            'time': bar.date,
            'open': bar.open,
            'high': bar.high,
            'low': bar.low,
            'close': bar.close,
            'volume': bar.volume
        } for bar in bars])
        
        # Convert time to timestamp index
        df['time'] = pd.to_datetime(df['time'])
        if df['time'].dt.tz is None:
            df['time'] = df['time'].dt.tz_localize('America/New_York')
        else:
            df['time'] = df['time'].dt.tz_convert('America/New_York')
        
        df = df.set_index('time')
        
        return df
    
    except Exception as e:
        print(f"✗ Error fetching {ticker} data: {e}")
        pacer.record()  # Count failed request for pacing
        return None


# ============================================================================
# Backtesting Engine
# ============================================================================

@dataclass
class NewsEvent:
    """Breaking news event."""
    ticker: str
    timestamp: pd.Timestamp
    title: str
    cluster_label: str
    quality: float
    exchange: str = "SMART"  # Will be refined


def determine_exchange(ticker: str) -> str:
    """Determine exchange for ticker.
    
    Simple heuristic - can be enhanced with actual exchange mapping.
    """
    # Common NASDAQ stocks
    nasdaq_common = {'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'NVDA', 'META', 'TSLA', 
                     'AMD', 'INTC', 'NFLX', 'ADBE', 'CSCO', 'AVGO', 'QCOM', 'TXN'}
    
    if ticker in nasdaq_common:
        return "NASDAQ"
    
    # Default to SMART for IB routing
    return "SMART"


def simulate_trade(ib: IB, event: NewsEvent, pacer: IBKRPacer) -> Optional[Dict]:
    """Simulate a single trade for a news event.
    
    Args:
        ib: IBKR connection
        event: NewsEvent
        pacer: IBKR pacing controller
    
    Returns:
        Trade result dictionary or None if trade skipped
    """
    # Entry target: news time + 60 seconds, vol_z > 1.5
    entry_target = event.timestamp + pd.Timedelta(seconds=60)
    
    # Fetch window: news time to 20 minutes after (for snapshots up to 15min)
    start_fetch = event.timestamp
    end_fetch = event.timestamp + pd.Timedelta(minutes=20)
    
    print(f"\n{'='*80}")
    print(f"Event: {event.ticker} @ {event.timestamp.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print(f"Title: {event.title[:80]}...")
    print(f"Cluster: {event.cluster_label}, Quality: {event.quality:.3f}")
    print(f"Entry target: {entry_target.strftime('%H:%M:%S')}")
    
    # Fetch 1-second bars
    df = fetch_1s_bars(ib, event.ticker, event.exchange, start_fetch, end_fetch, pacer)
    
    if df is None or len(df) == 0:
        print(f"✗ No data available for {event.ticker}")
        return None
    
    print(f"✓ Fetched {len(df)} bars ({df.index[0].strftime('%H:%M:%S')} to {df.index[-1].strftime('%H:%M:%S')})")
    
    # Compute features first (needed for vol_z filter)
    df = compute_features_adaptive(df)
    
    # Entry logic: First trade ≥ (event time + 60s) AND vol_z > 1.5
    # This matches halt-resume logic: First trade ≥ Halt Lift AND Vol_Z > 1.5
    entry_target = event.timestamp + pd.Timedelta(seconds=60)
    
    # Filter: time >= entry_target AND vol_z > 1.5
    entry_candidates = df[(df.index >= entry_target) & (df['vol_z'] > 1.5)]
    
    if len(entry_candidates) == 0:
        print(f"✗ No entry signal: no bars with vol_z > 1.5 after {entry_target.strftime('%H:%M:%S')}")
        return None
    
    entry_idx = entry_candidates.index[0]
    entry_price = df.loc[entry_idx, 'close']
    entry_vol_z = df.loc[entry_idx, 'vol_z']
    
    # Get news-time price for passive PnL comparison
    news_bars = df[df.index >= event.timestamp]
    if len(news_bars) == 0:
        print(f"✗ No data at news time")
        return None
    
    news_time_idx = news_bars.index[0]
    news_price = news_bars['close'].iloc[0]
    
    # Calculate passive entry drift (news time to actual entry)
    passive_drift = ((entry_price - news_price) / news_price) * 100
    
    print(f"✓ Entry: {entry_idx.strftime('%H:%M:%S')} @ ${entry_price:.4f} (vol_z={entry_vol_z:.2f})")
    print(f"  - Passive drift (news→entry): {passive_drift:+.4f}%")
    
    # Classify pattern based on first 60 seconds
    pattern = classify_pattern(df, entry_idx, entry_price)
    print(f"✓ Pattern: {pattern.pattern_type}")
    print(f"  - Price +{pattern.price_change_60s:.2f}% in 60s")
    print(f"  - Volume {pattern.volume_ratio:.2f}x baseline")
    print(f"  - Strategy: {pattern.exit_strategy}")
    print(f"  - Hold: {pattern.min_hold_sec}s to {pattern.max_hold_sec or 'unlimited'}s")
    
    # Find exit
    exit_idx, exit_price, exit_reason = find_exit_adaptive(df, entry_idx, entry_price, pattern)
    
    # Active PnL: actual strategy performance (entry to exit)
    active_pnl = ((exit_price - entry_price) / entry_price) * 100
    hold_sec = (exit_idx - entry_idx).total_seconds()
    
    # Passive PnL: raw market drift (news time to exit time)
    # This is what you would get if you bought at news time and held to exit time
    passive_pnl = ((exit_price - news_price) / news_price) * 100
    
    # Strategy alpha: active PnL minus passive drift
    # Positive alpha = strategy outperformed passive hold from entry point
    strategy_alpha = active_pnl - passive_drift
    
    print(f"✓ Exit: {exit_idx.strftime('%H:%M:%S')} @ ${exit_price:.4f}")
    print(f"  - Active PnL (entry→exit): {active_pnl:+.2f}%")
    print(f"  - Passive PnL (news→exit): {passive_pnl:+.2f}%")
    print(f"  - Strategy Alpha: {strategy_alpha:+.2f}%")
    print(f"  - Hold time: {hold_sec:.0f}s")
    print(f"  - Reason: {exit_reason}")
    
    # Calculate snapshots
    snapshots = calculate_snapshots(df, entry_idx, entry_price, exit_idx, exit_price)
    
    # Print snapshot comparison
    print(f"\n  Snapshots:")
    for key in ['ret_30s', 'ret_1m', 'ret_2m', 'ret_3m', 'ret_5m', 'ret_10m', 'ret_15m']:
        val = snapshots.get(key)
        if val is not None and not np.isnan(val):
            print(f"    {key:10s}: {val:+7.2f}%")
    
    print(f"    {'Peak':10s}: {snapshots['peak_ret']:+7.2f}%")
    print(f"    {'Max DD':10s}: {snapshots['max_dd']:+7.2f}%")
    
    # Build result
    result = {
        'ticker': event.ticker,
        'news_time': event.timestamp,
        'news_price': news_price,
        'entry_time': entry_idx,
        'entry_price': entry_price,
        'entry_vol_z': entry_vol_z,
        'passive_drift': passive_drift,  # News→Entry drift
        'exit_time': exit_idx,
        'exit_price': exit_price,
        'active_pnl': active_pnl,  # Entry→Exit (strategy performance)
        'passive_pnl': passive_pnl,  # News→Exit (buy-and-hold)
        'strategy_alpha': strategy_alpha,  # Active minus passive drift
        'hold_seconds': hold_sec,
        'exit_reason': exit_reason,
        'pattern_type': pattern.pattern_type,
        'pattern_price_60s': pattern.price_change_60s,
        'pattern_volume_ratio': pattern.volume_ratio,
        'cluster_label': event.cluster_label,
        'quality': event.quality,
        'title': event.title,
        **snapshots  # Add all snapshot returns
    }
    
    return result


def run_backtest(events: List[NewsEvent], ports: List[int], max_trades: int = 100) -> pd.DataFrame:
    """Run backtest on news events.
    
    Args:
        events: List of NewsEvent objects
        ports: IBKR ports to try
        max_trades: Maximum number of trades to execute
    
    Returns:
        DataFrame with trade results
    """
    ib = connect_ibkr(ports)
    if ib is None:
        print("✗ Failed to connect to IBKR")
        return pd.DataFrame()
    
    pacer = IBKRPacer(max_per_10min=60, max_per_1s=8, hard_cap=0)
    results = []
    
    try:
        for i, event in enumerate(events):
            if len(results) >= max_trades:
                print(f"\n✓ Reached max trades limit ({max_trades})")
                break
            
            print(f"\n[{i+1}/{len(events)}] Processing {event.ticker}...")
            
            result = simulate_trade(ib, event, pacer)
            if result is not None:
                results.append(result)
                print(f"✓ Trade #{len(results)} completed")
        
    finally:
        ib.disconnect()
        print(f"\n✓ Disconnected from IBKR")
        print(f"✓ Completed {len(results)} trades out of {len(events)} events")
    
    if results:
        df = pd.DataFrame(results)
        return df
    else:
        return pd.DataFrame()


# ============================================================================
# Load News Events
# ============================================================================

def load_breaking_news(start_date: str, end_date: str, min_quality: float = 0.45) -> List[NewsEvent]:
    """Load breaking news events from cache.
    
    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        min_quality: Minimum quality threshold
    
    Returns:
        List of NewsEvent objects
    """
    # Load classification cache
    cache_path = Path(__file__).parent / "classification_cache.json"
    if not cache_path.exists():
        print(f"✗ Cache not found: {cache_path}")
        return []
    
    cache = load_cache(cache_path)
    
    # Parse date range
    start_dt = pd.to_datetime(start_date).tz_localize('America/New_York')
    end_dt = pd.to_datetime(end_date).tz_localize('America/New_York') + pd.Timedelta(days=1)
    
    events = []
    
    for key, entry in cache.items():
        # Check if entry has classification
        if 'classification' not in entry:
            continue
        
        classification = entry['classification']
        
        # Filter by quality and cluster
        if classification.get('quality', 0) < min_quality:
            continue
        
        cluster = classification.get('cluster_label', '')
        if cluster in ['noise', 'unclassified']:
            continue
        
        # Parse metadata
        ticker = entry.get('ticker', '')
        timestamp_str = entry.get('date', '')
        title = entry.get('title', '')
        
        if not ticker or not timestamp_str:
            continue
        
        # Parse timestamp
        try:
            timestamp = pd.to_datetime(timestamp_str)
            if timestamp.tzinfo is None:
                timestamp = timestamp.tz_localize('UTC')
            timestamp = timestamp.tz_convert('America/New_York')
        except Exception:
            continue
        
        # Filter by date range
        if not (start_dt <= timestamp < end_dt):
            continue
        
        # Determine exchange
        exchange = determine_exchange(ticker)
        
        events.append(NewsEvent(
            ticker=ticker,
            timestamp=timestamp,
            title=title,
            cluster_label=cluster,
            quality=classification.get('quality', 0),
            exchange=exchange
        ))
    
    # Sort by timestamp
    events.sort(key=lambda e: e.timestamp)
    
    print(f"\n✓ Loaded {len(events)} breaking news events")
    print(f"  Date range: {start_date} to {end_date}")
    print(f"  Min quality: {min_quality}")
    
    # Print cluster distribution
    from collections import Counter
    cluster_counts = Counter(e.cluster_label for e in events)
    print(f"\n  Cluster distribution:")
    for cluster, count in cluster_counts.most_common():
        print(f"    {cluster:30s}: {count:3d}")
    
    return events


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="IBKR News Backtest with Adaptive Exit Strategies",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--min-quality", type=float, default=0.45, help="Minimum quality threshold")
    parser.add_argument("--max-trades", type=int, default=100, help="Maximum trades to execute")
    parser.add_argument("--ports", nargs="*", type=int, default=[7497, 7496, 4002, 4001],
                       help="IBKR ports to try")
    parser.add_argument("--output", default="backtest_results.csv", help="Output CSV file")
    
    args = parser.parse_args()
    
    print("="*80)
    print("IBKR News Backtest - Adaptive Exit Strategies")
    print("="*80)
    print(f"Date range: {args.start} to {args.end}")
    print(f"Min quality: {args.min_quality}")
    print(f"Max trades: {args.max_trades}")
    print(f"Entry filter: vol_z > 1.5")
    print(f"IBKR ports: {args.ports}")
    print("="*80)
    
    # Load news events
    events = load_breaking_news(args.start, args.end, args.min_quality)
    
    if not events:
        print("\n✗ No events found matching criteria")
        return 1
    
    # Run backtest
    results_df = run_backtest(events, args.ports, args.max_trades)
    
    if results_df.empty:
        print("\n✗ No trades executed")
        return 1
    
    # Save results
    output_path = Path(args.output)
    results_df.to_csv(output_path, index=False)
    print(f"\n✓ Results saved to: {output_path}")
    
    # Print summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    total_trades = len(results_df)
    active_winners = (results_df['active_pnl'] > 0).sum()
    active_losers = (results_df['active_pnl'] < 0).sum()
    active_win_rate = (active_winners / total_trades) * 100 if total_trades > 0 else 0
    
    passive_winners = (results_df['passive_pnl'] > 0).sum()
    passive_win_rate = (passive_winners / total_trades) * 100 if total_trades > 0 else 0
    
    print(f"\n📊 Overall Performance:")
    print(f"  Total trades: {total_trades}")
    print(f"\n  🎯 ACTIVE (Strategy Entry→Exit):")
    print(f"     Winners: {active_winners} ({active_win_rate:.1f}%)")
    print(f"     Losers: {active_losers} ({100-active_win_rate:.1f}%)")
    print(f"     Avg return: {results_df['active_pnl'].mean():+.2f}%")
    print(f"     Median return: {results_df['active_pnl'].median():+.2f}%")
    print(f"     Best trade: {results_df['active_pnl'].max():+.2f}%")
    print(f"     Worst trade: {results_df['active_pnl'].min():+.2f}%")
    
    print(f"\n  📈 PASSIVE (News→Exit Buy-and-Hold):")
    print(f"     Winners: {passive_winners} ({passive_win_rate:.1f}%)")
    print(f"     Avg return: {results_df['passive_pnl'].mean():+.2f}%")
    print(f"     Median return: {results_df['passive_pnl'].median():+.2f}%")
    print(f"     Best: {results_df['passive_pnl'].max():+.2f}%")
    print(f"     Worst: {results_df['passive_pnl'].min():+.2f}%")
    
    print(f"\n  ⚡ STRATEGY ALPHA (Active - Passive Drift):")
    print(f"     Avg alpha: {results_df['strategy_alpha'].mean():+.2f}%")
    print(f"     Median alpha: {results_df['strategy_alpha'].median():+.2f}%")
    print(f"     Positive alpha: {(results_df['strategy_alpha'] > 0).sum()} trades ({(results_df['strategy_alpha'] > 0).sum()/total_trades*100:.1f}%)")
    
    print(f"\n  ⏱️  Timing:")
    print(f"     Avg passive drift (news→entry): {results_df['passive_drift'].mean():+.2f}%")
    print(f"     Avg hold time: {results_df['hold_seconds'].mean():.0f}s ({results_df['hold_seconds'].mean()/60:.1f} min)")
    print(f"     Avg entry vol_z: {results_df['entry_vol_z'].mean():.2f}")
    
    # By pattern
    print(f"\n📋 By Pattern:")
    for pattern in ['EXPLOSIVE_SPIKE', 'SUSTAINED_TREND', 'WEAK_MOMENTUM']:
        pattern_df = results_df[results_df['pattern_type'] == pattern]
        if len(pattern_df) > 0:
            pattern_wins = (pattern_df['active_pnl'] > 0).sum()
            pattern_wr = (pattern_wins / len(pattern_df)) * 100
            print(f"  {pattern:20s}: {len(pattern_df):3d} trades, WR={pattern_wr:5.1f}%, "
                  f"Active={pattern_df['active_pnl'].mean():+6.2f}%, "
                  f"Passive={pattern_df['passive_pnl'].mean():+6.2f}%")
    
    # By cluster
    print(f"\n🏷️  Top Clusters (by Active PnL):")
    cluster_stats = results_df.groupby('cluster_label').agg({
        'active_pnl': ['count', 'mean', 'median'],
        'passive_pnl': ['mean'],
        'strategy_alpha': ['mean']
    }).round(2)
    cluster_stats.columns = ['count', 'active_avg', 'active_med', 'passive_avg', 'alpha_avg']
    cluster_stats = cluster_stats.sort_values('active_avg', ascending=False).head(10)
    print(cluster_stats.to_string())
    
    # Snapshot comparison
    print(f"\n⏱️  Snapshot Analysis (Average Returns):")
    snapshot_cols = ['ret_30s', 'ret_1m', 'ret_2m', 'ret_3m', 'ret_5m', 'ret_10m', 'ret_15m', 'active_pnl']
    print(f"  {'Time':10s}  {'Avg Return':>10s}  {'vs Active':>10s}")
    print(f"  {'-'*10}  {'-'*10}  {'-'*10}")
    
    active_avg = results_df['active_pnl'].mean()
    for col in snapshot_cols:
        if col in results_df.columns:
            avg = results_df[col].mean()
            if col == 'active_pnl':
                label = 'ACTUAL'
                diff = 0.0
            else:
                label = col.replace('ret_', '')
                diff = avg - active_avg
            
            print(f"  {label:10s}  {avg:+10.2f}%  {diff:+10.2f}%")
    
    print("\n" + "="*80)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
