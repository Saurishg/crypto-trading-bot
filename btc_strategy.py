#!/usr/bin/env python3
"""
BTC EMA5/13 strategy backtest — 2-year window, incremental OHLC cache.

Outputs:
  equity_curve.png
  trade_log.csv
  Writes a one-line summary to fraqtoos ai_context for the daily digest.
"""
import warnings
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use('Agg')

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import ccxt
import pandas as pd
import numpy as np
import ta
import matplotlib.pyplot as plt

HERE       = Path(__file__).parent
CACHE_PATH = HERE / "btc_1h_cache.csv"
EQUITY_PNG = HERE / "equity_curve.png"
TRADES_CSV = HERE / "trade_log.csv"
WINDOW_DAYS = 730  # ~2 years


def load_cached() -> pd.DataFrame:
    if not CACHE_PATH.exists():
        return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df = pd.read_csv(CACHE_PATH)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df


def fetch_ohlc() -> pd.DataFrame:
    """Fetch BTC/USDT 1h OHLC, using cache for everything except the tail."""
    cached = load_cached()
    exchange = ccxt.binance({
        'rateLimit': 1200,
        'enableRateLimit': True,
        'timeout': 30000,  # 30s — prevents indefinite hang
    })

    if cached.empty:
        since = exchange.parse8601(
            (datetime.utcnow() - timedelta(days=WINDOW_DAYS)).strftime('%Y-%m-%dT%H:%M:%SZ')
        )
    else:
        # Re-fetch from last cached timestamp + 1h to pick up any new bars
        last = cached['timestamp'].max()
        since = int((last + timedelta(hours=1)).timestamp() * 1000)

    new_ohlc = []
    while True:
        batch = exchange.fetch_ohlcv('BTC/USDT', '1h', since=since, limit=1000)
        if not batch:
            break
        new_ohlc += batch
        since = batch[-1][0] + 3600000
        if len(new_ohlc) >= 20000:
            break

    if new_ohlc:
        df_new = pd.DataFrame(new_ohlc, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df_new['timestamp'] = pd.to_datetime(df_new['timestamp'], unit='ms')
        combined = pd.concat([cached, df_new], ignore_index=True) \
                     .drop_duplicates('timestamp') \
                     .sort_values('timestamp') \
                     .reset_index(drop=True)
    else:
        combined = cached

    # Trim to WINDOW_DAYS
    cutoff = datetime.utcnow() - timedelta(days=WINDOW_DAYS)
    combined = combined[combined['timestamp'] >= cutoff].reset_index(drop=True)

    combined.to_csv(CACHE_PATH, index=False)
    return combined


def write_ai_context(summary: str):
    """Append to fraqtoos ai_context so the daily digest has BTC results."""
    try:
        sys.path.insert(0, "/home/work/fraqtoos")
        from core.ai_context import write_summary
        write_summary("BTC Strategy Bot", summary)
    except Exception as e:
        print(f"[ai_context] skip: {e}", file=sys.stderr)


def main():
    try:
        df = fetch_ohlc()
    except Exception as e:
        msg = f"BTC data fetch failed: {e}"
        print(msg, file=sys.stderr)
        write_ai_context(msg)
        sys.exit(1)

    if len(df) < 200:
        msg = f"BTC cache too small ({len(df)} candles) — skipping backtest"
        print(msg, file=sys.stderr)
        write_ai_context(msg)
        sys.exit(1)

    print(f"Loaded {len(df)} candles: {df['timestamp'].iloc[0].date()} → {df['timestamp'].iloc[-1].date()}")

    df['ema5']  = ta.trend.EMAIndicator(df['close'], window=5).ema_indicator()
    df['ema13'] = ta.trend.EMAIndicator(df['close'], window=13).ema_indicator()
    df['rsi']   = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
    df['atr']   = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()

    capital      = 10000.0
    in_position  = False
    entry_price  = stop_loss = take_profit = position_size = 0.0
    equity_curve = [capital]
    trade_log    = []
    wins = losses = total_trades = 0

    circuit_breaker = False
    ROLLING_PEAK_BARS = 180 * 24  # 180-day rolling window for circuit breaker peak

    for i in range(14, len(df)):
        price     = df['close'].iloc[i]
        atr_val   = df['atr'].iloc[i]
        ema5_val  = df['ema5'].iloc[i]
        ema13_val = df['ema13'].iloc[i]
        rsi_val   = df['rsi'].iloc[i]

        # Total portfolio value = cash + open position mark-to-market
        portfolio_value = capital + (position_size * price if in_position else 0)

        # Rolling peak — prevents an early lucky high from permanently locking the strategy
        lookback = equity_curve[-ROLLING_PEAK_BARS:] if len(equity_curve) >= ROLLING_PEAK_BARS else equity_curve
        peak_equity = max(lookback) if lookback else portfolio_value

        # 70%/80% thresholds — appropriate for BTC (30% drawdown trip, recover to -20%)
        if not circuit_breaker and portfolio_value < peak_equity * 0.70:
            circuit_breaker = True
        if circuit_breaker and portfolio_value > peak_equity * 0.80:
            circuit_breaker = False

        if not in_position:
            if ema5_val > ema13_val and 25 <= rsi_val <= 75 and not circuit_breaker:
                stop_loss_price   = price - 1.5 * atr_val
                take_profit_price = price + 3.0 * atr_val
                risk_amount = capital * 0.02
                raw_size = risk_amount / (price - stop_loss_price)
                position_size = min(raw_size, capital * 0.95 / price)
                if capital <= 0 or position_size <= 0:
                    equity_curve.append(capital)
                    continue
                entry_price = price
                stop_loss   = stop_loss_price
                take_profit = take_profit_price
                capital -= position_size * entry_price
                in_position = True
                trade_log.append({'ts': df['timestamp'].iloc[i], 'action': 'BUY',
                                  'price': entry_price, 'sl': stop_loss, 'tp': take_profit})
        else:
            unrealized_pnl = position_size * (price - entry_price)
            if unrealized_pnl > 1.5 * atr_val * position_size:
                stop_loss = max(stop_loss, entry_price)
            if unrealized_pnl > 2.0 * atr_val * position_size:
                stop_loss = max(stop_loss, price - 1.0 * atr_val)

            exit_triggered = (
                ema5_val < ema13_val or
                rsi_val > 78 or
                price <= stop_loss or
                price >= take_profit
            )
            if exit_triggered:
                pnl = position_size * (price - entry_price)
                capital += position_size * entry_price + pnl
                in_position = False
                total_trades += 1
                if pnl > 0: wins += 1
                else:       losses += 1
                trade_log.append({'ts': df['timestamp'].iloc[i], 'action': 'SELL',
                                  'price': price, 'pnl': pnl})

        # Append total portfolio value (cash + open position MTM)
        equity_curve.append(capital + (position_size * price if in_position else 0))

    if in_position:
        price = df['close'].iloc[-1]
        pnl = position_size * (price - entry_price)
        capital += position_size * entry_price + pnl
        total_trades += 1
        if pnl > 0: wins += 1
        else:       losses += 1
        trade_log.append({'ts': df['timestamp'].iloc[-1], 'action': 'CLOSE', 'price': price, 'pnl': pnl})
        equity_curve.append(capital)

    equity_series = pd.Series(equity_curve)
    returns       = equity_series.pct_change().dropna()
    total_return  = (equity_series.iloc[-1] / 10000 - 1) * 100
    sharpe        = np.sqrt(8760) * returns.mean() / returns.std() if returns.std() > 0 else 0
    peak          = equity_series.cummax()
    max_dd        = ((peak - equity_series) / peak).max() * 100
    win_rate      = wins / total_trades * 100 if total_trades > 0 else 0
    gross_profit  = sum(t['pnl'] for t in trade_log if 'pnl' in t and t['pnl'] > 0)
    gross_loss    = sum(abs(t['pnl']) for t in trade_log if 'pnl' in t and t['pnl'] < 0)
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    print(f"\n{'='*45}")
    print(f"  BTC EMA5/13 Strategy — 2-Year Backtest")
    print(f"{'='*45}")
    print(f"  Total Return:   {total_return:>10.2f}%")
    print(f"  Sharpe Ratio:   {sharpe:>10.2f}")
    print(f"  Max Drawdown:   {max_dd:>10.2f}%")
    print(f"  Win Rate:       {win_rate:>10.2f}%")
    print(f"  Profit Factor:  {profit_factor:>10.2f}")
    print(f"  Total Trades:   {total_trades:>10}")
    print(f"  Final Capital:  ${capital:>12,.2f}")
    print(f"{'='*45}")

    plt.figure(figsize=(12, 6))
    plt.plot(equity_series.values)
    plt.title('BTC EMA5/13 Strategy — Equity Curve')
    plt.xlabel('Bars'); plt.ylabel('Capital ($)')
    plt.tight_layout()
    plt.savefig(EQUITY_PNG)
    plt.close()
    pd.DataFrame(trade_log).to_csv(TRADES_CSV, index=False)

    summary = (
        f"BTC 2y backtest: return {total_return:.1f}%, "
        f"Sharpe {sharpe:.2f}, MaxDD {max_dd:.1f}%, "
        f"WR {win_rate:.1f}% on {total_trades} trades, "
        f"final ${capital:,.0f}"
    )
    write_ai_context(summary)
    print(f"Saved: {EQUITY_PNG.name} + {TRADES_CSV.name}")


if __name__ == "__main__":
    main()
