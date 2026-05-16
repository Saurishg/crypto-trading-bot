#!/usr/bin/env python3
"""
ADX filter A/B backtest.

Compares 3 strategy variants against the same 4y BTC 4h dataset:
  V1: baseline    — no ADX filter (current bot logic)
  V2: ADX > 20    — skip entries when trend is weak/choppy
  V3: ADX > 25    — only trade strong trends (textbook quant filter)

Metrics: total return, # trades, win rate, profit factor, max drawdown,
         Sharpe ratio, avg trade.
"""
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import pandas as pd
import numpy as np
import ta

HERE = Path(__file__).parent
CACHE = HERE / "btc_4h_cache.csv"

# Strategy params (match live bot)
FEE_RATE   = 0.00075
SLIPPAGE   = 0.0005
RISK_PCT   = 0.015
ATR_SL     = 2.0   # match live_bot.py
ATR_TP     = 6.0
RSI_LO     = 50
RSI_HI     = 70


def load_data() -> pd.DataFrame:
    df = pd.read_csv(CACHE)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp')

    # Indicators
    df['ema21']  = ta.trend.EMAIndicator(df['close'], window=21).ema_indicator()
    df['ema55']  = ta.trend.EMAIndicator(df['close'], window=55).ema_indicator()
    df['ema200'] = ta.trend.EMAIndicator(df['close'], window=200).ema_indicator()
    df['rsi']    = ta.momentum.RSIIndicator(df['close'], window=14).rsi()

    macd = ta.trend.MACD(df['close'], window_slow=26, window_fast=12, window_sign=9)
    df['macd']     = macd.macd()
    df['macd_sig'] = macd.macd_signal()

    df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()

    # ADX
    adx_obj = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], window=14)
    df['adx'] = adx_obj.adx()
    df['pdi'] = adx_obj.adx_pos()
    df['ndi'] = adx_obj.adx_neg()

    return df.dropna().copy()


def backtest(df: pd.DataFrame, adx_min: float = 0.0, label: str = "") -> dict:
    """Run backtest with optional ADX > adx_min filter. Mirrors live bot logic."""
    capital = 10_000.0
    equity_curve = []
    trades = []

    in_pos = False
    entry_price = stop_loss = take_profit = position_size = 0.0
    bars_held = 0
    peak = capital

    for i in range(len(df)):
        row = df.iloc[i]
        price = row['close']

        # Update equity
        if in_pos:
            cur_equity = capital + position_size * (price - entry_price)
        else:
            cur_equity = capital
        equity_curve.append(cur_equity)
        peak = max(peak, cur_equity)

        # Entry signal (match live bot: trend + momentum + RSI band)
        if not in_pos:
            cond_trend  = row['ema21'] > row['ema55'] and price > row['ema200']
            cond_macd   = row['macd'] > row['macd_sig']
            cond_rsi    = RSI_LO <= row['rsi'] <= RSI_HI
            cond_adx    = row['adx'] >= adx_min  # new filter

            if cond_trend and cond_macd and cond_rsi and cond_adx:
                atr = row['atr']
                sl_price = price - ATR_SL * atr
                tp_price = price + ATR_TP * atr
                # Position sizing: risk RISK_PCT of capital
                risk_per_unit = price - sl_price
                if risk_per_unit <= 0: continue
                position_size = (capital * RISK_PCT) / risk_per_unit
                entry_cost = position_size * price * (1 + FEE_RATE + SLIPPAGE)
                if entry_cost > capital: continue
                capital -= entry_cost
                entry_price = price
                stop_loss = sl_price
                take_profit = tp_price
                bars_held = 0
                in_pos = True
        else:
            bars_held += 1
            # Trailing stop after 1×ATR profit
            if price > entry_price + row['atr']:
                stop_loss = max(stop_loss, entry_price + 1.0 * row['atr'])

            hit_sl = price <= stop_loss
            hit_tp = price >= take_profit
            time_exit = bars_held >= 6 and row['ema21'] < row['ema55'] and row['rsi'] > 75

            if hit_sl or hit_tp or time_exit:
                exit_price = price
                pnl = position_size * (exit_price - entry_price)
                exit_proceeds = position_size * exit_price - (FEE_RATE + SLIPPAGE) * exit_price * position_size
                capital += exit_proceeds
                trades.append({
                    'entry_idx': i - bars_held,
                    'exit_idx': i,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'bars_held': bars_held,
                    'pnl': pnl,
                    'pnl_pct': (exit_price - entry_price) / entry_price * 100,
                    'reason': 'TP' if hit_tp else ('SL' if hit_sl else 'TIME'),
                })
                in_pos = False
                bars_held = 0

    # Final equity
    if in_pos:
        capital += position_size * df.iloc[-1]['close'] - (FEE_RATE + SLIPPAGE) * df.iloc[-1]['close'] * position_size
    equity_curve.append(capital)

    # ── Metrics ──
    n_trades = len(trades)
    wins = [t for t in trades if t['pnl'] > 0]
    losses = [t for t in trades if t['pnl'] <= 0]
    win_rate = (len(wins) / n_trades * 100) if n_trades else 0
    gross_win = sum(t['pnl'] for t in wins)
    gross_loss = abs(sum(t['pnl'] for t in losses))
    profit_factor = gross_win / gross_loss if gross_loss > 0 else float('inf') if gross_win > 0 else 0

    eq = np.array(equity_curve)
    returns = np.diff(eq) / eq[:-1]
    sharpe = (returns.mean() / returns.std() * np.sqrt(365 * 6)) if returns.std() > 0 else 0  # 6 candles/day

    dd = (eq - np.maximum.accumulate(eq)) / np.maximum.accumulate(eq)
    max_dd_pct = abs(dd.min() * 100) if len(dd) else 0

    total_return_pct = (capital / 10_000 - 1) * 100
    avg_trade_pct = np.mean([t['pnl_pct'] for t in trades]) if trades else 0
    avg_win_pct = np.mean([t['pnl_pct'] for t in wins]) if wins else 0
    avg_loss_pct = np.mean([t['pnl_pct'] for t in losses]) if losses else 0

    return {
        'label':         label,
        'adx_min':       adx_min,
        'final_capital': capital,
        'total_return_pct': total_return_pct,
        'n_trades':      n_trades,
        'win_rate_pct':  win_rate,
        'profit_factor': profit_factor,
        'sharpe':        sharpe,
        'max_dd_pct':    max_dd_pct,
        'avg_trade_pct': avg_trade_pct,
        'avg_win_pct':   avg_win_pct,
        'avg_loss_pct':  avg_loss_pct,
        'trades':        trades,
    }


def fmt(v, suffix='', dec=2):
    if v == float('inf'): return '∞'
    return f'{v:+.{dec}f}{suffix}' if (isinstance(v, (int,float)) and suffix == '%') else f'{v:.{dec}f}{suffix}'


def print_results(results: list):
    print('=' * 95)
    print(f'  {"Variant":<22}  {"Trades":>7} {"WinRate":>9} {"PF":>6} {"Return":>10} {"MaxDD":>9} {"Sharpe":>8}')
    print('-' * 95)
    for r in results:
        print(
            f'  {r["label"]:<22}  '
            f'{r["n_trades"]:>7} '
            f'{r["win_rate_pct"]:>8.1f}% '
            f'{r["profit_factor"]:>6.2f} '
            f'{r["total_return_pct"]:>+9.2f}% '
            f'{r["max_dd_pct"]:>+8.2f}% '
            f'{r["sharpe"]:>+8.2f}'
        )
    print('=' * 95)


def main():
    print('Loading 4h BTC data...')
    df = load_data()
    print(f'Loaded {len(df)} candles  ({df.index[0]}  →  {df.index[-1]})')
    print(f'ADX range: min={df["adx"].min():.1f}  median={df["adx"].median():.1f}  max={df["adx"].max():.1f}')
    print()

    variants = [
        (0.0,  'V1 baseline (no ADX)'),
        (20.0, 'V2 ADX > 20'),
        (25.0, 'V3 ADX > 25'),
        (30.0, 'V4 ADX > 30 (strict)'),
    ]

    results = []
    for adx_min, label in variants:
        print(f'Running {label}...')
        r = backtest(df, adx_min=adx_min, label=label)
        results.append(r)

    print()
    print_results(results)
    print()

    # Best by Sharpe
    best = max(results, key=lambda r: r['sharpe'])
    print(f'🏆 Best by Sharpe: {best["label"]}  (Sharpe {best["sharpe"]:+.2f}, return {best["total_return_pct"]:+.2f}%)')

    # Best by return
    best_ret = max(results, key=lambda r: r['total_return_pct'])
    print(f'💰 Best by return: {best_ret["label"]}  (return {best_ret["total_return_pct"]:+.2f}%)')

    return results


if __name__ == '__main__':
    main()
