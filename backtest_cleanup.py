#!/usr/bin/env python3
"""
Test the impact of removing/replacing weak components.

All variants have ADX > 25 (already validated). Comparing:
  V1: Baseline   — keeps broken time-exit (current bot logic)
  V2: Replace time-exit with simple EMA-cross exit
  V3: Remove time-exit entirely (only SL/TP)
  V4: V2 + breakeven trailing stop after +1.5 ATR profit
"""
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import pandas as pd
import numpy as np
import ta

HERE = Path(__file__).parent
CACHE = HERE / "btc_4h_cache.csv"
FEE_RATE   = 0.00075
SLIPPAGE   = 0.0005
RISK_PCT   = 0.015
ATR_SL     = 2.0
ATR_TP     = 6.0
RSI_LO     = 50
RSI_HI     = 70
ADX_MIN    = 25.0


def load_data() -> pd.DataFrame:
    df = pd.read_csv(CACHE)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp')
    df['ema21']  = ta.trend.EMAIndicator(df['close'], window=21).ema_indicator()
    df['ema55']  = ta.trend.EMAIndicator(df['close'], window=55).ema_indicator()
    df['ema200'] = ta.trend.EMAIndicator(df['close'], window=200).ema_indicator()
    df['rsi']    = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
    macd = ta.trend.MACD(df['close'], 26, 12, 9)
    df['macd']     = macd.macd()
    df['macd_sig'] = macd.macd_signal()
    df['atr']      = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()
    df['adx']      = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], window=14).adx()
    return df.dropna().copy()


def backtest(df, exit_mode='current', breakeven=False, label='') -> dict:
    """exit_mode: 'current' (broken), 'ema_cross', 'sl_tp_only'"""
    capital = 10_000.0
    eq_curve = []; trades = []
    in_pos = False
    entry_price = stop_loss = take_profit = position_size = 0.0
    bars_held = 0

    for i in range(len(df)):
        row = df.iloc[i]
        price = row['close']

        if in_pos:
            cur_eq = capital + position_size * (price - entry_price)
        else:
            cur_eq = capital
        eq_curve.append(cur_eq)

        if not in_pos:
            if (row['ema21'] > row['ema55'] and price > row['ema200']
                    and RSI_LO <= row['rsi'] <= RSI_HI
                    and row['macd'] > row['macd_sig']
                    and row['adx'] >= ADX_MIN):
                atr_v = row['atr']
                sl = price - ATR_SL * atr_v
                tp = price + ATR_TP * atr_v
                risk = price - sl
                if risk <= 0: continue
                position_size = (capital * RISK_PCT) / risk
                cost = position_size * price * (1 + FEE_RATE + SLIPPAGE)
                if cost > capital: continue
                capital -= cost
                entry_price = price; stop_loss = sl; take_profit = tp
                bars_held = 0; in_pos = True
        else:
            bars_held += 1
            # Breakeven move (V4 only)
            if breakeven and price >= entry_price + 1.5 * row['atr']:
                stop_loss = max(stop_loss, entry_price + 0.2 * row['atr'])  # lock small profit
            elif not breakeven and price > entry_price + row['atr']:
                stop_loss = max(stop_loss, entry_price + 1.0 * row['atr'])

            hit_sl = price <= stop_loss
            hit_tp = price >= take_profit

            time_exit = False
            if exit_mode == 'current':
                time_exit = bars_held >= 6 and row['ema21'] < row['ema55'] and row['rsi'] > 75
            elif exit_mode == 'ema_cross':
                time_exit = bars_held >= 6 and row['ema21'] < row['ema55']
            # 'sl_tp_only': time_exit stays False

            if hit_sl or hit_tp or time_exit:
                pnl = position_size * (price - entry_price)
                proceeds = position_size * price - (FEE_RATE + SLIPPAGE) * price * position_size
                capital += proceeds
                trades.append({
                    'pnl': pnl,
                    'pnl_pct': (price - entry_price) / entry_price * 100,
                    'reason': 'TP' if hit_tp else ('SL' if hit_sl else 'TIME'),
                    'bars': bars_held,
                })
                in_pos = False; bars_held = 0

    if in_pos:
        capital += position_size * df.iloc[-1]['close'] - (FEE_RATE + SLIPPAGE) * df.iloc[-1]['close'] * position_size
    eq_curve.append(capital)

    n = len(trades)
    wins = [t for t in trades if t['pnl'] > 0]
    losses = [t for t in trades if t['pnl'] <= 0]
    wr = (len(wins)/n*100) if n else 0
    gw = sum(t['pnl'] for t in wins); gl = abs(sum(t['pnl'] for t in losses))
    pf = gw/gl if gl > 0 else float('inf') if gw > 0 else 0
    eq = np.array(eq_curve)
    rets = np.diff(eq) / eq[:-1]
    sharpe = (rets.mean()/rets.std() * np.sqrt(365*6)) if rets.std() > 0 else 0
    dd = (eq - np.maximum.accumulate(eq)) / np.maximum.accumulate(eq)
    max_dd = abs(dd.min()*100) if len(dd) else 0
    ret_pct = (capital/10_000 - 1) * 100

    time_exits = sum(1 for t in trades if t['reason'] == 'TIME')

    return {
        'label': label, 'n': n, 'wr': wr, 'pf': pf,
        'ret': ret_pct, 'dd': max_dd, 'sharpe': sharpe,
        'time_exits': time_exits,
    }


def print_table(results):
    print('=' * 100)
    print(f'  {"Variant":<48} {"Trades":>7} {"WR":>6} {"PF":>5} {"Return":>9} {"MaxDD":>8} {"Sharpe":>7} {"TimeEx":>6}')
    print('-' * 100)
    for r in results:
        print(f'  {r["label"]:<48} {r["n"]:>7} {r["wr"]:>5.1f}% {r["pf"]:>5.2f} {r["ret"]:>+8.2f}% {r["dd"]:>+7.2f}% {r["sharpe"]:>+6.2f} {r["time_exits"]:>6}')
    print('=' * 100)


def main():
    df = load_data()
    print(f'Loaded {len(df)} candles. Running 4 variants (all with ADX > 25)...\n')
    results = [
        backtest(df, exit_mode='current',    breakeven=False, label='V1 current (broken time-exit)'),
        backtest(df, exit_mode='ema_cross',  breakeven=False, label='V2 EMA-cross exit instead'),
        backtest(df, exit_mode='sl_tp_only', breakeven=False, label='V3 no time-exit (SL/TP only)'),
        backtest(df, exit_mode='ema_cross',  breakeven=True,  label='V4 V2 + breakeven stop @ +1.5 ATR'),
    ]
    print_table(results)
    print()
    best = max(results, key=lambda r: r['ret'])
    print(f'🏆 Best by return:  {best["label"]}  ({best["ret"]:+.2f}%)')
    best_pf = max(results, key=lambda r: r['pf'])
    print(f'📊 Best by PF:      {best_pf["label"]}  (PF {best_pf["pf"]:.2f})')

if __name__ == '__main__':
    main()
