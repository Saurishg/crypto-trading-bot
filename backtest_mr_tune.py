#!/usr/bin/env python3
"""
Mean-reversion parameter tuning.

Testing key hypotheses about why the naive MR strategy lost money:

  H1: TP at BB-mid is too tight. Try BB-opposite (full reversion) and
      BB-mid + 0.5σ (partial).
  H2: ADX < 20 isn't choppy enough. Try ADX < 15.
  H3: Need a BB-SQUEEZE filter (width < threshold) so we only fade in
      genuinely range-bound conditions, not contracting volatility.
  H4: Need RSI-extreme confirmation (RSI < 30 / > 70) on top of StochRSI.
  H5: Tighter %B threshold (only fade extreme tails).
  H6: Trade only LONG MR (BTC has positive drift; shorts fight the tape).

This script grid-searches across these variables. ADX > 25 trend trades
are excluded (we're isolating MR signal quality).
"""
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from itertools import product
import pandas as pd
import numpy as np
import ta

HERE = Path(__file__).parent
CACHE = HERE / "btc_4h_cache.csv"

FEE_RATE = 0.00075
SLIPPAGE = 0.0005
RISK_PCT = 0.015
MAX_POS  = 0.30


def load_data() -> pd.DataFrame:
    df = pd.read_csv(CACHE)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp')
    df['rsi']    = ta.momentum.RSIIndicator(df['close'], 14).rsi()
    df['atr']    = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], 14).average_true_range()
    df['adx']    = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], 14).adx()
    bb = ta.volatility.BollingerBands(df['close'], 20, 2)
    df['bb_upper'] = bb.bollinger_hband()
    df['bb_mid']   = bb.bollinger_mavg()
    df['bb_lower'] = bb.bollinger_lband()
    df['bb_pct']   = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    df['bb_width_pct'] = (df['bb_upper'] - df['bb_lower']) / df['bb_mid'] * 100
    stoch = ta.momentum.StochRSIIndicator(df['close'], 14, 14, 3)
    df['stoch_rsi'] = stoch.stochrsi() * 100
    df['bb_sd']    = (df['bb_upper'] - df['bb_mid']) / 2  # 1 SD
    return df.dropna().copy()


def run_mr(df, *,
           direction: str,                # 'long', 'short', 'both'
           bb_long_thresh: float,         # %B threshold for long entry
           bb_short_thresh: float,        # %B threshold for short entry (1 - long)
           adx_max: float,                # only trade when ADX < this
           bb_width_max: float | None,    # only trade if BB width < this (None = no filter)
           rsi_required: bool,            # require RSI < 30 (long) / > 70 (short)
           stoch_required: bool,          # require StochRSI < 20 (long) / > 80 (short)
           tp_target: str,                # 'mid', 'opposite', 'mid_half_sd'
           sl_atr: float,                 # SL = band ± sl_atr * ATR
           time_exit_bars: int,           # max bars before market exit
           ) -> dict:
    cash = 10_000.0
    eq_curve = []
    trades = []
    pos = None

    for i in range(len(df)):
        row = df.iloc[i]
        price = row['close']

        # MTM
        if pos:
            if pos['side'] == 'long':  eq = cash + pos['size'] * (price - pos['entry'])
            else:                       eq = cash + pos['size'] * (pos['entry'] - price)
        else:
            eq = cash
        eq_curve.append(eq)

        # Exit
        if pos:
            pos['bars'] += 1
            if pos['side'] == 'long':
                hit_sl = price <= pos['sl']
                hit_tp = price >= pos['tp']
            else:
                hit_sl = price >= pos['sl']
                hit_tp = price <= pos['tp']
            time_exit = pos['bars'] >= time_exit_bars

            if hit_sl or hit_tp or time_exit:
                if pos['side'] == 'long':
                    pnl = pos['size'] * (price - pos['entry'])
                else:
                    pnl = pos['size'] * (pos['entry'] - price)
                fees = (FEE_RATE + SLIPPAGE) * price * pos['size']
                cash += pnl - fees
                trades.append({
                    'side': pos['side'],
                    'pnl': pnl - fees,
                    'reason': 'TP' if hit_tp else ('SL' if hit_sl else 'TIME'),
                    'bars': pos['bars'],
                })
                pos = None
                continue

        # Entry
        if pos: continue
        atr = row['atr']
        if pd.isna(atr) or atr <= 0: continue
        if row['adx'] >= adx_max: continue
        if bb_width_max is not None and row['bb_width_pct'] >= bb_width_max: continue
        eq_now = cash

        def _open(side, entry, sl, tp):
            nonlocal cash, pos
            risk = abs(entry - sl)
            reward = abs(tp - entry)
            if risk <= 0 or reward <= 0: return False
            size_risk = (eq_now * RISK_PCT) / risk
            size_cap  = (eq_now * MAX_POS) / entry
            size = min(size_risk, size_cap)
            if size <= 0: return False
            cash -= (FEE_RATE + SLIPPAGE) * entry * size
            pos = {'side': side, 'entry': entry, 'sl': sl, 'tp': tp, 'size': size, 'bars': 0}
            return True

        # Long entry (oversold fade)
        if direction in ('long', 'both'):
            if row['bb_pct'] <= bb_long_thresh and price > row['bb_lower']:
                if rsi_required and row['rsi'] > 30: pass
                elif stoch_required and row['stoch_rsi'] > 20: pass
                else:
                    if tp_target == 'mid':           tp = row['bb_mid']
                    elif tp_target == 'opposite':    tp = row['bb_upper']
                    elif tp_target == 'mid_half_sd': tp = row['bb_mid'] + 0.5 * row['bb_sd']
                    sl = row['bb_lower'] - sl_atr * atr
                    if tp > price > sl:
                        if _open('long', price, sl, tp): continue

        # Short entry (overbought fade)
        if direction in ('short', 'both'):
            if row['bb_pct'] >= bb_short_thresh and price < row['bb_upper']:
                if rsi_required and row['rsi'] < 70: pass
                elif stoch_required and row['stoch_rsi'] < 80: pass
                else:
                    if tp_target == 'mid':           tp = row['bb_mid']
                    elif tp_target == 'opposite':    tp = row['bb_lower']
                    elif tp_target == 'mid_half_sd': tp = row['bb_mid'] - 0.5 * row['bb_sd']
                    sl = row['bb_upper'] + sl_atr * atr
                    if sl > price > tp:
                        if _open('short', price, sl, tp): continue

    if pos:
        last = df.iloc[-1]['close']
        if pos['side'] == 'long':
            pnl = pos['size'] * (last - pos['entry'])
        else:
            pnl = pos['size'] * (pos['entry'] - last)
        cash += pnl - (FEE_RATE + SLIPPAGE) * last * pos['size']
    eq_curve.append(cash)

    n = len(trades)
    if n == 0:
        return {'n': 0, 'wr': 0, 'pf': 0, 'ret': 0, 'sharpe': 0, 'dd': 0,
                'avg_bars': 0, 'reasons': {}}
    wins = [t for t in trades if t['pnl'] > 0]
    losses = [t for t in trades if t['pnl'] <= 0]
    wr = len(wins) / n * 100
    gw = sum(t['pnl'] for t in wins); gl = abs(sum(t['pnl'] for t in losses))
    pf = gw/gl if gl > 0 else (float('inf') if gw > 0 else 0)
    eq_arr = np.array(eq_curve)
    rets = np.diff(eq_arr) / eq_arr[:-1]
    rets = rets[np.isfinite(rets)]
    sharpe = (rets.mean()/rets.std() * np.sqrt(365*6)) if len(rets) and rets.std() > 0 else 0
    dd = (eq_arr - np.maximum.accumulate(eq_arr)) / np.maximum.accumulate(eq_arr)
    max_dd = abs(dd.min() * 100)
    ret = (cash / 10_000 - 1) * 100
    avg_bars = sum(t['bars'] for t in trades) / n
    reasons = {}
    for t in trades: reasons[t['reason']] = reasons.get(t['reason'], 0) + 1

    return {'n': n, 'wr': wr, 'pf': pf, 'ret': ret, 'sharpe': sharpe,
            'dd': max_dd, 'avg_bars': avg_bars, 'reasons': reasons}


def main():
    df = load_data()
    print(f'Loaded {len(df)} candles\n')

    # Grid:
    directions      = ['long', 'short', 'both']
    bb_thresholds   = [0.05, 0.10]           # 0.05 = extreme, 0.10 = normal
    adx_maxes       = [15.0, 20.0]
    bb_width_caps   = [None, 4.0, 6.0]       # None = no filter, else max BB width %
    tp_targets      = ['mid', 'opposite', 'mid_half_sd']
    sl_atrs         = [0.5, 1.0]
    extras          = [('none', False, False), ('stoch', False, True), ('rsi', True, False)]

    combos = list(product(directions, bb_thresholds, adx_maxes, bb_width_caps,
                          tp_targets, sl_atrs, extras))
    print(f'Testing {len(combos)} combinations...\n')

    results = []
    for direction, bb_th, adx_max, bb_w, tp, sl, (ex_name, rsi_req, st_req) in combos:
        r = run_mr(df,
                   direction=direction,
                   bb_long_thresh=bb_th,
                   bb_short_thresh=1 - bb_th,
                   adx_max=adx_max,
                   bb_width_max=bb_w,
                   rsi_required=rsi_req,
                   stoch_required=st_req,
                   tp_target=tp,
                   sl_atr=sl,
                   time_exit_bars=12)
        r.update({'dir': direction, 'bb_th': bb_th, 'adx_max': adx_max,
                  'bb_w': bb_w, 'tp': tp, 'sl_atr': sl, 'extra': ex_name})
        results.append(r)

    # Filter: at least 20 trades for statistical confidence
    valid = [r for r in results if r['n'] >= 20]

    # Top by profit factor
    by_pf = sorted(valid, key=lambda r: r['pf'], reverse=True)
    print(f'TOP 12 PROFITABLE MR CONFIGS (≥20 trades, by PF):')
    print('=' * 130)
    print(f'  {"Dir":<5} {"BB%":>5} {"ADX<":>5} {"BBw":>5} {"TP":<10} {"SL":>4} {"Filter":>6}'
          f'  {"Trades":>7} {"WR":>6} {"PF":>5} {"Return":>9} {"DD":>7} {"Sharpe":>7} {"AvgBars":>7}')
    print('-' * 130)
    for r in by_pf[:12]:
        bbw_str = f'{r["bb_w"]:.0f}' if r['bb_w'] else '-'
        print(f'  {r["dir"]:<5} {r["bb_th"]:>5.2f} {r["adx_max"]:>5.0f} {bbw_str:>5} '
              f'{r["tp"]:<10} {r["sl_atr"]:>4.1f} {r["extra"]:>6}  '
              f'{r["n"]:>7} {r["wr"]:>5.1f}% {r["pf"]:>5.2f} {r["ret"]:>+8.2f}% '
              f'{r["dd"]:>+6.2f}% {r["sharpe"]:>+6.2f} {r["avg_bars"]:>7.1f}')
    print('=' * 130)

    profitable = [r for r in valid if r['ret'] > 0]
    print(f'\n{len(profitable)} / {len(valid)} configurations profitable (≥20 trades)')

    if profitable:
        best = max(profitable, key=lambda r: r['ret'])
        print(f'\n💰 Best profitable MR config:')
        print(f'   dir={best["dir"]}, BB%B≤{best["bb_th"]}, ADX<{best["adx_max"]},'
              f' BBwidth<{best["bb_w"]}, TP={best["tp"]}, SL={best["sl_atr"]}ATR, filter={best["extra"]}')
        print(f'   → {best["ret"]:+.2f}% return on {best["n"]} trades  '
              f'(WR {best["wr"]:.1f}%, PF {best["pf"]:.2f}, DD {best["dd"]:.2f}%, Sharpe {best["sharpe"]:.2f})')
        print(f'   Exits: {best["reasons"]}')

    # Hypothesis tests
    print('\n── Hypothesis breakdown ──')
    by_tp = {}
    for tp in ['mid', 'opposite', 'mid_half_sd']:
        v = [r['ret'] for r in valid if r['tp'] == tp]
        by_tp[tp] = (np.mean(v), len(v)) if v else (0, 0)
    print('H1 (TP target):', ', '.join(f'{k}: avg ret {v[0]:+.2f}% (n={v[1]})' for k, v in by_tp.items()))

    for adx in [15, 20]:
        v = [r['ret'] for r in valid if r['adx_max'] == adx]
        print(f'H2 (ADX<{adx}): avg ret {np.mean(v):+.2f}% (n={len(v)})')

    for bbw in [None, 4.0, 6.0]:
        v = [r['ret'] for r in valid if r['bb_w'] == bbw]
        label = 'no filter' if bbw is None else f'width<{bbw}'
        print(f'H3 (BB squeeze {label}): avg ret {np.mean(v):+.2f}% (n={len(v)})')

    for ex_name in ['none', 'stoch', 'rsi']:
        v = [r['ret'] for r in valid if r['extra'] == ex_name]
        print(f'H4 ({ex_name}-confirmation): avg ret {np.mean(v):+.2f}% (n={len(v)})')

    for dirn in ['long', 'short', 'both']:
        v = [r['ret'] for r in valid if r['dir'] == dirn]
        print(f'H6 ({dirn} only):  avg ret {np.mean(v):+.2f}% (n={len(v)})')


if __name__ == '__main__':
    main()
