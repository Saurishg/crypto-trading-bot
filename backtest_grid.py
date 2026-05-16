#!/usr/bin/env python3
"""
Grid-search parameter optimization. All variants have ADX > 25.
Searches: ATR_SL × ATR_TP × RSI_LO × volume filter
Picks best by risk-adjusted return (Sharpe × sqrt(trade_count)).
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
    df['vol_sma20'] = df['volume'].rolling(20).mean()
    df['vol_ratio'] = df['volume'] / df['vol_sma20']
    return df.dropna().copy()


def backtest(df, atr_sl, atr_tp, rsi_lo, rsi_hi=70, vol_filter=False) -> dict:
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
            cond = (row['ema21'] > row['ema55'] and price > row['ema200']
                    and rsi_lo <= row['rsi'] <= rsi_hi
                    and row['macd'] > row['macd_sig']
                    and row['adx'] >= ADX_MIN)
            if vol_filter:
                cond = cond and row['vol_ratio'] >= 1.2

            if cond:
                atr_v = row['atr']
                sl = price - atr_sl * atr_v
                tp = price + atr_tp * atr_v
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
            # Trailing stop (same as live bot): BE at +3%, then 1xATR below at +5%
            unrealized_pct = (price - entry_price) / entry_price * 100
            if unrealized_pct > 3: stop_loss = max(stop_loss, entry_price)
            if unrealized_pct > 5: stop_loss = max(stop_loss, price - row['atr'])

            if price <= stop_loss or price >= take_profit:
                pnl = position_size * (price - entry_price)
                proceeds = position_size * price - (FEE_RATE + SLIPPAGE) * price * position_size
                capital += proceeds
                trades.append({'pnl': pnl, 'pnl_pct': (price - entry_price)/entry_price*100,
                              'reason': 'TP' if price >= take_profit else 'SL'})
                in_pos = False; bars_held = 0

    if in_pos:
        capital += position_size * df.iloc[-1]['close'] - (FEE_RATE + SLIPPAGE) * df.iloc[-1]['close'] * position_size
    eq_curve.append(capital)

    n = len(trades)
    wins = [t for t in trades if t['pnl'] > 0]
    wr = (len(wins)/n*100) if n else 0
    gw = sum(t['pnl'] for t in wins); gl = abs(sum(t['pnl'] for t in trades if t['pnl'] <= 0))
    pf = gw/gl if gl > 0 else (float('inf') if gw > 0 else 0)
    eq = np.array(eq_curve); rets = np.diff(eq) / eq[:-1]
    sharpe = (rets.mean()/rets.std() * np.sqrt(365*6)) if rets.std() > 0 else 0
    ret_pct = (capital/10_000 - 1) * 100
    dd = (eq - np.maximum.accumulate(eq)) / np.maximum.accumulate(eq)
    max_dd = abs(dd.min()*100) if len(dd) else 0
    # Risk-adjusted score: prefer setups with more trades (statistical confidence)
    score = ret_pct * (n / 100) if n >= 20 else -999
    return {'n': n, 'wr': wr, 'pf': pf, 'ret': ret_pct,
            'sharpe': sharpe, 'dd': max_dd, 'score': score}


def main():
    df = load_data()
    print(f'Loaded {len(df)} candles\n')

    SL_grid  = [1.5, 2.0, 2.5, 3.0]
    TP_grid  = [3.0, 4.5, 6.0, 8.0]
    RSI_grid = [45, 50, 55]
    VOL_grid = [False, True]

    results = []
    total = len(SL_grid) * len(TP_grid) * len(RSI_grid) * len(VOL_grid)
    cnt = 0
    for sl in SL_grid:
        for tp in TP_grid:
            for rsi_lo in RSI_grid:
                for vol in VOL_grid:
                    cnt += 1
                    r = backtest(df, sl, tp, rsi_lo, 70, vol)
                    r.update({'sl': sl, 'tp': tp, 'rsi_lo': rsi_lo, 'vol': vol})
                    results.append(r)

    print(f'Tested {cnt} combinations\n')

    # Sort by return * sample size (penalize under-traded variants)
    results.sort(key=lambda r: r['score'], reverse=True)
    print('TOP 12 BY RISK-ADJ SCORE (return × trades/100, min 20 trades):')
    print('=' * 105)
    print(f'  {"SL":>4} {"TP":>4} {"R:R":>5} {"RSI":>4} {"Vol":>5}  {"Trades":>7} {"WR":>6} {"PF":>5} {"Return":>9} {"MaxDD":>8} {"Sharpe":>7}')
    print('-' * 105)
    for r in results[:12]:
        rr = r['tp'] / r['sl']
        print(f'  {r["sl"]:>4.1f} {r["tp"]:>4.1f} 1:{rr:>3.1f}  {r["rsi_lo"]:>4} {"yes" if r["vol"] else "no":>5}  '
              f'{r["n"]:>7} {r["wr"]:>5.1f}% {r["pf"]:>5.2f} {r["ret"]:>+8.2f}% {r["dd"]:>+7.2f}% {r["sharpe"]:>+6.2f}')
    print('=' * 105)

    # Best by raw return
    by_ret = sorted(results, key=lambda r: r['ret'], reverse=True)
    print('\nTOP 5 BY RAW RETURN:')
    for r in by_ret[:5]:
        rr = r['tp']/r['sl']
        print(f'  SL={r["sl"]} TP={r["tp"]} R:R=1:{rr:.1f} RSI≥{r["rsi_lo"]} vol={r["vol"]}  '
              f'→ {r["ret"]:+.2f}% on {r["n"]} trades (WR {r["wr"]:.1f}%, PF {r["pf"]:.2f})')

    # Best by PF (with min 30 trades for confidence)
    by_pf = sorted([r for r in results if r['n'] >= 30], key=lambda r: r['pf'], reverse=True)
    print('\nTOP 5 BY PROFIT FACTOR (≥30 trades):')
    for r in by_pf[:5]:
        rr = r['tp']/r['sl']
        print(f'  SL={r["sl"]} TP={r["tp"]} R:R=1:{rr:.1f} RSI≥{r["rsi_lo"]} vol={r["vol"]}  '
              f'→ PF {r["pf"]:.2f}  ({r["ret"]:+.2f}% on {r["n"]} trades)')

    # Best by Sharpe
    by_sh = sorted([r for r in results if r['n'] >= 30], key=lambda r: r['sharpe'], reverse=True)
    print('\nTOP 5 BY SHARPE (≥30 trades):')
    for r in by_sh[:5]:
        rr = r['tp']/r['sl']
        print(f'  SL={r["sl"]} TP={r["tp"]} R:R=1:{rr:.1f} RSI≥{r["rsi_lo"]} vol={r["vol"]}  '
              f'→ Sharpe {r["sharpe"]:.2f}  ({r["ret"]:+.2f}%, PF {r["pf"]:.2f})')

if __name__ == '__main__':
    main()
