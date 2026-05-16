#!/usr/bin/env python3
"""
Multi-strategy comparative backtest (margin-style accounting).

Variants:
  V1 — current:           Long-only TREND  (ADX>25)
  V2 — add shorts:        Long + Short TREND  (ADX>25, mirrored)
  V3 — add mean-rev:      Long TREND  +  Long MEAN-REVERSION  (ADX<20, BB extreme)
  V4 — MR only:           Long+Short MR  (no trend)
  V5 — full regime:       Long+Short TREND  +  Long+Short MR

Accounting: equity = cash + unrealized_pnl
  - Entry: cash -= entry_fees only
  - Exit:  cash += realized_pnl - exit_fees
  - Position size: min(RISK_PCT/SL_distance, MAX_POS) × equity
"""
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import pandas as pd
import numpy as np
import ta

HERE = Path(__file__).parent
CACHE = HERE / "btc_4h_cache.csv"

FEE_RATE = 0.00075
SLIPPAGE = 0.0005
RISK_PCT = 0.015
MAX_POS  = 0.30
ADX_TREND = 25.0
ADX_CHOP  = 20.0


def load_data() -> pd.DataFrame:
    df = pd.read_csv(CACHE)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp')
    df['ema21']  = ta.trend.EMAIndicator(df['close'], 21).ema_indicator()
    df['ema55']  = ta.trend.EMAIndicator(df['close'], 55).ema_indicator()
    df['ema200'] = ta.trend.EMAIndicator(df['close'], 200).ema_indicator()
    df['rsi']    = ta.momentum.RSIIndicator(df['close'], 14).rsi()
    macd = ta.trend.MACD(df['close'], 26, 12, 9)
    df['macd']     = macd.macd()
    df['macd_sig'] = macd.macd_signal()
    df['atr']      = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], 14).average_true_range()
    df['adx']      = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], 14).adx()
    bb = ta.volatility.BollingerBands(df['close'], 20, 2)
    df['bb_upper'] = bb.bollinger_hband()
    df['bb_mid']   = bb.bollinger_mavg()
    df['bb_lower'] = bb.bollinger_lband()
    df['bb_pct']   = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    stoch = ta.momentum.StochRSIIndicator(df['close'], 14, 14, 3)
    df['stoch_rsi'] = stoch.stochrsi() * 100
    return df.dropna().copy()


def equity(cash, pos, price):
    if not pos: return cash
    if pos['side'] == 'long':  return cash + pos['size'] * (price - pos['entry'])
    else:                       return cash + pos['size'] * (pos['entry'] - price)


def run_strategy(df, *,
                 allow_trend_long=True, allow_trend_short=False,
                 allow_mr_long=False, allow_mr_short=False) -> dict:
    cash = 10_000.0
    eq_curve = []
    trades = []
    pos = None

    for i in range(len(df)):
        row = df.iloc[i]
        price = row['close']

        eq = equity(cash, pos, price)
        eq_curve.append(eq)

        # ── Exits ────────────────────────────────────────────────────────
        if pos:
            pos['bars'] += 1
            atr = row['atr']

            if pos['side'] == 'long':
                if pos['kind'] == 'trend':
                    upct = (price - pos['entry']) / pos['entry'] * 100
                    if upct >= 3.0: pos['sl'] = max(pos['sl'], pos['entry'])
                    if upct >= 5.0: pos['sl'] = max(pos['sl'], price - atr)
                hit_sl = price <= pos['sl']
                hit_tp = price >= pos['tp']
            else:
                if pos['kind'] == 'trend':
                    upct = (pos['entry'] - price) / pos['entry'] * 100
                    if upct >= 3.0: pos['sl'] = min(pos['sl'], pos['entry'])
                    if upct >= 5.0: pos['sl'] = min(pos['sl'], price + atr)
                hit_sl = price >= pos['sl']
                hit_tp = price <= pos['tp']

            time_exit = pos['kind'] == 'mr' and pos['bars'] >= 12

            if hit_sl or hit_tp or time_exit:
                exit_p = price
                # Realized PnL (long: exit>entry good; short: entry>exit good)
                if pos['side'] == 'long':
                    pnl = pos['size'] * (exit_p - pos['entry'])
                else:
                    pnl = pos['size'] * (pos['entry'] - exit_p)
                exit_fees = (FEE_RATE + SLIPPAGE) * exit_p * pos['size']
                cash += pnl - exit_fees
                trades.append({
                    'kind': pos['kind'], 'side': pos['side'],
                    'entry': pos['entry'], 'exit': exit_p,
                    'pnl': pnl - exit_fees,
                    'pnl_pct': ((exit_p - pos['entry']) / pos['entry'] * 100
                                if pos['side'] == 'long'
                                else (pos['entry'] - exit_p) / pos['entry'] * 100),
                    'reason': 'TP' if hit_tp else ('SL' if hit_sl else 'TIME'),
                    'bars': pos['bars'],
                })
                pos = None
                continue

        # ── Entries (only if flat) ───────────────────────────────────────
        if pos: continue
        atr = row['atr']
        if pd.isna(atr) or atr <= 0: continue
        eq_now = cash  # we're flat

        regime_trend = row['adx'] >= ADX_TREND
        regime_chop  = row['adx'] <  ADX_CHOP

        def open_pos(side, entry, sl, tp, kind):
            nonlocal cash, pos
            risk = abs(entry - sl)
            if risk <= 0: return False
            size_risk = (eq_now * RISK_PCT) / risk
            size_cap  = (eq_now * MAX_POS) / entry
            size = min(size_risk, size_cap)
            if size <= 0: return False
            entry_fees = (FEE_RATE + SLIPPAGE) * entry * size
            cash -= entry_fees
            pos = {'side': side, 'entry': entry, 'sl': sl, 'tp': tp,
                   'size': size, 'kind': kind, 'bars': 0}
            return True

        # TREND LONG
        if regime_trend and allow_trend_long:
            if (row['ema21'] > row['ema55'] and price > row['ema200']
                    and 50 <= row['rsi'] <= 70
                    and row['macd'] > row['macd_sig']):
                if open_pos('long', price, price - 2*atr, price + 8*atr, 'trend'):
                    continue

        # TREND SHORT (mirror)
        if regime_trend and allow_trend_short:
            if (row['ema21'] < row['ema55'] and price < row['ema200']
                    and 30 <= row['rsi'] <= 50
                    and row['macd'] < row['macd_sig']):
                if open_pos('short', price, price + 2*atr, price - 8*atr, 'trend'):
                    continue

        # MR LONG (oversold fade in chop)
        if regime_chop and allow_mr_long:
            if (row['bb_pct'] < 0.10 and row['stoch_rsi'] < 20
                    and row['close'] > row['bb_lower']):
                sl = row['bb_lower'] - 0.5 * atr
                tp = row['bb_mid']
                if tp > price and price > sl:
                    if open_pos('long', price, sl, tp, 'mr'):
                        continue

        # MR SHORT (overbought fade in chop)
        if regime_chop and allow_mr_short:
            if (row['bb_pct'] > 0.90 and row['stoch_rsi'] > 80
                    and row['close'] < row['bb_upper']):
                sl = row['bb_upper'] + 0.5 * atr
                tp = row['bb_mid']
                if price > tp and sl > price:
                    if open_pos('short', price, sl, tp, 'mr'):
                        continue

    # Close any open position
    if pos:
        last = df.iloc[-1]['close']
        if pos['side'] == 'long':
            pnl = pos['size'] * (last - pos['entry'])
        else:
            pnl = pos['size'] * (pos['entry'] - last)
        cash += pnl - (FEE_RATE + SLIPPAGE) * last * pos['size']
    eq_curve.append(cash)

    # Metrics
    n = len(trades)
    wins = [t for t in trades if t['pnl'] > 0]
    losses = [t for t in trades if t['pnl'] <= 0]
    wr = (len(wins) / n * 100) if n else 0
    gw = sum(t['pnl'] for t in wins); gl = abs(sum(t['pnl'] for t in losses))
    pf = gw / gl if gl > 0 else (float('inf') if gw > 0 else 0)
    eq_arr = np.array(eq_curve)
    rets = np.diff(eq_arr) / eq_arr[:-1]
    rets = rets[np.isfinite(rets)]
    sharpe = (rets.mean() / rets.std() * np.sqrt(365 * 6)) if len(rets) and rets.std() > 0 else 0
    dd = (eq_arr - np.maximum.accumulate(eq_arr)) / np.maximum.accumulate(eq_arr)
    max_dd = abs(dd.min() * 100) if len(dd) else 0
    ret = (cash / 10_000 - 1) * 100

    by_kind = {}
    for k in ['trend', 'mr']:
        kts = [t for t in trades if t['kind'] == k]
        if kts:
            kw = [t for t in kts if t['pnl'] > 0]
            by_kind[k] = {'n': len(kts), 'wr': len(kw)/len(kts)*100,
                          'pnl': sum(t['pnl'] for t in kts)}

    return {'n': n, 'wr': wr, 'pf': pf, 'ret': ret, 'sharpe': sharpe,
            'dd': max_dd, 'by_kind': by_kind, 'trades': trades}


def main():
    df = load_data()
    print(f'Loaded {len(df)} candles ({df.index[0]} → {df.index[-1]})\n')

    variants = [
        ('V1 long-only TREND        ', dict(allow_trend_long=True)),
        ('V2 long+short TREND       ', dict(allow_trend_long=True, allow_trend_short=True)),
        ('V3 long TREND + long MR   ', dict(allow_trend_long=True, allow_mr_long=True)),
        ('V4 long+short MR only     ', dict(allow_mr_long=True, allow_mr_short=True)),
        ('V5 FULL regime-adaptive   ', dict(allow_trend_long=True, allow_trend_short=True,
                                            allow_mr_long=True, allow_mr_short=True)),
    ]

    results = []
    for label, params in variants:
        r = run_strategy(df, **params)
        r['label'] = label
        results.append(r)

    print('=' * 110)
    print(f'  {"Variant":<28} {"Trades":>7} {"WR":>6} {"PF":>5} {"Return":>9} {"MaxDD":>9} {"Sharpe":>7}')
    print('-' * 110)
    for r in results:
        print(f'  {r["label"]:<28} {r["n"]:>7} {r["wr"]:>5.1f}% {r["pf"]:>5.2f} '
              f'{r["ret"]:>+8.2f}% {r["dd"]:>+8.2f}% {r["sharpe"]:>+6.2f}')
    print('=' * 110)

    print('\nBreakdown by trade kind:')
    for r in results:
        if r['by_kind']:
            parts = [f'{k}: {v["n"]} trades, WR {v["wr"]:.1f}%, P&L ${v["pnl"]:+,.0f}'
                     for k, v in r['by_kind'].items()]
            print(f'  {r["label"]}: ' + ' | '.join(parts))

    best_ret = max(results, key=lambda r: r['ret'])
    print(f'\n💰 Best return:   {best_ret["label"]}  ({best_ret["ret"]:+.2f}%)')
    best_pf = max([r for r in results if r['n'] >= 30], key=lambda r: r['pf'])
    print(f'📊 Best PF:       {best_pf["label"]}  (PF {best_pf["pf"]:.2f})')
    best_sharpe = max([r for r in results if r['n'] >= 30], key=lambda r: r['sharpe'])
    print(f'🏆 Best Sharpe:   {best_sharpe["label"]}  (Sharpe {best_sharpe["sharpe"]:+.2f})')


if __name__ == '__main__':
    main()
