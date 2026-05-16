#!/usr/bin/env python3
"""
BTC Live Trading Bot — Binance Demo Mode.

Strategy (all components backtest-validated on 4y / 8588 4h candles):
  Entry:   price > EMA200  AND  EMA21 > EMA55  AND  50 ≤ RSI ≤ 70
           AND MACD > signal  AND  ADX ≥ 25
  SL:      2.0 × ATR below entry
  TP:      8.0 × ATR above entry (1:4 R:R)
  Trail:   move SL to entry at +3%; to (price − 1×ATR) at +5%
  Sizing:  risk RISK_PCT of equity per trade, hard-capped at MAX_POSITION_PCT of equity

Backtest (4y BTC 4h, ADX>25 filter):
  V1 baseline (TP6): +1.19%, 89 trades, WR 68.5%, PF 1.29
  V2 winner   (TP8): +17.07%, 65 trades, WR 38.5%, PF 1.45, Sharpe 1.57
"""
import os, sys, time, hmac, hashlib, requests, json
from datetime import datetime, date
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / '.env')

# ── Config ────────────────────────────────────────────────────────────────────
BASE_URL   = 'https://demo-api.binance.com'
API_KEY    = os.getenv('BINANCE_API_KEY')
API_SECRET = os.getenv('BINANCE_API_SECRET')
SYMBOL     = 'BTCUSDT'
INTERVAL   = '4h'

# Strategy parameters (backtest-optimized)
RISK_PCT          = 0.015   # 1.5% account risk per trade
MAX_POSITION_PCT  = 0.30    # never put more than 30% of equity in one position (cuts MaxDD)
ATR_SL            = 2.0
ATR_TP            = 8.0     # raised from 6 → +15.88% backtest improvement
RSI_LO            = 50
RSI_HI            = 70
ADX_MIN           = 25      # backtest: +4.6pp WR, flips return positive
FEE_RATE          = 0.001   # 0.1% per side

# Trailing stop levels (percent gain → SL move)
TRAIL_BE_PCT   = 3.0   # at +3% unrealized, move SL to break-even
TRAIL_ATR_PCT  = 5.0   # at +5% unrealized, trail SL to (price − 1 ATR)

STATE_FILE      = Path(__file__).parent / 'live_state.json'
PNL_LOG         = Path(__file__).parent / 'pnl_log.json'
INDICATORS_FILE = Path(__file__).parent / 'indicators.json'
LOCK_FILE       = Path(__file__).parent / '.bot_running'

sys.path.insert(0, '/home/work/fraqtoos')


# ── Binance API ──────────────────────────────────────────────────────────────

def _sign(params: dict) -> dict:
    params['timestamp'] = int(time.time() * 1000)
    query = '&'.join(f'{k}={v}' for k, v in params.items())
    sig = hmac.new(API_SECRET.encode(), query.encode(), hashlib.sha256).hexdigest()
    params['signature'] = sig
    return params

def _get(path, params=None):
    r = requests.get(BASE_URL + path, params=_sign(dict(params or {})),
                     headers={'X-MBX-APIKEY': API_KEY}, timeout=10)
    return r.json()

def _post(path, params=None):
    r = requests.post(BASE_URL + path, params=_sign(dict(params or {})),
                      headers={'X-MBX-APIKEY': API_KEY}, timeout=10)
    return r.json()

def get_balance() -> float:
    for b in _get('/api/v3/account').get('balances', []):
        if b['asset'] == 'USDT':
            return float(b['free'])
    return 0.0

def get_position() -> dict:
    for b in _get('/api/v3/account').get('balances', []):
        if b['asset'] == 'BTC':
            qty = float(b['free']) + float(b['locked'])
            if qty > 0.0001:
                return {'qty': qty}
    return {}

def get_klines(limit=250) -> list:
    r = requests.get(f'{BASE_URL}/api/v3/klines',
                     params={'symbol': SYMBOL, 'interval': INTERVAL, 'limit': limit}, timeout=10)
    return r.json()

def place_order(side: str, qty: float) -> dict:
    return _post('/api/v3/order', {
        'symbol': SYMBOL, 'side': side,
        'type': 'MARKET', 'quantity': f'{qty:.5f}',
    })


# ── Indicators ────────────────────────────────────────────────────────────────
# Split into TRADING (gates decisions) and DASHBOARD (display-only).
# Only TRADING indicators interfere with logic.

def _ema(data, n):
    k = 2/(n+1); e = data[0]
    for p in data[1:]: e = p*k + e*(1-k)
    return e

def _ema_arr(data, n):
    k = 2/(n+1); out = [data[0]]
    for p in data[1:]: out.append(p*k + out[-1]*(1-k))
    return out

def _atr(h, l, c, n=14):
    trs = [max(h[i]-l[i], abs(h[i]-c[i-1]), abs(l[i]-c[i-1])) for i in range(1, len(h))]
    return sum(trs[-n:]) / n

def _rsi(data, n=14):
    gains  = [max(data[i]-data[i-1], 0) for i in range(1, len(data))]
    losses = [max(data[i-1]-data[i], 0) for i in range(1, len(data))]
    ag = sum(gains[-n:])/n; al = sum(losses[-n:])/n
    return 100 - (100/(1+ag/al)) if al > 0 else 100

def _adx(h, l, c, n=14):
    """Wilder ADX. Returns (adx, +DI, -DI)."""
    pdm, ndm, trs = [], [], []
    for i in range(1, len(h)):
        up   = h[i] - h[i-1]
        dn   = l[i-1] - l[i]
        pdm.append(up if (up > dn and up > 0) else 0.0)
        ndm.append(dn if (dn > up and dn > 0) else 0.0)
        trs.append(max(h[i]-l[i], abs(h[i]-c[i-1]), abs(l[i]-c[i-1])))
    if len(trs) < n*2: return 0.0, 0.0, 0.0
    atr_s = sum(trs[:n]) / n
    pdi_s = sum(pdm[:n]) / n
    ndi_s = sum(ndm[:n]) / n
    dxs = []
    for i in range(n, len(trs)):
        atr_s = (atr_s * (n-1) + trs[i]) / n
        pdi_s = (pdi_s * (n-1) + pdm[i]) / n
        ndi_s = (ndi_s * (n-1) + ndm[i]) / n
        pdi = 100 * pdi_s / atr_s if atr_s else 0
        ndi = 100 * ndi_s / atr_s if atr_s else 0
        dxs.append(100 * abs(pdi-ndi) / (pdi+ndi) if (pdi+ndi) else 0)
    adx = sum(dxs[-n:]) / min(len(dxs), n) if dxs else 0
    return adx, pdi, ndi

def trading_signals(klines: list) -> dict:
    """Indicators that GATE trading decisions. Nothing else."""
    closes = [float(k[4]) for k in klines]
    highs  = [float(k[2]) for k in klines]
    lows   = [float(k[3]) for k in klines]

    fast = _ema_arr(closes, 12); slow = _ema_arr(closes, 26)
    macd_line = [f-s for f,s in zip(fast, slow)]
    macd_sig  = _ema_arr(macd_line, 9)
    adx_v, pdi, ndi = _adx(highs, lows, closes)

    return {
        'price':    closes[-1],
        'ema21':    _ema(closes[-50:],  21),
        'ema55':    _ema(closes[-100:], 55),
        'ema200':   _ema(closes,       200),
        'atr':      _atr(highs, lows, closes),
        'rsi':      _rsi(closes),
        'macd':     macd_line[-1],
        'macd_sig': macd_sig[-1],
        'adx':      adx_v,
        'pdi':      pdi,
        'ndi':      ndi,
    }


def dashboard_extras(klines: list) -> dict:
    """Indicators for the public status page. Do NOT influence trading."""
    closes  = [float(k[4]) for k in klines]
    highs   = [float(k[2]) for k in klines]
    lows    = [float(k[3]) for k in klines]
    volumes = [float(k[5]) for k in klines]

    # Bollinger Bands (20, 2σ)
    n = 20
    recent = closes[-n:]
    mid = sum(recent) / n
    sd  = (sum((p - mid)**2 for p in recent) / n) ** 0.5
    up, lo = mid + 2*sd, mid - 2*sd
    pct_b = (closes[-1] - lo) / (up - lo) if (up - lo) else 0.5

    # Rolling VWAP (last 24 candles ≈ 4 days)
    typ = [(highs[i]+lows[i]+closes[i])/3 for i in range(len(closes))][-24:]
    vols = volumes[-24:]
    vwap_v = sum(t*v for t, v in zip(typ, vols)) / sum(vols) if sum(vols) > 0 else closes[-1]

    # Stochastic RSI (%K)
    rsis = []
    for end in range(15, len(closes) + 1):
        w = closes[end-15:end]
        gains  = [max(w[i]-w[i-1], 0) for i in range(1, len(w))]
        losses = [max(w[i-1]-w[i], 0) for i in range(1, len(w))]
        ag = sum(gains)/14; al = sum(losses)/14
        rsis.append(100 - (100/(1+ag/al)) if al > 0 else 100)
    rmax, rmin = max(rsis[-14:]), min(rsis[-14:])
    stoch = (rsis[-1] - rmin) / (rmax - rmin) * 100 if rmax != rmin else 50.0

    # OBV trend
    obv = 0.0; series = [0.0]
    for i in range(1, len(closes)):
        if closes[i] > closes[i-1]:   obv += volumes[i]
        elif closes[i] < closes[i-1]: obv -= volumes[i]
        series.append(obv)
    if len(series) >= 48:
        r24 = sum(series[-24:]) / 24
        p24 = sum(series[-48:-24]) / 24
        ratio = (r24 - p24) / abs(p24) if p24 != 0 else 0
        obv_lbl = 'bullish' if ratio > 0.05 else 'bearish' if ratio < -0.05 else 'neutral'
    else:
        obv_lbl = 'neutral'

    return {
        'bb_upper': up, 'bb_mid': mid, 'bb_lower': lo,
        'bb_pct_b': pct_b, 'bb_width': (up - lo) / mid * 100,
        'vwap': vwap_v,
        'stoch_rsi': stoch,
        'obv': obv, 'obv_trend': obv_lbl,
    }


# ── State, P&L, locks ────────────────────────────────────────────────────────

def load_state() -> dict:
    return json.loads(STATE_FILE.read_text()) if STATE_FILE.exists() else {}

def save_state(state: dict):
    STATE_FILE.write_text(json.dumps(state, indent=2))

def log_trade(action: str, price: float, qty: float, pnl: float = 0, reason: str = ''):
    log = json.loads(PNL_LOG.read_text()) if PNL_LOG.exists() else []
    log.append({
        'ts': datetime.now().isoformat(), 'action': action,
        'price': price, 'qty': qty, 'pnl': round(pnl, 2),
        'pnl_after_fees': round(pnl - 2 * FEE_RATE * price * qty, 2),
        'reason': reason,
    })
    PNL_LOG.write_text(json.dumps(log, indent=2))

def get_daily_pnl() -> float:
    if not PNL_LOG.exists(): return 0.0
    today = date.today().isoformat()
    return sum(t.get('pnl_after_fees', 0) for t in json.loads(PNL_LOG.read_text())
               if t['ts'].startswith(today))

def acquire_lock() -> bool:
    if LOCK_FILE.exists():
        if time.time() - LOCK_FILE.stat().st_mtime < 300:
            return False
        LOCK_FILE.unlink()
    LOCK_FILE.write_text(str(os.getpid()))
    return True

def release_lock():
    try: LOCK_FILE.unlink()
    except: pass

def verify_position_on_startup(state: dict, pos: dict) -> dict:
    """If exchange shows BTC but state has no entry, recover from trade history."""
    if not pos or state.get('entry_price'): return state
    try:
        trades = _get('/api/v3/myTrades', {'symbol': SYMBOL, 'limit': 5})
        buys = [t for t in trades if t.get('isBuyer')]
        if buys:
            fill = float(buys[-1]['price'])
            atr_est = fill * 0.015
            state.update({
                'entry_price': fill,
                'stop_loss':   fill - ATR_SL * atr_est,
                'take_profit': fill + ATR_TP * atr_est,
                'qty':         pos['qty'],
                'entered_at':  datetime.now().isoformat(),
                'recovered':   True,
            })
            notify(f'⚠️ Recovered position: {pos["qty"]:.5f} BTC @ ${fill:,.2f}')
    except Exception as e:
        print(f'  Recovery failed: {e}')
    return state


def notify(msg: str):
    try:
        from core.notifier import send
        send(f'🤖 *BTC Bot*\n{msg}')
    except Exception:
        pass
    print(msg)


# ── Main loop ────────────────────────────────────────────────────────────────

def _run():
    print(f'\n[{datetime.now().strftime("%Y-%m-%d %H:%M")}] BTC Live Bot running...')

    klines = get_klines(250)
    if len(klines) < 210:
        print('Not enough candles'); return

    sig   = trading_signals(klines)
    extra = dashboard_extras(klines)

    price, ema21, ema55, ema200 = sig['price'], sig['ema21'], sig['ema55'], sig['ema200']
    atr, rsi, macd, macd_sig    = sig['atr'], sig['rsi'], sig['macd'], sig['macd_sig']
    adx_v                       = sig['adx']

    state = load_state()
    pos   = get_position()
    state = verify_position_on_startup(state, pos)

    print(f'  Price: ${price:,.2f} | RSI: {rsi:.1f} | MACD: {"▲" if macd>macd_sig else "▼"} | ADX: {adx_v:.1f} ({"trending" if adx_v>=ADX_MIN else "chop"})')

    # Write indicators snapshot for the public dashboard
    try:
        INDICATORS_FILE.write_text(json.dumps({
            'ts':        datetime.now().isoformat(),
            **sig, **extra,
            'has_pos':   bool(pos),
            'daily_pnl': get_daily_pnl(),
        }, indent=2))
    except Exception: pass

    # ── In a position: manage exit ──────────────────────────────────────
    if pos:
        entry = state.get('entry_price', price)
        sl    = state.get('stop_loss',   price - ATR_SL * atr)
        tp    = state.get('take_profit', price + ATR_TP * atr)
        upnl_pct = (price - entry) / entry * 100

        # Trailing stop
        if upnl_pct >= TRAIL_BE_PCT:   sl = max(sl, entry)
        if upnl_pct >= TRAIL_ATR_PCT:  sl = max(sl, price - atr)
        state['stop_loss'] = sl

        print(f'  Position: {pos["qty"]:.5f} BTC | Entry: ${entry:,.2f} | SL: ${sl:,.2f} | TP: ${tp:,.2f} | PnL: {upnl_pct:+.2f}%')

        if price <= sl or price >= tp:
            reason = 'TP' if price >= tp else 'SL'
            result = place_order('SELL', pos['qty'])
            if result.get('status') != 'FILLED':
                notify(f'⚠️ SELL order failed: {result}')
                save_state(state)
                return
            pnl     = (price - entry) * pos['qty']
            pnl_net = pnl - 2 * FEE_RATE * price * pos['qty']
            log_trade('SELL', price, pos['qty'], pnl, reason)
            notify(f'SELL {pos["qty"]:.5f} BTC @ ${price:,.2f} | Net P&L: ${pnl_net:+,.2f} | {reason}')
            save_state({})
        else:
            save_state(state)
        return

    # ── No position: scan for entry ─────────────────────────────────────
    cond_macro  = price > ema200
    cond_trend  = ema21 > ema55
    cond_rsi    = RSI_LO <= rsi <= RSI_HI
    cond_macd   = macd > macd_sig
    cond_adx    = adx_v >= ADX_MIN
    entry_ok    = cond_macro and cond_trend and cond_rsi and cond_macd and cond_adx

    if not entry_ok:
        blocked = []
        if not cond_macro:  blocked.append('price<EMA200')
        if not cond_trend:  blocked.append('EMA bearish')
        if not cond_rsi:    blocked.append(f'RSI={rsi:.0f}')
        if not cond_macd:   blocked.append('MACD bearish')
        if not cond_adx:    blocked.append(f'ADX={adx_v:.0f}<{ADX_MIN}')
        print(f'  No position | ❌ Waiting: {", ".join(blocked)}')
        return

    # Sizing: risk RISK_PCT per trade, hard-cap at MAX_POSITION_PCT of equity
    usdt     = get_balance()
    sl_price = price - ATR_SL * atr
    tp_price = price + ATR_TP * atr
    risk_per_unit = price - sl_price

    qty_by_risk = (usdt * RISK_PCT) / risk_per_unit
    qty_by_cap  = (usdt * MAX_POSITION_PCT) / price
    qty = round(min(qty_by_risk, qty_by_cap), 5)

    if qty < 0.0001 or usdt < 10:
        print(f'  Skipped — qty {qty} too small or usdt {usdt} too low')
        return

    result = place_order('BUY', qty)
    if result.get('status') != 'FILLED':
        print(f'  Order failed: {result}')
        return

    fill = float(result.get('fills', [{}])[0].get('price', price))
    fee  = 2 * FEE_RATE * fill * qty
    save_state({
        'entry_price': fill, 'stop_loss': sl_price, 'take_profit': tp_price,
        'qty': qty, 'entered_at': datetime.now().isoformat(),
    })
    log_trade('BUY', fill, qty, -fee, 'entry')
    notify(f'BUY {qty:.5f} BTC @ ${fill:,.2f} | SL ${sl_price:,.2f} | TP ${tp_price:,.2f} | Fee ${fee:.2f}')


def run():
    if not acquire_lock():
        print('Another instance running — skipping'); return
    try:
        _run()
    except Exception as e:
        notify(f'🚨 Bot crashed: {e}')
        raise
    finally:
        release_lock()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--loop', action='store_true', help='Run continuously every 5 min')
    args = parser.parse_args()

    if args.loop:
        print('Running in loop mode (5 min cycle)...')
        while True:
            run()
            time.sleep(300)
    else:
        run()
