#!/usr/bin/env python3
"""
BTC Live Trading Bot — Binance Demo Mode
Uses EMA21/55 + EMA200 trend filter strategy from btc_strategy.py
Runs once per hour via cron or orchestrator.
"""
import os, sys, time, hmac, hashlib, requests
from datetime import datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / '.env')

BASE_URL   = 'https://demo-api.binance.com'
API_KEY    = os.getenv('BINANCE_API_KEY')
API_SECRET = os.getenv('BINANCE_API_SECRET')
SYMBOL     = 'BTCUSDT'
RISK_PCT   = 0.015   # 1.5% risk per trade
ATR_SL     = 2.0     # stop loss = 2×ATR
ATR_TP     = 6.0     # take profit = 6×ATR (1:3 R:R)
STATE_FILE = Path(__file__).parent / 'live_state.json'

sys.path.insert(0, '/home/work/fraqtoos')


def sign(params: dict) -> dict:
    params['timestamp'] = int(time.time() * 1000)
    query = '&'.join(f'{k}={v}' for k, v in params.items())
    sig = hmac.new(API_SECRET.encode(), query.encode(), hashlib.sha256).hexdigest()
    params['signature'] = sig
    return params


def get(path: str, params: dict = {}) -> dict:
    r = requests.get(BASE_URL + path, params=sign(params),
                     headers={'X-MBX-APIKEY': API_KEY}, timeout=10)
    return r.json()


def post(path: str, params: dict = {}) -> dict:
    r = requests.post(BASE_URL + path, params=sign(params),
                      headers={'X-MBX-APIKEY': API_KEY}, timeout=10)
    return r.json()


def get_balance() -> float:
    acc = get('/api/v3/account')
    for b in acc.get('balances', []):
        if b['asset'] == 'USDT':
            return float(b['free'])
    return 0.0


def get_position() -> dict:
    """Check if we have an open BTC position."""
    acc = get('/api/v3/account')
    for b in acc.get('balances', []):
        if b['asset'] == 'BTC':
            qty = float(b['free']) + float(b['locked'])
            if qty > 0.0001:
                return {'qty': qty}
    return {}


def get_klines(limit=250) -> list:
    r = requests.get(f'{BASE_URL}/api/v3/klines',
                     params={'symbol': SYMBOL, 'interval': '1h', 'limit': limit}, timeout=10)
    return r.json()


def calc_indicators(klines: list) -> dict:
    closes = [float(k[4]) for k in klines]
    highs  = [float(k[2]) for k in klines]
    lows   = [float(k[3]) for k in klines]

    def ema(data, n):
        k = 2 / (n + 1)
        e = data[0]
        for p in data[1:]:
            e = p * k + e * (1 - k)
        return e

    def atr(h, l, c, n=14):
        trs = [max(h[i]-l[i], abs(h[i]-c[i-1]), abs(l[i]-c[i-1])) for i in range(1, len(h))]
        return sum(trs[-n:]) / n

    def rsi(data, n=14):
        gains = [max(data[i]-data[i-1], 0) for i in range(1, len(data))]
        losses = [max(data[i-1]-data[i], 0) for i in range(1, len(data))]
        ag = sum(gains[-n:]) / n
        al = sum(losses[-n:]) / n
        return 100 - (100 / (1 + ag / al)) if al > 0 else 100

    return {
        'price':  closes[-1],
        'ema21':  ema(closes[-50:],  21),
        'ema55':  ema(closes[-100:], 55),
        'ema200': ema(closes,       200),
        'atr':    atr(highs, lows, closes),
        'rsi':    rsi(closes),
    }


def load_state() -> dict:
    import json
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {}


def save_state(state: dict):
    import json
    STATE_FILE.write_text(json.dumps(state, indent=2))


def notify(msg: str):
    try:
        from core.notifier import send
        send(f'🤖 *BTC Bot*\n{msg}')
    except Exception:
        pass
    print(msg)


def place_order(side: str, qty: float) -> dict:
    qty_str = f'{qty:.5f}'
    return post('/api/v3/order', {
        'symbol':    SYMBOL,
        'side':      side,
        'type':      'MARKET',
        'quantity':  qty_str,
    })


def run():
    print(f'\n[{datetime.now().strftime("%Y-%m-%d %H:%M")}] BTC Live Bot running...')

    klines = get_klines(250)
    if len(klines) < 210:
        print('Not enough candles'); return

    ind = calc_indicators(klines)
    price  = ind['price']
    ema21  = ind['ema21']
    ema55  = ind['ema55']
    ema200 = ind['ema200']
    atr    = ind['atr']
    rsi    = ind['rsi']
    state  = load_state()
    pos    = get_position()

    print(f'  Price: ${price:,.2f} | EMA21: {ema21:,.0f} | EMA55: {ema55:,.0f} | EMA200: {ema200:,.0f} | RSI: {rsi:.1f}')

    if pos:
        # Check exit conditions
        entry  = state.get('entry_price', price)
        sl     = state.get('stop_loss', price - ATR_SL * atr)
        tp     = state.get('take_profit', price + ATR_TP * atr)
        bars   = state.get('bars_held', 0) + 1
        state['bars_held'] = bars

        # Trail stop
        unrealized_pct = (price - entry) / entry * 100
        if unrealized_pct > 3:
            sl = max(sl, entry)
            state['stop_loss'] = sl
        if unrealized_pct > 5:
            sl = max(sl, price - atr)
            state['stop_loss'] = sl

        exit_signal = (
            price <= sl or
            price >= tp or
            (bars >= 24 and ema21 < ema55 and rsi > 75)
        )

        print(f'  Position: {pos["qty"]:.5f} BTC | Entry: ${entry:,.2f} | SL: ${sl:,.2f} | TP: ${tp:,.2f} | PnL: {unrealized_pct:+.2f}%')

        if exit_signal:
            reason = 'SL hit' if price <= sl else 'TP hit' if price >= tp else 'EMA cross exit'
            result = place_order('SELL', pos['qty'])
            pnl = (price - entry) * pos['qty']
            msg = f'SELL {pos["qty"]:.5f} BTC @ ${price:,.2f} | PnL: ${pnl:+,.2f} | Reason: {reason}'
            notify(msg)
            save_state({})
        else:
            save_state(state)
    else:
        # Check entry conditions
        entry_signal = (
            price > ema200 and
            ema21 > ema55 and
            45 <= rsi <= 70
        )

        print(f'  No position | Signal: {"✅ ENTRY" if entry_signal else "❌ wait"}')

        if entry_signal:
            usdt = get_balance()
            sl_price = price - ATR_SL * atr
            tp_price = price + ATR_TP * atr
            risk_amt = usdt * RISK_PCT
            qty = round(risk_amt / (price - sl_price), 5)
            max_qty = round(usdt * 0.95 / price, 5)
            qty = min(qty, max_qty)

            if qty > 0.0001 and usdt > 10:
                result = place_order('BUY', qty)
                if result.get('status') == 'FILLED':
                    fill_price = float(result.get('fills', [{}])[0].get('price', price))
                    save_state({
                        'entry_price':  fill_price,
                        'stop_loss':    sl_price,
                        'take_profit':  tp_price,
                        'qty':          qty,
                        'bars_held':    0,
                        'entered_at':   datetime.now().isoformat(),
                    })
                    msg = f'BUY {qty:.5f} BTC @ ${fill_price:,.2f} | SL: ${sl_price:,.2f} | TP: ${tp_price:,.2f}'
                    notify(msg)
                else:
                    print(f'  Order result: {result}')


if __name__ == '__main__':
    run()
