#!/usr/bin/env python3
"""
BTC Live Trading Bot — Binance Demo Mode
Fixes: duplicate order guard, daily loss limit, fee accounting,
       P&L log, position verification on startup, news notifications,
       confidence-weighted news scoring.
"""
import os, sys, time, hmac, hashlib, requests, json
from datetime import datetime, date, timedelta
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / '.env')

BASE_URL   = 'https://demo-api.binance.com'
API_KEY    = os.getenv('BINANCE_API_KEY')
API_SECRET = os.getenv('BINANCE_API_SECRET')
SYMBOL     = 'BTCUSDT'
RISK_PCT   = 0.015
ATR_SL     = 2.0
ATR_TP     = 6.0
RSI_LO     = 50
RSI_HI     = 70
FEE_RATE   = 0.001   # 0.1% per side
MAX_DAILY_LOSS_PCT = 0.05  # stop trading if down 5% today

STATE_FILE = Path(__file__).parent / 'live_state.json'
PNL_LOG    = Path(__file__).parent / 'pnl_log.json'
LOCK_FILE  = Path(__file__).parent / '.bot_running'

sys.path.insert(0, '/home/work/fraqtoos')


# ── API helpers ───────────────────────────────────────────────────────────────

def sign(params: dict) -> dict:
    params['timestamp'] = int(time.time() * 1000)
    query = '&'.join(f'{k}={v}' for k, v in params.items())
    sig = hmac.new(API_SECRET.encode(), query.encode(), hashlib.sha256).hexdigest()
    params['signature'] = sig
    return params

def get(path, params={}):
    r = requests.get(BASE_URL + path, params=sign(dict(params)),
                     headers={'X-MBX-APIKEY': API_KEY}, timeout=10)
    return r.json()

def post(path, params={}):
    r = requests.post(BASE_URL + path, params=sign(dict(params)),
                      headers={'X-MBX-APIKEY': API_KEY}, timeout=10)
    return r.json()

def get_balance() -> float:
    acc = get('/api/v3/account')
    for b in acc.get('balances', []):
        if b['asset'] == 'USDT':
            return float(b['free'])
    return 0.0

def get_position() -> dict:
    acc = get('/api/v3/account')
    for b in acc.get('balances', []):
        if b['asset'] == 'BTC':
            qty = float(b['free']) + float(b['locked'])
            if qty > 0.0001:
                return {'qty': qty}
    return {}

def get_klines(limit=250) -> list:
    r = requests.get(f'{BASE_URL}/api/v3/klines',
                     params={'symbol': SYMBOL, 'interval': '4h', 'limit': limit}, timeout=10)
    return r.json()

def place_order(side: str, qty: float) -> dict:
    return post('/api/v3/order', {
        'symbol': SYMBOL, 'side': side,
        'type': 'MARKET', 'quantity': f'{qty:.5f}',
    })


# ── Indicators ────────────────────────────────────────────────────────────────

def calc_indicators(klines: list) -> dict:
    closes = [float(k[4]) for k in klines]
    highs  = [float(k[2]) for k in klines]
    lows   = [float(k[3]) for k in klines]

    def ema(data, n):
        k = 2/(n+1); e = data[0]
        for p in data[1:]: e = p*k + e*(1-k)
        return e

    def ema_arr(data, n):
        k = 2/(n+1); out = [data[0]]
        for p in data[1:]: out.append(p*k + out[-1]*(1-k))
        return out

    def atr(h, l, c, n=14):
        trs = [max(h[i]-l[i], abs(h[i]-c[i-1]), abs(l[i]-c[i-1])) for i in range(1, len(h))]
        return sum(trs[-n:]) / n

    def rsi(data, n=14):
        gains  = [max(data[i]-data[i-1], 0) for i in range(1, len(data))]
        losses = [max(data[i-1]-data[i], 0) for i in range(1, len(data))]
        ag = sum(gains[-n:])/n; al = sum(losses[-n:])/n
        return 100 - (100/(1+ag/al)) if al > 0 else 100

    fast = ema_arr(closes, 12); slow = ema_arr(closes, 26)
    macd_line = [f-s for f,s in zip(fast, slow)]
    macd_sig  = ema_arr(macd_line, 9)

    return {
        'price':    closes[-1],
        'ema21':    ema(closes[-50:],  21),
        'ema55':    ema(closes[-100:], 55),
        'ema200':   ema(closes,       200),
        'atr':      atr(highs, lows, closes),
        'rsi':      rsi(closes),
        'macd':     macd_line[-1],
        'macd_sig': macd_sig[-1],
    }


# ── State & P&L ───────────────────────────────────────────────────────────────

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
    log = json.loads(PNL_LOG.read_text())
    return sum(t['pnl_after_fees'] for t in log if t['ts'].startswith(today) and 'pnl_after_fees' in t)

def verify_position_on_startup(state: dict, pos: dict) -> dict:
    """If we have a BTC position but no state entry_price, recover from exchange."""
    if pos and not state.get('entry_price'):
        # Try to get last fill price from recent trades
        try:
            trades = get('/api/v3/myTrades', {'symbol': SYMBOL, 'limit': 5})
            buys = [t for t in trades if t.get('isBuyer')]
            if buys:
                fill = float(buys[-1]['price'])
                atr_est = float(buys[-1]['price']) * 0.015  # ~1.5% ATR estimate
                state.update({
                    'entry_price': fill,
                    'stop_loss':   fill - ATR_SL * atr_est,
                    'take_profit': fill + ATR_TP * atr_est,
                    'qty':         pos['qty'],
                    'bars_held':   0,
                    'recovered':   True,
                })
                notify(f'⚠️ Recovered position: {pos["qty"]:.5f} BTC @ ${fill:,.2f}')
        except Exception as e:
            print(f'  Position recovery failed: {e}')
    return state


# ── Notifications ─────────────────────────────────────────────────────────────

def notify(msg: str):
    try:
        from core.notifier import send
        send(f'🤖 *BTC Bot*\n{msg}')
    except Exception:
        pass
    print(msg)


# ── Duplicate order guard ─────────────────────────────────────────────────────

def acquire_lock() -> bool:
    if LOCK_FILE.exists():
        age = time.time() - LOCK_FILE.stat().st_mtime
        if age < 300:  # 5 min — another instance is running
            return False
        LOCK_FILE.unlink()  # stale lock
    LOCK_FILE.write_text(str(os.getpid()))
    return True

def release_lock():
    try: LOCK_FILE.unlink()
    except: pass


# ── News sentiment ────────────────────────────────────────────────────────────

def get_news(state: dict) -> tuple[float, dict]:
    """Returns (weighted_score, news_dict). Score is confidence-weighted: -1 to +1."""
    news = state.get('news', {'score': 0, 'confidence': 0.5, 'reason': 'not fetched'})
    last = state.get('last_news_ts', '')
    if not last or (datetime.now() - datetime.fromisoformat(last)).seconds > 14400:
        try:
            from news_sentiment import get_news_signal
            news = get_news_signal()
            state['news'] = news
            state['last_news_ts'] = datetime.now().isoformat()
        except Exception as e:
            print(f'  News fetch failed: {e}')
    # Confidence-weighted score: e.g. score=+1, conf=0.6 → weighted=0.6
    weighted = news.get('score', 0) * news.get('confidence', 0.5)
    return weighted, news


# ── Main ──────────────────────────────────────────────────────────────────────

def run():
    if not acquire_lock():
        print('Another instance running — skipping'); return

    try:
        _run()
    finally:
        release_lock()


def _run():
    print(f'\n[{datetime.now().strftime("%Y-%m-%d %H:%M")}] BTC Live Bot running...')

    # Daily loss circuit breaker
    daily_pnl = get_daily_pnl()
    usdt_start = get_balance()
    if usdt_start > 0 and daily_pnl < -(usdt_start * MAX_DAILY_LOSS_PCT):
        msg = f'🛑 Daily loss limit hit (${daily_pnl:,.2f}). No new trades today.'
        notify(msg); return

    klines = get_klines(250)
    if len(klines) < 210:
        print('Not enough candles'); return

    ind      = calc_indicators(klines)
    price    = ind['price']
    ema21    = ind['ema21']
    ema55    = ind['ema55']
    ema200   = ind['ema200']
    atr      = ind['atr']
    rsi      = ind['rsi']
    macd     = ind['macd']
    macd_sig = ind['macd_sig']

    state = load_state()
    pos   = get_position()
    state = verify_position_on_startup(state, pos)

    news_weighted, news = get_news(state)
    news_emoji = '🟢' if news_weighted > 0.3 else '🔴' if news_weighted < -0.3 else '🟡'

    print(f'  Price: ${price:,.2f} | RSI: {rsi:.1f} | MACD: {"▲" if macd>macd_sig else "▼"} | News: {news_emoji}({news_weighted:+.2f}) | Daily PnL: ${daily_pnl:+,.2f}')

    if pos:
        entry = state.get('entry_price', price)
        sl    = state.get('stop_loss',   price - ATR_SL * atr)
        tp    = state.get('take_profit', price + ATR_TP * atr)
        bars  = state.get('bars_held', 0) + 1
        state['bars_held'] = bars

        unrealized_pct = (price - entry) / entry * 100
        if unrealized_pct > 3:  sl = max(sl, entry);       state['stop_loss'] = sl
        if unrealized_pct > 5:  sl = max(sl, price - atr); state['stop_loss'] = sl

        exit_signal = price <= sl or price >= tp or (bars >= MIN_HOLD and ema21 < ema55)
        print(f'  Position: {pos["qty"]:.5f} BTC | Entry: ${entry:,.2f} | SL: ${sl:,.2f} | TP: ${tp:,.2f} | PnL: {unrealized_pct:+.2f}%')

        if exit_signal:
            reason = 'SL' if price <= sl else 'TP' if price >= tp else 'EMA cross'
            result = place_order('SELL', pos['qty'])
            pnl = (price - entry) * pos['qty']
            pnl_net = pnl - 2 * FEE_RATE * price * pos['qty']
            log_trade('SELL', price, pos['qty'], pnl, reason)
            notify(f'SELL {pos["qty"]:.5f} BTC @ ${price:,.2f} | Net PnL: ${pnl_net:+,.2f} | {reason}')
            save_state({})
        else:
            save_state(state)

    else:
        entry_signal = (
            price > ema200 and ema21 > ema55 and
            RSI_LO <= rsi <= RSI_HI and
            macd > macd_sig and
            news_weighted >= -0.3   # allow neutral/bullish, block strongly bearish
        )

        if not entry_signal:
            blocked_by = []
            if price <= ema200:       blocked_by.append('price<EMA200')
            if ema21 <= ema55:        blocked_by.append('EMA bearish')
            if not (RSI_LO<=rsi<=RSI_HI): blocked_by.append(f'RSI={rsi:.0f}')
            if macd <= macd_sig:      blocked_by.append('MACD bearish')
            if news_weighted < -0.3:  blocked_by.append(f'news={news_weighted:+.2f}')
            print(f'  No position | ❌ Waiting: {", ".join(blocked_by)}')
        else:
            # Confidence-weighted position sizing
            risk = RISK_PCT * (1 + max(0, news_weighted) * 0.3)  # up to +30% on bullish news
            usdt    = get_balance()
            sl_price = price - ATR_SL * atr
            tp_price = price + ATR_TP * atr
            qty = min(usdt * risk / (price - sl_price), usdt * 0.95 / price)
            qty = round(qty, 5)

            if qty > 0.0001 and usdt > 10:
                result = place_order('BUY', qty)
                if result.get('status') == 'FILLED':
                    fill = float(result.get('fills', [{}])[0].get('price', price))
                    fee_cost = 2 * FEE_RATE * fill * qty
                    save_state({
                        'entry_price': fill, 'stop_loss': sl_price,
                        'take_profit': tp_price, 'qty': qty,
                        'bars_held': 0, 'entered_at': datetime.now().isoformat(),
                    })
                    log_trade('BUY', fill, qty, -fee_cost, 'entry')
                    notify(f'BUY {qty:.5f} BTC @ ${fill:,.2f} | SL: ${sl_price:,.2f} | TP: ${tp_price:,.2f} | Fee: ${fee_cost:.2f}')
                else:
                    print(f'  Order failed: {result}')


if __name__ == '__main__':
    run()
