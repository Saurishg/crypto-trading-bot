#!/usr/bin/env python3
"""
News sentiment scorer for BTC trading.
Fetches latest crypto news via RSS feeds, scores with phi4, returns -1/0/+1.
"""
import sys, json, re, requests
from datetime import datetime, timezone

FEEDS = [
    'https://cointelegraph.com/rss',
    'https://coindesk.com/arc/outboundfeeds/rss/',
    'https://decrypt.co/feed',
]

OLLAMA = 'http://localhost:11434/v1/chat/completions'
MODEL  = 'phi4:latest'


def fetch_headlines(max_items=10) -> list[str]:
    headlines = []
    for url in FEEDS:
        try:
            r = requests.get(url, timeout=8, headers={'User-Agent': 'Mozilla/5.0'})
            # Simple regex parse — no xml lib needed
            titles = re.findall(r'<title><!\[CDATA\[(.*?)\]\]></title>', r.text)
            if not titles:
                titles = re.findall(r'<title>(.*?)</title>', r.text)[1:]  # skip feed title
            headlines += [t.strip() for t in titles[:5] if t.strip()]
        except Exception:
            continue
    return headlines[:max_items]


def score_sentiment(headlines: list[str]) -> dict:
    if not headlines:
        return {'score': 0, 'reason': 'no headlines', 'headlines': []}

    prompt = f"""You are a crypto trading assistant. Score the overall BTC market sentiment from these headlines.

Headlines:
{chr(10).join(f'- {h}' for h in headlines)}

Return JSON only:
{{"score": <-1 bearish | 0 neutral | 1 bullish>, "confidence": <0.0-1.0>, "reason": "<one sentence>"}}"""

    try:
        r = requests.post(OLLAMA, json={
            'model': MODEL,
            'messages': [{'role': 'user', 'content': prompt}],
            'temperature': 0.1,
            'max_tokens': 100,
        }, timeout=30)
        content = r.json()['choices'][0]['message']['content']
        # Strip thinking tags
        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
        # Extract JSON
        m = re.search(r'\{.*\}', content, re.DOTALL)
        if m:
            d = json.loads(m.group())
            return {
                'score':      int(d.get('score', 0)),
                'confidence': float(d.get('confidence', 0.5)),
                'reason':     d.get('reason', ''),
                'headlines':  headlines,
                'timestamp':  datetime.now(timezone.utc).isoformat(),
            }
    except Exception as e:
        pass

    return {'score': 0, 'confidence': 0, 'reason': f'AI error', 'headlines': headlines}


def get_news_signal() -> dict:
    headlines = fetch_headlines()
    return score_sentiment(headlines)


if __name__ == '__main__':
    result = get_news_signal()
    score = result['score']
    emoji = '🟢' if score > 0 else '🔴' if score < 0 else '🟡'
    print(f'{emoji} Sentiment: {score:+d} (confidence: {result["confidence"]:.0%})')
    print(f'   Reason: {result["reason"]}')
    print(f'   Headlines: {len(result["headlines"])} fetched')
