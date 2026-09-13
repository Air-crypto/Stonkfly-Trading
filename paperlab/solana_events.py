"""Read-only confirmed Pump logs. No signer, RPC transaction builder or submission.

Event layouts are pinned to pump-sdk 2.0.0. Unknown/truncated layouts fail closed.
The public RPC stream has no completeness guarantee or historical backfill.
"""
import asyncio
import base64
from collections import deque
import json
from pathlib import Path
import re
import sqlite3
import struct
import threading
import time

SCHEMA = json.loads((Path(__file__).parent/'solana_schema/pump_events.json').read_text())
PROGRAM = SCHEMA['address']
ENDPOINT = 'wss://api.mainnet-beta.solana.com'
SOL = 'So11111111111111111111111111111111111111112'
ZERO = '11111111111111111111111111111111'
TOKEN = 'TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA'
TOKEN22 = 'TokenzQdBNbLqP5VEhdkAS6EPFLC1PHnBqCXEpPxuEb'
ALPHABET = '123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz'


def base58(raw):
    n = int.from_bytes(raw, 'big'); out = ''
    while n:
        n, r = divmod(n, 58); out = ALPHABET[r] + out
    return '1' * (len(raw)-len(raw.lstrip(b'\0'))) + out


class Reader:
    def __init__(self, raw): self.raw, self.pos = raw, 0
    def take(self, n):
        if self.pos+n > len(self.raw): raise ValueError('Truncated event')
        v = self.raw[self.pos:self.pos+n]; self.pos += n; return v
    def read(self, typ):
        if typ == 'pubkey': return base58(self.take(32))
        if typ == 'bool':
            v = self.take(1)[0]
            if v not in (0,1): raise ValueError('Invalid boolean')
            return bool(v)
        if typ in ('u16','u32','u64','i64'):
            fmt = {'u16':'H','u32':'I','u64':'Q','i64':'q'}[typ]
            return struct.unpack('<'+fmt, self.take(struct.calcsize('<'+fmt)))[0]
        if typ == 'string':
            n = self.read('u32')
            if n > 4096: raise ValueError('Oversized event string')
            return self.take(n).decode('utf-8')
        if isinstance(typ,dict) and 'vec' in typ:
            n = self.read('u32')
            if n > 100: raise ValueError('Oversized event vector')
            return [self.read(typ['vec']) for _ in range(n)]
        if isinstance(typ,dict) and 'defined' in typ:
            fields = next(t['type']['fields'] for t in SCHEMA['types'] if t['name']==typ['defined']['name'])
            return {f['name']:self.read(f['type']) for f in fields}
        raise ValueError('Unsupported event type')


def decode(payload):
    raw = base64.b64decode(payload, validate=True)
    event = next((e for e in SCHEMA['events'] if raw[:8]==bytes(e['discriminator'])), None)
    if event is None: return None
    r = Reader(raw[8:]); fields = next(t['type']['fields'] for t in SCHEMA['types'] if t['name']==event['name'])
    data = {f['name']:r.read(f['type']) for f in fields}
    if r.pos != len(r.raw): raise ValueError('Event layout changed')
    return {'kind':event['name'], **data}


def parse_notification(message, received):
    if message.get('method') != 'logsNotification': return [], 0
    result = message['params']['result']; value = result['value']
    if value.get('err') is not None: return [], 0
    stack = []; output = []; errors = 0
    for index, line in enumerate(value['logs']):
        match = re.fullmatch(r'Program (\w+) invoke \[(\d+)\]', line)
        if match:
            depth = int(match[2]); stack = stack[:depth-1]; stack.append(match[1]); continue
        match = re.fullmatch(r'Program (\w+) (?:success|failed:.*)', line)
        if match:
            if stack and stack[-1]==match[1]: stack.pop()
            else: stack=[]
            continue
        if not line.startswith('Program data: ') or not stack or stack[-1]!=PROGRAM: continue
        try:
            event = decode(line[len('Program data: '):])
            if event:
                event.update(signature=value['signature'], slot=result['context']['slot'],
                             log_index=index, received=received)
                output.append(event)
        except (ValueError,UnicodeError,KeyError,struct.error): errors += 1
    return output, errors


def connect_db(path):
    db = sqlite3.connect(path, timeout=10)
    db.execute('PRAGMA journal_mode=WAL')
    db.executescript('''CREATE TABLE IF NOT EXISTS events
      (signature TEXT, log_index INTEGER, received REAL, kind TEXT, mint TEXT, body TEXT,
       PRIMARY KEY(signature,log_index));
      CREATE INDEX IF NOT EXISTS event_mint ON events(mint,received);
      CREATE TABLE IF NOT EXISTS health(at REAL, kind TEXT, detail TEXT);''')
    return db


class Feed:
    """One network reader and DB writer; immutable copies cross the thread boundary."""
    def __init__(self, root, max_events=200000, max_tokens=512):
        self.root = Path(root); self.root.mkdir(parents=True,exist_ok=True)
        self.max_events, self.max_tokens = max_events, max_tokens
        self.stop = threading.Event(); self.lock = threading.Lock()
        self.tokens = {}; self.pinned = None
        self.stats = dict(notifications=0, events=0, duplicates=0, decode_errors=0,
                          connections=0, overflow=0, last_message=0., last_event=0., status='starting')
    def start(self):
        self.thread = threading.Thread(target=self._thread,daemon=True); self.thread.start()
    def close(self):
        self.stop.set(); self.thread.join(timeout=15)
        if self.thread.is_alive(): raise RuntimeError('Collector failed to close')
    def _thread(self):
        try: asyncio.run(self._run())
        except Exception as exc:
            with self.lock: self.stats.update(status='failed',error_type=type(exc).__name__)
    def health(self):
        with self.lock: return dict(self.stats)
    def snapshot(self):
        with self.lock:
            return {k:{**v,'trades':list(v['trades'])} for k,v in self.tokens.items()}
    def accept(self, event):
        mint = event['mint']; kind = event['kind']
        with self.lock:
            if kind=='CreateEvent':
                if mint in self.tokens: return
                if len(self.tokens)>=self.max_tokens:
                    candidates = [k for k in self.tokens if k!=self.pinned]
                    if not candidates: self.stats['overflow']+=1; return
                    del self.tokens[min(candidates,key=lambda k:self.tokens[k]['created']['received'])]
                    self.stats['overflow']+=1
                self.tokens[mint] = {'created':event,'trades':deque(maxlen=256),'complete':False}
            elif mint in self.tokens and kind=='TradeEvent':
                trades = self.tokens[mint]['trades']
                if not trades or (event['slot'],event['timestamp']) >= (trades[-1]['slot'],trades[-1]['timestamp']):
                    trades.append(event)
            elif mint in self.tokens and kind=='CompleteEvent': self.tokens[mint]['complete']=True
    async def _run(self):
        import websockets
        db = connect_db(self.root/'events.db'); retry = 1
        try:
            while not self.stop.is_set():
                try:
                    async with websockets.connect(ENDPOINT,open_timeout=10,close_timeout=2,
                            max_size=4*1024*1024,max_queue=32,ping_interval=15,ping_timeout=15) as ws:
                        await ws.send(json.dumps({'jsonrpc':'2.0','id':1,'method':'logsSubscribe',
                            'params':[{'mentions':[PROGRAM]},{'commitment':'confirmed'}]}))
                        reply = json.loads(await asyncio.wait_for(ws.recv(),10))
                        if not isinstance(reply.get('result'),int): raise ValueError('Subscription rejected')
                        with self.lock:
                            self.stats['connections']+=1; self.stats['status']='connected'
                        db.execute('INSERT INTO health VALUES (?,?,?)',(time.time(),'connected','No backfill; reconnect gaps retained')); db.commit()
                        retry=1
                        while not self.stop.is_set():
                            try: message = json.loads(await asyncio.wait_for(ws.recv(),2))
                            except TimeoutError: continue
                            now=time.time(); events,errors=parse_notification(message,now)
                            with self.lock:
                                self.stats['notifications']+=1; self.stats['last_message']=now
                                self.stats['decode_errors']+=errors
                            for event in events:
                                cursor=db.execute('INSERT OR IGNORE INTO events VALUES (?,?,?,?,?,?)',
                                    (event['signature'],event['log_index'],now,event['kind'],event['mint'],json.dumps(event)))
                                if cursor.rowcount:
                                    self.accept(event)
                                    with self.lock: self.stats['events']+=1; self.stats['last_event']=now
                                else:
                                    with self.lock: self.stats['duplicates']+=1
                            db.commit()
                            if self.stats['events']>=self.max_events:
                                with self.lock: self.stats['status']='capacity_stopped'
                                self.stop.set()
                except Exception as exc:
                    with self.lock: self.stats.update(status='disconnected',error_type=type(exc).__name__)
                    db.execute('INSERT INTO health VALUES (?,?,?)',(time.time(),'disconnected',type(exc).__name__)); db.commit()
                    for _ in range(retry):
                        if self.stop.is_set(): break
                        await asyncio.sleep(1)
                    retry=min(30,retry*2)
        finally:
            db.execute('PRAGMA wal_checkpoint(TRUNCATE)'); db.close()


def eligible(token, now):
    """Only observed native-SOL launches, with fresh two-sided actual events."""
    c=token['created']; trades=token['trades']
    if token['complete']: return 'curve_completed'
    if c['quote_mint'] not in (SOL,ZERO): return 'unsupported_quote'
    if c['token_program'] not in (TOKEN,TOKEN22): return 'unsupported_token_program'
    if c['is_mayhem_mode']: return 'mayhem_mode'
    if not 30 <= now-c['timestamp'] <= 3600: return 'launch_age'
    if abs(c['received']-c['timestamp'])>30: return 'late_creation'
    if len(trades)<6: return 'insufficient_trades'
    last=trades[-1]
    if not 0<=now-last['received']<=10 or not -2<=now-last['timestamp']<=20: return 'stale_trade'
    if last.get('quote_mint') not in (SOL,ZERO): return 'unsupported_trade_quote'
    if last['mayhem_mode']: return 'mayhem_mode'
    recent=[t for t in trades if now-t['received']<=30]
    if sum(t['is_buy'] for t in recent)<2 or sum(not t['is_buy'] for t in recent)<2: return 'one_sided'
    if min(last['virtual_sol_reserves'],last['virtual_token_reserves'])<=0: return 'invalid_reserves'
    if last['real_sol_reserves']<5_000_000_000: return 'thin_real_reserves'
    return None
