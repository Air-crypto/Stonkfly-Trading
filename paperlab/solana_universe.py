"""All observed Pump/PumpSwap markets, with explicit discovery/price coverage.

No token allowlist or popularity, age, flow, or mayhem filter. Public streaming
and batched account resolution cannot prove historical or gap-free completeness.
Raw unknown-pool events are retained; metadata is applied only when received.
"""
import asyncio
import base64
from collections import deque
import json
import math
import os
from pathlib import Path
import time

from .solana_events import Feed, Reader, SOL, ZERO, TOKEN, AMM_PROGRAM

UNIVERSE_PROTOCOL='all_observed_pump_v1'
QUOTE_PROTOCOL='solana_all_observed_quotes_v3'
USDC='EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v'
POOL_SCHEMA=json.loads((Path(__file__).parent/'solana_schema/pump_pool_account.json').read_text())


def pool_metadata(address, account, received, slot):
    if not account or account['owner']!=AMM_PROGRAM or account.get('executable'):
        raise ValueError('Unverified PumpSwap pool owner')
    raw=base64.b64decode(account['data'][0],validate=True)
    if raw[:8]!=bytes(POOL_SCHEMA['account']['discriminator']):
        raise ValueError('Unknown pool discriminator')
    reader=Reader(raw[8:]);fields={}
    # Identity prefix is stable across appended pool fields. No reserve snapshot
    # is injected into historical trades; fresh stream events supply reserves.
    for field in POOL_SCHEMA['type']['type']['fields'][:5]:
        fields[field['name']]=reader.read(field['type'])
    return dict(kind='PoolMetadata',pool=address,base_mint=fields['base_mint'],
        quote_mint=fields['quote_mint'],received=received,timestamp=int(received),slot=slot,
        signature='metadata:'+address+':'+str(received),log_index=0,
        program=AMM_PROGRAM,discovery_source='verified_rpc_pool_account',creation_time_known=False)


class AllObservedFeed(Feed):
    all_observed=True

    def __init__(self,root,*,rpc=None,max_archive_bytes=2*1024**3):
        super().__init__(root,max_events=None,max_tokens=None)
        self.rpc=rpc or os.environ.get('PAPERLAB_SOLANA_HTTP_RPC','https://api.mainnet-beta.solana.com')
        self.endpoint=os.environ.get('PAPERLAB_SOLANA_WS_RPC','wss://api.mainnet-beta.solana.com')
        self.max_archive_bytes=max_archive_bytes
        self.pool_info={};self.unresolved={};self.ready=deque();self.resolver=None
        self.stats.update(universe_protocol=UNIVERSE_PROTOCOL,all_tokens_guaranteed=False,
            discovery='All received program events; existing curves on trade, existing pools via account lookup',
            pool_metadata_resolved=0,pool_metadata_failures=0,unresolved_pools=0,
            unresolved_pool_trade_events=0,unknown_curve_discoveries=0,
            resource_limit_bytes=max_archive_bytes)

    def _ensure(self,mint,event,quote,*,known=False):
        if mint not in self.tokens:
            created=dict(kind='CreateEvent',mint=mint,quote_mint=quote,token_program=TOKEN,
                is_mayhem_mode=event.get('is_mayhem_mode',event.get('mayhem_mode',False)),
                timestamp=event['timestamp'],received=event['received'],
                creation_time_known=known,discovery_source=event.get('discovery_source','observed_program_trade'))
            super().accept(created)

    def accept(self,event,*,event_cursor=None):
        with self.lock:
            kind=event['kind']
            if kind in ('CreatePoolEvent','PoolMetadata'):
                self.pool_info[event['pool']]=dict(event)
                self.pools[event['pool']]=event['base_mint']
                self._ensure(event['base_mint'],event,event['quote_mint'])
                self.unresolved.pop(event['pool'],None)
                self.stats['unresolved_pools']=len(self.unresolved)
                if kind=='PoolMetadata':self.stats['pool_metadata_resolved']+=1
                # Consume the archived cursor even though Feed ignores this kind.
                super().accept(event,event_cursor=event_cursor)
                return
            if kind=='CreateEvent':
                event={**event,'creation_time_known':True,'discovery_source':'observed_create_event'}
            if kind=='TradeEvent':
                if event['mint'] not in self.tokens:self.stats['unknown_curve_discoveries']+=1
                self._ensure(event['mint'],event,event.get('quote_mint',ZERO))
                event={**event,'venue':'pump'}
            if kind in ('BuyEvent','SellEvent'):
                info=self.pool_info.get(event['pool'])
                # Canonical SOL pools already identified from a real create event.
                if info is None and event['pool'] in self.pools:
                    info=dict(base_mint=self.pools[event['pool']],quote_mint=SOL)
                if info is None:
                    self.unresolved.setdefault(event['pool'],0.)
                    self.stats['unresolved_pools']=len(self.unresolved)
                    self.stats['unresolved_pool_trade_events']+=1
                    if event_cursor is not None:self.stats['event_cursor']=event_cursor
                    return
                quote=info['quote_mint'];mint=info['base_mint']
                self._ensure(mint,event,quote)
                event={**event,'kind':'TradeEvent','mint':mint,'venue':'pumpswap',
                    'quote_mint':quote,'mayhem_mode':info.get('is_mayhem_mode',False),
                    'is_buy':kind=='BuyEvent','sol_amount':event.get('quote_amount_in',event.get('quote_amount_out',0)),
                    'virtual_sol_reserves':event['pool_quote_token_reserves']+event.get('virtual_quote_reserves',0),
                    'virtual_token_reserves':event['pool_base_token_reserves'],
                    'real_sol_reserves':event['pool_quote_token_reserves']}
            super().accept(event,event_cursor=event_cursor)

    async def prepare_events(self,events,now):
        if self.resolver is None:self.resolver=asyncio.create_task(self._resolve())
        ready=[]
        while self.ready:ready.append(self.ready.popleft())
        return ready+events

    def _lookup(self,addresses):
        import requests
        r=requests.post(self.rpc,json=dict(jsonrpc='2.0',id=1,method='getMultipleAccounts',
            params=[addresses,dict(encoding='base64',commitment='confirmed')]),timeout=8)
        r.raise_for_status();data=r.json()
        if 'error' in data:raise ValueError('RPC account lookup failed')
        return data['result']

    async def _resolve(self):
        while not self.stop.is_set():
            now=time.time()
            with self.lock:addresses=[p for p,attempt in self.unresolved.items() if now-attempt>=60][:100]
            if addresses:
                with self.lock:
                    for p in addresses:self.unresolved[p]=now
                try:
                    result=await asyncio.to_thread(self._lookup,addresses);received=time.time()
                    if len(result['value'])!=len(addresses):raise ValueError('RPC result length differs')
                    for address,account in zip(addresses,result['value']):
                        try:self.ready.append(pool_metadata(address,account,received,result['context']['slot']))
                        except (ValueError,KeyError,TypeError):
                            with self.lock:self.stats['pool_metadata_failures']+=1
                except Exception:
                    with self.lock:self.stats['pool_metadata_failures']+=len(addresses)
                    # Do not retry aggressively after provider limits or failures.
                    await asyncio.sleep(10)
            await asyncio.sleep(2)

    async def finish_metadata(self):
        if self.resolver:
            self.resolver.cancel()
            try:await self.resolver
            except asyncio.CancelledError:pass

    def resource_exhausted(self):
        size=sum(p.stat().st_size for p in self.root.glob('events.db*'))
        if size>=self.max_archive_bytes:
            self.stats['coverage_stop_reason']='raw_archive_byte_limit'
            return True
        return False


def raw_reserves(trade):
    quote=trade.get('quote_mint',ZERO)
    if trade.get('venue')=='pumpswap' or quote in (SOL,ZERO):
        return trade['virtual_sol_reserves'],trade['real_sol_reserves'],trade['sol_amount']
    return trade.get('virtual_quote_reserves',0),trade.get('real_quote_reserves',0),trade.get('quote_amount',0)


def prepare_snapshots(snapshots,now,sol_usd,fx_seen,quote_usd=None):
    """Convert reserve units through fresh observed prices, never assume USDC=$1.

    Unknown quote assets remain visible but unpriced. Reachable quote currencies
    may use an observed path to SOL/USDC; all paths require fresh trade receipts.
    The model's flow/liquidity inputs are consistently normalized to SOL units.
    """
    if not math.isfinite(sol_usd) or sol_usd<=0 or not 0<=now-fx_seen<=90:return snapshots
    rates={SOL:sol_usd/1e9,ZERO:sol_usd/1e9}
    usdc=(quote_usd or {}).get(USDC,0)
    if math.isfinite(usdc) and usdc>0:rates[USDC]=usdc/1e6
    pending=dict(snapshots)
    while pending:
        progress=False
        for mint,token in list(pending.items()):
            if not token['trades']:del pending[mint];continue
            t=token['trades'][-1];q=t.get('quote_mint',ZERO);virtual,real,_=raw_reserves(t)
            if q not in rates:continue
            del pending[mint]
            if not 0<=now-t['received']<=10 or not -2<=now-t['timestamp']<=20:continue
            if virtual<=0 or real<0 or t['virtual_token_reserves']<=0:continue
            value=virtual/t['virtual_token_reserves']*rates[q]
            if math.isfinite(value) and value>0 and mint not in rates:
                rates[mint]=value;progress=True
        if not progress:break
    result={}
    for mint,token in snapshots.items():
        trades=[]
        for t in token['trades']:
            quote=t.get('quote_mint',ZERO);rate=rates.get(quote)
            if rate is None:trades.append(t);continue
            virtual,real,amount=raw_reserves(t);factor=rate/sol_usd*1e9
            trades.append({**t,'original_quote_mint':quote,'quote_unit_usd':rate,
                'quote_mint':SOL,'virtual_sol_reserves':virtual*factor,
                'real_sol_reserves':real*factor,'sol_amount':amount*factor})
        last_quote=trades[-1].get('quote_mint') if trades else token['created']['quote_mint']
        result[mint]={**token,'created':{**token['created'],'quote_mint':last_quote},'trades':trades}
    return result


def update_all_active(active,watched,retired,inactive,portfolio,snapshots,quotes,now,**kwargs):
    """All observable markets enter the queue. No permanent retirement/8-token cap."""
    if not kwargs.get('connected',True):return [],[]
    removed=[];admitted=[]
    for mint in list(active):
        if mint in quotes:inactive.pop(mint,None)
        else:inactive.setdefault(mint,now)
        if mint in inactive and now-inactive[mint]>=120:
            active.pop(mint);removed.append(mint)
    for mint in quotes:
        if mint not in active:
            active[mint]=snapshots[mint]['created'];watched[mint]=active[mint];admitted.append(mint)
    return admitted,removed
