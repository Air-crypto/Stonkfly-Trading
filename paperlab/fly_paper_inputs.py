"""Seal the registered future price and timestamp-correct news inputs."""
from dataclasses import asdict
import json
from pathlib import Path
import sqlite3

import numpy as np

from .core import Tick,atomic_json,digest
from .fly_market_study import quote_at,signature,validate as validate_parent
from .fly_paper_protocol import validate_registration
from .multi import MIN_CONTEXT,read_archive
from .news import News


class SnapshotNews(News):
    def __init__(self,path):
        self.db=sqlite3.connect(f'file:{Path(path).resolve()}?mode=ro',uri=True)
        self.enabled=True


class SealedNews:
    label='sealed_timestamped'
    def __init__(self,values):self.values=values
    def features(self,now):
        value=np.asarray(self.values[str(float(now))],dtype=np.float32)
        if value.shape!=(50,) or not np.isfinite(value).all():raise ValueError('Invalid sealed news vector')
        return value.copy()


def news_stamps(registration,series):
    stamps=set()
    for raw in series.values():
        ticks=[Tick(**t) for t in raw]
        for phase in ('development_start','test_start'):
            for i in range(4):stamps.add(quote_at(ticks,registration[phase]+300*i)[1].ts)
    return sorted(stamps)


def validate(envelope):
    if set(envelope)!={'plan','sha256'} or signature(envelope['plan'])!=envelope['sha256']:
        raise ValueError('Sealed paper comparison hash differs')
    p=envelope['plan'];r=validate_registration(p['registration'],p['training_audit'])
    if set(p['series'])!=set(r['cohort']) or p['snapshot_end']<r['end']:
        raise ValueError('Incomplete cohort or snapshot')
    for key,raw in p['series'].items():
        if not 100<=len(raw)<=512:raise ValueError('Retain bounded visual history')
        ticks=[Tick(**t) for t in raw]
        if any(t.product!=key or t.ts>r['end'] or type(t.available) is not bool for t in ticks) or any(a.ts>=b.ts for a,b in zip(ticks,ticks[1:])):
            raise ValueError('Mixed, unordered or future price inputs')
        if sum(t.available for t in ticks if t.ts<=r['development_start'])<MIN_CONTEXT:
            raise ValueError('Cohort admission must precede development')
    if set(p['news_features'])!={str(float(t)) for t in news_stamps(r,p['series'])}:
        raise ValueError('News vectors do not match all selected quote timestamps')
    news=SealedNews(p['news_features'])
    for stamp in news_stamps(r,p['series']):news.features(stamp)
    for key in ('snapshot_sha256','news_snapshot_sha256'):
        value=p.get(key)
        if not isinstance(value,str) or len(value)!=64 or any(c not in '0123456789abcdef' for c in value):raise ValueError('Missing snapshot hash')
    return p


def seal(archive,news_archive,registration,training_audit,parent,output):
    if Path(output).exists():raise ValueError('Refuse to overwrite a sealed paper comparison')
    r=json.loads(Path(registration).read_text());a=json.loads(Path(training_audit).read_text());validate_registration(r,a)
    if digest(training_audit)!=r['training_audit_sha256']:raise ValueError('Training audit file differs from registration')
    old=json.loads(Path(parent).read_text())
    if r['study']=='10':
        validate(old);parent_cohort=old['plan']['registration']['cohort']
    else:
        validate_parent(old['plan']);parent_cohort=old['plan']['cohort']
    if signature(old['plan'])!=old['sha256'] or old['sha256']!=r['parent_plan_sha256'] or parent_cohort!=r['cohort']:
        raise ValueError('Original cohort or parent plan differs')
    db=sqlite3.connect(f'file:{Path(archive).resolve()}?mode=ro',uri=True)
    try:last=db.execute("SELECT max(json_extract(payload,'$.observed')) FROM observations").fetchone()[0]
    finally:db.close()
    if last is None or last<r['end']:raise ValueError('Snapshot has not reached the registered endpoint')
    _,raw_series=read_archive(archive,r['end']);series={}
    for key in r['cohort']:
        if key not in raw_series:raise ValueError('Keep missing cohort members; never replace with survivors')
        ticks=raw_series[key];past=[t for t in ticks if t.ts<=r['development_start']];future=[t for t in ticks if t.ts>r['development_start']]
        if len(future)>412:raise ValueError('Future observations exceed the fixed context budget')
        series[key]=[asdict(t) for t in past[-(512-len(future)):]+future]
    news=SnapshotNews(news_archive)
    try:vectors={str(float(t)):news.features(t).tolist() for t in news_stamps(r,series)}
    finally:news.db.close()
    p={'registration':r,'training_audit':a,'series':series,'news_features':vectors,
       'snapshot_sha256':digest(archive),'snapshot_end':last,'news_snapshot_sha256':digest(news_archive)}
    envelope={'plan':p,'sha256':signature(p)};validate(envelope);atomic_json(output,envelope);return envelope


def audit_news(envelope,archive):
    p=validate(envelope)
    if digest(archive)!=p['news_snapshot_sha256']:raise ValueError('News snapshot hash differs')
    news=SnapshotNews(archive);sealed=SealedNews(p['news_features'])
    try:
        for t in news_stamps(p['registration'],p['series']):
            if not np.array_equal(news.features(t),sealed.features(t)):
                raise ValueError('Sealed news differs from timestamp-eligible archived revisions')
    finally:news.db.close()
    return {'news_snapshot_sha256':p['news_snapshot_sha256'],'vectors_reconstructed':len(p['news_features'])}
