"""Seal the registered future price and timestamp-correct news inputs."""
from dataclasses import asdict
import json
from pathlib import Path
import sqlite3

import numpy as np

from .core import Tick,atomic_json,digest
from .fly_market_study import quote_at,signature
from .fly_rate_protocol import validate_registration
from .multi import MIN_CONTEXT,read_archive
from .news import News

REFERENCE_FILES = {'audit.json':'training_audit_sha256', 'cloud-result.json':'capture_result_sha256',
    'selection.json':'cohort_selection_sha256', 'array-audit.json':'checkpoint_array_audit_sha256',
    'parent-report.json':'parent_report_sha256', 'mechanism-audit.json':'mechanism_audit_sha256'}


def require_seed():
    if hash('fly-news-seed-probe') != 7584921261715552910:
        raise ValueError('Start a fresh process with PYTHONHASHSEED=0 before sealing, news audit or simulation')


def verify_reference(r, audit, root):
    """Verify the completed parent, mechanism and new checkpoint before input sealing."""
    root=Path(root);validate_registration(r,audit)
    hashes={name:digest(root/name) for name in REFERENCE_FILES}
    if hashes!={name:r[field] for name,field in REFERENCE_FILES.items()}:
        raise ValueError('Pinned rate-study reference files differ')
    documents={name:json.loads((root/name).read_text()) for name in REFERENCE_FILES}
    captured=documents['cloud-result.json'];selection=documents['selection.json'];arrays=documents['array-audit.json']
    if (documents['audit.json']!=audit or captured.get('status')!='rate_checkpoint_exported'
            or captured['training_audit']!=audit or captured['selection']!=selection
            or captured['cohort']!=r['cohort'] or selection['cohort']!=r['cohort']
            or selection['policy']!=r['cohort_policy'] or selection['selected_at']>audit['captured_at']
            or arrays.get('status')!='rate_checkpoint_arrays_audited'
            or arrays['training_audit_sha256']!=hashes['audit.json']
            or arrays['cloud_result_sha256']!=hashes['cloud-result.json']
            or arrays['call_id']!=captured['call_id'] or arrays['cohort_selection_verified']!=r['cohort']):
        raise ValueError('Checkpoint was not captured and independently audited for this cohort')
    for i,key in enumerate(r['cohort']):
        file=f'pool{i}-memory.npz'
        if digest(root/file)!=r['source_memories'][key]['memory_file_sha256']:
            raise ValueError('Pinned exported memory file differs')
        if captured['artifact_sha256']['export/'+file]!=digest(root/file):
            raise ValueError('Memory does not belong to the completed cloud call')
    parent=documents['parent-report.json'];mechanism=documents['mechanism-audit.json']
    if (parent.get('status')!='study_11_complete_independently_audited' or parent['conditions']!=16
            or parent['registration']['end']!=r['parent_end'] or parent['policy_promoted'] is not False
            or not all(parent['verification'].values())
            or mechanism.get('status')!='learning_scale_audited'
            or not mechanism['verification']['all_weights_exact']):
        raise ValueError('Required completed financial and mechanism evidence differs')
    return hashes


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
            for i in range(registration['phase_steps']+1):stamps.add(quote_at(ticks,registration[phase]+300*i)[1].ts)
    return sorted(stamps)


def validate(envelope):
    if set(envelope)!={'plan','sha256'} or signature(envelope['plan'])!=envelope['sha256']:
        raise ValueError('Sealed paper comparison hash differs')
    p=envelope['plan'];r=validate_registration(p['registration'],p['training_audit'])
    if p.get('reference_sha256')!={name:r[field] for name,field in REFERENCE_FILES.items()}:
        raise ValueError('Sealed reference provenance differs')
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


def seal(archive,news_archive,registration,training_audit,evidence_root,output):
    require_seed()
    if Path(output).exists():raise ValueError('Refuse to overwrite a sealed paper comparison')
    r=json.loads(Path(registration).read_text());a=json.loads(Path(training_audit).read_text());validate_registration(r,a)
    if digest(training_audit)!=r['training_audit_sha256']:raise ValueError('Training audit file differs from registration')
    reference=verify_reference(r,a,evidence_root)
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
       'reference_sha256':reference,
       'snapshot_sha256':digest(archive),'snapshot_end':last,'news_snapshot_sha256':digest(news_archive)}
    envelope={'plan':p,'sha256':signature(p)};validate(envelope);atomic_json(output,envelope);return envelope


def audit_news(envelope,archive):
    require_seed()
    p=validate(envelope)
    if digest(archive)!=p['news_snapshot_sha256']:raise ValueError('News snapshot hash differs')
    news=SnapshotNews(archive);sealed=SealedNews(p['news_features'])
    try:
        for t in news_stamps(p['registration'],p['series']):
            if not np.array_equal(news.features(t),sealed.features(t)):
                raise ValueError('Sealed news differs from timestamp-eligible archived revisions')
    finally:news.db.close()
    return {'news_snapshot_sha256':p['news_snapshot_sha256'],'vectors_reconstructed':len(p['news_features'])}
