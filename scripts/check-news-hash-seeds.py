"""Reproduce sealed-news reconstruction across fresh Python hash seeds."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

CHILD = '''
import json,sys
import numpy as np
from paperlab.fly_paper_inputs import SnapshotNews,SealedNews,news_stamps,validate
from pathlib import Path
envelope=json.loads(Path(sys.argv[1]).read_text());p=validate(envelope)
news=SnapshotNews(sys.argv[2]);sealed=SealedNews(p['news_features']);differences=[]
try:
 for stamp in news_stamps(p['registration'],p['series']):
  actual=news.features(stamp);expected=sealed.features(stamp);indices=np.flatnonzero(actual!=expected)
  if len(indices):differences.append({'stamp':stamp,'channels':indices.tolist(),'max_abs':float(np.max(np.abs(actual-expected)))})
finally:news.db.close()
print(json.dumps({'vectors':len(p['news_features']),'differing_vectors':len(differences),
 'max_abs':max((d['max_abs'] for d in differences),default=0),
 'channels':sorted({i for d in differences for i in d['channels']}),'differences':differences,
 'python':sys.version.split()[0],'numpy':np.__version__}))
'''


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('plan','news','out'):parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--seeds',default='0,1,2,3,4,5,6,7')
    args=parser.parse_args();seeds=args.seeds.split(',')
    if not 1<=len(seeds)<=64 or len(set(seeds))!=len(seeds) or any(not s.isdigit() or int(s)>=2**32 for s in seeds):
        raise ValueError('Choose 1-64 distinct integer Python hash seeds')
    if args.out.exists():raise ValueError('Preserve the previous reproduction report')
    root=Path(__file__).resolve().parents[1];envelope=json.loads(args.plan.read_text())
    if sha(args.news)!=envelope['plan']['news_snapshot_sha256']:raise ValueError('Archive is not the sealed news snapshot')
    reports=[]
    for seed in seeds:
        env={**os.environ,'PYTHONHASHSEED':seed,'PYTHONPATH':str(root)}
        result=subprocess.run([sys.executable,'-c',CHILD,str(args.plan.resolve()),str(args.news.resolve())],
            env=env,cwd=root,check=True,capture_output=True,text=True,timeout=60)
        report={'seed':seed,**json.loads(result.stdout)};reports.append(report)
        print(json.dumps({k:v for k,v in report.items() if k!='differences'}),flush=True)
    report={'status':'news_hash_seed_reproduction','plan_sha256':envelope['sha256'],
        'news_snapshot_sha256':sha(args.news),'news_source_sha256':sha(root/'paperlab/news.py'),
        'script_sha256':sha(__file__),'runs':reports,'cloud_submissions':0,'neural_observations':0}
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
