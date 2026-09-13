"""Read-only cloud diagnosis of hash-seed initialization; no neural computation."""
import modal
from recovery_cloud import image, volume

app = modal.App('fly-news-diagnostic-11')


@app.function(image=image, volumes={'/state':volume}, cpu=(.125,.125), memory=(512,512),
    max_containers=1, min_containers=0, timeout=60, retries=0)
def probe():
    import json
    import os
    import subprocess
    import sys
    from pathlib import Path
    import numpy as np
    from paperlab.fly_paper_inputs import SnapshotNews, SealedNews
    script = '''
import json,os,sys
from pathlib import Path
import numpy as np
from paperlab.fly_paper_inputs import SnapshotNews,SealedNews
p=json.loads(Path('/state/registered-paper-11/plan.json').read_text())['plan']
n=SnapshotNews('/state/registered-paper-11/news.db');sealed=SealedNews(p['news_features']);diff=[]
try:
 for ts in p['news_features']:
  a=n.features(float(ts));b=sealed.features(float(ts))
  if not np.array_equal(a,b):diff.append({'ts':ts,'channels':np.flatnonzero(a!=b).tolist(),'max_abs':float(np.max(np.abs(a-b)))})
finally:n.db.close()
print(json.dumps({'seed_env':os.environ.get('PYTHONHASHSEED'),'hash_probe':hash('fly-news-seed-probe'),
 'python':sys.version.split()[0],'numpy':np.__version__,'differing_vectors':len(diff),'differences':diff}))
'''
    # Execute exactly the same calculation in the worker interpreter and in a
    # fresh child with the seed in its environment before Python starts.
    import contextlib
    import io
    output=io.StringIO()
    with contextlib.redirect_stdout(output):exec(script, {})
    parent=json.loads(output.getvalue())
    child=subprocess.run([sys.executable,'-c',script],env={**os.environ,'PYTHONHASHSEED':'0'},
        check=True,capture_output=True,text=True,timeout=40)
    return {'parent':parent,'fresh_seed0_child':json.loads(child.stdout),
        'cloud_call_id':modal.current_function_call_id(),'input_id':modal.current_input_id(),
        'new_neural_observations':0,'archive_writes':0}


@app.local_entrypoint()
def main():
    import json
    from pathlib import Path
    path=Path('runs/online-news-repro-11/cloud-process-probe.json')
    if path.exists():raise ValueError('Preserve previous probe evidence')
    result=probe.remote()
    path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
