"""Read already-published pilot status without dispatching or restarting workers."""
import argparse
import asyncio
import json
from pathlib import Path
import re
import time


def status(run_id):
    import modal
    if not re.fullmatch(r'solana-[a-z0-9-]{1,60}',run_id): raise ValueError('Invalid run id')
    v=modal.Volume.from_name('fly-paper-lab-state',environment_name='main')
    async def read(name):
        try:
            async with asyncio.timeout(20):
                data=b''
                async for b in v.read_file.aio(f'/solana-live/{run_id}/{name}'):data+=b
            return name,json.loads(data)
        except (FileNotFoundError,modal.exception.NotFoundError):return name,None
    async def collect():
        return dict(await asyncio.gather(*(read(n) for n in
            ('owner.json','started.json','opening.json','selection.json','latest.json','result.json','completed.json','failed.json','next.json'))))
    files=asyncio.run(collect())
    return {'observed_at':time.time(),'run_id':run_id,'files':{k:v for k,v in files.items() if v is not None},'cloud_submissions':0}


def main():
    p=argparse.ArgumentParser();p.add_argument('run_id');p.add_argument('--out');a=p.parse_args()
    result=status(a.run_id)
    if a.out:
        from .core import atomic_json
        atomic_json(Path(a.out),result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
