"""Paired holdout measurements of fixed packed/split/self-loop triples.

No tuning or deployment selection is performed. All profiles, candidates,
references and measurement counts are frozen before generating fresh rows.
"""
import argparse
import json
from pathlib import Path
import platform
import subprocess

import numpy as np
import xgboost as xgb

from xgb_latency import Forest
from .interference import PROFILES, measure, oracle
from .optimize import fingerprint


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ['manifest','model','harness','output']:parser.add_argument('--'+key,required=True)
    parser.add_argument('--seed',type=int,default=33623)
    args=parser.parse_args();manifest=json.loads(Path(args.manifest).read_text())
    if manifest['model_sha256']!=fingerprint(args.model):raise ValueError('Model mismatch')
    out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=False)
    entries={p['name']:list({e['name']:e for e in
        manifest['entries']+manifest['references'][p['name']]}.values()) for p in PROFILES}
    hashes={e['library']:fingerprint(e['library']) for es in entries.values() for e in es}
    sources=['xgb_latency/separated.py','xgb_latency/compiler.py','xgb_latency/interleaved.py',
             'benchmarks/packed_candidates.py','benchmarks/packed_diagnostic.py',
             'benchmarks/interference.py','benchmarks/interference.cc']
    source_hashes={p:fingerprint(p) for p in sources}
    plan=dict(entries=entries,pairs=manifest['pairs'],profiles=PROFILES,seed=args.seed,
        rows=4096,samples=1024,rounds=10,processes=7,manifest_sha256=fingerprint(args.manifest),
        model_sha256=fingerprint(args.model),harness_sha256=fingerprint(args.harness),
        binary_sha256=hashes,source_sha256=source_hashes)
    plan_path=out/'plan.json';plan_path.write_text(json.dumps(plan,indent=2)+'\n');plan_hash=fingerprint(plan_path)
    rng=np.random.default_rng(args.seed)
    rows=rng.normal(size=(plan['rows'],Forest.load(args.model).num_feature)).astype(np.float32)
    rows[rng.random(rows.shape)<.03]=np.nan
    raw=out/'raw.f32';rows.tofile(raw)
    booster=xgb.Booster(model_file=args.model);booster.set_param({'nthread':1})
    expected,feature_hashes={},{}
    for history in sorted({p['history_bytes'] for p in PROFILES}):
        expected[history],feature_hashes[str(history)]=oracle(Path(args.harness),raw,rows,history,booster,out)
    results={}
    for i,profile in enumerate(PROFILES):
        name=profile['name'];runs=[]
        for rep in range(plan['processes']):
            seed=args.seed+200000+i*plan['processes']+rep
            print(f'Packed diagnostic {name} {rep+1}/{plan["processes"]}',flush=True)
            result=measure(Path(args.harness),raw,expected[profile['history_bytes']],rows,entries[name],profile,
                out/f'{name}-{seed}.json',plan['samples'],plan['rounds'],seed,paired=True)
            runs.append(dict(seed=seed,measurement=result))
        results[name]=dict(entries=entries[name],runs=runs)
    assert fingerprint(plan_path)==plan_hash
    assert fingerprint(args.manifest)==plan['manifest_sha256']
    assert fingerprint(args.model)==plan['model_sha256']
    assert fingerprint(args.harness)==plan['harness_sha256']
    for p,h in {**hashes,**source_hashes}.items():assert fingerprint(p)==h
    report=dict(plan=plan,plan_sha256=plan_hash,results=results,raw_sha256=fingerprint(raw),
        feature_hashes=feature_hashes,builds=manifest['builds'],platform=platform.platform(),
        compiler=subprocess.check_output(['clang','--version'],text=True),
        caveats=['Fixed diagnostic; no new winner selection or default changes.',
                 'Fresh synthetic inputs, software cache/predictor interference; no hardware-flush claim.',
                 'Desktop CPU without affinity/frequency isolation; seven processes are comparison units.',
                 'Static spill/reload counts include all assembly, not dynamic PMU measurements.'])
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print('Report '+str(out/'report.json'),flush=True)


if __name__=='__main__':main()
