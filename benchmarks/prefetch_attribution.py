"""Fresh-data paired checks for every tuning-selected prefetch configuration.

Selections come exclusively from the frozen main selection.json, never from
main evaluation scores. There is no new winner selection in this diagnostic.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import xgboost as xgb

from xgb_latency import Forest
from .interference import PROFILES, measure, oracle
from .optimize import fingerprint


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ['selection','manifest','builds','harness','model','output']:
        parser.add_argument('--'+key,required=True)
    parser.add_argument('--seed',type=int,default=31423)
    args=parser.parse_args()
    selection=json.loads(Path(args.selection).read_text())
    manifest=json.loads(Path(args.manifest).read_text());builds=json.loads(Path(args.builds).read_text())
    if selection['model_sha256']!=fingerprint(args.model):raise ValueError('Model mismatch')
    def object_hash(e):return fingerprint(Path(e['library']).parent/'model.o')
    canonical={object_hash(e):e for e in manifest['entries'] if e['family']=='engine'}
    cases={}
    for profile,choices in selection['selected'].items():
        pairs={}
        for metric,choice in choices.items():
            e=choice['engine'];meta=json.loads((Path(e['library']).parent/'metadata.json').read_text())
            if meta.get('traversal_prefetch','none')=='none':continue
            keys=['backend','optimization','traversal_lanes','traversal_mode','traversal_leaf_layout',
                  'traversal_data_layout','traversal_load_schedule','traversal_alignment']
            control=next(x for x in builds if x['traversal_prefetch']=='none' and all(x[k]==meta[k] for k in keys))
            obj=Path(args.builds).parent/control['name']/'model.o'
            base=canonical[fingerprint(obj)]
            pair=pairs.setdefault(e['name'],dict(enabled=e,disabled=base,selected_objectives=[]))
            pair['selected_objectives'].append(metric)
        if pairs:cases[profile]=list(pairs.values())
    out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=False)
    plan=dict(source_selection_sha256=fingerprint(args.selection),cases=cases,seed=args.seed,
              rows=4096,samples=1024,rounds=10,processes=7,harness_sha256=fingerprint(args.harness),
              model_sha256=fingerprint(args.model))
    plan_path=out/'plan.json';plan_path.write_text(json.dumps(plan,indent=2)+'\n');plan_hash=fingerprint(plan_path)
    nf=Forest.load(args.model).num_feature
    rng=np.random.default_rng(args.seed);rows=rng.normal(size=(4096,nf)).astype(np.float32)
    rows[rng.random(rows.shape)<.03]=np.nan
    raw=out/'raw.f32';rows.tofile(raw)
    booster=xgb.Booster(model_file=args.model);booster.set_param({'nthread':1})
    expected,feature_hashes={},{}
    for history in sorted({p['history_bytes'] for p in PROFILES if p['name'] in cases}):
        expected[history],feature_hashes[str(history)]=oracle(Path(args.harness),raw,rows,history,booster,out)
    results={};hashes={}
    for i,profile in enumerate(p for p in PROFILES if p['name'] in cases):
        name=profile['name']
        entries=list({e['name']:e for pair in cases[name] for e in [pair['enabled'],pair['disabled']]}.values())
        for e in entries:hashes[e['name']]=fingerprint(e['library'])
        runs=[]
        for rep in range(7):
            seed=args.seed+200000+i*7+rep
            print(f'Attribution {name} {rep+1}/7',flush=True)
            result=measure(Path(args.harness),raw,expected[profile['history_bytes']],rows,entries,profile,
                           out/f'{name}-{seed}.json',1024,10,seed,paired=True)
            runs.append(dict(seed=seed,measurement=result))
        results[name]=dict(entries=entries,runs=runs)
    assert fingerprint(plan_path)==plan_hash
    assert fingerprint(args.selection)==plan['source_selection_sha256']
    for d in results.values():
        for e in d['entries']:assert fingerprint(e['library'])==hashes[e['name']]
    report=dict(plan=plan,plan_sha256=plan_hash,results=results,raw_sha256=fingerprint(raw),
                feature_hashes=feature_hashes,binary_sha256=hashes,source_sha256=fingerprint(__file__),
                caveats=['Diagnostic candidates fixed from tuning selections, not main evaluation.',
                         'Fresh synthetic inputs; no claim of deployment representativeness.',
                         'No new tuning, deployment selection, or post-hoc promotion.'])
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print('Report '+str(out/'report.json'),flush=True)


if __name__=='__main__':main()
