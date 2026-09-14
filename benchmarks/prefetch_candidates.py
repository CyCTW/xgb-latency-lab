"""Model-aware root lookahead and exact-next-node prefetch candidates."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

from xgb_latency import compile_model
from .cold_candidates import deduplicate
from .optimize import fingerprint


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('previous','model','output'):
        parser.add_argument('--'+name,required=True)
    args=parser.parse_args()
    previous=json.loads(Path(args.previous).read_text())
    if fingerprint(args.model)!=previous['selection']['model_sha256']:
        raise ValueError('Previous report used a different model')
    out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=False)
    entries=list(previous['entries']);builds=[]
    presets=[('aos',12,'lane'),('aos',16,'lane'),('soa',12,'lane'),('soa',16,'direct')]
    variants=[('none',1,3),('roots',1,3),('roots',2,3),('roots',4,3),
              ('next',1,2),('next',1,3),('both',1,3),('both',2,3)]
    controls=[]
    for layout,lanes,schedule in presets:
        for mode,distance,locality in variants:
            name=f'prefetch_{layout}{lanes}_{schedule}_{mode}_d{distance}_l{locality}'
            print('Building '+name,flush=True)
            lib=compile_model(args.model,out/name,backend='clang',traversal_lanes=lanes,
                traversal_leaf_layout='self_loop',traversal_data_layout=layout,traversal_load_schedule=schedule,
                traversal_prefetch=mode,traversal_prefetch_distance=distance,traversal_prefetch_locality=locality)
            entry=dict(name=name,family='engine',library=str(lib),symbol='predict_row',prepared=False)
            entries.append(entry)
            assembly=(lib.parent/'model.s').read_text()
            builds.append(dict(name=name,**json.loads((lib.parent/'metadata.json').read_text()),
                static_prefetch_instructions=len(re.findall(r'^\s*(?:prfm|prfum|prefetch\w*)\s',assembly,re.M)),
                sections=subprocess.check_output(['size','-m' if sys.platform=='darwin' else '-A',str(lib)],text=True)))
            if (layout,lanes,mode,distance,locality) in [('aos',16,'roots',1,3),('aos',16,'next',1,3)]:
                controls.append(entry)
    (out/'builds.json').write_text(json.dumps(builds,indent=2)+'\n')
    # multiobjective keeps the warm winner as the first explicit reference.
    warm=previous['selection']['references'][0]
    manifest=deduplicate(dict(entries=entries,warm_winner=warm,previous_selection=previous['selection']))
    def object_hash(entry):
        return hashlib.sha256((Path(entry['library']).parent/'model.o').read_bytes()).hexdigest()
    canonical={object_hash(e):e for e in manifest['entries'] if e['family']=='engine'}
    references=[*previous['selection']['references'],*controls]
    references += [choice['engine'] for choices in previous['selection']['selected'].values() for choice in choices.values()]
    references=[canonical[object_hash(e)] for e in references]
    manifest['references']=list({e['name']:e for e in references}.values())
    (out/'candidates.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__=='__main__':main()
