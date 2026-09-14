"""Compile code-size candidates from a previous cold tuning decision."""
import argparse
import json
import hashlib
from pathlib import Path
import subprocess
import sys

import numpy as np

from xgb_latency import compile_model
from .combine import OPTIONS


def deduplicate(manifest):
    """Do not treat identical engine objects as separate tuning opportunities."""
    canonical,aliases={},{}
    entries=[]
    ordered=[manifest['warm_winner'],*manifest['entries']]
    names=set()
    for entry in ordered:
        if entry['name'] in names:
            continue
        names.add(entry['name'])
        if entry['family']=='engine':
            obj=Path(entry['library']).parent/'model.o'
            key=hashlib.sha256(obj.read_bytes()).hexdigest()
            aliases.setdefault(key,[]).append(entry['name'])
            if key in canonical:
                continue
            canonical[key]=entry['name']
        entries.append(entry)
    return dict(manifest,entries=entries,engine_object_aliases=aliases)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('previous','model','calibration','output'):p.add_argument('--'+name,required=True)
    args=p.parse_args()
    previous=json.loads(Path(args.previous).read_text())
    seed=previous['selection']['selected']['mixed_pressure']['engine']
    meta=json.loads((Path(seed['library']).parent/'metadata.json').read_text())
    options={k:meta[k] for k in OPTIONS if k in meta}
    rows=np.load(args.calibration,allow_pickle=False)
    out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=False)
    variants=[('seed',{}),('compact4',dict(compact_leaf_depth=4)),
              ('outlined',dict(machine_outliner=True)),
              ('compact4_outlined',dict(compact_leaf_depth=4,machine_outliner=True)),
              ('compact4_Os',dict(compact_leaf_depth=4,optimization='Os')),
              ('compact4_Oz',dict(compact_leaf_depth=4,optimization='Oz')),
              ('depth1',dict(select_depth=1)),('depth2',dict(select_depth=2)),('depth3',dict(select_depth=3)),
              ('depth2_no_table',dict(select_depth=2,compact_leaf_depth=0))]
    variants += [(f'compact4_block{n}',dict(compact_leaf_depth=4,tree_block_size=n)) for n in (8,32,64)]
    variants += [(f'compact4_cost{n}',dict(compact_leaf_depth=4,select_policy='cost',select_depth=6,select_branch_penalty=n)) for n in (4,8)]
    entries=list(previous['entries']);builds=[]
    for label,changes in variants:
        name='cold_'+label;config=dict(options,**changes)
        print('Building '+name,flush=True)
        lib=compile_model(args.model,out/name,calibration=rows,**config)
        entries.append(dict(name=name,family='engine',library=str(lib),symbol='predict_row',prepared=False))
        metadata=json.loads((lib.parent/'metadata.json').read_text())
        builds.append(dict(name=name,**metadata,sections=subprocess.check_output(['size','-m' if sys.platform=='darwin' else '-A',str(lib)],text=True)))
    (out/'builds.json').write_text(json.dumps(builds,indent=2)+'\n')
    manifest=dict(entries=entries,warm_winner=previous['selection']['warm_winner'],seed=seed,
                  previous_selection=previous['selection'])
    (out/'candidates.json').write_text(json.dumps(deduplicate(manifest),indent=2)+'\n')


if __name__=='__main__':main()
