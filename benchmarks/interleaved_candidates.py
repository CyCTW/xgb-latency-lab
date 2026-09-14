"""Compile scalar and vector cross-tree traversal with fixed native controls."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from xgb_latency import compile_model
from .cold_candidates import deduplicate
from .optimize import fingerprint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('previous','model','output'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--leaf-layout',choices=['sentinel','self_loop'],default='sentinel')
    args = parser.parse_args()
    previous = json.loads(Path(args.previous).read_text())
    if fingerprint(args.model) != previous['selection']['model_sha256']:
        raise ValueError('Previous report used a different model')
    out = Path(args.output).resolve(); out.mkdir(parents=True,exist_ok=False)
    entries = list(previous['entries'])
    builds = []
    variants = [('scalar',lanes,'O3') for lanes in (1,2,4,8,16)]
    variants += [('vector',lanes,'O3') for lanes in (4,8,16)]
    variants += [('scalar',8,'O2'),('vector',4,'O2')]
    for mode,lanes,optimization in variants:
        prefix='interleaved_self' if args.leaf_layout=='self_loop' else 'interleaved'
        name = f'{prefix}_{mode}{lanes}_{optimization}'
        print('Building '+name,flush=True)
        lib = compile_model(args.model,out/name,backend='clang',traversal_lanes=lanes,
                            traversal_mode=mode,optimization=optimization,traversal_leaf_layout=args.leaf_layout)
        entries.append(dict(name=name,family='engine',library=str(lib),symbol='predict_row',prepared=False))
        builds.append(dict(name=name,**json.loads((lib.parent/'metadata.json').read_text()),
                           sections=subprocess.check_output(['size','-m' if sys.platform=='darwin' else '-A',str(lib)],text=True)))
    (out/'builds.json').write_text(json.dumps(builds,indent=2)+'\n')
    references=list(previous['selection'].get('references',[]))
    # Retain the prior phase's frozen winners without consulting evaluation.
    if args.leaf_layout=='self_loop':
        references += [choice['engine'] for choice in previous['selection']['selected'].values()]
    references += [next(e for e in entries if e['name']==name)
                   for name in (f'{prefix}_scalar1_O3',f'{prefix}_scalar4_O3',f'{prefix}_vector4_O3')]
    references=list({e['name']:e for e in references}.values())
    manifest = dict(entries=entries,warm_winner=previous['selection']['warm_winner'],references=references,
                    previous_selection=previous['selection'])
    (out/'candidates.json').write_text(json.dumps(deduplicate(manifest),indent=2)+'\n')


if __name__=='__main__':main()
