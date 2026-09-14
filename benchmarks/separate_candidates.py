"""Separate leaves from split records, with predeclared equivalent controls."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from xgb_latency import compile_model
from .cold_candidates import deduplicate
from .optimize import fingerprint


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['previous','model','output']:p.add_argument('--'+key,required=True)
    args=p.parse_args();previous=json.loads(Path(args.previous).read_text())
    if fingerprint(args.model)!=previous['selection']['model_sha256']:raise ValueError('Previous model mismatch')
    out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=False)
    entries=list(previous['entries']);builds=[];pairs=[]
    variants=[(layout,n,'scalar','lane') for layout in ('aos','soa','soa8') for n in (4,8,12,16,24,32)]
    variants += [('aos',n,'scalar','staged') for n in (8,12,16)]
    variants += [(layout,n,'vector','lane') for layout in ('aos','soa') for n in (8,16)]
    for layout,n,mode,schedule in variants:
        pair=[]
        for leaf_layout in ['separate','self_loop']:
            name=f'{leaf_layout}_{layout}_{mode}{n}_{schedule}'
            print('Building '+name,flush=True)
            lib=compile_model(args.model,out/name,backend='clang',traversal_lanes=n,traversal_mode=mode,
                traversal_leaf_layout=leaf_layout,traversal_data_layout=layout,traversal_load_schedule=schedule)
            entry=dict(name=name,family='engine',library=str(lib),symbol='predict_row',prepared=False)
            entries.append(entry);pair.append(entry)
            builds.append(dict(name=name,**json.loads((lib.parent/'metadata.json').read_text()),
                object_sha256=fingerprint(lib.parent/'model.o'),
                sections=subprocess.check_output(['size','-m' if sys.platform=='darwin' else '-A',str(lib)],text=True)))
        pairs.append(pair)
    (out/'builds.json').write_text(json.dumps(builds,indent=2)+'\n')
    warm=previous['selection']['references'][0]
    manifest=deduplicate(dict(entries=entries,warm_winner=warm,previous_selection=previous['selection']))
    def object_hash(e):return fingerprint(Path(e['library']).parent/'model.o')
    canonical={object_hash(e):e for e in manifest['entries'] if e['family']=='engine'}
    references=list(previous['selection']['references'])
    references += [c['engine'] for choices in previous['selection']['selected'].values() for c in choices.values()]
    references += [pair[0] for pair in pairs if pair[0]['name'] in
                   ['separate_aos_scalar8_lane','separate_aos_scalar12_lane','separate_soa_scalar8_lane']]
    references=[canonical[object_hash(e)] for e in references]
    manifest['references']=list({e['name']:e for e in references}.values())
    manifest['matched_controls']={canonical[object_hash(a)]['name']:canonical[object_hash(b)] for a,b in pairs}
    (out/'candidates.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__=='__main__':main()
