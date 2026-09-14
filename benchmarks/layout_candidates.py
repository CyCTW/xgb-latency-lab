"""Compare data layouts, widths and alignment using the same model and native ABI."""
import argparse
import hashlib
import json
from pathlib import Path
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
    variants=[(layout,lanes,'scalar',16) for layout in ('aos','soa','soa8') for lanes in (4,8,12,16,24,32)]
    variants += [(layout,16,'scalar',64) for layout in ('aos','soa','soa8')]
    variants += [('soa',lanes,'vector',16) for lanes in (16,32)]
    for layout,lanes,mode,alignment in variants:
        name=f'layout_{layout}_{mode}{lanes}_a{alignment}'
        print('Building '+name,flush=True)
        lib=compile_model(args.model,out/name,backend='clang',traversal_lanes=lanes,traversal_mode=mode,
                          traversal_leaf_layout='self_loop',traversal_data_layout=layout,traversal_alignment=alignment,
                          traversal_load_schedule='direct')
        entries.append(dict(name=name,family='engine',library=str(lib),symbol='predict_row',prepared=False))
        builds.append(dict(name=name,**json.loads((lib.parent/'metadata.json').read_text()),
            sections=subprocess.check_output(['size','-m' if sys.platform=='darwin' else '-A',str(lib)],text=True)))
    (out/'builds.json').write_text(json.dumps(builds,indent=2)+'\n')
    manifest=deduplicate(dict(entries=entries,warm_winner=previous['selection']['warm_winner'],
                              previous_selection=previous['selection']))
    # A freshly built AoS control can be identical to an incumbent object.
    # Resolve diagnostic references to the canonical retained entry after deduplication.
    def object_hash(entry):
        return hashlib.sha256((Path(entry['library']).parent/'model.o').read_bytes()).hexdigest()
    canonical={object_hash(e):e for e in manifest['entries'] if e['family']=='engine'}
    references=[choice['engine'] for choice in previous['selection']['selected'].values()]
    references += [next(e for e in entries if e['name']==name) for name in
                   ('layout_aos_scalar16_a16','layout_soa_scalar16_a16','layout_soa8_scalar16_a16','layout_aos_scalar32_a16')]
    references=[canonical[object_hash(e)] for e in references]
    manifest['references']=list({e['name']:e for e in references}.values())
    (out/'candidates.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__=='__main__':main()
