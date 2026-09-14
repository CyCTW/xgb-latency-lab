"""Build a fixed state-packing diagnostic and verify unchanged controls."""
import argparse
import json
from pathlib import Path

from xgb_latency import compile_model
from .optimize import fingerprint


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ['previous','model','output']:parser.add_argument('--'+key,required=True)
    args=parser.parse_args();previous=json.loads(Path(args.previous).read_text())
    if fingerprint(args.model)!=previous['selection']['model_sha256']:
        raise ValueError('Previous model mismatch')
    out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=False)
    old={e['name']:e for e in previous['entries']}
    entries=[];pairs=[];builds=[]
    # Fixed mechanism test, without selecting widths using evaluation scores.
    for layout in ['aos','soa']:
        for lanes in [8,12,16]:
            group={}
            old_name=f'separate_{layout}_scalar{lanes}_lane'
            old_control=previous['selection']['matched_controls'][old_name]
            for kind,leaf,state in [('packed','separate','packed'),('split','separate','split'),
                                    ('self_loop','self_loop','split')]:
                name=f'{kind}_{layout}_scalar{lanes}_lane'
                print('Building '+name,flush=True)
                lib=compile_model(args.model,out/name,backend='clang',traversal_lanes=lanes,
                    traversal_leaf_layout=leaf,traversal_leaf_state=state,
                    traversal_data_layout=layout,traversal_load_schedule='lane')
                obj_hash=fingerprint(lib.parent/'model.o')
                matched=None
                if kind!='packed':
                    control=old[old_name] if kind=='split' else old_control
                    matched=fingerprint(Path(control['library']).parent/'model.o')
                    if obj_hash!=matched:raise RuntimeError('Control machine code changed: '+name)
                entry=dict(name=name,family='engine',library=str(lib),symbol='predict_row',prepared=False)
                entries.append(entry);group[kind]=name
                asm=(lib.parent/'model.s').read_text()
                builds.append(dict(name=name,metadata=json.loads((lib.parent/'metadata.json').read_text()),
                    object_sha256=obj_hash,previous_object_sha256=matched,
                    static_spills=asm.count('Folded Spill'),static_reloads=asm.count('Folded Reload')))
            pairs.append(group)
    # References are fixed solely by the previous tuning selections.
    references={p:list({e['name']:e for families in choices.values() for e in families.values()}.values())
                for p,choices in previous['selection']['selected'].items()}
    manifest=dict(entries=entries,pairs=pairs,references=references,builds=builds,
        model_sha256=fingerprint(args.model),source_selection=previous['selection'],
        source_selection_sha256=previous['selection_sha256'])
    (out/'candidates.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__=='__main__':main()
