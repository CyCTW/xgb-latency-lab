"""Build mixed code/data candidates from a frozen cold tuning selection."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from xgb_latency import compile_model
from .cold_candidates import deduplicate
from .combine import OPTIONS
from .optimize import fingerprint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('previous', 'model', 'calibration', 'output'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--layout', choices=['wide','compact'], default='wide')
    parser.add_argument('--include-manifest', help='Also retain an earlier compiled candidate set')
    args = parser.parse_args()
    previous = json.loads(Path(args.previous).read_text())
    if fingerprint(args.model) != previous['selection']['model_sha256']:
        raise ValueError('Previous selection used a different model')
    seed = previous['selection']['selected']['mixed_pressure']['engine']
    metadata = json.loads((Path(seed['library']).parent/'metadata.json').read_text())
    options = {key: metadata[key] for key in OPTIONS if key in metadata}
    calibration = np.load(args.calibration, allow_pickle=False)
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    # All previous baselines, plus every frozen profile winner and warm control.
    entries = [e for e in previous['entries'] if e['family'] != 'engine']
    entries += [choice['engine'] for choice in previous['selection']['selected'].values()]
    if args.include_manifest:
        entries += json.loads(Path(args.include_manifest).read_text())['entries']
    variants = [(f'h{height}_p{probability:g}', dict(hybrid_depth=height, hybrid_max_probability=probability))
                for height, probability in [(2,1.),(3,1.),(4,1.),(6,1.),
                                             (2,.1),(3,.1),(4,.1),(4,.02),(4,.05),(3,.25)]]
    variants += [(f'h{height}_p{probability:g}_no_rank',
                  dict(hybrid_depth=height, hybrid_max_probability=probability, rank_feature_limit=0))
                 for height, probability in [(6,1.),(3,1.),(3,.1)]]
    builds = []
    for label, changes in variants:
        name = ('hybrid8_' if args.layout == 'compact' else 'hybrid_') + label
        print('Building ' + name, flush=True)
        lib = compile_model(args.model, out/name, calibration=calibration,
                            **dict(options, **changes, hybrid_layout=args.layout))
        entries.append(dict(name=name, family='engine', library=str(lib), symbol='predict_row', prepared=False))
        builds.append(dict(name=name, **json.loads((lib.parent/'metadata.json').read_text()),
                           sections=subprocess.check_output(['size', '-m' if sys.platform=='darwin' else '-A', str(lib)],text=True)))
    (out/'builds.json').write_text(json.dumps(builds, indent=2)+'\n')
    references=[seed]
    if args.layout=='compact':
        # Predeclared paired representation controls, kept even if neither wins.
        references += [next(e for e in entries if e['name']==name)
                       for name in ('hybrid_h2_p1','hybrid8_h2_p1') if any(e['name']==name for e in entries)]
    manifest = dict(entries=entries, warm_winner=previous['selection']['warm_winner'], references=references, seed=seed,
                    previous_selection=previous['selection'])
    (out/'candidates.json').write_text(json.dumps(deduplicate(manifest), indent=2)+'\n')


if __name__ == '__main__':
    main()
