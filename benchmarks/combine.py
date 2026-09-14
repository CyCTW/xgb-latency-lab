"""Combine a tuning-selected configuration with scheduling, layout and PGO.

The preceding run's tuning decision is the seed, never its evaluation scores.
New evaluation is opened only after the new tuning decision is persisted.
"""
import argparse
import json
from pathlib import Path
import platform
import shutil
import sys

import xgboost as xgb

from xgb_latency import Forest, compile_model
from .optimize import choose, fingerprint, load_rows, measure, validate
from .pgo import build_pgo

OPTIONS = ('backend', 'select_depth', 'tree_block_size', 'select_policy',
           'select_branch_penalty', 'rank_feature_limit', 'rank_strategy', 'rank_bucket_bits',
           'compact_leaf_depth', 'accumulation_batch', 'optimization', 'machine_outliner',
           'hybrid_depth', 'hybrid_max_probability', 'hybrid_layout',
           'traversal_lanes', 'traversal_mode', 'traversal_leaf_layout',
           'traversal_data_layout', 'traversal_alignment', 'traversal_load_schedule',
           'traversal_prefetch', 'traversal_prefetch_distance', 'traversal_prefetch_locality',
           'predicate_hoist_limit', 'leaf_table_bits', 'preload')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('previous', 'model', 'calibration', 'tuning', 'evaluation', 'output'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--seed', type=int, default=14870)
    parser.add_argument('--mode', choices=['combine', 'bucket', 'prefix', 'directory'], default='combine')
    args = parser.parse_args()
    previous_path = Path(args.previous).resolve()
    previous = json.loads(previous_path.read_text())
    model = Path(args.model).resolve()
    paths = [Path(getattr(args, name)).resolve() for name in ('calibration', 'tuning', 'evaluation')]
    if len(set(paths)) != 3:
        raise ValueError('Use separate calibration, tuning, evaluation files')
    if fingerprint(model) != previous['selection']['model_sha256']:
        raise ValueError('Previous run used a different model')
    forest = Forest.load(model)
    calibration, tuning = (load_rows(p, forest.num_feature) for p in paths[:2])
    if fingerprint(paths[0]) == fingerprint(paths[1]):
        raise ValueError('Calibration and tuning data are identical')
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    harness = out/'native'
    shutil.copy2(previous_path.parent/'native', harness)
    booster = xgb.Booster(model_file=str(model)); booster.set_param({'nthread': 1})
    seed_entry = previous['selection']['selected']['engine']
    metadata = json.loads((Path(seed_entry['library']).parent/'metadata.json').read_text())
    options = {key: metadata[key] for key in OPTIONS if key in metadata}
    variants = [('seed', {}), ('eytzinger', dict(rank_strategy='eytzinger')),
                ('batch4', dict(accumulation_batch=4)), ('batch16', dict(accumulation_batch=16)),
                ('eytzinger_batch4', dict(rank_strategy='eytzinger', accumulation_batch=4)),
                ('Os', dict(optimization='Os')), ('Oz', dict(optimization='Oz')),
                ('compact2', dict(compact_leaf_depth=2)), ('compact4', dict(compact_leaf_depth=4))]
    for penalty in (2, 4, 8):
        variants.append((f'cost{penalty}', dict(select_policy='cost', select_depth=6,
                                               select_branch_penalty=penalty)))
    for block in (0, 8, 16, 32, 64):
        variants.append((f'block{block}', dict(tree_block_size=block)))
    prefix = args.mode if args.mode in ('bucket', 'prefix', 'directory') else 'combo'
    if args.mode == 'bucket':
        variants = [('seed', {}), ('rank', dict(rank_strategy='bucket')),
                    ('compact4', dict(rank_strategy='bucket', compact_leaf_depth=4)),
                    ('block8', dict(rank_strategy='bucket', tree_block_size=8)),
                    ('block32', dict(rank_strategy='bucket', tree_block_size=32)),
                    ('rank4', dict(rank_strategy='bucket', rank_feature_limit=4)),
                    ('rank8', dict(rank_strategy='bucket', rank_feature_limit=8)),
                    ('cost2', dict(rank_strategy='bucket', select_policy='cost', select_depth=6, select_branch_penalty=2)),
                    ('cost8', dict(rank_strategy='bucket', select_policy='cost', select_depth=6, select_branch_penalty=8))]
    if args.mode == 'prefix':
        variants = [('seed', {})] + [(f'bits{bits}', dict(rank_strategy='bucket', rank_bucket_bits=bits))
                                     for bits in (8, 10, 12, 14, 16)]
        if options.get('rank_feature_limit', 0) < 32:
            variants += [(f'bits{bits}_rank{limit}', dict(rank_strategy='bucket', rank_bucket_bits=bits, rank_feature_limit=limit))
                         for bits in (12, 14, 16) for limit in (16, 32)]
    if args.mode == 'directory':
        variants = [('seed', {})] + [(f'split{bits}', dict(rank_strategy='bucket_split', rank_bucket_bits=bits))
                                    for bits in (12, 14, 16)]
        variants += [(f'split16_block{block}', dict(rank_strategy='bucket_split', rank_bucket_bits=16, tree_block_size=block))
                     for block in (8, 32)]
    reference_names = previous['selection'].get('reference_names',
        previous['selection'].get('previous_selection', {}).get('reference_names', []))
    entries = [e for e in previous['entries'] if e['family'] != 'engine']
    references = [e for e in previous['entries'] if e['name'] in
                  [seed_entry['name'], *reference_names]]
    entries.extend(references)
    builds, configurations = [], set()
    for label, changes in variants:
        config = dict(options, **changes)
        key = json.dumps(config, sort_keys=True)
        if key in configurations:
            continue
        configurations.add(key)
        name = prefix+'_'+label
        print('Building '+name, flush=True)
        lib = compile_model(model, out/name, calibration=calibration, **config)
        entries.append(dict(name=name, family='engine', library=str(lib), symbol='predict_row', prepared=False))
        builds.append(dict(name=name, **json.loads((lib.parent/'metadata.json').read_text())))
    # Train PGO for the seed (or the new bucket lowering). Evaluation is unopened.
    base_entry = next(e for e in entries if e['name'] == (prefix+'_rank' if args.mode == 'bucket' else prefix+'_seed'))
    native = '-mcpu=native' if platform.machine() in ('arm64','aarch64') else '-march=native'
    command = ['clang', '-'+options.get('optimization','O3'), native, '-fPIC', '-fno-fast-math',
               '-ffp-contract=off', *(['-mllvm','-enable-machine-outliner=always'] if options.get('machine_outliner') else []), '-x', 'ir', str(Path(base_entry['library']).parent/'model.ll'),
               '-dynamiclib' if sys.platform == 'darwin' else '-shared', '-o', base_entry['library']]
    lib, meta = build_pgo(command, base_entry, out/(prefix+'_pgo'), harness, calibration)
    entries.append(dict(base_entry, name=prefix+'_pgo', library=str(lib)))
    builds.append(dict(name=prefix+'_pgo', **meta))
    # Fairness control: nonquantized TL profiles train BOTH public interfaces.
    # Preserve the earlier dense-trained PGO as additional tuning candidates.
    previous_builds = json.loads((previous_path.parent/'builds.json').read_text())
    for build in previous_builds:
        if not build['name'].startswith('tl_') or 'command' not in build:
            continue
        candidates = [e for e in entries if e['library'] == build['command'][-1] and not e['prepared']]
        if not candidates:
            continue
        entry = candidates[0]
        interfaces = sorted({e['prepared'] for e in entries if e['library'] == entry['library']})
        name = build['name']+'_dual_pgo'
        print('Training '+name, flush=True)
        lib, meta = build_pgo(build['command'], entry, out/name, harness, calibration,
                              training_interfaces=interfaces)
        builds.append(dict(name=name, **meta))
        for prepared in interfaces:
            entries.append(dict(name=name+('_prepared' if prepared else '_dense'),
                                library=str(lib), family=entry['family']+'_pgo',
                                symbol='predict' if prepared else 'predict_row', prepared=prepared))
    tuning_errors = validate(booster, tuning, entries)
    timing = dict(samples=8192, rounds=11, seed=args.seed)
    measured = measure(harness, tuning, entries, out/'tuning.json', **timing)
    metric = 'block_median_ns_per_row'
    selected = choose(measured, entries, metric)
    selection = dict(selected=selected, metric=metric, seed=args.seed,
                     model_sha256=fingerprint(model), calibration_sha256=fingerprint(paths[0]),
                     tuning_sha256=fingerprint(paths[1]), seed_configuration=seed_entry,
                     seed_options=options, reference_names=reference_names, mode=args.mode,
                     previous_selection=previous['selection'])
    (out/'selection.json').write_text(json.dumps(selection, indent=2)+'\n')
    evaluation = load_rows(paths[2], forest.num_feature)
    evaluation_hash = fingerprint(paths[2])
    if evaluation_hash in (selection['calibration_sha256'], selection['tuning_sha256']):
        raise ValueError('Evaluation duplicates calibration or tuning')
    final_entries = list({e['name']: e for e in [*selected.values(), *references]}.values())
    errors = validate(booster, evaluation, final_entries)
    timing['seed'] += 1
    final = measure(harness, evaluation, final_entries, out/'evaluation.json', **timing)
    # Three independent native processes, same frozen binaries and evaluation.
    confirmations = []
    for seed in range(args.seed+2, args.seed+5):
        confirmations.append(dict(seed=seed, measurement=measure(harness, evaluation, final_entries,
                             out/f'confirmation-{seed}.json', samples=8192, rounds=11, seed=seed)))
    deploy = out/'selected_model'; deploy.mkdir()
    lib = Path(selected['engine']['library'])
    for filename in (lib.name, 'model.o', 'model.h', 'metadata.json'):
        shutil.copy2(lib.parent/filename, deploy/filename)
    report = dict(selection=selection, evaluation=final, evaluation_sha256=evaluation_hash,
                  entries=entries, tuning_errors=tuning_errors, evaluation_errors=errors,
                  confirmations=confirmations, platform=previous['platform'], compiler=previous['compiler'],
                  versions=previous['versions'], caveats=previous['caveats'],
                  binary_sha256={e['name']:fingerprint(e['library']) for e in final_entries},
                  selected_library=str(deploy/lib.name))
    (out/'builds.json').write_text(json.dumps(builds, indent=2)+'\n')
    (out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    for e in final['engines']:
        print(e['name'], e['block_median_ns_per_row'], flush=True)


if __name__ == '__main__':
    main()
