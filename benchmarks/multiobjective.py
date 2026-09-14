"""Freeze several latency objectives together, using balanced paired rounds.

All objectives share the same measurements and held-out data. A winner for one
metric must never be substituted for another after evaluation.
"""
import argparse
import json
from pathlib import Path
import platform
import shutil
import statistics
import subprocess

import xgboost as xgb

from xgb_latency import Forest
from .interference import PROFILES, build_harness, measure, oracle
from .optimize import choose, fingerprint, load_rows

METRICS=['model_mean_median_ns','model_p99_ns','pipeline_block_median_ns','pipeline_p99_ns']


def unique(entries):
    return list({e['name']:e for e in entries}.values())


def with_controls(entries, controls):
    """Include predeclared controls, including transitive dependencies, once."""
    result=unique(entries);seen={e['name'] for e in result}
    for entry in result:
        control=controls.get(entry['name'])
        if control is not None and control['name'] not in seen:
            result.append(control);seen.add(control['name'])
    return result


def aggregate(runs, entries, metrics):
    return {'engines':[dict(name=e['name'],**{metric:statistics.median(
        next(row[metric] for row in r['engines'] if row['name']==e['name']) for r in runs)
        for metric in metrics}) for e in entries]}


def shortlist(result, entries, references, metrics):
    by_name={e['name']:e for e in result['engines']}
    chosen=list(references)
    for metric in metrics:
        for family in sorted({e['family'] for e in entries}):
            chosen += sorted((e for e in entries if e['family']==family),
                key=lambda e:(by_name[e['name']][metric],e['name']))[:2]
    return unique(chosen)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ['model','candidate_manifest','tuning','evaluation','output']:
        parser.add_argument('--'+key.replace('_','-'),required=True)
    parser.add_argument('--metrics',nargs='+',choices=METRICS,default=METRICS)
    parser.add_argument('--profiles',nargs='+',choices=[p['name'] for p in PROFILES],default=[p['name'] for p in PROFILES])
    for key,default in [('screen_samples',256),('screen_rounds',4),('confirm_samples',512),('confirm_rounds',8),
                        ('confirm_runs',5),('evaluation_samples',1024),('evaluation_rounds',10),('evaluation_runs',7)]:
        parser.add_argument('--'+key.replace('_','-'),type=int,default=default)
    parser.add_argument('--seed',type=int,default=30320)
    args=parser.parse_args()
    counts={key:value for key,value in vars(args).items() if key.endswith(('_samples','_rounds','_runs'))}
    if min(counts.values())<1 or any(v%2 for k,v in counts.items() if k.endswith('_rounds')):
        parser.error('Counts must be positive; paired round counts must be even')
    metrics=list(dict.fromkeys(args.metrics))
    profiles=[p for p in PROFILES if p['name'] in args.profiles]
    model=Path(args.model).resolve();nf=Forest.load(model).num_feature
    tuning_path,evaluation_path=Path(args.tuning).resolve(),Path(args.evaluation).resolve()
    if tuning_path==evaluation_path:parser.error('Use separate tuning and evaluation files')
    manifest=json.loads(Path(args.candidate_manifest).read_text())
    entries=manifest['entries'];names={e['name'] for e in entries}
    if len(names)!=len(entries):raise ValueError('Candidate names must be unique')
    references=unique([manifest['warm_winner'],*manifest.get('references',[])])
    if not all(e in entries for e in references):raise ValueError('Reference missing from candidates')
    controls=manifest.get('matched_controls',{})
    if any(name not in names or entry not in entries for name,entry in controls.items()):
        raise ValueError('Matched control or candidate missing from manifest')
    for e in entries:
        if e['family']=='engine':
            metadata=json.loads((Path(e['library']).parent/'metadata.json').read_text())
            if metadata['source_sha256']!=fingerprint(model):raise ValueError('Candidate model mismatch')
    binary_hashes={e['name']:fingerprint(e['library']) for e in entries}
    tuning=load_rows(tuning_path,nf)
    out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=False)
    harness=build_harness(out)
    booster=xgb.Booster(model_file=str(model));booster.set_param({'nthread':1})

    def prepare(rows, subdir):
        directory=out/subdir;directory.mkdir();raw=directory/'raw.f32';rows.tofile(raw)
        expected,hashes={},{}
        for history in sorted({p['history_bytes'] for p in profiles}):
            expected[history],hashes[str(history)]=oracle(harness,raw,rows,history,booster,directory)
        return directory,raw,expected,hashes

    directory,raw,expected,tuning_hashes=prepare(tuning,'tuning')
    selected,screening,confirmations={},{},{}
    for i,profile in enumerate(profiles):
        name=profile['name'];print('Screening '+name,flush=True)
        screen=measure(harness,raw,expected[profile['history_bytes']],tuning,entries,profile,
                       directory/(name+'-screen.json'),args.screen_samples,args.screen_rounds,args.seed+i,paired=True)
        screening[name]=screen
        short=with_controls(shortlist(screen,entries,references,metrics),controls);runs=[]
        for rep in range(args.confirm_runs):
            seed=args.seed+1000+i*args.confirm_runs+rep
            print(f'Confirming {name} {rep+1}/{args.confirm_runs}: {len(short)} candidates',flush=True)
            runs.append(measure(harness,raw,expected[profile['history_bytes']],tuning,short,profile,
                directory/f'{name}-confirm-{seed}.json',args.confirm_samples,args.confirm_rounds,seed,paired=True))
        scores=aggregate(runs,short,metrics)
        confirmations[name]=dict(entries=short,runs=runs,aggregate=scores)
        selected[name]={metric:choose(scores,short,metric) for metric in metrics}
        print('Selected '+name+': '+json.dumps({m:s['engine']['name'] for m,s in selected[name].items()}),flush=True)
    selection=dict(selected=selected,metrics=metrics,profiles=profiles,counts=counts,seed=args.seed,
        model_sha256=fingerprint(model),tuning_sha256=fingerprint(tuning_path),tuning_feature_hashes=tuning_hashes,
        references=references,matched_controls=controls,paired_rounds=True,candidate_manifest_sha256=fingerprint(args.candidate_manifest),
        binary_sha256=binary_hashes)
    selection_path=out/'selection.json';selection_path.write_text(json.dumps(selection,indent=2)+'\n')
    frozen_hash=fingerprint(selection_path)
    # No holdout read (even a fingerprint) until every profile/objective is frozen.
    evaluation=load_rows(evaluation_path,nf)
    if fingerprint(evaluation_path)==selection['tuning_sha256']:raise ValueError('Evaluation duplicates tuning')
    directory,raw,expected,evaluation_hashes=prepare(evaluation,'evaluation')
    results={}
    for i,profile in enumerate(profiles):
        name=profile['name'];final=with_controls([*references,*[e for choice in selected[name].values() for e in choice.values()]],controls)
        runs=[]
        for rep in range(args.evaluation_runs):
            seed=args.seed+100000+i*args.evaluation_runs+rep
            print(f'Evaluating {name} {rep+1}/{args.evaluation_runs}: {len(final)} candidates',flush=True)
            result=measure(harness,raw,expected[profile['history_bytes']],evaluation,final,profile,
                directory/f'{name}-{seed}.json',args.evaluation_samples,args.evaluation_rounds,seed,paired=True)
            runs.append(dict(seed=seed,measurement=result))
        results[name]=dict(entries=final,runs=runs)
    if fingerprint(selection_path)!=frozen_hash:raise RuntimeError('Frozen selection changed')
    if any(fingerprint(e['library'])!=binary_hashes[e['name']] for e in entries):
        raise RuntimeError('Candidate binary changed during measurement')
    deployed={}
    for profile,choices in selected.items():
        deployed[profile]={}
        for metric,choice in choices.items():
            chosen=Path(choice['engine']['library']);directory=out/'selected_models'/profile/metric
            directory.mkdir(parents=True)
            for filename in (chosen.name,'model.o','model.h','metadata.json'):
                shutil.copy2(chosen.parent/filename,directory/filename)
            deployed[profile][metric]=str(directory/chosen.name)
    report=dict(selection=selection,selection_sha256=frozen_hash,entries=entries,screening=screening,
        confirmations=confirmations,evaluation=results,selected_libraries=deployed,
        evaluation_sha256=fingerprint(evaluation_path),evaluation_feature_hashes=evaluation_hashes,
        platform=platform.platform(),compiler=subprocess.check_output(['clang++','--version'],text=True),
        source_sha256={str(p):fingerprint(p) for p in [Path(__file__),Path(__file__).with_name('interference.py'),Path(__file__).with_name('interference.cc')]},
        caveats=['Balanced order reduces position imbalance, not all time drift or CPU interference.',
                 'Round pairs share input rows; they are not independent observations.',
                 'Software pressure is not verified hardware cache invalidation.',
                 'Core and per-call pipeline p99 include clock overhead; pipeline block metric has no inner timers.',
                 'Independent process summaries are the comparison units, not individual repeated rows.',
                 'All metrics share evaluation data; no post-hoc objective switching.',
                 'Synthetic feature producer, without CPU pinning or frequency isolation.'])
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print('Report '+str(out/'report.json'),flush=True)


if __name__=='__main__':main()
