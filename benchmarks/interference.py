"""Benchmark feature calculation and cache interference on the inference thread."""
import argparse
import json
from pathlib import Path
import platform
import shutil
import statistics
import subprocess
import sys

import numpy as np
import xgboost as xgb

from .optimize import choose, fingerprint, load_rows
from xgb_latency import Forest

PROFILES = [
    dict(name='hot_control', history_bytes=0, eviction_bytes=0, code_blocks=0),
    dict(name='features_64k', history_bytes=64*1024, eviction_bytes=0, code_blocks=0),
    dict(name='features_4m', history_bytes=4*1024*1024, eviction_bytes=0, code_blocks=0),
    dict(name='data_pressure_2m', history_bytes=64*1024, eviction_bytes=2*1024*1024, code_blocks=0),
    dict(name='code_pressure', history_bytes=64*1024, eviction_bytes=0, code_blocks=256),
    dict(name='mixed_pressure', history_bytes=64*1024, eviction_bytes=2*1024*1024, code_blocks=256),
]


def build_harness(out):
    """Unique noinline code blocks keep actual executed code resident/competing."""
    source = ['#include <cstdint>\n#include <cstddef>\n']
    for i in range(256):
        source.append(f'__attribute__((noinline)) static uint64_t code_{i}(uint64_t x) {{')
        for j in range(32):
            shift = 1+(i*7+j*13)%63
            constant = ((i+1)*0x9e3779b97f4a7c15+(j+1)*0x85ebca77c2b2ae63)&((1<<64)-1)
            source.append(f'x ^= {constant}ULL; x = (x<<{shift})|(x>>{64-shift}); x *= {constant|1}ULL;')
        source.append('return x;}')
    source.append('extern "C" uint64_t feature_code_work(uint64_t x, size_t count) {')
    source.append('static uint64_t (*const functions[])(uint64_t) = {'+','.join(f'code_{i}' for i in range(256))+'};')
    source.append('for (size_t i=0;i<count;++i) x=functions[i](x); return x;}')
    code = out/'feature-code.cc'; code.write_text('\n'.join(source)+'\n')
    native = '-mcpu=native' if platform.machine() in ('arm64','aarch64') else '-march=native'
    flags = ['clang++','-O3','-std=c++17',native,'-fno-fast-math','-ffp-contract=off']
    obj = out/'feature-code.o'
    subprocess.run([*flags,'-c',str(code),'-o',str(obj)],check=True,capture_output=True)
    harness = out/'interference'
    subprocess.run([*flags,str(Path(__file__).with_suffix('.cc')),str(obj),'-o',str(harness),
                    *([] if sys.platform=='darwin' else ['-ldl'])],check=True,capture_output=True)
    metadata = dict(command=flags, code_object_bytes=obj.stat().st_size,
                    code_size_output=subprocess.check_output(['size',str(obj)],text=True),
                    source_sha256=fingerprint(code), harness_sha256=fingerprint(harness))
    (out/'harness.json').write_text(json.dumps(metadata,indent=2)+'\n')
    return harness


def oracle(harness, raw_path, rows, history_bytes, booster, out):
    features_path=out/f'features-{history_bytes}.f32'
    subprocess.run([str(harness),'generate',str(raw_path),str(features_path),str(len(rows)),
                    str(rows.shape[1]),str(history_bytes),'0'],check=True,capture_output=True)
    features=np.fromfile(features_path,dtype=np.float32).reshape(rows.shape)
    expected=booster.predict(xgb.DMatrix(features),output_margin=True)
    path=out/f'expected-{history_bytes}.f32'; expected.tofile(path)
    return path, dict(feature_sha256=fingerprint(features_path),expected_sha256=fingerprint(path))


def measure(harness, raw, expected, rows, entries, profile, output, samples, rounds, seed, *, paired=False):
    command=[str(harness),'bench_paired' if paired else 'bench',str(raw),str(expected),str(len(rows)),str(rows.shape[1]),
             str(samples),str(rounds),str(seed),str(profile['history_bytes']),str(profile['eviction_bytes']),str(profile['code_blocks'])]
    for e in entries: command.extend([e['name'],e['library'],e['symbol'],str(int(e['prepared']))])
    run=subprocess.run(command,check=True,capture_output=True,text=True)
    result=json.loads(run.stdout)
    families={e['name']:e['family'] for e in entries}
    for e in result['engines']:
        if families[e['name']]=='engine' and e['max_abs_error']!=0:
            raise RuntimeError('Prototype differs from XGBoost on generated features: '+e['name'])
    output.write_text(json.dumps(result,indent=2)+'\n')
    return result


def candidates(shape):
    entries=[]
    def read(phase): return json.loads(Path(f'results/{phase}-macos-{shape}/report.json').read_text())
    # Baselines retain all existing stock / patched / PGO / ABI choices.
    directory=read('directory')
    entries.extend(e for e in directory['entries'] if e['family']!='engine')
    base='clang_cost4_block32_rank4' if shape=='100x4' else 'clang_d4_rank32'
    initial=read('exploration')
    names={base+suffix for suffix in ('','_eytzinger','_simd','_compact2','_compact4','_O2','_Oz','_pgo')}
    entries.extend(e for e in initial['entries'] if e['name'] in names)
    old=json.loads(Path(f'results/optimized-ranks-{shape}/report.json').read_text())
    entries.extend(e for e in old['entries'] if e['name'] in ('clang_d4_adaptive','clang_cost4_block32'))
    for phase in ('combined','bucket','prefix','directory'):
        r=read(phase); entries.append(r['selection']['selected']['engine'])
        if phase=='directory':
            entries.extend(e for e in r['entries'] if e['name'] in ('directory_split12','directory_split14','directory_split16'))
        if phase=='prefix':
            entries.extend(e for e in r['entries'] if e['name'] in ('prefix_bits10','prefix_bits12','prefix_bits14','prefix_bits16'))
    return list({e['name']:e for e in entries}.values()),directory['selection']['selected']['engine']


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('model','tuning','evaluation','output','shape'): parser.add_argument('--'+name,required=True)
    parser.add_argument('--candidate-manifest')
    parser.add_argument('--shortlist-runs',type=int,default=0)
    parser.add_argument('--metric',choices=['model_mean_median_ns','model_p99_ns','pipeline_block_median_ns'],default='model_mean_median_ns')
    parser.add_argument('--samples',type=int,default=512)
    parser.add_argument('--rounds',type=int,default=7)
    parser.add_argument('--seed',type=int,default=19410)
    args=parser.parse_args()
    if args.shortlist_runs<0: parser.error('shortlist-runs must be nonnegative')
    if min(args.samples,args.rounds)<1: parser.error('Counts must be positive')
    model=Path(args.model).resolve(); nf=Forest.load(model).num_feature
    tuning_path,evaluation_path=Path(args.tuning).resolve(),Path(args.evaluation).resolve()
    if tuning_path==evaluation_path: parser.error('Use separate tuning and evaluation files')
    tuning=load_rows(tuning_path,nf)
    out=Path(args.output).resolve(); out.mkdir(parents=True,exist_ok=False)
    harness=build_harness(out)
    if args.candidate_manifest:
        manifest=json.loads(Path(args.candidate_manifest).read_text())
        entries,warm_winner=manifest['entries'],manifest['warm_winner']
        references=manifest.get('references',[])
    else:
        entries,warm_winner=candidates(args.shape)
        references=[]
    for e in entries:
        if e['family']=='engine':
            metadata=json.loads((Path(e['library']).parent/'metadata.json').read_text())
            if metadata['source_sha256']!=fingerprint(model): raise ValueError('Candidate model mismatch')
    booster=xgb.Booster(model_file=str(model)); booster.set_param({'nthread':1})
    tune_out=out/'tuning'; tune_out.mkdir(); raw=tune_out/'raw.f32'; tuning.tofile(raw)
    expected={}; feature_hashes={}
    for history in sorted({p['history_bytes'] for p in PROFILES}):
        expected[history],feature_hashes[str(history)]=oracle(harness,raw,tuning,history,booster,tune_out)
    selected={}; tuning_results={}; shortlist_results={}
    for i,profile in enumerate(PROFILES):
        print('Tuning '+profile['name'],flush=True)
        result=measure(harness,raw,expected[profile['history_bytes']],tuning,entries,profile,
                       tune_out/(profile['name']+'.json'),min(args.samples,256),min(args.rounds,5),args.seed+i)
        tuning_results[profile['name']]=result
        if args.shortlist_runs:
            scores={e['name']:e[args.metric] for e in result['engines']}
            shortlist=[warm_winner,*references]
            for family in sorted({e['family'] for e in entries}):
                shortlist += sorted((e for e in entries if e['family']==family),key=lambda e:(scores[e['name']],e['name']))[:2]
            shortlist=list({e['name']:e for e in shortlist}.values())
            repeats=[]
            for repetition in range(args.shortlist_runs):
                run_seed=args.seed+20+i*args.shortlist_runs+repetition
                print(f"Confirming tuning shortlist {profile['name']} {repetition+1}",flush=True)
                repeats.append(measure(harness,raw,expected[profile['history_bytes']],tuning,shortlist,profile,
                    tune_out/(profile['name']+f'-shortlist-{run_seed}.json'),min(args.samples,256),min(args.rounds,5),run_seed))
            aggregate={'engines':[dict(name=e['name'],**{args.metric:statistics.median(
                next(row[args.metric] for row in r['engines'] if row['name']==e['name']) for r in repeats)}) for e in shortlist]}
            shortlist_results[profile['name']]=dict(entries=shortlist,runs=repeats,aggregate=aggregate)
            selected[profile['name']]=choose(aggregate,shortlist,args.metric)
            result=aggregate
        else:
            selected[profile['name']]=choose(result,entries,args.metric)
        winner=selected[profile['name']]['engine']['name']
        score=next(e[args.metric] for e in result['engines'] if e['name']==winner)
        print(f'Selected {winner}: {score:.1f} ns ({args.metric})',flush=True)
    selection=dict(selected=selected,metric=args.metric,profiles=PROFILES,
                   tuning_sha256=fingerprint(tuning_path),tuning_feature_hashes=feature_hashes,
                   model_sha256=fingerprint(model),warm_winner=warm_winner,references=references,seed=args.seed,shape=args.shape,
                   shortlist_runs=args.shortlist_runs,samples=args.samples,rounds=args.rounds,tuning_samples=min(args.samples,256),
                   tuning_rounds=min(args.rounds,5))
    # Freeze ALL workload decisions before opening ANY evaluation data.
    (out/'selection.json').write_text(json.dumps(selection,indent=2)+'\n')
    evaluation=load_rows(evaluation_path,nf)
    if fingerprint(evaluation_path)==selection['tuning_sha256']: raise ValueError('Evaluation duplicates tuning')
    eval_out=out/'evaluation'; eval_out.mkdir(); raw=eval_out/'raw.f32'; evaluation.tofile(raw)
    expected={}; eval_hashes={}
    for history in sorted({p['history_bytes'] for p in PROFILES}):
        expected[history],eval_hashes[str(history)]=oracle(harness,raw,evaluation,history,booster,eval_out)
    results={}
    for i,profile in enumerate(PROFILES):
        final_entries=list({e['name']:e for e in [*selected[profile['name']].values(),warm_winner,*references]}.values())
        runs=[]
        for repetition in range(3):
            seed=args.seed+100+i*3+repetition
            print(f"Evaluating {profile['name']} process {repetition+1}",flush=True)
            result=measure(harness,raw,expected[profile['history_bytes']],evaluation,final_entries,profile,
                           eval_out/(profile['name']+f'-{seed}.json'),args.samples,args.rounds,seed)
            runs.append(dict(seed=seed,measurement=result))
        results[profile['name']]=dict(entries=final_entries,runs=runs)
    deployed={}
    for profile in PROFILES:
        chosen=Path(selected[profile['name']]['engine']['library'])
        directory=out/'selected_models'/profile['name'];directory.mkdir(parents=True)
        for filename in (chosen.name,'model.o','model.h','metadata.json'):
            shutil.copy2(chosen.parent/filename,directory/filename)
        deployed[profile['name']]=str(directory/chosen.name)
    report=dict(selection=selection,entries=entries,tuning=tuning_results,evaluation=results,
                selected_libraries=deployed,tuning_shortlists=shortlist_results,
                evaluation_sha256=fingerprint(evaluation_path),evaluation_feature_hashes=eval_hashes,
                platform=platform.platform(),compiler=subprocess.check_output(['clang++','--version'],text=True),
                binary_sha256={e['name']:fingerprint(e['library']) for e in entries},
                caveats=['Software cache/branch-predictor pressure is not a verified hardware cache flush',
                         'Feature producer is synthetic; replace with the deployment feature function',
                         'Model timings include per-call clock overhead; pipeline blocks avoid inner timers',
                         'Prepared TL receives Entry packing in feature producer; full pipeline includes it',
                         'All workload selections frozen before evaluation; unchanged native binaries',
                         'Warm desktop VM-free macOS measurements without CPU/frequency isolation'])
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print('Report '+str(out/'report.json'),flush=True)


if __name__=='__main__': main()
