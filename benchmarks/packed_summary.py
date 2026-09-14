"""Verify completed packed diagnostics and save portable paired comparisons."""
import argparse
import json
from pathlib import Path
import statistics

from .optimize import fingerprint


METRICS=['model_mean_median_ns','model_p50_ns','model_p99_ns',
         'pipeline_block_median_ns','pipeline_p99_ns']


def ranges(values):
    return dict(min=min(values),median=statistics.median(values),max=max(values))


def summarize(path):
    path=Path(path);r=json.loads(path.read_text());plan=r['plan']
    assert fingerprint(path.parent/'plan.json')==r['plan_sha256']
    assert fingerprint(path.parent/'raw.f32')==r['raw_sha256']
    for p,h in {**plan['source_sha256'],**plan['binary_sha256']}.items():assert fingerprint(p)==h
    result=dict(source_report=str(path),source_report_sha256=fingerprint(path),plan=plan,
        plan_sha256=r['plan_sha256'],raw_sha256=r['raw_sha256'],feature_hashes=r['feature_hashes'],
        platform=r['platform'],compiler=r['compiler'],caveats=r['caveats'],builds=r['builds'],profiles={})
    for name,profile in r['results'].items():
        engines={};comparisons={}
        assert len(profile['runs'])==plan['processes']
        assert profile['entries']==plan['entries'][name]
        for run in profile['runs']:
            m=run['measurement'];assert m['paired_rounds'] is True
            assert m['samples_per_engine']==plan['samples']*plan['rounds']
            assert len(m['round_orders'])==plan['rounds']
            for i in range(0,len(m['round_orders']),2):
                assert m['round_orders'][i]==m['round_orders'][i+1][::-1]
                assert m['row_sequence_hashes'][i]==m['row_sequence_hashes'][i+1]
        for entry in profile['entries']:
            runs=[dict(seed=run['seed'],**next(e for e in run['measurement']['engines'] if e['name']==entry['name']))
                  for run in profile['runs']]
            if entry['family']=='engine':assert all(e['max_abs_error']==0 for e in runs)
            engines[entry['name']]=dict(entry=entry,runs=runs,ranges={k:ranges([e[k] for e in runs]) for k in METRICS})
        for triple in plan['pairs']:
            a=engines[triple['packed']]['runs'];controls={}
            for kind in ['split','self_loop']:
                b=engines[triple[kind]]['runs'];scores={}
                for k in METRICS:
                    changes=[100*(1-x[k]/y[k]) for x,y in zip(a,b)]
                    scores[k]=dict(reduction_percent=ranges(changes),individual_reductions=changes,
                        wins=sum(x>0 for x in changes),ties=sum(x==0 for x in changes),processes=len(changes))
                controls[kind]=dict(name=triple[kind],metrics=scores)
            comparisons[triple['packed']]=controls
        result['profiles'][name]=dict(engines=engines,comparisons=comparisons)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reports',nargs='+',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();root=str(Path.cwd())+'/'
    def portable(value):
        if isinstance(value,str):return value.replace(root,'')
        if isinstance(value,list):return [portable(x) for x in value]
        if isinstance(value,dict):return {k.replace(root,''):portable(v) for k,v in value.items()}
        return value
    results={Path(path).parent.name:summarize(path) for path in args.reports}
    Path(args.output).write_text(json.dumps(portable(results),indent=2)+'\n')
    for label,r in results.items():
        for profile in ['code_pressure','mixed_pressure']:
            print(label,profile)
            for name,controls in r['profiles'][profile]['comparisons'].items():
                print(name,{kind:{k:round(c['metrics'][k]['reduction_percent']['median'],2)
                    for k in ['model_mean_median_ns','model_p99_ns','pipeline_block_median_ns']}
                    for kind,c in controls.items()})
    print('Verified frozen plans, sources, binaries, balanced rows/order and zero engine errors.')


if __name__=='__main__':main()
