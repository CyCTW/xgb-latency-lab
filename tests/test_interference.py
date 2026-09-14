"""Check generated features independently and exercise actual native call paths."""
import json
from pathlib import Path
import subprocess

import numpy as np
import pytest
import xgboost as xgb

from benchmarks.interference import PROFILES, build_harness, oracle, measure
from benchmarks.run import build_tl
from xgb_latency import compile_model


@pytest.fixture(scope='module')
def workload(tmp_path_factory):
    root=tmp_path_factory.mktemp('interference')
    harness=build_harness(root)
    rng=np.random.default_rng(293)
    train=rng.normal(size=(128,4)).astype(np.float32)
    booster=xgb.train({'nthread':1,'max_depth':2,'base_score':.23},
                      xgb.DMatrix(train,label=train[:,0]-train[:,1]),5)
    path=root/'model.json';booster.save_model(path)
    lib=compile_model(path,root/'engine',backend='clang',calibration=train,rank_feature_limit=4,
                      rank_strategy='bucket_split',rank_bucket_bits=16,compact_leaf_depth=2)
    entries=[dict(name='engine',family='engine',library=str(lib),symbol='predict_row',prepared=False)]
    for quantized in (False,True):
        lib,_=build_tl(path,root/('tlq' if quantized else 'tl'),4,quantize=quantized,lto=True)
        for prepared in ([False] if quantized else [False,True]):
            entries.append(dict(name=f'tl_{quantized}_{prepared}',family='tl_stock',library=str(lib),
                                symbol='predict' if prepared else 'predict_row',prepared=prepared))
    rows=rng.normal(size=(16,4)).astype(np.float32);rows[::3,0]=np.nan;rows[1]=np.nan
    raw=root/'raw.f32';rows.tofile(raw)
    return root,harness,booster,entries,rows,raw


@pytest.mark.parametrize('history_bytes',[0,65536])
@pytest.mark.parametrize('packed',[False,True])
def test_feature_producer(workload,history_bytes,packed):
    root,harness,_,_,rows,raw=workload
    output=root/f'producer-{history_bytes}-{packed}.f32'
    subprocess.run([str(harness),'generate',str(raw),str(output),str(len(rows)),str(rows.shape[1]),
                    str(history_bytes),str(int(packed))],check=True)
    actual=np.fromfile(output,dtype=np.float32).reshape(rows.shape)
    expected=rows.copy()
    if history_bytes:
        state=0x12345678;history=[]
        for _ in range(history_bytes//4):
            state=(state*1664525+1013904223)&0xffffffff
            history.append(np.float32((state&65535)/65536))
        for i,row in enumerate(rows):
            for j,value in enumerate(row):
                key=(i*0x9e3779b9+j*0x85ebca6b)&0xffffffff;total=np.float32(0)
                for _ in range(16):
                    key=(key*1664525+1013904223)&0xffffffff
                    total=np.float32(total+history[(key>>4)&(len(history)-1)])
                expected[i,j]=np.float32(value+np.float32(np.float32(np.float32(total/np.float32(16))-.5)*np.float32(1/64)))
    if packed: expected.view(np.int32)[np.isnan(expected)]=-1
    np.testing.assert_array_equal(actual.view(np.uint32),expected.view(np.uint32))
    np.testing.assert_array_equal(np.fromfile(raw,dtype=np.uint32),rows.view(np.uint32).ravel())


@pytest.mark.parametrize('profile',PROFILES,ids=[p['name'] for p in PROFILES])
def test_native_interference_abis(workload,profile):
    root,harness,booster,entries,rows,raw=workload
    expected,_=oracle(harness,raw,rows,profile['history_bytes'],booster,root)
    result=measure(harness,raw,expected,rows,entries,profile,root/(profile['name']+'.json'),8,2,93)
    assert result['samples_per_engine']==16
    assert result['history_bytes']==profile['history_bytes']
    for e in result['engines']:
        assert e['model_mean_median_ns']>=0
        assert e['pipeline_block_median_ns']>=0
        assert len(e['model_round_means'])==2
        assert len(e['pipeline_blocks'])==2
    assert result['engines'][0]['max_abs_error']==0


@pytest.mark.parametrize('shortlist_runs',[0,2])
@pytest.mark.parametrize('with_reference',[False,True])
def test_all_workloads_freeze_before_evaluation(workload,tmp_path,monkeypatch,shortlist_runs,with_reference):
    from benchmarks import interference
    root,harness,_,entries,rows,_=workload
    tuning=tmp_path/'tuning.npy';evaluation=tmp_path/'evaluation.npy'
    np.save(tuning,rows);np.save(evaluation,np.float32(rows+.01))
    output=tmp_path/'run'
    monkeypatch.setattr(interference,'build_harness',lambda out:harness)
    monkeypatch.setattr(interference,'candidates',lambda shape:(entries,entries[0]))
    monkeypatch.setattr(interference,'PROFILES',PROFILES[:2])
    original=interference.load_rows
    def audited(path,nf):
        if path==evaluation:
            selection=json.loads((output/'selection.json').read_text())
            assert set(selection['selected'])=={'hot_control','features_64k'}
        return original(path,nf)
    monkeypatch.setattr(interference,'load_rows',audited)
    extra=[]
    if with_reference:
        reference=dict(entries[0],name='previous_cold')
        entries=[*entries,reference]
        manifest=tmp_path/'candidates.json'
        manifest.write_text(json.dumps(dict(entries=entries,warm_winner=entries[0],references=[reference])))
        extra=['--candidate-manifest',str(manifest)]
    monkeypatch.setattr('sys.argv',['interference','--model',str(root/'model.json'),
        '--tuning',str(tuning),'--evaluation',str(evaluation),'--output',str(output),
        '--shape','fixture','--samples','8','--rounds','1','--shortlist-runs',str(shortlist_runs),*extra])
    interference.main()
    report=json.loads((output/'report.json').read_text())
    for p in PROFILES[:2]:
        if shortlist_runs:
            short=report['tuning_shortlists'][p['name']]
            selected=interference.choose(short['aggregate'],short['entries'],'model_mean_median_ns')
        else:
            selected=interference.choose(report['tuning'][p['name']],entries,'model_mean_median_ns')
        assert report['selection']['selected'][p['name']]==selected
        assert len(report['evaluation'][p['name']]['runs'])==3
        if with_reference:
            assert reference in report['evaluation'][p['name']]['entries']
            assert report['selection']['references']==[reference]
            if shortlist_runs:assert reference in report['tuning_shortlists'][p['name']]['entries']
        assert Path(report['selected_libraries'][p['name']]).read_bytes()==Path(entries[0]['library']).read_bytes()


def test_balanced_native_rounds(workload):
    root,harness,booster,entries,rows,raw=workload;profile=PROFILES[-1]
    expected,_=oracle(harness,raw,rows,profile['history_bytes'],booster,root)
    result=measure(harness,raw,expected,rows,entries,profile,root/'paired.json',16,4,93,paired=True)
    assert result['paired_rounds'] is True
    assert result['samples_per_engine']==64
    for start in [0,2]:
        assert result['round_orders'][start]==result['round_orders'][start+1][::-1]
        assert sorted(result['round_orders'][start])==list(range(len(entries)))
        assert result['row_sequence_hashes'][start]==result['row_sequence_hashes'][start+1]
    assert result['row_sequence_hashes'][0]!=result['row_sequence_hashes'][2]
    assert all(e['max_abs_error']==0 for e in result['engines'] if e['name']=='engine')
    with pytest.raises(subprocess.CalledProcessError):
        measure(harness,raw,expected,rows,entries,profile,root/'invalid-paired.json',16,3,93,paired=True)


def test_multiobjective_freezes_all_choices(workload,tmp_path,monkeypatch):
    from benchmarks import multiobjective as multi
    root,harness,_,entries,rows,_=workload
    tuning=tmp_path/'tuning.npy';evaluation=tmp_path/'evaluation.npy'
    np.save(tuning,rows);np.save(evaluation,np.float32(rows+.01))
    output=tmp_path/'run';manifest=tmp_path/'manifest.json'
    # Controls can be outside the winner's metric/family and must survive to evaluation.
    manifest.write_text(json.dumps(dict(entries=entries,warm_winner=entries[0],references=[entries[0]],
                                       matched_controls={entries[0]['name']:entries[-1]})))
    monkeypatch.setattr(multi,'build_harness',lambda out:harness)
    original=multi.load_rows;original_fingerprint=multi.fingerprint
    def check_frozen():
        selection=json.loads((output/'selection.json').read_text())
        assert set(selection['selected'])=={'hot_control','features_64k'}
        for choices in selection['selected'].values():assert set(choices)==set(multi.METRICS)
    def audited_load(path,nf):
        if path==evaluation:check_frozen()
        return original(path,nf)
    def audited_hash(path):
        if Path(path)==evaluation:check_frozen()
        return original_fingerprint(path)
    monkeypatch.setattr(multi,'load_rows',audited_load)
    monkeypatch.setattr(multi,'fingerprint',audited_hash)
    monkeypatch.setattr('sys.argv',['multiobjective','--model',str(root/'model.json'),
        '--candidate-manifest',str(manifest),'--tuning',str(tuning),'--evaluation',str(evaluation),'--output',str(output),
        '--profiles','hot_control','features_64k','--screen-samples','8','--screen-rounds','2',
        '--confirm-samples','17','--confirm-rounds','4','--confirm-runs','2',
        '--evaluation-samples','19','--evaluation-rounds','6','--evaluation-runs','2'])
    multi.main()
    r=json.loads((output/'report.json').read_text())
    assert r['selection_sha256']==original_fingerprint(output/'selection.json')
    for profile,d in r['evaluation'].items():
        assert entries[-1] in d['entries']
        assert len(d['runs'])==2
        assert all(x['measurement']['paired_rounds'] for x in d['runs'])
        assert all(x['measurement']['samples_per_engine']==19*6 for x in d['runs'])
        confirm=r['confirmations'][profile]
        assert entries[-1] in confirm['entries']
        assert all(x['samples_per_engine']==17*4 for x in confirm['runs'])
        scores=multi.aggregate(confirm['runs'],confirm['entries'],multi.METRICS)
        for metric in multi.METRICS:
            assert r['selection']['selected'][profile][metric]==multi.choose(scores,confirm['entries'],metric)
            chosen=r['selection']['selected'][profile][metric]['engine']
            assert Path(r['selected_libraries'][profile][metric]).read_bytes()==Path(chosen['library']).read_bytes()


def test_shortlist_preserves_different_metric_winners():
    from benchmarks.multiobjective import METRICS, shortlist, aggregate
    entries=[dict(name=n,family='engine') for n in ['a','b','c','d','reference']]
    rows=[dict(name=e['name'],**dict(zip(METRICS,values))) for e,values in zip(entries,
        [(1,100,100,100),(2,50,50,50),(100,2,2,2),(200,1,1,1),(300,300,300,300)])]
    result={'engines':rows}
    short=shortlist(result,entries,[entries[-1]],METRICS)
    assert {e['name'] for e in short}=={e['name'] for e in entries}
    # A large interruption in one process must not dominate the median score.
    outlier={'engines':[dict(row,model_mean_median_ns=10000) for row in rows]}
    scores={e['name']:e for e in aggregate([result,outlier,result],short,METRICS)['engines']}
    assert scores['a']['model_mean_median_ns']==1
    assert scores['d']['model_p99_ns']==1


def test_matched_controls_dependency_closure():
    from benchmarks.multiobjective import with_controls
    a,b,c=[dict(name=n,family='engine') for n in ['a','b','c']]
    initial=[a,a]
    assert with_controls(initial,{'a':b,'b':c,'c':a})==[a,b,c]
    assert initial==[a,a]
