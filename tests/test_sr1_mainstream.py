import json
from pathlib import Path
import numpy as np
import pytest
from sinkhorn_lmflow.experiments.sr1_mainstream import (build_plan,run_mainstream,Budget,
    make_input,audit_endpoint,load_warmup,artifact_valid)
from sinkhorn_lmflow.io.config import load_config
from sinkhorn_lmflow.io.result_io import load_result
from sinkhorn_lmflow.solvers.sinkhorn_fast import solve_sk_log_fast
from sinkhorn_lmflow.types import SolverConfig


def tiny(tmp_path):
    return dict(smoke=True,selection='all',repeats=1,warmup_iters=3,max_iter=1000,
        performance_budget_sec=60,diagnostic_budget_sec=5,track_limit_sec=10,memory_cap_gib=8,
        budget_ledger=str(tmp_path/'ledger.json'),mnist_cache='data/mnist',download_mnist=False,
        cases=[dict(case_id='mini',family='M2',n=9,m=11,data_seed=42,epsilon=.1,alpha=None)])


def test_fixed_plan_priorities_and_subset_counts():
    config=load_config('configs/sr1_v1_main.yaml')
    cases,jobs=build_plan(config)
    assert len([c for c in cases if c['family']!='toy'])==57
    assert len(jobs)==607
    assert sum(j['group']=='lbfgs' for j in jobs)==30
    assert sum(j['mode']=='diagnostic' for j in jobs)==7
    first_scaling=next(i for i,j in enumerate(jobs) if j['group']=='scaling')
    assert first_scaling==342
    assert all(j['repeat']==0 for j in jobs[:114])
    assert jobs[first_scaling]['case_id']=='M6_N5000_s42_e0.003'
    assert not any(j['group']=='warmup' and j['warmup_iters']==20 for j in jobs)


def test_resume_no_double_machine_charge_and_audit(tmp_path):
    config=tiny(tmp_path);root=run_mainstream(config,tmp_path)
    manifest=json.loads((root/'manifest.json').read_text())
    assert all(j['status'] in ('success','library_stop') for j in manifest['jobs'])
    used=json.loads((tmp_path/'ledger.json').read_text())['performance_used_sec']
    main=[j for j in manifest['jobs'] if j['group']=='main']
    warmhash={j['attempts'][-1]['warmup_sha256'] for j in main}
    assert len(warmhash)==1
    for job in manifest['jobs']:
        attempt=job['attempts'][-1];r=load_result(root/attempt['run_path'])
        assert r.summary['audit']['passed']
        assert r.elapsed_sec>=r.summary['warmup_elapsed_sec']>=0
        assert artifact_valid(root,job,manifest['implementation_sha256'],attempt['problem_sha256'])
    run_mainstream(config,tmp_path,resume=root)
    after=json.loads((tmp_path/'ledger.json').read_text())['performance_used_sec']
    assert used==after
    assert all(len(j['attempts'])==1 for j in json.loads((root/'manifest.json').read_text())['jobs'])
    bad=dict(config,max_iter=999)
    with pytest.raises(ValueError):run_mainstream(bad,tmp_path,resume=root)
    # A changed initial fingerprint in a saved endpoint must invalidate resume.
    path=root/main[0]['attempts'][-1]['run_path']/'summary.json'
    summary=json.loads(path.read_text());summary['initial_sha256']='wrong';path.write_text(json.dumps(summary))
    assert not artifact_valid(root,main[0],manifest['implementation_sha256'],main[0]['attempts'][-1]['problem_sha256'])


def test_budget_exhaustion_and_resource_skips(tmp_path,monkeypatch):
    config=tiny(tmp_path);config['performance_budget_sec']=0
    root=run_mainstream(config,tmp_path)
    manifest=json.loads((root/'manifest.json').read_text())
    assert all(j['status']=='pending' for j in manifest['jobs'])
    with pytest.raises(ValueError):Budget(tmp_path/'ledger.json',60,5)
    import sinkhorn_lmflow.experiments.sr1_mainstream as mod
    monkeypatch.setattr(mod,'estimated_peak',lambda *a:10**15)
    config=tiny(tmp_path/'resources');root=run_mainstream(config,tmp_path)
    manifest=json.loads((root/'manifest.json').read_text())
    assert all(j['status']=='skipped_resource' for j in manifest['jobs'])


def test_blocked_data_kept_and_cancellation_endpoint(tmp_path,monkeypatch):
    import sinkhorn_lmflow.experiments.sr1_mainstream as mod
    monkeypatch.setattr(mod,'ensure_mnist',lambda *a,**kw:(_ for _ in ()).throw(FileNotFoundError('offline fixture')))
    config=tiny(tmp_path)
    config['cases']=[dict(case_id='image',family='M5',pair_name='same_0',n=784,m=784,epsilon=.01,data_seed=20261110)]
    root=run_mainstream(config,tmp_path)
    manifest=json.loads((root/'manifest.json').read_text())
    assert all(j['status']=='blocked_data' for j in manifest['jobs'])
    p,_,z=make_input(tiny(tmp_path)['cases'][0],None)
    r=solve_sk_log_fast(p,SolverConfig(max_iter=100,compute_spectrum_every=0),deadline=lambda:True)
    assert r.stop_reason=='timeout' and r.n_iter==0 and len(r.history)==1
    assert audit_endpoint(p,r)['passed']


def test_pause_boundary_resume_and_last_job(tmp_path):
    config=tiny(tmp_path);_,jobs=build_plan(config)
    root=run_mainstream(config,tmp_path,stop_after_job=jobs[0]['job_id'])
    manifest=json.loads((root/'manifest.json').read_text())
    assert manifest['jobs'][0]['status']=='success'
    assert all(j['status']=='pending' for j in manifest['jobs'][1:])
    assert json.loads((root/'pause.json').read_text())['paused']
    used=json.loads((tmp_path/'ledger.json').read_text())['performance_used_sec']
    run_mainstream(config,tmp_path,resume=root)
    assert json.loads((tmp_path/'ledger.json').read_text())['performance_used_sec']==used
    run_mainstream(config,tmp_path,resume=root,continue_paused=True,stop_after_job=jobs[-1]['job_id'])
    final=json.loads((root/'manifest.json').read_text())
    assert all(j['status'] in ('success','library_stop') for j in final['jobs'])
    assert all(len(j['attempts'])==1 for j in final['jobs'])
    assert json.loads((root/'pause.json').read_text())['paused']


def test_live_pause_request_at_job_boundary(tmp_path,monkeypatch):
    import sinkhorn_lmflow.experiments.sr1_mainstream as mod
    config=tiny(tmp_path);original=mod.save_result;_,jobs=build_plan(config)
    def save_and_request(result,path,*args,**kwargs):
        saved=original(result,path,*args,**kwargs)
        if Path(path).parent.name=='runs':
            (Path(path).parents[1]/'pause_request.json').write_text(json.dumps({'after_job_id':jobs[0]['job_id']}))
        return saved
    monkeypatch.setattr(mod,'save_result',save_and_request)
    root=run_mainstream(config,tmp_path)
    manifest=json.loads((root/'manifest.json').read_text())
    assert manifest['jobs'][0]['status']=='success'
    assert all(j['status']=='pending' for j in manifest['jobs'][1:])
    assert json.loads((root/'pause.json').read_text())['paused']


def test_atomic_checkpoint_retries_windows_sharing_violation(tmp_path,monkeypatch):
    import sinkhorn_lmflow.experiments.sr1_mainstream as mod
    path=tmp_path/'manifest.json';path.write_text('{"old": true}')
    original=Path.replace;attempts=[]
    def busy_once(source,target):
        attempts.append(str(source))
        if len(attempts)==1:
            assert json.loads(path.read_text())=={'old':True}
            raise PermissionError('simulated Windows sharing violation')
        return original(source,target)
    monkeypatch.setattr(Path,'replace',busy_once)
    mod.atomic_json({'new':True},path)
    assert len(attempts)==2 and json.loads(path.read_text())=={'new':True}
    assert not path.with_name('manifest.json.next').exists()


def test_user_exclusion_preserves_completed_tracks():
    from sinkhorn_lmflow.experiments.sr1_mainstream import apply_operator_policy
    manifest={'jobs':[{'case_id':'large','status':s,'attempts':[]} for s in ('success','skipped_resource','pending')]
                      +[{'case_id':'small','status':'pending','attempts':[]}]}
    apply_operator_policy(manifest,{'excluded_cases':{'large':{'reason':'explicit user cancellation'}}})
    assert [j['status'] for j in manifest['jobs']]==['success','skipped_resource','skipped_by_user','pending']
    assert all(j['comparison_excluded_reason'] for j in manifest['jobs'][:3])
    assert 'comparison_excluded_reason' not in manifest['jobs'][3]
