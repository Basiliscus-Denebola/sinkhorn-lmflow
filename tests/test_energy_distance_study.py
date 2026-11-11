import json
from collections import Counter
from time import sleep
import numpy as np
from sinkhorn_lmflow.io.config import load_config
from sinkhorn_lmflow.experiments.energy_distance_study import (
    build_plan, formal_cases, choose_kappa, BudgetLedger, run_study)


def test_plan_count_order_and_seed_separation():
    cfg=load_config('configs/energy_distance_v0.yaml');cases,jobs=build_plan(cfg)
    assert len(formal_cases())==39 and len(jobs)==262
    assert Counter(j['stage'] for j in jobs)==dict(calibration=16,main=234,ablation=12)
    assert all(max(c['n'],c['m'])<=2048 for c in cases)
    assert {c['data_seed'] for c in cases if c['case_id'].startswith('cal_')}=={41}
    main=[j for j in jobs if j['stage']=='main']
    assert all(j['repeat']==0 for j in main[:117])
    assert len({j['case_id'] for j in main[:21]})==7


def test_selection_defaults_ties_and_common_success():
    cfg=load_config('configs/energy_distance_v0.yaml')
    assert choose_kappa([],cfg)==dict(kappa=1.,reason='calibration_incomplete')
    rows=[]
    for cid in range(4):
        rows.append(dict(case_id=cid,method='sk_log_fast',status='success',total_sec=2.,warmup_converged=False))
        for k in cfg['kappa_grid']:
            rows.append(dict(case_id=cid,method='energy_distance',kappa=k,status='success',
                             total_sec=1.,warmup_converged=False))
    assert choose_kappa(rows,cfg)['kappa']==.3
    for r in rows:r['warmup_converged']=True
    assert choose_kappa(rows,cfg)['kappa']==1.
    for r in rows:r['warmup_converged']=False
    rows[1]['status']='numerical_failure'
    assert choose_kappa(rows,cfg)['kappa']==1.


def test_budget_resume_never_resets(tmp_path):
    cfg=load_config('configs/energy_distance_v0.yaml');path=tmp_path/'ledger.json'
    b=BudgetLedger(path,cfg);sleep(.015);b.stage('main');b.close()
    used=json.loads(path.read_text())['used_sec']
    b=BudgetLedger(path,cfg)
    assert b.used()>=used
    assert b.remaining('calibration')<360
    b.close()


def test_tiny_pipeline_resume_and_report(tmp_path,monkeypatch):
    # No formal workload is run by the validation gate.
    import sinkhorn_lmflow.experiments.energy_distance_study as study
    import sinkhorn_lmflow.plotting.energy_distance as plotting
    monkeypatch.setattr(plotting,'plot_results',lambda *args: None)
    cfg=load_config('configs/energy_distance_v0.yaml')
    tiny=dict(case_id='tiny',family='M2',n=9,m=11,data_seed=42,epsilon=.1,alpha=None)
    cal=dict(tiny,case_id='cal_tiny',data_seed=41)
    cfg.update(test_cases=[tiny],test_calibration_cases=[cal],test_representatives=['tiny'],
        repeats=1,warmup_iters=2,max_iter=500,budget_ledger=str(tmp_path/'budget.json'))
    # Keep testing artifacts separate from formal reports while retaining the
    # real source root for fingerprints and imports.
    original=plotting.generate_report
    def report(root,path,**kwargs):return original(root,tmp_path/'report.md',**kwargs)
    monkeypatch.setattr(plotting,'generate_report',report)
    root=run_study(cfg)
    manifest=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    assert len(manifest['jobs'])==9
    assert all(j['status']=='success' for j in manifest['jobs'])
    before=[j['artifact_sha256'] for j in manifest['jobs']]
    used=json.loads((tmp_path/'budget.json').read_text())['used_sec']
    assert run_study(cfg)==root
    assert json.loads((tmp_path/'budget.json').read_text())['used_sec']>=used
    after=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    assert [j['artifact_sha256'] for j in after['jobs']]==before
    assert (root/'endpoint_audit.json').exists()
    original(root,tmp_path/'report.md',audit=True)
    import pandas as pd
    ratios=pd.read_csv(root/'paired_speedups.csv')
    assert len(ratios)==2 and set(ratios.baseline)=={'sk_log_fast','sk_scaling_fast'}
    assert (ratios.speedup>0).all()

    # Report-only revisions must handle Windows fingerprint path separators
    # and must not execute a single solver or change stored measurements.
    real_fingerprint=study.fingerprint
    source_sha,cfg_sha,hashes=real_fingerprint(cfg)
    def report_change(config):
        revised=dict(hashes)
        key=next(k for k in revised if k.replace('\\','/')=='src/sinkhorn_lmflow/plotting/energy_distance.py')
        revised[key]='test-report-only-revision'
        return 'test-revision-sha',cfg_sha,revised
    monkeypatch.setattr(study,'fingerprint',report_change)
    def no_solve(*args,**kwargs):raise AssertionError('report revision launched a solver')
    for name in ('solve_sk_log_fast','solve_sk_scaling_fast','solve_energy_distance','prepare_fast_warmup'):
        monkeypatch.setattr(study,name,no_solve)
    revised=run_study(cfg,report_from=root)
    latest=json.loads((revised/'manifest.json').read_text(encoding='utf-8'))
    assert revised!=root and latest['measurement_source_sha256']==source_sha
    assert [j['artifact_sha256'] for j in latest['jobs']]==before
