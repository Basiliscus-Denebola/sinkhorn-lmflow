"""Read-only mathematical verification plus a durable, budgeted final audit."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import json
from collections import Counter
import numpy as np
import pandas as pd
from sinkhorn_lmflow.io.config import load_config
from sinkhorn_lmflow.io.result_io import load_result
from sinkhorn_lmflow.experiments.energy_distance_study import BudgetLedger, artifact_hashes, fingerprint
from sinkhorn_lmflow.experiments.sr1_mainstream import ledger_lock, audit_endpoint, atomic_json
from sinkhorn_lmflow.plotting.energy_distance import restored_input

base=Path(__file__).resolve().parents[1]
cfg=load_config(base/'configs/energy_distance_v0.yaml')
ledger_path=base/cfg['budget_ledger']
with ledger_lock(ledger_path):
    ledger=BudgetLedger(ledger_path,cfg)
    try:
        ledger.stage('report')
        root=Path(ledger.data['active_run_dir'])
        manifest=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
        original=Path(manifest['inherited_measurements_from'])
        before=json.loads((original/'manifest.json').read_text(encoding='utf-8'))
        assert len(manifest['jobs'])==262
        assert [j['artifact_sha256'] for j in manifest['jobs']]==[j['artifact_sha256'] for j in before['jobs']]
        assert fingerprint(cfg)[:2]==(manifest['source_sha256'],manifest['config_sha256'])
        audited={}
        by_case={}
        results={}
        for j in manifest['jobs']:by_case.setdefault(j['case_id'],[]).append(j)
        for cid,jobs in by_case.items():
            problem=restored_input(root,cid)
            for job in jobs:
                path=root/job['run_path']
                assert artifact_hashes(path)==job['artifact_sha256']
                result=load_result(path)
                results[job['job_id']]=(job,result.summary)
                audit=audit_endpoint(problem,result)
                assert audit['passed'] is not False,job['job_id']
                if result.converged:assert audit['residual_l1']<=1e-8
                audited[job['job_id']]=audit
                if job['method']=='energy_distance':
                    s=result.summary
                    parts=sum(s[k] for k in ('warmup_sec','distance_build_sec','gamma_setup_sec','bridge_setup_sec','iteration_sec'))
                    assert np.isclose(parts,result.elapsed_sec,rtol=0,atol=1e-10)
        for job,s in results.values():
            if job['stage']=='main' and job['method']=='energy_distance':
                _,baseline=results[job['job_id'].replace('__energy_distance__','__sk_log_fast__')]
                assert s['warmup_sha256']==baseline['warmup_sha256']
                assert s['warmup_sec']==baseline['warmup_sec']
            if job['stage']=='ablation':
                fullkey='main__'+job['case_id']+'__energy_distance__r0'
                _,full=results[fullkey]
                assert s['warmup_sha256']==full['warmup_sha256']
                if job['method']=='energy_amp':assert s['gamma']==full['gamma']
        atomic_json(audited,root/'endpoint_audit.json')
        runs=pd.read_csv(root/'runs.csv')
        pairs=pd.read_csv(root/'paired_speedups.csv')
        assert len(runs)==262 and len(pairs)==144
        assert len(pd.read_csv(root/'threshold_costs.csv'))==786
        assert len(pd.read_csv(root/'pending.csv'))==0
        assert runs[runs.stage=='main'].case_id.nunique()==39
        correction_dir=base/'results/20261111_125616_100831_energy_distance_v0_rev1'
        correction=json.loads((correction_dir/'manifest.json').read_text(encoding='utf-8'))
        correction_counts=dict(Counter(j['status'] for j in correction['jobs']))
        note=dict(reason='Windows slash mismatch in reporting-only inheritance mistakenly started a fresh revision; stopped and excluded',
                  excluded_revision=str(correction_dir),statuses_at_stop=correction_counts,
                  included_in_comparisons=False,budget_reset=False,
                  completed_formal_measurements=str(original),final_report_revision=str(root))
        atomic_json(note,correction_dir/'EXCLUDED_FROM_FORMAL_STUDY.json')
        atomic_json(note,root/'excluded_revision.json')
        verification=dict(planned=262,formal_cases=39,status_counts=dict(Counter(runs.status)),
            independent_audits_passed=sum(v['passed'] is True for v in audited.values()),
            not_applicable_endpoints=sum(v['passed'] is None for v in audited.values()),
            paired_comparisons=len(pairs),threshold_rows=786,
            identical_original_measurement_artifact_hashes=True,source_config_verified=True,
            shared_warmup_and_gamma_verified=True,exclusive_total_timing_verified=True,
            tests='31 core/related regression tests passed; 18 core gate tests passed after report-only guard additions',
            excluded_revision=note)
        atomic_json(verification,root/'final_verification.json')
        print(json.dumps(verification,ensure_ascii=False,indent=2))
        for method,g in pairs.groupby('baseline'):
            print(method,'pairs',len(g),'wins',sum(g.speedup>1),'median',g.speedup.median())
        main=runs[runs.stage=='main']
        print(pairs.groupby(['family','baseline']).speedup.agg(['median','min','max']).to_string())
        print('support range',main[main.family=='M5'][['n','m']].agg(['min','max']).to_string())
    finally:
        ledger.close()
        atomic_json(ledger.data,Path(ledger.data['active_run_dir'])/'budget_snapshot.json')
