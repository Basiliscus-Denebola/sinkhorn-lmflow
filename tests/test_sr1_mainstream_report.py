import json
import pandas as pd
from sinkhorn_lmflow.plotting.sr1_mainstream import pair_rows,report_mainstream
from sinkhorn_lmflow.experiments.sr1_mainstream import run_mainstream


def test_pair_censoring_warmup_and_not_applicable():
    base=dict(case_id='x',family='M2',epsilon=.01,alpha=None,marginal_type='uniform',repeat=0,
        warmup_iters=20,group='main',mode='performance',threshold=1e-8,problem_sha256='p',initial_sha256='i',
        warmup_sha256='w',artifact_audit_passed=True,reached_during_warmup=False,
        hit=True,total_sec=2.,terminal_sec=2.,terminal_status='success')
    sr=dict(base,method='mirror_sr1');sk=dict(base,method='sk_log_fast',hit=False,total_sec=None,terminal_sec=300.,terminal_status='timeout')
    scaling=dict(base,method='sk_scaling_fast',hit=False,total_sec=None,terminal_status='not_applicable_kernel_underflow')
    pairs=pair_rows([sr,sk,scaling]);log,raw=pairs
    assert log['speedup'] is None and log['bound']==150 and log['bound_type']=='lower'
    assert raw['reason']=='reference_kernel_not_applicable' and raw['bound'] is None
    sk.update(hit=True,total_sec=1.,terminal_status='success')
    assert pair_rows([sr,sk])[0]['speedup']==.5  # Keep slowdowns.
    sk['warmup_sha256']='wrong'
    assert pair_rows([sr,sk])[0]['reason']=='fingerprint_mismatch'
    sr['comparison_excluded_reason']='user cancelled this scale'
    assert pair_rows([sr,sk])[0]['reason']=='excluded_by_user'
    assert not pair_rows([sr,sk])[0]['paired_success']


def test_persisted_report_coverage_and_corruption(tmp_path):
    config=dict(smoke=True,selection='all',repeats=1,warmup_iters=3,max_iter=1000,
        performance_budget_sec=60,diagnostic_budget_sec=5,track_limit_sec=10,memory_cap_gib=8,
        budget_ledger=str(tmp_path/'ledger.json'),mnist_cache='data/mnist',download_mnist=False,
        cases=[dict(case_id='mini',family='M2',n=9,m=11,data_seed=42,epsilon=.1,alpha=None)])
    root=run_mainstream(config,tmp_path)
    output=report_mainstream(root,make_plots=False,reaudit=True)
    assert output['planned']==4 and output['audit_passed']
    thresholds=pd.read_csv(root/'tables/threshold_costs.csv')
    assert len(thresholds)==12 and set(thresholds.threshold)=={1e-4,1e-6,1e-8}
    coverage=pd.read_csv(root/'tables/coverage.csv')
    assert coverage.planned.sum()==4
    manifest=json.loads((root/'manifest.json').read_text())
    path=root/manifest['jobs'][0]['attempts'][0]['run_path']/'summary.json'
    summary=json.loads(path.read_text());summary['post_elapsed_sec']+=1;path.write_text(json.dumps(summary))
    output=report_mainstream(root,make_plots=False)
    assert not output['audit_passed']
