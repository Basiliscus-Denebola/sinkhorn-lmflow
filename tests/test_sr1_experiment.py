from dataclasses import asdict
import csv
import json
import numpy as np
import pytest
from sinkhorn_lmflow.problems.weak_coupling import weak_coupling_problem, initial_columns, initial_fingerprint
from sinkhorn_lmflow.core.semi_dual import evaluate_reduced
from sinkhorn_lmflow.io.problem_io import problem_fingerprint
from sinkhorn_lmflow.io.result_io import save_result, load_result
from sinkhorn_lmflow.solvers.mirror_sr1 import prepare_warmup, solve_mirror_sr1, solve_sinkhorn_reduced
from sinkhorn_lmflow.types import SolverConfig
from sinkhorn_lmflow.experiments.sr1_study import run_study


def test_known_solution_and_fingerprints():
    p = weak_coupling_problem(32,4,.01)
    K = np.exp(-p.M)
    assert np.all(K>0)
    np.testing.assert_allclose(K.sum(0),1,atol=1e-14)
    np.testing.assert_allclose(K.sum(1),1,atol=1e-14)
    s = evaluate_reduced(p,np.zeros(32))
    np.testing.assert_allclose(s.P,K/32,atol=1e-16)
    assert abs(s.f)<1e-14 and s.metrics['residual_l1']<1e-14
    assert problem_fingerprint(p)==problem_fingerprint(weak_coupling_problem(32,4,.01))
    assert initial_fingerprint(initial_columns(32,4,42))!=initial_fingerprint(initial_columns(32,4,43))
    with pytest.raises(ValueError): weak_coupling_problem(32,4,2)


def test_pair_cache_roundtrip_and_snapshot_isolation(tmp_path):
    p = weak_coupling_problem(16,2,.01)
    cfg = SolverConfig(max_iter=35,sr1_warmup_iters=3,compute_spectrum_every=0,max_backtracks=5,
                       initial_nu=-initial_columns(16,2,42))
    wr,snap = prepare_warmup(p,cfg,diagnostic_reference_z=np.zeros(16))
    twin = snap.clone()
    twin.state.z_col[0] += 1
    assert snap.fingerprint == snap.digest() and twin.fingerprint != twin.digest()
    with pytest.raises(ValueError): solve_mirror_sr1(p,cfg,warmup=twin)
    for fn in (solve_mirror_sr1,solve_sinkhorn_reduced):
        result = fn(p,cfg,warmup=snap)
        assert result.history[:4]==wr.history
        save_result(result,tmp_path/fn.__name__,asdict(cfg))
        read = load_result(tmp_path/fn.__name__)
        assert read.history==result.history
        assert isinstance(read.history[-1].state_eval_count,int)
        assert isinstance(read.history[-1].accept_reason,str)
        assert read.elapsed_sec>=snap.elapsed_sec
    assert snap.fingerprint==snap.digest()


def tiny_config():
    return dict(stage='smoke',n=16,blocks=[2],deltas=[.01],initial_seeds=[42],repeats=1,
                track_budget_sec=10,total_budget_sec=30,
                solver=dict(tol=1e-8,max_iter=2000,sr1_warmup_iters=3,compute_spectrum_every=0,max_backtracks=5))


def test_study_resume_and_failure_preserved(tmp_path):
    config = tiny_config()
    config['solver']['max_iter']=4
    root = run_study(config,tmp_path)
    manifest = json.loads((root/'manifest.json').read_text())
    assert manifest['pairs'][0]['status']=='partial'
    assert len(manifest['pairs'][0]['attempts'][0]['runs'])==2
    from sinkhorn_lmflow.io.config import load_config
    warm_path=root/manifest['pairs'][0]['attempts'][0]['warmup']
    assert load_config(warm_path/'config_resolved.yaml')['max_iter']==3
    run_study(config,resume=root)
    manifest = json.loads((root/'manifest.json').read_text())
    assert len(manifest['pairs'][0]['attempts'])==2
    from sinkhorn_lmflow.plotting.sr1 import report_study
    import pandas as pd
    report_study(root,make_plots=False)
    table=pd.read_csv(root/'tables'/'paired_speedups.csv')
    assert table.speedup.isna().all()
    assert len(pd.read_csv(root/'tables'/'all_runs.csv'))==4
    config['n']=32
    with pytest.raises(ValueError): run_study(config,resume=root)


def test_completed_resume_and_old_csv(tmp_path):
    config=tiny_config()
    config['solver']['tol']=1.
    root=run_study(config,tmp_path)
    run_study(config,resume=root)
    manifest=json.loads((root/'manifest.json').read_text())
    pair=manifest['pairs'][0]
    assert pair['status']=='completed' and len(pair['attempts'])==1
    paths=pair['attempts'][0]['runs']
    for path in paths.values():
        assert load_result(root/path).stop_reason=='warmup_converged'
    damaged=root/next(iter(paths.values()))/'summary.json'
    summary=json.loads(damaged.read_text())
    summary['initial_sha256']='corrupted'
    damaged.write_text(json.dumps(summary))
    run_study(config,resume=root)
    updated=json.loads((root/'manifest.json').read_text())
    assert len(updated['pairs'][0]['attempts'])==2
    path=root/next(iter(paths.values()))/'history.csv'
    with path.open(newline='') as f:
        reader=csv.DictReader(f); rows=list(reader); names=reader.fieldnames
    cut=names.index('reduced_f'); names=names[:cut]
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=names,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    assert load_result(path.parent).history[0].reduced_f is None


def test_pending_budget_and_report(tmp_path):
    config=tiny_config();config['total_budget_sec']=1e-9
    root=run_study(config,tmp_path)
    manifest=json.loads((root/'manifest.json').read_text())
    assert manifest['pairs'][0]['status']=='pending'
    from sinkhorn_lmflow.plotting.sr1 import report_study
    report_study(root,make_plots=False)
    assert (root/'tables'/'all_runs.csv').exists()
