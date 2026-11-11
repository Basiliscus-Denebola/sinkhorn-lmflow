import argparse
import os
import subprocess
import sys
from pathlib import Path
import _bootstrap  # noqa: F401


def main():
    parser = argparse.ArgumentParser(description='Frozen energy-distance study with cumulative one-hour budget')
    parser.add_argument('--config', required=True)
    parser.add_argument('--new-revision', action='store_true')
    parser.add_argument('--report-from', help='Inherit a fully finished plan into a report-only revision; never run solvers')
    args = parser.parse_args()
    if Path(sys.executable).resolve() != Path('D:/miniconda_envs/OT/python.exe').resolve():
        raise RuntimeError('Use scripts/run_ot.cmd and the OT environment')
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
        if os.environ.get(key) != '1':
            raise RuntimeError('Formal timing requires BLAS threads=1: use scripts/run_ot.cmd')
    # Development validation precedes the machine ledger; a failing gate never
    # spends the experiment budget. No broad expensive tests run in this gate.
    subprocess.run([sys.executable, '-m', 'pytest', '-q', 'tests/test_cost_profile_distance.py',
        'tests/test_energy_distance.py', 'tests/test_energy_distance_study.py',
        '--basetemp=tmp/energy_gate', '-o', 'cache_dir=tmp/energy_pytest_cache'], check=True)
    from sinkhorn_lmflow.io.config import load_config
    from sinkhorn_lmflow.experiments.energy_distance_study import run_study
    run_study(load_config(args.config), args.new_revision, args.report_from)


if __name__ == '__main__':
    main()
