"""Read persisted v1 artifacts, optionally independently re-audit every endpoint."""
import argparse
import _bootstrap  # noqa: F401
from sinkhorn_lmflow.plotting.sr1_mainstream import report_mainstream

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-dir',required=True)
    parser.add_argument('--no-plots',action='store_true')
    parser.add_argument('--audit',action='store_true')
    parser.add_argument('--report',default=None)
    args=parser.parse_args()
    print(report_mainstream(args.run_dir,not args.no_plots,args.audit,args.report))
