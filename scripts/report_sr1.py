import argparse
import _bootstrap  # noqa: F401
from sinkhorn_lmflow.plotting.sr1 import report_study


def main():
    parser=argparse.ArgumentParser(description='Plot and report saved SR1 study artifacts')
    parser.add_argument('--run-dir',required=True)
    parser.add_argument('--no-plots',action='store_true')
    args=parser.parse_args()
    print(report_study(args.run_dir,not args.no_plots))


if __name__=='__main__':
    main()
