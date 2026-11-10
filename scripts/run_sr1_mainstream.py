import argparse
import _bootstrap  # noqa: F401
from sinkhorn_lmflow.io.config import load_config
from sinkhorn_lmflow.experiments.sr1_mainstream import run_mainstream


def main():
    parser=argparse.ArgumentParser(description='Serial SR1 v1 suite with a shared cumulative budget ledger')
    parser.add_argument('--config',required=True)
    parser.add_argument('--output-root',default='results')
    parser.add_argument('--resume')
    parser.add_argument('--new-revision',action='store_true')
    parser.add_argument('--retry-failures',action='store_true')
    parser.add_argument('--stop-after-job',help='Finish this frozen-plan job and persist a user pause')
    parser.add_argument('--continue-paused',action='store_true',help='Explicitly resume a user-paused run')
    args=parser.parse_args()
    run_mainstream(load_config(args.config),args.output_root,args.resume,args.new_revision,args.retry_failures,
                   args.stop_after_job,args.continue_paused)


if __name__=='__main__':main()
