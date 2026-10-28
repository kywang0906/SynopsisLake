import argparse
from .runner import run

parser = argparse.ArgumentParser(description="Measured raw-data range-count execution")
parser.add_argument("--config", required=True)
args = parser.parse_args()
run(args.config)
