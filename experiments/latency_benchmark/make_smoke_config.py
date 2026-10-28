"""Generate a fixture-only configuration without embedding machine paths in Git."""
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--classpath", required=True, help="Maven target/classpath.txt")
parser.add_argument("--jar", required=True)
parser.add_argument("--output", required=True, help="New result directory")
parser.add_argument("--config", required=True)
args = parser.parse_args()
fixture = Path(__file__).resolve().parent / "fixtures"
cfg = dict(experiment_kind="correctness_smoke_not_research", bridge="jpype",
           classpath=[str(Path(args.jar).resolve())] + Path(args.classpath).read_text().strip().split(":"),
           histogram=str(fixture / "histogram.json"), queries=str(fixture / "queries.csv"),
           exact={"backend": "scan", "cache": "none", "files": [{"path": str(fixture / "points.csv")}]},
           methods=["always_exact", "skewness"], histogram_cache="resident",
           epsilon=0.25, delta=0.05, repeats=3, warmup_queries=2, seed=0,
           output=str(Path(args.output).resolve()))
Path(args.config).write_text(json.dumps(cfg, indent=2))
