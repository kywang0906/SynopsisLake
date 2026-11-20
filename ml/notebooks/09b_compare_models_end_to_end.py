# Databricks notebook source
# MAGIC %md
# MAGIC # 09b · Shared Java E2E: SA, saved models, notebook 22 source shape, and exact
# MAGIC
# MAGIC Compare every method on **the same raw data, query IDs, original ARM histogram H, calibration queries,
# MAGIC merge rates, and Java execution path**. No model is trained and no new queries are generated.
# MAGIC
# MAGIC | Setup | Decision and returned answer |
# MAGIC |---|---|
# MAGIC | `skewness` | Skewness-Align decides whether to return H or count the raw points |
# MAGIC | Saved models in `models` | Existing model weights score the query; the gate returns H or a raw count |
# MAGIC | `source_shape` | Notebook 22's source-only distribution reference scores the same H; the gate returns H or a raw count |
# MAGIC | `exact` | Every query is counted from the raw points |
# MAGIC
# MAGIC All gate cutoffs are recalibrated with **live Java scores on the same calibration IDs**, using the existing
# MAGIC empirical cutoff rule. Original model weights and cutoff files are not modified. Test truth never selects a
# MAGIC cutoff. The 5% calibration target is not a guarantee that future test violation will be at most 5%.
# MAGIC
# MAGIC Source-shape metadata is built once from this dataset's exported source histograms and raw files, or loaded
# MAGIC from `shape_dir`. It uses the frozen notebook 22 source-only rule and byte cap. It neither replaces H nor
# MAGIC reruns notebook 22's model study. Metadata construction and calibration are separate from query timing.
# MAGIC
# MAGIC **Timing:** full-batch Java process plus CSV-to-Parquet wall time, including JVM startup, data/model loading,
# MAGIC online scoring and true raw fallback. Batch order is randomized within repeats. Saved Java phase timers
# MAGIC additionally provide request-compute mean/p95; those are a different timing scope from batch wall time.
# MAGIC Raw points are loaded once per JVM, then scanned in memory. OS cache is not flushed; this is not a Spark or
# MAGIC per-query cloud-storage benchmark.
# MAGIC
# MAGIC **Inputs:** a classic cluster with Java, a jar built from the current code, the existing run folder, the
# MAGIC matching dataset query files, real numbered raw CSVs, and saved model folders. Dataset/seed provenance for
# MAGIC legacy model exports cannot be inferred from weights alone; supply a model manifest when available.

# COMMAND ----------

# MAGIC %pip install lightgbm pyarrow

# COMMAND ----------

# MAGIC %md
# MAGIC ## Settings for the common Europe comparison
# MAGIC
# MAGIC Select the actual dataset explicitly; no AUTHORS dataset is silently substituted. For Europe, set:
# MAGIC
# MAGIC | Widget | Value |
# MAGIC |---|---|
# MAGIC | `dataset` | The exact dataset identifier used for the team's Europe model export |
# MAGIC | `run_dir` | That Europe run folder containing `rows.csv`, `arm_*.txt` and `src_*.txt` |
# MAGIC | `dataset_dir` | The matching folder containing `queries-Ans.txt` (query IDs are its original row numbers) |
# MAGIC | `data_dir` | The genuine Europe raw tile folder, with `0.csv`, `1.csv`, ...; required for non-AUTHORS datasets |
# MAGIC | `models_root` | The folder containing the Europe-trained `v1/` and `v1_no_skew/` model exports |
# MAGIC | `models` | `v1,v1_no_skew` (add `v0` if that matching export is available) |
# MAGIC | `seed`, `model_training_seed` | The seed used for those model exports; `0` for the reported Europe run |
# MAGIC | `eps`, `delta` | `0.25`, `0.05` |
# MAGIC | `merge_rates` | `1,0.8,0.6,0.4,0.2`; the nearest existing rates are used and recorded |
# MAGIC | `shape_enabled`, `shape_packet_budget` | `true`, `49152` |
# MAGIC | `n_queries`, `repeats` | `2000`, `3`; use a smaller query count for the first input check |
# MAGIC
# MAGIC Leave `shape_dir` blank to build metadata from the current dataset. An existing shape directory must match
# MAGIC the current source exports; do not point it to the earlier AUTHORS results. A blank `output_root` creates
# MAGIC a unique run under the selected run folder's `compare/` directory. Completed runs are not overwritten.
# MAGIC Databricks preserves widget values across pulls, so review them after switching datasets or branches.

# COMMAND ----------

import os
from pathlib import Path

VOL = "/Volumes/cs598_ccc_research_7474657373242960/synopsis_lake"


def _repo_root():
    try:
        nb = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()  # noqa: F821
        root = nb.split("/ml/notebooks/")[0]
        return root if root.startswith("/Workspace/") else "/Workspace" + root
    except Exception:
        try:
            return str(Path(__file__).resolve().parents[2])
        except NameError:
            return os.getcwd()


DEFAULTS = {
    "volume": VOL,
    "repo_root": _repo_root(),
    "dataset": "",                  # required: exact identifier used for the saved Europe model export
    "dataset_dir": "",              # default <volume>/data/ml/<dataset>
    "run_dir": "",                  # default <volume>/output/ml/<dataset>
    "models": "v1,v1_no_skew",
    "models_root": "",              # default <run_dir>/models; v0 may use <run_dir>/gate
    "data_dir": "",                 # AUTHORS default only; explicitly set for Europe
    "seed": "0",
    "model_training_seed": "0",      # must match seed and the saved models' actual training split
    "model_manifest": "",           # optional dataset/seed/model-SHA provenance JSON
    "eps": "0.25",
    "delta": "0.05",
    "merge_rates": "1,0.8,0.6,0.4,0.2",
    "n_queries": "2000",            # "all" uses all eligible test IDs
    "repeats": "3",
    "java": "java",
    "jar_path": "",                 # default <volume>/jars/synopsislake.jar
    "classpath": "",                # optional complete jar/classes + commons-math3 classpath
    "shape_enabled": "true",
    "shape_dir": "",                # blank builds source-only metadata for this run
    "shape_packet_budget": "49152",  # child packet cap; native metadata is charged separately
    "chunk_size": "250000",
    "output_root": "",              # unique 09b-... run under <run_dir>/compare by default
}
try:
    for name, value in DEFAULTS.items():
        dbutils.widgets.text(name, value)  # noqa: F821
    get = dbutils.widgets.get  # noqa: F821
except NameError:
    def get(name):
        return os.environ.get(name.upper(), DEFAULTS[name])

# COMMAND ----------

import importlib
import sys

config = {name: str(get(name)).strip() for name in DEFAULTS}
repo_root = Path(config["repo_root"]).expanduser().resolve()
if not (repo_root / "ml" / "compare_shape_e2e.py").is_file():
    raise FileNotFoundError("Set repo_root to the checkout containing ml/compare_shape_e2e.py.")
sys.path.insert(0, str(repo_root))
importlib.invalidate_caches()
from ml import compare_shape_e2e  # noqa: E402

if Path(compare_shape_e2e.__file__).resolve() != repo_root / "ml" / "compare_shape_e2e.py":
    raise RuntimeError("Python has cached ml from another checkout. Restart Python, then rerun this notebook.")

print("Dataset:", config["dataset"])
print("Run folder:", config["run_dir"] or "<volume>/output/ml/<dataset>")
print("Models:", config["models"], "| source shape:", config["shape_enabled"])
print("Split seed:", config["seed"], "| declared model training seed:", config["model_training_seed"])
print("Error limit:", config["eps"], "| accepted-risk target:", config["delta"])
print("Queries:", config["n_queries"], "| timing repeats:", config["repeats"])
print("Every method will use the same selected test IDs, original H and common Java calibration.")

# Read-only, shallow inventory: names are candidates, not evidence of matching data or model provenance.
print("\nAvailable run/model candidates (no dataset is selected automatically):")
candidate_root = Path(config["volume"]) / "output" / "ml"
try:
    candidates = sorted(p for p in candidate_root.iterdir() if p.is_dir())[:30]
    if config["run_dir"]:
        explicit = Path(config["run_dir"]).expanduser()
        if explicit not in candidates:
            candidates.insert(0, explicit)
    for candidate in candidates:
        models = (Path(config["models_root"]).expanduser()
                  if config["models_root"] and str(candidate) == config["run_dir"] else candidate / "models")
        print("Run candidate:", candidate, "| rows.csv:", (candidate / "rows.csv").is_file())
        for name in ("v1", "v1_no_skew"):
            folder = models / name
            print(" ", folder, "| model.txt:", (folder / "model.txt").is_file(),
                  "| model.properties:", (folder / "model.properties").is_file())
    if not candidates:
        print("No run folders found under", candidate_root, "; use the actual Europe run_dir/models_root.")
except OSError as error:
    print("Could not list", candidate_root, ":", error)
if not config["dataset"]:
    raise ValueError("Set dataset, run_dir, dataset_dir, data_dir and models_root to the existing Europe export, then rerun this cell.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Run the shared comparison
# MAGIC
# MAGIC This checks the inputs, builds or verifies source metadata, scores and calibrates every gate on calibration
# MAGIC queries, freezes the cutoffs, and executes the held-out query batches. Exact queries and rejected queries scan
# MAGIC the real raw data. The manifest distinguishes verified input identities from legacy model-training provenance
# MAGIC that is only declared externally. Review that status before making a held-out generalization claim.
# MAGIC
# MAGIC After changing code, restart Python before running the notebook again to avoid cached helper modules.

# COMMAND ----------

result_dir = Path(compare_shape_e2e.run_comparison(config)).resolve()
print("Completed:", result_dir)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Quality and time, together
# MAGIC
# MAGIC `coverage` is the fraction answered from H. `violation` is the fraction of accepted answers whose actual
# MAGIC error exceeds `eps`; no acceptance means undefined violation, not 0%. Read both with accepted/bad counts.
# MAGIC `score_auc` uses all tested queries and does not depend on the cutoff. Final-answer mean error includes exact
# MAGIC fallbacks, which have zero error. `synopsis_*` metrics concern only the accepted approximate answers.
# MAGIC Batch latency and request-compute latency have different scopes; do not substitute one for the other.

# COMMAND ----------

import pandas as pd


def show(table):
    if "display" in globals():
        display(table)  # noqa: F821
    else:
        print(table.to_string(index=False))


summary = pd.read_csv(result_dir / "summary.csv")
show(summary)
print("Full results:", result_dir / "summary.csv")
print("Per-query records:", result_dir / "per_query.csv")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cost, calibration and correctness evidence
# MAGIC
# MAGIC Source-shape construction and common calibration are recorded separately from batch timing. Native source
# MAGIC metadata and the child packet both count toward storage; the packet budget alone is not the total cost.
# MAGIC The manifest records H/query/model/input identities, split roles, settings, and execution provenance.
# MAGIC A passing raw parity audit is required before interpreting timing or answer quality.

# COMMAND ----------

for filename in ("payload_costs.csv", "setup_times.csv", "parity.csv"):
    file = result_dir / filename
    if file.is_file():
        print("\n" + filename)
        show(pd.read_csv(file))
print("Calibration scores:", result_dir / "calibration.csv")
print("Frozen cutoffs:", result_dir / "frozen_cutoffs.csv")
print("Input and execution manifest:", result_dir / "manifest.json")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Files to share for review
# MAGIC
# MAGIC Share `09b_comparison_results.zip` from the completed run. It contains the comparison tables, per-query
# MAGIC evidence and provenance needed to assess whether source shape improves on the saved Europe models under
# MAGIC the same conditions. Keep the original run and model folders for replay. No new notebook 22 run is required.
# MAGIC ZIP files are built and verified on driver-local `/tmp`, then copied sequentially to the result directory.
# MAGIC This avoids random-write failures when the result directory is on a Databricks Volume.

# COMMAND ----------

review_zip = result_dir / "09b_comparison_results.zip"
if not review_zip.is_file():
    raise FileNotFoundError("The comparison completed without its review ZIP. Check the run status and packaging logs.")
print("Share this file:", review_zip)
print("Keep this completed run directory:", result_dir)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Optional: recover a failed ZIP export without rerunning the comparison
# MAGIC
# MAGIC Leave the path below blank during normal execution. If the comparison passed its numerical checks but
# MAGIC failed while writing the ZIP, set it to that exact `09b-...` result directory and run **only the next cell**.
# MAGIC It checks the saved validation flag, result tables, protocol and frozen input hashes before exporting.
# MAGIC It does not execute queries, retrain, recalibrate or change the saved timings. An incomplete numerical run
# MAGIC is refused. An old partial ZIP may be replaced; result CSVs and prior error details are preserved.
# MAGIC After pulling this fix, restart Python and run only the imports/settings cell and this recovery cell.

# COMMAND ----------

RECOVERY_RUN_DIR = ""  # Exact failed 09b result directory; never a personal path committed to Git.
if RECOVERY_RUN_DIR.strip():
    result_dir = Path(RECOVERY_RUN_DIR).expanduser().resolve()
    review_zip = compare_shape_e2e.package_results(result_dir, overwrite=True)
    print("Saved results exported without rerunning queries:", review_zip)
    print("Original result CSVs and experiment timings are unchanged:", result_dir)
else:
    print("Export-only recovery is inactive. Leave RECOVERY_RUN_DIR blank for a normal run.")
