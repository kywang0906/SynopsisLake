# Learned error predictor: how to run it

SynopsisLake can answer a range count from a merged histogram (synopsis) and say how far to trust it. This folder
adds a trained LightGBM model that estimates that error, replacing the Skewness-Align score. A **gate** answers a
query from the synopsis when the estimated error is at or below a cutoff, and reads the raw data otherwise.

**Training happens in Databricks** (notebooks `ml/notebooks/`). The Java side only reads what training produces.
Formats: [SCHEMA.md](SCHEMA.md).

```
Databricks                              Java (SynopsisLake jar)
notebook 02 / ml/gate_artifacts.py  -->  model.txt, model.properties, cutoffs.csv
                                         |
ErrorPredictionLogExp  -> run folder (src_*.txt, arm_*.txt, rows.csv)
                                         |
ApproxQueryExp  (model + cutoffs + run folder + queries)  -> approx.csv: answer, error estimate, decision
```

## 1. Build

```
cd SynopsisLake && mvn -DskipTests package          # SynopsisLake/target/SynopsisLake-0.0.1-SNAPSHOT.jar
```

Every `java` command below needs the jar and commons-math3 on the classpath. From the repo root:

```
JAR=SynopsisLake/target/SynopsisLake-0.0.1-SNAPSHOT.jar
CM=$(find ~/.m2 -name "commons-math3-*.jar" | head -1)
CP=$JAR:$CM
```

## 2. What you need from training

Put these three files together (the names matter):

| File | From | |
|---|---|---|
| `model.txt` | `Booster.save_model` (notebook 02 logs it to MLflow) | any LightGBM regression model; its feature list is read from the file |
| `model.properties` | `ml/gate_artifacts.py` (notebook 02 logs it too) | `label.transform=log` if the model predicts `log(relErr)`, else `identity`; optional `gate.cutoff` |
| `cutoffs.csv` | `ml/gate_artifacts.py` | one cutoff per merge rate, per `eps` and `delta` |

For a model you already have, create the last two from the training data:

```
python ml/gate_artifacts.py out/ml/authors_range data/ml/authors_range model.txt out/gate --seed 0
```

Use the seed the model was trained with, so the calibration queries are not training queries. It also prints the
held-out coverage and violation, and writes `gate_report.csv` per merge rate.

## 3. Create the synopses (once per dataset)

The data files are `range_query/0.csv … 3.csv` (`gunzip -k range_query/2.csv.gz range_query/3.csv.gz` first).

```
java -cp $CP sigspatial25.ErrorPredictionLogExp \
  -dataDir range_query -queryFile range_query/dataDrivenQuery/srcQuery-40-56-32-48-Ans.csv \
  -outDir out/ml/authors_range
```

This writes the synopses the next step reads (`src_<i>.txt`, `arm_<lonRes>.txt`, one per merge rate) and `rows.csv`.
Add `-modelPath model.txt` to also get a `learnedErr` column, and `-cutoffs cutoffs.csv -eps 0.10 -delta 0.05
[-rawFallback true]` to run the gate over every query and merge rate into `gated.csv`. `-rawFallback true` scans the
raw points for every rejected query, which is slow on the full dataset (about 4 ms per query here); leave it off to
only record the decisions.

## 4. Get approximate answers with estimated errors

```
java -cp $CP sigspatial25.ApproxQueryExp \
  -runDir out/ml/authors_range -queryFile queries.csv -outFile out/approx.csv \
  -modelPath model.txt -cutoffs cutoffs.csv -eps 0.10 -delta 0.05 -mergeRate 0.5
```

`queries.csv`: one query per line, `minLon,maxLon,minLat,maxLat[,exactCount]`. `-mergeRate` picks the nearest merged
synopsis. Without `-modelPath` the original Skewness-Align score is used (it needs its own `-cutoffs`).

`out/approx.csv` columns:

| column | meaning |
|---|---|
| `answer` | the count returned: the synopsis estimate, or the exact count if the gate sent it to raw data |
| `fromSynopsis` | `true` if the gate trusted the synopsis. `false` means read the raw data (use `-rawFallback true -dataDir range_query` to have the program do it) |
| `estRelError` | the model's estimated relative error of the synopsis answer; 0 for an exact raw answer; NaN for Skewness-Align |
| `errorScore`, `cutoff` | the score the gate compared with its cutoff |
| `gateNs`, `synNs`, `rawNs` | time to score, to answer from the synopsis, and to read raw data |
| `exact`, `relErr` | only when the query file has the exact count: the true error of `answer` |

Parquet: `python ml/to_parquet.py out/approx.csv` (needs pandas and pyarrow).

## 5. Choose the merge rate automatically

Instead of `-mergeRate`, let the planner pick the smallest synopsis that meets the accuracy target on queries with
exact counts:

```
java -cp $CP sigspatial25.ApproxQueryExp \
  -runDir out/ml/authors_range -queryFile queries.csv -outFile out/approx.csv \
  -modelPath model.txt -cutoffs cutoffs.csv -eps 0.10 -delta 0.05 \
  -plan true -planQueryFile plan_queries.csv -minCoverage 0.01
```

It evaluates every merge rate, writes `out/approx.csv.plan.csv` (coverage, violation and mean error per rate, and which
one was chosen) and answers `queries.csv` from the chosen rate. If no rate meets the target it prints that and exits
with status 2, or answers everything from the raw data with `-rawFallback true`.

Use planning queries that were **not** used to calibrate `cutoffs.csv`; otherwise the observed violation is
optimistic. `-minCoverage` is the share of queries a synopsis must answer for a rate to count (a synopsis that
answers almost nothing saves nothing).

## 5b. Exact execution mode

`-mode` on `ApproxQueryExp` chooses how queries are answered:

| `-mode` | behavior |
|---|---|
| `approx` | synopsis only (default; the same as omitting `-rawFallback`) |
| `gated` | synopsis, with raw data read for queries the gate rejects (same as `-rawFallback true`) |
| `exact` | every query is answered from the raw data; no synopsis, model or cutoff is used |

Exact mode needs only the queries and the raw points, not `-runDir`, `-modelPath`, `-cutoffs`, `-mergeRate` or `-plan`:

```
java -cp $CP sigspatial25.ApproxQueryExp -mode exact \
  -queryFile queries.csv -dataDir range_query -numFile 4 -outFile out/exact.csv
```

`-dataDir` must hold `0.csv .. <numFile-1>.csv`, tab-separated `lon lat[ ...]` lines. To use a single file such as
`SynopsisLake/osm21_pois_shuffle_sample.csv`, link it as `0.csv` in a folder of its own and pass `-numFile 1`.

`out/exact.csv` has the same columns as section 4. `answer` is the true count, `fromSynopsis` is `false`,
`estRelError` is 0, and `rawNs` is the scan time. `mergeRate`, `errorScore` and `cutoff` are empty. If the query file
has exact counts, `exact` and `relErr` are filled in as well.

The scan is linear in the number of points (a few ms per query at 1M points once the file is loaded), so use this mode
as a correctness baseline or for small data. In code, `GatedRangeQuery.exact(rawCounter)` gives the same behavior.

## 6. Check the Java side matches the model

```
java -cp $CP sigspatial25.ErrorPredictionLogExp -dataDir range_query -queryFile … -outDir out/ml/authors_range -modelPath model.txt
python ml/check_java_parity.py out/ml/authors_range data/ml/authors_range model.txt
```

It compares the Java scores with `Booster.predict` and fails if more than 0.5% of rows differ or the rank correlation
is below 0.9999. Features differ from numpy in the last bit, so a model that splits on rounding noise sends about 0.1%
of rows down another path; rank and AUC do not change. Run it once for every new model.

## 7. Learned merge policy (replaces ARM's reshape score)

Everything above scores the *error* of an answer. This section changes how the synopsis itself is built: ARM's reshape
step drops boundaries of the aligned histogram in the order of a hand-written density-difference score, and the learned
policy picks the drops with a LightGBM model instead (`ml/merge_policy.py` trains it, `predictor.MergePolicy` runs it).
Align and Merge are unchanged; only the choice of boundaries differs.

**How it works.** A candidate is one boundary on one axis (dropping it merges the two adjacent buckets). Its features
describe those buckets (widths, counts, density gap, how different their profiles across the other axis are, how many
sources have a boundary there, ...) and no query workload is needed to use the policy. The label is the true increase in
mean relative error, on training queries with known counts, when that boundary is dropped from the current histogram.
The policy repeatedly drops the lowest predicted increase, on the axis with the most drops left.

**Train** (needs `lightgbm`; the run folder from section 3 and a dataset folder with `queries-Ans.txt`):

```
python ml/merge_policy.py train out/ml/authors_range data/ml/authors_range out/merge_policy [--synthetic-data range_query]
python ml/merge_policy.py eval  out/ml/authors_range data/ml/authors_range out/merge_policy/merge_policy.txt
```

`--synthetic-data DIR` trains on uniformly placed synthetic queries of many sizes (exact counts from the raw points)
instead of the dataset's own queries; use `--synthetic-width/--synthetic-height` (degrees) so they cover the real query
sizes. On Databricks, notebook `08_train_merge_policy` does the training, runs the Java side and shows the comparison.

**Use it in Java:**

```
java -cp $CP sigspatial25.ErrorPredictionLogExp -dataDir range_query -queryFile ... -outDir out/ml/authors_range \
  -mergePolicy out/merge_policy/merge_policy.txt
```

This also writes `learned_<res>.txt` (same format as `arm_<res>.txt`) and `rows_learned.csv` (the `rows.csv` columns
without `learnedErr`, `method=LEARNED`), so ARM and the learned merge can be compared on identical queries.

**What to expect.** On held-out queries, mean relative error at merge rates at or below 73% of full resolution, against
ARM (Java, same queries):

| dataset | trained on | change vs ARM |
|---|---|---|
| authors' US data (10.5M points) | synthetic uniform queries | about -15% (up to -24% at the lowest resolutions) |
| global OSM sample, sources at resolution 120..168 (1M points) | synthetic uniform queries | about -9% |
| authors' US data | the dataset's own queries (3 seeds) | -8% to -25% |
| global OSM sample | the dataset's own queries (3 seeds) | about -14% |

**Guardrail: choose per dataset.** The learned policy is an option. `select` compares three ways of reshaping on the
dataset's validation queries (a split the policy never trained on): ARM, the simple rule "merge the pair of buckets with
the fewest points" (`mass`), and the learned policy. It picks the lowest mean error among the strategies that are no
worse than ARM by more than `--tolerance` (default 2%) in any of the small / medium / large query-size thirds; ARM is
always eligible, so nothing is adopted without evidence. It then repeats the comparison on the held-out test queries:

```
python ml/merge_policy.py select out/ml/authors_range data/ml/authors_range out/merge_policy [--model M] [--tolerance 0.02]
```

It writes `merge_strategy.properties` (`strategy=arm|mass|learned`, `model=` relative to the file) and
`merge_strategy_report.csv`. Build the synopses with the choice: `-mergeStrategy out/merge_policy/merge_strategy.properties`
writes `selected_<res>.txt` (ARM's histogram when `arm`). `-mergePolicy mass` runs the simple rule alone. Validation
queries are the dataset's own workload, so re-run `select` when the workload changes.

Result on the three datasets tried (mean relative error, resolutions at or below 73% of full, held-out test queries):

| dataset | ARM | mass rule | learned | chosen |
|---|---|---|---|---|
| authors' US data | 0.538 | 0.613 (+28% on small queries: rejected) | 0.411 | learned |
| global OSM sample (fine sources) | 0.291 | 0.270 | 0.266 | learned |
| spatial join | 0.270 | 0.188 | 0.180 | learned |

On the OSM and spatial-join data the mass rule gets most of the gain on its own, so the margin of the model over the
rule is small there (0.270 vs 0.266 and 0.188 vs 0.180). The guardrail takes the lowest error, but if you want to avoid
a model you can pass `-mergePolicy mass` directly.

Caveats: it is trained per dataset and needs a pass over the raw points for exact counts. With synthetic training the
policy is about 1-2% *worse* than ARM at the very lightest merge rates (just below full resolution). Training on one
query size and testing on another was 3-4x worse than ARM on the authors' data (not on the OSM data), so train on a
broad workload and check held-out query sizes before relying on it. A dataset whose error barely depends on the merge
rate (error set by the source resolution) has nothing to gain: ARM and the oracle policy differ by about 3% there.

`python ml/merge_policy.py golden FILE` rewrites the fixture `SynopsisLake/src/test/resources/merge_policy_golden.txt`
that `MergePolicyTest` uses to check the Java features against the Python ones; regenerate it if `candidate_features`
changes.

## Using it from Java

```java
SynopsisStore store = SynopsisStore.load("out/ml/authors_range");
ErrorPredictor predictor = PredictorFactory.predictor("model.txt", null);          // null = Skewness-Align
CutoffTable cutoffs = PredictorFactory.cutoffs("cutoffs.csv", null, 0.10, 0.05, predictor);
double rate = store.nearestMergeRate(0.5);
GatedRangeQuery q = new GatedRangeQuery(store.context(rate, predictor.needsBeta()), predictor,
        cutoffs.forMergeRate(rate), null);                                         // or a RawRangeCounter
ApproxAnswer a = q.answer(minLon, maxLon, minLat, maxLat);                          // a.answer, a.estimatedRelError, a.fromSynopsis
```

Classes are in `SynopsisLake/src/main/java/predictor/`. A model that names a feature `FeatureRegistry` does not know is
rejected at load, with the name; add features with `FeatureRegistry.register` (its value must match the training code).

## Troubleshooting

| Message | Cause |
|---|---|
| `no cutoff for merge rate X and no default cutoff given` | `cutoffs.csv` has no row for that rate (or for the `-eps`/`-delta` asked for). Pass `-cutoff` or set `gate.cutoff` |
| `model uses features with no Java implementation: …` | the model uses a feature `FeatureRegistry` lacks |
| `this predictor needs the merged histogram's beta distribution` | only from a hand-built `SynopsisContext` with a null beta; `SynopsisStore.context(rate, predictor.needsBeta())` handles it |
| `only single-output regression models are supported` | the model is multiclass, linear-tree, or not a regression objective |
| `needs -cutoffs, -cutoff, or gate.cutoff` | no cutoff was given for the gate |
| every query goes to raw data | the cutoff is `-inf`: the calibration found no cutoff meeting `delta` at that merge rate |

## Run the tests

```
cd SynopsisLake && mvn test
```

## 8. Budgeted source-shape assessment (notebook 22)

`notebooks/22_source_shape_assessment.ipynb` is a focused source-information study. It keeps the original ARM answer
and compares the full 28-feature learner with the same learner plus two R10-reference inputs, or the same two inputs
from a count grid at the same **sidecar** byte budget. The model family, log-error target, TRAIN roles and stopping
policy are shared. Native R08, analytic R10, SA and the fine reference remain controls. These are refits at one fixed
histogram, not the unavailable saved model or the joint multi-rate training run from notebook 12. Model sizes are
measured separately; matching sidecar bytes does not make fitted model sizes identical.

The filename follows the team sequence after notebook 12. Earlier research labels R01–R09 are now filenames
13–21 on the historical `experiments/bounded-sidecar-e2e` branch; the latest source-shape notebook is 22 on
`experiments/source-shape-integration`. Scientific method IDs and saved result names retain their original labels
so existing frozen artifacts and provenance remain valid.

New public outputs default to `synopsislake-results/22_source_shape_assessment`, with `22-review-...` review
folders and `22_source_shape_results.zip`. The bundle's `manifest.json` and `README.txt` map legacy method IDs,
including `R10` and `native_R08`, to readable names. Existing results and hash-locked run paths stay in place.
Databricks retains widget values after a pull: clear `output_root` or set it to the new default to use the new root;
an explicit path, including a previous `R10` path, is preserved.

The live Python adapter reconstructs histogram estimates. The team's original v1 table instead takes some
merge-shift inputs from Java-logged `est`/`srcSumEst`. Tiny arithmetic differences can change learned tree paths.
Consequently these results concern this explicitly reconstructed-input refit, not the team's original saved model;
the Python-to-Java transfer discrepancy is not evidence of a defect in that original training pipeline.

R10 saves exact child masses and quantized child-local positional moments for selected source cells. Its source-only
allocation and Beta/copula integration use established statistical components. This is an experimental representation
and integration study, not a claim of new probability mathematics, universal superiority, bounded arbitrary-source
merging or guaranteed future risk.

### Inputs and reproducibility

Set `run_dir` to a real source/ARM export folder from the existing pipeline, or set `histogram_dir`, `queries_file`,
and `raw_data_dir` explicitly. The grouped-cell membership helper (`shape_split_inputs.py`) reproduces the known
AUTHORS split byte-for-byte without running historical notebooks. For other inputs it records a newly declared
cohort. Truth>=10 is an offline eligibility/reporting condition, never a runtime query filter. Outputs go outside Git
in unique directories. Models, gates and code identities freeze before fresh geometry and labels; every timed fallback
performs a new raw scan. The independent R08 numerical test reference is a small fixture with provenance, not a
dependency on an old notebook.

The notebook contains the full workflow, resume fields, three English diagnostic figures and a small review ZIP.
Model weights remain in `models/seed_<seed>/<method>/`; their measured bytes and hashes remain in the review packet.
Python-request timing can use an explicit compiled SciPy/C++ backend or the indexed Python backend. Compilation,
metadata/model initialization, raw loading, first requests, resident requests and local reloads are separate costs.
Neither backend implies a Databricks/Spark/network latency result.

### Reuse the Java query engine

Build this branch's jar before using the optional Java profile. The existing gate and exact counter are reused:

```bash
java -cp "$CP" sigspatial25.ApproxQueryExp -mode gated \
  -runDir out/ml/authors_range -queryFile queries.csv -outFile out/source_shape.csv \
  -shapeDir out/source_shape/native -shapePacket out/source_shape/shape.bin \
  -cutoffs out/source_shape/cutoffs.csv -eps 0.25 -delta 0.05 -mergeRate 0.5 \
  -dataDir range_query -numFile 4 -setupFile out/source_shape/setup.csv
```

Omitting `-shapePacket` uses native R08. Standalone source shape is mutually exclusive with `-modelPath`. To use a
properly paired full-v1+R10 model, explicitly add `-augmentShape true -modelPath .../ML_v1_shape/model.txt` alongside
the shape options. Do **not** pair the `ML_v1_fine` model with source-shape metadata: the two reference types share
feature names but have different meanings. The locked comparison profile validates model/source identities; manual
CLI users must preserve that pairing. Optional `critic.reference=source_shape` in a separately managed properties
file makes the intended family explicit (`fine` is rejected). Do not mutate a frozen model properties file to add it.

`shape_java_comparison.py` invokes this existing CLI for SA, full-v1, R10 and full-v1+R10. It calibrates the Java backend
on original calibration IDs, freezes all Java scores before fresh truth, and checks every exact/fallback answer.
The profile's query files contain geometry only; evaluation labels are joined afterward. Native source shape has
independent numerical parity checks. Learned Python/Java scores are **not assumed identical**: tiny feature rounding
can change tree paths, so backend differences and separately frozen gates remain explicit.

`-setupFile` records named JVM initialization/query-loop phases. The Java profile reports complete subprocess wall
time to CSV separately from the sum of logged gate/synopsis/raw times; it does not include Parquet conversion.
Raw text inputs are uncompressed `0.csv...`; gzip inputs are materialized once outside the repository and checked
against original identities. Zero-truth output error is zero for a zero answer and infinity for a positive answer.
Zero accepted queries have undefined accepted risk. Calibration's empirical 5% target is not a future-risk guarantee.

To diagnose JVM/load amortization **after** a completed Java profile, the optional batch-cost runner retains the
original small-batch result and executes all 2,048 already-fixed queries, seed 0, with three repetitions. This is an
explicitly post-hoc operating-regime check, not new method validation or threshold selection. It refuses overwrites
and verifies the recorded raw/model/gate/profile identities. Prepare the intent, then pass its printed path:

```bash
python -m ml.shape_java_batch_cost --profile-dir /path/to/completed-java-profile --output-root /path/outside-git
python -m ml.shape_java_batch_cost --run-dir /path/printed-by-the-first-command
```

Keep the locked profile at its original paths; to test another environment, first create a new profile there.
Complete-process and logged query-compute times have different scopes. Report accepted/bad counts with any speedup;
the observed larger-batch benefit does not imply that the empirical 5% target was achieved.

Python regression checks from the repository root:

```bash
python -m unittest discover -s tests -v
```

### Compare source shape with the saved models in notebook 09b

`notebooks/09b_compare_models_end_to_end.py` now includes notebook 22's standalone `source_shape` assessor beside
SA, exact, and the models in `models`. Every arm uses the same original ARM H, held-out query IDs, calibration
IDs, raw data, Java entry point and batch timer. It does not retrain the models, generate queries or replace H.
Live Java scores recalibrate all cutoffs on the common calibration set; existing model files are unchanged.

For the team's Europe comparison, use the **actual Europe dataset identifier and paths** from that model export:
set `dataset`, `dataset_dir`, `run_dir`, `data_dir`, and `models_root`; choose `models=v1,v1_no_skew`,
`seed=0`, `model_training_seed=0`, `eps=0.25`, `delta=0.05`, `merge_rates=1,0.8,0.6,0.4,0.2`,
`n_queries=2000`, `repeats=3`, and `shape_enabled=true`. Non-AUTHORS datasets require an explicit raw `data_dir`.
Use the seed that actually trained the models. Optional `model_manifest` supplies dataset/seed/model-SHA
provenance; legacy exports without it remain explicitly marked as unverified training-split provenance.
The notebook now requires an explicit dataset and defaults to `models=v1,v1_no_skew`, `eps=0.25` and the Europe
rate grid. A shallow read-only inventory prints available run/model candidates. Pulling does not reset old
Databricks widgets: manually replace `authors_range` or a model list containing `v0` when using Europe.
Missing files first require checking the selected dataset and model directory; do not synthesize
`model.properties` or retrain notebook 06 when the correct exports already exist elsewhere.

Leave `shape_dir` blank to build the fixed source-only metadata from that run's source exports and numbered raw
CSVs, with `shape_packet_budget=49152`; existing AUTHORS shape artifacts cannot be reused for Europe. A supplied
shape directory is identity-checked. Use a current jar containing `SourceShapePredictor`; `jar_path` or a complete
`classpath` can override the shared Volume jar without deploying anything. Existing Databricks widget values
persist across pulls, so check the settings when changing datasets.

Each execution creates a unique `09b-...` directory under `output_root` (default `<run_dir>/compare`) and exports
`09b_comparison_results.zip`. Share that ZIP for review. It includes `summary.csv`, `per_query.csv`, `timing.csv`,
`setup_times.csv`, `calibration.csv`, `frozen_cutoffs.csv`, `payload_costs.csv`, `parity.csv`, and `manifest.json`.
Batch wall time includes JVM startup/loading, query execution, CSV output and Parquet conversion; construction
and calibration are separate. Request-compute mean/p95 from logged phases have a narrower scope. Always report
accepted counts, bad counts and violation with any speedup; empirical calibration is not a future 5% guarantee.

The common Java calibration intentionally replaces the old Python-derived cutoff values **for this comparison
only**, so results need not reproduce the earlier PR table exactly. The saved weights and original cutoff files
are preserved. Query eligibility and the query-ID split match 09b; this does not turn a query split into an
unseen-location test. A small TRAIN-only exact audit fails a mismatched raw/query input before timed batches.

For a local replay, put the same widget settings in a JSON file and run:

```bash
python -m ml.compare_shape_e2e --config /path/to/comparison-settings.json
```

The review ZIP is built and CRC-checked on driver-local `/tmp`, then published with sequential writes and a
SHA256 readback check. Building ZIP headers directly on a Unity Catalog Volume can fail with `OSError: [Errno 5]`.
If a prior run failed **only during ZIP export**, use the notebook's optional export-only recovery cell or:

```bash
python -m ml.compare_shape_e2e --package-only /path/to/compare/09b-EXACT-RUN --overwrite-archive
```

This verifies the persisted numerical-validation flag, final tables, protocol identity and frozen input hashes;
it refuses a run that failed before numerical validation. It replaces an old partial ZIP, preserves CSVs and
original timings, and keeps the prior failure in `manifest.json` under `export_recoveries`. It performs no Java
query execution, training, recalibration or retiming. Local staging disk must fit the compressed results archive.
Export time and input rechecks remain outside the original experimental timing scope.

The optional training `model_manifest` must come from the training run, not be invented from the test results.
Its schema is `dataset`, integer `seed`, `split_kind="query"`, `eligible_query_ids_sha256` (SHA256 of JSON-encoded
sorted eligible IDs with separators `(',', ':')`), and `models: {name: {model_sha256: ...}}`. It declares and
checks the matching training cohort and weights; it does not replay or prove the training procedure.

Focused correctness checks (the real Java tests require a JDK and commons-math3):

```bash
COMMONS_MATH3_JAR=/path/to/commons-math3.jar python -m unittest discover -s tests -p 'test_compare_shape_e2e.py' -v
python -m unittest discover -s tests -p 'test_shape_e2e_inputs.py' -v
```
