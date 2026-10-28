Suggested title: **Add reproducible end-to-end range-query latency benchmark with raw-data fallback**

The existing range-query experiments compare histogram estimates against supplied answers. They do not measure a live accept/fallback execution path. This PR adds a serial per-query benchmark that includes synopsis/source access, histogram evaluation, online feature extraction, model prediction, acceptance decisions, and a real raw-point query whenever a synopsis answer is rejected.

The benchmark compares always-exact execution with a gate around the team's existing LightGBM model and, where supported, the repository's per-query Skewness score. It reuses `SimpleSpatialHistogramOpt` through a small JVM bridge without changing histogram construction, merging, or estimation algorithms. Exact execution uses the closed-rectangle predicate from `BuildRangeAns`, with either a streaming CSV scan or a Spark aggregate. No supplied answer table is used for fallback.

## Experimental controls

- Require explicit existing `query_id` train/calibration/test assignments; do not resplit queries. Training rows are not used by the benchmark.
- Select each empirical acceptance cutoff using calibration rows only. Defaults are 25% per-answer relative error and a 5% accepted-answer violation target. Test outcomes do not tune cutoffs, and the target is not claimed as a statistical guarantee.
- Define zero-truth error as zero for a zero answer and positive infinity otherwise. Include these queries in calibration and evaluation, and label their truth stratum in per-query output.
- Use the same raw inputs, queries, executor, and environment for all policies. Randomize query/method order with a recorded seed and report repetitions and cache settings.
- Measure model/bridge loading, resident synopsis/source initialization, warmup, and calibration separately from steady-state query latency. Record externally reported training time rather than pretending this benchmark retrains the model.
- Rejected policies execute the raw executor inside the timed call. The separately measured always-exact answer is attached only afterward for quality evaluation and fallback consistency checks.

## Outputs

Each run creates a new output directory containing `queries.csv`, `calibration.csv` and `calibration_summary.csv` when gates are enabled, `summary.csv`, and `run.json`. The outputs include stage timings, mean and nearest-rank p95 latency, speedup relative to always-exact execution, accepted/violating query counts, observed risk, initialization/calibration costs, run configuration, and artifact/environment provenance. Repeated measurements are not counted as additional independent quality samples. Undefined accepted-answer risk is left empty when no answers are accepted.

`databricks.example.json` documents explicit replaceable settings for the team's environment. It does not create a cluster, upload artifacts, deploy a job, or contain credentials.

## Validation

- Focused Java tests exercise existing histogram estimates, nonuniform/rectangular grids, domain-edge cases, Skewness scoring, input validation, and the explicit rectangular-score guard.
- Python tests cover calibration ties and cutoffs, query split validation, zero-answer policy, repeat-aware summaries, source/cache accounting, and raw-data fallback behavior.
- The bundled small-data smoke run exercises the real Java bridge and real CSV point scans. It is a correctness demonstration, not a research performance result.
- Maven package passed with 12 tests (11 new bridge tests); all 14 Python tests passed, including actual Spark 3.5.0 and a fixture-only LightGBM Booster loader check. Commands and environment are recorded in `VALIDATION.md`. Formal Databricks performance validation remains outstanding.

## Current integration limitations

At the inspected base revision, the checkout does not include the report's v0/v1 feature pipeline, trained Booster, original query split artifact, or full point workload. The ML adapter therefore requires the team's existing model and exact online feature extractor/schema. It does not introduce a substitute feature set or model. Missing ML configuration fails explicitly.

The existing `ComputeBetaDist` uses the wrong flattening stride for rectangular grids. This PR leaves that core code unchanged and blocks Skewness scoring for affected shapes; histogram estimates remain supported. Use a square merged histogram for the three-policy comparison, or omit Skewness and document why until a separate core fix is reviewed.

Resident-cache speedups are steady-state measurements. Per-query artifact reopening does not imply cold OS/storage caches. End-to-end results must be interpreted together with observed answer quality and accepted sample counts; this PR claims no workload speedup or guaranteed 5% test violation rate.
