"""Serial per-query benchmark. Labels never enter a measured policy call."""
import csv
from dataclasses import dataclass
import hashlib
import importlib
import json
import math
from pathlib import Path
import platform
import random
import statistics
import subprocess
import time

from .backends import JavaHistograms, exact_executor


@dataclass(frozen=True)
class Query:
    query_id: str
    xmin: float
    ymin: float
    xmax: float
    ymax: float

    @property
    def bounds(self):
        return self.xmin, self.ymin, self.xmax, self.ymax


def relative_error(answer, truth):
    # No smoothing denominator: a false nonzero answer to an empty query is bad.
    if truth == 0:
        return 0.0 if answer == 0 else math.inf
    return abs(answer - truth) / truth


def calibrate_cutoff(scores_and_bad, delta):
    """Largest empirical-feasible score prefix, accepting all ties together.

    This is NOT a statistical test-set risk guarantee. An empty feasible set
    returns None, explicitly meaning reject all (not score <= 0).
    """
    rows = sorted((s, int(b)) for s, b in scores_and_bad if math.isfinite(s))
    threshold = None
    count = bad = i = 0
    while i < len(rows):
        score = rows[i][0]
        while i < len(rows) and rows[i][0] == score:
            count += 1
            bad += rows[i][1]
            i += 1
        if bad / count <= delta:
            threshold = score
    return threshold


def read_queries(path):
    groups = {"train": [], "calibration": [], "test": []}
    ids = set()
    with open(path, newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            qid = row["query_id"]
            if not qid or qid in ids:
                raise ValueError("query_id must be nonempty and unique; one histogram per run")
            ids.add(qid)
            q = Query(qid, *(float(row[k]) for k in ("xmin", "ymin", "xmax", "ymax")))
            if not all(map(math.isfinite, q.bounds)) or q.xmin >= q.xmax or q.ymin >= q.ymax:
                raise ValueError("query bounds must be finite and increasing")
            if row["split"] not in groups:
                raise ValueError("split must be train, calibration or test; no automatic resplit")
            groups[row["split"]].append(q)
    if not groups["test"]:
        raise ValueError("test split is empty")
    return groups


class HistogramStore:
    def __init__(self, config, method, loader):
        self.config, self.method, self.loader = config, method, loader
        self.resident = config.get("histogram_cache", "resident") == "resident"
        if config.get("histogram_cache", "resident") not in ("resident", "per_query"):
            raise ValueError("histogram_cache must be resident or per_query")
        self.item = self._load() if self.resident else None

    def _load(self):
        hist = self.loader.load(self.config["histogram"])
        sources = []
        if self.method == "ml":
            sources = [self.loader.load(p) for p in self.config.get("sources", [])]
        if self.method == "skewness":
            opts = self.config.get("skewness", {})
            hist.initSkewness(opts.get("scale", 0.5), opts.get("epsilon", 1e-6))
        return hist, sources

    def get(self):
        return self.item if self.resident else self._load()


def execute(method, query, executor, store=None, model=None, cutoff=None, score_only=False):
    times = {name: 0 for name in
             ("lookup_ns", "estimate_ns", "features_ns", "inference_ns", "decision_ns", "exact_ns")}
    start = time.perf_counter_ns()
    estimate = score = None
    accepted = False
    if method != "always_exact":
        phase = time.perf_counter_ns()
        hist, sources = store.get()
        times["lookup_ns"] = time.perf_counter_ns() - phase
        phase = time.perf_counter_ns()
        estimate = float(hist.estimate(*query.bounds))
        if not math.isfinite(estimate) or estimate < 0:
            raise ValueError("invalid histogram estimate")
        times["estimate_ns"] = time.perf_counter_ns() - phase
        phase = time.perf_counter_ns()
        if method == "ml":
            features = model.features(query, estimate, hist, sources)
            times["features_ns"] = time.perf_counter_ns() - phase
            phase = time.perf_counter_ns()
            score = float(model.predict(features))
            times["inference_ns"] = time.perf_counter_ns() - phase
        else:
            # Existing Java method also computes an estimate internally. Charge
            # ALL that work to features_ns; do not hide the duplicate work.
            score = float(hist.score(*query.bounds))
            times["features_ns"] = time.perf_counter_ns() - phase
        phase = time.perf_counter_ns()
        accepted = cutoff is not None and math.isfinite(score) and score <= cutoff
        times["decision_ns"] = time.perf_counter_ns() - phase
    if method == "always_exact" or (not accepted and not score_only):
        phase = time.perf_counter_ns()
        answer = float(executor.count(query))
        if not math.isfinite(answer) or answer < 0:
            raise ValueError("invalid exact count")
        times["exact_ns"] = time.perf_counter_ns() - phase
        answer_source = "raw_scan"
    else:
        answer, answer_source = estimate, "histogram"
    elapsed = time.perf_counter_ns() - start
    return dict(query_id=query.query_id, method=method, accepted=accepted,
                estimate=estimate, score=score, answer=answer, answer_source=answer_source,
                e2e_ns=elapsed, overhead_ns=elapsed - sum(times.values()), **times)


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def percentile95(values):
    values = sorted(values)
    # Explicit nearest-rank quantile, meaningful also for one observation.
    return values[max(0, math.ceil(0.95 * len(values)) - 1)]


def summaries(rows, epsilon, delta):
    out = []
    for repeat in sorted({r["repeat"] for r in rows}) + ["all"]:
        group = rows if repeat == "all" else [r for r in rows if r["repeat"] == repeat]
        exact_times = [r["e2e_ns"] for r in group if r["method"] == "always_exact"]
        for method in dict.fromkeys(r["method"] for r in group):
            measurements = [r for r in group if r["method"] == method]
            # Repeats are latency observations, NOT new independent quality samples.
            first = {}
            for row in measurements:
                if row["query_id"] in first:
                    old = first[row["query_id"]]
                    if old["accepted"] != row["accepted"] or not math.isclose(
                            old["answer"], row["answer"], rel_tol=1e-12, abs_tol=1e-9):
                        raise ValueError("non-deterministic policy across repeats; analyze separately")
                first[row["query_id"]] = row
            unique = list(first.values())
            accepted = [r for r in unique if r["accepted"]]
            violations = sum(r["relative_error"] > epsilon for r in accepted)
            risk = violations / len(accepted) if accepted else None
            result = dict(repeat=repeat, method=method, measurements=len(measurements),
                          unique_queries=len(unique), accepted_queries=len(accepted),
                          violating_accepted_queries=violations,
                          acceptance_rate=len(accepted) / len(unique), violation_rate=risk,
                          quality_target_met=(risk <= delta) if risk is not None else None,
                          epsilon=epsilon, delta=delta,
                          speedup_vs_exact=statistics.mean(exact_times) /
                          statistics.mean(r["e2e_ns"] for r in measurements),
                          speedup_scope="steady_state_excludes_separately_recorded_initialization")
            for field in ("e2e_ns", "lookup_ns", "estimate_ns", "features_ns", "inference_ns",
                          "decision_ns", "exact_ns", "overhead_ns"):
                values = [r[field] for r in measurements]
                result["mean_" + field] = statistics.mean(values)
                result["p95_" + field] = percentile95(values)
            out.append(result)
    return out


def _resolve(config, root):
    def resolve(value):
        p = Path(value)
        return str(p if p.is_absolute() else (root / p).resolve())
    for key in ("queries", "histogram", "output"):
        if key in config:
            config[key] = resolve(config[key])
    config["sources"] = [resolve(p) for p in config.get("sources", [])]
    for spec in config["exact"]["files"]:
        spec["path"] = resolve(spec["path"])
    # Classpath/model options are explicit user-owned paths, never guessed.
    return config


def run(config_path, *, loader_factory=JavaHistograms, executor_factory=exact_executor):
    config_path = Path(config_path).resolve()
    with config_path.open(encoding="utf-8") as stream:
        cfg = _resolve(json.load(stream), config_path.parent)
    methods = cfg.get("methods", ["always_exact", "ml", "skewness"])
    if not methods or len(set(methods)) != len(methods) or "always_exact" not in methods:
        raise ValueError("methods must be unique and include always_exact")
    if set(methods) - {"always_exact", "ml", "skewness"}:
        raise ValueError("unknown method")
    epsilon, delta = cfg.get("epsilon", 0.25), cfg.get("delta", 0.05)
    if not math.isfinite(epsilon) or epsilon < 0 or not 0 <= delta < 1:
        raise ValueError("invalid epsilon/delta")
    repeats, warmups = cfg.get("repeats", 3), cfg.get("warmup_queries", 2)
    if not isinstance(repeats, int) or repeats < 1 or not isinstance(warmups, int) or warmups < 0:
        raise ValueError("repeats >= 1 and warmup_queries >= 0 required")
    groups = read_queries(cfg["queries"])
    if len(methods) > 1 and not groups["calibration"]:
        raise ValueError("calibration split required for gates")
    if "ml" in methods and not cfg.get("ml", {}).get("factory"):
        raise ValueError("ML pipeline absent: provide ml.factory and existing model; no substitute model")
    # Refuse overwrite, including accidentally mixing old/new run result files.
    output = Path(cfg["output"])
    output.mkdir(parents=True, exist_ok=False)
    init = {}
    model = None
    if "ml" in methods:
        phase = time.perf_counter_ns()
        module, factory = cfg["ml"]["factory"].split(":")
        model = getattr(importlib.import_module(module), factory)(cfg["ml"].get("options", {}))
        if not callable(getattr(model, "features", None)) or not callable(getattr(model, "predict", None)):
            raise ValueError("ML adapter needs features() and predict()")
        init["model_import_load_init_ns"] = time.perf_counter_ns() - phase
    phase = time.perf_counter_ns()
    executor = executor_factory(cfg["exact"])
    init["exact_executor_init_ns"] = time.perf_counter_ns() - phase
    try:
        phase = time.perf_counter_ns()
        loader = loader_factory(cfg) if len(methods) > 1 else None
        init["jvm_bridge_init_ns"] = time.perf_counter_ns() - phase
        stores = {}
        for method in methods:
            if method != "always_exact":
                phase = time.perf_counter_ns()
                stores[method] = HistogramStore(cfg, method, loader)
                init[method + "_hist_source_beta_init_ns"] = time.perf_counter_ns() - phase
        cal_start = time.perf_counter_ns()
        calibration = []
        for q in groups["calibration"] if len(methods) > 1 else []:
            exact = execute("always_exact", q, executor)
            exact.update(truth=exact["answer"], relative_error=0.0, phase="calibration_exact_label")
            calibration.append(exact)
            for method in methods:
                if method == "always_exact":
                    continue
                row = execute(method, q, executor, stores[method], model, score_only=True)
                row.update(truth=exact["answer"],
                           relative_error=relative_error(row["estimate"], exact["answer"]),
                           phase="calibration_score_only")
                calibration.append(row)
        cutoffs = {method: calibrate_cutoff(
            [(r["score"], r["relative_error"] > epsilon) for r in calibration if r["method"] == method],
            delta) for method in methods if method != "always_exact"}
        calibration_ns = time.perf_counter_ns() - cal_start
        write_csv(output / "calibration.csv", calibration)
        calibration_summary = []
        for method, threshold in cutoffs.items():
            scored = [r for r in calibration if r["method"] == method]
            accepted = [r for r in scored if threshold is not None and math.isfinite(r["score"])
                        and r["score"] <= threshold]
            bad = sum(r["relative_error"] > epsilon for r in accepted)
            calibration_summary.append(dict(method=method, cutoff=threshold, queries=len(scored),
                                            accepted_queries=len(accepted), violating_accepted_queries=bad,
                                            violation_rate=bad / len(accepted) if accepted else None))
        write_csv(output / "calibration_summary.csv", calibration_summary)
        phase = time.perf_counter_ns()
        for q in groups["calibration"][:warmups]:
            for method in methods:
                execute(method, q, executor, stores.get(method), model, cutoffs.get(method))
        init["warmup_ns"] = time.perf_counter_ns() - phase
        rng = random.Random(cfg.get("seed", 0))
        rows = []
        for repeat in range(repeats):
            queries = groups["test"].copy()
            rng.shuffle(queries)
            for q in queries:
                order = methods.copy()
                rng.shuffle(order)
                block = []
                for position, method in enumerate(order):
                    row = execute(method, q, executor, stores.get(method), model, cutoffs.get(method))
                    row.update(repeat=repeat, method_position=position,
                               histogram_cache=cfg.get("histogram_cache", "resident"),
                               exact_backend=cfg["exact"].get("backend", "scan"),
                               raw_cache=cfg["exact"].get("cache", "none"))
                    block.append(row)
                # Evaluation AFTER measured calls. No label goes to model or fallback.
                truth = next(r["answer"] for r in block if r["method"] == "always_exact")
                for row in block:
                    if row["answer_source"] == "raw_scan" and not math.isclose(
                            row["answer"], truth, rel_tol=1e-12, abs_tol=1e-9):
                        raise AssertionError("raw fallback disagrees with measured exact baseline")
                    row.update(truth=truth, relative_error=relative_error(row["answer"], truth),
                               truth_group="zero" if truth == 0 else "small" if truth < 10 else "at_least_10")
                rows.extend(block)
        write_csv(output / "queries.csv", rows)
        summary = summaries(rows, epsilon, delta)
        write_csv(output / "summary.csv", summary)
        try:
            commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
            dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True))
        except (OSError, subprocess.CalledProcessError):
            commit, dirty = None, None
        artifact_paths = [cfg["queries"]] + ([cfg["histogram"]] if "histogram" in cfg else []) + cfg["sources"]
        # Fingerprint small synopses after timing. Do not scan raw data merely to
        # hash it (which would introduce another unreported cache-warming pass).
        artifacts = {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in artifact_paths}
        raw_files = []
        for spec in cfg["exact"]["files"]:
            item = dict(path=spec["path"])
            try:
                stat = Path(spec["path"]).stat()
                item.update(size_bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)
            except OSError:
                item["stat"] = "unavailable; use immutable dataset_version in config"
            raw_files.append(item)
        metadata = dict(config=cfg, git_commit=commit, git_dirty=dirty,
                        artifact_sha256=artifacts, raw_files=raw_files,
                        bridge_metadata=getattr(loader, "metadata", {}),
                        exact_metadata=getattr(executor, "metadata", {}),
                        model_metadata=getattr(model, "metadata", {}),
                        query_file_sha256=hashlib.sha256(Path(cfg["queries"]).read_bytes()).hexdigest(),
                        python=platform.python_version(), platform=platform.platform(),
                        initialization_ns=init, calibration_ns=calibration_ns,
                        training_seconds=cfg.get("ml", {}).get("training_seconds"),
                        training_status="external_report_only_not_run_by_benchmark",
                        split_sizes={k: len(v) for k, v in groups.items()}, cutoffs=cutoffs,
                        zero_truth_policy="error=0 if answer=0, otherwise +inf; included in calibration/test",
                        p95_definition="nearest rank ceil(0.95*n)",
                        risk_note="empirical calibration, not guaranteed on test; repeated queries count once",
                        cache_note="resident loads charged to initialization; per_query reopens artifacts; OS/storage caches NOT flushed",
                        result_status="complete", experiment_kind=cfg.get("experiment_kind", "research"))
        with (output / "run.json").open("w", encoding="utf-8") as stream:
            json.dump(metadata, stream, indent=2, allow_nan=False)
        print("Completed:", output)
        return summary
    finally:
        executor.close()
