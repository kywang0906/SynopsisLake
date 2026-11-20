"""Shared Java 09b comparison, including notebook 22's source-shape assessor.

No model fitting, new query generation, or histogram rebuilding occurs here.
Every arm is recalibrated from live Java scores on identical calibration IDs.
Test truth is kept out of Java query files and is used only for post-run checks.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import time
import traceback
from uuid import uuid4
import zipfile

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .features import load_hist
from .gate_eval import cutoff, split_ids
from .result_archive import package_directory
from .shape_e2e_inputs import prepare_shape_metadata, verify_shape_metadata

MAIN_CLASS = "sigspatial25.ApproxQueryExp"
BOUNDS = ["minLon", "maxLon", "minLat", "maxLat"]


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=str) + "\n", encoding="utf-8")


def _bool(value):
    if isinstance(value, bool):
        return value
    if str(value).lower() not in ("true", "false"):
        raise ValueError("Boolean settings must be true or false")
    return str(value).lower() == "true"


def relative_error(answers, truth):
    a, t = np.asarray(answers, float), np.asarray(truth, float)
    if a.shape != t.shape or not np.isfinite(a).all() or not np.isfinite(t).all() or (t < 0).any():
        raise ValueError("Answers and nonnegative truth must be aligned finite arrays")
    out = np.full(a.shape, np.inf)
    np.divide(np.abs(a - t), t, out=out, where=t > 0)
    out[(t == 0) & (a == 0)] = 0.
    return out


def calibrated_cutoff(scores, errors, eps, delta):
    scores, errors = np.asarray(scores, float), np.asarray(errors, float)
    if not len(scores) or scores.shape != errors.shape or not np.isfinite(scores).all() or np.isnan(errors).any():
        raise ValueError("Calibration requires aligned nonempty finite scores and non-NaN errors")
    return float(cutoff(scores, errors, eps, delta))


def _syn(frame):
    s = frame.fromSynopsis
    if s.dtype == bool:
        return s.to_numpy()
    if not s.astype(str).str.lower().isin(["true", "false"]).all():
        raise ValueError("Invalid fromSynopsis values")
    return s.astype(str).str.lower().eq("true").to_numpy()


def validate_answers(frame, histogram_answers, exact_answers, expected_cutoff, *, exact=False):
    """Check real fallback and acceptance; arrays follow Java query-file order."""
    h, raw = np.asarray(histogram_answers, float), np.asarray(exact_answers, float)
    if len(frame) != len(raw) or h.shape != raw.shape:
        raise ValueError("Answer/parity arrays are not aligned")
    answer, accepted = frame.answer.to_numpy(float), _syn(frame)
    if not np.isfinite(answer).all():
        raise ValueError("Nonfinite returned answer")
    phases = frame[["gateNs", "synNs", "rawNs"]].to_numpy(float)
    if not np.isfinite(phases).all() or (phases < 0).any():
        raise ValueError("Invalid Java request timers")
    if exact:
        if accepted.any() or not np.array_equal(answer, raw) or (frame.rawNs <= 0).any():
            raise ValueError("Exact mode must return real raw counts with positive scan time")
        return
    score = frame.errorScore.to_numpy(float)
    if not np.isfinite(score).all() or not np.array_equal(accepted, score <= expected_cutoff):
        raise ValueError("Accepted decisions disagree with the frozen cutoff")
    if not np.all(frame.cutoff.to_numpy(float) == expected_cutoff):
        raise ValueError("Java used a different cutoff")
    if not np.allclose(answer[accepted], h[accepted], rtol=0, atol=1e-8):
        raise ValueError("Accepted answer is not the shared histogram H")
    if not np.array_equal(answer[~accepted], raw[~accepted]):
        raise ValueError("Fallback answer differs from the real exact baseline")
    if (frame.rawNs.to_numpy()[accepted] != 0).any() or (frame.rawNs.to_numpy()[~accepted] <= 0).any():
        raise ValueError("Raw-scan timing does not match acceptance/fallback")


def summarize_answers(frame, H, truth, eps):
    accepted = _syn(frame)
    err, h_err = relative_error(frame.answer, truth), relative_error(H, truth)
    bad = h_err > eps
    n, count = len(frame), int(accepted.sum())
    violations = int((accepted & bad).sum())
    def average(a):
        return float(np.mean(a)) if len(a) else np.nan
    def p95(a):
        return float(np.quantile(a, .95)) if len(a) else np.nan
    score = frame.errorScore.to_numpy(float)
    score_ok = np.isfinite(score).all() and np.unique(bad).size == 2
    return dict(queries=n, coverage=count/n, synopsis_queries=count,
                accepted_bad_count=violations, accepted_good_count=count-violations,
                violation=violations/count if count else np.nan,
                good_answer_recall=float((accepted & ~bad).sum()/(~bad).sum()) if (~bad).any() else np.nan,
                mean_error=average(err), p95_error=p95(err),
                synopsis_mean_error=average(err[accepted]), synopsis_p95_error=p95(err[accepted]),
                score_auc=float(roc_auc_score(bad, score)) if score_ok else np.nan)


def _config(config):
    c = dict(config)
    c.setdefault("volume", "/Volumes/cs598_ccc_research_7474657373242960/synopsis_lake")
    c["dataset"] = str(c.get("dataset", "")).strip()
    if not c["dataset"]:
        raise ValueError("Set dataset explicitly to the saved models' dataset identifier; Europe is not AUTHORS")
    for key, default in (("seed", 0), ("model_training_seed", 0), ("repeats", 3),
                         ("shape_packet_budget", 49152), ("chunk_size", 250000)):
        c[key] = int(c.get(key, default))
    c["eps"], c["delta"] = float(c.get("eps", .25)), float(c.get("delta", .05))
    if not (0 < c["eps"] and 0 <= c["delta"] < 1) or c["repeats"] < 1:
        raise ValueError("Need eps>0, 0<=delta<1, and repeats>=1")
    if c["seed"] != c["model_training_seed"]:
        raise ValueError("seed must match model_training_seed; no new test split for old weights")
    c["shape_enabled"] = _bool(c.get("shape_enabled", True))
    models = c.get("models", "v1,v1_no_skew")
    c["models"] = [x.strip() for x in models.split(",") if x.strip()] if isinstance(models, str) else list(models)
    if (not c["models"] or len(set(c["models"])) != len(c["models"])
            or any(not re.fullmatch(r"[A-Za-z0-9_-]+", x) or x in ("source_shape", "skewness", "exact") for x in c["models"])):
        raise ValueError("models must contain unique, safe saved-model directory names")
    rates = c.get("merge_rates", "1,0.8,0.6,0.4,0.2")
    c["merge_rates"] = [float(x) for x in rates.split(",") if x.strip()] if isinstance(rates, str) else list(rates)
    if not c["merge_rates"] or not np.isfinite(c["merge_rates"]).all():
        raise ValueError("merge_rates must be nonempty and finite")
    c["n_queries"] = str(c.get("n_queries", "2000")).lower()
    if c["n_queries"] != "all" and int(c["n_queries"]) < 1:
        raise ValueError("n_queries must be all or a positive integer")
    volume = Path(c["volume"])
    defaults = dict(repo_root=Path(__file__).resolve().parents[1],
                    run_dir=volume/"output/ml"/c["dataset"], dataset_dir=volume/"data/ml"/c["dataset"],
                    jar_path=volume/"jars/synopsislake.jar")
    for key, default in defaults.items():
        c[key] = str(Path(c.get(key) or default).expanduser().resolve())
    if not c.get("data_dir") and c["dataset"] != "authors_range":
        raise ValueError("Set data_dir explicitly for Europe/another dataset; AUTHORS raw data is not substituted")
    c["data_dir"] = str(Path(c.get("data_dir") or volume/"data/range_query").expanduser().resolve())
    c["models_root"] = str(Path(c.get("models_root") or Path(c["run_dir"])/"models").expanduser().resolve())
    c["output_root"] = str(Path(c.get("output_root") or Path(c["run_dir"])/"compare").expanduser().resolve())
    c["java"] = c.get("java") or "java"
    return c


def _classpath(c):
    if c.get("classpath"):
        entries = c["classpath"].split(os.pathsep)
    else:
        jar = Path(c["jar_path"])
        candidates = sorted((Path(c["volume"])/"jars").glob("commons-math3*.jar"))
        if not candidates:
            candidates = sorted(Path.home().glob(".m2/repository/org/apache/commons/commons-math3/*/*.jar"))
        if not candidates:
            candidates = sorted(Path("/databricks/jars").glob("**/*commons-math3*.jar"))
        if not candidates:
            raise FileNotFoundError("Set classpath to the current SynopsisLake jar/classes and commons-math3 jar")
        entries = [str(jar), str(candidates[0])]
    paths = [Path(x).expanduser().resolve() for x in entries]
    if not all(x.exists() for x in paths):
        raise FileNotFoundError("Every explicit Java classpath entry must exist")
    needed = ["sigspatial25/ApproxQueryExp.class"]
    if c["shape_enabled"]:
        needed.append("predictor/SourceShapePredictor.class")
    def contains(p, member):
        if p.is_dir():
            return (p/member).is_file()
        if not zipfile.is_zipfile(p):
            return False
        with zipfile.ZipFile(p) as archive:
            return member in archive.namelist()
    for member in needed:
        if not any(contains(p, member) for p in paths):
            raise ValueError("Java classpath lacks " + member + "; build the current team main jar")
    c["classpath"] = os.pathsep.join(map(str, paths))
    return [f for p in paths for f in (sorted(p.rglob("*.class")) if p.is_dir() else [p])]


def _model_dirs(c):
    out = {}
    for name in c["models"]:
        p = Path(c["models_root"])/name
        if name == "v0" and not (p/"model.txt").is_file() and (Path(c["run_dir"])/"gate/model.txt").is_file():
            p = Path(c["run_dir"])/"gate"
        for f in ("model.txt", "model.properties"):
            if not (p/f).is_file():
                raise FileNotFoundError(
                    f"Missing {p/f} for selected model '{name}'. "
                    f"Resolved dataset={c['dataset']}, run_dir={c['run_dir']}, models_root={c['models_root']}. "
                    "Check these settings and use the original matching model export; do not create placeholder properties. "
                    "The Europe comparison needs models=v1,v1_no_skew; remove v0 unless its export is required and complete. "
                    "Databricks retains old widget values after a pull. Training is needed only if the intended model export does not exist."
                )
        out[name] = p
    return out


def _workload(c):
    run, ds = Path(c["run_dir"]), Path(c["dataset_dir"])
    q = pd.read_csv(ds/"queries-Ans.txt", header=None, float_precision="round_trip")
    if q.shape[1] != 5:
        raise ValueError("queries-Ans.txt must have four bounds and the exact count")
    q.columns = [*BOUNDS, "truth"]
    q.insert(0, "source_qId", np.arange(len(q)))
    if not np.isfinite(q[[*BOUNDS, "truth"]].to_numpy(float)).all() or (q.truth < 0).any() or (q.truth % 1 != 0).any():
        raise ValueError("Query bounds/counts must be finite; counts nonnegative integers")
    if (q.minLon >= q.maxLon).any() or (q.minLat >= q.maxLat).any():
        raise ValueError("Queries must have positive width/height")
    rows = pd.read_csv(run/"rows.csv", float_precision="round_trip")
    rows = rows[rows.method == "ARM"].copy()
    if rows.empty or rows.duplicated(["qId", "mergeRate"]).any():
        raise ValueError("Need unique ARM rows per query and merge rate")
    if not np.isfinite(rows.qId).all() or (rows.qId % 1 != 0).any() or rows.qId.min() < 0 or rows.qId.max() >= len(q):
        raise ValueError("Invalid logged query IDs")
    if not np.array_equal(rows["gt"].to_numpy(float), q.truth.to_numpy()[rows.qId.to_numpy(int)]):
        raise ValueError("Logged truth differs from queries-Ans.txt")
    eligible = np.sort(rows.loc[rows["gt"] >= 10, "qId"].unique().astype(int))
    tr, ca, te = split_ids(eligible, c["seed"])
    if not tr or not ca or not te:
        raise ValueError("Insufficient eligible query IDs for the inherited 50/20/30 split")
    selected = np.array(sorted(te))
    if c["n_queries"] != "all" and int(c["n_queries"]) < len(selected):
        selected = np.sort(np.random.default_rng(c["seed"]).choice(selected, int(c["n_queries"]), replace=False))
    roles = pd.DataFrame([dict(source_qId=int(i), split=role, selected_for_timing=(i in selected if role == "test" else False))
                          for role, ids in (("train", tr), ("calibration", ca), ("test", te)) for i in sorted(ids)])
    hists = {}
    for p in sorted(run.glob("arm_*.txt")):
        rate, _, _, counts = load_hist(p)
        if rate in hists or not np.isfinite(rate) or not np.isfinite(counts).all():
            raise ValueError("Duplicate/nonfinite merged histogram rate/counts")
        hists[rate] = p
    if not hists:
        raise FileNotFoundError("No arm_*.txt in run_dir")
    # Java resolves a distance tie to the lower rate (TreeMap ascending order).
    rates = sorted({min(sorted(hists), key=lambda r: abs(r-x)) for x in c["merge_rates"]}, reverse=True)
    for r in rates:
        present = rows.loc[np.isclose(rows.mergeRate, r, rtol=0, atol=1e-12), "qId"]
        if set(ca).union(selected)-set(present):
            raise ValueError("Selected histogram rate is missing calibration/test logged IDs")
    return q.set_index("source_qId", drop=False), roles, sorted(ca), selected, hists, rates


def _write_queries(path, frame):
    frame[BOUNDS].to_csv(path, header=False, index=False, float_format="%.17g")
    pd.DataFrame(dict(java_qId=np.arange(len(frame)), source_qId=frame.source_qId.to_numpy(int))).to_csv(
        path.with_suffix(".ids.csv"), index=False)


def _java(c, method, model_dirs, shape, query, out, setup, rate=None, cutoffs=None, mode="gated"):
    args = [c["java"], "-cp", c["classpath"], MAIN_CLASS, "-mode", mode,
            "-queryFile", str(query), "-outFile", str(out), "-setupFile", str(setup)]
    if method == "exact":
        args += ["-dataDir", c["data_dir"], "-numFile", str(c["num_files"])]
        return args
    args += ["-runDir", c["run_dir"], "-mergeRate", repr(float(rate)),
             "-eps", str(c["eps"]), "-delta", str(c["delta"])]
    args += ["-cutoffs", str(cutoffs)] if cutoffs is not None else ["-cutoff", "0"]
    if method in model_dirs:
        args += ["-modelPath", str(model_dirs[method]/"model.txt")]
    elif method == "source_shape":
        args += ["-shapeDir", str(shape/"native"), "-shapePacket", str(shape/"shape.bin")]
    elif method != "skewness":
        raise ValueError("Unknown setup: " + method)
    if mode == "gated":
        args += ["-dataDir", c["data_dir"], "-numFile", str(c["num_files"])]
    return args


def _run_java(c, method, models, shape, query, ids, folder, rate=None, cutoffs=None, mode="gated", parquet=False):
    folder.mkdir(parents=True, exist_ok=False)
    output, setup = folder/"answers.csv", folder/"setup.csv"
    args = _java(c, method, models, shape, query, output, setup, rate, cutoffs, mode)
    _json(folder/"command.json", dict(argv=args, query_file_has_truth=False))
    t0 = time.perf_counter()
    result = subprocess.run(args, capture_output=True, text=True)
    t1 = time.perf_counter()
    if result.returncode:
        (folder/"stdout.log").write_text(result.stdout)
        (folder/"stderr.log").write_text(result.stderr)
        raise RuntimeError(f"{method} Java failed: {result.stderr[-3000:]}")
    if parquet:
        pd.read_csv(output, float_precision="round_trip").to_parquet(folder/"answers.parquet", index=False)
    t2 = time.perf_counter()
    (folder/"stdout.log").write_text(result.stdout)
    (folder/"stderr.log").write_text(result.stderr)
    a = pd.read_csv(output, float_precision="round_trip")
    if not np.array_equal(a.qId.to_numpy(), np.arange(len(ids))):
        raise ValueError("Java answer IDs/count differ from the frozen query-file rows")
    if method != "exact" and not np.allclose(a.mergeRate, rate, rtol=0, atol=1e-12):
        raise ValueError("Java selected a different merged histogram rate")
    a["source_qId"] = np.asarray(ids, int)
    a["fromSynopsis"] = _syn(a)
    record = dict(java_s=t1-t0, parquet_s=t2-t1 if parquet else 0., total_s=t2-t0,
                  setup=method, rate=rate, query_count=len(a), folder=str(folder))
    _json(folder/"timing.json", record)
    return a, record


def _check_hashes(hashes):
    for name, digest in hashes.items():
        if _sha(name) != digest:
            raise ValueError("Frozen input identity changed: " + name)


def _model_provenance(c, models, roles):
    manifest = c.get("model_manifest")
    ids = sorted(roles.source_qId.astype(int))
    digest = hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()
    status = dict(status="external_training_split_unverified", declared_dataset=c["dataset"],
                  declared_training_seed=c["model_training_seed"], split_kind="query",
                  eligible_query_ids_sha256=digest,
                  limitation="Legacy model exports lack training split provenance; matching weights/roles must be confirmed from the training run.")
    if manifest:
        m = json.loads(Path(manifest).read_text())
        if (m.get("dataset") != c["dataset"] or m.get("seed") != c["seed"] or m.get("split_kind") != "query"
                or m.get("eligible_query_ids_sha256") != digest):
            raise ValueError("Model manifest does not match the common dataset/query split")
        for name, p in models.items():
            if m.get("models", {}).get(name, {}).get("model_sha256") != _sha(p/"model.txt"):
                raise ValueError("Model manifest hash mismatch: " + name)
        status.update(status="provided_manifest_matches", manifest_sha256=_sha(manifest),
                      limitation="Training provenance is supplied by the manifest; this runner does not replay training.")
    return status


def package_results(run_dir, *, overwrite=False) -> Path:
    """Export an already validated run; never execute Java or change a gate.

    Legacy runs can say ``failed`` after a ZIP write error despite passing all
    numerical checks. The persisted parity flag and protocol digest distinguish
    that case from a failure during execution or answer validation.
    """
    out = Path(run_dir).expanduser().resolve(strict=True)
    state = json.loads((out/"manifest.json").read_text(encoding="utf-8"))
    if (state.get("status") not in ("complete", "failed")
            or state.get("exact_truth_parity") is not True
            or state.get("no_training") is not True
            or state.get("unchanged_H") is not True
            or state.get("model_weights_modified") is not False):
        raise ValueError("Export-only recovery requires completed numerical validation; do not mark an unfinished run complete")
    protocol_file = out/"protocol.json"
    if state.get("input_protocol_sha256") != _sha(protocol_file):
        raise ValueError("Frozen protocol hash differs; export-only recovery refused")
    protocol = json.loads(protocol_file.read_text(encoding="utf-8"))
    if (protocol.get("frozen_before_test_execution") is not True
            or protocol.get("no_test_threshold_selection") is not True
            or not protocol.get("files_sha256")):
        raise ValueError("Missing frozen test/calibration protocol; export-only recovery refused")
    required = ("summary.csv", "per_query.csv", "parity.csv", "timing.csv", "setup_times.csv",
                "payload_costs.csv", "raw_audit.csv", "frozen_cutoffs.csv", "calibration.csv")
    for name in required:
        table = pd.read_csv(out/name)
        if table.empty:
            raise ValueError("Incomplete result table: " + name)
        if name == "parity.csv":
            for column in ("same_H", "fallback_matches_exact", "stable_repeat"):
                if not table[column].astype(str).str.lower().eq("true").all():
                    raise ValueError("Numerical parity check failed: " + column)
    if not (out/"README.txt").is_file():
        raise FileNotFoundError("Missing README.txt; this is not the completed-results ZIP failure")
    archive = out/"09b_comparison_results.zip"
    if archive.exists() and not overwrite:
        raise FileExistsError(f"Archive already exists: {archive}; use overwrite=True for a partial failed export")
    # This is I/O verification only, outside all saved experiment timings.
    _check_hashes(protocol["files_sha256"])
    previous = json.loads(json.dumps(state))
    state.setdefault("export_recoveries", []).append(dict(
        requested_utc=datetime.now(timezone.utc).isoformat(),
        previous_status=state["status"], previous_error=state.get("error"),
        previous_phase=state.get("phase"), previous_traceback=state.get("traceback"),
        action="package saved results only; no Java, training, calibration, or retiming"))
    state.update(status="complete", phase="complete", archive_storage="local ZIP staging then sequential copy to result directory")
    state.pop("error", None)
    state.pop("traceback", None)
    try:
        _json(out/"manifest.json", state)
        return package_directory(out, overwrite=overwrite)
    except BaseException:
        # A failed recovery must not turn a failed export into a success.
        try:
            _json(out/"manifest.json", previous)
        except OSError:
            pass  # Preserve the original export exception on an unhealthy mount.
        raise


def run_comparison(config) -> Path:
    """Run the existing 09b workload with a shared Java calibration and E2E timer."""
    c = _config(config)
    if shutil.which(c["java"]) is None:
        raise FileNotFoundError("Java executable not found")
    import pyarrow  # noqa: F401 - fail before expensive source preparation
    cp_files = _classpath(c)
    models = _model_dirs(c)
    q, roles, ca_ids, test_ids, hists, rates = _workload(c)
    raw = sorted(Path(c["data_dir"]).glob("[0-9]*.csv"), key=lambda p: int(p.stem))
    c["num_files"] = len(raw)
    if not raw or [int(p.stem) for p in raw] != list(range(len(raw))):
        raise ValueError("Java raw files must be contiguous 0.csv..N-1.csv")
    for p in raw:
        with p.open() as f:
            if len(f.readline().rstrip().split("\t")) < 2:
                raise ValueError("Java needs TAB-separated numbered raw CSVs: " + str(p))
    root = Path(c["output_root"])
    if root == Path(c["repo_root"]) or Path(c["repo_root"]) in root.parents:
        raise ValueError("output_root must be outside the Git checkout")
    root.mkdir(parents=True, exist_ok=True)
    out = root/("09b-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid4().hex[:8])
    out.mkdir(exist_ok=False)
    state = dict(status="running", config=c, started_utc=datetime.now(timezone.utc).isoformat(),
                 headline_latency="median whole-batch JVM + input/model loads + queries + CSV-to-Parquet seconds",
                 instrumented_query_latency="gateNs+synNs+rawNs; excludes per-request I/O and batch setup",
                 cache="raw loaded once per JVM; OS cache uncontrolled, no cache flush",
                 zero_policy="truth>0: abs(answer-truth)/truth; both zero:0; truth0/answer!=0:Infinity",
                 calibration="live Java scores; common original calibration IDs; identical empirical tied-prefix rule",
                 no_training=True, unchanged_H=True)
    _json(out/"manifest.json", state)
    started = time.perf_counter()
    try:
        roles.to_csv(out/"splits.csv", index=False)
        q.loc[sorted(roles.source_qId)].to_csv(out/"query_labels_offline.csv", index=False)
        _write_queries(out/"calibration_queries.csv", q.loc[ca_ids])
        _write_queries(out/"test_queries.csv", q.loc[test_ids])
        shape, shape_info = None, None
        tick = time.perf_counter()
        if c["shape_enabled"]:
            shape = Path(c.get("shape_dir") or out/"source_shape").expanduser().resolve()
            if c.get("shape_dir"):
                shape_info = verify_shape_metadata(shape, run_dir=c["run_dir"])
            else:
                prepare_shape_metadata(c["run_dir"], c["data_dir"], shape,
                                       packet_budget_bytes=c["shape_packet_budget"], chunk_size=c["chunk_size"])
                shape_info = verify_shape_metadata(shape, run_dir=c["run_dir"])
            for r in shape_info["source_inputs"]:
                sid = int(r["source_id"])
                if sid >= len(raw) or _sha(raw[sid]) != r["raw_sha256"]:
                    raise ValueError("Source-shape metadata does not match this benchmark's raw files")
            if shape_info["source_count"] != len(raw):
                raise ValueError("Source-shape source count differs from Java raw files")
            if shape_info["packet_budget_bytes"] != c["shape_packet_budget"]:
                raise ValueError("Reused source-shape packet has a different declared byte cap")
        state["source_preparation_or_verification_seconds"] = time.perf_counter()-tick
        state["shape_metadata"] = str(shape) if shape else None
        state["source_metadata_constructed_this_run"] = bool(shape and not c.get("shape_dir"))
        state["source_metadata_original_construction_seconds"] = shape_info.get("construction_seconds") if shape_info else None
        state["raw_rows_validated_by_source_construction"] = shape_info.get("raw_rows") if shape_info else None
        state["original_model_training_seconds"] = None  # saved models; not retrained or timed here
        state["model_provenance"] = _model_provenance(c, models, roles)
        if state["model_provenance"]["status"] == "external_training_split_unverified":
            print("Model training split provenance is unverified; use the matching dataset and seed from notebook 06.")
        methods = ["skewness", *c["models"], *(["source_shape"] if shape else [])]
        # A deterministic TRAIN-only raw audit fails bad data/query pairings
        # early. These checks never choose or change any gate.
        train_ids = roles.loc[roles.split == "train", "source_qId"].to_numpy(int)
        audit_ids = train_ids[np.linspace(0, len(train_ids)-1, min(8, len(train_ids)), dtype=int)]
        _write_queries(out/"raw_audit_queries.csv", q.loc[audit_ids])
        tick = time.perf_counter()
        audit_answers, _ = _run_java(c, "exact", models, shape, out/"raw_audit_queries.csv", audit_ids,
                                     out/"raw_input_audit", mode="exact")
        expected_audit = q.loc[audit_ids].truth.to_numpy(float)
        raw_audit = pd.DataFrame(dict(source_qId=audit_ids, stored_truth=expected_audit,
                                     exact_raw_answer=audit_answers.answer,
                                     matches=audit_answers.answer.to_numpy() == expected_audit))
        raw_audit.to_csv(out/"raw_audit.csv", index=False)
        if not raw_audit.matches.all():
            raise ValueError("TRAIN raw audit differs from stored truth; check raw/query dataset identities")
        state["raw_input_audit_seconds"] = time.perf_counter()-tick
        source_files = sorted(Path(c["run_dir"]).glob("src_*.txt"))
        source_mass = sum(float(load_hist(p)[3].sum()) for p in source_files)
        for rate in rates:
            h_mass = float(load_hist(hists[rate])[3].sum())
            if not np.isclose(h_mass, source_mass, rtol=0, atol=1e-6+64*np.finfo(float).eps*source_mass):
                raise ValueError("Selected H mass differs from the exact source histograms")
        model_files = [f for p in models.values() for f in (p/"model.txt", p/"model.properties")]
        files = [*source_files, *hists.values(), *raw, *model_files, *cp_files,
                 Path(c["run_dir"])/"rows.csv", Path(c["dataset_dir"])/"queries-Ans.txt",
                 out/"splits.csv", out/"calibration_queries.csv", out/"test_queries.csv",
                 out/"query_labels_offline.csv", out/"calibration_queries.ids.csv", out/"test_queries.ids.csv"]
        if shape:
            files += [shape/"shape.bin", shape/"manifest.json", *sorted((shape/"native").iterdir())]
        if c.get("model_manifest"):
            files.append(Path(c["model_manifest"]))
        hashes = {str(f.resolve()): _sha(f) for f in files if f.is_file()}
        cal_records, cut_records = [], []
        cut_files, cal_H = {}, {}
        tick = time.perf_counter()
        for method in methods:
            for rid, rate in enumerate(rates):
                a, _ = _run_java(c, method, models, shape, out/"calibration_queries.csv", ca_ids,
                                 out/"calibration"/method/f"rate_{rid}", rate, mode="approx")
                H = a.answer.to_numpy(float)
                if rate in cal_H and not np.allclose(H, cal_H[rate], rtol=0, atol=1e-8):
                    raise ValueError("Calibration arms did not use the same H")
                cal_H[rate] = H
                error = relative_error(H, q.loc[ca_ids].truth)
                value = calibrated_cutoff(a.errorScore, error, c["eps"], c["delta"])
                a["setup"], a["actual_histogram_error"] = method, error
                cal_records.append(a)
                cut_records.append(dict(setup=method, mergeRate=rate, eps=c["eps"], delta=c["delta"],
                                        cutoff=value, calibration_queries=len(a)))
        pd.concat(cal_records, ignore_index=True).to_csv(out/"calibration.csv", index=False)
        cuts = pd.DataFrame(cut_records)
        cuts.to_csv(out/"frozen_cutoffs.csv", index=False, float_format="%.17g")
        for method in methods:
            p = out/"cutoffs"/(method+".csv")
            p.parent.mkdir(exist_ok=True)
            cuts[cuts.setup == method][["mergeRate", "eps", "delta", "cutoff"]].to_csv(p, index=False, float_format="%.17g")
            cut_files[method] = p
            hashes[str(p.resolve())] = _sha(p)
        hashes[str((out/"frozen_cutoffs.csv").resolve())] = _sha(out/"frozen_cutoffs.csv")
        hashes[str((out/"calibration.csv").resolve())] = _sha(out/"calibration.csv")
        state["common_calibration_seconds"] = time.perf_counter()-tick
        protocol = dict(frozen_before_test_execution=True, files_sha256=hashes, rates=rates,
                        calibration_ids=ca_ids, test_ids=list(map(int, test_ids)),
                        model_provenance=state["model_provenance"], no_test_threshold_selection=True,
                        code_sha256={p.name:_sha(p) for p in (Path(__file__), Path(__file__).with_name("shape_e2e_inputs.py"))})
        _json(out/"protocol.json", protocol)
        _check_hashes(hashes)
        # Reference H is produced by the same Java engine without truth or raw fallback.
        test_H = {}
        for rid, rate in enumerate(rates):
            a, _ = _run_java(c, "skewness", models, shape, out/"test_queries.csv", test_ids,
                             out/"histogram_answers"/f"rate_{rid}", rate, mode="approx")
            test_H[rate] = a.answer.to_numpy(float)
        jobs = []
        for repeat in range(c["repeats"]):
            batch = [(m, r, i) for i, r in enumerate(rates) for m in methods] + [("exact", None, 0)]
            np.random.default_rng(c["seed"]+repeat+91273).shuffle(batch)
            jobs += [dict(repeat=repeat, setup=m, rate=r, rate_index=i) for m, r, i in batch]
        _json(out/"execution_order.json", jobs)
        pd.DataFrame({"warm_up": [1.]}).to_parquet(out/"_warm_up.parquet", index=False)
        records, frames, setup_records = [], [], []
        for job in jobs:
            method, rate, rep = job["setup"], job["rate"], job["repeat"]
            a, record = _run_java(c, method, models, shape, out/"test_queries.csv", test_ids,
                                  out/"timed"/f"repeat_{rep}"/method/f"rate_{job['rate_index']}",
                                  rate, cut_files.get(method), mode="exact" if method == "exact" else "gated", parquet=True)
            record["repeat"] = rep
            records.append(record)
            a["setup"], a["repeat"] = method, rep
            frames.append(a)
            st = pd.read_csv(Path(record["folder"])/"setup.csv")
            setup_records.append(st.assign(setup=method, repeat=rep, merge_rate=rate))
            print(f"Repeat {rep}: {method}, rate={rate}, batch={record['total_s']:.3f}s")
        state["phase"] = "finalizing_results"
        timing = pd.DataFrame(records)
        timing.to_csv(out/"timing.csv", index=False)
        pd.concat(setup_records, ignore_index=True).to_csv(out/"setup_times.csv", index=False)
        all_answers = pd.concat(frames, ignore_index=True)
        truth = q.loc[test_ids].truth.to_numpy(float)
        exact = all_answers[all_answers.setup == "exact"]
        exact_answers = exact[exact.repeat == 0].answer.to_numpy(float)
        if not np.array_equal(exact_answers, truth):
            raise ValueError("Exact baseline differs from stored truth; check dataset/raw/query identities")
        parity, summaries, enriched = [], [], []
        for method in [*methods, "exact"]:
            for rate in ([None] if method == "exact" else rates):
                group = all_answers[all_answers.setup == method]
                if rate is not None:
                    group = group[np.isclose(group.mergeRate, rate, rtol=0, atol=1e-12)]
                H = test_H[rates[0]] if rate is None else test_H[rate]
                value = None if rate is None else float(cuts[(cuts.setup == method) & (cuts.mergeRate == rate)].cutoff.iloc[0])
                first = None
                for repeat, part in group.groupby("repeat", sort=True):
                    part = part.sort_values("qId").copy()
                    validate_answers(part, H, exact_answers, value, exact=method == "exact")
                    semantic = part[["answer", "fromSynopsis", "errorScore", "cutoff"]].to_numpy()
                    if first is not None and not pd.DataFrame(semantic).equals(pd.DataFrame(first)):
                        raise ValueError("Answers/scores/decisions changed across timing repeats")
                    first = semantic
                    part["H"], part["truth"] = H, truth
                    part["actual_histogram_error"] = relative_error(H, truth)
                    part["returned_error"] = relative_error(part.answer, truth)
                    part["instrumented_query_ms"] = part[["gateNs", "synNs", "rawNs"]].sum(axis=1)/1e6
                    enriched.append(part)
                    parity.append(dict(setup=method, merge_rate=rate, repeat=int(repeat), queries=len(part),
                                       same_H=True, fallback_matches_exact=True, stable_repeat=True))
                result = summarize_answers(part, H, truth, c["eps"])
                timed = timing[timing.setup == method]
                if rate is not None:
                    timed = timed[np.isclose(timed.rate.astype(float), rate, rtol=0, atol=1e-12)]
                compute = group[["gateNs", "synNs", "rawNs"]].sum(axis=1)/1e6
                result.update(setup=method, merge_rate=rate, eps=c["eps"], delta_target=c["delta"],
                              latency_s=float(timed.total_s.median()), latency_min_s=float(timed.total_s.min()),
                              latency_max_s=float(timed.total_s.max()), java_s=float(timed.java_s.median()),
                              parquet_s=float(timed.parquet_s.median()),
                              instrumented_query_mean_ms=float(compute.mean()), instrumented_query_p95_ms=float(compute.quantile(.95)),
                              gate_ms_per_query=float(group.gateNs.mean()/1e6), synopsis_ms_per_query=float(group.synNs.mean()/1e6),
                              raw_ms_per_query=float(group.rawNs.mean()/1e6), repeats=c["repeats"],
                              observed_within_target=(result["violation"] <= c["delta"]) if result["synopsis_queries"] else None)
                if method == "exact":
                    result["score_auc"] = np.nan
                summaries.append(result)
        summary = pd.DataFrame(summaries)
        exact_seconds = float(summary.loc[summary.setup == "exact", "latency_s"].iloc[0])
        summary["speedup_vs_exact"] = exact_seconds/summary.latency_s
        summary.to_csv(out/"summary.csv", index=False)
        pd.concat(enriched, ignore_index=True).to_csv(out/"per_query.csv", index=False)
        pd.DataFrame(parity).to_csv(out/"parity.csv", index=False)
        costs = []
        store_bytes = sum(p.stat().st_size for p in [*source_files, *hists.values()])
        for method in methods:
            model_bytes = sum((models[method]/f).stat().st_size for f in ("model.txt", "model.properties")) if method in models else 0
            aux = int(shape_info["auxiliary_bytes"]) if method == "source_shape" else 0
            for rate in rates:
                costs.append(dict(setup=method, merge_rate=rate, selected_H_bytes=hists[rate].stat().st_size,
                                  shared_store_bytes=store_bytes, model_bytes=model_bytes, shape_auxiliary_bytes=aux,
                                  cutoffs_bytes=cut_files[method].stat().st_size,
                                  loaded_artifact_bytes=store_bytes+model_bytes+aux+cut_files[method].stat().st_size,
                                  scope="actual shared Java SynopsisStore loads every ARM/source export; excludes JAR/raw/query/output files"))
        pd.DataFrame(costs).to_csv(out/"payload_costs.csv", index=False)
        _check_hashes(hashes)
        state.update(status="complete", phase="complete", finished_utc=datetime.now(timezone.utc).isoformat(),
                     comparison_seconds=time.perf_counter()-started, unique_timed_queries=len(test_ids),
                     calibration_queries=len(ca_ids), rates=rates, input_protocol_sha256=_sha(out/"protocol.json"),
                     versions={name:importlib.metadata.version(name) for name in ("numpy", "pandas", "pyarrow", "lightgbm", "scikit-learn")},
                     java_version=subprocess.run([c["java"], "-version"], capture_output=True, text=True).stderr,
                     environment=dict(platform=platform.platform(), machine=platform.machine(),
                                      logical_cpu_count=os.cpu_count(), python=platform.python_version(),
                                      databricks_runtime=os.environ.get("DATABRICKS_RUNTIME_VERSION", "not available")),
                     exact_truth_parity=True, model_weights_modified=False,
                     archive_storage="local ZIP staging then sequential copy to result directory",
                     output_zip_excludes="external raw files, external model weights and source/H inputs; their identities are frozen in protocol.json")
        _json(out/"manifest.json", state)
        (out/"README.txt").write_text(
            "09b shared Java source-shape comparison\n"
            "Read summary.csv with coverage and violation together. Zero accepted risk is undefined.\n"
            "latency_s is median whole-batch Java+loading+CSV-to-Parquet.\n"
            "instrumented_query_* excludes batch loading/output; it is not single-query wall latency.\n"
            "All arms use the same query geometry, H and Java calibration IDs/rule. No model was fitted.\n"
            "Check manifest.json model_provenance before making held-out/generalization claims.\n"
            "Raw data is loaded once per Java process and scanned for each exact/rejected query.\n"
            "No OS cache flushing or future-risk guarantee. Training/construction/calibration are outside timed batches.\n",
            encoding="utf-8")
        state["phase"] = "packaging_results"
        package_directory(out)
        print("Completed:", out)
        return out
    except BaseException as exc:
        state.update(status="failed", error=str(exc), traceback=traceback.format_exc())
        try:
            _json(out/"manifest.json", state)
        except OSError as metadata_error:
            print("Could not save failure metadata:", metadata_error)
        if hasattr(exc, "add_note"):
            exc.add_note(f"09b results: {out}; phase={state.get('phase', 'execution')}. "
                         "For a packaging-only failure, use package_results(run_dir, overwrite=True); do not rerun the timing cell.")
        raise


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument("--config", help="JSON settings; paths and models match the existing 09b run")
    source.add_argument("--package-only", metavar="RUN_DIR", help="Export an already validated run without rerunning queries")
    ap.add_argument("--overwrite-archive", action="store_true", help="Replace a partial ZIP during export-only recovery")
    args = ap.parse_args(argv)
    if args.package_only:
        print(package_results(args.package_only, overwrite=args.overwrite_archive))
    else:
        if args.overwrite_archive:
            ap.error("--overwrite-archive requires --package-only")
        print(run_comparison(json.loads(Path(args.config).read_text())))


if __name__ == "__main__":
    main()
