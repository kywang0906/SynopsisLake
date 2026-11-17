"""Portable R10 development and frozen confirmation for local shape metadata.

Allocation is source-only. The extra-byte cap is chosen on grouped inner TRAIN
validation, then models and calibration gates freeze before any new geometry or
raw label is produced. The density correction is classical quadrant raking;
this experiment does not establish novelty or a speed improvement.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shutil
import time
import traceback
from uuid import uuid4
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pandas as pd

from . import bounded_sidecar as bounded
from . import local_shape as shape
from . import shape_error_model as model
from . import sidecar_workflow as prior
from .sidecar_e2e import Arm, Query, ScanCounter, ScoreResult, run_benchmark


DEFAULTS = dict(phase="develop", family="raking", run_dir="", preparation_dir="", development_dir="",
    output_root="", raw_data_dir="", queries_file="", seeds=[0, 1, 2],
    extra_budgets=[12288, 49152, 1000000000], selection_budgets=[], train_queries=2048,
    validation_queries=512, calibration_queries=512, test_queries=256,
    include_ml=True, evaluate_old_test=True, chunk_size=250000,
    epsilon=.25, delta=.05, min_exact=10, parity_queries=8,
    fresh_query_count=0, fresh_query_seed=2026111801,
    timing_queries=0, timing_repeats=2, force_confirmation=False)
GEOMETRY = ["x0", "x1", "y0", "y1"]


def _cfg(config):
    cfg = {**DEFAULTS, **dict(config)}
    cfg["seeds"] = prior._list(cfg["seeds"], int)
    cfg["extra_budgets"] = prior._list(cfg["extra_budgets"], int)
    cfg["selection_budgets"] = prior._list(cfg["selection_budgets"], int)
    if not set(cfg["selection_budgets"]) <= set(cfg["extra_budgets"]):
        raise ValueError("selection_budgets must be a subset of measured extra_budgets")
    if cfg["family"] not in ("raking", "hierarchical"):
        raise ValueError("family must be raking or hierarchical")
    if not cfg["seeds"] or len(set(cfg["seeds"])) != len(cfg["seeds"]):
        raise ValueError("seeds must be unique and nonempty")
    if not cfg["extra_budgets"] or len(set(cfg["extra_budgets"])) != len(cfg["extra_budgets"]):
        raise ValueError("extra_budgets must be unique and nonempty")
    for key in ("train_queries", "validation_queries", "calibration_queries", "test_queries",
                "chunk_size", "parity_queries", "fresh_query_count", "timing_queries", "timing_repeats"):
        if isinstance(cfg[key], bool) or int(cfg[key]) != cfg[key] or int(cfg[key]) < 0:
            raise ValueError(f"{key} must be a nonnegative integer")
        cfg[key] = int(cfg[key])
    if not cfg["chunk_size"] or not cfg["parity_queries"] or not cfg["timing_repeats"]:
        raise ValueError("chunk_size, parity_queries and timing_repeats must be positive")
    cfg["output_root"] = str(Path(cfg["output_root"] or Path(__file__).resolve().parents[2]/"synopsislake-results").expanduser().resolve())
    return cfg


def _new(root, prefix):
    path = Path(root)/(prefix+"-"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")+"-"+uuid4().hex[:8])
    path.mkdir(parents=True, exist_ok=False)
    return path


def _write_state(path, state):
    prior._json(Path(path)/"run.json", state)


def _fail(path, state, exc):
    state.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    _write_state(path, state)
    (Path(path)/"failure.txt").write_text(traceback.format_exc())


def _implementation():
    names = ("local_shape_workflow.py", "local_shape.py", "shape_error_model.py",
             "sidecar_workflow.py", "sidecar_e2e.py", "bounded_sidecar.py", "_distribution_math.py", "features.py")
    paths = [Path(__file__).with_name(name) for name in names]
    hierarchy = Path(__file__).with_name("hierarchical_shape.py")
    if hierarchy.is_file():
        paths.append(hierarchy)
    return {p.name: prior._sha(p) for p in paths}


def _backend(cfg):
    if cfg["family"] == "hierarchical":
        from . import hierarchical_shape
        return hierarchical_shape
    return shape


def _r09(cfg):
    root = Path(cfg["run_dir"]).expanduser().resolve()
    if not (root/"run.json").is_file():
        raise FileNotFoundError(f"Set run_dir to a completed R09 experiment: {root}")
    state = json.loads((root/"run.json").read_text())
    if state.get("status") != "complete" or state.get("study") != "bounded-sidecar-e2e":
        raise ValueError("run_dir must be a completed bounded-sidecar-e2e R09 run")
    for relative in ("payloads/original_H.hist", "payloads/structured_reference", "outer_splits.csv", "input_manifest.json"):
        if not (root/relative).exists():
            raise FileNotFoundError(f"Required R09 artifact missing: {root/relative}")
    return root, state


def prepare_experiment(config) -> Path:
    """Parse real raw sources once; persist source quadrants and private raw caches."""
    cfg = _cfg(config)
    if cfg["preparation_dir"]:
        path = Path(cfg["preparation_dir"]).expanduser().resolve()
        _load_preparation(path)
        return path
    path = _new(cfg["output_root"], "local-shape-inputs")
    state = dict(status="preparing", study="local-shape-preparation", config=cfg,
                 started_utc=datetime.now(timezone.utc).isoformat())
    _write_state(path, state)
    tick = time.perf_counter()
    try:
        source, old = _r09(cfg)
        native = path/"native"
        shutil.copytree(source/"payloads/structured_reference", native)
        shutil.copyfile(source/"payloads/original_H.hist", path/"original_H.hist")
        shutil.copyfile(source/"outer_splits.csv", path/"outer_splits.csv")
        hists, encoded = shape.load_native(native)
        original = prior.load_histogram(path/"original_H.hist")
        queries_path = Path(cfg["queries_file"] or old["config"]["queries_file"]).expanduser().resolve()
        queries = prior._load_queries(queries_path, old["config"].get("query_metadata_file", ""))
        outer = pd.read_csv(path/"outer_splits.csv")
        selected = set(outer.loc[outer.seed.isin(cfg["seeds"]), "qId"])
        omitted = queries[~queries.qId.isin(selected)]
        if (omitted.truth >= cfg["min_exact"]).any():
            raise ValueError("Inherited outer splits omit eligible queries")
        queries = queries[queries.qId.isin(selected)].copy()
        queries = queries[["qId", *GEOMETRY, "truth"]]
        prior._splits(queries, {**old["config"], "seeds": cfg["seeds"]}, path/"outer_splits.csv")
        queries.to_csv(path/"queries_offline.csv", index=False)
        cache = path/"raw_cache"
        cache.mkdir()
        quad_dir = path/"quadrants"
        quad_dir.mkdir()
        manifest = json.loads((source/"input_manifest.json").read_text())
        identities = {str(Path(item["path"]).resolve()): item for item in manifest}
        raw_dir = Path(cfg["raw_data_dir"] or old["config"]["raw_data_dir"]).expanduser().resolve()
        audit = []
        for sid, hist in enumerate(hists):
            raw = next((p for p in (raw_dir/f"{sid}.csv", raw_dir/f"{sid}.csv.gz") if p.is_file()), None)
            if raw is None:
                raise FileNotFoundError(f"Missing real raw source {raw_dir/f'{sid}.csv'} or .csv.gz")
            before = (raw.stat().st_size, raw.stat().st_mtime_ns)
            digest = prior._sha(raw)
            expected = identities.get(str(raw), next((v for k, v in identities.items() if Path(k).name == raw.name), None))
            if expected is None or digest != expected["sha256"]:
                raise ValueError(f"Raw source differs from the inherited R09 identity: {raw}")
            rows = int(hist[3].sum())
            points_path = cache/f"source_{sid}.npy"
            points = np.lib.format.open_memmap(points_path, mode="w+", dtype=np.float64, shape=(rows, 2))
            quadrants = np.zeros((hist[3].size, 4), np.uint64)
            position, started = 0, time.perf_counter()
            for chunk in prior.iter_raw(raw, cfg["chunk_size"]):
                if position+len(chunk) > rows:
                    raise ValueError("Raw rows exceed inherited source mass")
                points[position:position+len(chunk)] = chunk
                if ((chunk[:, 0] < original[1][0]) | (chunk[:, 0] > original[1][-1]) |
                        (chunk[:, 1] < original[2][0]) | (chunk[:, 1] > original[2][-1])).any():
                    raise ValueError("Real raw source lies outside original H root")
                position += len(chunk)
                quadrants += shape.quadrant_counts_from_points(chunk, hist)
            if position != rows or not np.array_equal(quadrants.sum(axis=1), hist[3].ravel()):
                raise ValueError("Source quadrant counts do not reproduce each inherited cell mass")
            if before != (raw.stat().st_size, raw.stat().st_mtime_ns):
                raise ValueError("Raw input changed during construction")
            points.flush()
            del points
            np.save(quad_dir/f"source_{sid}.npy", quadrants, allow_pickle=False)
            audit.append(dict(source_id=sid, raw_path=str(raw), raw_sha256=digest, raw_rows=rows,
                              source_cells=hist[3].size, quadrant_mass_matches=True,
                              raw_cache_bytes=points_path.stat().st_size,
                              raw_cache_sha256=prior._sha(points_path),
                              quadrant_cache_bytes=(quad_dir/f"source_{sid}.npy").stat().st_size,
                              construction_seconds=time.perf_counter()-started))
            print(f"R10 source {sid}: {rows:,} real rows, exact quadrant masses", flush=True)
        pd.DataFrame(audit).to_csv(path/"source_audit.csv", index=False)
        if int(original[3].sum()) != sum(item["raw_rows"] for item in audit):
            raise ValueError("Native/H/raw total masses are incompatible")
        counter = ScanCounter(_raw_arrays(path))
        parity = []
        indexed_queries = queries.set_index("qId")
        for qid in prior._select(queries.qId, cfg["parity_queries"], 0, "r10-source-parity"):
            row = indexed_queries.loc[qid]
            value = counter.count(row[GEOMETRY].to_numpy(float))
            parity.append(dict(qId=qid, raw_count=value, stored_truth=int(row.truth), matches=value == int(row.truth)))
        pd.DataFrame(parity).to_csv(path/"raw_truth_parity.csv", index=False)
        if not all(row["matches"] for row in parity):
            raise ValueError("Offline query truth failed independent closed-bound raw parity scans")
        state.update(status="prepared", source_r09_run=str(source), source_r09_manifest_sha256=prior._sha(source/"run.json"),
                     queries_file=str(queries_path), raw_rows=sum(item["raw_rows"] for item in audit),
                     native_bytes=sum(p.stat().st_size for p in native.iterdir()),
                     original_H_bytes=(path/"original_H.hist").stat().st_size,
                     preparation_seconds=time.perf_counter()-tick,
                     input_sha256={str(p.relative_to(path)): prior._sha(p) for p in [path/"original_H.hist", path/"outer_splits.csv", path/"queries_offline.csv", path/"source_audit.csv", path/"raw_truth_parity.csv", *native.iterdir(), *quad_dir.iterdir()]})
        _write_state(path, state)
        return path
    except BaseException as exc:
        _fail(path, state, exc)
        raise


def _load_preparation(path):
    path = Path(path).resolve()
    state = json.loads((path/"run.json").read_text())
    if state.get("status") != "prepared":
        raise ValueError("preparation_dir must be a completed R10 preparation")
    for relative, digest in state["input_sha256"].items():
        if prior._sha(path/relative) != digest:
            raise ValueError(f"Prepared source-only input changed: {relative}")
    return path, state


def _raw_arrays(preparation):
    preparation = Path(preparation)
    audit = pd.read_csv(preparation/"source_audit.csv")
    arrays = []
    for row in audit.sort_values("source_id").itertuples(index=False):
        path = preparation/"raw_cache"/f"source_{row.source_id}.npy"
        if path.stat().st_size != row.raw_cache_bytes or prior._sha(path) != row.raw_cache_sha256:
            raise ValueError(f"Raw coordinate cache identity changed: {path}")
        points = np.load(path, mmap_mode="r", allow_pickle=False)
        if points.dtype != np.float64 or points.shape != (int(row.raw_rows), 2) or not np.isfinite(points).all():
            raise ValueError(f"Raw coordinate cache shape/dtype/values changed: {path}")
        arrays.append(points)
    return arrays


def _children(preparation, hists, quadrants):
    """Reusable source-only child bytes, built from verified caches without CSV parsing."""
    from . import hierarchical_shape as hierarchy
    preparation = Path(preparation)
    folder = preparation/"child_moment_cache"
    manifest = folder/"manifest.json"
    identity = {"prepared_manifest": prior._sha(preparation/"run.json"),
                "source_audit": prior._sha(preparation/"source_audit.csv")}
    if manifest.is_file():
        saved = json.loads(manifest.read_text())
        if saved["source_identity"] != identity:
            raise ValueError("Cached child moments belong to different source inputs")
        for name, digest in saved["files_sha256"].items():
            if prior._sha(folder/name) != digest:
                raise ValueError("Source child-moment cache identity changed")
    else:
        folder.mkdir(exist_ok=True)
        for sid, (points, hist, expected) in enumerate(zip(_raw_arrays(preparation), hists, quadrants)):
            counts = np.zeros_like(expected)
            sums = np.zeros(counts.shape+(5,), float)
            for start in range(0, len(points), 250000):
                count, additive = hierarchy.accumulate_child_moments(points[start:start+250000], hist)
                counts += count
                sums += additive
            if not np.array_equal(counts, expected):
                raise ValueError("Child moments and prepared source quadrant counts disagree")
            np.save(folder/f"source_{sid}.npy", hierarchy.encode_child_moments(counts, sums), allow_pickle=False)
        prior._json(manifest, dict(source_identity=identity,
            files_sha256={p.name: prior._sha(p) for p in folder.glob("source_*.npy")},
            source_only=True, raw_csv_parsing=False))
    return [np.load(folder/f"source_{sid}.npy", allow_pickle=False) for sid in range(len(hists))]


def _base_query_features(rectangles, components, original, answers=None):
    rectangles = np.asarray(rectangles, float)
    reference = bounded.batch_reference(components, rectangles)
    full_mass, boundary_mass = np.zeros(len(rectangles)), np.zeros(len(rectangles))
    b, mass = components.bounds, components.mass
    for i, q in enumerate(rectangles):
        touched = (b[:, 0] <= q[1]) & (b[:, 1] >= q[0]) & (b[:, 2] <= q[3]) & (b[:, 3] >= q[2])
        full = (q[0] <= b[:, 0]) & (b[:, 1] <= q[1]) & (q[2] <= b[:, 2]) & (b[:, 3] <= q[3])
        full_mass[i], boundary_mass[i] = mass[full].sum(), mass[touched & ~full].sum()
    return dict(reference_base=reference, reference_hybrid=reference,
                full_mass=full_mass, boundary_mass=boundary_mass,
                corrected_boundary_fraction=np.zeros(len(rectangles)),
                H=np.array([prior.histogram_count(q, original) for q in rectangles]) if answers is None else np.asarray(answers))


def model_features(rectangles, described, root, *, hybrid=False):
    """The explicit BASE8/HYBRID12 whitelist; no IDs, group or truth columns."""
    rects = np.asarray(rectangles, float)
    H = np.asarray(described["H"], float)
    base = np.asarray(described["reference_base"], float)
    reference = np.asarray(described["reference_hybrid"] if hybrid else base, float)
    width = (rects[:, 1]-rects[:, 0])/(root[1]-root[0])
    height = (rects[:, 3]-rects[:, 2])/(root[3]-root[2])
    boundary = np.asarray(described["boundary_mass"], float)
    total = boundary+np.asarray(described["full_mass"], float)
    frame = pd.DataFrame(dict(log_H=np.log1p(H), log_reference=np.log1p(reference),
        proxy_score=np.abs(H-reference)/np.maximum(reference, 1.), query_width=width,
        query_height=height, query_area=width*height, query_aspect=width/height,
        partial_mass_fraction=np.divide(boundary, total, out=np.zeros_like(boundary), where=total > 0)))
    if hybrid:
        frame["log_base_reference"] = np.log1p(base)
        frame["base_proxy_score"] = np.abs(H-base)/np.maximum(base, 1.)
        frame["reference_shift"] = (reference-base)/np.maximum(base, 1.)
        frame["corrected_mass_fraction"] = described["corrected_boundary_fraction"]
    expected = model.HYBRID_FEATURES if hybrid else model.BASE_FEATURES
    if tuple(frame.columns) != tuple(expected) or not np.isfinite(frame.to_numpy()).all():
        raise ValueError("Live feature implementation differs from the model whitelist")
    return frame


def _inner(queries, outer, seed, cfg):
    train = outer[(outer.seed == seed) & (outer.split == "train")].copy().sort_values("qId")
    if "cell" not in train:
        raise ValueError("R10 requires inherited source-cell groups for inner validation")
    fit, validation, info = model._inner_split(len(train), seed, train.cell.to_numpy())
    fit_ids = prior._select(train.iloc[fit].qId, cfg["train_queries"], seed, "r10-inner-fit")
    val_ids = prior._select(train.iloc[validation].qId, cfg["validation_queries"], seed, "r10-inner-validation")
    if not fit_ids or not val_ids:
        raise ValueError("Inner fit and validation must be nonempty")
    indexed = queries.set_index("qId")
    fit_frame = indexed.loc[fit_ids].reset_index().merge(train[["qId", "cell"]], on="qId", validate="one_to_one")
    val_frame = indexed.loc[val_ids].reset_index().merge(train[["qId", "cell"]], on="qId", validate="one_to_one")
    if set(fit_frame.cell) & set(val_frame.cell):
        raise ValueError("Grouped inner fit/validation cells overlap")
    assignments = pd.concat((fit_frame[["qId", "cell"]].assign(role="fit"), val_frame[["qId", "cell"]].assign(role="validation")))
    return fit_frame, val_frame, assignments.assign(seed=seed), info


def _metrics(qids, truth, answers, score, seed, method, family="inner_validation", epsilon=.25):
    errors = prior._actual_error(answers, truth)
    finite = np.isfinite(errors)
    return dict(seed=seed, method=method, family=family, n_queries=len(truth),
                mae=float(np.mean(abs(np.asarray(score)-errors))),
                mae_pp=float(100*np.mean(abs(np.asarray(score)-errors))),
                auc_bad=prior._auc(errors > epsilon, score), nonfinite_actual_errors=int((~finite).sum()))


def _fine_controls(caps, arrays, root, folder, chunk_size):
    folder = Path(folder)
    folder.mkdir(exist_ok=True)
    controls = {}
    for cap in sorted(set(map(int, caps))):
        n = math.isqrt((cap-64)//4)
        if n < 1:
            raise ValueError("Matched finer-control cap cannot hold a cell")
        controls[cap] = dict(n=n, counts=np.zeros((n, n), np.uint64))
    for points in arrays:
        for start in range(0, len(points), chunk_size):
            chunk = points[start:start+chunk_size]
            for item in controls.values():
                n = item["n"]
                ix = np.clip(((chunk[:, 0]-root[0])/(root[1]-root[0])*n).astype(np.int64), 0, n-1)
                iy = np.clip(((chunk[:, 1]-root[2])/(root[3]-root[2])*n).astype(np.int64), 0, n-1)
                item["counts"] += np.bincount(ix*n+iy, minlength=n*n).reshape(n, n).astype(np.uint64)
    for cap, item in controls.items():
        path = folder/f"cap_{cap}.counts"
        size = prior._save_fine(path, item.pop("counts"), root)
        item.update(path=str(path), auxiliary_bytes=size, matched_cap_bytes=cap, unused_bytes=cap-size)
    return controls


def _copy_inputs(preparation, destination):
    shutil.copytree(Path(preparation)/"native", destination/"native")
    for name in ("original_H.hist", "outer_splits.csv", "source_audit.csv"):
        shutil.copyfile(Path(preparation)/name, destination/name)


def develop_experiment(config) -> Path:
    """Choose the source-only packet cap using outer-TRAIN inner validation only."""
    cfg = _cfg(config)
    preparation = prepare_experiment(cfg)
    _, prepared = _load_preparation(preparation)
    path = _new(Path(preparation)/"development", "development")
    state = dict(status="developing", study="local-shape-development", config=cfg,
                 preparation_dir=str(preparation), started_utc=datetime.now(timezone.utc).isoformat())
    _write_state(path, state)
    started = time.perf_counter()
    try:
        _copy_inputs(preparation, path)
        hists, encoded = shape.load_native(path/"native")
        original = prior.load_histogram(path/"original_H.hist")
        root = (original[1][0], original[1][-1], original[2][0], original[2][-1])
        components = prior.load_structured_reference(path/"native")
        queries = pd.read_csv(Path(preparation)/"queries_offline.csv", float_precision="round_trip")
        outer = pd.read_csv(path/"outer_splits.csv")
        quad = [np.load(p, allow_pickle=False) for p in sorted((Path(preparation)/"quadrants").glob("source_*.npy"))]
        backend = _backend(cfg)
        children = _children(preparation, hists, quad) if cfg["family"] == "hierarchical" else None
        packets_dir = path/"packets"
        packets_dir.mkdir()
        packets, allocations = {}, []
        for budget in cfg["extra_budgets"]:
            payload = (backend.build_selection(hists, encoded, quad, children, budget) if children is not None
                       else backend.build_selection(hists, encoded, quad, budget))
            packet_path = packets_dir/f"extra_{budget}.bin"
            packet_path.write_bytes(payload)
            runtime = backend.load_runtime(hists, encoded, payload)
            packets[budget] = dict(path=str(packet_path), payload_bytes=len(payload),
                                   auxiliary_bytes=runtime.auxiliary_bytes,
                                   selected_cells=len(runtime.selection.selected_ids))
            audit = pd.DataFrame(backend.selection_audit(hists, encoded, quad, children, payload) if children is not None
                                 else backend.selection_audit(hists, encoded, quad, payload))
            audit.assign(extra_budget=budget).to_csv(packets_dir/f"allocation_{budget}.csv", index=False)
            allocations.append(dict(extra_budget=budget, packet_bytes=len(payload),
                                    native_bytes=runtime.native_bytes, auxiliary_bytes=runtime.auxiliary_bytes,
                                    selected_cells=len(runtime.selection.selected_ids),
                                    packet_sha256=prior._sha(packet_path)))
        pd.DataFrame(allocations).to_csv(path/"allocation_summary.csv", index=False)
        arrays = _raw_arrays(preparation)
        fine = _fine_controls([prepared["native_bytes"], *[p["auxiliary_bytes"] for p in packets.values()]], arrays, root, path/"fine", cfg["chunk_size"])
        inners, assignment_rows, rows, predictions = {}, [], [], []
        base_cache, hybrid_cache = {}, {}
        for seed in cfg["seeds"]:
            fit, validation, assignments, split_info = _inner(queries, outer, seed, cfg)
            inners[seed] = (fit, validation, split_info)
            assignment_rows.append(assignments)
            rects, truth = validation[GEOMETRY].to_numpy(float), validation.truth.to_numpy(float)
            described = _base_query_features(rects, components, original)
            base_cache[seed] = described
            answers = described["H"]
            actual = prior._actual_error(answers, truth)
            base_score = np.abs(answers-described["reference_base"])/np.maximum(described["reference_base"], 1.)
            rows.append(_metrics(validation.qId, truth, answers, base_score, seed, "R08", epsilon=cfg["epsilon"]))
            predictions.append(pd.DataFrame(dict(seed=seed, method="R08", qId=validation.qId,
                truth=truth, H=answers, reference_count=described["reference_base"], score=base_score, actual_error=actual)))
            base_fine = prior._load_fine(fine[prepared["native_bytes"]]["path"])
            base_fine_reference = np.array([prior.histogram_count(q, base_fine) for q in rects])
            rows.append(_metrics(validation.qId, truth, answers, np.abs(answers-base_fine_reference)/np.maximum(base_fine_reference, 1.),
                                 seed, "fine_R08", epsilon=cfg["epsilon"]))
            for budget, packet in packets.items():
                runtime = backend.load_runtime(hists, encoded, Path(packet["path"]).read_bytes())
                described = runtime.query_features(rects)
                described["H"] = answers
                hybrid_cache[seed, budget] = described
                for mode, column in (("hybrid", "reference_hybrid"), ("uniform", "reference_uniform")):
                    reference = described[column]
                    score = np.abs(answers-reference)/np.maximum(reference, 1.)
                    name = f"{mode}_{budget}"
                    rows.append(_metrics(validation.qId, truth, answers, score, seed, name, epsilon=cfg["epsilon"]))
                    predictions.append(pd.DataFrame(dict(seed=seed, method=name, qId=validation.qId,
                        truth=truth, H=answers, reference_count=reference, score=score, actual_error=actual)))
                hist = prior._load_fine(fine[packet["auxiliary_bytes"]]["path"])
                reference = np.array([prior.histogram_count(q, hist) for q in rects])
                score = np.abs(answers-reference)/np.maximum(reference, 1.)
                rows.append(_metrics(validation.qId, truth, answers, score, seed, f"fine_{budget}", epsilon=cfg["epsilon"]))
                predictions.append(pd.DataFrame(dict(seed=seed, method=f"fine_{budget}", qId=validation.qId,
                    truth=truth, H=answers, reference_count=reference, score=score, actual_error=actual)))
            print(f"R10 seed {seed}: {len(fit)} fit / {len(validation)} grouped inner validation", flush=True)
        pd.concat(assignment_rows, ignore_index=True).to_csv(path/"inner_assignments.csv", index=False)
        metrics = pd.DataFrame(rows)
        metrics.to_csv(path/"validation_metrics.csv", index=False)
        validation_predictions = pd.concat(predictions, ignore_index=True)
        validation_predictions.to_csv(path/"validation_predictions.csv", index=False)
        _validation_coverage(validation_predictions, cfg).to_csv(path/"validation_fixed_coverage.csv", index=False)
        macro = metrics.groupby("method", as_index=False).agg(mae=("mae", "mean"), auc_bad=("auc_bad", "mean"))
        macro.to_csv(path/"validation_macro.csv", index=False)
        score_by_method = macro.set_index("method").mae
        eligible_budgets = cfg["selection_budgets"] or cfg["extra_budgets"]
        candidate = min(eligible_budgets, key=lambda b: (score_by_method[f"hybrid_{b}"], packets[b]["payload_bytes"], b))
        chosen = candidate if score_by_method[f"hybrid_{candidate}"] < score_by_method["R08"] else 0
        promising = bool(chosen and score_by_method[f"hybrid_{candidate}"] < score_by_method[f"fine_{candidate}"])
        selection = dict(status="selected_on_inner_validation", selected_extra_budget=chosen,
                         candidate_extra_budget=candidate,
                         selected_packet_bytes=packets[chosen]["payload_bytes"] if chosen else 0,
                         selected_auxiliary_bytes=packets[chosen]["auxiliary_bytes"] if chosen else prepared["native_bytes"],
                         eligible_selection_budgets=eligible_budgets,
                         measured_upper_controls=[b for b in cfg["extra_budgets"] if b not in eligible_budgets],
                         selection_rule="minimum macro-seed grouped inner-validation proxy MAE including R08 with zero correction bytes; packet bytes then budget break ties",
                         promising=promising, promising_rule="selected hybrid macro MAE strictly below R08 and byte-matched fine",
                         source_only_allocation=True, source_protocol=backend.protocol(), family=cfg["family"],
                         base_mae=float(score_by_method["R08"]), hybrid_mae=float(score_by_method[f"hybrid_{candidate}"]),
                         matched_fine_mae=float(score_by_method[f"fine_{candidate}"]))
        prior._json(path/"selection.json", selection)
        state.update(status="allocation_selected", selected_extra_budget=chosen, candidate_extra_budget=candidate, promising=promising,
                     packets=packets, fine_controls=fine, development_seconds=time.perf_counter()-started)
        _write_state(path, state)
        if cfg["include_ml"]:
            _fit_controls(path, cfg, inners, hists, encoded, components, original, root, packets[candidate])
        _freeze_and_evaluate(path, cfg, preparation, queries, outer, selection)
        state = json.loads((path/"run.json").read_text())
        state.update(status="complete", development_seconds=time.perf_counter()-started)
        _write_state(path, state)
        _report(path)
        package_results(path)
        return path
    except BaseException as exc:
        _fail(path, state, exc)
        raise


def _fit_controls(path, cfg, inners, hists, encoded, components, original, root, packet):
    """Fit on supplied outer TRAIN only; optional explicit inner roles are audited."""
    metadata = []
    runtime = _backend(cfg).load_runtime(hists, encoded, Path(packet["path"]).read_bytes())
    for seed, (fit, validation, _) in inners.items():
        train = pd.concat((fit, validation), ignore_index=True)
        rects = train[GEOMETRY].to_numpy(float)
        described = runtime.query_features(rects)
        described["H"] = np.array([prior.histogram_count(q, original) for q in rects])
        target = prior._actual_error(described["H"], train.truth.to_numpy(float))
        for hybrid, name in ((False, "R08_tinyML"), (True, "hybrid_tinyML")):
            features = model_features(rects, described, root, hybrid=hybrid)
            features["actual_error"], features["split"], features["cell"] = target, "train", train.cell.to_numpy()
            features["qId"] = train.qId.to_numpy()
            info = model.fit_models(features, model.HYBRID_FEATURES if hybrid else model.BASE_FEATURES,
                                    seed, path/"models"/f"seed_{seed}"/name, groups="cell",
                                    inner_fit_indices=np.arange(len(fit)),
                                    inner_validation_indices=np.arange(len(fit), len(train)))
            info.update(seed=seed, method=name, train_qIds=train.qId.tolist())
            metadata.append(info)
    prior._json(path/"model_metadata.json", metadata)


def _validation_coverage(predictions, cfg):
    rows = []
    for (seed, method), part in predictions.groupby(["seed", "method"], sort=False):
        score, err = part.score.to_numpy(), part.actual_error.to_numpy()
        for fraction in (.1, .25, .5, 1.):
            k = max(1, math.ceil(fraction*len(part)))
            threshold = np.sort(score)[k-1]
            accepted = score <= threshold
            rows.append(dict(seed=seed, method=method, nominal_coverage=fraction,
                actual_coverage=float(accepted.mean()), selected_queries=int(accepted.sum()),
                extra_tied_queries=int(accepted.sum())-k, violation=float((err[accepted] > cfg["epsilon"]).mean())))
    return pd.DataFrame(rows)


def _freeze_and_evaluate(path, cfg, preparation, queries, outer, selection):
    """Finish the concrete frozen package before exposing old test outcomes."""
    state = json.loads((path/"run.json").read_text())
    native_bytes = sum(p.stat().st_size for p in (path/"native").iterdir())
    packet = state["packets"][str(state["candidate_extra_budget"])]
    caps = [native_bytes, packet["auxiliary_bytes"]]
    metadata = json.loads((path/"model_metadata.json").read_text()) if (path/"model_metadata.json").is_file() else []
    for info in metadata:
        caps.append((packet["auxiliary_bytes"] if info["method"] == "hybrid_tinyML" else native_bytes)+info["model_bytes"])
    original = prior.load_histogram(path/"original_H.hist")
    root = (original[1][0], original[1][-1], original[2][0], original[2][-1])
    fine = _fine_controls(caps, _raw_arrays(preparation), root, path/"fine", cfg["chunk_size"])
    state["fine_controls"].update({str(k): v for k, v in fine.items()})
    _write_state(path, state)
    if metadata:
        _model_validation(path, cfg, queries, metadata, state)
    indexed = queries.set_index("qId")
    calibration_rows, selected_rows, gate_rows, costs = [], [], [], []
    for seed in cfg["seeds"]:
        arms = load_arms(path, seed)
        cal_ids = prior._select(outer.loc[(outer.seed == seed) & (outer.split == "calibration"), "qId"], cfg["calibration_queries"], seed, "r10-calibration")
        test_ids = prior._select(outer.loc[(outer.seed == seed) & (outer.split == "test"), "qId"], cfg["test_queries"], seed, "r10-test")
        calibration = indexed.loc[cal_ids].reset_index()
        scores = _score_live(calibration[["qId", *GEOMETRY]], arms, original, seed, "calibration")
        scored = prior._attach_truth(scores, calibration)
        calibration_rows.append(scored)
        for arm in arms:
            part = scored[scored.method == arm.name]
            gate_rows.append(dict(seed=seed, method=arm.name, epsilon=cfg["epsilon"], delta=cfg["delta"],
                cutoff=prior._cutoff(part.score.to_numpy(), part.actual_error.to_numpy(), cfg["epsilon"], cfg["delta"]),
                calibration_queries=len(part)))
            costs.append({"seed": seed, "method": arm.name, **dict(arm.metadata or {})})
        selected_rows.extend(dict(seed=seed, qId=qid, role=role) for role, ids in (("calibration", cal_ids), ("test", test_ids)) for qid in ids)
    pd.concat(calibration_rows, ignore_index=True).to_csv(path/"calibration_predictions.csv", index=False)
    pd.DataFrame(gate_rows).to_csv(path/"frozen_gate_cutoffs.csv", index=False)
    pd.DataFrame(selected_rows).to_csv(path/"selected_query_ids.csv", index=False)
    pd.DataFrame(costs).to_csv(path/"payload_costs.csv", index=False)
    protocol = dict(status="frozen_before_test", config=cfg, selection_sha256=prior._sha(path/"selection.json"),
        implementation_sha256=_implementation(), allocation_reads_query_truth=False,
        model_training="only supplied inherited outer TRAIN; grouped inner-validation selection followed by outer TRAIN refit",
        calibration="only original inherited outer calibration; largest complete score tie prefix satisfying empirical delta",
        fresh_labels_generated=False, source_protocol=_backend(cfg).protocol(),
        source_preparation_manifest_sha256=prior._sha(Path(preparation)/"run.json"),
        files_sha256={str(p.relative_to(path)): prior._sha(p) for p in path.rglob("*")
                      if p.is_file() and p.name not in {"run.json", "results.zip"} and "feature_cache" not in p.parts})
    prior._json(path/"protocol.json", protocol)
    state.update(status="frozen", protocol_sha256=prior._sha(path/"protocol.json"), evaluation_pending=False)
    _write_state(path, state)
    verify_protocol_lock(path)

    if cfg["evaluate_old_test"]:
        predictions, metrics, coverage, timing = [], [], [], []
        arrays = _raw_arrays(preparation) if cfg["timing_queries"] else None
        for seed in cfg["seeds"]:
            test_ids = [x["qId"] for x in selected_rows if x["seed"] == seed and x["role"] == "test"]
            test = indexed.loc[test_ids].reset_index()
            scores = _score_live(test[["qId", *GEOMETRY]], load_arms(path, seed), original, seed, "test")
            quality = pd.concat((calibration_rows[cfg["seeds"].index(seed)], prior._attach_truth(scores, test)), ignore_index=True)
            quality["family"] = "old_test_exploratory"
            m, c, _ = prior._quality(quality, seed, cfg["epsilon"], cfg["delta"], cfg["min_exact"])
            metrics.append(m.assign(family="old_test_exploratory"))
            coverage.append(c.assign(family="old_test_exploratory"))
            predictions.append(quality)
            if cfg["timing_queries"]:
                cal_ids = [x["qId"] for x in selected_rows if x["seed"] == seed and x["role"] == "calibration"]
                timed = _timing(path, path, seed, test, indexed.loc[cal_ids].reset_index(), original, arrays, cfg, "old_test_exploratory")
                timing.append(timed.summary.assign(seed=seed, family="old_test_exploratory"))
        pd.concat(predictions, ignore_index=True).to_csv(path/"per_query_quality.csv", index=False)
        pd.concat(metrics, ignore_index=True).to_csv(path/"metrics.csv", index=False)
        pd.concat(coverage, ignore_index=True).to_csv(path/"fixed_coverage.csv", index=False)
        if timing:
            pd.concat(timing, ignore_index=True).to_csv(path/"e2e_summary.csv", index=False)
    verify_protocol_lock(path)


def _model_validation(path, cfg, queries, metadata, state):
    """Use saved pre-refit predictions; final models never score their validation labels."""
    original = prior.load_histogram(path/"original_H.hist")
    indexed = queries.set_index("qId")
    native_bytes = sum(p.stat().st_size for p in (path/"native").iterdir())
    packet = state["packets"][str(state["candidate_extra_budget"])]
    rows, prediction_rows = [], []
    for info in metadata:
        heldout_path = path/"models"/f"seed_{info['seed']}"/info["method"]/"inner_validation_predictions.csv"
        heldout = pd.read_csv(heldout_path, float_precision="round_trip")
        ids = heldout.qId.astype(np.int64).to_numpy()
        part = indexed.loc[ids].reset_index()
        rects, truth = part[GEOMETRY].to_numpy(float), part.truth.to_numpy(float)
        answers = np.array([prior.histogram_count(q, original) for q in rects])
        actual = prior._actual_error(answers, truth)
        if not np.allclose(actual, heldout.actual_error.to_numpy(float), rtol=1e-12, atol=1e-12):
            raise ValueError("Saved pre-refit validation target differs from unchanged original H error")
        score = heldout.prediction.to_numpy(float)
        rows.append(_metrics(ids, truth, answers, score, info["seed"], info["method"], epsilon=cfg["epsilon"]))
        prediction_rows.append(pd.DataFrame(dict(seed=info["seed"], method=info["method"], qId=ids,
            truth=truth, H=answers, score=score, actual_error=actual)))
        cap = (packet["auxiliary_bytes"] if info["method"] == "hybrid_tinyML" else native_bytes)+info["model_bytes"]
        item = state["fine_controls"][str(cap)]
        hist = prior._load_fine(path/"fine"/Path(item["path"]).name)
        reference = np.array([prior.histogram_count(q, hist) for q in rects])
        score = np.abs(answers-reference)/np.maximum(reference, 1.)
        fine_name = "fine_for_"+info["method"]
        rows.append(_metrics(ids, truth, answers, score, info["seed"], fine_name, epsilon=cfg["epsilon"]))
        prediction_rows.append(pd.DataFrame(dict(seed=info["seed"], method=fine_name, qId=ids,
            truth=truth, H=answers, reference_count=reference, score=score, actual_error=actual)))
    metrics = pd.DataFrame(rows)
    predictions = pd.concat(prediction_rows, ignore_index=True)
    metrics.to_csv(path/"model_validation_metrics.csv", index=False)
    predictions.to_csv(path/"model_validation_predictions.csv", index=False)
    _validation_coverage(predictions, cfg).to_csv(path/"model_validation_fixed_coverage.csv", index=False)
    combined = pd.concat((pd.read_csv(path/"validation_metrics.csv"), metrics), ignore_index=True)
    combined.to_csv(path/"validation_metrics.csv", index=False)
    combined.groupby("method", as_index=False).agg(mae=("mae", "mean"), mae_pp=("mae_pp", "mean"), auc_bad=("auc_bad", "mean")).to_csv(path/"validation_macro.csv", index=False)


def _score_live(geometry, arms, original, seed, role):
    if set(geometry) != {"qId", *GEOMETRY}:
        raise ValueError("Scoring accepts geometry only")
    rows = []
    for arm in arms:
        artifact = arm.artifact_loader() if arm.artifact_loader else None
        for row in geometry.itertuples(index=False):
            rect = np.array((row.x0, row.x1, row.y0, row.y1), float)
            answer = prior.histogram_count(rect, original)
            value = (arm.score_fn(rect, answer, artifact, original) if arm.score_uses_histogram
                     else arm.score_fn(rect, answer, artifact))
            value = value if isinstance(value, ScoreResult) else ScoreResult(float(value))
            if not np.isfinite(value.score) or value.score < 0:
                raise ValueError("Assessor returned a nonfinite or negative score")
            rows.append(dict(seed=seed, method=arm.name, split=role, qId=int(row.qId),
                x0=row.x0, x1=row.x1, y0=row.y0, y1=row.y1,
                H=answer, score=float(value.score), reference_count=value.reference))
    return pd.DataFrame(rows)


def load_arms(development_dir, seed):
    """Live native/packet/model loaders suitable for the R09 E2E engine."""
    path = Path(development_dir)
    state = json.loads((path/"run.json").read_text())
    native = path/"native"
    native_bytes = sum(p.stat().st_size for p in native.iterdir())
    packet_info = state["packets"][str(state["candidate_extra_budget"])]
    packet_path = path/"packets"/Path(packet_info["path"]).name
    original_bytes = (path/"original_H.hist").stat().st_size
    def runtime_loader():
        hists, encoded = shape.load_native(native)
        return _backend(state["config"]).load_runtime(hists, encoded, packet_path.read_bytes())
    def ref_score(rect, answer, components):
        reference = bounded.reference_count(components, rect)
        return ScoreResult(prior._proxy(answer, reference), reference)
    def correction_score(rect, answer, runtime, *, uniform=False):
        reference = runtime.reference_count(rect, mode="empirical_uniform" if uniform else "conditional_r08")
        return ScoreResult(prior._proxy(answer, reference), reference)
    def cost(auxiliary, packet_bytes=0, model_bytes=0, **extra):
        return dict(original_H_bytes=original_bytes, native_bytes=native_bytes,
                    packet_bytes=packet_bytes, model_schema_and_weights_bytes=model_bytes,
                    auxiliary_bytes=auxiliary, total_serialized_bytes=original_bytes+auxiliary, **extra)
    arms = [Arm("R08", ref_score, artifact_loader=lambda: prior.load_structured_reference(native),
                metadata=cost(native_bytes)),
            Arm("hybrid", correction_score, artifact_loader=runtime_loader,
                metadata=cost(packet_info["auxiliary_bytes"], packet_info["payload_bytes"])),
            Arm("empirical_uniform", lambda rect, answer, rt: correction_score(rect, answer, rt, uniform=True),
                artifact_loader=runtime_loader,
                metadata=cost(packet_info["auxiliary_bytes"], packet_info["payload_bytes"]))]
    models = json.loads((path/"model_metadata.json").read_text()) if (path/"model_metadata.json").is_file() else []
    for info in models:
        if info["seed"] != seed:
            continue
        hybrid = info["method"] == "hybrid_tinyML"
        def loader(info=info, hybrid=hybrid):
            schema = path/"models"/f"seed_{seed}"/info["method"]/"model.schema.json"
            return (runtime_loader() if hybrid else prior.load_structured_reference(native), model.load_model(schema))
        def score(rect, answer, artifact, histogram, hybrid=hybrid):
            rect = np.asarray(rect, dtype=float)
            runtime, predictor = artifact
            described = runtime.query_features(rect[None, :]) if hybrid else _base_query_features(rect[None, :], runtime, histogram, [answer])
            described["H"] = np.array([answer])
            root = (histogram[1][0], histogram[1][-1], histogram[2][0], histogram[2][-1])
            features = model_features(rect[None, :], described, root, hybrid=hybrid)
            return ScoreResult(predictor.predict_row(features.iloc[0]))
        auxiliary = (packet_info["auxiliary_bytes"] if hybrid else native_bytes)+info["model_bytes"]
        arms.append(Arm(info["method"], score, artifact_loader=loader, score_uses_histogram=True,
                        metadata=cost(auxiliary, packet_info["payload_bytes"] if hybrid else 0, info["model_bytes"])))
    # Each individual matched control charges its actual compact header/count bytes.
    fine = state["fine_controls"]
    for target in list(arms):
        if target.name == "empirical_uniform":
            continue
        cap = target.metadata["auxiliary_bytes"]
        item = fine[str(cap)]
        file = path/"fine"/Path(item["path"]).name
        def fine_score(rect, answer, hist):
            reference = prior.histogram_count(rect, hist)
            return ScoreResult(prior._proxy(answer, reference), reference)
        arms.append(Arm("fine_for_"+target.name, fine_score,
            artifact_loader=lambda p=file: prior._load_fine(p),
            metadata=cost(item["auxiliary_bytes"], matched_target=target.name,
                          matched_cap_bytes=cap, unused_bytes=item["unused_bytes"])))
    return arms


def verify_protocol_lock(development_dir):
    path = Path(development_dir)
    state = json.loads((path/"run.json").read_text())
    if prior._sha(path/"protocol.json") != state["protocol_sha256"]:
        raise ValueError("Frozen R10 protocol changed")
    protocol = json.loads((path/"protocol.json").read_text())
    if protocol["implementation_sha256"] != _implementation():
        raise ValueError("R10 implementation changed since freeze")
    for relative, digest in protocol["files_sha256"].items():
        if prior._sha(path/relative) != digest:
            raise ValueError(f"Frozen R10 artifact changed: {relative}")
    return protocol


def verify_confirmation_lock(confirmation_dir):
    path = Path(confirmation_dir)
    state = json.loads((path/"run.json").read_text())
    if prior._sha(path/"protocol.json") != state["protocol_sha256"]:
        raise ValueError("Frozen confirmation protocol changed")
    protocol = json.loads((path/"protocol.json").read_text())
    if protocol["implementation_sha256"] != _implementation():
        raise ValueError("Frozen confirmation implementation changed")
    if prior._sha(path/"frozen_gate_cutoffs.csv") != protocol["frozen_gate_cutoffs_sha256"]:
        raise ValueError("Frozen confirmation gate values changed")
    verify_protocol_lock(state["development_dir"])
    if (path/"fresh_score_lock.json").is_file():
        lock = json.loads((path/"fresh_score_lock.json").read_text())
        if lock["protocol_sha256"] != state["protocol_sha256"]:
            raise ValueError("Fresh score lock refers to another confirmation protocol")
        for name, key in (("fresh_queries_unlabeled.csv", "geometry_sha256"),
                          ("fresh_scores_before_labels.csv", "scores_sha256")):
            if prior._sha(path/name) != lock[key]:
                raise ValueError(f"Frozen fresh artifact changed: {name}")
    elif state.get("status") == "complete" or (path/"fresh_scores_before_labels.csv").exists() or (path/"fresh_queries_unlabeled.csv").exists():
        raise ValueError("Fresh geometry/scores require their immutable score lock")
    return protocol


def _timing(output, development, seed, test, calibration, original, arrays, cfg, family):
    development = Path(development)
    ids = prior._select(test.qId, cfg["timing_queries"], seed, "r10-timing-"+family)
    timed = test.set_index("qId").loc[ids].reset_index()
    def records(frame):
        return [Query(int(q.qId), (q.x0, q.x1, q.y0, q.y1), int(q.truth)) for q in frame.itertuples(index=False)]
    gates = pd.read_csv(development/"frozen_gate_cutoffs.csv")
    return run_benchmark(queries=records(timed), calibration_queries=records(calibration),
        arms=load_arms(development, seed), histogram_loader=lambda: prior.load_histogram(development/"original_H.hist"),
        estimate_fn=prior.histogram_count, raw_counter=ScanCounter(arrays),
        output_dir=Path(output)/"e2e"/f"seed_{seed}"/family, repeats=cfg["timing_repeats"],
        epsilon=cfg["epsilon"], delta=cfg["delta"], seed=seed,
        frozen_cutoffs=gates[gates.seed == seed].set_index("method").to_dict("index"),
        frozen_cutoff_provenance=dict(path=str(development/"frozen_gate_cutoffs.csv"), sha256=prior._sha(development/"frozen_gate_cutoffs.csv")),
        metadata=dict(family=family, backend="Python live scorers and real vector scan fallback", no_speed_claim=True))


def confirm_experiment(config) -> Path:
    cfg = _cfg(config)
    development = Path(cfg["development_dir"]).expanduser().resolve()
    protocol = verify_protocol_lock(development)
    old = json.loads((development/"run.json").read_text())
    if old.get("status") != "complete":
        raise ValueError("development_dir must be a complete frozen R10 development run")
    selection = json.loads((development/"selection.json").read_text())
    if not selection["promising"] and not cfg["force_confirmation"]:
        raise ValueError("Development did not meet the frozen validation gain criterion; fresh confirmation is not authorized by that criterion")
    if cfg["fresh_query_count"] < 1000:
        raise ValueError("Fresh confirmation requires fresh_query_count>=1000")
    path = _new(development/"confirmation", "confirmation")
    state = dict(status="frozen_before_geometry", study="local-shape-confirmation", config=cfg,
                 development_dir=str(development), development_protocol_sha256=old["protocol_sha256"],
                 independent_dataset=False, forced_exploratory=not selection["promising"])
    _write_state(path, state)
    try:
        preparation = Path(old["preparation_dir"])
        _load_preparation(preparation)
        if prior._sha(preparation/"source_audit.csv") != prior._sha(development/"source_audit.csv"):
            raise ValueError("Prepared raw-cache identities differ from the frozen development audit")
        cfg = {**old["config"], **{key: cfg[key] for key in ("fresh_query_count", "fresh_query_seed", "timing_queries", "timing_repeats", "force_confirmation")}}
        state["config"] = cfg
        original = prior.load_histogram(development/"original_H.hist")
        root = (original[1][0], original[1][-1], original[2][0], original[2][-1])
        queries = pd.read_csv(preparation/"queries_offline.csv", float_precision="round_trip")
        assignments = pd.read_csv(development/"inner_assignments.csv")
        donor_seed = cfg["seeds"][0]
        donors = assignments[assignments.seed == donor_seed].qId.to_numpy()
        donor_rects = queries.set_index("qId").loc[donors, GEOMETRY].to_numpy(float)
        shutil.copyfile(development/"frozen_gate_cutoffs.csv", path/"frozen_gate_cutoffs.csv")
        prior._json(path/"protocol.json", dict(status="frozen_before_fresh_geometry", config=cfg,
            development_protocol_sha256=old["protocol_sha256"], implementation_sha256=_implementation(),
            frozen_gate_cutoffs_sha256=prior._sha(path/"frozen_gate_cutoffs.csv"),
            donor_training_qIds=donors.tolist(), generator="sample TRAIN width/height pairs and uniformly shift inside unchanged H root",
            selection_locked=True, new_dataset_claim=False))
        state["protocol_sha256"] = prior._sha(path/"protocol.json")
        _write_state(path, state)
        rectangles = prior._fresh_rectangles(cfg["fresh_query_count"], root, cfg["fresh_query_seed"], donor_rects)
        geometry = pd.DataFrame(rectangles, columns=GEOMETRY)
        geometry.insert(0, "qId", np.arange(len(geometry))+int(queries.qId.max())+1)
        geometry.to_csv(path/"fresh_queries_unlabeled.csv", index=False)
        scores = [_score_live(geometry, load_arms(development, seed), original, seed, "test") for seed in cfg["seeds"]]
        pd.concat(scores, ignore_index=True).to_csv(path/"fresh_scores_before_labels.csv", index=False)
        prior._json(path/"fresh_score_lock.json", dict(protocol_sha256=state["protocol_sha256"],
            geometry_sha256=prior._sha(path/"fresh_queries_unlabeled.csv"),
            scores_sha256=prior._sha(path/"fresh_scores_before_labels.csv")))
        verify_confirmation_lock(path)
        arrays = _raw_arrays(preparation)
        labels, seconds = prior._indexed_labels(arrays, rectangles)
        labeled = geometry.assign(truth=labels)
        labeled.to_csv(path/"fresh_queries_labeled_offline.csv", index=False)
        counter = ScanCounter(arrays)
        audits = []
        for row in labeled.head(3).itertuples(index=False):
            exact = int(counter.count((row.x0, row.x1, row.y0, row.y1)))
            audits.append(dict(qId=row.qId, indexed_truth=int(row.truth), brute_raw_truth=exact, matches=exact == int(row.truth)))
        pd.DataFrame(audits).to_csv(path/"fresh_raw_label_audit.csv", index=False)
        if not all(item["matches"] for item in audits):
            raise ValueError("Fresh indexed labels disagree with independent exact scans")
        gates = pd.read_csv(path/"frozen_gate_cutoffs.csv")
        metrics, coverage, predictions, timing = [], [], [], []
        for seed, scored in zip(cfg["seeds"], scores):
            quality = prior._attach_truth(scored, labeled)
            # Use the saved gate values directly: no recalibration on fresh truth.
            m, c = _quality_frozen(quality, gates[gates.seed == seed], cfg)
            metrics.append(m.assign(seed=seed, family="fresh_same_dataset"))
            coverage.append(c.assign(seed=seed, family="fresh_same_dataset"))
            predictions.append(quality.assign(family="fresh_same_dataset"))
            if cfg["timing_queries"]:
                selected = pd.read_csv(development/"selected_query_ids.csv")
                ids = selected[(selected.seed == seed) & (selected.role == "calibration")].qId
                cal = queries.set_index("qId").loc[ids].reset_index()
                timing.append(_timing(path, development, seed, labeled, cal, original, arrays, cfg, "fresh_same_dataset").summary.assign(seed=seed, family="fresh_same_dataset"))
        pd.concat(metrics, ignore_index=True).to_csv(path/"metrics.csv", index=False)
        pd.concat(coverage, ignore_index=True).to_csv(path/"fixed_coverage.csv", index=False)
        pd.concat(predictions, ignore_index=True).to_csv(path/"per_query_quality.csv", index=False)
        if timing:
            pd.concat(timing, ignore_index=True).to_csv(path/"e2e_summary.csv", index=False)
        verify_confirmation_lock(path)
        state.update(status="complete", offline_indexed_label_seconds=seconds,
                     raw_rows=sum(len(p) for p in arrays), fresh_queries=len(labeled))
        _write_state(path, state)
        (path/"analysis_en.md").write_text("# Frozen fresh-query confirmation\n\nAll method packets, models, byte caps and gates were frozen in the referenced development protocol. New geometry and every method score were persisted before exact indexed raw labels. Raw label audits use independent scans. These are fresh rectangles on the same inspected dataset, not a new population. Fixed-coverage rows are descriptive ranking diagnostics; empirical calibration does not guarantee future risk. Timing, when enabled, executes live scoring and genuine raw scan fallback.\n")
        package_results(path)
        return path
    except BaseException as exc:
        _fail(path, state, exc)
        raise


def _quality_frozen(quality, gates, cfg):
    rows, fixed = [], []
    cutoffs = gates.set_index("method").cutoff
    for method, part in quality.groupby("method", sort=False):
        for cohort, selected in (("all_queries", part), ("truth_ge_min_exact", part[part.truth >= cfg["min_exact"]]),
                                 ("truth_lt_min_exact", part[part.truth < cfg["min_exact"]]), ("truth_zero", part[part.truth == 0])):
            if not len(selected):
                continue
            err, score = selected.actual_error.to_numpy(), selected.score.to_numpy()
            accepted = score <= cutoffs[method]
            rows.append(dict(method=method, cohort=cohort, n_queries=len(selected),
                mae_pp=float(100*np.abs(score-err).mean()), auc_bad=prior._auc(err > cfg["epsilon"], score),
                coverage=float(accepted.mean()), accepted_queries=int(accepted.sum()),
                violation=float((err[accepted] > cfg["epsilon"]).mean()) if accepted.any() else np.nan,
                cutoff=cutoffs[method], nonfinite_actual_errors=int((~np.isfinite(err)).sum())))
            for fraction in (.1, .25, .5, 1.):
                k = max(1, math.ceil(fraction*len(selected)))
                threshold = np.sort(score)[k-1]
                keep = score <= threshold
                fixed.append(dict(method=method, cohort=cohort, nominal_coverage=fraction,
                    actual_coverage=float(keep.mean()), selected_queries=int(keep.sum()),
                    extra_tied_queries=int(keep.sum())-k, violation=float((err[keep] > cfg["epsilon"]).mean())))
    return pd.DataFrame(rows), pd.DataFrame(fixed)


def _report(path):
    state = json.loads((Path(path)/"run.json").read_text())
    selection = json.loads((Path(path)/"selection.json").read_text())
    method = "classical hierarchical child moment distributions" if state["config"]["family"] == "hierarchical" else "classical source-quadrant density raking"
    (Path(path)/"analysis_en.md").write_text(
        f"# Local shape development\n\nThis run evaluates {method}; no novelty or speed claim follows. "
        "Allocation reads only native cell moments and exact source quadrant counts. Its extra-byte cap was selected "
        "on grouped inner validation inside inherited outer TRAIN. Calibration and test truth selected no allocation.\n\n"
        f"Selected extra cap: {selection['selected_extra_budget']:,} bytes; actual packet: {selection['selected_packet_bytes']:,} bytes. "
        f"Macro inner-validation MAE: R08 {selection['base_mae']:.6f}, hybrid {selection['hybrid_mae']:.6f}, "
        f"matched fine {selection['matched_fine_mae']:.6f}. Promising under the declared criterion: {selection['promising']}.\n\n"
        "Validation comparisons are development evidence after candidate selection, not independent confirmation. "
        "Three seeds reuse the same real dataset. Native R08, packet, model text and model schema bytes are separate charges. "
        "Full-cap controls report actual materialized bytes, not their requested ceiling. Raw coordinate caches remain outside results.zip.\n")


def package_results(run_dir):
    path = Path(run_dir)
    archive = path/"results.zip"
    with ZipFile(archive, "w", ZIP_DEFLATED) as zipped:
        for item in sorted(path.rglob("*")):
            parts = item.relative_to(path).parts
            if item.is_file() and item != archive and not set(parts) & {"raw_cache", "feature_cache"}:
                zipped.write(item, item.relative_to(path))
    return archive


def run_experiment(config) -> Path:
    cfg = _cfg(config)
    if cfg["phase"] == "prepare":
        return prepare_experiment(cfg)
    if cfg["phase"] == "develop":
        return develop_experiment(cfg)
    if cfg["phase"] == "confirm":
        return confirm_experiment(cfg)
    raise ValueError("phase must be prepare, develop or confirm")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    for key, value in DEFAULTS.items():
        option = "--"+key.replace("_", "-")
        if isinstance(value, bool):
            parser.add_argument(option, action=argparse.BooleanOptionalAction, default=None)
        else:
            parser.add_argument(option, type=str if isinstance(value, list) else type(value), default=None)
    args = vars(parser.parse_args(argv))
    config_path = args.pop("config")
    config = json.loads(Path(config_path).read_text()) if config_path else {}
    config.update({k: v for k, v in args.items() if v is not None})
    print(run_experiment(config))


if __name__ == "__main__":
    main()
