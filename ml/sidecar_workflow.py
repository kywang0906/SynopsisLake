"""Real-data bounded sidecar experiment; no fitting or model dependencies.

Run ``python -m ml.sidecar_workflow --help``.  The compressor receives only
serialized summaries and training rectangles.  Raw arrays belong to metadata
construction, offline label checks, and the separately timed fallback engine.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import shutil
import struct
import time
import traceback
from uuid import uuid4
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pandas as pd
from scipy.special import betainc
from scipy.stats import rankdata

from .features import load_hist
from . import bounded_sidecar as bounded
from .sidecar_e2e import Arm, Query, ScanCounter, ScoreResult, run_benchmark


DEFAULTS = dict(
    output_root="", histogram_dir="", raw_data_dir="", queries_file="",
    query_metadata_file="", split_file="", split_kind="", run_dir="",
    base_resolution=130, budget_bytes=54032, seeds=[0],
    source_partition_factor=1, modes=["geometry", "workload"],
    merge_trees=["balanced", "left", "right"], workload_queries=32,
    calibration_queries=32, test_queries=64, timing_queries=12,
    timing_repeats=3, epsilon=.25, delta=.05, min_exact=10,
    parity_queries=8, chunk_size=250000, fresh_query_count=0,
    fresh_query_seed=731,
)
_HIST_HEADER = struct.Struct("<IId")
_MOMENT_HEADER = struct.Struct("<8sBBHIII4d8x")
_FINE_HEADER = struct.Struct("<8sII4d16x")


def _json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False,
                                    default=_json_default) + "\n")


def _json_default(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _list(value, convert=str):
    if isinstance(value, str):
        value = value.split(",")
    return [convert(x) for x in value]


def _config(config):
    cfg = {**DEFAULTS, **dict(config)}
    repo = Path(cfg.get("repo_root") or Path(__file__).resolve().parents[1]).expanduser().resolve()
    cfg["repo_root"] = str(repo)
    cfg["output_root"] = str(Path(cfg["output_root"] or repo.parent / "synopsislake-results").expanduser().resolve())
    for key, convert in (("seeds", int), ("modes", str), ("merge_trees", str)):
        cfg[key] = _list(cfg[key], convert)
        if not cfg[key] or len(set(cfg[key])) != len(cfg[key]):
            raise ValueError(f"{key} must be a nonempty list without duplicates")
    if not set(cfg["modes"]) <= {"geometry", "workload"}:
        raise ValueError("modes must contain geometry and/or workload")
    if not set(cfg["merge_trees"]) <= {"balanced", "left", "right"}:
        raise ValueError("merge_trees must contain balanced, left, and/or right")
    for key in ("budget_bytes", "source_partition_factor", "workload_queries", "calibration_queries",
                "test_queries", "timing_queries", "timing_repeats", "parity_queries", "chunk_size",
                "fresh_query_count", "min_exact"):
        value = cfg[key]
        if int(value) != value or int(value) < 0:
            raise ValueError(f"{key} must be a nonnegative integer")
        cfg[key] = int(value)
    if cfg["budget_bytes"] < bounded.HEADER_BYTES + bounded.COMPONENT_BYTES:
        raise ValueError("budget_bytes cannot hold one sidecar component and its header")
    if cfg["source_partition_factor"] not in (1, 4, 16):
        raise ValueError("source_partition_factor must be 1, 4, or 16 (4/16/64 virtual leaves)")
    if not cfg["chunk_size"] or not cfg["timing_repeats"] or not cfg["parity_queries"]:
        raise ValueError("chunk_size, timing_repeats, and parity_queries must be positive")
    if not 0 < float(cfg["epsilon"]) or not 0 <= float(cfg["delta"]) < 1:
        raise ValueError("epsilon must be positive and delta in [0,1)")
    return cfg


def _manifest_paths(run_dir):
    """Resolve existing artifact paths from prior manifests without requiring models."""
    found = []
    if not run_dir:
        return found
    root = Path(run_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Prior run_dir does not exist: {root}")
    found.append(root)
    for name in ("run.json", "input_manifest.json", "confirmation_config.json"):
        path = root / name
        if not path.is_file():
            continue
        def visit(value):
            if isinstance(value, dict):
                for item in value.values():
                    visit(item)
            elif isinstance(value, list):
                for item in value:
                    visit(item)
            elif isinstance(value, str) and ("/" in value or value.endswith((".txt", ".csv"))):
                p = Path(value).expanduser()
                for candidate in (p, root / p):
                    if candidate.exists():
                        found.append(candidate.resolve())
        visit(json.loads(path.read_text()))
    return list(dict.fromkeys(found))


def _resolve_inputs(cfg):
    prior = _manifest_paths(cfg["run_dir"])
    repo = Path(cfg["repo_root"])
    if cfg["histogram_dir"]:
        hist_dir = Path(cfg["histogram_dir"]).expanduser().resolve()
    else:
        choices = [p if p.is_dir() else p.parent for p in prior]
        choices += [p / "histograms" for p in choices]
        hist_dir = next((p for p in choices if list(p.glob("src_*.txt")) and list(p.glob("arm_*.txt"))), None)
        if hist_dir is None:
            raise FileNotFoundError("Set histogram_dir to real src_<id>.txt and arm_<resolution>.txt exports; "
                                    f"no histogram directory was found through run_dir={cfg['run_dir']!r}")
    source_paths = sorted(hist_dir.glob("src_*.txt"), key=lambda p: int(p.stem.split("_")[-1]))
    if not source_paths:
        raise FileNotFoundError(f"No source histogram exports: {hist_dir / 'src_*.txt'}")
    base_path = hist_dir / f"arm_{cfg['base_resolution']}.txt"
    if not base_path.is_file():
        arms = sorted(hist_dir.glob("arm_*.txt"))
        if len(arms) == 1:
            base_path = arms[0]
        else:
            raise FileNotFoundError(f"Original H is missing: {base_path}; set base_resolution explicitly")
    raw_dir = Path(cfg["raw_data_dir"] or repo / "range_query").expanduser().resolve()
    raw_paths = []
    for source in source_paths:
        sid = int(source.stem.split("_")[-1])
        path = next((p for p in (raw_dir / f"{sid}.csv", raw_dir / f"{sid}.csv.gz") if p.is_file()), None)
        if path is None:
            raise FileNotFoundError(f"Real raw source missing: {raw_dir / f'{sid}.csv'} or {raw_dir / f'{sid}.csv.gz'}")
        raw_paths.append(path)
    if cfg["queries_file"]:
        queries = Path(cfg["queries_file"]).expanduser().resolve()
    else:
        candidates = [p for p in prior if p.is_file() and p.name in ("queries-Ans.txt", "srcQuery-40-56-32-48-Ans.csv")]
        candidates += [raw_dir / "dataDrivenQuery/srcQuery-40-56-32-48-Ans.csv"]
        queries = next((p for p in candidates if p.is_file()), candidates[-1])
    if not queries.is_file():
        raise FileNotFoundError(f"Query bounds and offline truth file missing: {queries}")
    split = Path(cfg["split_file"]).expanduser().resolve() if cfg["split_file"] else None
    if split is None:
        candidates = []
        for p in prior:
            root = p if p.is_dir() else p.parent
            candidates += [root / "outer_splits.csv", root / "splits.csv"]
        split = next((p for p in candidates if p.is_file()), None)
    if split is not None and not split.is_file():
        raise FileNotFoundError(f"Inherited split file missing: {split}")
    return hist_dir, source_paths, base_path, raw_paths, queries, split


def _validate_hist(hist, *, exact=False):
    rate, bx, by, counts = hist
    if (not np.isfinite(rate) or rate < 0 or bx.ndim != 1 or by.ndim != 1
            or len(bx) < 2 or len(by) < 2 or counts.shape != (len(bx)-1, len(by)-1)
            or not np.isfinite(np.r_[bx, by, counts.ravel()]).all()
            or (np.diff(bx) <= 0).any() or (np.diff(by) <= 0).any() or (counts < 0).any()):
        raise ValueError("Invalid histogram geometry/counts")
    if exact and not np.equal(counts, np.floor(counts)).all():
        raise ValueError("Source histograms must contain exact integer counts")
    return hist


def save_histogram(path, hist):
    """R08 SLHIST01 codec, including every float64 boundary and count."""
    rate, bx, by, counts = _validate_hist(hist)
    with Path(path).open("wb") as f:
        f.write(b"SLHIST01")
        f.write(_HIST_HEADER.pack(*counts.shape, float(rate)))
        for array in (bx, by, counts):
            f.write(np.asarray(array, dtype="<f8").tobytes(order="C"))
    return Path(path).stat().st_size


def load_histogram(path):
    payload = Path(path).read_bytes()
    if len(payload) < 24 or payload[:8] != b"SLHIST01":
        raise ValueError(f"Invalid histogram payload: {path}")
    nx, ny, rate = _HIST_HEADER.unpack_from(payload, 8)
    if not nx or not ny or len(payload) != 24 + 8*(nx+ny+2+nx*ny):
        raise ValueError(f"Invalid histogram payload dimensions: {path}")
    values = np.frombuffer(payload, dtype="<f8", offset=24)
    return _validate_hist((rate, values[:nx+1].copy(), values[nx+1:nx+ny+2].copy(),
                           values[nx+ny+2:].reshape(nx, ny).copy()))


def iter_raw(path, chunk_size=250000):
    """The repository's TAB-separated raw x/y/timestamp files, including gzip."""
    for frame in pd.read_csv(path, sep=r"\s+", header=None, usecols=[0, 1],
                             dtype=np.float64, chunksize=chunk_size, compression="infer",
                             float_precision="round_trip", skip_blank_lines=False):
        points = np.ascontiguousarray(frame.to_numpy(), dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
            raise ValueError(f"Invalid raw coordinates: {path}")
        yield points


def _save_fine(path, counts, root):
    if counts.max(initial=0) > np.iinfo(np.uint32).max:
        raise OverflowError("Finer raw-count cell exceeds uint32 codec")
    Path(path).write_bytes(_FINE_HEADER.pack(b"SLCNT001", counts.shape[0], 1, *root)
                          + np.asarray(counts, dtype="<u4").tobytes(order="C"))
    return Path(path).stat().st_size


def _load_fine(path):
    payload = Path(path).read_bytes()
    if len(payload) < 64:
        raise ValueError("Truncated compact finer-grid payload")
    magic, n, version, x0, x1, y0, y1 = _FINE_HEADER.unpack_from(payload)
    if (magic != b"SLCNT001" or version != 1 or not n or len(payload) != 64+4*n*n
            or payload[48:64] != bytes(16) or not x0 < x1 or not y0 < y1):
        raise ValueError("Invalid compact finer-grid payload")
    counts = np.frombuffer(payload, dtype="<u4", offset=64).reshape(n, n).copy()
    return _validate_hist((0., np.linspace(x0, x1, n+1), np.linspace(y0, y1, n+1), counts))


def _sufficient(points, hist):
    """R08 arithmetic-floor ownership and normalized additive moments."""
    _, bx, by, coarse = _validate_hist(hist, exact=True)
    nx, ny = coarse.shape
    if (not np.allclose(np.diff(bx), (bx[-1]-bx[0])/nx, rtol=1e-11, atol=0)
            or not np.allclose(np.diff(by), (by[-1]-by[0])/ny, rtol=1e-11, atol=0)):
        raise ValueError("Moment accumulation requires the original regular source grid")
    if ((points[:, 0] < bx[0]) | (points[:, 0] >= bx[-1]) |
            (points[:, 1] < by[0]) | (points[:, 1] >= by[-1])).any():
        raise ValueError("Raw point lies outside the original half-open source domain")
    ix = ((points[:, 0]-bx[0])/((bx[-1]-bx[0])/nx)).astype(np.int64)
    iy = ((points[:, 1]-by[0])/((by[-1]-by[0])/ny)).astype(np.int64)
    if (ix < 0).any() or (ix >= nx).any() or (iy < 0).any() or (iy >= ny).any():
        raise ValueError("Arithmetic-floor source ownership produced an invalid cell")
    cell = ix*ny + iy
    u = (points[:, 0]-bx[ix])/np.diff(bx)[ix]
    v = (points[:, 1]-by[iy])/np.diff(by)[iy]
    outside = (u < 0) | (u > 1) | (v < 0) | (v > 1)
    if ((u < -1e-10) | (u > 1+1e-10) | (v < -1e-10) | (v > 1+1e-10)).any():
        raise ValueError("Assigned raw point exceeds source-cell roundoff tolerance")
    u, v = np.clip(u, 0, 1), np.clip(v, 0, 1)
    counts = np.bincount(cell, minlength=nx*ny).astype(np.uint64)
    sums = np.column_stack([np.bincount(cell, weights=value, minlength=nx*ny)
                            for value in (u, v, u*u, v*v, u*v)])
    return counts, sums, int(outside.sum())


def _moments(counts, sums):
    avg = np.divide(sums, counts[:, None], out=np.zeros_like(sums), where=counts[:, None] > 0)
    moments = np.column_stack((avg[:, 0], avg[:, 1], avg[:, 2]-avg[:, 0]**2,
                               avg[:, 3]-avg[:, 1]**2, avg[:, 4]-avg[:, 0]*avg[:, 1]))
    if (moments[:, 2:4] < -1e-10).any():
        raise ValueError("Negative raw variance beyond roundoff")
    moments[:, 2:4] = np.maximum(moments[:, 2:4], 0)
    return moments


def _save_structured_moments(path, hist, moments, source_id):
    _, bx, by, counts = hist
    encoded = bounded.encode_moments(moments)
    Path(path).write_bytes(_MOMENT_HEADER.pack(b"SLMOM001", 1, 5, 0, source_id,
                          *counts.shape, bx[0], bx[-1], by[0], by[-1]) + encoded.tobytes())
    return Path(path).stat().st_size


def load_structured_reference(directory):
    """Load the actual R08 coarse+moment files; mixture arrays are derived RAM."""
    parts = []
    paths = sorted(Path(directory).glob("src_*.hist"), key=lambda p: int(p.stem.split("_")[-1]))
    if not paths:
        raise FileNotFoundError(f"No native R08 source payloads: {directory}")
    for path in paths:
        sid = int(path.stem.split("_")[-1])
        hist = load_histogram(path)
        payload = path.with_name(f"src_{sid}-moments.bin").read_bytes()
        if len(payload) < _MOMENT_HEADER.size:
            raise ValueError("Truncated native R08 moment payload")
        magic, version, components, flags, source_id, nx, ny, *root = _MOMENT_HEADER.unpack_from(payload)
        expected_root = (hist[1][0], hist[1][-1], hist[2][0], hist[2][-1])
        if (magic != b"SLMOM001" or version != 1 or components != 5 or flags or source_id != sid
                or (nx, ny) != hist[3].shape or tuple(root) != expected_root
                or payload[56:64] != bytes(8) or len(payload) != 64+5*nx*ny):
            raise ValueError("Native R08 moment/histogram identities or geometry disagree")
        encoded = np.frombuffer(payload, np.uint8, offset=64).reshape(nx*ny, 5)
        parts.append(bounded.components_from_histogram_moments(hist, encoded))
    return _join_components(parts)


def _load_queries(path, metadata_file=""):
    frame = pd.read_csv(path, float_precision="round_trip")
    aliases = {"minLon": "x0", "maxLon": "x1", "minLat": "y0", "maxLat": "y1",
               "min_x": "x0", "max_x": "x1", "min_y": "y0", "max_y": "y1",
               "gt": "truth", "answer": "truth", "Ans": "truth"}
    frame = frame.rename(columns=aliases)
    if not {"x0", "x1", "y0", "y1", "truth"} <= set(frame.columns):
        frame = pd.read_csv(path, header=None, float_precision="round_trip")
        if frame.shape[1] != 5:
            raise ValueError("queries_file must contain minLon,maxLon,minLat,maxLat,truth (header optional)")
        frame.columns = ["x0", "x1", "y0", "y1", "truth"]
    if "qId" not in frame:
        frame.insert(0, "qId", np.arange(len(frame), dtype=np.int64))
    if metadata_file:
        metadata = pd.read_csv(Path(metadata_file).expanduser())
        if "qId" not in metadata or metadata.qId.duplicated().any() or len(metadata) != len(frame):
            raise ValueError("Query metadata requires one unique qId per query row")
        # Prepared authors data associates qId with the row index, not a label-derived ordering.
        if set(metadata.qId) != set(frame.qId):
            raise ValueError("Query metadata and query-file qIds differ")
        frame = frame.merge(metadata.drop(columns=[c for c in metadata if c in frame and c != "qId"]), on="qId", validate="one_to_one")
    values = frame[["x0", "x1", "y0", "y1", "truth"]].to_numpy(float)
    if (not np.isfinite(values).all() or (values[:, 0] >= values[:, 1]).any()
            or (values[:, 2] >= values[:, 3]).any() or (values[:, 4] < 0).any()
            or not np.equal(values[:, 4], np.floor(values[:, 4])).all()
            or frame.qId.duplicated().any() or not np.equal(frame.qId, np.floor(frame.qId)).all()):
        raise ValueError("Queries require unique integer qId, finite positive-area bounds and integer nonnegative truth")
    frame["qId"] = frame.qId.astype(np.int64)
    return frame


def _splits(frame, cfg, split_file):
    rows = []
    if split_file is None:
        for seed in cfg["seeds"]:
            # Exact gate_eval.split_ids algorithm, copied to avoid importing LightGBM.
            u = np.random.default_rng(seed).permutation(np.unique(frame.qId))
            roles = [("train", u[:len(u)//2]), ("calibration", u[len(u)//2:int(.7*len(u))]),
                     ("test", u[int(.7*len(u)):])]
            rows.extend(dict(seed=seed, qId=int(qid), split=role, split_kind="query_exploratory")
                        for role, ids in roles for qid in ids)
        return pd.DataFrame(rows), "query split exploratory; exact gate_eval.split_ids; no grouped holdout claim"
    outer = pd.read_csv(split_file)
    if not {"qId", "split"} <= set(outer):
        raise ValueError(f"Inherited split file must contain qId and split: {split_file}")
    if "seed" not in outer:
        if len(cfg["seeds"]) != 1:
            raise ValueError("A split file without seed supports exactly one configured seed")
        outer["seed"] = cfg["seeds"][0]
    if "split_kind" in outer:
        kinds = sorted(set(outer.split_kind.astype(str)))
        kind = cfg["split_kind"] or ("cell" if "cell" in kinds else kinds[0])
        outer = outer[outer.split_kind.astype(str) == kind].copy()
    else:
        outer["split_kind"] = "inherited"
    outer["split"] = outer["split"].replace({"cal": "calibration", "validation": "calibration"})
    outer = outer[outer.seed.isin(cfg["seeds"]) & outer.qId.isin(set(frame.qId))].copy()
    for seed in cfg["seeds"]:
        part = outer[outer.seed == seed]
        if (part.qId.duplicated().any() or set(part.qId) != set(frame.qId)
                or set(part.split) != {"train", "calibration", "test"}):
            raise ValueError(f"Inherited split seed {seed} is not a complete disjoint query partition")
        if "cell" in part and set(part.split_kind.astype(str)) == {"cell"}:
            groups = [set(part.loc[part.split == role, "cell"].astype(str)) for role in ("train", "calibration", "test")]
            if any(groups[i] & groups[j] for i in range(3) for j in range(i)):
                raise ValueError("Inherited source-cell groups overlap across outer splits")
    return outer, "inherited outer memberships; grouping claims limited to supplied split_kind and audited cells"


def _select(ids, cap, seed, salt):
    ids = sorted(map(int, ids))
    order = sorted(ids, key=lambda qid: hashlib.sha256(f"sidecar:{seed}:{salt}:{qid}".encode()).digest())
    return order if not cap else order[:cap]


def _alpha(hist):
    _, bx, by, counts = hist
    density = counts/np.outer(np.diff(bx), np.diff(by))
    median = float(np.median(density))
    mad = float(np.median(np.abs(density-median)))
    return .5/(mad/(median+1e-6)+1e-6)+1


def _overlap(boundaries, lo, hi):
    n = len(boundaries)-1
    i0 = int(np.clip(np.searchsorted(boundaries, lo, "right")-1, 0, n-1))
    i1 = int(np.clip(np.searchsorted(boundaries, hi, "left")-1, 0, n-1))
    frac = np.clip((np.minimum(hi, boundaries[i0+1:i1+2])-
                    np.maximum(lo, boundaries[i0:i1+1]))/np.diff(boundaries)[i0:i1+1], 0, 1)
    return i0, i1, frac


def histogram_count(rect, hist):
    """Canonical positive-area uniform estimate from the fixed original H."""
    _, bx, by, counts = hist
    if rect[1] <= bx[0] or rect[0] >= bx[-1] or rect[3] <= by[0] or rect[2] >= by[-1]:
        return 0.
    i0, i1, fx = _overlap(bx, rect[0], rect[1])
    j0, j1, fy = _overlap(by, rect[2], rect[3])
    return float(np.sum(np.outer(fx, fy)*counts[i0:i1+1, j0:j1+1]))


def _sa_score(rect, hist, alpha):
    _, bx, by, _ = hist
    if rect[1] <= bx[0] or rect[0] >= bx[-1] or rect[3] <= by[0] or rect[2] >= by[-1]:
        return 0.
    _, _, fx = _overlap(bx, rect[0], rect[1])
    _, _, fy = _overlap(by, rect[2], rect[3])
    ratios = np.outer(fx, fy).ravel()
    ratios = ratios[ratios > 0]
    if not len(ratios):
        return 0.
    errors = 1-betainc(alpha, alpha, .5+ratios/2)+betainc(alpha, alpha, .5-ratios/2)
    return float(np.sum(ratios*errors)/np.sum(ratios))


def _proxy(answer, reference):
    return abs(float(answer)-float(reference))/max(float(reference), 1.)


def _actual_error(answer, truth):
    a, y = np.broadcast_arrays(np.asarray(answer, float), np.asarray(truth, float))
    return np.divide(abs(a-y), y, out=np.where(a == y, 0., np.inf), where=y > 0)


def _join_components(parts):
    return bounded.Components(bounds=np.concatenate([p.bounds for p in parts]),
                              mass=np.concatenate([p.mass for p in parts]),
                              encoded=np.concatenate([p.encoded for p in parts]))


def _mass(payload):
    return int(bounded.decode(payload).total_mass)


def _save_payload(path, payload):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_bytes(payload)
    if Path(path).read_bytes() != payload:
        raise ValueError("Sidecar persistence round trip failed")


def _build_tree(leaves, tree, folder, cfg, probes, mode, audit, seed):
    """Every internal node consumes only its two saved bounded children."""
    counter = 0
    def merge(left, right):
        nonlocal counter
        tick = time.perf_counter()
        left_bytes, right_bytes = left.read_bytes(), right.read_bytes()
        expected = _mass(left_bytes)+_mass(right_bytes)
        payload = bounded.merge_bytes(left_bytes, right_bytes, cfg["budget_bytes"],
                                      training_rectangles=probes, mode=mode,
                                      max_training_queries=len(probes))
        path = folder / f"node_{counter:04d}.bin"
        counter += 1
        _save_payload(path, payload)
        observed = _mass(payload)
        if observed != expected or len(payload) > cfg["budget_bytes"]:
            raise ValueError("Internal sidecar mass or byte budget violation")
        audit.append(dict(seed=seed, mode=mode, tree=tree, node=path.stem, node_kind="internal",
                          left=str(left), right=str(right), payload_path=str(path),
                          payload_bytes=len(payload), budget_bytes=cfg["budget_bytes"],
                          total_auxiliary_budget_bytes=cfg.get("total_auxiliary_budget_bytes", cfg["budget_bytes"]),
                          global_training_geometry_bytes=cfg.get("global_training_geometry_bytes", 0),
                          components=len(bounded.decode(payload).components.mass), mass=observed,
                          input_mass=expected, mass_matches=True, raw_rows_read=0,
                          raw_bytes_read=0, merge_seconds=time.perf_counter()-tick,
                          sha256=_sha(path)))
        return path
    paths = list(leaves)
    if tree == "left":
        result = paths[0]
        for path in paths[1:]:
            result = merge(result, path)
    elif tree == "right":
        result = paths[-1]
        for path in reversed(paths[:-1]):
            result = merge(path, result)
    else:
        while len(paths) > 1:
            paths = [merge(paths[i], paths[i+1]) if i+1 < len(paths) else paths[i]
                     for i in range(0, len(paths), 2)]
        result = paths[0]
    root = folder / "root.bin"
    shutil.copyfile(result, root)
    return root


def _cutoff(score, errors, epsilon, delta):
    order = np.argsort(score, kind="stable")
    s, bad = np.asarray(score)[order], np.asarray(errors)[order] > epsilon
    violation = np.cumsum(bad)/np.arange(1, len(s)+1)
    ok = np.flatnonzero((violation <= delta) & np.r_[s[1:] != s[:-1], True])
    return -np.inf if not len(ok) else float(s[ok[-1]])


def _auc(labels, scores):
    labels = np.asarray(labels, bool)
    n1, n0 = int(labels.sum()), int((~labels).sum())
    if not n1 or not n0:
        return np.nan
    return float((rankdata(scores)[labels].sum()-n1*(n1+1)/2)/(n1*n0))


def _quality(frame, seed, epsilon, delta, min_exact):
    metrics, fixed, cutoffs = [], [], []
    for method, part in frame.groupby("method", sort=False):
        cal, test = (part[part.split == role] for role in ("calibration", "test"))
        cutoff = _cutoff(cal.score.to_numpy(), cal.actual_error.to_numpy(), epsilon, delta)
        cutoffs.append(dict(seed=seed, method=method, epsilon=epsilon, delta=delta,
                            calibration_queries=len(cal), cutoff=cutoff,
                            calibration_accepted=int((cal.score <= cutoff).sum())))
        for cohort, selected in (("all_queries", test), ("truth_ge_min_exact", test[test.truth >= min_exact]),
                                 ("truth_lt_min_exact", test[test.truth < min_exact]),
                                 ("truth_zero", test[test.truth == 0])):
            if not len(selected):
                continue
            err, score = selected.actual_error.to_numpy(), selected.score.to_numpy()
            accept = score <= cutoff
            finite = np.isfinite(err)
            mae = np.nan if method == "original_SA" else float(np.mean(abs(score-err)))
            metrics.append(dict(seed=seed, method=method, cohort=cohort, n_queries=len(selected),
                                auc_bad=_auc(err > epsilon, score), mae=mae, mae_pp=100*mae,
                                nonfinite_actual_errors=int((~finite).sum()),
                                coverage=float(accept.mean()), accepted_queries=int(accept.sum()),
                                violation=float((err[accept] > epsilon).mean()) if accept.any() else np.nan,
                                cutoff=cutoff, epsilon=epsilon, delta=delta))
            sorted_scores = np.sort(score)
            for fraction in (.1, .25, .5, 1.):
                k = max(1, int(np.ceil(fraction*len(selected))))
                threshold = sorted_scores[k-1]
                accepted = score <= threshold
                fixed.append(dict(seed=seed, method=method, cohort=cohort, nominal_coverage=fraction,
                                  actual_coverage=float(accepted.mean()), selected_queries=int(accepted.sum()),
                                  extra_tied_queries=int(accepted.sum())-k,
                                  violation=float((err[accepted] > epsilon).mean()),
                                  mean_actual_error=float(err[accepted].mean()), score_cutoff=threshold))
    return pd.DataFrame(metrics), pd.DataFrame(fixed), pd.DataFrame(cutoffs)


def _make_arms(base, structured_dir, alpha_path, fine_path, roots, virtual_union_path=None):
    def ref_score(rect, answer, artifact):
        reference = bounded.reference_count(artifact, rect)
        return ScoreResult(_proxy(answer, reference), reference)
    def fine_score(rect, answer, hist):
        reference = histogram_count(rect, hist)
        return ScoreResult(_proxy(answer, reference), reference)
    arms = [Arm("uncompressed_R08_reference", ref_score,
                artifact_loader=lambda: load_structured_reference(structured_dir)),
            Arm("original_SA", lambda rect, answer, alpha, hist: _sa_score(rect, hist, alpha),
                artifact_loader=lambda: struct.unpack("<d", alpha_path.read_bytes())[0],
                score_uses_histogram=True),
            Arm("fine_matched", fine_score, artifact_loader=lambda: _load_fine(fine_path))]
    arms.extend(Arm(name, ref_score, artifact_loader=lambda p=path: bounded.decode(p.read_bytes()))
                for name, path in roots.items())
    if virtual_union_path is not None:
        arms.append(Arm("uncompressed_virtual_partition_reference", ref_score,
                        artifact_loader=lambda: bounded.decode(virtual_union_path.read_bytes())))
    return arms


def _score_geometry(part, arms, base, seed, role):
    """Intentionally accepts geometry only, never an offline truth column."""
    if set(part.columns) != {"qId", "x0", "x1", "y0", "y1"}:
        raise ValueError("Geometry scorer accepts qId and four bounds only")
    rows = []
    for arm in arms:
        artifact = arm.artifact_loader() if arm.artifact_loader else None
        for q in part.itertuples(index=False):
            rect = np.array((q.x0, q.x1, q.y0, q.y1), float)
            answer = histogram_count(rect, base)
            result = (arm.score_fn(rect, answer, artifact, base) if arm.score_uses_histogram
                      else arm.score_fn(rect, answer, artifact))
            score = result.score if isinstance(result, ScoreResult) else float(result)
            reference = result.reference if isinstance(result, ScoreResult) else None
            if not np.isfinite(score) or score < 0:
                raise ValueError("Assessor produced a nonfinite or negative score")
            rows.append(dict(seed=seed, method=arm.name, split=role, qId=int(q.qId),
                             x0=q.x0, x1=q.x1, y0=q.y0, y1=q.y1,
                             H=answer, score=float(score), reference_count=reference))
    return pd.DataFrame(rows)


def _attach_truth(scores, labels):
    result = scores.merge(labels[["qId", "truth"]], on="qId", validate="many_to_one")
    if len(result) != len(scores):
        raise ValueError("Offline truth join lost query scores")
    result["actual_error"] = _actual_error(result.H.to_numpy(), result.truth.to_numpy())
    return result


def verify_protocol_lock(run_dir):
    """Fail if frozen summaries, training geometry, gates, or protocol changed."""
    root = Path(run_dir)
    state = json.loads((root / "run.json").read_text())
    if _sha(root / "protocol.json") != state["protocol_sha256"]:
        raise ValueError("Frozen protocol hash changed")
    protocol = json.loads((root / "protocol.json").read_text())
    for relative, digest in protocol["artifact_sha256"].items():
        path = root / relative
        if not path.is_file() or _sha(path) != digest:
            raise ValueError(f"Frozen artifact changed: {relative}")
    for name, key in (("input_manifest.json", "input_manifest_sha256"),
                      ("selected_query_ids.csv", "selected_query_ids_sha256"),
                      ("frozen_gate_cutoffs.csv", "frozen_gate_cutoffs_sha256"),
                      ("outer_splits.csv", "outer_splits_sha256")):
        if _sha(root / name) != protocol[key]:
            raise ValueError(f"Frozen protocol input changed: {name}")
    for name, digest in protocol["implementation_sha256"].items():
        path = Path(__file__).with_name(name)
        if not path.is_file() or _sha(path) != digest:
            raise ValueError(f"Frozen implementation changed: {name}")
    fresh_lock = root / "fresh_score_lock.json"
    if fresh_lock.is_file():
        lock = json.loads(fresh_lock.read_text())
        if lock["protocol_sha256"] != state["protocol_sha256"]:
            raise ValueError("Fresh scores belong to a different frozen protocol")
        for name, key in (("fresh_queries_unlabeled.csv", "geometry_sha256"),
                          ("fresh_scores_before_labels.csv", "scores_sha256"),
                          ("fresh_query_donors.csv", "donors_sha256")):
            if _sha(root / name) != lock[key]:
                raise ValueError(f"Frozen fresh query input changed: {name}")
    return protocol


def _fresh_rectangles(count, root, seed, donor_rectangles):
    rng = np.random.default_rng(seed)
    x0, x1, y0, y1 = root
    donors = np.asarray(donor_rectangles, float)
    indices = rng.integers(0, len(donors), count)
    width = donors[indices, 1]-donors[indices, 0]
    height = donors[indices, 3]-donors[indices, 2]
    if (width > x1-x0).any() or (height > y1-y0).any():
        raise ValueError("Training donor query does not fit the original H root")
    lo_x = rng.uniform(x0, x1-width)
    lo_y = rng.uniform(y0, y1-height)
    return np.column_stack((lo_x, lo_x+width, lo_y, lo_y+height))


def _indexed_labels(raw_sources, rectangles):
    """Offline labels only; the timed fallback always scans original arrays."""
    from scipy.spatial import cKDTree
    total = np.zeros(len(rectangles), np.int64)
    started = time.perf_counter()
    for points in raw_sources:
        tree = cKDTree(points, leafsize=128, copy_data=False)
        for i, (x0, x1, y0, y1) in enumerate(rectangles):
            pending = [(tree.tree, *tree.mins, *tree.maxes)]
            answer = 0
            while pending:
                node, lx, ly, hx, hy = pending.pop()
                if hx < x0 or lx > x1 or hy < y0 or ly > y1:
                    continue
                if x0 <= lx and hx <= x1 and y0 <= ly and hy <= y1:
                    answer += int(node.children)
                elif node.split_dim < 0:
                    p = tree.data[tree.indices[node.start_idx:node.end_idx]]
                    answer += int(np.count_nonzero((p[:, 0] >= x0) & (p[:, 0] <= x1) &
                                                  (p[:, 1] >= y0) & (p[:, 1] <= y1)))
                elif node.split_dim == 0:
                    pending.extend(((node.lesser, lx, ly, float(node.split), hy),
                                    (node.greater, float(node.split), ly, hx, hy)))
                else:
                    pending.extend(((node.lesser, lx, ly, hx, float(node.split)),
                                    (node.greater, lx, float(node.split), hx, hy)))
            total[i] += answer
    return total, time.perf_counter()-started


def _figures(destination, metrics, timing):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    selected = metrics[metrics.cohort == "truth_ge_min_exact"]
    summary = selected.groupby(["family", "method"], sort=False).agg(auc=("auc_bad", "mean"), mae=("mae_pp", "mean"))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    summary.auc.plot.barh(ax=axes[0], color="#2c7fb8")
    summary.mae.plot.barh(ax=axes[1], color="#f28e2b")
    axes[0].set_xlabel("Bad-answer AUC (higher is better)")
    axes[1].set_xlabel("Error-proxy MAE, percentage points (SA undefined)")
    fig.savefig(destination / "quality.png", dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    if len(timing):
        group = timing.groupby(["family", "method", "cache_mode"], sort=False).total_mean_ms.mean()
        group.plot.barh(ax=ax, color="#59a14f")
    ax.set_xlabel("Mean E2E milliseconds, including genuine scan fallback")
    fig.savefig(destination / "e2e_cost.png", dpi=150)
    plt.close(fig)


def package_results(run_dir):
    """Archive reports and bounded artifacts; never include raw data or models."""
    root = Path(run_dir).resolve()
    archive = root / "results.zip"
    with ZipFile(archive, "w", ZIP_DEFLATED) as z:
        for path in sorted(root.rglob("*")):
            if (path.is_file() and path != archive and "models" not in path.relative_to(root).parts
                    and "raw" not in path.relative_to(root).parts):
                z.write(path, path.relative_to(root))
    return archive


def run_experiment(config) -> Path:
    """Construct summaries, freeze the protocol, and evaluate the real fallback."""
    cfg = _config(config)
    destination = Path(cfg["output_root"]) / ("bounded-sidecar-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid4().hex[:8])
    destination.mkdir(parents=True, exist_ok=False)
    state = dict(status="running", study="bounded-sidecar-e2e", config=cfg,
                 started_utc=datetime.now(timezone.utc).isoformat(), training=False,
                 learned_weights_bytes=0, timings={}, limitations=[
                     "Existing inspected query cohorts are exploratory; repeated seeds are not new datasets.",
                     "Quantized moment references and empirical calibration are not confidence guarantees.",
                     "Virtual source partitions measure merge behavior on the same real dataset.",
                     "Serialized artifact bytes exclude Python objects and resident numerical temporaries."])
    _json(destination / "run.json", state)
    started = time.perf_counter()
    try:
        hist_dir, source_paths, base_path, raw_paths, query_path, split_path = _resolve_inputs(cfg)
        base = _validate_hist(load_hist(base_path))
        source_hists = [_validate_hist(load_hist(p), exact=True) for p in source_paths]
        expected_mass = sum(int(h[3].sum()) for h in source_hists)
        if not np.isclose(base[3].sum(), expected_mass, rtol=0, atol=1e-6):
            raise ValueError("Original H and source histograms have incompatible total mass")
        frame = _load_queries(query_path, cfg["query_metadata_file"])
        all_query_count = len(frame)
        if split_path is not None:
            # R02/R03 lineage stores only the eligible truth>=10 cohort. Keep original row IDs.
            split_ids = set(pd.read_csv(split_path, usecols=["qId"]).qId)
            missing = frame[~frame.qId.isin(split_ids)]
            if len(missing) and (missing.truth >= cfg["min_exact"]).any():
                raise ValueError("Inherited split file omits queries meeting min_exact")
            if len(missing):
                missing.to_csv(destination / "queries_excluded_from_inherited_cohort.csv", index=False)
                frame = frame[frame.qId.isin(split_ids)].copy()
        state.update(all_query_count=all_query_count, inherited_eligible_query_count=len(frame),
                     excluded_query_count=all_query_count-len(frame))
        outer, split_note = _splits(frame, cfg, split_path)
        outer.to_csv(destination / "outer_splits.csv", index=False)
        by_id = frame.set_index("qId")
        bounds_columns = ["x0", "x1", "y0", "y1"]
        payload_root = destination / "payloads"
        payload_root.mkdir()
        structured_dir = payload_root / "structured_reference"
        structured_dir.mkdir()
        h_path = payload_root / "original_H.hist"
        h_bytes = save_histogram(h_path, base)
        alpha_path = payload_root / "original_SA.alpha"
        alpha_path.write_bytes(struct.pack("<d", _alpha(base)))
        # R08 compact fine codec charges a 64-byte root/dimensions header and uint32 counts.
        n = math.isqrt((cfg["budget_bytes"]-64)//4)
        if n < 1:
            raise ValueError("Sidecar budget cannot hold byte-matched compact finer histogram")
        root = (float(base[1][0]), float(base[1][-1]), float(base[2][0]), float(base[2][-1]))
        fine_bx, fine_by = np.linspace(root[0], root[1], n+1), np.linspace(root[2], root[3], n+1)
        fine_counts = np.zeros((n, n), np.uint64)
        raw_sources, raw_audit, leaf_components, original_components = [], [], [], []
        structured_bytes = 0
        input_manifest = []
        for path in [base_path, *source_paths, query_path] + ([split_path] if split_path else []):
            input_manifest.append(dict(path=str(path), bytes=path.stat().st_size, sha256=_sha(path)))
        construction_started = time.perf_counter()
        for source_path, raw_path, hist in zip(source_paths, raw_paths, source_hists):
            tick = time.perf_counter()
            sid = int(source_path.stem.split("_")[-1])
            rows = int(hist[3].sum())
            before = (raw_path.stat().st_size, raw_path.stat().st_mtime_ns)
            hash_tick = time.perf_counter()
            digest = _sha(raw_path)
            hash_seconds = time.perf_counter()-hash_tick
            points = np.empty((rows, 2), np.float64)
            counts = np.zeros(hist[3].size, np.uint64)
            sums = np.zeros((hist[3].size, 5), np.float64)
            clips, position = 0, 0
            for chunk in iter_raw(raw_path, cfg["chunk_size"]):
                if position+len(chunk) > rows:
                    raise ValueError(f"Raw source has more rows than source histogram: {raw_path}")
                points[position:position+len(chunk)] = chunk
                position += len(chunk)
                c, s, clipped = _sufficient(chunk, hist)
                counts += c
                sums += s
                clips += clipped
                if ((chunk[:, 0] < root[0]) | (chunk[:, 0] > root[1]) |
                        (chunk[:, 1] < root[2]) | (chunk[:, 1] > root[3])).any():
                    raise ValueError("Raw point outside original H root")
                ix = np.minimum(((chunk[:, 0]-root[0])/(root[1]-root[0])*n).astype(int), n-1)
                iy = np.minimum(((chunk[:, 1]-root[2])/(root[3]-root[2])*n).astype(int), n-1)
                fine_counts += np.bincount(ix*n+iy, minlength=n*n).reshape(n, n).astype(np.uint64)
            if position != rows or not np.array_equal(counts.reshape(hist[3].shape), hist[3]):
                raise ValueError(f"Raw arithmetic-floor counts do not exactly reproduce {source_path}")
            if before != (raw_path.stat().st_size, raw_path.stat().st_mtime_ns):
                raise ValueError(f"Raw source changed during construction: {raw_path}")
            moments = _moments(counts, sums)
            original_components.append(bounded.components_from_histogram_moments(hist, moments))
            structured_bytes += save_histogram(structured_dir / f"src_{sid}.hist", hist)
            structured_bytes += _save_structured_moments(structured_dir / f"src_{sid}-moments.bin", hist, moments, sid)
            factor = cfg["source_partition_factor"]
            partition_mass = np.zeros_like(counts)
            for partition, part in enumerate(np.array_split(points, factor)):
                if factor == 1:
                    c, s = counts, sums
                else:
                    c, s, _ = _sufficient(part, hist)
                partition_mass += c
                partition_hist = (hist[0], hist[1], hist[2], c.reshape(hist[3].shape).astype(float))
                leaf_components.append((f"src_{sid}_partition_{partition}",
                                        bounded.components_from_histogram_moments(partition_hist, _moments(c, s))))
            if not np.array_equal(partition_mass, counts):
                raise ValueError("Virtual partition masses do not sum to original source cells")
            raw_sources.append(points)
            input_manifest.append(dict(path=str(raw_path), bytes=raw_path.stat().st_size, sha256=digest))
            raw_audit.append(dict(source_id=sid, raw_path=str(raw_path), raw_rows=rows,
                                  raw_coordinate_bytes=points.nbytes, original_cells=hist[3].size,
                                  nonempty_cells=int(np.count_nonzero(counts)), mass_matches=True,
                                  normalized_coordinate_clips=clips, source_partition_factor=factor,
                                  construction_seconds=time.perf_counter()-tick,
                                  raw_hash_seconds=hash_seconds, raw_sha256=digest))
            print(f"Source {sid}: {rows:,} real rows; exact cell masses verified", flush=True)
        state["timings"]["shared_raw_construction_seconds"] = time.perf_counter()-construction_started
        if int(fine_counts.sum()) != expected_mass:
            raise ValueError("Byte-matched finer raw-count histogram lost mass")
        fine = (0., fine_bx, fine_by, fine_counts.astype(float))
        fine_path = payload_root / "fine_matched.counts"
        fine_bytes = _save_fine(fine_path, fine_counts, root)
        if fine_bytes > cfg["budget_bytes"]:
            raise ValueError("Finer histogram exceeded byte-matched auxiliary budget")
        union = _join_components(original_components)
        union_budget = bounded.HEADER_BYTES + bounded.COMPONENT_BYTES*len(union.mass)
        union_path = payload_root / "reference_union_common_codec.bin"
        _save_payload(union_path, bounded.encode(union, union_budget))
        del union, original_components
        virtual_union_path = None
        if cfg["source_partition_factor"] > 1:
            virtual_union = _join_components([component for _, component in leaf_components])
            virtual_union_path = payload_root / "reference_virtual_partition_union.bin"
            _save_payload(virtual_union_path, bounded.encode(virtual_union,
                          bounded.HEADER_BYTES+bounded.COMPONENT_BYTES*len(virtual_union.mass)))
            del virtual_union
        pd.DataFrame(raw_audit).to_csv(destination / "source_construction.csv", index=False)
        _json(destination / "input_manifest.json", input_manifest)
        audit, costs, all_metrics, all_fixed, all_cutoffs, quality_frames, timing_frames = [], [], [], [], [], [], []
        seed_data, selections = {}, []
        for seed in cfg["seeds"]:
            membership = outer[outer.seed == seed]
            roles = {role: _select(membership.loc[membership.split == role, "qId"],
                                  cfg[{"train": "workload_queries", "calibration": "calibration_queries", "test": "test_queries"}[role]],
                                  seed, role) for role in ("train", "calibration", "test")}
            if not all(roles.values()):
                raise ValueError(f"Seed {seed} requires nonempty training/calibration/test sets")
            timing_ids = _select(roles["test"], cfg["timing_queries"], seed, "timing")
            seed_dir = payload_root / f"seed_{seed}"
            seed_dir.mkdir()
            probes = by_id.loc[roles["train"], bounds_columns].to_numpy(float)
            if len(probes) > 4096:
                raise ValueError("Set workload_queries to at most4096 explicit training probes; 32 or64 is recommended")
            probe_path = seed_dir / "training_geometry.npy"
            np.save(probe_path, probes, allow_pickle=False)
            ids_path = seed_dir / "training_ids.json"
            _json(ids_path, roles["train"])
            global_bytes = probe_path.stat().st_size+ids_path.stat().st_size
            roots = {}
            for mode in cfg["modes"]:
                mode_cfg = dict(cfg, budget_bytes=cfg["budget_bytes"]-(global_bytes if mode == "workload" else 0),
                                total_auxiliary_budget_bytes=cfg["budget_bytes"],
                                global_training_geometry_bytes=global_bytes if mode == "workload" else 0)
                if mode_cfg["budget_bytes"] < bounded.HEADER_BYTES+bounded.COMPONENT_BYTES:
                    raise ValueError("Total auxiliary budget cannot hold saved workload geometry/IDs plus one sidecar component")
                mode_dir = seed_dir / mode
                mode_dir.mkdir()
                leaves = []
                for name, component in leaf_components:
                    tick = time.perf_counter()
                    payload = bounded.build_leaf(component, mode_cfg["budget_bytes"],
                                                 training_rectangles=probes, mode=mode,
                                                 max_training_queries=len(probes))
                    path = mode_dir / "leaves" / f"{name}.bin"
                    _save_payload(path, payload)
                    if _mass(payload) != int(component.mass.sum()) or len(payload) > mode_cfg["budget_bytes"]:
                        raise ValueError("Leaf sidecar mass or byte budget violation")
                    audit.append(dict(seed=seed, mode=mode, tree="shared", node=name, node_kind="leaf",
                                      payload_path=str(path), payload_bytes=len(payload), budget_bytes=mode_cfg["budget_bytes"],
                                      total_auxiliary_budget_bytes=cfg["budget_bytes"],
                                      global_training_geometry_bytes=global_bytes if mode == "workload" else 0,
                                      components=len(bounded.decode(payload).components.mass), mass=_mass(payload),
                                      input_mass=int(component.mass.sum()), mass_matches=True,
                                      raw_rows_read=0, raw_bytes_read=0,
                                      merge_seconds=time.perf_counter()-tick, sha256=_sha(path)))
                    leaves.append(path)
                for tree in cfg["merge_trees"]:
                    folder = mode_dir / tree
                    folder.mkdir()
                    root_path = _build_tree(leaves, tree, folder, mode_cfg, probes, mode, audit, seed)
                    if _mass(root_path.read_bytes()) != expected_mass:
                        raise ValueError("Root sidecar total mass differs from original data")
                    roots[f"bounded_{mode}_{tree}"] = root_path
            seed_data[seed] = dict(roles=roles, timing_ids=timing_ids, roots=roots,
                                   global_bytes=global_bytes, probes=probe_path)
            selections.extend(dict(seed=seed, qId=qid, role=role, timed=qid in timing_ids if role == "test" else False)
                              for role, ids in roles.items() for qid in ids)
        del leaf_components
        pd.DataFrame(audit).to_csv(destination / "merge_audit.csv", index=False)
        pd.DataFrame(selections).to_csv(destination / "selected_query_ids.csv", index=False)
        # Gates are frozen on old calibration truth before fresh geometry or labels exist.
        frozen_rows = []
        geometry_columns = ["qId", *bounds_columns]
        for seed, data in seed_data.items():
            data["arms"] = _make_arms(base, structured_dir, alpha_path, fine_path, data["roots"], virtual_union_path)
            cal_frame = by_id.loc[data["roles"]["calibration"]].reset_index()
            cal_scores = _score_geometry(cal_frame[geometry_columns], data["arms"], base, seed, "calibration")
            data["calibration_frame"] = cal_frame
            data["calibration_scores"] = _attach_truth(cal_scores, cal_frame)
            seed_gates = []
            for method, cal in data["calibration_scores"].groupby("method", sort=False):
                cutoff = _cutoff(cal.score.to_numpy(), cal.actual_error.to_numpy(), cfg["epsilon"], cfg["delta"])
                gate = dict(seed=seed, method=method, cutoff=cutoff, epsilon=cfg["epsilon"], delta=cfg["delta"],
                            calibration_queries=len(cal), calibration_accepted=int((cal.score <= cutoff).sum()))
                seed_gates.append(gate)
                frozen_rows.append(gate)
            data["gates"] = pd.DataFrame(seed_gates)
        pd.DataFrame(frozen_rows).to_csv(destination / "frozen_gate_cutoffs.csv", index=False)
        protocol = dict(status="frozen_before_test_evaluation", config=cfg,
                        input_manifest_sha256=_sha(destination / "input_manifest.json"),
                        outer_splits_sha256=_sha(destination / "outer_splits.csv"),
                        split_note=split_note, selected_query_ids_sha256=_sha(destination / "selected_query_ids.csv"),
                        frozen_gate_cutoffs_sha256=_sha(destination / "frozen_gate_cutoffs.csv"),
                        original_H_sha256=_sha(h_path), original_H_bytes=h_bytes,
                        compression_labels_read=False, training_geometry_only=True,
                        actual_error="abs(H-truth)/truth; truth=0 gives 0 iff H=0 else infinity",
                        score="abs(H-reference)/max(reference,1); SA canonical heuristic",
                        calibration="largest whole-score-tie prefix with empirical violation <= delta",
                        fixed_coverage="10/25/50/100 percent; whole-score ties included",
                        fresh_queries="geometry and scores persisted after this freeze and before raw labels" if cfg["fresh_query_count"] else "disabled",
                        fresh_geometry_policy="sample width/height pairs from the first configured seed's selected TRAIN probes, uniformly shift within original H root",
                        fresh_donor_seed=cfg["seeds"][0],
                        virtual_partition_scope="contiguous equal-row partitions of each original source; same real dataset",
                        auxiliary_budget="workload payload + once-per-seed saved geometry and IDs <= B; geometry payload <= B; fine compact payload <= B",
                        artifact_sha256={str(p.relative_to(destination)): _sha(p) for p in payload_root.rglob("*") if p.is_file()},
                        implementation_sha256={p.name: _sha(p) for p in (Path(__file__), Path(bounded.__file__),
                            Path(__file__).with_name("sidecar_e2e.py"), Path(__file__).with_name("_distribution_math.py"),
                            Path(__file__).with_name("features.py"))})
        _json(destination / "protocol.json", protocol)
        state.update(status="frozen_before_evaluation", protocol_sha256=_sha(destination / "protocol.json"),
                     split_note=split_note, original_H_bytes=h_bytes,
                     structured_reference_auxiliary_bytes=structured_bytes,
                     union_common_codec_bytes=union_path.stat().st_size,
                     fine_matched_auxiliary_bytes=fine_bytes, fine_matched_resolution=n,
                     raw_rows=expected_mass, virtual_leaves=len(source_paths)*cfg["source_partition_factor"])
        _json(destination / "run.json", state)
        verify_protocol_lock(destination)
        # Fallback never receives stored truth; parity labels are offline diagnostics only.
        raw_counter = ScanCounter(raw_sources)
        parity_ids = _select(frame.qId, cfg["parity_queries"], 0, "raw-parity")
        parity = []
        parity_started = time.perf_counter()
        for qid in parity_ids:
            row = by_id.loc[qid]
            counted = int(raw_counter.count(row[bounds_columns].to_numpy(float)))
            parity.append(dict(qId=qid, stored_truth=int(row.truth), raw_count=counted,
                               matches=counted == int(row.truth)))
        state["timings"]["stored_truth_parity_scan_seconds"] = time.perf_counter()-parity_started
        pd.DataFrame(parity).to_csv(destination / "raw_truth_parity.csv", index=False)
        if not all(item["matches"] for item in parity):
            raise ValueError("Stored query truth disagrees with genuine closed-bound raw scans; offline truth reuse refused")
        fresh_frame = None
        if cfg["fresh_query_count"]:
            donor_seed = cfg["seeds"][0]
            donor_ids = seed_data[donor_seed]["roles"]["train"]
            donor_rectangles = by_id.loc[donor_ids, bounds_columns].to_numpy(float)
            fresh_rects = _fresh_rectangles(cfg["fresh_query_count"], root, cfg["fresh_query_seed"], donor_rectangles)
            fresh_geometry = pd.DataFrame(fresh_rects, columns=bounds_columns)
            fresh_geometry.insert(0, "qId", np.arange(len(fresh_geometry), dtype=np.int64)+int(frame.qId.max())+1)
            fresh_geometry.to_csv(destination / "fresh_queries_unlabeled.csv", index=False)
            donor_indices = np.random.default_rng(cfg["fresh_query_seed"]).integers(0, len(donor_ids), len(fresh_geometry))
            pd.DataFrame(dict(qId=fresh_geometry.qId, donor_training_qId=np.asarray(donor_ids)[donor_indices],
                              donor_split_seed=donor_seed)).to_csv(destination / "fresh_query_donors.csv", index=False)
            fresh_scores = []
            for seed, data in seed_data.items():
                scores = _score_geometry(fresh_geometry, data["arms"], base, seed, "test")
                data["fresh_scores"] = scores
                fresh_scores.append(scores)
            pd.concat(fresh_scores, ignore_index=True).to_csv(destination / "fresh_scores_before_labels.csv", index=False)
            _json(destination / "fresh_score_lock.json", dict(status="scored_before_labels",
                  protocol_sha256=state["protocol_sha256"],
                  geometry_sha256=_sha(destination / "fresh_queries_unlabeled.csv"),
                  donors_sha256=_sha(destination / "fresh_query_donors.csv"),
                  scores_sha256=_sha(destination / "fresh_scores_before_labels.csv")))
            verify_protocol_lock(destination)
            labels, seconds = _indexed_labels(raw_sources, fresh_rects)
            fresh_frame = fresh_geometry.copy()
            fresh_frame["truth"] = labels
            fresh_frame.to_csv(destination / "fresh_queries_labeled_offline.csv", index=False)
            state["timings"]["fresh_offline_indexed_label_seconds"] = seconds
            for item in fresh_frame.head(min(3, len(fresh_frame))).itertuples(index=False):
                if int(raw_counter.count(np.array((item.x0, item.x1, item.y0, item.y1)))) != int(item.truth):
                    raise ValueError("Fresh indexed labels disagree with independent raw scans")
        for seed, data in seed_data.items():
            arms = data["arms"]
            for arm in arms:
                payload_bytes = (structured_bytes if arm.name == "uncompressed_R08_reference"
                                 else virtual_union_path.stat().st_size if arm.name == "uncompressed_virtual_partition_reference"
                                 else 8 if arm.name == "original_SA"
                                 else fine_bytes if arm.name == "fine_matched" else data["roots"][arm.name].stat().st_size)
                global_bytes = data["global_bytes"] if arm.name.startswith("bounded_workload_") else 0
                aux = payload_bytes+global_bytes
                runtime_bytes = payload_bytes
                costs.append(dict(seed=seed, method=arm.name, original_H_bytes=h_bytes, auxiliary_bytes=aux,
                                  payload_bytes=payload_bytes, total_serialized_bytes=h_bytes+aux, model_weight_bytes=0,
                                  global_training_geometry_bytes=global_bytes, total_auxiliary_budget_bytes=cfg["budget_bytes"],
                                  runtime_representation_bytes=runtime_bytes,
                                  total_runtime_serialized_bytes=h_bytes+runtime_bytes+global_bytes,
                                  offline_union_verification_bytes=union_path.stat().st_size if arm.name == "uncompressed_R08_reference" else 0,
                                  persisted_control_bytes=h_bytes+aux+(union_path.stat().st_size if arm.name == "uncompressed_R08_reference" else 0),
                                  reference_storage_codec="R08 structured coarse + five-byte moments" if arm.name == "uncompressed_R08_reference" else "actual persisted codec",
                                  fine_unused_budget_bytes=cfg["budget_bytes"]-fine_bytes if arm.name == "fine_matched" else 0))
                if arm.name.startswith("bounded_") and aux > cfg["budget_bytes"]:
                    raise ValueError("Payload plus shared workload input exceeds total auxiliary budget")
            cal_frame = data["calibration_frame"]
            test_frame = by_id.loc[data["roles"]["test"]].reset_index()
            old_scores = _score_geometry(test_frame[geometry_columns], arms, base, seed, "test")
            evaluations = [("existing_exploratory", test_frame, old_scores, data["timing_ids"])]
            if fresh_frame is not None:
                evaluations.append(("fresh_same_dataset", fresh_frame, data["fresh_scores"],
                                    _select(fresh_frame.qId, cfg["timing_queries"], seed, "timing-fresh")))
            for family, evaluation_frame, scores, timed_ids in evaluations:
                quality = pd.concat((data["calibration_scores"], _attach_truth(scores, evaluation_frame)), ignore_index=True)
                quality["family"] = family
                quality_frames.append(quality)
                metrics, fixed, cutoffs = _quality(quality, seed, cfg["epsilon"], cfg["delta"], cfg["min_exact"])
                all_metrics.append(metrics.assign(family=family))
                all_fixed.append(fixed.assign(family=family))
                all_cutoffs.append(cutoffs.assign(family=family))
                if not np.array_equal(cutoffs.set_index("method").loc[data["gates"].method, "cutoff"].to_numpy(), data["gates"].cutoff.to_numpy()):
                    raise ValueError("Evaluation gate changed after protocol freeze")
                timing_frame = evaluation_frame.set_index("qId").loc[timed_ids].reset_index()
                query_objects = lambda part: [Query(int(q.qId), (q.x0, q.x1, q.y0, q.y1), float(q.truth))
                                               for q in part.itertuples(index=False)]
                verify_protocol_lock(destination)
                result = run_benchmark(queries=query_objects(timing_frame), calibration_queries=query_objects(cal_frame),
                                       arms=arms, histogram_loader=lambda: load_histogram(h_path),
                                       estimate_fn=histogram_count, raw_counter=raw_counter,
                                       output_dir=destination / "e2e" / f"seed_{seed}" / family,
                                       epsilon=cfg["epsilon"], delta=cfg["delta"],
                                       cache_modes=("resident", "per_query"), repeats=cfg["timing_repeats"], seed=seed,
                                       frozen_cutoffs=data["gates"].set_index("method").to_dict("index"),
                                       frozen_cutoff_provenance=dict(path=str(destination / "frozen_gate_cutoffs.csv"),
                                                                     sha256=_sha(destination / "frozen_gate_cutoffs.csv")),
                                       metadata=dict(protocol_sha256=state["protocol_sha256"],
                                                     timing_selection="deterministic qId hash before truth evaluation", cohort=family))
                timing_frames.append(result.summary.assign(seed=seed, family=family))
                print(f"Seed {seed} {family}: quality {len(evaluation_frame)} queries; live E2E {len(timing_frame)} queries", flush=True)
        verify_protocol_lock(destination)
        metrics, fixed, cutoffs = (pd.concat(items, ignore_index=True) for items in (all_metrics, all_fixed, all_cutoffs))
        metrics.to_csv(destination / "metrics.csv", index=False)
        fixed.to_csv(destination / "fixed_coverage.csv", index=False)
        cutoffs.to_csv(destination / "gate_cutoffs.csv", index=False)
        pd.concat(quality_frames, ignore_index=True).to_csv(destination / "per_query_quality.csv", index=False)
        timing = pd.concat(timing_frames, ignore_index=True)
        timing.to_csv(destination / "e2e_summary.csv", index=False)
        pd.DataFrame(costs).to_csv(destination / "payload_costs.csv", index=False)
        summary = metrics.groupby(["family", "method", "cohort"], as_index=False).agg(mean_auc=("auc_bad", "mean"),
                     mean_mae_pp=("mae_pp", "mean"), mean_coverage=("coverage", "mean"),
                     mean_violation=("violation", "mean"), min_queries=("n_queries", "min"))
        summary.to_csv(destination / "summary.csv", index=False)
        _figures(destination, metrics, timing)
        scope = "fresh random rectangles on the same inspected real dataset" if fresh_frame is not None else "the previously inspected query cohort; exploratory evidence"
        report = (f"# Bounded sidecar merge and genuine fallback\n\n"
                  f"This run evaluated {scope}. It used {expected_mass:,} real points and {state['virtual_leaves']} "
                  f"virtual leaves, with a fixed {cfg['budget_bytes']:,}-byte auxiliary cap. Workload payload caps subtract saved training geometry and IDs once per seed. "
                  f"The original histogram H stayed identical across all assessors. No model was trained.\n\n"
                  f"Split protocol: {split_note}. Workload compression received only selected training rectangles. "
                  "Calibration truth selected each empirical gate cutoff. Test truth selected no budget, mode, merge tree, "
                  "query subset, or cutoff. Whole score ties are retained in the 10/25/50/100% coverage diagnostics. "
                  "The truth>=min_exact metrics are numerical-error diagnostics; all query counts remain in gate/E2E reports.\n\n"
                  f"Common original H costs {h_bytes:,} bytes using the R08 SLHIST01 binary codec. The original-cell "
                  f"R08 reference costs {structured_bytes:,} auxiliary bytes in its structured coarse+moment codecs. "
                  f"Its explicit common-component union costs {state['union_common_codec_bytes']:,} bytes as a separate "
                  "offline verification artifact. Runtime requests load the native structured files and construct mixture "
                  "arrays in RAM; they never load the larger generic union. Comparing a bounded payload with that larger generic codec "
                  "does not establish compression relative to original R08 storage. Every bounded artifact is independently "
                  "decodable without the reference union. Training geometry and IDs are separate shared compressor input "
                  "bytes in payload_costs.csv. Resident RAM includes additional arrays and Python objects.\n\n"
                  f"The finer control has {n}x{n} exact raw-count cells and costs {fine_bytes:,} auxiliary bytes, "
                  "using the R08 64-byte root/dimensions header plus uint32 counts; no unused alpha is stored. Its resolution was selected by bytes alone. "
                  "Moment construction and fine counts share one raw parsing pass per original source. Virtual partition "
                  "summaries revisit in-memory construction points; all merges read zero raw rows and zero raw bytes.\n\n"
                  "Quality covers the selected full evaluation subset; timed queries are a separate deterministic qId "
                  "subset. E2E measurements execute live H estimation, live scoring, a frozen gate, and actual closed-bound "
                  "raw scans on rejection. The all-raw baseline scans the same original coordinate arrays. Setup, first-query, "
                  "resident and per-query-loading measurements are reported separately. Raw fallback returns the scanned "
                  "answer and never looks up stored truth. Stored truth was reused only after independent parity scans.\n\n"
                  "AUC measures bad-answer ranking. Error-proxy MAE is undefined for SA because SA is a ranking heuristic. "
                  "Moment matching is approximate; no confidence guarantee or novelty claim follows. Repeated seeds and "
                  "virtual partitions are not independent datasets. Inspect metrics.csv, fixed_coverage.csv, gate_cutoffs.csv, "
                  "merge_audit.csv and the e2e directory before making quality or speed claims.\n")
        (destination / "analysis_en.md").write_text(report)
        state.update(status="complete", completed_utc=datetime.now(timezone.utc).isoformat(),
                     total_seconds=time.perf_counter()-started,
                     package_versions={name: importlib.metadata.version(name) for name in ("numpy", "pandas", "scipy", "matplotlib")})
        _json(destination / "run.json", state)
        package_results(destination)
        return destination
    except BaseException as exc:
        state.update(status="failed", failed_utc=datetime.now(timezone.utc).isoformat(),
                     error=f"{type(exc).__name__}: {exc}", total_seconds=time.perf_counter()-started)
        _json(destination / "run.json", state)
        (destination / "failure.txt").write_text(traceback.format_exc())
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="Optional JSON config; explicit CLI arguments override it")
    for key, value in DEFAULTS.items():
        if isinstance(value, list):
            parser.add_argument("--"+key.replace("_", "-"), default=None, help="comma-separated values")
        else:
            parser.add_argument("--"+key.replace("_", "-"), type=type(value), default=None)
    parser.add_argument("--repo-root", default=None)
    args = vars(parser.parse_args(argv))
    config_path = args.pop("config")
    config = json.loads(Path(config_path).read_text()) if config_path else {}
    config.update({key: value for key, value in args.items() if value is not None})
    print(run_experiment(config))


if __name__ == "__main__":
    main()
