"""Execution-only replay of frozen local-shape confirmations.

Native, packet, model, gate, H and matched fine bytes remain unchanged. Only
the derived source-axis indexes and required-feature evaluation are replaced.
This is a Python runtime comparison, with resident raw vector scans for exact
answers. No training, calibration, query lookup or distribution-model change
occurs. Existing research source files and result directories are never edited.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time
from typing import Mapping
from uuid import uuid4

import numpy as np
import pandas as pd

from . import local_shape_workflow as workflow
from . import shape_error_model as models
from . import shape_runtime
from . import sidecar_workflow as prior
from .sidecar_e2e import Arm, Query, ScanCounter, ScoreResult, run_benchmark


TOLERANCE = 1e-9
TIMING_SALT = "r10-timing-fresh_same_dataset"
METHOD_KINDS = {"R08": "native", "hybrid": "hybrid", "empirical_uniform": "uniform",
                "R08_tinyML": "ml_native", "hybrid_tinyML": "ml_hybrid"}


@dataclass
class _Artifact:
    runtime: object
    predictor: object = None
    canonical: object = None
    canonical_loaded: bool = False


def _cutoff_value(value):
    cutoff = float(value["cutoff"] if isinstance(value, Mapping) else value)
    if not (math.isfinite(cutoff) or cutoff == -math.inf):
        raise ValueError("Frozen cutoffs must be finite or negative infinity")
    return cutoff


def _backend_module(backend):
    if backend == "indexed":
        return shape_runtime
    if backend == "compiled":
        from . import shape_kernel
        return shape_kernel
    raise ValueError("backend must be explicitly indexed or compiled")


def _backend_setup(backend):
    """Prepare executable code outside artifact loaders/request timing."""
    module = _backend_module(backend)
    if backend == "indexed":
        return dict(backend=backend, compilation_seconds=0., executable_setup_seconds=0.)
    already_initialized = module._STATE is not None
    tick = time.perf_counter()
    try:
        setup = module.kernel_setup()
    except (OSError, RuntimeError) as exc:
        raise RuntimeError("Explicit compiled backend could not initialize; install a supported g++/SciPy "
                           "toolchain or explicitly select backend='indexed': "+str(exc)) from exc
    source = Path(module.__file__)
    library = Path(setup["cache_dir"])/"shape_kernel.so"
    return dict(backend=backend, process_already_initialized=already_initialized,
        compilation_seconds=0. if already_initialized else setup["compilation_seconds"],
        executable_setup_seconds=time.perf_counter()-tick,
        library_load_seconds=0. if already_initialized else setup["library_load_seconds"],
        cache_hit=already_initialized or setup["compilation_seconds"] == 0.,
        cache_key=setup["cache_key"], cache_dir=setup["cache_dir"], library_sha256=setup["library_sha256"],
        build_identity=setup["identity"],
        executable_code_bytes=dict(wrapper_source=source.stat().st_size,
                                   kernel_source=source.with_suffix(".cc").stat().st_size,
                                   compiled_library=library.stat().st_size),
        code_bytes_policy="shared nondatabaked executable code; separate from native/packet/model payload budgets")


def index_arms(old_arms, *, native_dir, packet_path=None, packet_paths=None,
               model_paths=None, method_kinds=None, cutoffs=None, backend="indexed"):
    """Wrap existing arms, retaining fine controls and all serialized costs.

    Custom method names (including OSM controls) can supply ``method_kinds``
    with values native/hybrid/uniform/ml_native/ml_hybrid. Packet paths may be
    shared or provided per method in ``packet_paths``; ML arms require their
    original schema in ``model_paths``. Unmapped arms are returned unchanged.
    For a score within 1e-9 absolute/relative of its *unchanged* cutoff, the
    original scorer is executed live to preserve exact boundary decisions.
    This fallback never consults saved scores, labels or query identifiers.
    ``backend='compiled'`` explicitly prepares the shared code before returning
    arms. Its analytic references run C++; ML required-feature extraction stays
    on the unchanged indexed Python path and original model inference.
    """
    backend_module = _backend_module(backend)
    backend_setup = _backend_setup(backend)
    old_arms = list(old_arms)
    native_dir = Path(native_dir)
    kinds = dict(METHOD_KINDS if method_kinds is None else method_kinds)
    packet_paths, model_paths = dict(packet_paths or {}), dict(model_paths or {})
    names = [arm.name for arm in old_arms]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate arm names")
    if set(kinds)-set(names):
        # Default roles are allowed to describe controls absent from a packet.
        if method_kinds is not None:
            raise ValueError("Method kinds refer to absent arms")
        kinds = {k: v for k, v in kinds.items() if k in names}
    if cutoffs is not None and set(cutoffs) != set(names):
        raise ValueError("Frozen cutoff names must match all original arms")
    numeric_cutoffs = None if cutoffs is None else {name: _cutoff_value(v) for name, v in cutoffs.items()}
    result = []
    for old in old_arms:
        if old.name not in kinds:
            result.append(old)
            continue
        kind = kinds[old.name]
        if kind not in {"native", "hybrid", "uniform", "ml_native", "ml_hybrid"}:
            raise ValueError("Unknown indexed method kind: "+str(kind))
        is_ml = kind.startswith("ml_")
        uses_packet = kind in {"hybrid", "uniform", "ml_hybrid"}
        method_packet = packet_paths.get(old.name, packet_path) if uses_packet else None
        if uses_packet and method_packet is None:
            raise ValueError("Hierarchical methods require their original packet")
        if is_ml and old.name not in model_paths:
            raise ValueError("ML methods require their original model schema")
        schema = None if not is_ml else Path(model_paths[old.name])
        stats = dict(indexed_calls=0, canonical_near_cutoff_calls=0, artifact_loads=0)
        metadata = dict(old.metadata or {})
        metadata.update(execution=backend+" frozen source/packet evaluator", execution_stats=stats,
                        execution_backend=backend, executable_setup=backend_setup,
                        feature_backend="indexed Python required features" if is_ml else backend,
                        model_inference="original frozen predictor" if is_ml else None)

        def loader(old=old, method_packet=method_packet, schema=schema, stats=stats):
            runtime = (backend_module.load_compiled(native_dir, method_packet) if backend == "compiled"
                       else backend_module.load_indexed(native_dir, method_packet))
            predictor = models.load_model(schema) if schema is not None else None
            expected = (old.metadata or {}).get("auxiliary_bytes")
            model_bytes = 0 if predictor is None else predictor.model_bytes
            if expected is not None and runtime.auxiliary_bytes+model_bytes != int(expected):
                raise ValueError("Indexed loading changed serialized auxiliary bytes: "+old.name)
            stats["artifact_loads"] += 1
            stats["runtime_ram_bytes"] = int(runtime.ram_bytes)
            stats["derived_runtime_ram_bytes"] = int(runtime.derived_ram_bytes)
            return _Artifact(runtime, predictor)

        def score(rectangle, answer, artifact, histogram, *, old=old, kind=kind, stats=stats):
            rectangle = np.asarray(rectangle, dtype=float)
            runtime = artifact.runtime
            if kind.startswith("ml_"):
                hybrid = kind == "ml_hybrid"
                described = runtime.query_features(rectangle[None, :])
                # H is exactly the estimate already charged to this request.
                described["H"] = np.asarray([answer], dtype=float)
                root = (histogram[1][0], histogram[1][-1], histogram[2][0], histogram[2][-1])
                features = workflow.model_features(rectangle[None, :], described, root, hybrid=hybrid)
                value = ScoreResult(artifact.predictor.predict_row(features.iloc[0]))
            else:
                mode = "empirical_uniform" if kind == "uniform" else None
                if backend == "compiled":
                    cutoff = None if numeric_cutoffs is None else numeric_cutoffs[old.name]
                    reference, error_score = runtime.reference_and_score(rectangle, answer, cutoff=cutoff, mode=mode)
                    value = ScoreResult(error_score, reference)
                else:
                    reference = runtime.reference_count(rectangle, mode=mode)
                    value = ScoreResult(prior._proxy(answer, reference), reference)
            stats["indexed_calls"] += 1
            cutoff = None if numeric_cutoffs is None else numeric_cutoffs[old.name]
            near = (backend != "compiled" or kind.startswith("ml_")) and cutoff is not None and math.isfinite(cutoff) and math.isclose(
                value.score, cutoff, rel_tol=TOLERANCE, abs_tol=TOLERANCE)
            if near:
                if not artifact.canonical_loaded:
                    artifact.canonical = old.artifact_loader() if old.artifact_loader else None
                    artifact.canonical_loaded = True
                canonical = (old.score_fn(rectangle, answer, artifact.canonical, histogram)
                             if old.score_uses_histogram else old.score_fn(rectangle, answer, artifact.canonical))
                value = canonical if isinstance(canonical, ScoreResult) else ScoreResult(float(canonical))
                stats["canonical_near_cutoff_calls"] += 1
            return value

        result.append(Arm(old.name, score, artifact_loader=loader,
                          metadata=metadata, score_uses_histogram=True))
    return result


def build_indexed_arms(development_dir, seed, *, cutoffs=None, backend="indexed"):
    """Equivalent frozen R10 method roster with indexed runtime loaders."""
    development = Path(development_dir)
    state = json.loads((development/"run.json").read_text())
    if state["config"]["family"] != "hierarchical":
        raise ValueError("Indexed replay currently requires hierarchical packets")
    packet = state["packets"][str(state["candidate_extra_budget"])]
    packet_path = development/"packets"/Path(packet["path"]).name
    old_arms = workflow.load_arms(development, seed)
    if cutoffs is None:
        gates = pd.read_csv(development/"frozen_gate_cutoffs.csv", float_precision="round_trip")
        selected = gates[gates.seed == seed]
        if selected.method.duplicated().any():
            raise ValueError("Duplicate frozen gate methods")
        cutoffs = selected.set_index("method").to_dict("index")
    model_paths = {arm.name: development/"models"/f"seed_{seed}"/arm.name/"model.schema.json"
                   for arm in old_arms if arm.name in {"R08_tinyML", "hybrid_tinyML"}}
    return index_arms(old_arms, native_dir=development/"native", packet_path=packet_path,
                      model_paths=model_paths, cutoffs=cutoffs, backend=backend)


def verify_live_parity(measurements, frozen_scores, frozen_gates, *, tolerance=TOLERANCE):
    """Check every measured live value and exact gate decision without labels.

    Measurements require seed/method/qId/H/score/reference/accepted. Frozen
    scores are the persisted geometry-only scores written before raw labels.
    ML's intentionally absent reference is compared as a paired missing value.
    Exact baseline returns are audited by the E2E engine's fresh raw scans.
    """
    keys = ["seed", "method", "qId"]
    required = {*keys, "H", "score", "reference_count"}
    if not required <= set(frozen_scores) or frozen_scores.duplicated(keys).any():
        raise ValueError("Frozen score keys or columns are invalid")
    if frozen_gates.duplicated(["seed", "method"]).any():
        raise ValueError("Duplicate frozen gate keys")
    live = measurements[measurements.method != "always_exact"].copy()
    expected = frozen_scores[[*keys, "H", "score", "reference_count"]].rename(
        columns={"H": "expected_H", "score": "expected_score", "reference_count": "expected_reference"})
    live = live.merge(expected, on=keys, how="left", validate="many_to_one", indicator=True)
    if (live._merge != "both").any():
        raise ValueError("Live query/method absent from frozen scores")
    gates = frozen_gates[["seed", "method", "cutoff"]].rename(columns={"cutoff": "expected_cutoff"})
    live = live.merge(gates, on=["seed", "method"], how="left", validate="many_to_one")
    if live.expected_cutoff.isna().any():
        raise ValueError("Live method absent from frozen gates")
    for label, observed, expected_name in (("H", "H", "expected_H"),
                                           ("score", "score", "expected_score"),
                                           ("reference", "reference", "expected_reference")):
        values, old = pd.to_numeric(live[observed]).to_numpy(float), live[expected_name].to_numpy(float)
        live[label+"_matches"] = np.isclose(values, old, rtol=tolerance, atol=tolerance,
                                            equal_nan=label == "reference")
        live[label+"_abs_delta"] = np.abs(values-old)
    accepted = live.accepted
    if accepted.dtype != bool:
        accepted = accepted.astype(str).str.lower().map({"true": True, "false": False})
        if accepted.isna().any():
            raise ValueError("Live acceptance must contain booleans")
    expected_accepted = np.isfinite(live.expected_score) & (live.expected_score <= live.expected_cutoff)
    live["decision_matches"] = accepted.to_numpy(bool) == expected_accepted.to_numpy(bool)
    live["cutoff_matches"] = (live.cutoff.to_numpy(float) == live.expected_cutoff.to_numpy(float)
                               if "cutoff" in live else True)
    live["all_matches"] = live[["H_matches", "score_matches", "reference_matches", "decision_matches", "cutoff_matches"]].all(axis=1)
    columns = [*keys, *[c for c in ("cache_mode", "repeat", "phase") if c in live],
               "H_matches", "score_matches", "reference_matches", "decision_matches", "cutoff_matches", "all_matches",
               "H_abs_delta", "score_abs_delta", "reference_abs_delta"]
    return live[columns]


def _geometry_labels(geometry, labels):
    columns = ["qId", *workflow.GEOMETRY]
    if (geometry.qId.duplicated().any() or labels.qId.duplicated().any()
            or set(geometry.qId) != set(labels.qId)):
        raise ValueError("Labels do not cover the locked geometry exactly")
    old = geometry[columns].sort_values("qId").reset_index(drop=True)
    new = labels[columns].sort_values("qId").reset_index(drop=True)
    if not np.array_equal(old.to_numpy(), new.to_numpy()):
        raise ValueError("Labeled coordinates differ from locked geometry")
    truth = labels.truth.to_numpy(float)
    if not np.isfinite(truth).all() or (truth < 0).any() or (truth != np.floor(truth)).any():
        raise ValueError("Raw row-count labels must be finite nonnegative integers")


def _source_hashes(backend="indexed"):
    paths = [Path(__file__), Path(shape_runtime.__file__)]
    if backend == "compiled":
        source = Path(_backend_module(backend).__file__)
        paths += [source, source.with_suffix(".cc")]
    return {p.name: prior._sha(p) for p in paths}


def run_indexed_confirmation(confirmation_dir, output_root, timing_queries=32, repeats=3, *, backend="indexed"):
    """Replay the canonical timing cohort with frozen artifacts and live scans.

    Query IDs are selected from the locked *unlabeled* geometry with the exact
    original timing salt, before loading labels. Existing protocol verifiers
    remain unchanged. Every timed and optional first-request assessor output
    must match its original score-before-labels record and exact gate decision.
    """
    if int(timing_queries) != timing_queries or timing_queries < 0 or int(repeats) != repeats or repeats < 1:
        raise ValueError("Invalid timing query limit or repeats")
    backend_module = _backend_module(backend)
    confirmation = Path(confirmation_dir).expanduser().resolve()
    original_protocol = workflow.verify_confirmation_lock(confirmation)
    old_state = json.loads((confirmation/"run.json").read_text())
    if old_state.get("status") != "complete":
        raise ValueError("Execution replay requires a completed frozen confirmation")
    development = Path(old_state["development_dir"])
    development_state = json.loads((development/"run.json").read_text())
    preparation, preparation_state = workflow._load_preparation(development_state["preparation_dir"])
    if prior._sha(preparation/"source_audit.csv") != prior._sha(development/"source_audit.csv"):
        raise ValueError("Prepared raw caches differ from frozen development")
    if prior._sha(preparation/"run.json") != json.loads((development/"protocol.json").read_text())["source_preparation_manifest_sha256"]:
        raise ValueError("Source preparation manifest changed")
    geometry = pd.read_csv(confirmation/"fresh_queries_unlabeled.csv", float_precision="round_trip")
    scores = pd.read_csv(confirmation/"fresh_scores_before_labels.csv", float_precision="round_trip")
    gates = pd.read_csv(confirmation/"frozen_gate_cutoffs.csv", float_precision="round_trip")
    cfg = original_protocol["config"]
    selected = {int(seed): prior._select(geometry.qId, int(timing_queries), int(seed), TIMING_SALT)
                for seed in cfg["seeds"]}
    if not len(geometry) or any(not ids for ids in selected.values()):
        raise ValueError("No locked test geometry available")
    # The selection above depends on geometry IDs, never scores or labels.
    labels_path = confirmation/"fresh_queries_labeled_offline.csv"
    labels = pd.read_csv(labels_path, float_precision="round_trip")
    _geometry_labels(geometry, labels)
    selected_calibration = pd.read_csv(development/"selected_query_ids.csv")
    offline_queries = pd.read_csv(preparation/"queries_offline.csv", float_precision="round_trip").set_index("qId")
    destination = Path(output_root).expanduser().resolve()/(backend+"-shape-execution-"+
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")+"-"+uuid4().hex[:8])
    destination.mkdir(parents=True, exist_ok=False)
    state = dict(status="running", study=backend+"-shape-execution", backend=backend, request_engine="Python",
        source_confirmation=str(confirmation), source_confirmation_manifest_sha256=prior._sha(confirmation/"run.json"),
        source_confirmation_protocol_sha256=prior._sha(confirmation/"protocol.json"),
        source_development=str(development), source_development_protocol_sha256=prior._sha(development/"protocol.json"),
        source_preparation=str(preparation), source_preparation_manifest_sha256=prior._sha(preparation/"run.json"),
        source_score_lock_sha256=prior._sha(confirmation/"fresh_score_lock.json"),
        source_geometry_sha256=prior._sha(confirmation/"fresh_queries_unlabeled.csv"),
        source_scores_sha256=prior._sha(confirmation/"fresh_scores_before_labels.csv"),
        source_gate_sha256=prior._sha(confirmation/"frozen_gate_cutoffs.csv"),
        source_labels_sha256=prior._sha(labels_path), execution_implementation_sha256=_source_hashes(backend),
        frozen_implementation_sha256=original_protocol["implementation_sha256"],
        timing_queries=int(timing_queries), repeats=int(repeats), numeric_tolerance=dict(rtol=TOLERANCE, atol=TOLERANCE),
        gate_decisions="exact match; unchanged frozen cutoffs; live canonical fallback only near cutoff",
        training=False, recalibration=False, labels_used_for_selection=False,
        raw_policy="verified original float64 caches copied to resident RAM; every exact call vector-scans every row",
        runtime_protocol=backend_module.protocol(), family="fresh_same_dataset", independent_dataset=False,
        started_utc=datetime.now(timezone.utc).isoformat())
    prior._json(destination/"run.json", state)
    pd.DataFrame([dict(seed=seed, order=order, qId=qid, selection_salt=TIMING_SALT)
                  for seed, ids in selected.items() for order, qid in enumerate(ids)]).to_csv(
                      destination/"query_selection.csv", index=False)
    state["query_selection_sha256"] = prior._sha(destination/"query_selection.csv")
    started = time.perf_counter()
    try:
        state["executable_setup"] = _backend_setup(backend)
        prior._json(destination/"run.json", state)
        tick = time.perf_counter()
        arrays = [np.array(array, dtype=np.float64, copy=True) for array in workflow._raw_arrays(preparation)]
        raw_load_seconds = time.perf_counter()-tick
        state.update(raw_rows=sum(len(p) for p in arrays), raw_coordinate_bytes=sum(p.nbytes for p in arrays),
                     raw_shared_load_seconds=raw_load_seconds,
                     raw_cache_identities=pd.read_csv(preparation/"source_audit.csv")[[
                         "source_id", "raw_cache_sha256", "raw_rows"]].to_dict("records"))
        summaries, parities, costs = [], [], []
        label_index = labels.set_index("qId")
        for position, seed in enumerate(map(int, cfg["seeds"])):
            gate_rows = gates[gates.seed == seed]
            if gate_rows.method.duplicated().any():
                raise ValueError("Duplicate frozen cutoff methods")
            frozen = gate_rows.set_index("method").to_dict("index")
            arms = build_indexed_arms(development, seed, cutoffs=frozen, backend=backend)
            if set(frozen) != {arm.name for arm in arms}:
                raise ValueError("Indexed arm roster differs from frozen gates")
            if set(scores.loc[scores.seed == seed, "method"]) != set(frozen):
                raise ValueError("Indexed arm roster differs from frozen fresh scores")
            calibration_ids = selected_calibration.loc[(selected_calibration.seed == seed) &
                (selected_calibration.role == "calibration"), "qId"].to_numpy()
            calibration = offline_queries.loc[calibration_ids].reset_index()
            timed = label_index.loc[selected[seed]].reset_index()
            def records(frame):
                return [Query(int(q.qId), (q.x0, q.x1, q.y0, q.y1), int(q.truth))
                        for q in frame.itertuples(index=False)]
            counter = ScanCounter(arrays)
            # Shared raw loading is charged once, in the first seed's setup.
            counter.load_seconds = raw_load_seconds if position == 0 else 0.0
            result = run_benchmark(queries=records(timed), calibration_queries=records(calibration),
                arms=arms, histogram_loader=lambda: prior.load_histogram(development/"original_H.hist"),
                estimate_fn=prior.histogram_count, raw_counter=counter,
                output_dir=destination/"e2e"/f"seed_{seed}", epsilon=cfg["epsilon"], delta=cfg["delta"],
                repeats=int(repeats), seed=seed, cache_modes=("resident", "per_query"),
                frozen_cutoffs=frozen, frozen_cutoff_provenance=dict(
                    path=str(confirmation/"frozen_gate_cutoffs.csv"), sha256=state["source_gate_sha256"]),
                metadata=dict(execution_only=True, execution_backend=backend, family="fresh_same_dataset",
                              source_confirmation=str(confirmation), query_selection_salt=TIMING_SALT,
                              runtime_protocol=backend_module.protocol(), executable_setup=state["executable_setup"],
                              ML_feature_backend="indexed Python required features; original frozen model inference"))
            live = result.per_query.assign(seed=seed, phase="timed")
            first = result.first_query.assign(seed=seed, phase="first_request", repeat=-1)
            parity = verify_live_parity(pd.concat((live, first), ignore_index=True), scores, gates)
            parity.to_csv(destination/f"live_parity_seed_{seed}.csv", index=False)
            if not parity.all_matches.all():
                raise ValueError(f"Indexed live output differs from frozen pre-label scores for seed {seed}")
            parities.append(parity)
            summaries.append(result.summary.assign(seed=seed, family="fresh_same_dataset"))
            costs.extend(dict(seed=seed, method=arm.name, **dict(arm.metadata or {})) for arm in arms)
            print(f"{backend.capitalize()} execution seed {seed}: every live output/decision matches frozen scores", flush=True)
        pd.concat(summaries, ignore_index=True).to_csv(destination/"summary.csv", index=False)
        pd.concat(parities, ignore_index=True).to_csv(destination/"live_parity.csv", index=False)
        pd.DataFrame(costs).to_csv(destination/"payload_costs.csv", index=False)
        workflow.verify_confirmation_lock(confirmation)
        workflow._load_preparation(preparation)
        if _source_hashes(backend) != state["execution_implementation_sha256"]:
            raise ValueError("Execution implementation changed during measurement")
        state.update(status="complete", wall_seconds=time.perf_counter()-started,
                     completed_utc=datetime.now(timezone.utc).isoformat(), all_live_outputs_and_decisions_match=True)
        prior._json(destination/"run.json", state)
        (destination/"analysis_en.md").write_text(
            "# Frozen execution replay\n\nOnly execution changed. The original native/packet/model/fine bytes, "
            "H and frozen calibration gates are reused. Timing IDs use the original deterministic selection. "
            "Every timed/first-request assessor output matches the frozen pre-label scores within 1e-9; gate "
            "decisions match exactly. Every exact baseline or fallback executes a resident raw vector scan, "
            "independently checked after all timed requests. Initialization and first requests are separate. "
            "Compiled execution, when explicitly selected, records shared executable code/compiler/cache setup separately; "
            "ML feature math remains the indexed Python implementation and original model inference. "
            "Per-query mode reloads local payloads; OS caches are uncontrolled. Read latency and accepted "
            "violation together; acceleration does not imply an end-to-end speedup.\n")
        return destination
    except BaseException as exc:
        state.update(status="failed", error=f"{type(exc).__name__}: {exc}", wall_seconds=time.perf_counter()-started)
        prior._json(destination/"run.json", state)
        raise
