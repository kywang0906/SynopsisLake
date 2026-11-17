"""Python end-to-end gate benchmark with live assessors and real raw scans.

The histogram estimator is shared by every assessor. A scorer receives only
``(bounds, H, artifact)`` (plus the actual loaded histogram when explicitly
requested); neither offline truth nor a saved score enters a request.
``artifact_loader`` and ``histogram_loader`` must return fresh runtime
objects when called. In ``per_query`` mode their local reads and construction
are part of request time. Raw arrays are resident in both modes.

Example::

    arm = Arm('disagreement', lambda q, H, sidecar:
              ScoreResult(abs(H - sidecar.count(q))/max(H, 1)))
    result = run_benchmark(queries=test, calibration_queries=cal, arms=[arm],
        histogram_loader=load_H, estimate_fn=histogram_count,
        raw_counter=ScanCounter([points]), output_dir='out/e2e')

Use the repository's H/SA helpers in these callbacks. This module measures a
Python backend, not Java, Spark, or remote storage. The optional first-request
measurements follow offline calibration, and do not imply a cold OS cache.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence
import json
import hashlib
import math
import platform
import time

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Query:
    qId: Any
    bounds: Sequence[float]
    truth: float

    def __post_init__(self):
        bounds = tuple(map(float, self.bounds))
        if (len(bounds) != 4 or not np.isfinite(bounds).all()
                or bounds[0] > bounds[1] or bounds[2] > bounds[3]):
            raise ValueError('Query bounds must be four finite increasing closed bounds')
        if not math.isfinite(float(self.truth)) or float(self.truth) < 0:
            raise ValueError('Offline truth must be finite and nonnegative')
        object.__setattr__(self, 'bounds', bounds)
        object.__setattr__(self, 'truth', float(self.truth))


@dataclass(frozen=True)
class ScoreResult:
    score: float
    reference: Optional[float] = None


@dataclass(frozen=True)
class Arm:
    """Live assessor; return a score or ``ScoreResult`` (lower is safer).

    Set ``score_uses_histogram=True`` for callbacks accepting a fourth argument,
    the actual histogram object used for this request's H estimate. This lets
    SA reuse the current per-query payload instead of a captured resident H.
    """
    name: str
    score_fn: Callable[..., Any]
    artifact_loader: Optional[Callable[[], Any]] = None
    metadata: Optional[Mapping[str, Any]] = None
    score_uses_histogram: bool = False


@dataclass
class BenchmarkResult:
    per_query: pd.DataFrame
    quality_cases: pd.DataFrame
    summary: pd.DataFrame
    cutoffs: pd.DataFrame
    setup: pd.DataFrame
    first_query: pd.DataFrame
    output_dir: Path


class ScanCounter:
    """Exact inclusive vector scans of resident raw Nx2 float64 arrays.

    Every ``count`` scans every raw row, including duplicate coordinates. There
    is no index or answer cache. Float64 input arrays are retained by reference,
    so callers must keep them unchanged during a benchmark. Extra raw fields
    and row weights are not interpreted: every input row has unit mass.
    """

    def __init__(self, sources):
        started = time.perf_counter()
        self.sources = []
        for source in sources:
            points = np.asarray(source, dtype=np.float64)
            if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
                raise ValueError('Raw sources must be finite Nx2 arrays')
            self.sources.append(points)
        if not self.sources:
            raise ValueError('At least one raw source is required')
        self.raw_rows = sum(len(p) for p in self.sources)
        self.raw_bytes = sum(p.nbytes for p in self.sources)
        self.initialization_seconds = time.perf_counter() - started
        self.load_seconds = 0.0

    @classmethod
    def from_files(cls, paths, iter_points):
        """Load each source with an external TAB/gzip chunk reader, once."""
        started = time.perf_counter()
        sources = []
        for path in paths:
            chunks = [np.asarray(chunk, dtype=np.float64) for chunk in iter_points(path)]
            sources.append(np.concatenate(chunks) if chunks else np.empty((0, 2)))
        result = cls(sources)
        result.load_seconds = time.perf_counter() - started - result.initialization_seconds
        return result

    def count(self, bounds):
        x0, x1, y0, y1 = bounds
        return sum(int(np.count_nonzero((p[:, 0] >= x0) & (p[:, 0] <= x1)
                                        & (p[:, 1] >= y0) & (p[:, 1] <= y1)))
                   for p in self.sources)


def relative_error(answer, truth):
    """Zero truth has error zero only for an exactly zero answer."""
    return abs(answer-truth)/truth if truth > 0 else (0.0 if answer == 0 else math.inf)


def empirical_cutoff(scores, errors, epsilon=.25, delta=.05):
    """Largest observed finite cutoff meeting empirical calibration risk.

    Stable sorting and complete equal-score groups match ``gate_eval.cutoff``.
    A bad early prefix can recover when subsequent accepted queries are good.
    Nonfinite scores are always rejected. With no eligible prefix, return -inf.
    This empirical calibration is not a population-risk guarantee.
    """
    if not math.isfinite(epsilon) or epsilon < 0 or not 0 <= delta <= 1:
        raise ValueError('Invalid epsilon or delta')
    scores, errors = np.asarray(scores, float), np.asarray(errors, float)
    if scores.ndim != 1 or errors.shape != scores.shape or np.isnan(errors).any() or (errors < 0).any():
        raise ValueError('Scores and nonnegative errors must be equal-length vectors')
    finite = np.isfinite(scores)
    scores, errors = scores[finite], errors[finite]
    if not len(scores):
        return -math.inf
    order = np.argsort(scores, kind='stable')
    scores, bad = scores[order], errors[order] > epsilon
    risk = np.cumsum(bad)/np.arange(1, len(scores)+1)
    complete_tie = np.r_[scores[1:] != scores[:-1], True]
    valid = np.flatnonzero((risk <= delta) & complete_tie)
    return float(scores[valid[-1]]) if len(valid) else -math.inf


def _score(value):
    if isinstance(value, ScoreResult):
        return ScoreResult(float(value.score), None if value.reference is None else float(value.reference))
    return ScoreResult(float(value))


def _invoke_score(arm, bounds, estimate, artifact, histogram):
    """Optionally pass the actual current H object, e.g. for live SA math."""
    if arm.score_uses_histogram:
        return _score(arm.score_fn(bounds, estimate, artifact, histogram))
    return _score(arm.score_fn(bounds, estimate, artifact))


def _json_value(value):
    if isinstance(value, Mapping):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    if isinstance(value, (np.integer, np.floating)):
        return _json_value(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return '-inf' if value < 0 else 'inf' if value > 0 else None
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path, value):
    path.write_text(json.dumps(_json_value(value), indent=2, allow_nan=False, default=str)+'\n')


STAGES = ('histogram_load_ns', 'artifact_load_ns', 'estimate_ns', 'score_ns', 'gate_ns', 'raw_ns', 'total_ns')


def run_benchmark(*, queries, calibration_queries, arms, histogram_loader,
                  estimate_fn, raw_counter, output_dir, epsilon=.25, delta=.05,
                  cache_modes=('resident', 'per_query'), repeats=3,
                  warmup_queries=(), seed=0, measure_first_query=True,
                  truth_check_limit=8, metadata=None, frozen_cutoffs=None,
                  frozen_cutoff_provenance=None):
    """Calibrate offline, execute live paired requests, and persist CSV/JSON.

    Calibration and test IDs must be disjoint. Warm-up IDs must belong to the
    calibration cohort. Gate cutoffs depend only on calibration errors. Truth
    is consulted outside request timers for validation and final quality.

    To reuse a gate, pass ``frozen_cutoffs`` as a mapping from every assessor
    name to a record containing ``cutoff``, ``epsilon`` and ``delta``. For
    example, read ``gate_cutoffs.csv`` with pandas and call
    ``table.set_index('method').to_dict('index')``. Targets must exactly match
    this invocation; cutoffs must be finite or negative infinity. In this mode
    no calibration estimates, scores, or cutoff selections are computed.
    ``frozen_cutoff_provenance`` should identify the source file and its SHA256;
    loading that file is the caller's separately recorded initialization cost.

    ``raw_counter`` is a ``ScanCounter`` or any object whose ``count(bounds)``
    really scans the raw data. The same instance serves the exact baseline and
    every rejected request. After all timed requests, a fresh scan per unique
    test geometry checks every exact return. No validation scan is interleaved
    with timed arms. At most ``truth_check_limit`` deterministically selected
    calibration/test geometries are also audited before timing (zero disables
    this prepass). Sampling does not inspect truth. No supplied truth
    substitutes for raw execution.
    """
    queries, calibration_queries, arms = list(queries), list(calibration_queries), list(arms)
    warmup_queries, cache_modes = list(warmup_queries), tuple(cache_modes)
    empirical_cutoff([], [], epsilon, delta)  # Validate settings before writing artifacts.
    if not queries or not calibration_queries or not arms:
        raise ValueError('Nonempty test, calibration and assessor cohorts are required')
    if repeats < 1 or int(repeats) != repeats:
        raise ValueError('repeats must be a positive integer')
    if truth_check_limit < 0 or int(truth_check_limit) != truth_check_limit:
        raise ValueError('truth_check_limit must be a nonnegative integer')
    if not cache_modes or len(set(cache_modes)) != len(cache_modes) or not set(cache_modes) <= {'resident', 'per_query'}:
        raise ValueError('Use unique resident/per_query cache modes')
    names = [arm.name for arm in arms]
    if len(set(names)) != len(names) or 'always_exact' in names or any(not n for n in names):
        raise ValueError('Assessor names must be unique, nonempty and distinct from always_exact')
    for cohort in (queries, calibration_queries):
        if len({q.qId for q in cohort}) != len(cohort):
            raise ValueError('Duplicate query ID in cohort')
    calibration_by_id = {q.qId: q for q in calibration_queries}
    if set(calibration_by_id) & {q.qId for q in queries}:
        raise ValueError('Calibration and test IDs overlap')
    if any(q.qId not in calibration_by_id or q != calibration_by_id[q.qId] for q in warmup_queries):
        raise ValueError('Warm-up queries must come from calibration')
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    if (destination/'run.json').exists():
        raise FileExistsError('Refusing to overwrite an E2E run')
    state = dict(status='running', backend='Python', epsilon=epsilon, delta=delta,
                 repeats=int(repeats), seed=int(seed), cache_modes=cache_modes,
                 raw_access='resident raw arrays; fresh inclusive vector scan for every exact request',
                 histogram_access='resident or per-query local payload reload; OS cache uncontrolled',
                 calibration='empirical complete-tie prefixes; calibration only; no training',
                 first_query_scope='first measured request after offline calibration, before explicit warm-up; not cold storage',
                 truth_check_limit=int(truth_check_limit),
                 validation_scope='bounded deterministic prepass; fresh exact validation after all timed requests; no interleaved validation scans',
                 python=platform.python_version(), metadata=metadata or {},
                 assessors={a.name: a.metadata or {} for a in arms})
    state['gate_mode'] = 'frozen' if frozen_cutoffs is not None else 'calibration'
    if frozen_cutoffs is not None:
        state['calibration'] = 'serialized frozen cutoffs reused; no scoring or gate selection on supplied calibration truth'
        state['frozen_cutoff_provenance'] = frozen_cutoff_provenance or {'kind': 'caller supplied mapping'}
    _write_json(destination/'run.json', state)
    setup_rows, first_rows, rows = [], [], []
    started = time.perf_counter()

    def setup(stage, elapsed_ns, **info):
        setup_rows.append(dict(stage=stage, elapsed_ns=int(elapsed_ns), **info))

    def timed_load(loader, stage, **info):
        tick = time.perf_counter_ns()
        value = loader() if loader else None
        setup(stage, time.perf_counter_ns()-tick, **info)
        return value

    setup('raw_load', getattr(raw_counter, 'load_seconds', 0)*1e9,
          raw_rows=getattr(raw_counter, 'raw_rows', None), raw_bytes=getattr(raw_counter, 'raw_bytes', None))
    setup('raw_validation_and_initialization', getattr(raw_counter, 'initialization_seconds', 0)*1e9)
    try:
        cutoffs, cutoff_rows, calibration_rows = {}, [], []
        if frozen_cutoffs is not None:
            tick = time.perf_counter_ns()
            if not isinstance(frozen_cutoffs, Mapping) or set(frozen_cutoffs) != set(names):
                raise ValueError('Frozen cutoff method names must match assessor names exactly')
            for name in names:
                record = frozen_cutoffs[name]
                if not isinstance(record, Mapping) or not {'cutoff', 'epsilon', 'delta'} <= set(record):
                    raise ValueError('Frozen cutoff records require cutoff, epsilon and delta metadata')
                if float(record['epsilon']) != float(epsilon) or float(record['delta']) != float(delta):
                    raise ValueError('Frozen cutoff epsilon/delta metadata differs from requested targets')
                cutoff = float(record['cutoff'])
                if not (math.isfinite(cutoff) or cutoff == -math.inf):
                    raise ValueError('Frozen cutoff must be finite or negative infinity')
                cutoffs[name] = cutoff
                cutoff_rows.append(dict(record, method=name, cutoff=cutoff, epsilon=epsilon, delta=delta,
                                        calibration_source='frozen artifact; no recalibration'))
            setup('frozen_cutoff_validation', time.perf_counter_ns()-tick)
            setup('offline_calibration', 0)
            state['calibration_seconds'] = 0.0
        hist = timed_load(histogram_loader, 'histogram_initialization')
        artifacts = {a.name: timed_load(a.artifact_loader, 'assessor_initialization', method=a.name) for a in arms}
        if frozen_cutoffs is None:
            tick = time.perf_counter_ns()
            calibration_values = {a.name: [] for a in arms}
            for q in calibration_queries:
                estimate = float(estimate_fn(q.bounds, hist))
                if not math.isfinite(estimate) or estimate < 0:
                    raise ValueError('H must be finite and nonnegative')
                for arm in arms:
                    result = _invoke_score(arm, q.bounds, estimate, artifacts[arm.name], hist)
                    calibration_values[arm.name].append((result.score, relative_error(estimate, q.truth)))
                    calibration_rows.append(dict(qId=q.qId, method=arm.name, H=estimate,
                                                 score=result.score, reference=result.reference,
                                                 truth=q.truth, relative_error=relative_error(estimate, q.truth)))
            for arm in arms:
                scores, errors = np.asarray(calibration_values[arm.name]).T
                cutoff = empirical_cutoff(scores, errors, epsilon, delta)
                cutoffs[arm.name] = cutoff
                accepted = np.isfinite(scores) & (scores <= cutoff)
                n, bad = int(accepted.sum()), int(np.count_nonzero((errors > epsilon) & accepted))
                cutoff_rows.append(dict(method=arm.name, cutoff=cutoff, epsilon=epsilon, delta=delta,
                                        calibration_queries=len(scores), calibration_accepted=n,
                                        calibration_violations=bad,
                                        calibration_violation_rate=bad/n if n else np.nan,
                                        calibration_source='calibration only'))
            calibration_ns = time.perf_counter_ns()-tick
            setup('offline_calibration', calibration_ns)
            state['calibration_seconds'] = calibration_ns/1e9
        cutoff_frame = pd.DataFrame(cutoff_rows)
        cutoff_frame.to_csv(destination/'gate_cutoffs.csv', index=False)
        calibration_frame = pd.DataFrame(calibration_rows, columns=['qId', 'method', 'H', 'score', 'reference', 'truth', 'relative_error'])
        calibration_frame['cutoff'] = calibration_frame.method.map(cutoffs)
        calibration_frame['accepted'] = np.isfinite(calibration_frame.score.astype(float)) & (calibration_frame.score <= calibration_frame.cutoff)
        calibration_frame.to_csv(destination/'calibration_scores.csv', index=False)
        pd.DataFrame([dict(qId=q.qId, split=split, x0=q.bounds[0], x1=q.bounds[1],
                           y0=q.bounds[2], y1=q.bounds[3], truth=q.truth)
                      for split, cohort in [('calibration', calibration_queries), ('test', queries)]
                      for q in cohort]).to_csv(destination/'query_manifest.csv', index=False)
        state['gate_cutoffs'] = cutoffs
        state['training_seconds'] = 0.0
        _write_json(destination/'run.json', state)
        # Deterministic bounded validation only. No result is an exact answer.
        geometries = {}
        for q in [*calibration_queries, *queries]:
            if q.bounds in geometries and q.truth != geometries[q.bounds].truth:
                raise ValueError('Inconsistent supplied truth for one geometry')
            geometries[q.bounds] = q
        audit_queries = sorted(geometries.values(), key=lambda q: hashlib.sha256(
            (str(seed)+':'+repr(q.bounds)).encode()).digest())[:int(truth_check_limit)]
        tick = time.perf_counter_ns()
        for q in audit_queries:
            if raw_counter.count(q.bounds) != q.truth:
                raise ValueError('Offline truth differs from a fresh raw scan: '+str(q.qId))
        setup('offline_raw_truth_validation', time.perf_counter_ns()-tick, query_count=len(audit_queries))
        state['pre_timing_truth_audit_ids'] = [q.qId for q in audit_queries]
        arms_by_name = {a.name: a for a in arms}

        def execute(bounds, method, cache_mode):
            # No Query object, supplied truth, or saved score is available here.
            begin = time.perf_counter_ns()
            result = {stage: 0 for stage in STAGES}
            result.update(accepted=False, answer=np.nan, H=np.nan, score=np.nan, reference=np.nan, cutoff=np.nan)
            if method == 'always_exact':
                tick = time.perf_counter_ns()
                result['answer'] = float(raw_counter.count(bounds))
                result['raw_ns'] = time.perf_counter_ns()-tick
            else:
                arm = arms_by_name[method]
                tick = time.perf_counter_ns()
                current_hist = hist if cache_mode == 'resident' else histogram_loader()
                result['histogram_load_ns'] = time.perf_counter_ns()-tick
                tick = time.perf_counter_ns()
                artifact = artifacts[method] if cache_mode == 'resident' else (arm.artifact_loader() if arm.artifact_loader else None)
                result['artifact_load_ns'] = time.perf_counter_ns()-tick
                tick = time.perf_counter_ns()
                result['H'] = float(estimate_fn(bounds, current_hist))
                result['estimate_ns'] = time.perf_counter_ns()-tick
                tick = time.perf_counter_ns()
                score = _invoke_score(arm, bounds, result['H'], artifact, current_hist)
                result['score'], result['reference'] = score.score, score.reference
                result['score_ns'] = time.perf_counter_ns()-tick
                tick = time.perf_counter_ns()
                result['cutoff'] = cutoffs[method]
                result['accepted'] = math.isfinite(score.score) and score.score <= cutoffs[method]
                result['gate_ns'] = time.perf_counter_ns()-tick
                if result['accepted']:
                    result['answer'] = result['H']
                else:
                    tick = time.perf_counter_ns()
                    result['answer'] = float(raw_counter.count(bounds))
                    result['raw_ns'] = time.perf_counter_ns()-tick
            result['total_ns'] = time.perf_counter_ns()-begin
            return result

        rng = np.random.default_rng(seed)
        methods = ['always_exact', *names]
        for cache_mode in rng.permutation(cache_modes):
            if measure_first_query:
                for order, method in enumerate(rng.permutation(methods)):
                    result = execute(queries[0].bounds, method, str(cache_mode))
                    first_rows.append(dict(cache_mode=cache_mode, method=method, qId=queries[0].qId,
                                           arm_order=order, scope='after calibration, before warm-up', **result))
            tick = time.perf_counter_ns()
            for q in warmup_queries:
                for method in rng.permutation(methods):
                    execute(q.bounds, method, str(cache_mode))
            setup('explicit_calibration_warmup', time.perf_counter_ns()-tick, cache_mode=cache_mode,
                  query_count=len(warmup_queries))
            for repeat in range(int(repeats)):
                for index in rng.permutation(len(queries)):
                    q = queries[index]
                    for order, method in enumerate(rng.permutation(methods)):
                        result = execute(q.bounds, method, str(cache_mode))
                        rows.append(dict(cache_mode=cache_mode, repeat=repeat, qId=q.qId,
                                         method=method, arm_order=order, truth=q.truth, **result))
        # Verification runs only once every measured arm has finished. This
        # leaves paired-method cache effects to their randomized execution order.
        tick = time.perf_counter_ns()
        test_by_id = {q.qId: q for q in queries}
        fresh_counts = {}
        for q in queries:
            if q.bounds not in fresh_counts:
                fresh_counts[q.bounds] = float(raw_counter.count(q.bounds))
            if fresh_counts[q.bounds] != q.truth:
                raise ValueError('Offline truth differs from post-timing fresh raw scan: '+str(q.qId))
        expected_H = {}
        for result in [*first_rows, *rows]:
            q = test_by_id[result['qId']]
            if result['method'] != 'always_exact':
                H = result['H']
                if not math.isfinite(H) or H < 0:
                    raise ValueError('Invalid runtime H')
                prior = expected_H.setdefault(q.bounds, H)
                if not math.isclose(H, prior, rel_tol=1e-12, abs_tol=1e-8):
                    raise ValueError('H changed across assessor arms, repetitions or cache modes')
            if not result['accepted'] and result['answer'] != fresh_counts[q.bounds]:
                raise ValueError('Returned exact answer differs from post-timing fresh raw scan: '+str(q.qId))
        setup('post_request_exact_and_H_validation', time.perf_counter_ns()-tick,
              query_count=len(fresh_counts))
        frame = pd.DataFrame(rows)
        keys = ['cache_mode', 'method', 'qId']
        grouped = frame.groupby(keys, dropna=False)
        if not grouped.size().eq(repeats).all() or (grouped[['answer', 'accepted']].nunique() > 1).any().any():
            raise ValueError('Incomplete repetitions or unstable decisions/answers')
        quality = frame.drop_duplicates(keys).copy()
        quality['relative_error'] = [relative_error(a, t) for a, t in zip(quality.answer, quality.truth)]
        quality['violating_answer'] = quality.relative_error > epsilon
        quality['accepted_violation'] = quality.accepted & quality.violating_answer
        frame = frame.merge(quality[keys+['relative_error', 'violating_answer', 'accepted_violation']],
                            on=keys, validate='many_to_one')
        summaries = []
        for (cache_mode, method), group in frame.groupby(['cache_mode', 'method'], sort=True):
            unique = quality[(quality.cache_mode == cache_mode) & (quality.method == method)]
            n, accepted, bad = len(unique), int(unique.accepted.sum()), int(unique.accepted_violation.sum())
            row = dict(cache_mode=cache_mode, method=method, query_count=n,
                       timing_observations=len(group), repeats=int(repeats), accepted_count=accepted,
                       violation_count=bad, acceptance=accepted/n,
                       accepted_violation_rate=bad/accepted if accepted else np.nan,
                       quality_target_met=bad/accepted <= delta if accepted else None,
                       returned_violation_count=int(unique.violating_answer.sum()),
                       returned_mean_relative_error=float(unique.relative_error.mean()),
                       eligible_truth_ge_10_count=int((unique.truth >= 10).sum()),
                       small_truth_lt_10_count=int((unique.truth < 10).sum()),
                       zero_truth_count=int((unique.truth == 0).sum()),
                       epsilon=epsilon, delta=delta, backend='Python', raw_access='resident arrays')
            for stage in STAGES:
                row[stage.replace('_ns', '_mean_ms')] = float(group[stage].mean())/1e6
                row[stage.replace('_ns', '_p95_ms')] = float(group[stage].quantile(.95))/1e6
            summaries.append(row)
        summary = pd.DataFrame(summaries)
        baseline = summary[summary.method == 'always_exact'][['cache_mode', 'total_mean_ms', 'total_p95_ms']].rename(
            columns={'total_mean_ms': 'baseline_total_mean_ms', 'total_p95_ms': 'baseline_total_p95_ms'})
        summary = summary.merge(baseline, on='cache_mode', validate='many_to_one')
        summary['speedup_mean'] = summary.baseline_total_mean_ms/summary.total_mean_ms
        summary['speedup_p95_ratio'] = summary.baseline_total_p95_ms/summary.total_p95_ms
        diagnostics = []
        for (cache_mode, method), group in quality.groupby(['cache_mode', 'method'], sort=True):
            for cohort, mask in [('all', np.ones(len(group), dtype=bool)),
                                 ('truth_ge_10', group.truth >= 10),
                                 ('truth_lt_10', group.truth < 10),
                                 ('truth_zero', group.truth == 0)]:
                subset = group.loc[mask]
                n, accepted = len(subset), int(subset.accepted.sum())
                bad = int(subset.accepted_violation.sum())
                diagnostics.append(dict(cache_mode=cache_mode, method=method, truth_cohort=cohort,
                    query_count=n, accepted_count=accepted, violation_count=bad,
                    acceptance=accepted/n if n else np.nan,
                    accepted_violation_rate=bad/accepted if accepted else np.nan,
                    quality_target_met=bad/accepted <= delta if accepted else None,
                    returned_violation_count=int(subset.violating_answer.sum()),
                    returned_mean_relative_error=float(subset.relative_error.mean()) if n else np.nan))
        setup_frame, first_frame = pd.DataFrame(setup_rows), pd.DataFrame(first_rows)
        frame.to_csv(destination/'per_query.csv', index=False)
        quality.to_csv(destination/'quality_cases.csv', index=False)
        summary.to_csv(destination/'summary.csv', index=False)
        pd.DataFrame(diagnostics).to_csv(destination/'quality_by_truth.csv', index=False)
        setup_frame.to_csv(destination/'setup.csv', index=False)
        first_frame.to_csv(destination/'first_query.csv', index=False)
        if not first_frame.empty:
            cold = first_frame[['cache_mode', 'method', 'qId', 'total_ns']].copy()
            cold['first_request_ms'] = cold.total_ns/1e6
            cold.to_csv(destination/'first_query_summary.csv', index=False)
        state.update(status='complete', wall_seconds=time.perf_counter()-started,
                     raw_rows=getattr(raw_counter, 'raw_rows', None), raw_bytes=getattr(raw_counter, 'raw_bytes', None),
                     test_queries=len(queries), calibration_queries=len(calibration_queries))
        _write_json(destination/'run.json', state)
        return BenchmarkResult(frame, quality, summary, cutoff_frame, setup_frame, first_frame, destination)
    except BaseException as error:
        state.update(status='failed', error=f'{type(error).__name__}: {error}', wall_seconds=time.perf_counter()-started)
        pd.DataFrame(setup_rows).to_csv(destination/'setup.csv', index=False)
        _write_json(destination/'run.json', state)
        raise
