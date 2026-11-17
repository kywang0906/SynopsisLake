"""Tests for empirical calibration and the actual raw fallback request path."""
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'ml'))
from sidecar_e2e import Arm, Query, ScanCounter, ScoreResult, empirical_cutoff, run_benchmark


class CalibrationTest(unittest.TestCase):
    def test_complete_ties_and_recovery_after_bad_prefix(self):
        self.assertEqual(empirical_cutoff([0, 0, 1], [0, 1, 1], delta=.25), -math.inf)
        self.assertEqual(empirical_cutoff([0, 1, 2, 3], [1, 0, 0, 0], delta=.25), 3)
        self.assertEqual(empirical_cutoff([1, 0, 0, 2], [1, 0, 1, 1], delta=.25), -math.inf)

    def test_nonfinite_scores_and_empty_input_reject(self):
        self.assertEqual(empirical_cutoff([], []), -math.inf)
        self.assertEqual(empirical_cutoff([np.nan, np.inf], [0, 0]), -math.inf)


class ScanCounterTest(unittest.TestCase):
    def test_closed_bounds_duplicates_and_fresh_mutated_raw(self):
        points = np.array([[0., 0.], [1., 1.], [1., 1.], [2., 2.]])
        counter = ScanCounter([points])
        self.assertEqual(counter.count((0, 1, 0, 1)), 3)
        self.assertEqual(counter.count((1, 1, 1, 1)), 2)
        points[1] = [3, 3]
        self.assertEqual(counter.count((0, 1, 0, 1)), 2)


class EndToEndTest(unittest.TestCase):
    def test_scorer_can_use_actual_per_request_loaded_H(self):
        raw = ScanCounter([np.array([[0., 0.], [1., 1.]])])
        serial, observed = [0], []

        def load_histogram():
            serial[0] += 1
            return {'estimate': 1., 'serial': serial[0]}

        def score(bounds, H, artifact, histogram):
            self.assertEqual(H, histogram['estimate'])
            observed.append(histogram['serial'])
            return 0.

        with tempfile.TemporaryDirectory() as folder:
            run_benchmark(queries=[Query('test', (0, 1, 0, 1), 2)],
                calibration_queries=[Query('cal', (0, .5, 0, .5), 1)],
                arms=[Arm('sa', score, score_uses_histogram=True)], histogram_loader=load_histogram,
                estimate_fn=lambda q, h: h['estimate'], raw_counter=raw, output_dir=folder,
                cache_modes=('per_query',), repeats=2, measure_first_query=False, truth_check_limit=0)
            self.assertEqual(observed, [1, 2, 3])  # calibration, then two freshly loaded contexts

    def test_serialized_frozen_gate_overrides_new_calibration_without_scoring_it(self):
        raw = ScanCounter([np.array([[0., 0.], [1., 1.]])])
        cal = [Query('cal', (0, .5, 0, .5), 1)]
        query = [Query('test', (0, 1, 0, 1), 2)]
        with tempfile.TemporaryDirectory() as folder:
            frozen_path = Path(folder)/'saved_cutoffs.csv'
            pd.DataFrame([dict(method='assessor', cutoff=-math.inf, epsilon=.25, delta=.05)]).to_csv(frozen_path, index=False)
            frozen = pd.read_csv(frozen_path).set_index('method').to_dict('index')

            def no_calibration_score(bounds, H, artifact):
                self.assertNotEqual(bounds, cal[0].bounds)
                return 0.

            result = run_benchmark(queries=query, calibration_queries=cal,
                arms=[Arm('assessor', no_calibration_score)], histogram_loader=lambda: None,
                estimate_fn=lambda q, h: 1., raw_counter=raw, output_dir=Path(folder)/'frozen',
                cache_modes=('resident',), repeats=1, measure_first_query=False,
                frozen_cutoffs=frozen, frozen_cutoff_provenance={'path': str(frozen_path)}, truth_check_limit=0)
            self.assertEqual(result.cutoffs.cutoff.iloc[0], -math.inf)
            self.assertEqual(result.summary[result.summary.method == 'assessor'].accepted_count.iloc[0], 0)
            self.assertEqual(result.setup[result.setup.stage == 'offline_calibration'].elapsed_ns.iloc[0], 0)
            self.assertTrue(pd.read_csv(Path(folder)/'frozen/calibration_scores.csv').empty)
            state = json.loads((Path(folder)/'frozen/run.json').read_text())
            self.assertEqual(state['gate_mode'], 'frozen')
            self.assertEqual(state['frozen_cutoff_provenance']['path'], str(frozen_path))
            # Fresh calibration on the same good calibration row would accept;
            # reusing the persisted reject-all gate must retain rejection.
            fresh = run_benchmark(queries=query, calibration_queries=cal,
                arms=[Arm('assessor', lambda q, H, a: 0.)], histogram_loader=lambda: None,
                estimate_fn=lambda q, h: 1., raw_counter=raw, output_dir=Path(folder)/'recalibrated',
                cache_modes=('resident',), repeats=1, measure_first_query=False, truth_check_limit=0)
            self.assertEqual(fresh.summary[fresh.summary.method == 'assessor'].accepted_count.iloc[0], 1)

    def test_frozen_gate_rejects_different_targets_names_and_invalid_cutoffs(self):
        raw = ScanCounter([np.array([[0., 0.], [1., 1.]])])
        records = [({'assessor': {'cutoff': 0., 'epsilon': .1, 'delta': .05}}, 'metadata differs'),
                   ({'other': {'cutoff': 0., 'epsilon': .25, 'delta': .05}}, 'names must match'),
                   ({'assessor': {'cutoff': math.inf, 'epsilon': .25, 'delta': .05}}, 'finite or negative'),
                   ({'assessor': {'cutoff': math.nan, 'epsilon': .25, 'delta': .05}}, 'finite or negative')]
        with tempfile.TemporaryDirectory() as folder:
            for i, (record, message) in enumerate(records):
                with self.subTest(record=record), self.assertRaisesRegex(ValueError, message):
                    run_benchmark(queries=[Query('test', (0, 1, 0, 1), 2)],
                        calibration_queries=[Query('cal', (0, .5, 0, .5), 1)],
                        arms=[Arm('assessor', lambda q, H, a: 0.)], histogram_loader=lambda: None,
                        estimate_fn=lambda q, h: 1., raw_counter=raw, output_dir=Path(folder)/str(i),
                        cache_modes=('resident',), repeats=1, frozen_cutoffs=record)

    def test_exact_validation_scans_once_after_all_timed_requests(self):
        class CountingScan(ScanCounter):
            calls = 0

            def count(self, bounds):
                self.calls += 1
                return super().count(bounds)

        raw = CountingScan([np.array([[0., 0.], [1., 1.]])])
        seen_counts = []

        def score(q, H, a):
            seen_counts.append(raw.calls)
            return 1.

        with tempfile.TemporaryDirectory() as folder:
            run_benchmark(queries=[Query('test', (0, 1, 0, 1), 2)],
                calibration_queries=[Query('cal', (0, .5, 0, .5), 1)],
                arms=[Arm('reject', score)], histogram_loader=lambda: None,
                estimate_fn=lambda q, h: 100., raw_counter=raw, output_dir=folder,
                cache_modes=('resident',), repeats=2, measure_first_query=False, truth_check_limit=0)
            # Four real timed scans (baseline + fallback, twice), then one
            # fresh post-timing scan. No verification scan separates arms.
            self.assertEqual(raw.calls, 5)
            self.assertLess(max(seen_counts), 4)

    def test_reloads_live_scorers_exact_fallback_and_repeat_dedup(self):
        points = np.array([[0., 0.], [1., 1.], [1., 1.], [2., 2.]])
        raw = ScanCounter([points])
        cal = [Query('cal', (-1, .5, -1, .5), 1)]
        test = [Query('a', (0, 1, 0, 1), 3), Query('b', (2, 2, 2, 2), 1)]
        loads, seen = {'H': 0, 'artifact': 0}, []

        def histogram_loader():
            loads['H'] += 1
            return object()

        def artifact_loader():
            loads['artifact'] += 1
            return object()

        def score(bounds, estimate, artifact):
            self.assertIsNotNone(artifact)
            self.assertEqual(len(bounds), 4)
            seen.append(tuple(bounds))
            return ScoreResult(1., 2.)

        with tempfile.TemporaryDirectory() as folder:
            result = run_benchmark(queries=test, calibration_queries=cal,
                arms=[Arm('reject', score, artifact_loader)], histogram_loader=histogram_loader,
                estimate_fn=lambda q, h: 100., raw_counter=raw, output_dir=folder,
                repeats=3, warmup_queries=cal, seed=9)
            self.assertEqual(len(result.per_query), 24)
            self.assertEqual(len(result.quality_cases), 8)
            self.assertEqual(len(seen), 1+2+2+12)  # calibration, first, warm-up, timed
            self.assertEqual(loads, {'H': 1+1+1+6, 'artifact': 1+1+1+6})
            self.assertTrue((result.per_query.answer == result.per_query.truth).all())
            self.assertTrue((result.per_query.raw_ns > 0).all())
            rejected = result.summary[result.summary.method == 'reject']
            self.assertTrue(rejected.accepted_violation_rate.isna().all())
            self.assertTrue(rejected.quality_target_met.isna().all())
            self.assertTrue((rejected.query_count == 2).all())
            self.assertEqual(json.loads((Path(folder)/'run.json').read_text())['backend'], 'Python')

    def test_fallback_reads_raw_after_scorer_mutation(self):
        # Mutation occurs during the live scorer, after offline truth validation.
        # Removing one point changes the genuine scan; a truth-table fallback
        # would return the old supplied answer and fail the expected exception.
        points = np.array([[0., 0.], [1., 1.]])
        calls = [0]

        def scorer(bounds, estimate, artifact):
            calls[0] += 1
            if calls[0] > 1:
                points[1] = [9, 9]
            return 1.

        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError, 'fresh raw scan'):
                run_benchmark(queries=[Query('test', (0, 1, 0, 1), 2)],
                    calibration_queries=[Query('cal', (0, .5, 0, .5), 1)],
                    arms=[Arm('reject', scorer)], histogram_loader=lambda: None,
                    estimate_fn=lambda q, h: 100., raw_counter=ScanCounter([points]),
                    output_dir=folder, cache_modes=('resident',), repeats=1, measure_first_query=False)

    def test_accepts_with_live_H_and_calibration_only_gate(self):
        raw = ScanCounter([np.array([[0., 0.], [1., 1.]])])
        with tempfile.TemporaryDirectory() as folder:
            result = run_benchmark(queries=[Query('test', (0, 1, 0, 1), 2)],
                calibration_queries=[Query('cal', (0, .5, 0, .5), 1)],
                arms=[Arm('accept', lambda q, H, a: 0.)], histogram_loader=lambda: None,
                estimate_fn=lambda q, h: 1., raw_counter=raw, output_dir=folder,
                cache_modes=('resident',), repeats=2, measure_first_query=False)
            row = result.summary[result.summary.method == 'accept'].iloc[0]
            self.assertEqual(row.accepted_count, 1)
            self.assertEqual(row.violation_count, 1)
            self.assertFalse(row.quality_target_met)
            self.assertEqual(len(result.quality_cases), 2)

    def test_zero_truth_policy_and_separate_small_count_diagnostics(self):
        raw = ScanCounter([np.array([[0., 0.], [1., 1.]])])
        with tempfile.TemporaryDirectory() as folder:
            result = run_benchmark(queries=[Query('zero', (4, 5, 4, 5), 0)],
                calibration_queries=[Query('cal', (0, .5, 0, .5), 1)],
                arms=[Arm('accept', lambda q, H, a: 0.)], histogram_loader=lambda: None,
                estimate_fn=lambda q, h: 1., raw_counter=raw, output_dir=folder,
                cache_modes=('resident',), repeats=2, measure_first_query=False)
            accepted = result.quality_cases[result.quality_cases.method == 'accept'].iloc[0]
            self.assertTrue(math.isinf(accepted.relative_error))
            exact = result.quality_cases[result.quality_cases.method == 'always_exact'].iloc[0]
            self.assertEqual(exact.relative_error, 0)
            diagnostics = pd.read_csv(Path(folder)/'quality_by_truth.csv')
            selected = diagnostics[(diagnostics.method == 'accept') & (diagnostics.truth_cohort == 'truth_zero')].iloc[0]
            self.assertEqual(selected.query_count, 1)
            self.assertEqual(selected.violation_count, 1)


if __name__ == '__main__':
    unittest.main()
