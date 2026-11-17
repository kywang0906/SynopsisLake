"""Execution adapter guards; tiny fixtures make no real-data speed claim."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd

from ml import shape_execution as execution
from ml.sidecar_e2e import Arm, ScoreResult


class ArmExecutionTests(unittest.TestCase):
    def test_compiled_backend_prepares_once_and_calls_guard_with_frozen_gate(self):
        runtime = mock.Mock(auxiliary_bytes=10, ram_bytes=50, derived_ram_bytes=25)
        runtime.reference_and_score.return_value = (2., .25)
        backend = mock.Mock()
        backend.load_compiled.return_value = runtime
        old_score = mock.Mock(side_effect=AssertionError('Compiled analytic guard owns its canonical rounding'))
        old = Arm('uniform_custom', old_score, metadata={'auxiliary_bytes': 10})
        with (mock.patch.object(execution, '_backend_module', return_value=backend),
              mock.patch.object(execution, '_backend_setup', return_value={'cache_hit': True}) as setup):
            arms = execution.index_arms([old], native_dir='native', packet_path='packet.bin',
                method_kinds={'uniform_custom': 'uniform'}, cutoffs={'uniform_custom': {'cutoff': .25}},
                backend='compiled')
            setup.assert_called_once_with('compiled')
            backend.load_compiled.assert_not_called()  # Still outside artifact/query loads.
            artifact = arms[0].artifact_loader()
            value = arms[0].score_fn([0, .5, 0, .5], 2.5, artifact, None)
            backend.load_compiled.assert_called_once_with(Path('native'), 'packet.bin')
            runtime.reference_and_score.assert_called_once()
            self.assertEqual(runtime.reference_and_score.call_args.kwargs, {'cutoff': .25, 'mode': 'empirical_uniform'})
            self.assertEqual(value, ScoreResult(.25, 2.))
            self.assertEqual(arms[0].metadata['auxiliary_bytes'], 10)
            self.assertEqual(arms[0].metadata['execution_backend'], 'compiled')

    def test_compiler_failure_is_explicit_without_silent_runtime_fallback(self):
        from ml import shape_kernel
        with mock.patch.object(shape_kernel, 'kernel_setup', side_effect=RuntimeError('g++ unavailable')):
            with self.assertRaisesRegex(RuntimeError, 'Explicit compiled backend could not initialize'):
                execution.index_arms([], native_dir='native', backend='compiled')
        with self.assertRaisesRegex(ValueError, 'explicitly indexed or compiled'):
            execution.index_arms([], native_dir='native', backend='auto')

    def test_custom_roles_use_current_H_original_model_and_leave_fine_untouched(self):
        hist = (1., np.array([0., 1.]), np.array([0., 1.]), np.array([[2.]]))
        runtime = mock.Mock(auxiliary_bytes=10, ram_bytes=50, derived_ram_bytes=25)
        runtime.reference_count.return_value = 2.
        runtime.query_features.return_value = dict(reference_base=np.array([2.]), reference_hybrid=np.array([3.]),
            full_mass=np.array([1.]), boundary_mass=np.array([1.]), corrected_boundary_fraction=np.array([.5]))
        predictor = mock.Mock(model_bytes=7)
        predictor.predict_row.return_value = .125
        fine = Arm('fine_original', lambda q, H, a: ScoreResult(.2, 3.))
        old = [Arm('custom_native', lambda *args: 99., metadata={'auxiliary_bytes': 10}),
               Arm('custom_ml', lambda *args: 99., metadata={'auxiliary_bytes': 17}), fine]
        with (mock.patch.object(execution.shape_runtime, 'load_indexed', return_value=runtime) as load,
              mock.patch.object(execution.models, 'load_model', return_value=predictor) as model_load):
            arms = execution.index_arms(old, native_dir='native', packet_paths={'custom_ml': 'packet.bin'},
                model_paths={'custom_ml': 'schema.json'},
                method_kinds={'custom_native': 'native', 'custom_ml': 'ml_hybrid'})
            self.assertIs(arms[2], fine)
            artifact = arms[1].artifact_loader()
            actual = arms[1].score_fn([0, .5, 0, .5], 17., artifact, hist)
            self.assertEqual(actual.score, .125)
            model_load.assert_called_once_with(Path('schema.json'))
            self.assertEqual(load.call_args.args, (Path('native'), 'packet.bin'))
            features = predictor.predict_row.call_args.args[0]
            self.assertEqual(features.log_H, np.log1p(17.))
            self.assertEqual(tuple(features.index), execution.models.HYBRID_FEATURES)
            self.assertEqual(arms[1].metadata['auxiliary_bytes'], 17)

    def test_near_gate_calls_canonical_live_without_changing_cutoff(self):
        runtime = mock.Mock(auxiliary_bytes=10, ram_bytes=50, derived_ram_bytes=25)
        runtime.reference_count.return_value = 1.
        canonical_score, canonical_load = mock.Mock(return_value=ScoreResult(.25000000001, 1.)), mock.Mock(return_value='canonical')
        old = Arm('custom', canonical_score, artifact_loader=canonical_load, metadata={'auxiliary_bytes': 10})
        with mock.patch.object(execution.shape_runtime, 'load_indexed', return_value=runtime):
            arm = execution.index_arms([old], native_dir='native', method_kinds={'custom': 'native'},
                cutoffs={'custom': .25})[0]
            artifact = arm.artifact_loader()
            result = arm.score_fn([0, .5, 0, .5], 1.25, artifact, None)
            self.assertEqual(result.score, .25000000001)
            self.assertGreater(result.score, .25)
            canonical_score.assert_called_once()
            self.assertEqual(canonical_score.call_args.args[2], 'canonical')
            arm.score_fn([0, .5, 0, .5], 1.25, artifact, None)
            canonical_load.assert_called_once()


class FrozenParityTests(unittest.TestCase):
    def setUp(self):
        self.saved = pd.DataFrame([dict(seed=0, method='native', qId=8, H=10., score=.25, reference_count=8.),
                                   dict(seed=0, method='ml', qId=8, H=10., score=.1, reference_count=np.nan)])
        self.gates = pd.DataFrame([dict(seed=0, method='native', cutoff=.25), dict(seed=0, method='ml', cutoff=.2)])
        self.live = pd.DataFrame([dict(seed=0, method='native', qId=8, H=10., score=.25, reference=8., accepted=True),
                                  dict(seed=0, method='ml', qId=8, H=10., score=.1, reference=None, accepted=True)])

    def test_all_values_and_missing_ml_reference_match_without_labels(self):
        checked = execution.verify_live_parity(self.live, self.saved, self.gates)
        self.assertTrue(checked.all_matches.all())

    def test_numeric_tolerance_never_relaxes_exact_gate_decision(self):
        self.live.loc[0, 'score'] += 1e-12
        self.live.loc[0, 'accepted'] = False
        checked = execution.verify_live_parity(self.live, self.saved, self.gates)
        self.assertTrue(checked.score_matches.all())
        self.assertFalse(checked.decision_matches.all())

    def test_changed_reference_or_missing_frozen_key_fails(self):
        self.live.loc[0, 'reference'] = 9.
        checked = execution.verify_live_parity(self.live, self.saved, self.gates)
        self.assertFalse(checked.reference_matches.all())
        self.live.loc[0, 'qId'] = 9
        with self.assertRaisesRegex(ValueError, 'absent from frozen'):
            execution.verify_live_parity(self.live, self.saved, self.gates)

    def test_labels_must_match_every_locked_coordinate_and_integral_raw_count(self):
        geometry = pd.DataFrame([dict(qId=8, x0=0., x1=1., y0=0., y1=1.)])
        labels = geometry.assign(truth=3)
        execution._geometry_labels(geometry, labels)
        labels.loc[0, 'x1'] = np.nextafter(1., 2.)
        with self.assertRaisesRegex(ValueError, 'coordinates differ'):
            execution._geometry_labels(geometry, labels)


if __name__ == '__main__':
    unittest.main()
