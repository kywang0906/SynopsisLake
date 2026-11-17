"""Compiled execution parity; synthetic inputs establish mechanics only.

These tests exercise the frozen distribution policy, not dataset quality or
speed. The compiled backend receives metadata, never raw points or labels.
"""
from pathlib import Path
import hashlib
import json
import sysconfig
import tempfile
import unittest
from unittest import mock

import numpy as np
import scipy

from ml import bounded_sidecar as bounded
from ml import hierarchical_shape as hierarchy
from ml import local_shape
from ml import sidecar_workflow as workflow
from tests.test_shape_runtime import fixtures, queries


class CompiledKernelParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from ml import shape_kernel
        cls.kernel = shape_kernel
        cls.hists, cls.encoded, cls.payload = fixtures()
        cls.rectangles = queries(cls.hists)
        cls.canonical = hierarchy.load_runtime(cls.hists, cls.encoded, cls.payload)
        cls.compiled = shape_kernel.CompiledRuntime(cls.hists, cls.encoded, cls.payload)
        cls.native = shape_kernel.CompiledRuntime(cls.hists, cls.encoded)
        cls.components = [bounded.components_from_histogram_moments(h, e)
                          for h, e in zip(cls.hists, cls.encoded)]

    def test_native_matches_frozen_math_for_random_closed_edges_and_diracs(self):
        expected = sum((bounded.batch_reference(c, self.rectangles) for c in self.components))
        actual = self.native.batch_reference(self.rectangles)
        np.testing.assert_allclose(actual, expected, rtol=5e-13, atol=1e-10)
        for rect, value in zip(self.rectangles[::31], expected[::31]):
            self.assertAlmostEqual(self.native.reference_count(rect), value, delta=1e-10)

    def test_hierarchy_and_uniform_match_frozen_child_model(self):
        for mode in (None, "hierarchical_r08", "empirical_uniform"):
            with self.subTest(mode=mode):
                expected = self.canonical.batch_reference(self.rectangles, mode=mode)
                actual = self.compiled.batch_reference(self.rectangles, mode=mode)
                np.testing.assert_allclose(actual, expected, rtol=5e-13, atol=1e-10)
        for rect in self.rectangles[::29]:
            self.assertAlmostEqual(self.compiled.reference_count(rect),
                                   self.canonical.reference_count(rect), delta=1e-10)

    def test_required_feature_arrays_keep_frozen_contract(self):
        selected = self.rectangles[::7]
        expected = self.canonical.query_features(selected, histogram=self.hists[0])
        actual = self.compiled.query_features(selected, histogram=self.hists[0])
        keys = ("H", "reference_base", "reference_hybrid", "reference_uniform", "full_mass", "boundary_mass",
                "corrected_boundary_mass", "corrected_boundary_estimate_base", "corrected_boundary_estimate_hybrid",
                "corrected_boundary_fraction", "correction_delta", "correction_abs_delta")
        for key in keys:
            with self.subTest(feature=key):
                np.testing.assert_allclose(actual[key], expected[key], rtol=5e-13, atol=1e-10)

    def test_nearby_gate_cutoffs_preserve_frozen_decisions(self):
        rects = self.rectangles[:100]
        old, new = self.canonical.batch_reference(rects), self.compiled.batch_reference(rects)
        answers = np.arange(len(rects), dtype=float)+5
        old_score = abs(answers-old)/np.maximum(old, 1.)
        new_score = abs(answers-new)/np.maximum(new, 1.)
        np.testing.assert_allclose(new_score, old_score, rtol=1e-10, atol=1e-10)
        for cutoff in (.01, .25, 1., 5.):
            separated = abs(old_score-cutoff) > 1e-9
            np.testing.assert_array_equal(new_score[separated] <= cutoff, old_score[separated] <= cutoff)
        for direction in (-1, 1):
            threshold = old_score+direction*1e-9
            np.testing.assert_array_equal(new_score <= threshold, old_score <= threshold)

    def test_exact_cutoff_tie_uses_canonical_numerical_guard(self):
        runtime = self.kernel.CompiledRuntime(self.hists, self.encoded, self.payload)
        rect = self.rectangles[0]
        expected = self.canonical.reference_count(rect)
        answer = expected*1.25+2.
        cutoff = abs(answer-expected)/max(expected, 1.)
        before = runtime.derived_ram_bytes
        with mock.patch.object(runtime, "canonical_reference_count", wraps=runtime.canonical_reference_count) as fallback:
            reference, score = runtime.reference_and_score(rect, answer, cutoff=cutoff)
        fallback.assert_called_once()
        self.assertEqual(reference, expected)
        self.assertEqual(score, cutoff)
        self.assertTrue(score <= cutoff)
        self.assertGreater(runtime.derived_ram_bytes, before)
        with mock.patch.object(runtime, "canonical_reference_count", side_effect=AssertionError("Unneeded fallback")):
            reference, score = runtime.reference_and_score(rect, answer, cutoff=cutoff+1.)
        self.assertAlmostEqual(reference, expected, delta=1e-10)
        self.assertAlmostEqual(score, cutoff, delta=1e-10)

    def test_compiler_cache_binds_source_abi_scipy_and_actual_binary(self):
        setup = self.kernel.kernel_setup()
        identity = setup["identity"]
        required = {"source_sha256", "wrapper_sha256", "compiler", "compiler_version", "flags",
                    "python_abi", "python_implementation", "scipy_version", "platform", "pointer_bytes"}
        self.assertTrue(required <= set(identity))
        self.assertIn("-fno-fast-math", identity["flags"])
        self.assertIn("-ffp-contract=off", identity["flags"])
        self.assertNotIn("-ffast-math", identity["flags"])
        self.assertEqual(identity["python_abi"], sysconfig.get_config_var("SOABI"))
        self.assertEqual(identity["scipy_version"], scipy.__version__)
        self.assertTrue(Path(identity["compiler"]).is_file())
        wrapper = Path(self.kernel.__file__)
        self.assertEqual(identity["source_sha256"], hashlib.sha256(wrapper.with_suffix(".cc").read_bytes()).hexdigest())
        self.assertEqual(identity["wrapper_sha256"], hashlib.sha256(wrapper.read_bytes()).hexdigest())
        self.assertEqual(setup["cache_key"], hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest())
        cache = Path(setup["cache_dir"])
        self.assertEqual(setup["library_sha256"], hashlib.sha256((cache/"shape_kernel.so").read_bytes()).hexdigest())
        saved = json.loads((cache/"build.json").read_text())
        self.assertEqual(saved["identity"], identity)
        self.assertEqual(saved["library_sha256"], setup["library_sha256"])
        self.assertGreaterEqual(setup["compilation_seconds"], 0.)
        self.assertGreaterEqual(setup["library_load_seconds"], 0.)

    def test_unsupported_capsule_signature_and_missing_compiler_fail_explicitly(self):
        with self.assertRaisesRegex(RuntimeError, "Unsupported SciPy C API signature"):
            self.kernel._capsule("ndtri", b"wrong")
        with self.assertRaisesRegex(RuntimeError, "unavailable"):
            self.kernel._capsule("missing_capsule", b"wrong")
        with mock.patch.object(self.kernel, "_STATE", None), \
             mock.patch.object(self.kernel.shutil, "which", return_value=None), \
             self.assertRaisesRegex(RuntimeError, "requires g\\+\\+"):
            self.kernel.kernel_setup()

    def test_clamps_high_correlation_and_child_endpoint_atoms(self):
        hist = (1., np.array([0., 1.]), np.array([0., 1.]), np.array([[200]], np.uint64))
        encoded = bounded.encode_moments([[0., 1., .02, .03, .1]])
        child = bounded.encode_moments([[.1, .9, .005, .005, .25], [.9, .1, .005, .005, -.25],
                                        [0., 0., 0., 0., 0.], [1., 1., 0., 0., 0.]]).reshape(1, 4, 5)
        quarters = np.array([[20, 40, 60, 80]], np.uint64)
        payload = hierarchy.build_selection([hist], [encoded], [quarters], [child], 200,
                                            priority=np.ones(1))
        frozen = hierarchy.load_runtime([hist], [encoded], payload)
        compiled = self.kernel.CompiledRuntime([hist], [encoded], payload)
        rects = np.r_[local_shape.QUADRANTS,
                      [[0., .25, 0., .25], [.5, .75, 0., .25], [.9, 1., .9, 1.],
                       [0., 1., 0., 1.], [.5, .5, .5, .5], [0., 0., 0., 1.]]]
        np.testing.assert_allclose(compiled.batch_reference(rects), frozen.batch_reference(rects),
                                   rtol=5e-13, atol=1e-10)
        np.testing.assert_array_equal(compiled.batch_reference(local_shape.QUADRANTS), quarters[0])
        self.assertEqual(compiled.reference_count([0., 1., 0., 1.]), 200.)

    def test_no_diagnostic_quad_cdf_or_raw_reader_is_needed(self):
        with mock.patch.object(local_shape, "_quadrant_probabilities", side_effect=AssertionError("Diagnostic CDF")), \
             mock.patch.object(workflow, "iter_raw", side_effect=AssertionError("Runtime read raw points")):
            runtime = self.kernel.CompiledRuntime(self.hists, self.encoded, self.payload)
            first = runtime.reference_count([-.831, .287, 4.271, 5.133])
            second = runtime.query_features([[.414, .528, 5.111, 5.167]])["reference_hybrid"][0]
        self.assertTrue(np.isfinite([first, second]).all())
        self.assertEqual(runtime.backend, "compiled_scipy_capi")

    def test_empty_sources_empty_batch_and_invalid_rectangles(self):
        hist = (1., np.array([0., 1., 2.]), np.array([0., 1.]), np.zeros((2, 1), np.uint64))
        encoded = np.zeros((2, 5), np.uint8)
        quarters, children = np.zeros((2, 4), np.uint64), np.zeros((2, 4, 5), np.uint8)
        payload = hierarchy.build_selection([hist], [encoded], [quarters], [children], 200)
        for packet in (None, payload):
            runtime = self.kernel.CompiledRuntime([hist], [encoded], packet)
            np.testing.assert_array_equal(runtime.batch_reference([[0., 2., 0., 1.], [0., 0., 0., 1.]]), [0., 0.])
            self.assertEqual(runtime.batch_reference([]).shape, (0,))
            self.assertEqual(runtime.total_mass, 0)
        for rect in ([1., 0., 0., 1.], [0., 1., 1., 0.], [0., np.nan, 0., 1.], [0., 1., 0.]):
            with self.subTest(rect=rect), self.assertRaises(ValueError):
                self.compiled.reference_count(rect)
        with self.assertRaises(ValueError):
            self.compiled.reference_count([0., 1., 0., 1.], mode="unknown")

    def test_loader_validates_codec_and_preserves_byte_accounting(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/"snapshot"
            hierarchy.save_snapshot(root, self.hists, self.encoded, self.payload)
            with mock.patch.object(workflow, "iter_raw", side_effect=AssertionError("Loader read raw points")):
                runtime = self.kernel.load_compiled(root, root/"hierarchical.bin")
                native = self.kernel.load_compiled(root)
            self.assertEqual(runtime.auxiliary_bytes, self.canonical.auxiliary_bytes)
            self.assertEqual(runtime.native_bytes, self.canonical.native_bytes)
            self.assertEqual(native.auxiliary_bytes, self.canonical.native_bytes)
            self.assertGreater(runtime.ram_bytes, 0)
            self.assertGreater(runtime.derived_ram_bytes, 0)
            np.testing.assert_allclose(runtime.batch_reference(self.rectangles[:20]),
                                       self.canonical.batch_reference(self.rectangles[:20]), rtol=5e-13, atol=1e-10)
            bad = bytearray(self.payload)
            bad[-1] ^= 1
            (root/"hierarchical.bin").write_bytes(bad)
            with self.assertRaises(ValueError):
                self.kernel.load_compiled(root, root/"hierarchical.bin")
            (root/"hierarchical.bin").write_bytes(self.payload)
            moment_path = root/"src_0-moments.bin"
            changed = bytearray(moment_path.read_bytes())
            changed[64] ^= 1
            moment_path.write_bytes(changed)
            with self.assertRaises(ValueError):
                self.kernel.load_compiled(root, root/"hierarchical.bin")


if __name__ == "__main__":
    unittest.main()
