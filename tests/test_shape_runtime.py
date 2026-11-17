"""Indexed execution parity; these fixtures are not quality/speed evidence."""
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from ml import bounded_sidecar as bounded
from ml import hierarchical_shape as hierarchy
from ml import local_shape
from ml import sidecar_workflow as workflow


def fixtures():
    rng = np.random.default_rng(826)
    hists, encoded, children, quarters = [], [], [], []
    grids = [([-2., -1., .1, 1.7], [4., 4.2, 4.7, 6.5]),
             ([-1.5, -.7, .4, 2.], [3.5, 4.8, 5.3]),
             ([.3, .6, 1.2], [4.1, 4.4, 5.6, 7.])]
    for sid, (bx, by) in enumerate(grids):
        counts = rng.integers(1, 90, size=(len(bx)-1, len(by)-1), dtype=np.uint64)
        counts.ravel()[sid] = 0
        hist = (1., np.asarray(bx), np.asarray(by), counts)
        moments = np.column_stack((rng.uniform(.1, .9, (counts.size, 2)),
                                   rng.uniform(.005, .08, (counts.size, 2)),
                                   rng.uniform(-.005, .005, counts.size)))
        moments[0] = [0., 1., 0., 0., 0.]
        child_moments = np.column_stack((rng.uniform(.05, .95, (4*counts.size, 2)),
                                         rng.uniform(.005, .08, (4*counts.size, 2)),
                                         rng.uniform(-.005, .005, 4*counts.size)))
        child_moments[::5] = [0., 0., 0., 0., 0.]
        q = np.array([rng.multinomial(int(n), rng.dirichlet(np.ones(4))) for n in counts.ravel()], np.uint64)
        hists.append(hist)
        encoded.append(bounded.encode_moments(moments))
        children.append(bounded.encode_moments(child_moments).reshape(-1, 4, 5))
        quarters.append(q)
    n = sum(h[3].size for h in hists)
    budget = hierarchy.HEADER_BYTES+len(hists)*hierarchy.SOURCE_BYTES+(n+7)//8+32*9
    payload = hierarchy.build_selection(hists, encoded, quarters, children, budget,
                                        priority=np.arange(n, dtype=float)+1)
    return hists, encoded, payload


def queries(hists):
    rng = np.random.default_rng(741)
    first = rng.uniform([-3., 3.], [2.5, 7.5], (180, 2))
    second = first+rng.uniform(.00001, 1.2, (180, 2))
    result = [[a[0], b[0], a[1], b[1]] for a, b in zip(first, second)]
    result += [[-10., 10., -10., 10.], [9., 10., 9., 10.]]
    for _, bx, by, counts in hists:
        result.append([bx[0], bx[-1], by[0], by[-1]])
        for ix in range(counts.shape[0]):
            for iy in range(counts.shape[1]):
                x0, x1, y0, y1 = bx[ix], bx[ix+1], by[iy], by[iy+1]
                xm, ym = (x0+x1)/2, (y0+y1)/2
                result.extend([[x0, x1, y0, y1], [x0, xm, y0, ym],
                               [xm, x1, ym, y1], [x0, x0, y0, y1],
                               [xm, xm, ym, ym], [x0, x1, y1, y1],
                               [np.nextafter(x0, -np.inf), np.nextafter(x0, np.inf), y0, y1],
                               [np.nextafter(x1, -np.inf), np.nextafter(x1, np.inf), y0, y1]])
    return np.asarray(result)


class IndexedRuntimeParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from ml import shape_runtime
        cls.optimized = shape_runtime
        cls.hists, cls.encoded, cls.payload = fixtures()
        cls.rectangles = queries(cls.hists)
        cls.canonical = hierarchy.load_runtime(cls.hists, cls.encoded, cls.payload)
        cls.indexed = shape_runtime.IndexedHierarchicalRuntime(cls.hists, cls.encoded, cls.payload)
        cls.native = shape_runtime.FastNativeRuntime(cls.hists, cls.encoded)
        cls.components = [bounded.components_from_histogram_moments(h, e)
                          for h, e in zip(cls.hists, cls.encoded)]

    def test_native_matches_frozen_reference_for_random_and_closed_edges(self):
        expected = sum((bounded.batch_reference(c, self.rectangles) for c in self.components))
        np.testing.assert_allclose(self.native.batch_reference(self.rectangles), expected, rtol=2e-14, atol=2e-12)
        for rect, value in zip(self.rectangles[::31], expected[::31]):
            self.assertAlmostEqual(self.native.reference_count(rect), value, places=11)

    def test_hierarchy_matches_frozen_reference_for_all_modes(self):
        for mode in (None, "hierarchical_r08", "empirical_uniform"):
            expected = self.canonical.batch_reference(self.rectangles, mode=mode)
            actual = self.indexed.batch_reference(self.rectangles, mode=mode)
            with self.subTest(mode=mode):
                np.testing.assert_allclose(actual, expected, rtol=2e-14, atol=2e-12)
        for rect in self.rectangles[::29]:
            self.assertAlmostEqual(self.indexed.reference_count(rect), self.canonical.reference_count(rect), places=11)

    def test_required_feature_arrays_match_frozen_canonical(self):
        selected = self.rectangles[::4]
        expected = self.canonical.query_features(selected, histogram=self.hists[0])
        actual = self.indexed.query_features(selected, histogram=self.hists[0])
        keys = ("H", "reference_base", "reference_hybrid", "reference_uniform", "full_mass", "boundary_mass",
                "corrected_boundary_mass", "corrected_boundary_estimate_base", "corrected_boundary_estimate_hybrid",
                "corrected_boundary_fraction", "correction_delta", "correction_abs_delta")
        for key in keys:
            with self.subTest(feature=key):
                np.testing.assert_allclose(actual[key], expected[key], rtol=2e-14, atol=2e-12)
        self.assertNotIn("source_disagreement_mass", actual)

    def test_gate_decisions_equal_with_separated_and_near_cutoffs(self):
        rects = self.rectangles[:100]
        old = self.canonical.batch_reference(rects)
        new = self.indexed.batch_reference(rects)
        answers = np.arange(len(rects), dtype=float)+5
        old_score = abs(answers-old)/np.maximum(old, 1.)
        new_score = abs(answers-new)/np.maximum(new, 1.)
        np.testing.assert_allclose(new_score, old_score, rtol=5e-13, atol=5e-13)
        for cutoff in (.01, .25, 1., 5.):
            separated = abs(old_score-cutoff) > 1e-10
            np.testing.assert_array_equal(new_score[separated] <= cutoff, old_score[separated] <= cutoff)
        # A cutoff this close cannot materially differ without a model change.
        for direction in (-1, 1):
            threshold = old_score+direction*1e-10
            np.testing.assert_array_equal(new_score <= threshold, old_score <= threshold)

    def test_initialization_does_not_compute_diagnostic_quadrant_cdfs(self):
        with mock.patch.object(local_shape, "_quadrant_probabilities", side_effect=AssertionError("Diagnostic CDF in cold loader")):
            runtime = self.optimized.IndexedHierarchicalRuntime(self.hists, self.encoded, self.payload)
        self.assertAlmostEqual(runtime.reference_count(self.rectangles[10]), self.canonical.reference_count(self.rectangles[10]), places=11)

    def test_inference_has_no_raw_reader_and_accepts_unseen_geometry(self):
        with mock.patch.object(workflow, "iter_raw", side_effect=AssertionError("Runtime read raw points")):
            first = self.indexed.reference_count([-.831, .287, 4.271, 5.133])
            second = self.indexed.query_features([[.414, .528, 5.111, 5.167]])["reference_hybrid"][0]
        self.assertTrue(np.isfinite([first, second]).all())

    def test_empty_native_sources_conserve_zero_mass(self):
        hist = (1., np.array([0., 1., 2.]), np.array([0., 1.]), np.zeros((2, 1), np.uint64))
        encoded = np.zeros((2, 5), np.uint8)
        quarters = np.zeros((2, 4), np.uint64)
        children = np.zeros((2, 4, 5), np.uint8)
        payload = hierarchy.build_selection([hist], [encoded], [quarters], [children], 200)
        rects = [[0., 2., 0., 1.], [-1., 1., -.5, .5], [0., 0., 0., 1.]]
        for runtime in (self.optimized.IndexedHierarchicalRuntime([hist], [encoded], payload),
                        self.optimized.FastNativeRuntime([hist], [encoded])):
            np.testing.assert_array_equal(runtime.batch_reference(rects), np.zeros(3))
            self.assertEqual(runtime.total_mass, 0)

    def test_loader_validates_packet_and_preserves_serialized_cost(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/"snapshot"
            hierarchy.save_snapshot(root, self.hists, self.encoded, self.payload)
            with mock.patch.object(workflow, "iter_raw", side_effect=AssertionError("Loader read raw points")):
                runtime = self.optimized.load_indexed(root, root/"hierarchical.bin")
                native = self.optimized.load_indexed(root)
            self.assertEqual(runtime.auxiliary_bytes, self.canonical.auxiliary_bytes)
            self.assertEqual(runtime.native_bytes, self.canonical.native_bytes)
            self.assertEqual(native.auxiliary_bytes, self.canonical.native_bytes)
            self.assertGreater(runtime.ram_bytes, 0)
            self.assertGreater(native.ram_bytes, 0)
            np.testing.assert_allclose(runtime.batch_reference(self.rectangles[:20]),
                                       self.canonical.batch_reference(self.rectangles[:20]), atol=2e-12, rtol=2e-14)
            bad = bytearray(self.payload)
            bad[-1] ^= 1
            (root/"hierarchical.bin").write_bytes(bad)
            with self.assertRaises(ValueError):
                self.optimized.load_indexed(root, root/"hierarchical.bin")
            (root/"hierarchical.bin").write_bytes(self.payload)
            moment_path = root/"src_0-moments.bin"
            changed = bytearray(moment_path.read_bytes())
            changed[64] ^= 1
            moment_path.write_bytes(changed)
            with self.assertRaises(ValueError):
                self.optimized.load_indexed(root, root/"hierarchical.bin")


if __name__ == "__main__":
    unittest.main()
