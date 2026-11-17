"""Construction, codec and inference guards for source-quadrant density raking.

These synthetic checks establish mechanics, not accuracy on the authors data.
"""
from pathlib import Path
import struct
import tempfile
import unittest
from unittest import mock
import zlib

import numpy as np

from ml import bounded_sidecar as bounded
from ml import local_shape as shape
from ml import sidecar_workflow as workflow
from ml._distribution_math import boundary_probability


def fixture(counts=((100,),), moments=None, bx=None, by=None):
    counts = np.asarray(counts, np.uint64)
    bx = np.arange(counts.shape[0]+1, dtype=float) if bx is None else np.asarray(bx, float)
    by = np.arange(counts.shape[1]+1, dtype=float) if by is None else np.asarray(by, float)
    hist = (1., bx, by, counts)
    if moments is None:
        moments = np.tile([.5, .5, .08, .08, .025], (counts.size, 1))
    return hist, bounded.encode_moments(np.asarray(moments).reshape(-1, 5))


def repair_crc(payload):
    value = bytearray(payload)
    value[shape.HEADER_BYTES-4:shape.HEADER_BYTES] = struct.pack(
        "<I", zlib.crc32(value[:shape.HEADER_BYTES-4]+value[shape.HEADER_BYTES:]))
    return bytes(value)


class SourceConstruction(unittest.TestCase):
    def test_parent_arithmetic_and_midpoint_ownership(self):
        hist, _ = fixture(((4, 1), (1, 2)), bx=[0, .9, 1.8], by=[0, .7, 1.4])
        points = np.array([[.1, .1], [.45, .35], [.45, .1], [.1, .35],
                           [.2, .9], [1.2, .1], [1.35, 1.05], [1.7, 1.3]])
        quarter = shape.quadrant_counts_from_points(points, hist)
        parent, _, _ = workflow._sufficient(points, hist)
        np.testing.assert_array_equal(quarter.sum(axis=1), parent)
        np.testing.assert_array_equal(quarter[0], [1, 1, 1, 1])
        np.testing.assert_array_equal(quarter, shape.quadrant_counts_from_points(points[:3], hist)
                                      +shape.quadrant_counts_from_points(points[3:], hist))

    def test_split_plane_counts_are_partition_counts(self):
        hist, encoded = fixture(((5,),))
        points = np.array([[.1, .1], [.1, .6], [.6, .1], [.6, .6], [.5, .5]])
        counts = shape.quadrant_counts_from_points(points, hist)
        np.testing.assert_array_equal(counts, [[1, 1, 1, 2]])
        runtime = shape.load_runtime([hist], [encoded], shape.build_selection([hist], [encoded], [counts], 200))
        self.assertEqual(runtime.reference_count(shape.QUADRANTS[0]), 1.)
        # The closed raw query includes the midpoint atom assigned high/high.
        self.assertEqual(((points[:, 0] <= .5) & (points[:, 1] <= .5)).sum(), 2)
        self.assertIn("split_plane_atoms", shape.protocol())

    def test_native_byte_means_cannot_be_midpoint_diracs(self):
        encoded = np.zeros((256, 5), np.uint8)
        encoded[:, 0] = np.arange(256)
        encoded[:, 1] = np.arange(256)
        self.assertFalse((bounded.decode_moments(encoded)[:, :2] == .5).any())
        np.testing.assert_array_equal(shape._quadrant_probabilities(encoded).sum(axis=1), np.ones(256))


class SelectionAndCodec(unittest.TestCase):
    def setUp(self):
        self.hist, self.encoded = fixture(((100, 100), (100, 0)))
        self.counts = np.array([[100, 0, 0, 0], [40, 20, 20, 20], [0, 0, 0, 100], [0, 0, 0, 0]])
        self.fixed = shape.HEADER_BYTES+shape.SOURCE_BYTES+1

    def build(self, budget=1000, mode="conditional_r08"):
        return shape.build_selection([self.hist], [self.encoded], [self.counts], budget, mode=mode)

    def test_actual_packet_budget_and_selection_prefix(self):
        previous = set()
        for k in range(4):
            budget = self.fixed+shape.SELECTED_CELL_BYTES*k+11
            payload = self.build(budget)
            selection = shape.decode_selection([self.hist], [self.encoded], payload)
            self.assertEqual(len(payload), self.fixed+shape.SELECTED_CELL_BYTES*k)
            self.assertLessEqual(len(payload), budget)
            self.assertEqual(len(selection.selected_ids), k)
            self.assertTrue(previous <= set(selection.selected_ids))
            previous = set(selection.selected_ids)
        self.assertEqual(self.build(), self.build())
        with self.assertRaises(ValueError):
            self.build(self.fixed-1)

    def test_one_fixed_source_only_priority_and_shared_control_selection(self):
        payload = self.build(self.fixed+12)
        selected = shape.decode_selection([self.hist], [self.encoded], payload).selected_ids
        p = shape._quadrant_probabilities(self.encoded[:3])
        scores = .5*np.abs(self.counts[:3]-100*p).sum(axis=1)
        self.assertEqual(selected.tolist(), [int(np.argmax(scores))])
        control = shape.decode_selection([self.hist], [self.encoded], self.build(self.fixed+12, "empirical_uniform"))
        np.testing.assert_array_equal(control.selected_ids, selected)
        self.assertEqual(control.payload_bytes, len(payload))
        audit = shape.selection_audit([self.hist], [self.encoded], [self.counts], payload)
        np.testing.assert_array_equal(np.flatnonzero(audit["selected"]), selected)
        self.assertEqual(audit["source_rank"][selected[0]], 0)
        np.testing.assert_allclose(audit["priority"][:3], scores)

    def test_all_single_byte_corruption_is_detected(self):
        payload = self.build()
        for i in range(len(payload)):
            bad = bytearray(payload)
            bad[i] ^= 1
            with self.subTest(byte=i), self.assertRaises(ValueError):
                shape.decode_selection([self.hist], [self.encoded], bad)
        for payload2 in (payload[:-1], payload+b"x", b""):
            with self.assertRaises(ValueError):
                shape.decode_selection([self.hist], [self.encoded], payload2)

    def test_rechecksummed_invalid_counts_padding_and_source_table_rejected(self):
        payload = self.build()
        offset = shape.HEADER_BYTES+shape.SOURCE_BYTES+1
        bad = bytearray(payload)
        struct.pack_into("<I", bad, offset, 101)
        with self.assertRaisesRegex(ValueError, "exceed"):
            shape.decode_selection([self.hist], [self.encoded], repair_crc(bad))
        bad = bytearray(payload)
        bad[shape.HEADER_BYTES+shape.SOURCE_BYTES] |= 128
        with self.assertRaisesRegex(ValueError, "bitmap"):
            shape.decode_selection([self.hist], [self.encoded], repair_crc(bad))
        bad = bytearray(payload)
        struct.pack_into("<I", bad, shape.HEADER_BYTES, 5)
        with self.assertRaisesRegex(ValueError, "source table"):
            shape.decode_selection([self.hist], [self.encoded], repair_crc(bad))

    def test_native_identity_and_exact_uint32_counts(self):
        payload = self.build()
        changed = self.encoded.copy()
        changed[0, 0] += 1
        with self.assertRaisesRegex(ValueError, "identity"):
            shape.decode_selection([self.hist], [changed], payload)
        hist, encoded = fixture(((2**32,),))
        with self.assertRaises(OverflowError):
            shape.build_selection([hist], [encoded], [np.array([[2**32, 0, 0, 0]], np.uint64)], 200)
        with self.assertRaisesRegex(ValueError, "sum"):
            shape.build_selection([self.hist], [self.encoded], [self.counts+1], 200)
        selection = shape.decode_selection([self.hist], [self.encoded], payload)
        with self.assertRaises(ValueError):
            selection.quadrant_counts.setflags(write=True)


class ConditionalInference(unittest.TestCase):
    def setUp(self):
        self.hist, self.encoded = fixture(moments=[[.3, .65, .035, .025, .01]])
        self.counts = np.array([[10, 20, 30, 40]], np.uint64)
        self.payload = shape.build_selection([self.hist], [self.encoded], [self.counts], 200)
        self.runtime = shape.load_runtime([self.hist], [self.encoded], self.payload)

    def test_exact_quarter_quota_and_root_mass_conservation(self):
        np.testing.assert_array_equal(self.runtime.batch_reference(shape.QUADRANTS), self.counts[0])
        np.testing.assert_array_equal(self.runtime.batch_reference(shape.QUADRANTS, mode="empirical_uniform"), self.counts[0])
        self.assertEqual(self.runtime.reference_count([0, 1, 0, 1]), 100.)
        self.assertEqual(self.runtime.reference_count([-1, 2, -1, 2]), 100.)
        self.assertEqual(self.runtime.reference_count([2, 3, 2, 3]), 0.)

    def test_local_intersection_retains_conditional_native_shape(self):
        rect = np.array([.1, .3, .15, .4], np.float32).astype(float)
        moments = bounded.decode_moments(self.encoded).astype(float)
        p = boundary_probability(rect[None, :], moments, "beta_gaussian_copula")[0]
        q = boundary_probability(shape.QUADRANTS[:1], moments, "beta_gaussian_copula")[0]
        self.assertAlmostEqual(self.runtime.reference_count(rect), 10*p/q, places=12)
        self.assertAlmostEqual(self.runtime.reference_count(rect, mode="empirical_uniform"),
                               10*4*(rect[1]-rect[0])*(rect[3]-rect[2]), places=12)

    def test_uniform_live_path_skips_selected_cell_cdf_work(self):
        rect = [.1, .3, .15, .4]
        expected = self.runtime.query_features([rect])["reference_uniform"][0]
        with mock.patch.object(shape, "boundary_probability", side_effect=AssertionError("Uniform selected cell called CDF")):
            self.assertEqual(self.runtime.reference_count(rect, mode="empirical_uniform"), expected)

    def test_zero_cdf_positive_quarter_uses_fixed_uniform_fallback(self):
        hist, encoded = fixture(moments=[[.9, .9, 0, 0, 0]])
        counts = np.array([[40, 10, 20, 30]])
        runtime = shape.load_runtime([hist], [encoded], shape.build_selection([hist], [encoded], [counts], 200))
        self.assertEqual(shape.DENOMINATOR_TOLERANCE, 1e-12)
        self.assertEqual(runtime.reference_count([0, .5, 0, .5]), 40.)
        self.assertEqual(runtime.reference_count([0, .25, 0, .5]), 20.)
        features = runtime.query_features([[0, .25, 0, .5]])
        self.assertTrue(np.isfinite(np.concatenate(list(features.values()))).all())
        self.assertGreater(features["zero_cdf_boundary_mass"][0], 0.)

    def test_base_reuses_native_numerical_contract_without_selected_cells(self):
        hist2, encoded2 = fixture(((0, 20), (40, 60)), bx=[-1, .1, 1.2], by=[2, 3, 4])
        hists, encs = [self.hist, hist2], [self.encoded, encoded2]
        counts = [self.counts, np.array([[0, 0, 0, 0], [1, 2, 3, 14], [10, 10, 10, 10], [30, 10, 10, 10]])]
        fixed = shape.HEADER_BYTES+2*shape.SOURCE_BYTES+1
        runtime = shape.load_runtime(hists, encs, shape.build_selection(hists, encs, counts, fixed))
        queries = np.array([[0, .8, 0, .8], [-2, .3, 2.2, 3.8], [-10, 10, -10, 10], [0, 0, 0, 1]])
        a = bounded.components_from_histogram_moments(self.hist, self.encoded)
        b = bounded.components_from_histogram_moments(hist2, encoded2)
        expected = bounded.batch_reference(a, queries)+bounded.batch_reference(b, queries)
        np.testing.assert_allclose(runtime.batch_reference(queries), expected, rtol=1e-14)
        np.testing.assert_array_equal(runtime.query_features(queries)["reference_base"], runtime.batch_reference(queries))
        self.assertEqual(runtime.total_mass, 220)

    def test_features_need_only_synopses_and_geometry(self):
        queries = np.array([[.1, .7, .2, .8], [0, 1, 0, 1]])
        with mock.patch.object(workflow, "iter_raw", side_effect=AssertionError("Inference read raw points")):
            features = self.runtime.query_features(queries, histogram=self.hist)
        self.assertEqual(set(features["H"].shape), {2})
        np.testing.assert_array_equal(features["correction_delta"], features["reference_hybrid"]-features["reference_base"])
        self.assertEqual(features["corrected_boundary_mass"][0], 100.)
        self.assertEqual(features["corrected_boundary_mass"][1], 0.)
        self.assertEqual(features["full_mass"][1], 100.)

    def test_persistent_native_and_correction_account_every_byte(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)/"snapshot"
            nbytes = shape.save_snapshot(directory, [self.hist], [self.encoded], self.payload,
                                         original_histogram=self.hist)
            self.assertEqual(nbytes, sum(p.stat().st_size for p in directory.iterdir()))
            with mock.patch.object(workflow, "iter_raw", side_effect=AssertionError("Inference read raw points")):
                runtime = shape.load_snapshot(directory)
                self.assertEqual(runtime.nbytes, nbytes)
                self.assertEqual(runtime.auxiliary_bytes, self.runtime.auxiliary_bytes)
                self.assertEqual(runtime.reference_count(shape.QUADRANTS[0]), 10.)
                self.assertIn("H", runtime.query_features([[0, .5, 0, .5]]))
            (directory/"uncharged.bin").write_bytes(b"x")
            with self.assertRaisesRegex(ValueError, "uncharged"):
                shape.load_snapshot(directory)


if __name__ == "__main__":
    unittest.main()
