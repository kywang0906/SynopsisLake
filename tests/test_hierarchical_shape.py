"""Mechanics of compact within-quarter moments; no dataset accuracy claim."""
from pathlib import Path
import struct
import tempfile
import unittest
from unittest import mock
import zlib

import numpy as np

from ml import bounded_sidecar as bounded
from ml import hierarchical_shape as shape
from ml import local_shape
from ml import sidecar_workflow as workflow
from ml._distribution_math import boundary_probability


def fixture(counts=((100,),), moments=None, bx=None, by=None):
    counts = np.asarray(counts, np.uint64)
    bx = np.arange(counts.shape[0]+1, dtype=float) if bx is None else np.asarray(bx, float)
    by = np.arange(counts.shape[1]+1, dtype=float) if by is None else np.asarray(by, float)
    hist = (1., bx, by, counts)
    if moments is None:
        moments = np.tile([.5, .5, .08, .08, .025], (counts.size, 1))
    encoded = bounded.encode_moments(np.asarray(moments).reshape(-1, 5))
    child = np.tile(np.array([[.2, .2, .015, .015, 0.], [.35, .65, .04, .03, .01],
                              [.65, .35, .03, .04, -.01], [.8, .7, .015, .025, 0.]]),
                    (counts.size, 1))
    child_encoded = bounded.encode_moments(child).reshape(counts.size, 4, 5)
    return hist, encoded, child_encoded


def repair_crc(payload):
    value = bytearray(payload)
    value[shape.HEADER_BYTES-4:shape.HEADER_BYTES] = struct.pack(
        "<I", zlib.crc32(value[:shape.HEADER_BYTES-4]+value[shape.HEADER_BYTES:]))
    return bytes(value)


class ChildConstruction(unittest.TestCase):
    def test_parent_ownership_midpoints_and_additive_sums(self):
        hist, _, _ = fixture(((4, 1), (1, 2)))
        points = np.array([[.1, .1], [.5, .5], [.5, .1], [.1, .5],
                           [.2, 1.1], [1.2, .1], [1.5, 1.5], [1.7, 1.3]])
        counts, sums = shape.accumulate_child_moments(points, hist)
        parent, _, _ = workflow._sufficient(points, hist)
        np.testing.assert_array_equal(counts.sum(axis=1), parent)
        np.testing.assert_array_equal(counts, local_shape.quadrant_counts_from_points(points, hist))
        np.testing.assert_array_equal(counts[0], [1, 1, 1, 1])
        # Equality at the parent split belongs high; child coordinates reset.
        np.testing.assert_array_equal(sums[0, 3], [0, 0, 0, 0, 0])
        left_c, left_s = shape.accumulate_child_moments(points[:3], hist)
        right_c, right_s = shape.accumulate_child_moments(points[3:], hist)
        np.testing.assert_array_equal(left_c+right_c, counts)
        np.testing.assert_allclose(left_s+right_s, sums, rtol=0, atol=1e-15)

    def test_normalized_child_moment_codec(self):
        hist, _, _ = fixture(((4,),))
        points = np.array([[.05, .1], [.15, .2], [.3, .35], [.4, .45]])
        counts, sums = shape.accumulate_child_moments(points, hist)
        encoded = shape.encode_child_moments(counts, sums)
        child = points*2
        moments = np.array([[child[:, 0].mean(), child[:, 1].mean(),
                             child[:, 0].var(), child[:, 1].var(),
                             np.mean(child[:, 0]*child[:, 1])-child[:, 0].mean()*child[:, 1].mean()]])
        self.assertEqual(encoded.shape, (1, 4, 5))
        self.assertEqual(encoded.dtype, np.uint8)
        np.testing.assert_array_equal(encoded[0, :1], bounded.encode_moments(moments))
        self.assertTrue(np.isfinite(bounded.decode_moments(encoded.reshape(-1, 5))).all())

    def test_outside_parent_domain_rejected(self):
        hist, _, _ = fixture()
        for point in ([1., .2], [-.1, .2], [.2, float("nan")]):
            with self.assertRaises(ValueError):
                shape.accumulate_child_moments([point], hist)


class HierarchicalCodec(unittest.TestCase):
    def setUp(self):
        self.hist, self.encoded, self.children = fixture(((100, 100), (100, 0)))
        self.counts = np.array([[100, 0, 0, 0], [40, 20, 20, 20], [0, 0, 0, 100], [0, 0, 0, 0]])
        self.fixed = shape.HEADER_BYTES+shape.SOURCE_BYTES+1

    def build(self, budget=1000, **kwargs):
        return shape.build_selection([self.hist], [self.encoded], [self.counts], [self.children], budget, **kwargs)

    def test_exact_budget_and_source_selection_prefix(self):
        self.assertEqual(shape.SELECTED_CELL_BYTES, 32)
        previous = set()
        for k in range(4):
            budget = self.fixed+32*k+31
            payload = self.build(budget)
            selection = shape.decode_selection([self.hist], [self.encoded], payload)
            self.assertEqual(len(payload), self.fixed+32*k)
            self.assertLessEqual(len(payload), budget)
            self.assertEqual(len(selection.selected_ids), k)
            self.assertTrue(previous <= set(selection.selected_ids))
            previous = set(selection.selected_ids)
        with self.assertRaises(ValueError):
            self.build(self.fixed-1)
        self.assertEqual(self.build(), self.build())

    def test_explicit_geometry_priority_controls_allocation(self):
        payload = self.build(self.fixed+32, priority=np.array([1., 100., 2., 0.]))
        selection = shape.decode_selection([self.hist], [self.encoded], payload)
        self.assertEqual(selection.selected_ids.tolist(), [1])
        for priority in (np.array([1.]), np.array([1., np.nan, 1., 1.]), np.array([1., -1., 1., 1.])):
            with self.assertRaises(ValueError):
                self.build(priority=priority)

    def test_every_single_byte_corruption_is_detected(self):
        payload = self.build()
        for i in range(len(payload)):
            bad = bytearray(payload)
            bad[i] ^= 1
            with self.subTest(byte=i), self.assertRaises(ValueError):
                shape.decode_selection([self.hist], [self.encoded], bad)
        for bad in (b"", payload[:-1], payload+b"x"):
            with self.assertRaises(ValueError):
                shape.decode_selection([self.hist], [self.encoded], bad)

    def test_rechecksummed_invalid_counts_and_reserved_child_byte_rejected(self):
        payload = self.build()
        record = shape.HEADER_BYTES+shape.SOURCE_BYTES+1
        bad = bytearray(payload)
        struct.pack_into("<I", bad, record, 101)
        with self.assertRaises(ValueError):
            shape.decode_selection([self.hist], [self.encoded], repair_crc(bad))
        bad = bytearray(payload)
        bad[record+12+4] = 128  # Reserved signed covariance -128.
        with self.assertRaises(ValueError):
            shape.decode_selection([self.hist], [self.encoded], repair_crc(bad))

    def test_native_identity_exact_counts_and_invalid_children(self):
        payload = self.build()
        changed = self.encoded.copy()
        changed[0, 0] += 1
        with self.assertRaises(ValueError):
            shape.decode_selection([self.hist], [changed], payload)
        hist, encoded, children = fixture(((2**32,),))
        with self.assertRaises(OverflowError):
            shape.build_selection([hist], [encoded], [np.array([[2**32, 0, 0, 0]], np.uint64)], [children], 200)
        with self.assertRaises(ValueError):
            shape.build_selection([self.hist], [self.encoded], [self.counts+1], [self.children], 1000)
        with self.assertRaises(ValueError):
            shape.build_selection([self.hist], [self.encoded], [self.counts], [self.children.astype(float)], 1000)


class HierarchicalInference(unittest.TestCase):
    def setUp(self):
        self.hist, self.encoded, self.children = fixture()
        self.counts = np.array([[10, 20, 30, 40]], np.uint64)
        self.payload = shape.build_selection([self.hist], [self.encoded], [self.counts], [self.children], 200)
        self.runtime = shape.load_runtime([self.hist], [self.encoded], self.payload)

    def test_exact_quarter_quota_and_root_mass(self):
        np.testing.assert_array_equal(self.runtime.batch_reference(local_shape.QUADRANTS), self.counts[0])
        np.testing.assert_array_equal(self.runtime.batch_reference(local_shape.QUADRANTS, mode="empirical_uniform"), self.counts[0])
        self.assertEqual(self.runtime.reference_count([0, 1, 0, 1]), 100.)
        self.assertEqual(self.runtime.reference_count([-1, 2, -1, 2]), 100.)
        self.assertEqual(self.runtime.reference_count([2, 3, 2, 3]), 0.)

    def test_partial_child_shape_differs_from_uniform(self):
        query = np.array([0, .25, 0, .25])
        child_rect = np.array([[0, .5, 0, .5]])
        moments = bounded.decode_moments(self.children[0, :1]).astype(float)
        expected = 10*boundary_probability(child_rect, moments, "beta_gaussian_copula")[0]
        self.assertAlmostEqual(self.runtime.reference_count(query), expected, places=12)
        self.assertEqual(self.runtime.reference_count(query, mode="empirical_uniform"), 2.5)
        self.assertGreater(abs(expected-2.5), 1.)
        high_query = [.5, .75, .5, .75]
        moments = bounded.decode_moments(self.children[0, 3:]).astype(float)
        expected = 40*boundary_probability(child_rect, moments, "beta_gaussian_copula")[0]
        self.assertAlmostEqual(self.runtime.reference_count(high_query), expected, places=12)

    def test_child_endpoint_atoms_do_not_duplicate_adjacent_partition_quota(self):
        children = np.zeros((1, 4, 5), np.uint8)
        payload = shape.build_selection([self.hist], [self.encoded], [self.counts], [children], 200)
        runtime = shape.load_runtime([self.hist], [self.encoded], payload)
        np.testing.assert_array_equal(runtime.batch_reference(local_shape.QUADRANTS), self.counts[0])
        self.assertEqual(runtime.reference_count([0, 1, 0, 1]), 100.)
        self.assertEqual(runtime.reference_count([0, .25, 0, .25]), 10.)

    def test_no_selection_matches_native_base(self):
        fixed = shape.HEADER_BYTES+shape.SOURCE_BYTES+1
        runtime = shape.load_runtime([self.hist], [self.encoded],
                                     shape.build_selection([self.hist], [self.encoded], [self.counts], [self.children], fixed))
        queries = np.array([[.1, .7, .2, .8], [0, 1, 0, 1], [-1, .3, 0, .4]])
        components = bounded.components_from_histogram_moments(self.hist, self.encoded)
        np.testing.assert_allclose(runtime.batch_reference(queries), bounded.batch_reference(components, queries), rtol=1e-14)

    def test_features_and_persistence_use_no_raw_points(self):
        queries = np.array([[.1, .7, .2, .8], [0, 1, 0, 1]])
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)/"snapshot"
            nbytes = shape.save_snapshot(directory, [self.hist], [self.encoded], self.payload,
                                         original_histogram=self.hist)
            self.assertEqual(nbytes, sum(p.stat().st_size for p in directory.iterdir()))
            with mock.patch.object(workflow, "iter_raw", side_effect=AssertionError("Inference read raw points")):
                runtime = shape.load_snapshot(directory)
                features = runtime.query_features(queries)
                self.assertEqual(runtime.nbytes, nbytes)
                self.assertEqual(runtime.auxiliary_bytes, self.runtime.auxiliary_bytes)
                self.assertEqual(runtime.reference_count([0, 1, 0, 1]), 100.)
                self.assertIn("H", features)
                np.testing.assert_array_equal(features["correction_delta"], features["reference_hybrid"]-features["reference_base"])
                self.assertEqual(features["corrected_boundary_mass"][0], 100.)
                self.assertEqual(features["corrected_boundary_mass"][1], 0.)
                self.assertEqual(features["full_mass"][1], 100.)
            (directory/"uncharged.bin").write_bytes(b"x")
            with self.assertRaises(ValueError):
                shape.load_snapshot(directory)


if __name__ == "__main__":
    unittest.main()
