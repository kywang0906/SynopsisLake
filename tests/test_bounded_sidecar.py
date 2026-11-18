"""Behavioral checks for the bounded, child-only distribution sidecar.

Run without additional test dependencies with::

    python -m unittest discover -s tests -p 'test_bounded_sidecar.py'
"""

import ast
import json
from pathlib import Path
import types
import unittest
from unittest import mock

import numpy as np

from ml import bounded_sidecar as bs


def components(bounds, masses, moments=None):
    bounds = np.asarray(bounds, dtype=float).reshape(-1, 4)
    if moments is None:
        moments = np.tile([0.5, 0.5, 1 / 12, 1 / 12, 0.0], (len(bounds), 1))
    return bs.Components(bounds, np.asarray(masses, dtype=np.uint64),
                         bs.encode_moments(np.asarray(moments, dtype=float)))


def source_grid(n=4):
    bounds = [[i, i + 1, j, j + 1] for i in range(n) for j in range(n)]
    masses = np.arange(1, len(bounds) + 1, dtype=np.uint64)
    moments = [[0.2 + 0.1 * (i % 5), 0.3 + 0.05 * (i % 4),
                0.02 + 0.003 * (i % 3), 0.04, (-1) ** i * 0.01]
               for i in range(len(bounds))]
    return components(bounds, masses, moments)


TRAIN_RECTANGLES = np.asarray([
    [0.0, 1.4, 0.0, 3.0], [0.1, 3.2, 1.3, 2.8],
    [1.2, 2.2, 0.3, 3.1], [0.0, 4.0, 0.0, 4.0],
    [2.1, 3.9, 2.2, 3.6], [-1.0, 0.2, 0.2, 0.8],
], dtype=float)


def r08_analytic_module():
    """Read the retained reference, rather than mirroring its math in a test."""
    notebook = Path(__file__).resolve().parents[1] / "ml/notebooks/20_compact_distribution_critic.ipynb"
    document = json.loads(notebook.read_text())
    for cell in document["cells"]:
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell["source"])
        if not source.startswith("# Embedded helper: analytic_moments_fast.py"):
            continue
        tree = ast.parse(source)
        for node in tree.body:
            if isinstance(node, ast.Assign):
                module_source = ast.literal_eval(node.value)
                reference = types.ModuleType("_r08_analytic_test_reference")
                exec(compile(module_source, str(notebook), "exec"), reference.__dict__)
                return reference
    raise AssertionError("The retained R08 numerical reference was not found")


class CodecTests(unittest.TestCase):
    def test_round_trip_charges_every_record_and_uses_frozen_arrays(self):
        bounds = np.asarray([[0, 1, 0, 2], [-1, 2, -2, 1]], dtype=float)
        mass = np.asarray([13, 29], dtype=np.uint64)
        encoded = bs.encode_moments(np.asarray([
            [0.2, 0.8, 0.02, 0.1, -0.025], [0.8, 0.3, 0.04, 0.05, 0.02]]))
        c = bs.Components(bounds, mass, encoded)
        bounds[:] = 100
        mass[:] = 0
        encoded[:] = 0
        payload = bs.encode(c, 154)
        self.assertEqual(len(payload), 64 + 45 * 2)
        restored = bs.decode(payload)
        self.assertEqual(restored.total_mass, 42)
        self.assertEqual(restored.budget_bytes, 154)
        self.assertEqual(restored.payload_bytes, 154)
        # Serialization may canonicalize record order; bounds, mass and
        # encoded moments must remain associated with the same component.
        original_order = np.argsort(c.mass)
        restored_order = np.argsort(restored.components.mass)
        np.testing.assert_array_equal(restored.components.bounds[restored_order],
                                      c.bounds[original_order])
        np.testing.assert_array_equal(restored.components.mass[restored_order],
                                      c.mass[original_order])
        np.testing.assert_array_equal(restored.components.encoded[restored_order],
                                      c.encoded[original_order])
        self.assertEqual(bs.encode(restored.components, 154), payload)
        for array in (c.bounds, c.mass, c.encoded, restored.components.bounds,
                      restored.components.mass, restored.components.encoded):
            self.assertFalse(array.flags.writeable)
            with self.assertRaises(ValueError):
                array.flat[0] = 0

    def test_header_only_empty_and_exact_integer_mass(self):
        empty = components([], [])
        payload = bs.build_leaf(empty, 64, mode="geometry")
        self.assertEqual(len(payload), 64)
        self.assertEqual(bs.decode(payload).total_mass, 0)
        self.assertEqual(bs.reference_count(payload, [0, 1, 0, 1]), 0)
        large_mass = 2 ** 53 + 17
        payload = bs.encode(components([[0, 1, 0, 1]], [large_mass]), 109)
        self.assertEqual(int(bs.decode(payload).components.mass[0]), large_mass)
        self.assertEqual(bs.decode(payload).total_mass, large_mass)

    def test_budget_checks_are_on_actual_bytes(self):
        c = source_grid(2)
        with self.assertRaises((ValueError, OverflowError)):
            bs.encode(c, 64 + 45 * len(c.mass) - 1)
        for cap in (0, 63, 64, 108):
            with self.subTest(cap=cap):
                with self.assertRaises((ValueError, OverflowError)):
                    bs.build_leaf(c, cap, mode="geometry")
        payload = bs.build_leaf(c, 109, mode="geometry")
        self.assertEqual(len(payload), 109)
        self.assertEqual(len(bs.decode(payload).components.mass), 1)

    def test_malformed_serializations_are_rejected(self):
        payload = bs.encode(source_grid(2), 64 + 45 * 4)
        bad_magic = bytearray(payload)
        bad_magic[0] ^= 1
        bad_flags = bytearray(payload)
        bad_flags[10] ^= 1
        for malformed in (b"", payload[:63], payload[:-1], payload + b"\0",
                          bytes(bad_magic), bytes(bad_flags)):
            with self.subTest(length=len(malformed)):
                with self.assertRaises((ValueError, OverflowError)):
                    bs.decode(malformed)

    def test_invalid_component_information_and_overflow_fail(self):
        valid = bs.encode_moments(np.asarray([[0.5, 0.5, 0.02, 0.02, 0.0]]))
        for bounds, mass, encoded in (
            ([[0, 0, 0, 1]], [1], valid),
            ([[0, 1, 1, 0]], [1], valid),
            ([[0, np.nan, 0, 1]], [1], valid),
            ([[0, 1, 0, 1]], [-1], valid),
            ([[0, 1, 0, 1]], [1.5], valid),
            ([[0, 1, 0, 1]], [1], np.zeros((1, 4), np.uint8)),
        ):
            with self.subTest(bounds=bounds, mass=mass):
                with self.assertRaises((ValueError, TypeError, OverflowError)):
                    bs.Components(bounds, mass, encoded)
        bad_covariance = valid.copy()
        bad_covariance[:, 4] = 128
        with self.assertRaises((ValueError, TypeError)):
            bs.Components([[0, 1, 0, 1]], [1], bad_covariance)
        with self.assertRaises((ValueError, OverflowError)):
            c = components([[0, 1, 0, 1], [1, 2, 0, 1]], [2 ** 64 - 1, 1])
            bs.encode(c, 154)

    def test_original_moment_scaling_and_histogram_cell_ownership(self):
        moments = np.asarray([[0, 1, 0, 0.25, -0.25],
                              [1, 0, 0.25, 0, 0.25]])
        np.testing.assert_array_equal(bs.encode_moments(moments),
                                      [[0, 255, 0, 255, 129],
                                       [255, 0, 255, 0, 127]])
        np.testing.assert_array_equal(bs.decode_moments(bs.encode_moments(moments)), moments)
        hist = (1.0, np.asarray([0, 2, 4]), np.asarray([-2, 0, 2]),
                np.asarray([[0, 3], [7, 0]], dtype=np.uint64))
        per_cell = np.tile([0.3, 0.6, 0.02, 0.04, 0.01], (4, 1))
        c = bs.components_from_histogram_moments(hist, per_cell)
        np.testing.assert_array_equal(c.bounds, [[0, 2, 0, 2], [2, 4, -2, 0]])
        np.testing.assert_array_equal(c.mass, [3, 7])
        split = bs.components_from_histogram_moments(hist, per_cell[:, :2],
                                                     per_cell[:, 2:4], per_cell[:, 4])
        self.assertEqual(bs.encode(c, 154), bs.encode(split, 154))


class ReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reference = r08_analytic_module()

    def test_r08_beta_copula_parity_on_partial_intersection(self):
        c = components([[0, 2, -2, 2]], [37],
                       [[0.32, 0.71, 0.043, 0.031, -0.019]])
        decoded = bs.decode_moments(c.encoded).astype(float)
        local = np.asarray([[0.125, 0.75, 0.25, 0.75]])
        expected = 37 * self.reference.boundary_probability(
            local, decoded, "beta_gaussian_copula")[0]
        payload = bs.encode(c, 109)
        for information in (payload, bs.decode(payload), c):
            self.assertAlmostEqual(bs.reference_count(information, [0.25, 1.5, -1, 1]),
                                   expected, places=11)

    def test_mixture_full_containment_disjointness_and_batch(self):
        c = components([[-2, 0, 0, 1], [0, 3, -1, 2], [1, 2, 0, 1]], [5, 11, 17])
        payload = bs.encode(c, 64 + 45 * 3)
        queries = np.asarray([[-2, 3, -1, 2], [10, 11, 10, 11],
                              [-3, -2.5, 0, 1], [-2, 0, 0, 1]])
        answers = bs.batch_reference(payload, queries)
        self.assertEqual(answers.shape, (4,))
        self.assertEqual(answers[0], 33)
        self.assertEqual(answers[1], 0)
        self.assertEqual(answers[2], 0)
        np.testing.assert_allclose(answers, [bs.reference_count(payload, q) for q in queries],
                                   rtol=0, atol=0)
        self.assertEqual(bs.batch_reference(payload, np.empty((0, 4))).shape, (0,))

    def test_zero_variance_and_zero_correlation_use_reference_policy(self):
        c = components([[0, 1, 0, 1], [1, 2, 0, 1]], [2, 3],
                       [[0, 1, 0, 0, 0], [0.4, 0.6, 0.03, 0.02, 0]])
        payload = bs.encode(c, 154)
        self.assertEqual(bs.reference_count(payload, [0, 0, 1, 1]), 2)
        self.assertEqual(bs.reference_count(payload, [0.1, 0.9, 0.1, 0.9]), 0)
        local = np.asarray([[0.25, 0.75, 0.25, 0.75]])
        moments = bs.decode_moments(c.encoded[1:])
        expected = 3 * self.reference.boundary_probability(local, moments,
                                                          "beta_independent")[0]
        self.assertAlmostEqual(bs.reference_count(payload, [1.25, 1.75, 0.25, 0.75]),
                               expected, places=11)

    def test_invalid_queries_are_rejected(self):
        payload = bs.encode(source_grid(1), 109)
        for rect in ([0, 1, 0], [1, 0, 0, 1], [0, 1, np.nan, 1]):
            with self.subTest(rect=rect):
                with self.assertRaises((ValueError, TypeError)):
                    bs.reference_count(payload, rect)


class CompressionTests(unittest.TestCase):
    def test_uint64_limit_mass_is_exact_after_child_merge(self):
        left = bs.encode(components([[0, 1, 0, 1]], [2 ** 63]), 109)
        right = bs.encode(components([[2, 3, 0, 1]], [2 ** 63 - 1]), 109)
        merged = bs.merge_bytes(left, right, 109, mode="geometry")
        self.assertEqual(bs.decode(merged).total_mass, 2 ** 64 - 1)
        self.assertEqual(int(bs.decode(merged).components.mass[0]), 2 ** 64 - 1)

    def test_bounded_training_selection_is_independent_of_query_row_order(self):
        c = source_grid(3)
        rectangles = np.asarray([[i / 100, 1 + i / 100, 0.1, 2.4]
                                 for i in range(101)])
        first = bs.compress(c, 154, rectangles, mode="workload", max_training_queries=4)
        second = bs.compress(c, 154, np.concatenate((rectangles[::-1], rectangles[:20])),
                             mode="workload", max_training_queries=4)
        self.assertEqual(first, second)

    def test_pooling_uses_local_coordinates_at_large_global_offsets(self):
        # Two equally weighted endpoint atoms have mean .5, variance .25 and
        # covariance .25 in their union box, even far from the coordinate origin.
        x, y = 1e12, 2e12
        c = components([[x, x + 1, y, y + 1], [x + 3, x + 4, y + 3, y + 4]],
                       [7, 7], [[0, 0, 0, 0, 0], [1, 1, 0, 0, 0]])
        summary = bs.decode(bs.compress(c, 109, mode="geometry"))
        np.testing.assert_array_equal(summary.components.bounds,
                                      [[x, x + 4, y, y + 4]])
        np.testing.assert_array_equal(summary.components.encoded,
                                      [[128, 128, 255, 255, 127]])
        self.assertEqual(summary.total_mass, 14)

    def test_each_mode_respects_budget_and_exact_mass_at_each_tree_node(self):
        leaves = [source_grid(2),
                  components([[-3, -1, -2, 2], [-2, 0, 1, 4]], [31, 47]),
                  components([[0.2, 4.5, -3, 0.2]], [101])]
        for mode in ("geometry", "workload"):
            with self.subTest(mode=mode):
                payloads = [bs.build_leaf(c, 154, TRAIN_RECTANGLES, mode=mode) for c in leaves]
                expected_mass = [int(sum(map(int, c.mass))) for c in leaves]
                for p, mass in zip(payloads, expected_mass):
                    self.assertLessEqual(len(p), 154)
                    self.assertEqual(bs.decode(p).total_mass, mass)
                left = bs.merge_bytes(payloads[0], payloads[1], 154, TRAIN_RECTANGLES, mode=mode)
                self.assertEqual(bs.decode(left).total_mass, sum(expected_mass[:2]))
                root_left = bs.merge_bytes(left, payloads[2], 154, TRAIN_RECTANGLES, mode=mode)
                right = bs.merge_bytes(payloads[1], payloads[2], 154, TRAIN_RECTANGLES, mode=mode)
                root_right = bs.merge_bytes(payloads[0], right, 154, TRAIN_RECTANGLES, mode=mode)
                for root in (root_left, root_right):
                    summary = bs.decode(root)
                    self.assertEqual(summary.total_mass, sum(expected_mass))
                    self.assertLessEqual(len(root), 154)
                    self.assertEqual(len(root), 64 + 45 * len(summary.components.mass))
                    self.assertEqual(bs.reference_count(root, [-10, 10, -10, 10]),
                                     sum(expected_mass))

    def test_order_and_ties_are_reproducible(self):
        c = source_grid(3)
        reversed_c = bs.Components(c.bounds[::-1], c.mass[::-1], c.encoded[::-1])
        for mode in ("geometry", "workload"):
            with self.subTest(mode=mode):
                first = bs.compress(c, 199, TRAIN_RECTANGLES, mode=mode)
                self.assertEqual(first, bs.compress(c, 199, TRAIN_RECTANGLES, mode=mode))
                self.assertEqual(first, bs.compress(reversed_c, 199, TRAIN_RECTANGLES, mode=mode))
                p = bs.build_leaf(c, 244, TRAIN_RECTANGLES, mode=mode)
                q = bs.encode(components([[-2, 3, -3, 1]], [37]), 109)
                self.assertEqual(bs.merge_bytes(p, q, 199, TRAIN_RECTANGLES, mode=mode),
                                 bs.merge_bytes(q, p, 199, TRAIN_RECTANGLES, mode=mode))

    def test_child_only_merge_uses_serialized_information_without_file_reads(self):
        c = source_grid(3)
        p = bs.build_leaf(c, 244, TRAIN_RECTANGLES, mode="workload")
        q = bs.encode(components([[-3, 2, 1, 4]], [17]), 109)
        with mock.patch("builtins.open", side_effect=AssertionError("merge read a file")), \
             mock.patch.object(Path, "read_bytes", side_effect=AssertionError("merge read a path")):
            merged = bs.merge_bytes(p, q, 154, TRAIN_RECTANGLES, mode="workload")
        self.assertEqual(bs.decode(merged).total_mass, bs.decode(p).total_mass + 17)
        self.assertEqual(merged, bs.merge_bytes(bytes(p), bytes(q), 154,
                                               TRAIN_RECTANGLES, mode="workload"))

    def test_training_geometry_is_required_only_for_workload_compression(self):
        c = source_grid(2)
        self.assertEqual(len(bs.compress(c, 244, mode="workload")), 244)
        with self.assertRaises((ValueError, TypeError)):
            bs.compress(c, 109, mode="workload")
        with self.assertRaises((ValueError, TypeError)):
            bs.compress(c, 109, np.column_stack((TRAIN_RECTANGLES, np.ones(6))),
                        mode="workload")
        with self.assertRaises((ValueError, TypeError)):
            bs.compress(c, 109, TRAIN_RECTANGLES, mode="unknown")
        result = bs.compress(c, 109, mode="geometry")
        self.assertEqual(bs.decode(result).total_mass, int(c.mass.sum()))


if __name__ == "__main__":
    unittest.main()
