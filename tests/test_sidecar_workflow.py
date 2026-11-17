"""Independent checks of the scientific protocol and real-file workflow.

The generated fixture checks execution and accounting only. It is not a
performance or answer-quality experiment on the authors dataset.
"""
import gzip
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
from zipfile import ZipFile

import numpy as np
import pandas as pd

from ml import bounded_sidecar as bounded
from ml import sidecar_workflow as workflow


def write_histogram(path, counts):
    counts = np.asarray(counts)
    lines = ["1.0", "0,1,2,3", "0,1,2,3"]
    lines += [",".join(map(str, row)) for row in counts]
    path.write_text("\n".join(lines) + "\n")


def fixture(root):
    histograms, raw = root / "histograms", root / "raw_input"
    histograms.mkdir()
    raw.mkdir()
    points = np.array([[i + u, j + u] for i in range(3)
                       for j in range(3) for u in (.1, .2, .3, .4, .5, .6)])
    for source in (0, 1):
        write_histogram(histograms / f"src_{source}.txt", np.full((3, 3), 6))
        text = "".join(f"{x:.17g}\t{y:.17g}\t0\n" for x, y in points)
        if source:
            with gzip.open(raw / "1.csv.gz", "wt") as stream:
                stream.write(text)
        else:
            (raw / "0.csv").write_text(text)
    write_histogram(histograms / "arm_3.txt", np.full((3, 3), 12))
    rows = []
    for group in range(9):
        i, j = divmod(group, 3)
        for narrow in (False, True):
            pad, size = (.05, .8) if narrow else (0., 1.)
            rows.append(dict(qId=len(rows), x0=i + pad, x1=i + pad + size,
                             y0=j + pad, y1=j + pad + size, truth=12,
                             cell=f"cell_{group}"))
    queries = pd.DataFrame(rows)
    query_path = root / "queries.csv"
    queries.drop(columns="cell").to_csv(query_path, index=False)
    splits = []
    for seed in (0, 1, 2):
        for row in queries.itertuples(index=False):
            group = int(row.cell.split("_")[1])
            role = ("train", "calibration", "test")[(group // 3 + seed) % 3]
            splits.append(dict(seed=seed, qId=row.qId, cell=row.cell,
                               split=role, split_kind="cell"))
    split_path = root / "outer_splits.csv"
    pd.DataFrame(splits).to_csv(split_path, index=False)
    config = dict(histogram_dir=str(histograms), raw_data_dir=str(raw),
                  queries_file=str(query_path), split_file=str(split_path),
                  output_root=str(root / "results"), base_resolution=3,
                  budget_bytes=512, seeds=[0, 1, 2], workload_queries=0,
                  calibration_queries=0, test_queries=0, timing_queries=2,
                  timing_repeats=1, parity_queries=3,
                  modes=["geometry", "workload"],
                  merge_trees=["balanced", "left", "right"])
    return config, queries, pd.DataFrame(splits)


class ReaderAndQualityGuards(unittest.TestCase):
    def test_raw_reader_preserves_round_trip_coordinates(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "points.csv"
            path.write_text("0.8118980320571972\t0.12345678901234567\t0\n")
            actual = list(workflow.iter_raw(path))[0][0]
            expected = np.array([float("0.8118980320571972"),
                                 float("0.12345678901234567")])
            np.testing.assert_array_equal(actual, expected)

    def test_zero_truth_errors_are_not_dropped_from_all_query_mae(self):
        frame = pd.DataFrame([
            dict(method="reference", split="calibration", score=0.,
                 actual_error=1., truth=20),
            dict(method="reference", split="test", score=0.,
                 actual_error=np.inf, truth=0),
            dict(method="reference", split="test", score=0.,
                 actual_error=0., truth=20),
        ])
        metrics, _, _ = workflow._quality(frame, 0, .25, .05, 10)
        all_queries = metrics[metrics.cohort == "all_queries"].iloc[0]
        self.assertFalse(np.isfinite(all_queries.mae))
        self.assertEqual(all_queries.nonfinite_actual_errors, 1)
        self.assertEqual(all_queries.accepted_queries, 0)
        self.assertTrue(np.isnan(all_queries.violation))

    def test_grouped_role_overlap_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            config, queries, splits = fixture(Path(tmp))
            splits.loc[(splits.seed == 0) & (splits.qId == 6), "cell"] = "cell_0"
            path = Path(tmp) / "invalid_splits.csv"
            splits.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "overlap"):
                workflow._splits(queries, workflow._config(config), path)


class WorkflowExecutionGuards(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.config, cls.queries, cls.splits = fixture(Path(cls.temporary.name))
        # Plot rendering is unrelated to the scientific and execution guards.
        with mock.patch.object(workflow, "_figures"):
            cls.result = workflow.run_experiment(cls.config)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_seed_workloads_use_only_their_inherited_training_queries(self):
        by_id = self.queries.set_index("qId")
        all_ids = []
        for seed in (0, 1, 2):
            directory = self.result / "payloads" / f"seed_{seed}"
            ids = json.loads((directory / "training_ids.json").read_text())
            expected = set(self.splits.loc[(self.splits.seed == seed) &
                                           (self.splits.split == "train"), "qId"])
            self.assertEqual(set(ids), expected)
            np.testing.assert_array_equal(np.load(directory / "training_geometry.npy"),
                by_id.loc[ids, ["x0", "x1", "y0", "y1"]].to_numpy(float))
            all_ids.append(set(ids))
        self.assertFalse(all_ids[0] & all_ids[1])

    def test_every_saved_node_respects_actual_bytes_and_conserves_all_mass(self):
        audit = pd.read_csv(self.result / "merge_audit.csv")
        self.assertTrue(audit.mass_matches.all())
        self.assertTrue((audit.payload_bytes <= self.config["budget_bytes"]).all())
        self.assertTrue(audit.raw_rows_read.eq(0).all())
        self.assertTrue(audit.raw_bytes_read.eq(0).all())
        for row in audit.itertuples(index=False):
            payload = Path(row.payload_path).read_bytes()
            self.assertEqual(len(payload), row.payload_bytes)
            self.assertEqual(bounded.decode(payload).total_mass, row.input_mass)
        for seed in (0, 1, 2):
            for mode in self.config["modes"]:
                for tree in self.config["merge_trees"]:
                    root = self.result / "payloads" / f"seed_{seed}" / mode / tree / "root.bin"
                    self.assertEqual(bounded.decode(root.read_bytes()).total_mass, 108)

    def test_native_r08_storage_is_separate_from_common_union_representation(self):
        state = json.loads((self.result / "run.json").read_text())
        self.assertEqual(state["structured_reference_auxiliary_bytes"], 538)
        self.assertEqual(state["union_common_codec_bytes"], 64 + 45 * 18)
        self.assertNotEqual(state["structured_reference_auxiliary_bytes"],
                            state["union_common_codec_bytes"])
        costs = pd.read_csv(self.result / "payload_costs.csv")
        reference = costs[costs.method == "uncompressed_R08_reference"]
        self.assertTrue(reference.auxiliary_bytes.eq(538).all())
        self.assertTrue(reference.runtime_representation_bytes.eq(538).all())
        self.assertTrue(reference.offline_union_verification_bytes.eq(874).all())

    def test_total_auxiliary_bytes_charge_saved_workload_geometry(self):
        costs = pd.read_csv(self.result / "payload_costs.csv")
        bounded_costs = costs[costs.method.str.startswith("bounded_")]
        self.assertTrue((bounded_costs.auxiliary_bytes <= self.config["budget_bytes"]).all())
        self.assertTrue((bounded_costs.auxiliary_bytes == bounded_costs.payload_bytes +
                         bounded_costs.global_training_geometry_bytes).all())
        workload = bounded_costs[bounded_costs.method.str.startswith("bounded_workload_")]
        self.assertTrue((workload.global_training_geometry_bytes > 0).all())
        self.assertTrue((bounded_costs.total_serialized_bytes == bounded_costs.original_H_bytes +
                         bounded_costs.auxiliary_bytes).all())

    def test_fine_baseline_fits_largest_compact_u32_grid(self):
        state = json.loads((self.result / "run.json").read_text())
        n, size, cap = (state["fine_matched_resolution"],
                        state["fine_matched_auxiliary_bytes"], self.config["budget_bytes"])
        self.assertEqual(size, 64 + 4 * n * n)
        self.assertLessEqual(size, cap)
        self.assertGreater(64 + 4 * (n + 1) ** 2, cap)

    def test_protocol_lock_rejects_tampered_frozen_training_geometry(self):
        workflow.verify_protocol_lock(self.result)
        path = self.result / "payloads" / "seed_0" / "training_geometry.npy"
        original = path.read_bytes()
        try:
            path.write_bytes(original + b"tampered")
            with self.assertRaises((ValueError, RuntimeError)):
                workflow.verify_protocol_lock(self.result)
        finally:
            path.write_bytes(original)

    def test_archive_contains_evidence_and_no_raw_point_files(self):
        archive = workflow.package_results(self.result)
        with ZipFile(archive) as zipped:
            names = zipped.namelist()
        self.assertIn("protocol.json", names)
        self.assertIn("merge_audit.csv", names)
        self.assertIn("payload_costs.csv", names)
        self.assertFalse(any(name.endswith(("0.csv", "1.csv.gz")) for name in names))


class FreshQueryFreezeGuards(unittest.TestCase):
    def test_scores_and_calibration_gates_are_frozen_before_fresh_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            config, _, _ = fixture(Path(tmp))
            config.update(seeds=[0], modes=["workload"], merge_trees=["balanced"],
                          fresh_query_count=5, fresh_query_seed=2026111709)
            generate = workflow._fresh_rectangles
            label = workflow._indexed_labels
            phases = []

            def current_result():
                return next(Path(config["output_root"]).glob("bounded-sidecar-*"))

            def checked_generation(*args, **kwargs):
                result = current_result()
                self.assertTrue((result / "frozen_gate_cutoffs.csv").is_file())
                self.assertFalse((result / "fresh_queries_labeled_offline.csv").exists())
                workflow.verify_protocol_lock(result)
                phases.append("geometry")
                return generate(*args, **kwargs)

            def checked_labels(*args, **kwargs):
                result = current_result()
                lock = json.loads((result / "fresh_score_lock.json").read_text())
                self.assertEqual(lock["geometry_sha256"],
                                 workflow._sha(result / "fresh_queries_unlabeled.csv"))
                self.assertEqual(lock["scores_sha256"],
                                 workflow._sha(result / "fresh_scores_before_labels.csv"))
                scores = pd.read_csv(result / "fresh_scores_before_labels.csv")
                self.assertFalse({"truth", "actual_error"} & set(scores.columns))
                self.assertFalse((result / "fresh_queries_labeled_offline.csv").exists())
                workflow.verify_protocol_lock(result)
                phases.append("labels")
                return label(*args, **kwargs)

            with mock.patch.object(workflow, "_fresh_rectangles", side_effect=checked_generation), \
                    mock.patch.object(workflow, "_indexed_labels", side_effect=checked_labels), \
                    mock.patch.object(workflow, "_figures"):
                result = workflow.run_experiment(config)
            self.assertEqual(phases, ["geometry", "labels"])
            frozen = pd.read_csv(result / "frozen_gate_cutoffs.csv").set_index(["seed", "method"])
            cutoffs = pd.read_csv(result / "gate_cutoffs.csv")
            self.assertEqual(set(cutoffs.family), {"existing_exploratory", "fresh_same_dataset"})
            for family, part in cutoffs.groupby("family"):
                actual = part.set_index(["seed", "method"]).loc[frozen.index, "cutoff"]
                np.testing.assert_array_equal(actual, frozen.cutoff)
            scores_path = result / "fresh_scores_before_labels.csv"
            scores_path.write_text(scores_path.read_text() + "tampered\n")
            with self.assertRaises((ValueError, RuntimeError)):
                workflow.verify_protocol_lock(result)


if __name__ == "__main__":
    unittest.main()
