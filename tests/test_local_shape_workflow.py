"""Independent stage, leakage and accounting checks for the R10 workflow.

Tiny generated data exercise correctness only. They are not evidence of query
accuracy or performance on the authors dataset. Confirmation labels are mocked.
"""
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock
from zipfile import ZipFile

import numpy as np
import pandas as pd

from ml import local_shape as shape
from ml import local_shape_workflow as workflow
from ml import shape_error_model as models
from ml import sidecar_e2e as e2e
from ml import sidecar_workflow as prior
from tests.test_sidecar_workflow import fixture


class FeatureContractGuards(unittest.TestCase):
    def test_base_and_hybrid_features_use_their_own_reference_and_supplied_H(self):
        rectangles = np.array([[2., 6., 14., 18.], [0., 2., 10., 11.]])
        described = dict(H=np.array([30., 0.]), reference_base=np.array([10., 4.]),
                         reference_hybrid=np.array([20., 2.]),
                         boundary_mass=np.array([6., 0.]), full_mass=np.array([4., 0.]),
                         corrected_boundary_fraction=np.array([.25, 0.]))
        before = {name: value.copy() for name, value in described.items()}
        base = workflow.model_features(rectangles, described, [0, 10, 10, 30])
        hybrid = workflow.model_features(rectangles, described, [0, 10, 10, 30], hybrid=True)
        self.assertEqual(tuple(base), models.BASE_FEATURES)
        self.assertEqual(tuple(hybrid), models.HYBRID_FEATURES)
        np.testing.assert_array_equal(base.log_H, np.log1p([30., 0.]))
        np.testing.assert_array_equal(hybrid.log_H, base.log_H)
        np.testing.assert_array_equal(base.log_reference, np.log1p([10., 4.]))
        np.testing.assert_array_equal(hybrid.log_reference, np.log1p([20., 2.]))
        np.testing.assert_array_equal(base.proxy_score, [2., 1.])
        np.testing.assert_array_equal(hybrid.proxy_score, [.5, 1.])
        np.testing.assert_array_equal(hybrid.base_proxy_score, base.proxy_score)
        np.testing.assert_array_equal(hybrid.reference_shift, [1., -.5])
        np.testing.assert_allclose(base.query_width, [.4, .2])
        np.testing.assert_allclose(base.query_height, [.2, .05])
        np.testing.assert_allclose(base.query_area, [.08, .01])
        np.testing.assert_allclose(base.query_aspect, [2., 4.])
        np.testing.assert_array_equal(base.partial_mass_fraction, [.6, 0.])
        for name in before:
            np.testing.assert_array_equal(described[name], before[name])
        self.assertFalse({"qId", "cell", "truth", "actual_error", "SA"} & set(hybrid))

    def test_supplied_live_H_overrides_a_histogram_reestimate(self):
        hist = (1., np.array([0., 1.]), np.array([0., 1.]), np.array([[12.]]))
        encoded = np.array([[100, 120, 40, 40, 0]], np.uint8)
        components = prior.bounded.components_from_histogram_moments(hist, encoded)
        described = workflow._base_query_features(np.array([[0., .5, 0., .5]]),
                                                   components, hist, answers=[99.])
        self.assertEqual(described["H"].tolist(), [99.])
        features = workflow.model_features(np.array([[0., .5, 0., .5]]), described, [0, 1, 0, 1])
        self.assertEqual(features.log_H.iloc[0], np.log1p(99.))

    def test_live_scorer_rejects_offline_truth_columns(self):
        geometry = pd.DataFrame([dict(qId=1, x0=0., x1=1., y0=0., y1=1., truth=12)])
        with self.assertRaisesRegex(ValueError, "geometry only"):
            workflow._score_live(geometry, [], None, 0, "test")


class StagedWorkflowGuards(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        old_config, cls.queries, cls.splits = fixture(cls.root)
        old_config.update(seeds=[0], modes=["geometry"], merge_trees=["balanced"],
                          timing_queries=1, parity_queries=1)
        with mock.patch.object(prior, "_figures"):
            cls.old_run = prior.run_experiment(old_config)
        cls.config = dict(run_dir=str(cls.old_run), output_root=str(cls.root/"r10"),
                          seeds=[0], extra_budgets=[128, 256], include_ml=False,
                          evaluate_old_test=False, train_queries=0, validation_queries=0,
                          calibration_queries=0, test_queries=0, parity_queries=1,
                          chunk_size=17, timing_repeats=1)
        cls.preparation = workflow.prepare_experiment(cls.config)
        cls.config["preparation_dir"] = str(cls.preparation)
        real_metrics = workflow._metrics

        def losing_correction(*args, **kwargs):
            result = real_metrics(*args, **kwargs)
            # A deterministic negative development outcome tests whether the
            # uncorrected baseline can win; this is not an accuracy assertion.
            result["mae"] = 0. if result["method"] == "R08" else 1.
            result["mae_pp"] = 100*result["mae"]
            return result

        with mock.patch.object(workflow, "_metrics", side_effect=losing_correction):
            cls.development = workflow.develop_experiment(cls.config)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_preparation_matches_every_exact_parent_and_source_identity(self):
        _, state = workflow._load_preparation(self.preparation)
        hists, _ = shape.load_native(self.preparation/"native")
        audit = pd.read_csv(self.preparation/"source_audit.csv")
        arrays = workflow._raw_arrays(self.preparation)
        self.assertEqual(state["raw_rows"], 108)
        self.assertTrue(audit.quadrant_mass_matches.all())
        for sid, (hist, points) in enumerate(zip(hists, arrays)):
            stored = np.load(self.preparation/"quadrants"/f"source_{sid}.npy", allow_pickle=False)
            np.testing.assert_array_equal(stored.sum(axis=1), hist[3].ravel())
            np.testing.assert_array_equal(shape.quadrant_counts_from_points(points, hist), stored)
        self.assertEqual(state["native_bytes"], sum(p.stat().st_size for p in (self.preparation/"native").iterdir()))

    def test_preparation_hash_lock_rejects_quadrant_and_raw_audit_tampering(self):
        for relative in ("quadrants/source_0.npy", "source_audit.csv"):
            path = self.preparation/relative
            original = path.read_bytes()
            try:
                path.write_bytes(original+b"changed")
                with self.subTest(relative=relative), self.assertRaisesRegex(ValueError, "changed"):
                    workflow._load_preparation(self.preparation)
            finally:
                path.write_bytes(original)
        workflow._load_preparation(self.preparation)

    def test_inner_fit_and_selection_validation_are_grouped_outer_train_only(self):
        assignments = pd.read_csv(self.development/"inner_assignments.csv")
        allowed = self.splits[(self.splits.seed == 0) & (self.splits.split == "train")]
        forbidden = self.splits[(self.splits.seed == 0) & (self.splits.split != "train")]
        self.assertEqual(set(assignments.qId), set(allowed.qId))
        self.assertFalse(set(assignments.qId) & set(forbidden.qId))
        fit = assignments[assignments.role == "fit"]
        validation = assignments[assignments.role == "validation"]
        self.assertTrue(len(fit) and len(validation))
        self.assertFalse(set(fit.cell) & set(validation.cell))
        predictions = pd.read_csv(self.development/"validation_predictions.csv")
        self.assertEqual(set(predictions.qId), set(validation.qId))
        self.assertFalse(set(predictions.qId) & set(fit.qId))

    def test_model_controls_receive_only_train_and_the_predeclared_inner_roles(self):
        cfg = workflow._cfg(self.config)
        queries = pd.read_csv(self.preparation/"queries_offline.csv")
        outer = pd.read_csv(self.development/"outer_splits.csv")
        fit, validation, _, info = workflow._inner(queries, outer, 0, cfg)
        hists, encoded = shape.load_native(self.development/"native")
        components = prior.load_structured_reference(self.development/"native")
        original = prior.load_histogram(self.development/"original_H.hist")
        root = (original[1][0], original[1][-1], original[2][0], original[2][-1])
        state = json.loads((self.development/"run.json").read_text())
        packet = state["packets"][str(state["candidate_extra_budget"])]
        observed = []

        def inspect_fit(table, names, seed, out_dir, **kwargs):
            observed.append(tuple(names))
            self.assertTrue(table.split.eq("train").all())
            self.assertEqual(set(table.qId), set(fit.qId) | set(validation.qId))
            self.assertEqual(kwargs["groups"], "cell")
            train_positions = kwargs["inner_fit_indices"]
            validation_positions = kwargs["inner_validation_indices"]
            self.assertEqual(set(table.iloc[train_positions].qId), set(fit.qId))
            self.assertEqual(set(table.iloc[validation_positions].qId), set(validation.qId))
            self.assertFalse(set(table.iloc[train_positions].cell) & set(table.iloc[validation_positions].cell))
            self.assertFalse({"qId", "cell", "truth", "actual_error", "split"} & set(names))
            return {"model_bytes": 0}

        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(models, "fit_models", side_effect=inspect_fit):
            workflow._fit_controls(Path(tmp), cfg, {0: (fit, validation, info)},
                                   hists, encoded, components, original, root, packet)
        self.assertEqual(observed, [models.BASE_FEATURES, models.HYBRID_FEATURES])

    def test_losing_corrections_select_zero_increment_and_block_fresh_queries(self):
        selection = json.loads((self.development/"selection.json").read_text())
        self.assertEqual(selection["selected_extra_budget"], 0)
        self.assertEqual(selection["selected_packet_bytes"], 0)
        self.assertFalse(selection["promising"])
        prepared = json.loads((self.preparation/"run.json").read_text())
        self.assertEqual(selection["selected_auxiliary_bytes"], prepared["native_bytes"])
        existing = set((self.development/"confirmation").glob("*"))
        with mock.patch.object(prior, "_fresh_rectangles") as geometry, \
                mock.patch.object(prior, "_indexed_labels") as labels:
            with self.assertRaisesRegex(ValueError, "did not meet"):
                workflow.confirm_experiment({**self.config, "development_dir": str(self.development),
                                             "fresh_query_count": 1000})
            geometry.assert_not_called()
            labels.assert_not_called()
        self.assertEqual(set((self.development/"confirmation").glob("*")), existing)

    def test_packet_rank_audit_is_source_only_and_identical_to_fixed_builder(self):
        hists, encoded = shape.load_native(self.development/"native")
        quadrants = [np.load(p, allow_pickle=False) for p in sorted((self.preparation/"quadrants").glob("*.npy"))]
        for budget in (128, 256):
            packet = self.development/"packets"/f"extra_{budget}.bin"
            payload = packet.read_bytes()
            self.assertEqual(payload, shape.build_selection(hists, encoded, quadrants, budget))
            stored = pd.read_csv(self.development/"packets"/f"allocation_{budget}.csv")
            expected = pd.DataFrame(shape.selection_audit(hists, encoded, quadrants, payload))
            pd.testing.assert_frame_equal(stored.drop(columns="extra_budget"), expected, check_dtype=False)
            self.assertFalse({"truth", "qId", "actual_error", "split"} & set(stored))
            selected = shape.decode_selection(hists, encoded, payload)
            self.assertEqual(int(stored.selected.sum()), len(selected.selected_ids))
            self.assertLessEqual(len(payload), budget)

    def test_full_auxiliary_costs_and_maximal_finer_controls_use_actual_files(self):
        costs = pd.read_csv(self.development/"payload_costs.csv").set_index("method")
        native = sum(p.stat().st_size for p in (self.development/"native").iterdir())
        original = (self.development/"original_H.hist").stat().st_size
        self.assertEqual(costs.loc["R08", "auxiliary_bytes"], native)
        state = json.loads((self.development/"run.json").read_text())
        candidate = state["packets"][str(state["candidate_extra_budget"])]
        packet_bytes = Path(candidate["path"]).stat().st_size
        for method in ("hybrid", "empirical_uniform"):
            self.assertEqual(costs.loc[method, "packet_bytes"], packet_bytes)
            self.assertEqual(costs.loc[method, "auxiliary_bytes"], native+packet_bytes)
        self.assertTrue((costs.total_serialized_bytes == original+costs.auxiliary_bytes).all())
        for name, row in costs[costs.index.str.startswith("fine_for_")].iterrows():
            target = row.matched_target
            self.assertEqual(row.matched_cap_bytes, costs.loc[target, "auxiliary_bytes"])
            item = state["fine_controls"][str(int(row.matched_cap_bytes))]
            actual = Path(item["path"]).stat().st_size
            self.assertEqual(row.auxiliary_bytes, actual)
            n = prior._load_fine(item["path"])[3].shape[0]
            self.assertEqual(actual, 64+4*n*n)
            self.assertLessEqual(actual, row.matched_cap_bytes)
            self.assertGreater(64+4*(n+1)**2, row.matched_cap_bytes)

    def test_hierarchical_adapter_uses_its_own_child_codec_and_budget(self):
        from ml import hierarchical_shape as hierarchy
        real_metrics = workflow._metrics

        def losing_correction(*args, **kwargs):
            result = real_metrics(*args, **kwargs)
            result["mae"] = 0. if result["method"] == "R08" else 1.
            result["mae_pp"] = 100*result["mae"]
            return result

        with mock.patch.object(workflow, "_metrics", side_effect=losing_correction):
            development = workflow.develop_experiment({**self.config, "family": "hierarchical"})
        workflow.verify_protocol_lock(development)
        hists, encoded = hierarchy.load_native(development/"native")
        quadrants = [np.load(p, allow_pickle=False) for p in sorted((self.preparation/"quadrants").glob("*.npy"))]
        children = workflow._children(self.preparation, hists, quadrants)
        costs = pd.read_csv(development/"payload_costs.csv").set_index("method")
        native = sum(p.stat().st_size for p in (development/"native").iterdir())
        for budget in (128, 256):
            payload = (development/"packets"/f"extra_{budget}.bin").read_bytes()
            self.assertEqual(payload[:8], hierarchy.MAGIC)
            self.assertEqual(payload, hierarchy.build_selection(hists, encoded, quadrants, children, budget))
            decoded = hierarchy.decode_selection(hists, encoded, payload)
            fixed = hierarchy.HEADER_BYTES+len(hists)*hierarchy.SOURCE_BYTES+(sum(hist[3].size for hist in hists)+7)//8
            self.assertEqual(len(payload), fixed+32*len(decoded.selected_ids))
            self.assertLessEqual(len(payload), budget)
        self.assertEqual(costs.loc["hybrid", "auxiliary_bytes"], native+costs.loc["hybrid", "packet_bytes"])
        arms = {arm.name: arm for arm in workflow.load_arms(development, 0)}
        for name in ("R08", "hybrid", "empirical_uniform"):
            arm = arms[name]
            answer = arm.score_fn(np.array([0., 3., 0., 3.]), 108., arm.artifact_loader())
            self.assertEqual(answer.reference, 108.)
            self.assertEqual(answer.score, 0.)

    def test_model_and_schema_bytes_expand_each_corresponding_fine_cap(self):
        with tempfile.TemporaryDirectory() as tmp:
            copied = Path(tmp)/"development"
            shutil.copytree(self.development, copied)
            metadata = []
            for name in ("R08_tinyML", "hybrid_tinyML"):
                directory = copied/"models"/name
                directory.mkdir(parents=True)
                (directory/"model.txt").write_bytes(b"portable-weights")
                (directory/"model.schema.json").write_bytes(b"portable-schema")
                size = sum(p.stat().st_size for p in directory.iterdir())
                metadata.append(dict(seed=0, method=name, model_bytes=size,
                                     model_path=str(directory/"model.schema.json")))
            prior._json(copied/"model_metadata.json", metadata)
            state = json.loads((copied/"run.json").read_text())
            packet = state["packets"][str(state["candidate_extra_budget"])]
            native = sum(p.stat().st_size for p in (copied/"native").iterdir())
            original = prior.load_histogram(copied/"original_H.hist")
            root = (original[1][0], original[1][-1], original[2][0], original[2][-1])
            caps = [(packet["auxiliary_bytes"] if info["method"] == "hybrid_tinyML" else native)+info["model_bytes"] for info in metadata]
            fine = workflow._fine_controls(caps, workflow._raw_arrays(self.preparation), root, copied/"fine", 17)
            state["fine_controls"].update({str(k): value for k, value in fine.items()})
            prior._json(copied/"run.json", state)
            arms = {arm.name: arm for arm in workflow.load_arms(copied, 0)}
            for info, cap in zip(metadata, caps):
                cost = arms[info["method"]].metadata
                self.assertEqual(cost["model_schema_and_weights_bytes"], info["model_bytes"])
                self.assertEqual(cost["auxiliary_bytes"], cap)
                self.assertEqual(arms["fine_for_"+info["method"]].metadata["matched_cap_bytes"], cap)

    def test_development_lock_detects_saved_allocation_and_gate_changes(self):
        workflow.verify_protocol_lock(self.development)
        for relative in ("packets/allocation_128.csv", "frozen_gate_cutoffs.csv", "inner_assignments.csv"):
            path = self.development/relative
            original = path.read_bytes()
            try:
                path.write_bytes(original+b"changed")
                with self.subTest(relative=relative), self.assertRaisesRegex(ValueError, "changed"):
                    workflow.verify_protocol_lock(self.development)
            finally:
                path.write_bytes(original)
        workflow.verify_protocol_lock(self.development)

    def test_all_method_scores_and_gates_exist_before_any_confirmation_labels(self):
        phases = []
        generate = prior._fresh_rectangles
        expected_methods = {arm.name for arm in workflow.load_arms(self.development, 0)}

        def confirmation():
            return next((self.development/"confirmation").glob("confirmation-*"))

        def checked_generation(*args, **kwargs):
            path = confirmation()
            self.assertTrue((path/"protocol.json").is_file())
            self.assertTrue((path/"frozen_gate_cutoffs.csv").is_file())
            self.assertFalse((path/"fresh_queries_labeled_offline.csv").exists())
            phases.append("geometry")
            return generate(*args, **kwargs)

        def source_only_scores(geometry, arms, original, seed, role):
            self.assertEqual(set(geometry), {"qId", *workflow.GEOMETRY})
            return pd.concat([geometry.assign(seed=seed, method=arm.name, split=role,
                                              H=1., score=.1, reference_count=1.) for arm in arms], ignore_index=True)

        def checked_labels(arrays, rectangles):
            path = confirmation()
            lock = json.loads((path/"fresh_score_lock.json").read_text())
            self.assertEqual(lock["geometry_sha256"], prior._sha(path/"fresh_queries_unlabeled.csv"))
            self.assertEqual(lock["scores_sha256"], prior._sha(path/"fresh_scores_before_labels.csv"))
            scores = pd.read_csv(path/"fresh_scores_before_labels.csv")
            self.assertEqual(set(scores.method), expected_methods)
            self.assertTrue(scores.groupby("method").size().eq(1000).all())
            self.assertFalse({"truth", "actual_error"} & set(scores))
            self.assertFalse((path/"fresh_queries_labeled_offline.csv").exists())
            self.assertEqual((path/"frozen_gate_cutoffs.csv").read_bytes(),
                             (self.development/"frozen_gate_cutoffs.csv").read_bytes())
            workflow.verify_protocol_lock(self.development)
            workflow.verify_confirmation_lock(path)
            phases.append("labels")
            counter = e2e.ScanCounter(arrays)
            return np.array([counter.count(rect) for rect in rectangles], dtype=np.int64), 0.

        with mock.patch.object(prior, "_fresh_rectangles", side_effect=checked_generation), \
                mock.patch.object(workflow, "_score_live", side_effect=source_only_scores), \
                mock.patch.object(prior, "_indexed_labels", side_effect=checked_labels):
            result = workflow.confirm_experiment({**self.config, "development_dir": str(self.development),
                "fresh_query_count": 1000, "force_confirmation": True, "fresh_query_seed": 314159})
        self.assertEqual(phases, ["geometry", "labels"])
        workflow.verify_confirmation_lock(result)
        path = result/"fresh_scores_before_labels.csv"
        original = path.read_bytes()
        try:
            path.write_bytes(original+b"changed")
            with self.assertRaisesRegex(ValueError, "changed"):
                workflow.verify_confirmation_lock(result)
        finally:
            path.write_bytes(original)
        lock = result/"fresh_score_lock.json"
        payload = lock.read_bytes()
        try:
            lock.unlink()
            with self.assertRaisesRegex(ValueError, "lock|missing"):
                workflow.verify_confirmation_lock(result)
        finally:
            lock.write_bytes(payload)

    def test_live_arms_execute_real_exact_fallback_under_saved_reject_gates(self):
        arms = workflow.load_arms(self.development, 0)
        rows = self.queries.set_index("qId")
        membership = self.splits[self.splits.seed == 0]
        cal_id = membership[membership.split == "calibration"].qId.iloc[0]
        test_id = membership[membership.split == "test"].qId.iloc[0]
        make_query = lambda qid: e2e.Query(int(qid), rows.loc[qid, workflow.GEOMETRY].to_numpy(float), rows.loc[qid, "truth"])

        class CountingScan(e2e.ScanCounter):
            calls = 0

            def count(self, bounds):
                self.calls += 1
                return super().count(bounds)

        original = prior.load_histogram(self.development/"original_H.hist")
        frozen_scores = workflow._score_live(rows.loc[[test_id]].reset_index()[["qId", *workflow.GEOMETRY]],
                                              arms, original, 0, "test").set_index("method")
        raw = CountingScan(workflow._raw_arrays(self.preparation))
        cutoffs = {arm.name: dict(cutoff=-np.inf, epsilon=.25, delta=.05) for arm in arms}
        with tempfile.TemporaryDirectory() as tmp:
            result = e2e.run_benchmark(queries=[make_query(test_id)], calibration_queries=[make_query(cal_id)],
                arms=arms, histogram_loader=lambda: prior.load_histogram(self.development/"original_H.hist"),
                estimate_fn=prior.histogram_count, raw_counter=raw, output_dir=tmp,
                frozen_cutoffs=cutoffs, frozen_cutoff_provenance={"test_fixture": True},
                cache_modes=("resident", "per_query"), repeats=1, truth_check_limit=0,
                measure_first_query=False)
        self.assertTrue((result.per_query.answer == result.per_query.truth).all())
        self.assertTrue(result.per_query.raw_ns.gt(0).all())
        self.assertEqual(raw.calls, 2*(len(arms)+1)+1)
        scored = result.per_query[result.per_query.method != "always_exact"]
        self.assertFalse(scored.accepted.any())
        for row in scored.itertuples(index=False):
            expected = frozen_scores.loc[row.method]
            self.assertAlmostEqual(row.H, expected.H, places=12)
            self.assertAlmostEqual(row.score, expected.score, places=12)
            self.assertAlmostEqual(row.reference, expected.reference_count, places=12)

    def test_archive_contains_frozen_evidence_and_excludes_raw_caches(self):
        archive = workflow.package_results(self.development)
        with ZipFile(archive) as zipped:
            names = zipped.namelist()
        self.assertIn("protocol.json", names)
        self.assertIn("frozen_gate_cutoffs.csv", names)
        self.assertIn("allocation_summary.csv", names)
        self.assertFalse(any("raw_cache" in Path(name).parts for name in names))
        self.assertFalse(any(name.endswith(("0.csv", "1.csv.gz")) for name in names))


if __name__ == "__main__":
    unittest.main()
