"""Protocol/persistence checks for the R10 tiny-ML control, not accuracy claims."""
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd

from ml import shape_error_model as models


def training_table():
    rng = np.random.default_rng(39)
    count = 120
    width, height = rng.uniform(.01, .2, (2, count))
    frame = pd.DataFrame(dict(
        log_H=rng.uniform(1., 6., count), log_reference=rng.uniform(1., 6., count),
        proxy_score=rng.uniform(0., 1., count), query_width=width,
        query_height=height, query_area=width * height, query_aspect=width / height,
        partial_mass_fraction=rng.uniform(0., 1., count),
        log_base_reference=rng.uniform(1., 6., count),
        base_proxy_score=rng.uniform(0., 1., count),
        reference_shift=rng.uniform(-.5, .5, count),
        corrected_mass_fraction=rng.uniform(0., 1., count),
        actual_error=rng.uniform(0., 2., count),
        split="train", cell=np.repeat([f"cell_{i}" for i in range(20)], 6),
        H=rng.uniform(10., 500., count), qId=np.arange(count),
        SA=rng.uniform(10., 500., count), truth=20, sourceId=np.arange(count) % 4))
    return frame


class InputProtocolGuards(unittest.TestCase):
    def setUp(self):
        self.frame = training_table()

    def reject_without_fit(self, frame, names=models.BASE_FEATURES, groups=None, message="", **kwargs):
        with mock.patch.object(models, "_lightgbm") as backend:
            with self.assertRaisesRegex(ValueError, message):
                models.fit_models(frame, names, 7, "/tmp/unwritten-r10-model", groups=groups, **kwargs)
            backend.assert_not_called()

    def test_nontraining_rows_cannot_reach_backend(self):
        for role in ("calibration", "test", "fresh", None):
            frame = self.frame.copy()
            frame.loc[0, "split"] = role
            with self.subTest(role=role):
                self.reject_without_fit(frame, message="outer train rows only")

    def test_targets_identities_and_unrecognized_features_are_rejected(self):
        forbidden = ("SA", "truth", "ground_truth", "actual_error", "qId", "sourceId",
                     "cell", "label", "renamed_offline_label", "x0")
        for name in forbidden:
            with self.subTest(feature=name):
                self.reject_without_fit(self.frame, ["proxy_score", name],
                                        message="Unsupported or leaking")

    def test_feature_count_duplicates_and_column_presence_are_checked(self):
        for names in ([], ["proxy_score"] * 2, list(models.HYBRID_FEATURES) + ["extra"],
                      "proxy_score"):
            with self.subTest(names=names):
                self.reject_without_fit(self.frame, names)
        self.reject_without_fit(self.frame.drop(columns="log_H"), message="Missing model features")
        duplicate = pd.concat([self.frame, self.frame[["log_H"]]], axis=1)
        self.reject_without_fit(duplicate, message="unique columns")

    def test_nonfinite_features_and_invalid_targets_cannot_reach_backend(self):
        for column, value in (("proxy_score", np.inf), ("log_H", np.nan),
                              ("actual_error", np.inf), ("actual_error", -.1),
                              ("actual_error", np.nan)):
            frame = self.frame.copy()
            frame.loc[0, column] = value
            with self.subTest(column=column, value=value):
                self.reject_without_fit(frame, message="finite")

    def test_invalid_grouping_is_rejected_without_row_split_fallback(self):
        for groups in (["same"] * len(self.frame),
                       [None] + list(self.frame.cell.iloc[1:]),
                       [1, 2], "missing_group"):
            with self.subTest(groups=str(groups)[:30]):
                self.reject_without_fit(self.frame, groups=groups, message="group")
        frame = self.frame.copy()
        frame["cell"] = "only_one_cell"
        self.reject_without_fit(frame, message="two distinct groups")
        self.reject_without_fit(self.frame.drop(columns="cell"), message="requires groups")

    def test_missing_lightgbm_reports_required_backend(self):
        with mock.patch.object(models.importlib, "import_module", side_effect=ImportError("missing")):
            with self.assertRaisesRegex(ImportError, "requires LightGBM; no alternative backend"):
                models._lightgbm()

    def test_explicit_inner_indices_must_be_a_group_disjoint_integer_partition(self):
        fit, val = np.arange(96), np.arange(96, 120)
        invalid = (
            dict(inner_fit_indices=fit),
            dict(inner_validation_indices=val),
            dict(inner_fit_indices=[], inner_validation_indices=val),
            dict(inner_fit_indices=fit.astype(float), inner_validation_indices=val),
            dict(inner_fit_indices=np.ones(120, dtype=bool), inner_validation_indices=val),
            dict(inner_fit_indices=np.r_[fit, 0], inner_validation_indices=val),
            dict(inner_fit_indices=fit, inner_validation_indices=np.arange(96, 121)),
            dict(inner_fit_indices=np.arange(100), inner_validation_indices=val),
            dict(inner_fit_indices=np.arange(90), inner_validation_indices=val),
            dict(inner_fit_indices=np.arange(95), inner_validation_indices=np.arange(95, 120)),
        )
        for kwargs in invalid:
            with self.subTest(arguments=str(kwargs)[:50]):
                self.reject_without_fit(self.frame, message="inner", **kwargs)


class InnerValidationGuards(unittest.TestCase):
    def test_grouped_partition_is_exhaustive_disjoint_and_deterministic(self):
        groups = np.repeat([f"g{i}" for i in range(20)], 3)
        fit, val, audit = models._inner_split(len(groups), 19, groups)
        fit_again, val_again, same_audit = models._inner_split(len(groups), 19, groups)
        np.testing.assert_array_equal(fit, fit_again)
        np.testing.assert_array_equal(val, val_again)
        self.assertEqual(audit, same_audit)
        self.assertFalse(set(fit) & set(val))
        self.assertEqual(set(fit) | set(val), set(range(len(groups))))
        self.assertFalse(set(groups[fit]) & set(groups[val]))
        self.assertFalse(set(audit["fit_group_sha256"]) & set(audit["validation_group_sha256"]))
        self.assertEqual(audit["kind"], "grouped")
        # Group membership, unlike row positions, does not depend on input order.
        order = np.random.default_rng(33).permutation(len(groups))
        _, reordered_val, _ = models._inner_split(len(groups), 19, groups[order])
        self.assertEqual(set(groups[val]), set(groups[order][reordered_val]))

    def test_equal_numeric_group_ids_never_split_apart(self):
        groups = np.asarray([1, 1., 2, 2., 3, 3., 4, 4.], dtype=object)
        fit, val, _ = models._inner_split(len(groups), 0, groups)
        self.assertFalse(set(groups[fit]) & set(groups[val]))

    def test_row_split_is_explicit_deterministic_and_train_only(self):
        fit, val, audit = models._inner_split(20, 9, None)
        self.assertEqual(audit, {"kind": "deterministic_rows"})
        self.assertEqual(len(val), 4)
        self.assertFalse(set(fit) & set(val))
        np.testing.assert_array_equal(val, models._inner_split(20, 9, None)[1])


@unittest.skipUnless(importlib.util.find_spec("lightgbm"), "LightGBM is required; use benchmark venv")
class PortableModelGuards(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.frame = training_table()
        cls.directory = Path(cls.temporary.name) / "model"
        cls.info = models.fit_models(cls.frame, models.HYBRID_FEATURES, 7, cls.directory)
        cls.model = models.load_model(cls.info["model_path"])

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_frozen_candidates_use_only_grouped_outer_train_and_refit_all_train(self):
        audit = self.info["selection"]
        fit, val = audit["fit_row_positions"], audit["validation_row_positions"]
        self.assertFalse(set(fit) & set(val))
        self.assertEqual(set(fit) | set(val), set(range(len(self.frame))))
        self.assertFalse(set(self.frame.iloc[fit].cell) & set(self.frame.iloc[val].cell))
        self.assertEqual(self.info["refit_rows"], len(self.frame))
        self.assertEqual(self.info["backend"], "lightgbm")
        self.assertEqual([row["num_leaves"] for row in self.info["candidates"]], [7, 15])
        self.assertTrue(all(1 <= row["iterations"] <= 128 for row in self.info["candidates"]))
        self.assertTrue(all(np.isfinite(row["validation_mae"]) for row in self.info["candidates"]))
        expected = min(self.info["candidates"], key=lambda row:
                       (row["validation_mae"], row["num_leaves"], row["iterations"]))
        self.assertEqual(self.info["selected_num_leaves"], expected["num_leaves"])
        self.assertEqual(self.info["selected_iterations"], expected["iterations"])

    def test_numeric_scores_are_identity_target_predictions_and_leave_H_unchanged(self):
        before = self.frame.copy(deep=True)
        predictions = self.model.predict_frame(self.frame)
        raw = self.model.booster.predict(self.frame[list(models.HYBRID_FEATURES)].to_numpy(),
                                         num_threads=1)
        np.testing.assert_array_equal(predictions, np.maximum(raw, 0.))
        self.assertTrue(np.isfinite(predictions).all())
        self.assertTrue((predictions >= 0).all())
        self.assertEqual(self.model.schema["target_transform"], "identity")
        pd.testing.assert_frame_equal(self.frame, before)

    def test_live_row_and_frame_predictions_use_schema_order_and_ignore_labels(self):
        sample = self.frame.iloc[:5].copy()
        expected = self.model.predict_frame(sample)
        for i, (_, row) in enumerate(sample.iterrows()):
            self.assertAlmostEqual(self.model.predict_row(row), expected[i], places=14)
            self.assertAlmostEqual(self.model.predict_row(dict(reversed(list(row.items())))),
                                   expected[i], places=14)
        sample["SA"], sample["truth"], sample["actual_error"] = 99999, -99999, np.nan
        sample["qId"], sample["sourceId"], sample["cell"] = -1, -2, "changed"
        np.testing.assert_array_equal(self.model.predict_frame(sample[sample.columns[::-1]]), expected)
        self.assertEqual(self.model.booster.feature_name(), list(models.HYBRID_FEATURES))
        self.assertEqual(self.model.predict_frame(sample.iloc[:0]).shape, (0,))

    def test_reported_runtime_bytes_match_all_required_portable_files(self):
        self.assertEqual({path.name for path in self.directory.iterdir()},
                         {"model.txt", "model.schema.json", "inner_validation_predictions.csv"})
        total = sum((self.directory / name).stat().st_size for name in
                    ("model.txt", "model.schema.json"))
        self.assertEqual(self.info["model_bytes"], total)
        self.assertEqual(self.model.model_bytes, total)
        self.assertEqual(self.info["model_text_bytes"] + self.info["schema_bytes"], total)
        self.assertNotIn("selection", self.model.schema)
        audit = Path(self.info["validation_predictions_path"])
        self.assertEqual(self.info["validation_predictions_bytes"], audit.stat().st_size)
        self.assertNotIn("validation_predictions_path", self.model.schema)

    def test_only_text_and_schema_are_needed_for_portable_reload(self):
        with tempfile.TemporaryDirectory() as tmp:
            portable = Path(tmp)
            for name in ("model.txt", "model.schema.json"):
                shutil.copy2(self.directory / name, portable / name)
            reloaded = models.load_model(portable)
            np.testing.assert_array_equal(reloaded.predict_frame(self.frame),
                                          self.model.predict_frame(self.frame))

    def test_tampering_model_or_schema_feature_order_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            portable = Path(tmp)
            for path in self.directory.iterdir():
                shutil.copy2(path, portable / path.name)
            text = portable / "model.txt"
            original = text.read_bytes()
            text.write_bytes(original + b"\n")
            with self.assertRaisesRegex(ValueError, "checksum"):
                models.load_model(portable)
            text.write_bytes(original)
            schema_path = portable / "model.schema.json"
            schema = json.loads(schema_path.read_text())
            schema["feature_names"] = list(reversed(schema["feature_names"]))
            schema_path.write_text(json.dumps(schema))
            with self.assertRaisesRegex(ValueError, "feature order"):
                models.load_model(portable)

    def test_pre_refit_validation_audit_matches_selected_candidate_and_declared_roles(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            fit, val = np.arange(96), np.arange(119, 95, -1)
            lgb = models._lightgbm()
            real_train = lgb.train
            candidate_predictions, candidate_keys = [], []
            call_count = 0

            def tracked_train(params, dataset, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count == 3:
                    # The persisted audit must already exist before all-train fitting.
                    audit = pd.read_csv(directory / "inner_validation_predictions.csv",
                                        float_precision="round_trip")
                    chosen = min(range(2), key=lambda i: candidate_keys[i])
                    np.testing.assert_array_equal(audit.row_position, val)
                    np.testing.assert_array_equal(audit.qId, self.frame.iloc[val].qId)
                    np.testing.assert_array_equal(audit.actual_error,
                                                  self.frame.iloc[val].actual_error)
                    np.testing.assert_array_equal(audit.prediction, candidate_predictions[chosen])
                fitted = real_train(params, dataset, **kwargs)
                if call_count <= 2:
                    iteration = int(fitted.best_iteration or fitted.current_iteration())
                    prediction = np.maximum(0., fitted.predict(
                        self.frame.iloc[val][list(models.BASE_FEATURES)].to_numpy(),
                        num_iteration=iteration, num_threads=1))
                    candidate_predictions.append(prediction)
                    candidate_keys.append((float(np.mean(abs(prediction -
                                               self.frame.iloc[val].actual_error.to_numpy()))),
                                           params["num_leaves"], iteration))
                return fitted

            with mock.patch.object(lgb, "train", side_effect=tracked_train):
                info = models.fit_models(self.frame, models.BASE_FEATURES, 7, directory,
                    inner_fit_indices=fit, inner_validation_indices=val)
            self.assertEqual(call_count, 3)
            self.assertEqual(info["selection"]["kind"], "predeclared_grouped")
            self.assertEqual(info["validation_row_positions"], val.tolist())
            audit = pd.read_csv(info["validation_predictions_path"], float_precision="round_trip")
            self.assertAlmostEqual(info["validation_mae"],
                                   float(np.mean(abs(audit.prediction - audit.actual_error))), places=14)
            self.assertEqual(info["selection"]["validation_mae"], info["validation_mae"])
            self.assertEqual(info["refit_rows"], len(self.frame))

    def test_runtime_rejects_missing_or_nonfinite_features(self):
        with self.assertRaisesRegex(ValueError, "Missing model features"):
            self.model.predict_frame(self.frame.drop(columns="log_H"))
        row = self.frame.iloc[0].to_dict()
        row["log_H"] = np.inf
        with self.assertRaisesRegex(ValueError, "finite numeric"):
            self.model.predict_row(row)
        del row["log_H"]
        with self.assertRaisesRegex(ValueError, "every finite numeric"):
            self.model.predict_row(row)


if __name__ == "__main__":
    unittest.main()
