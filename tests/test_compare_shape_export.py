"""Export-only recovery checks; no Java, training, calibration, or retiming."""
from __future__ import annotations

from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from ml import compare_shape_e2e as comparison


def _hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class CompletedResultsExportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.out = root / "validated_run"
        self.out.mkdir()
        inputs = root / "frozen_inputs"
        inputs.mkdir()
        self.raw = inputs / "0.csv"
        self.raw.write_text("1\t2\n3\t4\n")
        self.model = inputs / "model.txt"
        self.model.write_text("frozen external model identity\n")
        self.protocol = dict(
            frozen_before_test_execution=True,
            no_test_threshold_selection=True,
            files_sha256={str(path): _hash(path) for path in (self.raw, self.model)},
        )
        self.protocol_file = self.out / "protocol.json"
        self._write_json(self.protocol_file, self.protocol)
        tables = {
            "summary.csv": "setup,coverage,violation,latency_s\nv1,0.5,0.0,15.625\n",
            "per_query.csv": "setup,source_qId,answer,fromSynopsis\nv1,7,12,true\nv1,8,5,false\n",
            "parity.csv": "same_H,fallback_matches_exact,stable_repeat\ntrue,true,true\n",
            "timing.csv": "setup,java_s,total_s\nv1,15.5,15.625\n",
            "setup_times.csv": "setup,loadNs\nv1,12345\n",
            "payload_costs.csv": "setup,loaded_artifact_bytes\nv1,277000\n",
            "raw_audit.csv": "file,points\n0.csv,2\n",
            "frozen_cutoffs.csv": "setup,merge_rate,cutoff\nv1,0.4,0.046907078801975435\n",
            "calibration.csv": "setup,source_qId,errorScore\nv1,1,0.03125\n",
        }
        for name, contents in tables.items():
            (self.out / name).write_text(contents)
        (self.out / "README.txt").write_text("Validated numerical results; export alone previously failed.\n")
        self.manifest = dict(
            status="failed", phase="packaging_results",
            error="[Errno 5] Input/output error",
            traceback="zipfile export failed after all numerical checks\n",
            exact_truth_parity=True, no_training=True, unchanged_H=True,
            model_weights_modified=False,
            finished_utc="2026-11-20T10:11:12+00:00", comparison_seconds=321.125,
            unique_timed_queries=2, calibration_queries=1,
            input_protocol_sha256=_hash(self.protocol_file),
        )
        self.manifest_file = self.out / "manifest.json"
        self._write_json(self.manifest_file, self.manifest)
        self.archive = self.out / "09b_comparison_results.zip"

    @staticmethod
    def _write_json(path, value):
        path.write_text(json.dumps(value, indent=2) + "\n")

    def _science_snapshot(self):
        return {path.relative_to(self.out).as_posix(): path.read_bytes()
                for path in self.out.rglob("*")
                if path.is_file() and path.name != "manifest.json" and path.suffix.lower() != ".zip"}

    def _forbid_experiment_execution(self, stack):
        mocks = []
        for target in ("subprocess.run", "_run_java", "calibrated_cutoff", "cutoff", "run_comparison"):
            mocks.append(stack.enter_context(patch(
                "ml.compare_shape_e2e." + target,
                side_effect=AssertionError("Export must not execute or recalibrate the experiment"))))
        return mocks

    def _assert_refused_unchanged(self, exception, pattern=None):
        manifest_before = self.manifest_file.read_bytes()
        science_before = self._science_snapshot()
        with ExitStack() as stack:
            self._forbid_experiment_execution(stack)
            context = self.assertRaisesRegex(exception, pattern) if pattern else self.assertRaises(exception)
            with context:
                comparison.package_results(self.out)
        self.assertEqual(self.manifest_file.read_bytes(), manifest_before)
        self.assertEqual(self._science_snapshot(), science_before)
        self.assertFalse(self.archive.exists())

    def test_legacy_zip_failure_recovers_without_rerun_or_changed_measurements(self):
        science_before = self._science_snapshot()
        input_hashes = {str(path): _hash(path) for path in (self.raw, self.model)}
        self.archive.write_bytes(b"incomplete ZIP from failed Volume random write")
        with ExitStack() as stack:
            execution_mocks = self._forbid_experiment_execution(stack)
            returned = comparison.package_results(self.out, overwrite=True)
        for mock in execution_mocks:
            mock.assert_not_called()
        self.assertEqual(returned, self.archive)
        self.assertEqual(self._science_snapshot(), science_before)
        self.assertEqual({str(path): _hash(path) for path in (self.raw, self.model)}, input_hashes)
        state = json.loads(self.manifest_file.read_text())
        self.assertEqual((state["status"], state["phase"]), ("complete", "complete"))
        for name in ("finished_utc", "comparison_seconds", "unique_timed_queries", "calibration_queries"):
            self.assertEqual(state[name], self.manifest[name])
        self.assertNotIn("error", state)
        self.assertNotIn("traceback", state)
        self.assertEqual(len(state["export_recoveries"]), 1)
        recovery = state["export_recoveries"][0]
        self.assertEqual(recovery["previous_status"], "failed")
        self.assertEqual(recovery["previous_phase"], "packaging_results")
        self.assertEqual(recovery["previous_error"], self.manifest["error"])
        self.assertEqual(recovery["previous_traceback"], self.manifest["traceback"])
        with zipfile.ZipFile(self.archive) as archive:
            self.assertIsNone(archive.testzip())
            self.assertNotIn(self.archive.name, archive.namelist())
            for name, content in science_before.items():
                self.assertEqual(archive.read(name), content)
            self.assertEqual(json.loads(archive.read("manifest.json")), state)

    def test_unfinished_or_unverified_runs_cannot_be_marked_complete(self):
        cases = (("status", "running"), ("exact_truth_parity", False), ("no_training", False),
                 ("unchanged_H", False), ("model_weights_modified", True))
        for field, value in cases:
            with self.subTest(field=field):
                self._write_json(self.manifest_file, dict(self.manifest, **{field: value}))
                self._assert_refused_unchanged(ValueError, "completed numerical validation")

    def test_protocol_hash_tampering_is_rejected(self):
        self.protocol_file.write_text(self.protocol_file.read_text() + "\n")
        self._assert_refused_unchanged(ValueError, "Frozen protocol hash differs")

    def test_protocol_requires_frozen_calibration_and_input_identities(self):
        for field, value in (("frozen_before_test_execution", False),
                             ("no_test_threshold_selection", False), ("files_sha256", {})):
            with self.subTest(field=field):
                self._write_json(self.protocol_file, dict(self.protocol, **{field: value}))
                self._write_json(self.manifest_file, dict(
                    self.manifest, input_protocol_sha256=_hash(self.protocol_file)))
                self._assert_refused_unchanged(ValueError, "Missing frozen test/calibration protocol")

    def test_changed_raw_or_model_identity_is_rejected(self):
        for path in (self.raw, self.model):
            with self.subTest(path=path.name):
                original = path.read_bytes()
                path.write_bytes(original + b"changed\n")
                self._assert_refused_unchanged(ValueError, "Frozen input identity changed")
                self.assertEqual(path.read_bytes(), original + b"changed\n")
                path.write_bytes(original)

    def test_failed_parity_cannot_be_overridden_by_manifest_flag(self):
        parity = self.out / "parity.csv"
        for column in ("same_H", "fallback_matches_exact", "stable_repeat"):
            with self.subTest(column=column):
                columns = ["same_H", "fallback_matches_exact", "stable_repeat"]
                values = ["false" if name == column else "true" for name in columns]
                parity.write_text(",".join(columns) + "\n" + ",".join(values) + "\n")
                self._assert_refused_unchanged(ValueError, "Numerical parity check failed: " + column)

    def test_missing_or_empty_saved_tables_cannot_be_recovered(self):
        for name in ("summary.csv", "per_query.csv", "timing.csv", "setup_times.csv",
                     "payload_costs.csv", "raw_audit.csv", "frozen_cutoffs.csv", "calibration.csv", "parity.csv"):
            path = self.out / name
            original = path.read_bytes()
            with self.subTest(name=name):
                path.unlink()
                self._assert_refused_unchanged(FileNotFoundError)
            path.write_bytes(original)
        (self.out / "summary.csv").write_text("setup,coverage,violation,latency_s\n")
        self._assert_refused_unchanged(ValueError, "Incomplete result table: summary.csv")

    def test_missing_readme_is_not_mistaken_for_just_a_zip_failure(self):
        (self.out / "README.txt").unlink()
        self._assert_refused_unchanged(FileNotFoundError, "Missing README.txt")

    def test_existing_archive_requires_explicit_overwrite(self):
        self.archive.write_bytes(b"existing export")
        manifest_before = self.manifest_file.read_bytes()
        science_before = self._science_snapshot()
        with self.assertRaisesRegex(FileExistsError, "overwrite=True"):
            comparison.package_results(self.out)
        self.assertEqual(self.archive.read_bytes(), b"existing export")
        self.assertEqual(self.manifest_file.read_bytes(), manifest_before)
        self.assertEqual(self._science_snapshot(), science_before)

    def test_publish_failure_restores_original_failed_manifest(self):
        science_before = self._science_snapshot()

        def fail_publish(out, *, overwrite=False):
            # Confirm publication was given the completion manifest, then fail.
            state = json.loads((out / "manifest.json").read_text())
            self.assertEqual(state["status"], "complete")
            self.archive.write_bytes(b"partial copied ZIP")
            raise OSError(5, "Simulated sequential Volume write failure")

        with ExitStack() as stack:
            execution_mocks = self._forbid_experiment_execution(stack)
            publisher = stack.enter_context(patch(
                "ml.compare_shape_e2e.package_directory", side_effect=fail_publish))
            with self.assertRaisesRegex(OSError, "sequential Volume write failure"):
                comparison.package_results(self.out, overwrite=True)
        publisher.assert_called_once_with(self.out, overwrite=True)
        for mock in execution_mocks:
            mock.assert_not_called()
        self.assertEqual(json.loads(self.manifest_file.read_text()), self.manifest)
        self.assertEqual(self._science_snapshot(), science_before)


if __name__ == "__main__":
    unittest.main()
