"""Direct export preparation guards; synthetic data establish mechanics only."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd

from ml import bounded_sidecar as bounded
from ml import local_shape as shape
from ml import local_shape_workflow as workflow
from ml import sidecar_workflow as prior
from tests.test_sidecar_workflow import fixture


def direct_fixture(root):
    config, queries, splits = fixture(root)
    queries.qId = 10+3*queries.qId
    splits.qId = 10+3*splits.qId
    zero = pd.DataFrame([dict(qId=999, x0=4., x1=5., y0=4., y1=5., truth=0)])
    pd.concat([queries.drop(columns="cell"), zero], ignore_index=True).to_csv(config["queries_file"], index=False)
    splits.to_csv(config["split_file"], index=False)
    config.update(seeds=[0], chunk_size=17, parity_queries=8,
                  output_root=str(root/"direct-results"))
    return config, queries, splits


class DirectInputPreparation(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.config, self.queries, self.splits = direct_fixture(self.root)

    def tearDown(self):
        self.temporary.cleanup()

    def prepare(self, **overrides):
        from ml import shape_inputs
        return shape_inputs.prepare_direct_inputs({**self.config, **overrides})

    def test_direct_exports_prepare_native_and_private_raw_without_r09(self):
        with mock.patch.object(prior, "run_experiment", side_effect=AssertionError("R09 must not run")):
            path = self.prepare(budget_bytes=0)
        _, state = workflow._load_preparation(path)
        self.assertEqual(state["status"], "prepared")
        self.assertEqual(state["study"], "local-shape-preparation")
        self.assertEqual(state["raw_rows"], 108)
        hists, encoded = shape.load_native(path/"native")
        audit = pd.read_csv(path/"source_audit.csv")
        arrays = workflow._raw_arrays(path)
        self.assertEqual(len(audit), 2)
        self.assertTrue(audit.quadrant_mass_matches.all())
        self.assertTrue(audit.raw_sha256.str.fullmatch(r"[0-9a-f]{64}").all())
        for sid, (hist, actual_encoded, points) in enumerate(zip(hists, encoded, arrays)):
            counts, sums, _ = prior._sufficient(points, hist)
            np.testing.assert_array_equal(counts, hist[3].ravel())
            np.testing.assert_array_equal(actual_encoded, bounded.encode_moments(prior._moments(counts, sums)))
            quadrants = np.load(path/"quadrants"/f"source_{sid}.npy", allow_pickle=False)
            np.testing.assert_array_equal(quadrants, shape.quadrant_counts_from_points(points, hist))
            np.testing.assert_array_equal(quadrants.sum(axis=1), hist[3].ravel())
        self.assertEqual(state["native_bytes"], sum(p.stat().st_size for p in (path/"native").iterdir()))
        parity = pd.read_csv(path/"raw_truth_parity.csv")
        self.assertEqual(len(parity), 8)
        self.assertTrue(parity.matches.all())

    def test_legacy_eligible_ids_preserved_and_small_exclusion_saved(self):
        path = self.prepare()
        retained = pd.read_csv(path/"queries_offline.csv")
        self.assertEqual(set(retained.qId), set(self.queries.qId))
        self.assertNotIn(999, set(retained.qId))
        outer = pd.read_csv(path/"outer_splits.csv")
        self.assertEqual(set(outer.qId), set(retained.qId))
        excluded = pd.read_csv(path/"queries_excluded_from_inherited_cohort.csv")
        self.assertEqual(excluded.qId.tolist(), [999])
        self.assertEqual(excluded.truth.tolist(), [0])

    def test_raw_and_quadrant_hashes_reject_tampering(self):
        path = self.prepare()
        quadrant = path/"quadrants"/"source_0.npy"
        original = quadrant.read_bytes()
        quadrant.write_bytes(original+b"changed")
        with self.assertRaisesRegex(ValueError, "changed"):
            workflow._load_preparation(path)
        quadrant.write_bytes(original)
        cache = path/"raw_cache"/"source_0.npy"
        raw_original = cache.read_bytes()
        cache.write_bytes(raw_original+b"changed")
        with self.assertRaisesRegex(ValueError, "changed"):
            workflow._raw_arrays(path)

    def test_wrong_exported_source_counts_are_rejected(self):
        source = Path(self.config["histogram_dir"])/"src_0.txt"
        lines = source.read_text().splitlines()
        # Keep total mass unchanged so the exact per-cell audit must catch it.
        lines[3] = "7,5,6"
        source.write_text("\n".join(lines)+"\n")
        with self.assertRaisesRegex(ValueError, "[Mm]ass|[Cc]ount|rows"):
            self.prepare()

    def test_wrong_query_truth_is_rejected_by_raw_audit(self):
        queries = pd.read_csv(self.config["queries_file"])
        queries.loc[queries.qId != 999, "truth"] = 13
        queries.to_csv(self.config["queries_file"], index=False)
        with self.assertRaisesRegex(ValueError, "truth|parity"):
            self.prepare()

    def test_grouped_inherited_roles_are_required_and_disjoint(self):
        with self.assertRaisesRegex((ValueError, FileNotFoundError), "split|group|cell"):
            self.prepare(split_file="")
        bad = self.splits.drop(columns="cell")
        bad.to_csv(self.config["split_file"], index=False)
        with self.assertRaisesRegex(ValueError, "group|cell"):
            self.prepare()
        bad = self.splits.copy()
        bad.loc[(bad.seed == 0) & (bad.qId == self.queries.qId.iloc[6]), "cell"] = "cell_0"
        bad.to_csv(self.config["split_file"], index=False)
        with self.assertRaisesRegex(ValueError, "overlap|group|cell"):
            self.prepare()

    def test_eligible_query_cannot_be_silently_omitted_from_inherited_splits(self):
        split = pd.read_csv(self.config["split_file"])
        split = split[split.qId != self.queries.qId.iloc[0]]
        split.to_csv(self.config["split_file"], index=False)
        with self.assertRaisesRegex(ValueError, "eligible|omit|min_exact"):
            self.prepare()

    def test_missing_split_groups_can_come_from_identity_matched_query_metadata(self):
        self.splits.drop(columns="cell").to_csv(self.config["split_file"], index=False)
        metadata = pd.concat([self.queries[["qId", "cell"]],
                              pd.DataFrame([dict(qId=999, cell="outside")])], ignore_index=True)
        metadata_path = self.root/"query_metadata.csv"
        metadata.to_csv(metadata_path, index=False)
        path = self.prepare(query_metadata_file=str(metadata_path))
        outer = pd.read_csv(path/"outer_splits.csv")
        expected = self.splits[self.splits.seed == 0].sort_values("qId").reset_index(drop=True)
        actual = outer[outer.seed == 0].sort_values("qId").reset_index(drop=True)
        np.testing.assert_array_equal(actual.cell, expected.cell)

    def test_r08_confirmation_config_resolves_direct_inputs(self):
        from ml import shape_inputs
        inherited = self.root/"r08"
        inherited.mkdir()
        (inherited/"confirmation_config.json").write_text(json.dumps(self.config))
        path = shape_inputs.prepare_direct_inputs(dict(run_dir=str(inherited),
                                                      output_root=str(self.root/"from-r08"),
                                                      seeds=[0], parity_queries=8))
        _, state = workflow._load_preparation(path)
        self.assertEqual(state["raw_rows"], 108)
        self.assertEqual(set(pd.read_csv(path/"queries_offline.csv").qId), set(self.queries.qId))

    def test_r08_inherited_raw_identity_rejects_changed_coordinates_before_parsing(self):
        from ml import shape_inputs
        inherited = self.root/"r08-identity"
        inherited.mkdir()
        (inherited/"confirmation_config.json").write_text(json.dumps(self.config))
        raw = Path(self.config["raw_data_dir"])/"0.csv"
        (inherited/"raw_input_manifest.json").write_text(json.dumps([
            dict(source_id=0, raw_sha256=prior._sha(raw))]))
        # A coordinate moves inside its existing parent cell; counts still fit.
        lines = raw.read_text().splitlines()
        fields = lines[0].split("\t")
        fields[0] = "0.11"
        lines[0] = "\t".join(fields)
        raw.write_text("\n".join(lines)+"\n")
        with mock.patch.object(prior, "iter_raw", side_effect=AssertionError("Changed source must fail before parsing")):
            with self.assertRaisesRegex(ValueError, "SHA|identity"):
                shape_inputs.prepare_direct_inputs(dict(run_dir=str(inherited),
                                                       output_root=str(self.root/"bad-identity"), seeds=[0]))


if __name__ == "__main__":
    unittest.main()
