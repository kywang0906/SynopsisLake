import csv
import json
import math
from pathlib import Path
import tempfile
import unittest

from experiments.latency_benchmark.backends import ScanExact
from experiments.latency_benchmark.runner import (
    Query, calibrate_cutoff, execute, read_queries, relative_error, run,
)


# Deliberately test doubles, never available as a CLI substitute for team ML.
FEATURE_CALLS = []


class TestModel:
    def features(self, q, estimate, hist, sources):
        FEATURE_CALLS.append(q.query_id)
        assert not hasattr(q, "truth") and not hasattr(q, "split")
        return hist.score(*q.bounds)

    def predict(self, features):
        return features


def test_model_factory(options):
    return TestModel()


class TestHist:
    def __init__(self, obj):
        self.obj = obj

    def initSkewness(self, scale, epsilon):
        pass

    def estimate(self, xmin, ymin, xmax, ymax):
        # Independent tiny geometric oracle for testing harness plumbing only.
        x, y, data = (self.obj[k] for k in ("x_edges", "y_edges", "counts"))
        return sum(data[i][j] * max(0, min(xmax, x[i+1]) - max(xmin, x[i]))
                   * max(0, min(ymax, y[j+1]) - max(ymin, y[j]))
                   / ((x[i+1]-x[i]) * (y[j+1]-y[j]))
                   for i in range(len(data)) for j in range(len(data[i])))

    def score(self, xmin, ymin, xmax, ymax):
        return 0.0 if xmax - xmin >= 1 else 1.0


class TestLoader:
    loads = 0

    def __init__(self, cfg):
        pass

    def load(self, path):
        type(self).loads += 1
        return TestHist(json.loads(Path(path).read_text()))


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.fixture = Path(__file__).resolve().parents[1] / "fixtures"

    def tearDown(self):
        self.tmp.cleanup()

    def config(self, **changes):
        cfg = dict(queries=str(self.fixture / "queries.csv"),
                   histogram=str(self.fixture / "histogram.json"), output=str(self.root / "results"),
                   exact={"backend": "scan", "files": [{"path": str(self.fixture / "points.csv")}]},
                   methods=["always_exact", "ml", "skewness"], repeats=2, warmup_queries=0,
                   ml={"factory": __name__ + ":test_model_factory"},
                   histogram_cache="resident", epsilon=0.25, delta=0.05)
        cfg.update(changes)
        path = self.root / "config.json"
        path.write_text(json.dumps(cfg))
        return path

    def test_calibration_includes_ties_and_can_recover_after_bad_prefix(self):
        self.assertIsNone(calibrate_cutoff([(0, False), (0, True)], 0.05))
        self.assertEqual(calibrate_cutoff([(0, True), (1, False)], 0.5), 1)
        self.assertIsNone(calibrate_cutoff([(math.nan, False), (1, True)], 0.05))
        self.assertIsNone(calibrate_cutoff([], 0.05))

    def test_zero_policy(self):
        self.assertEqual(relative_error(0, 0), 0)
        self.assertEqual(relative_error(1, 0), math.inf)
        self.assertEqual(relative_error(2, 4), 0.5)

    def test_raw_scan_is_closed_and_weighted(self):
        path = self.root / "points.csv"
        path.write_text("x;y;weight\n0;0;2\n1;1;3\n2;2;4\n")
        scan = ScanExact({"files": [{"path": str(path), "delimiter": ";", "header": True,
                                     "weight_col": 2}]})
        self.assertEqual(scan.count(Query("q", 0, 0, 1, 1)), 5)
        self.assertEqual(scan.count(Query("q", 3, 3, 4, 4)), 0)

    def test_fallback_reads_modified_raw_file_not_old_truth(self):
        path = self.root / "raw.csv"
        path.write_text("0.1,0.1\n")
        scan = ScanExact({"files": [{"path": str(path)}]})
        q = Query("q", 0, 0, 1, 1)
        self.assertEqual(execute("always_exact", q, scan)["answer"], 1)
        path.write_text("0.1,0.1\n0.2,0.2\n")
        class Store:
            def get(self):
                return TestHist({"x_edges": [0, 1], "y_edges": [0, 1], "counts": [[100]]}), []
        row = execute("ml", q, scan, Store(), TestModel(), cutoff=None)
        self.assertEqual(row["answer"], 2)
        self.assertEqual(row["answer_source"], "raw_scan")
        self.assertGreater(row["exact_ns"], 0)

    def test_runner_no_leakage_real_exact_and_repeat_sample_sizes(self):
        FEATURE_CALLS.clear()
        summaries = run(self.config(), loader_factory=TestLoader)
        out = self.root / "results"
        with (out / "queries.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        ml = [r for r in rows if r["method"] == "ml"]
        self.assertEqual(len(ml), 6)
        accepted = [r for r in ml if r["accepted"] == "True"]
        self.assertEqual(len(accepted), 4)
        self.assertTrue(all(int(r["exact_ns"]) == 0 for r in accepted))
        fallback = [r for r in ml if r["accepted"] == "False"]
        self.assertTrue(all(float(r["answer"]) == 1 for r in fallback))
        self.assertTrue(all(r["answer_source"] == "raw_scan" for r in fallback))
        self.assertNotIn("unused_train", FEATURE_CALLS)
        # No test features are extracted until calibration's complete pass ends.
        self.assertEqual(FEATURE_CALLS[:3], ["cal_full", "cal_partial", "cal_zero"])
        summary = next(r for r in summaries if r["method"] == "ml" and r["repeat"] == "all")
        self.assertEqual(summary["measurements"], 6)
        self.assertEqual(summary["accepted_queries"], 2)
        self.assertEqual(summary["unique_queries"], 3)
        self.assertEqual(summary["violation_rate"], 0)
        meta = json.loads((out / "run.json").read_text())
        self.assertEqual(meta["cutoffs"], {"ml": 0, "skewness": 0})
        self.assertIsNone(meta["training_seconds"])

    def test_cache_modes_have_same_answers_and_different_artifact_reads(self):
        TestLoader.loads = 0
        a = run(self.config(), loader_factory=TestLoader)
        resident_reads = TestLoader.loads
        TestLoader.loads = 0
        b = run(self.config(histogram_cache="per_query", output=str(self.root / "perquery")),
                loader_factory=TestLoader)
        self.assertGreater(TestLoader.loads, resident_reads)
        self.assertEqual([r["acceptance_rate"] for r in a], [r["acceptance_rate"] for r in b])

    def test_test_violations_are_reported_without_retuning_cutoff(self):
        queries = self.root / "changed-test.csv"
        queries.write_text((self.fixture / "queries.csv").read_text().replace(
            "test_full_cell,0,0,1,1,test", "test_full_cell,0.1,0.1,1.1,1.1,test"))
        result = run(self.config(queries=str(queries)), loader_factory=TestLoader)
        meta = json.loads((self.root / "results" / "run.json").read_text())
        self.assertEqual(meta["cutoffs"]["ml"], 0)
        summary = next(r for r in result if r["method"] == "ml" and r["repeat"] == "all")
        self.assertEqual(summary["violating_accepted_queries"], 1)
        self.assertGreater(summary["violation_rate"], 0.05)
        self.assertFalse(summary["quality_target_met"])

    def test_duplicate_query_ids_rejected(self):
        path = self.root / "queries.csv"
        path.write_text("query_id,xmin,ymin,xmax,ymax,split\na,0,0,1,1,calibration\na,0,0,1,1,test\n")
        with self.assertRaisesRegex(ValueError, "unique"):
            read_queries(path)

    def test_missing_ml_fails_instead_of_substituting_model(self):
        with self.assertRaisesRegex(ValueError, "ML pipeline absent"):
            run(self.config(ml={}), loader_factory=TestLoader)

    def test_truncated_raw_data_and_invalid_weights_fail(self):
        path = self.root / "raw.csv"
        path.write_text("0,0\n")
        with self.assertRaisesRegex(ValueError, "shorter"):
            ScanExact({"files": [{"path": str(path), "data_rows": 2}]}).count(Query("q", 0, 0, 1, 1))
        path.write_text("0,0,-2\n")
        with self.assertRaises(ValueError):
            ScanExact({"files": [{"path": str(path), "weight_col": 2}]}).count(Query("q", 0, 0, 1, 1))

    def test_refuses_to_overwrite_completed_run(self):
        cfg = self.config(methods=["always_exact"])
        run(cfg)
        with self.assertRaises(FileExistsError):
            run(cfg)


if __name__ == "__main__":
    unittest.main()
