"""Opt-in actual JVM/Spark/LightGBM checks; fixtures are not team models."""
import csv
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest

from experiments.latency_benchmark.backends import JavaHistograms, SparkExact
from experiments.latency_benchmark.runner import Query, run


def fixture_features(q, estimate, histogram, sources):
    # Loader contract test only. NEVER used as a team feature implementation.
    return {"width": q.xmax - q.xmin, "estimate": estimate}


@unittest.skipUnless(importlib.util.find_spec("lightgbm"), "optional LightGBM package absent")
class BoosterContractTests(unittest.TestCase):
    def test_real_booster_load_prediction_and_schema_guard(self):
        import lightgbm as lgb
        import numpy as np
        from experiments.latency_benchmark.lightgbm_adapter import load
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "fixture-model.txt"
            data = lgb.Dataset(np.array([[1., 1.], [2., 2.], [3., 1.], [4., 2.]]),
                               label=[0., 0.1, 0.5, 1.], feature_name=["width", "estimate"])
            model = lgb.train(dict(objective="regression", verbosity=-1, num_threads=1,
                                   min_data_in_leaf=1, num_leaves=2), data, num_boost_round=2)
            model.save_model(str(path))
            options = dict(model_path=str(path), feature_names=["width", "estimate"],
                           feature_callable=__name__ + ":fixture_features")
            adapter = load(options)
            features = adapter.features(Query("q", 0, 0, 1, 1), 1, None, [])
            self.assertAlmostEqual(float(adapter.predict(features)), float(model.predict(features)[0]))
            self.assertEqual(len(adapter.metadata["model_sha256"]), 64)
            options["feature_names"] = ["estimate", "width"]
            with self.assertRaisesRegex(ValueError, "order"):
                load(options)


@unittest.skipUnless(os.environ.get("SYNOPSISLAKE_TEST_JAR") and importlib.util.find_spec("pyspark"),
                     "set SYNOPSISLAKE_TEST_JAR and install optional pyspark")
class SparkIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from pyspark.sql import SparkSession
        cls.spark = (SparkSession.builder.master("local[2]").appName("latency-correctness-tests")
                     .config("spark.jars", os.environ["SYNOPSISLAKE_TEST_JAR"])
                     .config("spark.ui.enabled", "false")
                     .config("spark.ui.showConsoleProgress", "false")
                     .config("spark.sql.shuffle.partitions", "2").getOrCreate())
        cls.spark.sparkContext.setLogLevel("ERROR")

    @classmethod
    def tearDownClass(cls):
        cls.spark.stop()

    def test_spark_raw_weighted_closed_predicate_and_invalid_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "raw.csv"
            path.write_text("x,y,w\n0,0,2\n1,1,3\n2,2,4\n")
            cfg = {"files": [{"path": str(path), "header": True, "weight_col": 2}], "cache": "persist"}
            executor = SparkExact(cfg)
            try:
                self.assertEqual(executor.count(Query("q", 0, 0, 1, 1)), 5)
                self.assertEqual(executor.count(Query("q", 3, 3, 4, 4)), 0)
            finally:
                executor.close()
            bad = Path(tmp) / "bad.csv"
            bad.write_text("NaN,1\n")
            executor = SparkExact({"files": [{"path": str(bad)}]})
            try:
                with self.assertRaises(ValueError):
                    executor.count(Query("q", 0, 0, 1, 1))
            finally:
                executor.close()

    def test_real_spark_bridge_raw_fallback_end_to_end(self):
        fixture = Path(__file__).resolve().parents[1] / "fixtures"
        with tempfile.TemporaryDirectory() as tmp:
            config = dict(bridge="spark", histogram=str(fixture / "histogram.json"),
                          queries=str(fixture / "queries.csv"), output=str(Path(tmp) / "out"),
                          exact={"backend": "spark", "cache": "none",
                                 "files": [{"path": str(fixture / "points.csv")}]},
                          methods=["always_exact", "skewness"], repeats=1, warmup_queries=0,
                          experiment_kind="correctness_smoke_not_research")
            path = Path(tmp) / "config.json"
            path.write_text(json.dumps(config))
            run(path)
            with (Path(tmp) / "out" / "queries.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            partial = next(r for r in rows if r["method"] == "skewness" and r["query_id"] == "test_partial")
            self.assertEqual(partial["answer_source"], "raw_scan")
            self.assertEqual(float(partial["answer"]), 1)
            full = next(r for r in rows if r["method"] == "skewness" and r["query_id"] == "test_full_cell")
            self.assertEqual(full["answer_source"], "histogram")
            self.assertEqual(float(full["answer"]), 2)


if __name__ == "__main__":
    unittest.main()
