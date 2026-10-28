"""Real query executors and a thin bridge to the existing Java histogram code."""
import csv
import itertools
import json
import math
from pathlib import Path


class JavaHistograms:
    def __init__(self, config):
        if config["bridge"] == "spark":
            from pyspark.sql import SparkSession
            self.spark = SparkSession.builder.getOrCreate()
            self.bridge = self.spark._jvm.benchmark.HistogramBridge
            self.gateway = self.spark.sparkContext._gateway
            self.metadata = dict(spark_version=self.spark.version,
                                 java_version=self.spark._jvm.java.lang.System.getProperty("java.version"),
                                 spark_application_id=self.spark.sparkContext.applicationId)
        elif config["bridge"] == "jpype":
            import jpype
            if not jpype.isJVMStarted():
                jpype.startJVM(classpath=config["classpath"])
            self.bridge = jpype.JClass("benchmark.HistogramBridge")
            self.gateway = None
            self.metadata = dict(jpype_version=jpype.__version__,
                                 java_version=str(jpype.JClass("java.lang.System").getProperty("java.version")))
        else:
            raise ValueError("bridge must be spark or jpype")

    def load(self, path):
        with open(path, encoding="utf-8") as stream:
            obj = json.load(stream)
        x, y, counts = obj["x_edges"], obj["y_edges"], obj["counts"]
        if self.gateway is None:
            return self.bridge(x, y, counts)
        jdouble = self.spark._jvm.double
        def vector(values):
            out = self.gateway.new_array(jdouble, len(values))
            for i, value in enumerate(values):
                out[i] = float(value)
            return out
        matrix = self.gateway.new_array(jdouble, len(counts), len(y) - 1)
        for i, row in enumerate(counts):
            if len(row) != len(y) - 1:
                raise ValueError("ragged histogram counts")
            for j, value in enumerate(row):
                matrix[i][j] = float(value)
        return self.bridge(vector(x), vector(y), matrix)


class ScanExact:
    """Reopen and scan the original point records on EVERY exact query.

    Closed rectangles match BuildRangeAns. Optional weights match that class's
    weighted count. data_rows excludes an appended synopsis only when explicitly
    supplied by the owner; malformed/truncated records fail instead of skipping.
    """
    def __init__(self, config):
        if config.get("cache", "none") != "none":
            raise ValueError("scan executor supports cache=none only")
        self.files = config["files"]

    def count(self, query):
        total = 0.0
        for spec in self.files:
            with open(spec["path"], newline="", encoding="utf-8") as stream:
                reader = csv.reader(stream, delimiter=spec.get("delimiter", ","))
                if spec.get("header", False):
                    next(reader)
                rows = reader
                limit = spec.get("data_rows")
                if limit is not None:
                    if not isinstance(limit, int) or limit < 0:
                        raise ValueError("data_rows must be a nonnegative integer")
                    rows = itertools.islice(reader, limit)
                seen = 0
                for row in rows:
                    seen += 1
                    x = float(row[spec.get("x_col", 0)])
                    y = float(row[spec.get("y_col", 1)])
                    w = float(row[spec["weight_col"]]) if "weight_col" in spec else 1.0
                    if not all(map(math.isfinite, (x, y, w))) or w < 0:
                        raise ValueError("raw coordinates/weights must be finite; weights nonnegative")
                    if query.xmin <= x <= query.xmax and query.ymin <= y <= query.ymax:
                        total += w
                if limit is not None and seen != limit:
                    raise ValueError("raw file shorter than declared data_rows")
        return total

    def close(self):
        pass


class SparkExact:
    """Same closed-rectangle predicate; collect() executes a real Spark scan."""
    def __init__(self, config):
        from pyspark.sql import SparkSession, functions as F
        self.spark = SparkSession.builder.getOrCreate()
        self.F = F
        self.frame = None
        self.cached = config.get("cache", "none") == "persist"
        if config.get("cache", "none") not in ("none", "persist"):
            raise ValueError("spark cache must be none or persist")
        for spec in config["files"]:
            if "data_rows" in spec:
                raise ValueError("Spark needs point-only files; data_rows/tail format is scan-only")
            frame = (self.spark.read.option("header", spec.get("header", False))
                     .option("sep", spec.get("delimiter", ","))
                     .option("mode", "FAILFAST").csv(spec["path"]))
            cols = frame.columns
            x = F.col(cols[spec.get("x_col", 0)]).cast("double")
            y = F.col(cols[spec.get("y_col", 1)]).cast("double")
            w = (F.col(cols[spec["weight_col"]]).cast("double")
                 if "weight_col" in spec else F.lit(1.0))
            frame = frame.select(x.alias("x"), y.alias("y"), w.alias("w"))
            self.frame = frame if self.frame is None else self.frame.unionByName(frame)
        if self.cached:
            # Materialization is recorded as executor initialization, not hidden.
            self.frame = self.frame.persist()
            self.frame.count()
        self.metadata = dict(spark_version=self.spark.version,
                             partitions=self.frame.rdd.getNumPartitions(),
                             persisted=self.cached,
                             spark_settings={key: self.spark.sparkContext.getConf().get(key, "unset")
                                             for key in ("spark.executor.instances", "spark.executor.cores",
                                                         "spark.executor.memory", "spark.driver.memory")})

    def count(self, query):
        F = self.F
        bad = F.lit(False)
        for name in ("x", "y", "w"):
            col = F.col(name)
            bad = bad | col.isNull() | F.isnan(col) | (F.abs(col) == float("inf"))
        bad = bad | (F.col("w") < 0)
        inside = ((F.col("x") >= query.xmin) & (F.col("x") <= query.xmax)
                  & (F.col("y") >= query.ymin) & (F.col("y") <= query.ymax))
        result = self.frame.agg(
            F.sum(F.when(inside, F.col("w")).otherwise(0.0)).alias("answer"),
            F.sum(F.when(bad, 1).otherwise(0)).alias("bad")).collect()[0]
        if result["bad"]:
            raise ValueError("invalid raw coordinates/weights")
        return float(result["answer"] or 0.0)

    def close(self):
        if self.cached:
            self.frame.unpersist(blocking=True)


def exact_executor(config):
    if not config.get("files"):
        raise ValueError("exact.files must declare original point data")
    backend = config.get("backend", "scan")
    if backend == "scan":
        return ScanExact(config)
    if backend == "spark":
        return SparkExact(config)
    raise ValueError("exact backend must be scan or spark")
