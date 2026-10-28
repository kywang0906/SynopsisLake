"""Load the team's existing Booster AND existing online feature extractor.

No substitute features, training, resplitting, or test-based tuning occurs here.
The extractor must return a dict keyed by the model's exact feature names.
"""
import hashlib
import importlib
from pathlib import Path


def load(options):
    import lightgbm as lgb
    import numpy as np
    model_path = Path(options["model_path"])
    module, name = options["feature_callable"].split(":")
    module = importlib.import_module(module)
    extract = getattr(module, name)
    booster = lgb.Booster(model_file=str(model_path))
    names = options["feature_names"]
    if names != booster.feature_name():
        raise ValueError("feature_names/order must match the existing model exactly")
    threads = options.get("num_threads", 1)
    if not isinstance(threads, int) or threads < 1:
        raise ValueError("num_threads must be a positive integer")

    class Adapter:
        metadata = dict(lightgbm_version=lgb.__version__, numpy_version=np.__version__,
                        model_sha256=hashlib.sha256(model_path.read_bytes()).hexdigest(),
                        feature_names=names, num_threads=threads,
                        feature_module_sha256=hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest())

        def features(self, query, estimate, histogram, sources):
            values = extract(query, estimate, histogram, sources)
            if set(values) != set(names):
                raise ValueError("online features differ from model schema")
            return np.array([[values[name] for name in names]], dtype=float)

        def predict(self, features):
            prediction = booster.predict(features, num_threads=threads)
            if prediction.shape != (1,):
                raise ValueError("expected one scalar error/risk score per query")
            return prediction[0]

    return Adapter()
