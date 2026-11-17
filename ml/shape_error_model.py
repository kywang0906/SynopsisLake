"""Small, train-only LightGBM controls for the R10 error prediction study.

The learner predicts ``actual_error`` directly. It never changes H, calibrates
a gate, or receives calibration/test data. Two prespecified tree sizes are
selected on an inner validation split, then the chosen number of iterations is
refit on all supplied outer-training rows. Inference needs only the LightGBM
text file and its small JSON schema; no pickle or fitted preprocessing is used.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib
import json
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd


SCHEMA_VERSION = 1
TARGET_COLUMN = "actual_error"
MAX_FEATURES = 12
NUM_LEAVES = (7, 15)
MAX_ITERATIONS = 128
EARLY_STOPPING_ROUNDS = 16
VALIDATION_FRACTION = .2

# These are the frozen R10 features, all derived from synopsis/query state.
# Counts use log1p; query dimensions/area use normalized root geometry. IDs,
# source-cell coordinates, SA and error labels are deliberately unsupported.
BASE_FEATURES = ("log_H", "log_reference", "proxy_score", "query_width",
                 "query_height", "query_area", "query_aspect", "partial_mass_fraction")
HYBRID_FEATURES = BASE_FEATURES + ("log_base_reference", "base_proxy_score",
                                   "reference_shift", "corrected_mass_fraction")
SUPPORTED_FEATURES = frozenset(HYBRID_FEATURES)


def _lightgbm():
    try:
        return importlib.import_module("lightgbm")
    except ImportError as exc:
        raise ImportError(
            "The R10 tiny-ML control requires LightGBM; no alternative backend "
            "is substituted. Use the benchmark environment with LightGBM installed."
        ) from exc


def _feature_names(names):
    if isinstance(names, str):
        raise ValueError("feature_names must be an ordered sequence of names")
    names = list(names)
    if (not names or len(names) > MAX_FEATURES or
            any(not isinstance(name, str) for name in names) or
            len(set(names)) != len(names)):
        raise ValueError("Expected 1 to 12 unique supported feature names")
    unsupported = set(names) - SUPPORTED_FEATURES
    if unsupported:
        raise ValueError("Unsupported or leaking feature names: " + ", ".join(sorted(unsupported)))
    return names


def _matrix(frame, names):
    if not isinstance(frame, pd.DataFrame) or not frame.columns.is_unique:
        raise ValueError("Expected a DataFrame with unique columns")
    missing = set(names) - set(frame.columns)
    if missing:
        raise ValueError("Missing model features: " + ", ".join(sorted(missing)))
    try:
        values = frame.loc[:, names].to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("Model features must be finite numeric values") from exc
    if not np.isfinite(values).all():
        raise ValueError("Model features must be finite numeric values")
    return np.ascontiguousarray(values)


def _group_keys(groups, n):
    values = np.asarray(groups, dtype=object)
    if values.shape != (n,):
        raise ValueError("Expected one inner-validation group per training row")
    keys = []
    for value in values:
        if not np.isscalar(value) or pd.isna(value):
            raise ValueError("Inner-validation groups must be nonmissing scalar values")
        if isinstance(value, np.generic):
            value = value.item()
        # Equal numeric IDs (1 and 1.0) remain one group, while string '1' is a
        # distinct identifier. This also avoids input-order-dependent coercion.
        if isinstance(value, (int, float, bool)):
            if not np.isfinite(value):
                raise ValueError("Inner-validation groups must be finite")
            key = "number:" + (str(int(value)) if value == int(value) else float(value).hex())
        elif isinstance(value, str):
            key = "str:" + value
        else:
            raise ValueError("Inner-validation group IDs must be strings or numbers")
        keys.append(key)
    return np.asarray(keys, dtype=object)


def _inner_split(n, seed, groups):
    if n < 4:
        raise ValueError("At least four outer-training rows are required")
    rng = np.random.default_rng(seed)
    if groups is None:
        count = max(1, int(np.ceil(n * VALIDATION_FRACTION)))
        validation = np.sort(rng.permutation(n)[:count])
        fit = np.setdiff1d(np.arange(n), validation)
        return fit, validation, {"kind": "deterministic_rows"}
    keys = _group_keys(groups, n)
    unique = np.asarray(sorted(set(keys)), dtype=object)
    if len(unique) < 2:
        raise ValueError("Grouped inner validation requires at least two distinct groups")
    count = max(1, min(len(unique) - 1,
                       int(np.ceil(len(unique) * VALIDATION_FRACTION))))
    held_out = set(unique[rng.permutation(len(unique))[:count]])
    validation_mask = np.asarray([key in held_out for key in keys], dtype=bool)
    fit, validation = np.flatnonzero(~validation_mask), np.flatnonzero(validation_mask)
    digest = lambda key: hashlib.sha256(key.encode("utf-8")).hexdigest()
    return fit, validation, dict(
        kind="grouped", fit_groups=len(unique) - count, validation_groups=count,
        fit_group_sha256=[digest(key) for key in unique if key not in held_out],
        validation_group_sha256=[digest(key) for key in unique if key in held_out])


def _declared_inner_split(n, groups, fit_indices, validation_indices):
    """Validate caller-declared roles without changing their row ordering."""
    if n < 4:
        raise ValueError("At least four outer-training rows are required")
    positions = []
    for name, values in (("fit", fit_indices), ("validation", validation_indices)):
        values = np.asarray(values)
        if values.ndim != 1 or not len(values):
            raise ValueError("Predeclared inner " + name + " indices must be nonempty vectors")
        if values.dtype.kind not in "iu" or (values < 0).any() or (values >= n).any():
            raise ValueError("Predeclared inner " + name + " indices must be integer row positions")
        if len(np.unique(values)) != len(values):
            raise ValueError("Predeclared inner " + name + " indices contain duplicates")
        positions.append(values.astype(np.int64))
    fit, validation = positions
    if np.intersect1d(fit, validation).size:
        raise ValueError("Predeclared inner fit and validation positions overlap")
    if not np.array_equal(np.sort(np.concatenate(positions)), np.arange(n)):
        raise ValueError("Predeclared inner roles must partition every supplied training row")
    keys = _group_keys(groups, n)
    fit_groups, validation_groups = sorted(set(keys[fit])), sorted(set(keys[validation]))
    if set(fit_groups) & set(validation_groups):
        raise ValueError("Predeclared inner fit and validation groups overlap")
    digest = lambda key: hashlib.sha256(key.encode("utf-8")).hexdigest()
    return fit, validation, dict(
        kind="predeclared_grouped", fit_groups=len(fit_groups),
        validation_groups=len(validation_groups),
        fit_group_sha256=[digest(key) for key in fit_groups],
        validation_group_sha256=[digest(key) for key in validation_groups])


def _parameters(seed, leaves):
    return dict(objective="regression_l1", metric="l1", learning_rate=.05,
                num_leaves=int(leaves), min_data_in_leaf=20, lambda_l2=1.,
                feature_fraction=1., bagging_fraction=1., bagging_freq=0,
                seed=int(seed), feature_fraction_seed=int(seed),
                bagging_seed=int(seed), data_random_seed=int(seed),
                deterministic=True, force_col_wise=True, num_threads=1,
                verbosity=-1)


def fit_models(train_table, feature_names, seed, out_dir, groups=None, *,
               inner_fit_indices=None, inner_validation_indices=None):
    """Select a tiny direct-error model within outer TRAIN and save its refit.

    ``train_table.actual_error`` must contain finite nonnegative direct errors.
    When a ``split`` column is present, every row must equal ``train``. Groups
    can be a same-length sequence or a column name. If omitted, a ``cell``
    column is used when available. Missing grouping or fewer than two distinct
    groups is rejected, so an omitted argument cannot silently split siblings.
    Optional ``inner_fit_indices`` and ``inner_validation_indices`` must both be
    supplied as nonempty, disjoint, group-disjoint integer positions partitioning
    the input table. They preserve roles chosen before workflow query caps.

    The returned dictionary includes the portable ``model_path``, exact runtime
    ``model_bytes``, and a selection audit with positions relative to the input
    table. The audit is returned separately from the runtime schema so training
    provenance does not become an inference dependency or hidden model state.
    Selected candidate predictions are saved before refitting, in the separate
    ``validation_predictions_path`` CSV; its audit bytes are not runtime bytes.
    """
    names = _feature_names(feature_names)
    x = _matrix(train_table, names)
    if "split" in train_table and not train_table["split"].eq("train").all():
        raise ValueError("fit_models accepts outer train rows only")
    if TARGET_COLUMN not in train_table:
        raise ValueError("Missing direct-error target: " + TARGET_COLUMN)
    try:
        y = train_table[TARGET_COLUMN].to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("actual_error must be finite and nonnegative") from exc
    if not np.isfinite(y).all() or (y < 0).any():
        raise ValueError("actual_error must be finite and nonnegative")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if isinstance(groups, str):
        if groups not in train_table:
            raise ValueError("Missing grouping column: " + groups)
        groups = train_table[groups].to_numpy()
    elif groups is None and "cell" in train_table:
        groups = train_table["cell"].to_numpy()
    if groups is None:
        raise ValueError("R10 inner validation requires groups or a cell column")
    if (inner_fit_indices is None) != (inner_validation_indices is None):
        raise ValueError("Provide both predeclared inner fit and validation indices")
    if inner_fit_indices is None:
        fit, validation, split_info = _inner_split(len(x), int(seed), groups)
    else:
        fit, validation, split_info = _declared_inner_split(
            len(x), groups, inner_fit_indices, inner_validation_indices)
    lgb = _lightgbm()
    candidates, candidate_predictions = [], []
    for leaves in NUM_LEAVES:
        fitted = lgb.train(
            _parameters(seed, leaves),
            lgb.Dataset(x[fit], label=y[fit], feature_name=names, free_raw_data=True),
            num_boost_round=MAX_ITERATIONS,
            valid_sets=[lgb.Dataset(x[validation], label=y[validation],
                                    feature_name=names, free_raw_data=True)],
            valid_names=["inner_validation"],
            callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)])
        iteration = int(fitted.best_iteration or fitted.current_iteration())
        prediction = np.maximum(0., fitted.predict(x[validation], num_iteration=iteration,
                                                   num_threads=1))
        candidates.append(dict(num_leaves=leaves, iterations=iteration,
                               validation_mae=float(np.mean(abs(prediction - y[validation])))))
        candidate_predictions.append(prediction)
    # All tie-breaks are frozen before test/calibration predictions are available.
    chosen_index = min(range(len(candidates)), key=lambda i:
                       (candidates[i]["validation_mae"], candidates[i]["num_leaves"],
                        candidates[i]["iterations"]))
    chosen = candidates[chosen_index]
    out = Path(out_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    validation_path = out / "inner_validation_predictions.csv"
    heldout = pd.DataFrame(dict(row_position=validation,
                               actual_error=y[validation],
                               prediction=candidate_predictions[chosen_index]))
    if "qId" in train_table:
        heldout.insert(1, "qId", train_table.iloc[validation]["qId"].to_numpy())
    # Persist the selected holdout predictions while the candidate model still
    # precedes final fitting. Never substitute all-train-refit predictions here.
    heldout.to_csv(validation_path, index=False)
    final = lgb.train(_parameters(seed, chosen["num_leaves"]),
                      lgb.Dataset(x, label=y, feature_name=names, free_raw_data=True),
                      num_boost_round=chosen["iterations"])
    text_path, schema_path = out / "model.txt", out / "model.schema.json"
    final.save_model(str(text_path))
    model_payload = text_path.read_bytes()
    schema = dict(schema_version=SCHEMA_VERSION, backend="lightgbm",
                  backend_version=lgb.__version__, model_file=text_path.name,
                  model_sha256=hashlib.sha256(model_payload).hexdigest(),
                  feature_names=names, target_column=TARGET_COLUMN,
                  target_transform="identity", prediction_clip_min=0.,
                  inference_num_threads=1)
    schema_path.write_text(json.dumps(schema, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return dict(
        model_path=str(schema_path), model_text_path=str(text_path),
        backend="lightgbm", backend_version=lgb.__version__,
        model_bytes=text_path.stat().st_size + schema_path.stat().st_size,
        model_text_bytes=text_path.stat().st_size, schema_bytes=schema_path.stat().st_size,
        feature_names=names, target_column=TARGET_COLUMN, target_transform="identity",
        selected_num_leaves=chosen["num_leaves"], selected_iterations=chosen["iterations"],
        validation_mae=chosen["validation_mae"],
        validation_predictions_path=str(validation_path),
        validation_predictions_bytes=validation_path.stat().st_size,
        validation_row_positions=validation.tolist(),
        refit_rows=len(x), candidates=candidates,
        selection=dict(seed=int(seed), validation_fraction=len(validation) / len(x),
                       validation_mae=chosen["validation_mae"],
                       maximum_iterations=MAX_ITERATIONS,
                       early_stopping_rounds=EARLY_STOPPING_ROUNDS,
                       fit_row_positions=fit.tolist(),
                       validation_row_positions=validation.tolist(), **split_info))


@dataclass(frozen=True)
class ErrorModel:
    """A reloaded portable model; predictions are scores and leave H untouched."""

    booster: object
    schema: dict
    schema_path: Path

    @property
    def feature_names(self):
        return tuple(self.schema["feature_names"])

    @property
    def model_bytes(self):
        return self.schema_path.stat().st_size + (self.schema_path.parent /
                                                  self.schema["model_file"]).stat().st_size

    def predict_frame(self, frame):
        values = _matrix(frame, list(self.feature_names))
        if not len(values):
            return np.empty(0, dtype=float)
        prediction = np.asarray(self.booster.predict(values, num_threads=1), dtype=float)
        if prediction.shape != (len(values),) or not np.isfinite(prediction).all():
            raise ValueError("LightGBM returned invalid error predictions")
        return np.maximum(0., prediction)

    def predict_row(self, row: Mapping):
        if not isinstance(row, Mapping):
            # pandas Series is the common live-row representation.
            if not isinstance(row, pd.Series) or not row.index.is_unique:
                raise ValueError("Expected a row mapping or Series")
        try:
            values = np.asarray([[row[name] for name in self.feature_names]], dtype=np.float64)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Row must contain every finite numeric model feature") from exc
        if not np.isfinite(values).all():
            raise ValueError("Row must contain every finite numeric model feature")
        prediction = float(self.booster.predict(values, num_threads=1)[0])
        if not np.isfinite(prediction):
            raise ValueError("LightGBM returned an invalid error prediction")
        return max(0., prediction)

    predict = predict_frame


def load_model(path):
    """Load a checked LightGBM text + JSON pair from a schema path/directory."""
    path = Path(path).expanduser().resolve()
    if path.is_dir():
        path = path / "model.schema.json"
    schema = json.loads(path.read_text())
    if (not isinstance(schema, dict) or schema.get("schema_version") != SCHEMA_VERSION or
            schema.get("backend") != "lightgbm" or
            schema.get("target_column") != TARGET_COLUMN or
            schema.get("target_transform") != "identity" or
            schema.get("prediction_clip_min") != 0. or
            schema.get("inference_num_threads") != 1):
        raise ValueError("Unsupported R10 error-model schema")
    names = _feature_names(schema.get("feature_names", []))
    if schema.get("model_file") != "model.txt":
        raise ValueError("Model text must be the local model.txt artifact")
    text_path = path.parent / schema["model_file"]
    if hashlib.sha256(text_path.read_bytes()).hexdigest() != schema.get("model_sha256"):
        raise ValueError("LightGBM model checksum mismatch")
    booster = _lightgbm().Booster(model_file=str(text_path))
    if booster.feature_name() != names or booster.num_feature() != len(names):
        raise ValueError("LightGBM text and schema feature order disagree")
    return ErrorModel(booster=booster, schema=schema, schema_path=path)
