from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestRegressor
from sklearn.frozen import FrozenEstimator
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .features import CATEGORICAL_FEATURES, MODEL_FEATURES, NUMERIC_FEATURES


def preprocessor(scale: bool = True) -> ColumnTransformer:
    num = [("impute", SimpleImputer(strategy="median", add_indicator=True))]
    if scale:
        num.append(("scale", StandardScaler()))
    return ColumnTransformer(
        [
            ("num", Pipeline(num), NUMERIC_FEATURES),
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
        ]
    )


def ks_statistic(y: pd.Series | np.ndarray, p: np.ndarray) -> float:
    order = np.argsort(p)
    ys = np.asarray(y)[order]
    bad = np.cumsum(ys) / max(ys.sum(), 1)
    good = np.cumsum(1 - ys) / max((1 - ys).sum(), 1)
    return float(np.max(np.abs(bad - good)))


def metrics(y, p) -> dict[str, float]:
    auc = roc_auc_score(y, p)
    return {
        "roc_auc": float(auc),
        "pr_auc": float(average_precision_score(y, p)),
        "ks": ks_statistic(y, p),
        "gini": float(2 * auc - 1),
        "brier": float(brier_score_loss(y, p)),
    }


def lgd_metrics(actual: pd.Series | np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    y = np.asarray(actual, dtype=float)
    p = np.asarray(predicted, dtype=float).clip(0, 1)
    return {
        "mae": float(mean_absolute_error(y, p)),
        "rmse": float(np.sqrt(mean_squared_error(y, p))),
        "mean_actual_lgd": float(y.mean()),
        "mean_predicted_lgd": float(p.mean()),
    }


def temporal_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ordered = df.sort_values("application_date")
    a = int(0.6 * len(ordered))
    b = int(0.8 * len(ordered))
    return ordered.iloc[:a], ordered.iloc[a:b], ordered.iloc[b:]


@dataclass
class ModelBundle:
    pd_model: Any
    lgd_model: Any | None
    model_version: str = "pd-gb-cal-v1"
    feature_version: str = "application-v1"

    def predict_pd(self, frame: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.pd_model.predict_proba(frame[MODEL_FEATURES])[:, 1]).clip(
            0.001, 0.999
        )

    def predict_lgd(self, frame: pd.DataFrame) -> np.ndarray:
        if self.lgd_model is None:
            return np.full(len(frame), 0.62)
        return np.asarray(self.lgd_model.predict(frame[MODEL_FEATURES])).clip(0.1, 0.95)

    def save(self, path: str):
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str):
        return joblib.load(path)


def select_champion(candidate_metrics: dict[str, dict[str, float]]) -> tuple[str, dict]:
    """Select the best calibrated candidate subject to minimum risk-model gates."""
    gates: dict[str, dict[str, float | bool]] = {}
    eligible: list[str] = []
    for name, values in candidate_metrics.items():
        passed = values["roc_auc"] >= 0.65 and values["ks"] >= 0.20 and values["brier"] <= 0.25
        gates[name] = {
            "passed": passed,
            "minimum_auc": 0.65,
            "minimum_ks": 0.20,
            "maximum_brier": 0.25,
        }
        if passed:
            eligible.append(name)
    if not eligible:
        raise ValueError("No PD candidate passed the configured quality gates")
    champion = max(
        eligible,
        key=lambda name: (candidate_metrics[name]["roc_auc"], -candidate_metrics[name]["brier"]),
    )
    return champion, gates


def registry_manifest(bundle: ModelBundle, report: dict) -> dict:
    return {
        "champion": bundle.model_version,
        "feature_version": bundle.feature_version,
        "default_definition": "90+ DPD or charge-off within 12 months",
        "training_split": report["split"],
        "candidate_metrics": report["candidates"],
        "lgd_validation": report["lgd"],
        "quality_gates": report["quality_gates"],
        "promotion_decision": "PROMOTE",
        "compatible_policy_versions": ["policy-v1"],
    }


def save_registry_manifest(path: str, bundle: ModelBundle, report: dict) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(registry_manifest(bundle, report), handle, indent=2)


def train_models(df: pd.DataFrame) -> tuple[ModelBundle, dict]:
    train, val, test = temporal_split(df)
    ytrain = train.default_12m.astype(int)
    yval = val.default_12m.astype(int)
    logistic = Pipeline(
        [
            ("prep", preprocessor()),
            ("model", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )
    logistic.fit(train[MODEL_FEATURES], ytrain)
    logistic_calibrated = CalibratedClassifierCV(FrozenEstimator(logistic), method="sigmoid")
    logistic_calibrated.fit(val[MODEL_FEATURES], yval)
    gb = Pipeline(
        [
            ("prep", preprocessor(False)),
            (
                "model",
                HistGradientBoostingClassifier(
                    max_iter=130, max_leaf_nodes=15, learning_rate=0.05, l2_regularization=2
                ),
            ),
        ]
    )
    gb.fit(train[MODEL_FEATURES], ytrain)
    gb_calibrated = CalibratedClassifierCV(FrozenEstimator(gb), method="sigmoid")
    gb_calibrated.fit(val[MODEL_FEATURES], yval)
    defaults = train[train.default_12m.eq(1) & train.lgd_realized.notna()]
    test_defaults = test[test.default_12m.eq(1) & test.lgd_realized.notna()]
    lgd = None
    lgd_report: dict[str, Any] = {
        "training_defaults": len(defaults),
        "test_defaults": len(test_defaults),
        "status": "insufficient_defaults",
    }
    if len(defaults) >= 20:
        lgd = Pipeline(
            [
                ("prep", preprocessor(False)),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=80, min_samples_leaf=8, random_state=42, n_jobs=1
                    ),
                ),
            ]
        )
        lgd.fit(defaults[MODEL_FEATURES], defaults.lgd_realized)
        if len(test_defaults) > 0:
            predictions = np.asarray(lgd.predict(test_defaults[MODEL_FEATURES])).clip(0.1, 0.95)
            benchmark = np.full(len(test_defaults), float(defaults.lgd_realized.mean()))
            lgd_report.update(
                {
                    "status": "evaluated",
                    "model": lgd_metrics(test_defaults.lgd_realized, predictions),
                    "mean_lgd_benchmark": lgd_metrics(test_defaults.lgd_realized, benchmark),
                }
            )
    candidates = {
        "logistic_calibrated": metrics(
            test.default_12m.astype(int),
            logistic_calibrated.predict_proba(test[MODEL_FEATURES])[:, 1],
        ),
        "gradient_boosting_calibrated": metrics(
            test.default_12m.astype(int), gb_calibrated.predict_proba(test[MODEL_FEATURES])[:, 1]
        ),
    }
    champion, gates = select_champion(candidates)
    models = {
        "logistic_calibrated": logistic_calibrated,
        "gradient_boosting_calibrated": gb_calibrated,
    }
    versions = {
        "logistic_calibrated": "pd-logistic-cal-v2",
        "gradient_boosting_calibrated": "pd-gb-cal-v2",
    }
    reports = {
        "candidates": candidates,
        "champion": champion,
        "quality_gates": gates,
        "lgd": lgd_report,
        "split": {
            "train": len(train),
            "validation": len(val),
            "test": len(test),
            "train_end": str(train.application_date.max().date()),
            "validation_end": str(val.application_date.max().date()),
        },
    }
    return ModelBundle(models[champion], lgd, model_version=versions[champion]), reports
