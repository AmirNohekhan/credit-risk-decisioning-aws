from __future__ import annotations

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
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
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
    calibrated = CalibratedClassifierCV(FrozenEstimator(gb), method="sigmoid")
    calibrated.fit(val[MODEL_FEATURES], yval)
    defaults = train[train.default_12m.eq(1) & train.lgd_realized.notna()]
    lgd = None
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
    reports = {
        "logistic": metrics(
            test.default_12m.astype(int), logistic.predict_proba(test[MODEL_FEATURES])[:, 1]
        ),
        "gradient_boosting_calibrated": metrics(
            test.default_12m.astype(int), calibrated.predict_proba(test[MODEL_FEATURES])[:, 1]
        ),
        "split": {
            "train": len(train),
            "validation": len(val),
            "test": len(test),
            "train_end": str(train.application_date.max().date()),
            "validation_end": str(val.application_date.max().date()),
        },
    }
    return ModelBundle(calibrated, lgd), reports
