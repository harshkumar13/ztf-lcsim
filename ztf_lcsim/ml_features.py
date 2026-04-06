"""
ML-based feature augmentation for ZTF light curve similarity search.

Trains a calibrated Random Forest on labeled LC features and appends
class probability vectors as additional search features.

Usage
-----
from ztf_lcsim.ml_features import MLFeatureAugmenter
aug = MLFeatureAugmenter()
aug.fit(X_train, labels)
X_aug = aug.augment(X_new)
aug.save("ztf_data/ml_augmenter.pkl")
aug2 = MLFeatureAugmenter.load("ztf_data/ml_augmenter.pkl")
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

logger = logging.getLogger(__name__)


class MLFeatureAugmenter:
    """
    Trains a Random Forest on labeled light-curve features and uses the
    predicted class probabilities as additional similarity-search features.

    Why this helps
    --------------
    P(class | features) is a compressed, class-discriminating representation.
    Two objects with similar probability vectors are similar in class space
    even when their raw feature vectors differ in amplitude or phase.

    Parameters
    ----------
    n_estimators    : number of trees in the Random Forest
    max_depth       : max tree depth (None = unlimited)
    min_samples_leaf: min samples per leaf (prevents overfitting)
    calibrate       : whether to Platt-calibrate the RF probabilities
    random_state    : reproducibility seed
    """

    def __init__(
        self,
        n_estimators: int = 300,
        max_depth: Optional[int] = None,
        min_samples_leaf: int = 5,
        calibrate: bool = True,
        random_state: int = 42,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.calibrate = calibrate
        self.random_state = random_state

        self.pipeline_: Optional[Pipeline] = None
        self.label_encoder_: Optional[LabelEncoder] = None
        self.classes_: Optional[List[str]] = None
        self.feature_importances_: Optional[np.ndarray] = None
        self.cv_scores_: Optional[np.ndarray] = None
        self.is_fitted: bool = False

    # ── training ──────────────────────────────────────────────────────────────

    def fit(
        self,
        X: np.ndarray,
        labels: List[str],
        verbose: bool = True,
    ) -> "MLFeatureAugmenter":
        """
        Train on labeled feature vectors.

        Parameters
        ----------
        X      : ndarray, shape (N, D)  — may contain NaN (handled internally)
        labels : list of str, length N  — class labels e.g. ["RRL", "SNIa", ...]
        """
        X = np.asarray(X, dtype=np.float32)
        labels = list(labels)

        if len(X) == 0:
            raise ValueError("No training data supplied.")
        if len(X) != len(labels):
            raise ValueError(
                f"X has {len(X)} rows but labels has {len(labels)} entries."
            )

        # ── encode labels ─────────────────────────────────────────────────────
        self.label_encoder_ = LabelEncoder()
        y = self.label_encoder_.fit_transform(labels)
        self.classes_ = list(self.label_encoder_.classes_)
        n_classes = len(self.classes_)

        if verbose:
            counts = pd.Series(labels).value_counts()
            logger.info(
                f"Training MLFeatureAugmenter on {len(X):,} objects, "
                f"{n_classes} classes:\n{counts.to_string()}"
            )

        # ── build pipeline ────────────────────────────────────────────────────
        base_clf = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            max_features="sqrt",
            n_jobs=-1,
            random_state=self.random_state,
            class_weight="balanced",
        )

        if self.calibrate and len(X) >= 50:
            # use 3-fold isotonic calibration for reliable probabilities
            n_cv = min(3, n_classes)
            clf = CalibratedClassifierCV(base_clf, cv=n_cv, method="isotonic")
        else:
            clf = base_clf

        self.pipeline_ = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("clf", clf),
            ]
        )

        # ── cross-validation ─────────────────────────────────────────────────
        n_splits = min(5, min(pd.Series(labels).value_counts()))
        n_splits = max(2, int(n_splits))

        if verbose and len(X) >= 50 and n_splits >= 2:
            try:
                cv = StratifiedKFold(
                    n_splits=n_splits,
                    shuffle=True,
                    random_state=self.random_state,
                )
                self.cv_scores_ = cross_val_score(
                    self.pipeline_,
                    X,
                    y,
                    cv=cv,
                    scoring="balanced_accuracy",
                    n_jobs=-1,
                )
                logger.info(
                    f"{n_splits}-fold CV balanced accuracy: "
                    f"{self.cv_scores_.mean():.3f} "
                    f"± {self.cv_scores_.std():.3f}"
                )
            except Exception as exc:
                logger.warning(f"Cross-validation failed ({exc}) — skipping.")

        # ── final fit on all data ─────────────────────────────────────────────
        self.pipeline_.fit(X, y)
        self.is_fitted = True

        # ── extract feature importances ───────────────────────────────────────
        try:
            inner = self.pipeline_.named_steps["clf"]
            if hasattr(inner, "estimator"):
                fi = inner.estimator.feature_importances_
            elif hasattr(inner, "feature_importances_"):
                fi = inner.feature_importances_
            else:
                fi = None
            self.feature_importances_ = fi
        except Exception:
            self.feature_importances_ = None

        if verbose:
            logger.info("MLFeatureAugmenter training complete ✓")
            if self.feature_importances_ is not None:
                top5 = np.argsort(self.feature_importances_)[::-1][:5]
                logger.info(
                    "Top-5 feature importance indices: "
                    + ", ".join(
                        f"[{i}]={self.feature_importances_[i]:.4f}" for i in top5
                    )
                )

        return self

    # ── inference ─────────────────────────────────────────────────────────────
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        import warnings

        if not self.is_fitted:
            raise RuntimeError("MLFeatureAugmenter is not fitted. Call fit() first.")
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Skipping features without any observed values",
                category=UserWarning,
            )
            return self.pipeline_.predict_proba(X).astype(np.float32)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predicted class labels (argmax of probabilities)."""
        proba = self.predict_proba(X)
        idx = np.argmax(proba, axis=1)
        return np.array([self.classes_[i] for i in idx])

    def augment(self, X: np.ndarray) -> np.ndarray:
        """
        Concatenate raw features with class probability vector.

        Parameters
        ----------
        X : ndarray, shape (N, D)

        Returns
        -------
        ndarray, shape (N, D + n_classes)
        """
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        proba = self.predict_proba(X)
        return np.hstack([X, proba]).astype(np.float32)

    # ── properties ────────────────────────────────────────────────────────────

    @property
    def n_prob_features(self) -> int:
        """Number of class probability columns added by augment()."""
        return len(self.classes_) if self.classes_ else 0

    @property
    def prob_feature_names(self) -> List[str]:
        """Column names for the probability features."""
        return [f"p_{c}" for c in (self.classes_ or [])]

    # ── persistence ───────────────────────────────────────────────────────────

    def save(self, path: Union[str, Path]):
        """Serialise to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "pipeline": self.pipeline_,
            "label_encoder": self.label_encoder_,
            "classes": self.classes_,
            "feature_importances": self.feature_importances_,
            "cv_scores": self.cv_scores_,
            "is_fitted": self.is_fitted,
            "params": {
                "n_estimators": self.n_estimators,
                "max_depth": self.max_depth,
                "min_samples_leaf": self.min_samples_leaf,
                "calibrate": self.calibrate,
                "random_state": self.random_state,
            },
        }

        with open(path, "wb") as fh:
            pickle.dump(payload, fh, protocol=4)

        logger.info(f"MLFeatureAugmenter saved to {path}")

    @classmethod
    def load(cls, path: Union[str, Path]) -> "MLFeatureAugmenter":
        """Load from disk. Returns a ready-to-use instance."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"ML augmenter not found at {path}.\n"
                "Run:  python scripts/05_train_ml.py"
            )

        with open(path, "rb") as fh:
            data = pickle.load(fh)

        obj = cls(**data.get("params", {}))
        obj.pipeline_ = data["pipeline"]
        obj.label_encoder_ = data["label_encoder"]
        obj.classes_ = data["classes"]
        obj.feature_importances_ = data.get("feature_importances")
        obj.cv_scores_ = data.get("cv_scores")
        obj.is_fitted = data["is_fitted"]

        logger.info(
            f"MLFeatureAugmenter loaded — "
            f"{len(obj.classes_)} classes: {obj.classes_}"
        )
        return obj

    # ── diagnostics ───────────────────────────────────────────────────────────

    def explain(
        self,
        X: np.ndarray,
        feature_names: Optional[List[str]] = None,
        top_n: int = 15,
    ) -> pd.DataFrame:
        """
        Print class probabilities and top feature importances for one object.

        Parameters
        ----------
        X            : feature vector, shape (D,) or (1, D)
        feature_names: list of feature name strings
        top_n        : how many top features to print

        Returns
        -------
        pd.DataFrame of class probabilities sorted descending
        """
        if not self.is_fitted:
            raise RuntimeError("Not fitted.")

        proba = self.predict_proba(X.reshape(1, -1))[0]
        df_p = pd.DataFrame(
            {
                "class": self.classes_,
                "probability": proba,
            }
        ).sort_values("probability", ascending=False)

        print("\n── ML class probabilities ─────────────────────────")
        print(df_p.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

        if self.feature_importances_ is not None and feature_names is not None:
            n = min(top_n, len(feature_names), len(self.feature_importances_))
            top = np.argsort(self.feature_importances_)[::-1][:n]
            df_fi = pd.DataFrame(
                {
                    "feature": [feature_names[i] for i in top],
                    "importance": self.feature_importances_[top],
                }
            )
            print(f"\n── Top-{n} RF feature importances (global) ────────")
            print(df_fi.to_string(index=False, float_format=lambda x: f"{x:.5f}"))

        return df_p

    def class_report(
        self,
        X: np.ndarray,
        labels: List[str],
    ) -> str:
        """Return sklearn classification report string."""
        from sklearn.metrics import classification_report

        if not self.is_fitted:
            raise RuntimeError("Not fitted.")
        y_pred = self.predict(X)
        return classification_report(labels, y_pred, target_names=self.classes_)
