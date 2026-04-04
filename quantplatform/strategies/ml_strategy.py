"""
Machine Learning strategy: Random Forest + XGBoost with walk-forward validation.
Proper time-series train/val/test split — NO random shuffling.
"""
import logging
import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

from features.engineering import get_feature_columns

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


class MLStrategy:
    """
    Supervised classification strategy.
    Target: whether price is higher in `horizon` bars → long signal.
    Uses walk-forward validation to avoid data leakage.
    """
    name = "ML Strategy"

    def __init__(
        self,
        model_type: str = "random_forest",   # "random_forest", "xgboost", "logistic"
        horizon: int = 5,
        threshold: float = 0.55,              # confidence threshold to trade
        walk_forward_windows: int = 5,
    ):
        self.model_type = model_type
        self.horizon = horizon
        self.threshold = threshold
        self.walk_forward_windows = walk_forward_windows
        self.model = None
        self.scaler = None
        self.feature_cols = None
        self.feature_importances_ = None
        self.train_metrics = {}

    def _build_model(self):
        if self.model_type == "xgboost" and HAS_XGB:
            return XGBClassifier(
                n_estimators=200, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                use_label_encoder=False, eval_metric="logloss",
                random_state=42, verbosity=0
            )
        elif self.model_type == "logistic":
            return LogisticRegression(max_iter=1000, C=1.0, random_state=42)
        else:
            return RandomForestClassifier(
                n_estimators=200, max_depth=6, min_samples_leaf=20,
                max_features="sqrt", random_state=42, n_jobs=-1
            )

    def fit(self, feature_df: pd.DataFrame) -> dict:
        """
        Walk-forward training on feature_df.
        Returns training metrics dict.
        """
        self.feature_cols = get_feature_columns(feature_df)
        X = feature_df[self.feature_cols].values
        y = feature_df["target"].values

        n = len(X)
        window_size = n // (self.walk_forward_windows + 1)

        fold_metrics = []
        logger.info(f"Walk-forward training: {self.walk_forward_windows} folds, model={self.model_type}")

        for fold in range(self.walk_forward_windows):
            train_end = window_size * (fold + 1)
            test_start = train_end
            test_end = min(train_end + window_size, n)

            X_train, y_train = X[:train_end], y[:train_end]
            X_test, y_test = X[test_start:test_end], y[test_start:test_end]

            if len(X_test) == 0:
                break

            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)

            model = self._build_model()
            model.fit(X_train_s, y_train)
            preds = model.predict(X_test_s)
            acc = accuracy_score(y_test, preds)
            fold_metrics.append(acc)
            logger.info(f"  Fold {fold+1}: accuracy={acc:.3f}")

        # Final model trained on all data except last 20%
        split = int(n * 0.80)
        self.scaler = StandardScaler()
        X_train_final = self.scaler.fit_transform(X[:split])
        X_test_final = self.scaler.transform(X[split:])

        self.model = self._build_model()
        self.model.fit(X_train_final, y[:split])

        # Out-of-sample test metrics
        test_preds = self.model.predict(X_test_final)
        test_acc = accuracy_score(y[split:], test_preds)

        # Feature importances
        if hasattr(self.model, "feature_importances_"):
            self.feature_importances_ = pd.Series(
                self.model.feature_importances_, index=self.feature_cols
            ).sort_values(ascending=False)
        elif hasattr(self.model, "coef_"):
            self.feature_importances_ = pd.Series(
                np.abs(self.model.coef_[0]), index=self.feature_cols
            ).sort_values(ascending=False)

        self.train_metrics = {
            "walk_forward_acc": np.mean(fold_metrics),
            "walk_forward_std": np.std(fold_metrics),
            "oos_accuracy": test_acc,
            "fold_accuracies": fold_metrics,
            "n_train": split,
            "n_test": n - split,
        }
        logger.info(f"Final OOS accuracy: {test_acc:.3f}")
        return self.train_metrics

    def generate_signals(self, feature_df: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals from ML predictions.
        Returns Series of {1=long, -1=short, 0=flat}.
        """
        if self.model is None:
            raise RuntimeError("Model not fitted. Call .fit() first.")

        X = feature_df[self.feature_cols].values
        X_scaled = self.scaler.transform(X)

        proba = self.model.predict_proba(X_scaled)
        long_prob = proba[:, 1]   # probability of upward move
        short_prob = proba[:, 0]  # probability of downward move

        signals = pd.Series(0, index=feature_df.index)
        signals[long_prob >= self.threshold] = 1
        signals[short_prob >= self.threshold] = -1

        # Shift by 1 to avoid lookahead
        signals = signals.shift(1).fillna(0)
        logger.info(f"[{self.name}] Long: {(signals==1).sum()}, Short: {(signals==-1).sum()}, Flat: {(signals==0).sum()}")
        return signals

    def get_prediction_df(self, feature_df: pd.DataFrame) -> pd.DataFrame:
        """Return full prediction dataframe with probabilities for visualization."""
        if self.model is None:
            return pd.DataFrame()

        X = feature_df[self.feature_cols].values
        X_scaled = self.scaler.transform(X)
        proba = self.model.predict_proba(X_scaled)

        return pd.DataFrame({
            "long_prob": proba[:, 1],
            "short_prob": proba[:, 0],
            "signal": self.generate_signals(feature_df).values,
        }, index=feature_df.index)
