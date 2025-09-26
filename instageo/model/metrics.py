"""Streaming / running classification metrics for large datasets.

Classes
-------
RunningConfusionMatrix
    Tracks a confusion matrix online; yields accuracy, precision, recall,
    F1 and Jaccard (IoU) with *macro* averages **and** per‑class vectors.

RunningAUC
    Histogram‑based streaming ROC‑AUC that supports one‑vs‑rest macro AUC and
    per‑class AUCs.

RunningRegressionMetrics
    Maintain streaming statistics for regression metrics.

Both classes expose two public methods:
    • update(y_true, y_pred, y_score=None) – Add a (mini‑)batch.
    • compute() – Return a dict of metrics.
"""
from __future__ import annotations

import numpy as np

__all__ = ["RunningConfusionMatrix", "RunningAUC", "RunningRegressionMetrics"]


# -----------------------------------------------------------------------------
# Helper utilities
# -----------------------------------------------------------------------------


def _safe_div(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    """Element‑wise num / den, returns 0 where den == 0."""
    den = den.astype(float)
    out = np.zeros_like(den, dtype=float)
    np.divide(num, den, out=out, where=den != 0)
    return out


# -----------------------------------------------------------------------------
# Confusion‑matrix‑based metrics
# -----------------------------------------------------------------------------


class RunningConfusionMatrix:
    """Maintain a streaming confusion matrix for *single‑label* classification.

    Parameters
    ----------
    num_classes : int
        Total number of classes.
    ignore_index : int | None, default None
        A label to mask out when updating (e.g. background in segmentation).
    """

    def __init__(self, num_classes: int, ignore_index: int | None = None) -> None:
        """Initialize the confusion matrix.

        Args:
            num_classes: Total number of classes.
            ignore_index: A label to mask out when updating.
        """
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
        self.total = 0  # valid samples so far

    def update(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """Update the confusion matrix with new predictions.

        Args:
            y_true: Ground truth labels.
            y_pred: Predicted labels.
        """
        y_true = np.asarray(y_true).ravel()
        y_pred = np.asarray(y_pred).ravel()
        if y_true.shape != y_pred.shape:
            raise ValueError("y_true and y_pred shapes differ.")

        if self.ignore_index is not None:
            mask = y_true != self.ignore_index
            y_true, y_pred = y_true[mask], y_pred[mask]
        if y_true.size == 0:
            return

        k = self.num_classes
        indices = y_true * k + y_pred
        binc = np.bincount(indices, minlength=k * k)
        self.matrix += binc.reshape(k, k)
        self.total += y_true.size

    @property
    def _tp(self) -> np.ndarray:
        """Get true positives."""
        return np.diag(self.matrix)

    @property
    def _fp(self) -> np.ndarray:
        """Get false positives."""
        return self.matrix.sum(axis=0) - self._tp

    @property
    def _fn(self) -> np.ndarray:
        """Get false negatives."""
        return self.matrix.sum(axis=1) - self._tp

    def accuracy(self) -> float:
        """Compute accuracy."""
        if self.total == 0:
            return float("nan")
        return self._tp.sum() / self.total

    def precision(self) -> np.ndarray:
        """Compute precision per class."""
        return _safe_div(self._tp, self._tp + self._fp)

    def recall(self) -> np.ndarray:
        """Compute recall per class."""
        return _safe_div(self._tp, self._tp + self._fn)

    def f1(self) -> np.ndarray:
        """Compute F1 score per class."""
        p, r = self.precision(), self.recall()
        return _safe_div(2 * p * r, p + r)

    def jaccard(self) -> np.ndarray:
        """Compute Jaccard (IoU) per class."""
        return _safe_div(self._tp, self._tp + self._fp + self._fn)

    def compute(self, include_per_class: bool = True) -> dict[str, float | list[float]]:
        """Return a dict with macro averages + (optionally) per‑class lists."""
        metrics = {
            "accuracy": self.accuracy(),
            "precision": self.precision().mean(),
            "recall": self.recall().mean(),
            "f1": self.f1().mean(),
            "jaccard": self.jaccard().mean(),
        }
        if include_per_class:
            metrics.update(
                {
                    "precision_per_class": self.precision().tolist(),
                    "recall_per_class": self.recall().tolist(),
                    "f1_per_class": self.f1().tolist(),
                    "jaccard_per_class": self.jaccard().tolist(),
                }
            )
        return metrics

    def reset(self) -> None:
        """Reset the confusion matrix and total count."""
        self.matrix.fill(0)
        self.total = 0


# -----------------------------------------------------------------------------
# Streaming ROC‑AUC (histogram) – one‑vs‑rest
# -----------------------------------------------------------------------------


class RunningAUC:
    """Histogram‑based streaming ROC‑AUC.

    Notes
    -----
    • Supports *macro* OVR AUC as well as per‑class AUCs.
    • Memory/CPU O(n_bins * C).
    """

    def __init__(
        self,
        num_classes: int,
        n_bins: int = 1024,
        min_score: float = 0.0,
        max_score: float = 1.0,
    ) -> None:
        """Initialize the AUC calculator.

        Args:
            num_classes: Total number of classes.
            n_bins: Number of histogram bins.
            min_score: Minimum score value.
            max_score: Maximum score value.
        """
        self.num_classes = num_classes
        self.n_bins = n_bins
        self.min_score = min_score
        self.max_score = max_score
        self.reset()

    def _bin(self, score: float) -> int:
        """Convert a score to a bin index."""
        score = min(self.max_score, max(self.min_score, score))
        return int((score - self.min_score) / (self.max_score - self.min_score) * (self.n_bins - 1))

    def update(self, y_true: np.ndarray, y_score: np.ndarray) -> None:
        """Update the AUC histograms with new predictions.

        Args:
            y_true: Ground truth labels.
            y_score: Predicted probabilities.
        """
        y_true = np.asarray(y_true).ravel()
        y_score = np.asarray(y_score)
        if y_score.ndim == 1:
            # Binary case: probability of positive class only
            if self.num_classes != 2:
                raise ValueError("For 1‑D y_score, num_classes must be 2.")
            y_score = np.stack([1 - y_score, y_score], axis=1)
        if y_true.shape[0] != y_score.shape[0]:
            raise ValueError("y_true and y_score length mismatch.")
        if y_score.shape[1] != self.num_classes:
            raise ValueError("Second dim of y_score must equal num_classes.")

        for cls in range(self.num_classes):
            cls_scores = y_score[:, cls]
            pos_mask = y_true == cls
            neg_mask = ~pos_mask
            if pos_mask.any():
                pos_bins = np.fromiter((self._bin(s) for s in cls_scores[pos_mask]), dtype=np.int32)
                np.add.at(self.pos_hist[cls], pos_bins, 1)
                self.n_pos[cls] += pos_bins.size
            if neg_mask.any():
                neg_bins = np.fromiter((self._bin(s) for s in cls_scores[neg_mask]), dtype=np.int32)
                np.add.at(self.neg_hist[cls], neg_bins, 1)
                self.n_neg[cls] += neg_bins.size

    def _auc_one_class(self, c: int) -> float:
        """Compute AUC for a single class."""
        if self.n_pos[c] == 0 or self.n_neg[c] == 0:
            return float("nan")
        auc = 0.0
        cum_neg = 0
        for pos_cnt, neg_cnt in zip(self.pos_hist[c], self.neg_hist[c]):
            auc += pos_cnt * cum_neg  # positives outrank earlier negatives
            auc += 0.5 * pos_cnt * neg_cnt  # ties inside bin
            cum_neg += neg_cnt
        return auc / (self.n_pos[c] * self.n_neg[c])

    def score(self, include_per_class: bool = True) -> dict[str, float | list[float]]:
        """Compute ROC-AUC scores.

        Args:
            include_per_class: Whether to include per-class scores.

        Returns:
            Dictionary containing macro average and optionally per-class scores.
        """
        per_class = np.array([self._auc_one_class(c) for c in range(self.num_classes)])
        macro = np.nanmean(per_class)
        if include_per_class:
            return {
                "roc_auc_macro": macro,
                "roc_auc_per_class": per_class.tolist(),
            }
        return {"roc_auc_macro": macro}

    def reset(self) -> None:
        """Reset the histograms and counts."""
        self.pos_hist = np.zeros((self.num_classes, self.n_bins), dtype=np.int64)
        self.neg_hist = np.zeros((self.num_classes, self.n_bins), dtype=np.int64)
        self.n_pos = np.zeros(self.num_classes, dtype=np.int64)
        self.n_neg = np.zeros(self.num_classes, dtype=np.int64)


# -----------------------------------------------------------------------------
# Streaming Regression Metrics
# -----------------------------------------------------------------------------


class RunningRegressionMetrics:
    """Maintain streaming statistics for regression metrics.

    This class tracks metrics like RMSE, MAE, R2 score, Pearson correlation,
    and expected error metrics in an online fashion.

    Parameters
    ----------
    ee_bias : float, default 0.05
        Bias term for expected error calculation.
    ee_coef : float, default 0.15
        Coefficient for expected error calculation.
    """

    def __init__(
        self, ee_bias: float = 0.05, ee_coef: float = 0.15, include_ee: bool = False
    ) -> None:
        """Initialize the regression metrics tracker.

        Args:
            ee_bias: Bias term for expected error calculation.
            ee_coef: Coefficient for expected error calculation.
            include_ee: Whether to include expected error metrics.
        """
        self.ee_bias = ee_bias
        self.ee_coef = ee_coef
        self.include_ee = include_ee
        self.reset()

    def reset(self) -> None:
        """Reset all running statistics."""
        self.n = 0
        self.sum_x = 0.0
        self.sum_y = 0.0
        self.sum_xy = 0.0
        self.sum_x2 = 0.0
        self.sum_y2 = 0.0
        self.sum_abs_error = 0.0
        self.sum_squared_error = 0.0
        self.within_ee_count = 0

    def update(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """Update the running statistics with new predictions.

        Args:
            y_true: Ground truth values.
            y_pred: Predicted values.
        """
        y_true = np.asarray(y_true).ravel()
        y_pred = np.asarray(y_pred).ravel()
        if y_true.shape != y_pred.shape:
            raise ValueError("y_true and y_pred shapes differ.")

        # Update basic statistics
        self.n += y_true.size
        self.sum_x += y_true.sum()
        self.sum_y += y_pred.sum()
        self.sum_xy += (y_true * y_pred).sum()
        self.sum_x2 += (y_true * y_true).sum()
        self.sum_y2 += (y_pred * y_pred).sum()

        # Update error metrics
        abs_error = np.abs(y_pred - y_true)
        self.sum_abs_error += abs_error.sum()
        self.sum_squared_error += (abs_error * abs_error).sum()

        # Update expected error metrics
        if self.include_ee:
            expected_error = self.ee_bias + self.ee_coef * y_true
            self.within_ee_count += np.sum(abs_error <= expected_error)

    def mae(self) -> float:
        """Compute Mean Absolute Error."""
        if self.n == 0:
            return float("nan")
        return self.sum_abs_error / self.n

    def rmse(self) -> float:
        """Compute Root Mean Squared Error."""
        if self.n == 0:
            return float("nan")
        return np.sqrt(self.sum_squared_error / self.n)

    def r2_score(self) -> float:
        """Compute R2 score."""
        if self.n < 2:
            return float("nan")

        x_mean = self.sum_x / self.n

        # Calculate total sum of squares and residual sum of squares
        ss_tot = self.sum_x2 - self.n * x_mean * x_mean
        ss_res = self.sum_squared_error

        if ss_tot == 0:
            return float("nan")
        return 1 - (ss_res / ss_tot)

    def pearson_corrcoef(self) -> float:
        """Compute Pearson correlation coefficient."""
        if self.n < 2:
            return float("nan")

        x_mean = self.sum_x / self.n
        y_mean = self.sum_y / self.n

        # Calculate covariance and standard deviations
        cov_xy = self.sum_xy - self.n * x_mean * y_mean
        std_x = np.sqrt(self.sum_x2 - self.n * x_mean * x_mean)
        std_y = np.sqrt(self.sum_y2 - self.n * y_mean * y_mean)

        if std_x == 0 or std_y == 0:
            return float("nan")
        return cov_xy / (std_x * std_y)

    def ee_percentage(self) -> float:
        """Compute percentage of predictions within expected error."""
        if self.n == 0:
            return float("nan")
        return (self.within_ee_count / self.n) * 100

    def compute(self) -> dict[str, float | None]:
        """Return a dict with all regression metrics."""
        return {
            "mae": self.mae(),
            "rmse": self.rmse(),
            "r2_score": self.r2_score(),
            "pearson_corrcoef": self.pearson_corrcoef(),
            "ee_percentage": self.ee_percentage() if self.include_ee else None,
            "ee_bias": self.ee_bias,
            "ee_coef": self.ee_coef,
        }
