"""test_metrics.py
Sanity checks for metrics.py against scikit‑learn.
Run with:  python -m pytest test_metrics.py
"""

import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    jaccard_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

from instageo.model.metrics import (
    RunningAUC,
    RunningConfusionMatrix,
    RunningRegressionMetrics,
)


def generate_dummy(num_samples: int = 1000, num_classes: int = 3, seed: int = 0):
    rng = np.random.default_rng(seed)
    y_true = rng.integers(0, num_classes, size=num_samples)
    y_pred = rng.integers(0, num_classes, size=num_samples)

    # probabilities
    y_score = rng.random((num_samples, num_classes))
    y_score = y_score / y_score.sum(axis=1, keepdims=True)
    return y_true, y_pred, y_score


def generate_regression_dummy(num_samples: int = 1000, seed: int = 0):
    """Generate dummy data for regression testing."""
    rng = np.random.default_rng(seed)
    # Generate true values between 0 and 1
    y_true = rng.random(num_samples)
    # Generate predictions with some noise
    noise = rng.normal(0, 0.1, num_samples)
    y_pred = y_true + noise
    # Clip predictions to [0, 1] range
    y_pred = np.clip(y_pred, 0, 1)
    return y_true, y_pred


def test_confusion_metrics():
    y_true, y_pred, _ = generate_dummy()
    num_classes = 3

    cm = RunningConfusionMatrix(num_classes)
    # simulate streaming in 10 chunks
    chunk_size = len(y_true) // 10
    for start in range(0, len(y_true), chunk_size):
        cm.update(y_true[start : start + chunk_size], y_pred[start : start + chunk_size])

    assert np.isclose(cm.accuracy(), accuracy_score(y_true, y_pred))
    assert np.isclose(
        cm.precision().mean(),
        precision_score(y_true, y_pred, average="macro", zero_division=0),
    )
    assert np.isclose(
        cm.recall().mean(),
        recall_score(y_true, y_pred, average="macro", zero_division=0),
    )
    assert np.isclose(cm.f1().mean(), f1_score(y_true, y_pred, average="macro", zero_division=0))
    assert np.isclose(
        cm.jaccard().mean(),
        jaccard_score(y_true, y_pred, average="macro", zero_division=0),
    )


def test_running_auc():
    # Create a simple binary classification scenario
    # 10 samples with clear positive/negative cases
    y_true = np.array([1, 1, 1, 1, 1, 0, 1, 1, 0, 1])  # 5 negative, 5 positive
    y_score = np.array(
        [
            [0.1, 0.9],
            [0.2, 0.8],
            [0.3, 0.7],
            [0.4, 0.6],
            [0.45, 0.55],
            [0.55, 0.45],
            [0.6, 0.4],
            [0.7, 0.3],
            [0.8, 0.2],
            [0.9, 0.1],
        ]
    )

    # Initialize AUC calculator
    auc_stream = RunningAUC(num_classes=2, n_bins=2048)

    # Stream in two passes
    half = len(y_true) // 2
    auc_stream.update(y_true[:half], y_score[:half])
    auc_stream.update(y_true[half:], y_score[half:])

    # Get AUC score from our implementation
    auc_score = auc_stream.score(include_per_class=False)["roc_auc_macro"]

    # Get AUC score from scikit-learn
    skl_auc_score = roc_auc_score(y_true, y_score[:, 1])

    # Compare the scores
    assert auc_score == skl_auc_score == 0.75

    # Test with more ambiguous predictions
    y_score_ambiguous = np.array(
        [
            [0.4, 0.6],
            [0.45, 0.55],
            [0.5, 0.5],
            [0.55, 0.45],
            [0.6, 0.4],
            [0.4, 0.6],
            [0.45, 0.55],
            [0.5, 0.5],
            [0.55, 0.45],
            [0.6, 0.4],
        ]
    )

    # Reset and test with ambiguous predictions
    auc_stream.reset()
    auc_stream.update(y_true, y_score_ambiguous)
    auc_score_ambiguous = auc_stream.score(include_per_class=False)["roc_auc_macro"]

    # Get AUC score from scikit-learn for ambiguous case
    skl_auc_score_ambiguous = roc_auc_score(
        y_true, y_score_ambiguous[:, 1], multi_class="ovr", average="macro"
    )

    assert auc_score_ambiguous == skl_auc_score_ambiguous == 0.375


def test_regression_metrics():
    """Test regression metrics against scikit-learn equivalents."""
    # Generate dummy regression data
    y_true, y_pred = generate_regression_dummy(num_samples=1000)

    # Initialize our regression metrics
    reg_metrics = RunningRegressionMetrics(include_ee=True)

    # Simulate streaming in 10 chunks
    chunk_size = len(y_true) // 10
    for start in range(0, len(y_true), chunk_size):
        reg_metrics.update(y_true[start : start + chunk_size], y_pred[start : start + chunk_size])

    # Get metrics from our implementation
    metrics = reg_metrics.compute()

    # Compare with scikit-learn and scipy metrics
    assert np.isclose(metrics["mae"], mean_absolute_error(y_true, y_pred))
    assert np.isclose(metrics["rmse"], np.sqrt(mean_squared_error(y_true, y_pred)))
    assert np.isclose(metrics["r2_score"], r2_score(y_true, y_pred))
    assert np.isclose(metrics["pearson_corrcoef"], pearsonr(y_true, y_pred)[0])

    # Test expected error metrics
    # For a simple test, we'll verify that the EE percentage is between 0 and 100
    assert 0 <= metrics["ee_percentage"] <= 100

    # Test with perfect predictions
    reg_metrics.reset()
    reg_metrics.update(y_true, y_true)  # Perfect predictions
    perfect_metrics = reg_metrics.compute()

    assert np.isclose(perfect_metrics["mae"], 0.0)
    assert np.isclose(perfect_metrics["rmse"], 0.0)
    assert np.isclose(perfect_metrics["r2_score"], 1.0)
    assert np.isclose(perfect_metrics["pearson_corrcoef"], 1.0)

    # Test with constant predictions
    reg_metrics.reset()
    constant_pred = np.full_like(y_true, y_true.mean())
    reg_metrics.update(y_true, constant_pred)
    constant_metrics = reg_metrics.compute()

    # R2 should be 0 for constant predictions
    assert np.isclose(constant_metrics["r2_score"], 0.0)

    # Test with negative correlation
    reg_metrics.reset()
    negative_pred = 1 - y_true  # Perfect negative correlation
    reg_metrics.update(y_true, negative_pred)
    negative_metrics = reg_metrics.compute()

    assert np.isclose(negative_metrics["pearson_corrcoef"], -1.0)
