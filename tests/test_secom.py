from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.analysis import load_secom, secom_classification, secom_model_comparison, secom_quality, secom_sensor_missingness, secom_temporal_model_comparison

SECOM = Path(__file__).parents[1] / "data" / "raw" / "uci_secom"


def test_official_secom_adapter_preserves_shape_and_label_semantics():
    data, labels = load_secom(SECOM)
    quality = secom_quality(data)
    assert quality["rows"] == 1567
    # UCI reports 591 features including the outcome; secom.data contains 590 sensors.
    assert quality["sensors"] == 590
    assert quality["failures"] == 104
    assert labels["timestamp"].dt.year.min() == 2008
    assert set(data["label"].unique()) == {-1, 1}


def test_secom_classification_uses_stratified_holdout_and_balanced_metrics():
    data, _ = load_secom(SECOM)
    metrics, predictions = secom_classification(data)
    assert metrics["train_rows"] + metrics["test_rows"] == 1567
    assert metrics["features_after_constant_filter"] < 591
    assert 0 <= metrics["balanced_accuracy"] <= 1
    assert len(predictions) == metrics["test_rows"]


def test_secom_missingness_helper_returns_sorted_sensor_rates():
    data, _ = load_secom(SECOM)
    missingness = secom_sensor_missingness(data, top_n=5)
    assert list(missingness.columns) == ["sensor", "missing_rate"]
    assert len(missingness) == 5
    assert missingness["missing_rate"].is_monotonic_decreasing


def test_secom_model_comparison_uses_fixed_models_and_stratified_folds():
    data, _ = load_secom(SECOM)
    summary, folds = secom_model_comparison(data, n_splits=2, n_repeats=1)
    assert set(summary["model"]) == {"dummy_prior", "logistic_regression", "pca_logistic", "random_forest"}
    assert len(folds) == 8
    assert folds.groupby("model")["fold"].nunique().eq(2).all()
    assert folds[["pr_auc", "balanced_accuracy", "roc_auc"]].notna().all().all()
    dummy = summary.loc[summary["model"] == "dummy_prior", "pr_auc_mean"].iloc[0]
    assert dummy == pytest.approx(data["fail"].mean(), abs=0.01)


def test_temporal_model_comparison_uses_future_only_timestamp_boundaries():
    rng = np.random.default_rng(42)
    rows = 90
    fail = np.tile([0, 0, 0, 0, 1], rows // 5)
    data = pd.DataFrame(
        {
            "sensor_001": fail + rng.normal(0, 0.5, rows),
            "sensor_002": rng.normal(0, 1, rows),
            "sensor_003": rng.normal(0, 1, rows),
            "timestamp": pd.date_range("2025-01-01", periods=rows, freq="h"),
            "fail": fail,
        }
    )

    summary, folds = secom_temporal_model_comparison(data, n_splits=3)

    assert set(summary["model"]) == {"dummy_prior", "logistic_regression", "pca_logistic", "random_forest"}
    assert set(summary["validation_strategy"]) == {"temporal_expanding"}
    assert len(folds) == 12
    assert (folds["train_end"] < folds["test_start"]).all()
    assert folds.groupby("model")["fold"].nunique().eq(3).all()
