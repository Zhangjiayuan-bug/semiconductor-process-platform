import numpy as np
import pandas as pd
import pytest

from src.analysis import (
    benjamini_hochberg,
    capability,
    detect_anomalies,
    drift_report,
    population_stability_index,
    secom_report,
    secom_sensor_association,
    spc_chart,
)


def test_capability_returns_expected_mean_and_cpk():
    result = capability(pd.Series([95, 96, 97, 96]), 90, 100)
    assert result["mean"] == 96.0
    assert result["cpk"] > 0


def test_spc_marks_outlier():
    result = spc_chart(pd.Series([10] * 30 + [20]))
    assert bool(result.iloc[-1]["out_of_control"])


def test_imr_limits_match_published_constants():
    result = spc_chart(pd.Series([0, 1, 0, 1, 0, 3.2]), baseline_size=5)
    assert result.iloc[0]["center"] == pytest.approx(0.4)
    assert result.iloc[0]["mr_center"] == pytest.approx(1.0)
    assert result.iloc[0]["ucl"] == pytest.approx(3.059574468)
    assert result.iloc[0]["lcl"] == pytest.approx(-2.259574468)
    assert result.iloc[0]["mr_ucl"] == pytest.approx(3.267)
    assert result.iloc[-1]["rules"] == "WE-1"


def test_imr_detects_western_electric_run_rule():
    result = spc_chart(pd.Series([0, 1, 0, 1, 0] + [0.5] * 8), baseline_size=5)
    assert "WE-4" in result.iloc[-1]["rules"]
    assert bool(result.iloc[-1]["individual_signal"])


def test_anomaly_detector_marks_injected_point():
    data = pd.DataFrame({"temperature": np.r_[np.ones(60), 20], "pressure": np.r_[np.ones(60), 20]})
    result = detect_anomalies(data, contamination=0.05)
    assert bool(result.iloc[0]["anomaly"])


def test_secom_report_returns_quality_and_predictions():
    data = pd.DataFrame(
        {
            "sensor_001": [1.0, 2.0, np.nan, 4.0],
            "sensor_002": [5.0, 6.0, 7.0, 8.0],
            "fail": [0, 1, 0, 1],
        }
    )
    report = secom_report(data, test_size=0.5)
    assert report["quality"]["rows"] == 4
    assert list(report["predictions"].columns) == ["actual_fail", "predicted_fail", "failure_probability"]


def test_secom_report_rejects_single_class():
    data = pd.DataFrame({"sensor_001": [1.0, 2.0, 3.0, 4.0], "fail": [0, 0, 0, 0]})
    with pytest.raises(ValueError, match="requires both pass and fail"):
        secom_report(data, test_size=0.5)


def test_population_stability_index_detects_shift():
    reference = pd.Series(np.linspace(0, 1, 200))
    current = pd.Series(np.linspace(2, 3, 200))
    assert population_stability_index(reference, current) > 1


def test_drift_report_returns_machine_readable_status():
    reference = pd.DataFrame({"temperature": np.linspace(100, 110, 120), "constant": 1})
    current = pd.DataFrame({"temperature": np.linspace(125, 135, 120), "constant": 1})
    result = drift_report(reference, current)
    assert list(result.columns) == [
        "feature",
        "reference_mean",
        "current_mean",
        "mean_delta",
        "reference_missing_pct",
        "current_missing_pct",
        "psi",
        "ks_statistic",
        "ks_pvalue",
        "status",
    ]
    assert result.loc[result["feature"] == "temperature", "status"].iloc[0] == "drift"
    assert "constant" not in set(result["feature"])


def test_benjamini_hochberg_is_monotonic_in_pvalue_order():
    pvalues = pd.Series([0.04, 0.001, 0.02, np.nan])
    adjusted = benjamini_hochberg(pvalues)
    ordered = adjusted[pvalues.dropna().sort_values().index]
    assert ordered.is_monotonic_increasing
    assert adjusted.iloc[1] == pytest.approx(0.003)
    assert pd.isna(adjusted.iloc[3])


def test_sensor_association_ranks_shifted_sensor():
    rng = np.random.default_rng(42)
    data = pd.DataFrame(
        {
            "sensor_001": np.r_[rng.normal(0, 1, 80), rng.normal(3, 1, 40)],
            "sensor_002": rng.normal(0, 1, 120),
            "fail": [0] * 80 + [1] * 40,
        }
    )
    result = secom_sensor_association(data)
    assert result.iloc[0]["sensor"] == "sensor_001"
    assert result.iloc[0]["pvalue_bh"] < 0.05
    assert result.iloc[0]["effect_size"] > 1
