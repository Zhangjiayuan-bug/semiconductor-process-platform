from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, ttest_ind
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import IsolationForest, RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, balanced_accuracy_score, make_scorer, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, TimeSeriesSplit, cross_validate, train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

REQUIRED = {"lot_id", "equipment", "process_step", "yield_pct"}


def validate_data(frame: pd.DataFrame) -> list[str]:
    return sorted(REQUIRED - set(frame.columns))


def clean_data(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    data.columns = [str(c).strip().lower().replace(" ", "_") for c in data.columns]
    for column in data.select_dtypes(include=["object", "string"]):
        data[column] = data[column].astype(str).str.strip()
    numeric = data.select_dtypes(include=np.number).columns
    data[numeric] = data[numeric].replace([np.inf, -np.inf], np.nan)
    if "yield_pct" in data:
        data = data.dropna(subset=["yield_pct"])
    return data.reset_index(drop=True)


def load_secom(root: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the official UCI SECOM sensor matrix and aligned pass/fail labels."""
    root = Path(root)
    sensor_path = root / "secom.data"
    label_path = root / "secom_labels.data"
    sensors = pd.read_csv(sensor_path, sep=r"\s+", header=None, na_values="NaN", engine="python")
    sensors.columns = [f"sensor_{i + 1:03d}" for i in range(sensors.shape[1])]
    label_rows = []
    with label_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            parts = line.strip().split(maxsplit=1)
            if len(parts) != 2:
                raise ValueError(f"SECOM label line {line_number} is malformed")
            label_rows.append((int(parts[0]), parts[1].strip().strip('"')))
    labels = pd.DataFrame(label_rows, columns=["label", "timestamp"])
    if len(sensors) != len(labels):
        raise ValueError(f"SECOM row mismatch: {len(sensors)} sensor rows vs {len(labels)} labels")
    labels["timestamp"] = pd.to_datetime(labels["timestamp"], errors="coerce", dayfirst=True)
    if labels["timestamp"].isna().any():
        raise ValueError("SECOM contains an invalid timestamp")
    if not labels["label"].isin([-1, 1]).all():
        raise ValueError("SECOM labels must be -1 (pass) or 1 (fail)")
    data = pd.concat([sensors, labels], axis=1)
    data = pd.concat([data, (data["label"] == 1).rename("fail")], axis=1)
    return data, labels


def secom_quality(data: pd.DataFrame) -> dict[str, float | int]:
    sensor_cols = [c for c in data.columns if c.startswith("sensor_")]
    missing = int(data[sensor_cols].isna().sum().sum())
    return {
        "rows": int(len(data)),
        "sensors": int(len(sensor_cols)),
        "failures": int((data["fail"] == 1).sum()),
        "missing_values": missing,
        "missing_rate_pct": missing / max(1, len(data) * len(sensor_cols)) * 100,
    }


def secom_sensor_missingness(data: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    sensor_cols = [c for c in data.columns if c.startswith("sensor_")]
    if not sensor_cols:
        return pd.DataFrame(columns=["sensor", "missing_rate"])
    missingness = data[sensor_cols].isna().mean().sort_values(ascending=False).head(top_n)
    result = missingness.rename("missing_rate").reset_index()
    result.columns = ["sensor", "missing_rate"]
    return result


def secom_report(data: pd.DataFrame, test_size: float = 0.25) -> dict[str, object]:
    quality = secom_quality(data)
    metrics, predictions = secom_classification(data, test_size=test_size)
    return {
        "quality": quality,
        "missingness": secom_sensor_missingness(data),
        "metrics": metrics,
        "predictions": predictions,
    }


def population_stability_index(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    ref = pd.to_numeric(reference, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    cur = pd.to_numeric(current, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(ref) == 0 or len(cur) == 0 or ref.nunique() < 2:
        return np.nan
    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.unique(np.nanquantile(ref, quantiles))
    if len(edges) < 3:
        return np.nan
    edges[0] = -np.inf
    edges[-1] = np.inf
    ref_counts = pd.cut(ref, edges, include_lowest=True).value_counts(sort=False).to_numpy(dtype=float)
    cur_counts = pd.cut(cur, edges, include_lowest=True).value_counts(sort=False).to_numpy(dtype=float)
    ref_pct = np.clip(ref_counts / ref_counts.sum(), 1e-6, None)
    cur_pct = np.clip(cur_counts / cur_counts.sum(), 1e-6, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def drift_report(reference: pd.DataFrame, current: pd.DataFrame, columns: list[str] | None = None, bins: int = 10) -> pd.DataFrame:
    if columns is None:
        shared = sorted(set(reference.select_dtypes(include=np.number).columns) & set(current.select_dtypes(include=np.number).columns))
    else:
        shared = [c for c in columns if c in reference.columns and c in current.columns]
    rows = []
    for column in shared:
        ref = pd.to_numeric(reference[column], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        cur = pd.to_numeric(current[column], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if len(ref) == 0 or len(cur) == 0:
            continue
        if ref.nunique() < 2 and cur.nunique() < 2:
            continue
        ks = ks_2samp(ref, cur)
        psi = population_stability_index(ref, cur, bins=bins)
        rows.append(
            {
                "feature": column,
                "reference_mean": float(ref.mean()),
                "current_mean": float(cur.mean()),
                "mean_delta": float(cur.mean() - ref.mean()),
                "reference_missing_pct": float(reference[column].isna().mean() * 100),
                "current_missing_pct": float(current[column].isna().mean() * 100),
                "psi": psi,
                "ks_statistic": float(ks.statistic),
                "ks_pvalue": float(ks.pvalue),
                "status": "drift"
                if (pd.notna(psi) and psi >= 0.25) or ks.pvalue < 0.01
                else "watch"
                if (pd.notna(psi) and psi >= 0.1) or ks.pvalue < 0.05
                else "stable",
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=[
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
        )
    return pd.DataFrame(rows).sort_values(["status", "psi", "ks_statistic"], ascending=[True, False, False]).reset_index(drop=True)


def benjamini_hochberg(pvalues: pd.Series) -> pd.Series:
    values = pd.to_numeric(pvalues, errors="coerce")
    adjusted = pd.Series(np.nan, index=values.index, dtype=float)
    valid = values.dropna().sort_values()
    if valid.empty:
        return adjusted
    n = len(valid)
    ranks = np.arange(1, n + 1)
    ranked = (valid.to_numpy() * n / ranks)[::-1]
    corrected = np.minimum.accumulate(ranked)[::-1]
    adjusted.loc[valid.index] = np.clip(corrected, 0, 1)
    return adjusted


def secom_sensor_association(data: pd.DataFrame, top_n: int = 30) -> pd.DataFrame:
    sensor_cols = [c for c in data.columns if c.startswith("sensor_")]
    rows = []
    for column in sensor_cols:
        passed = pd.to_numeric(data.loc[data["fail"] == 0, column], errors="coerce").dropna()
        failed = pd.to_numeric(data.loc[data["fail"] == 1, column], errors="coerce").dropna()
        if len(passed) < 2 or len(failed) < 2 or passed.nunique() < 2 and failed.nunique() < 2:
            continue
        statistic, pvalue = ttest_ind(failed, passed, equal_var=False, nan_policy="omit")
        pooled = np.sqrt((failed.var(ddof=1) + passed.var(ddof=1)) / 2)
        effect = (failed.mean() - passed.mean()) / pooled if pooled > 0 else np.nan
        rows.append(
            {
                "sensor": column,
                "pass_mean": float(passed.mean()),
                "fail_mean": float(failed.mean()),
                "mean_delta": float(failed.mean() - passed.mean()),
                "effect_size": float(effect),
                "welch_t": float(statistic),
                "pvalue": float(pvalue),
                "pass_count": int(len(passed)),
                "fail_count": int(len(failed)),
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=["sensor", "pass_mean", "fail_mean", "mean_delta", "effect_size", "welch_t", "pvalue", "pvalue_bh", "pass_count", "fail_count"]
        )
    result = pd.DataFrame(rows)
    result["pvalue_bh"] = benjamini_hochberg(result["pvalue"])
    result["abs_effect_size"] = result["effect_size"].abs()
    result = result.sort_values(["pvalue_bh", "abs_effect_size"], ascending=[True, False]).head(top_n)
    return result.drop(columns="abs_effect_size").reset_index(drop=True)


def secom_classification(data: pd.DataFrame, test_size: float = 0.25) -> tuple[dict[str, float], pd.DataFrame]:
    """Leakage-aware baseline: remove constant sensors, impute/scale in a pipeline, stratify split."""
    sensor_cols = [c for c in data.columns if c.startswith("sensor_")]
    x = data[sensor_cols].replace([np.inf, -np.inf], np.nan)
    y = data["fail"]
    if y.nunique(dropna=True) < 2:
        raise ValueError("SECOM classification requires both pass and fail labels")
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, stratify=y, random_state=42)
    model = Pipeline(
        [
            ("remove_constant", VarianceThreshold()),
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=2500, class_weight="balanced", random_state=42)),
        ]
    )
    model.fit(x_train, y_train)
    probability = model.predict_proba(x_test)[:, 1]
    predicted = (probability >= 0.5).astype(int)
    metrics = {
        "balanced_accuracy": float(balanced_accuracy_score(y_test, predicted)),
        "precision": float(precision_score(y_test, predicted, zero_division=0)),
        "recall": float(recall_score(y_test, predicted, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probability)),
        "pr_auc": float(average_precision_score(y_test, probability)),
        "train_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
        "features_after_constant_filter": int(model.named_steps["remove_constant"].get_support().sum()),
    }
    predictions = pd.DataFrame({"actual_fail": y_test.to_numpy(), "predicted_fail": predicted, "failure_probability": probability})
    return metrics, predictions


def _secom_models() -> dict[str, Pipeline]:
    def preprocessing() -> list[tuple[str, object]]:
        return [("remove_constant", VarianceThreshold()), ("imputer", SimpleImputer(strategy="median", add_indicator=True))]

    return {
        "dummy_prior": Pipeline(preprocessing() + [("classifier", DummyClassifier(strategy="prior"))]),
        "logistic_regression": Pipeline(
            preprocessing() + [("scale", StandardScaler()), ("classifier", LogisticRegression(max_iter=2500, class_weight="balanced", random_state=42))]
        ),
        "pca_logistic": Pipeline(
            preprocessing()
            + [
                ("scale", StandardScaler()),
                ("pca", PCA(n_components=0.95, svd_solver="full")),
                ("classifier", LogisticRegression(max_iter=2500, class_weight="balanced", random_state=42)),
            ]
        ),
        "random_forest": Pipeline(
            preprocessing()
            + [
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=160, max_features="sqrt", min_samples_leaf=2, class_weight="balanced_subsample", random_state=42, n_jobs=1
                    ),
                )
            ]
        ),
    }


def _evaluate_secom_models(
    x: pd.DataFrame,
    y: pd.Series,
    splits: list[tuple[np.ndarray, np.ndarray]],
    fold_metadata: list[dict[str, object]],
    strategy: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    scoring = {
        "pr_auc": "average_precision",
        "balanced_accuracy": "balanced_accuracy",
        "roc_auc": "roc_auc",
        "recall": make_scorer(recall_score, zero_division=0),
        "precision": make_scorer(precision_score, zero_division=0),
    }
    fold_rows = []
    for model_name, model in _secom_models().items():
        scores = cross_validate(model, x, y, cv=splits, scoring=scoring, n_jobs=1, error_score="raise")
        for fold, metadata in enumerate(fold_metadata):
            row = {
                "validation_strategy": strategy,
                "model": model_name,
                "fold": fold + 1,
                "fit_time_seconds": float(scores["fit_time"][fold]),
                **metadata,
            }
            row.update({metric: float(scores[f"test_{metric}"][fold]) for metric in scoring})
            fold_rows.append(row)
    folds = pd.DataFrame(fold_rows)
    summary = folds.groupby(["validation_strategy", "model"], as_index=False).agg(
        pr_auc_mean=("pr_auc", "mean"),
        pr_auc_std=("pr_auc", "std"),
        balanced_accuracy_mean=("balanced_accuracy", "mean"),
        roc_auc_mean=("roc_auc", "mean"),
        recall_mean=("recall", "mean"),
        precision_mean=("precision", "mean"),
        fit_time_mean_seconds=("fit_time_seconds", "mean"),
        validation_folds=("fold", "count"),
    )
    summary["pr_auc_ci95_half_width"] = 1.96 * summary["pr_auc_std"] / np.sqrt(summary["validation_folds"])
    return summary.sort_values("pr_auc_mean", ascending=False).reset_index(drop=True), folds


def secom_model_comparison(data: pd.DataFrame, n_splits: int = 5, n_repeats: int = 2) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare fixed baselines with preprocessing fitted inside repeated stratified CV."""
    sensor_cols = [c for c in data.columns if c.startswith("sensor_")]
    x = data[sensor_cols].replace([np.inf, -np.inf], np.nan)
    y = data["fail"].astype(int)
    if y.nunique() < 2:
        raise ValueError("SECOM model comparison requires both pass and fail labels")
    if int(y.value_counts().min()) < n_splits:
        raise ValueError("Each class must contain at least n_splits observations")

    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)
    splits = list(cv.split(x, y))
    fold_metadata = []
    for fold, (train_index, test_index) in enumerate(splits):
        fold_metadata.append(
            {
                "repeat": fold // n_splits + 1,
                "split": fold % n_splits + 1,
                "train_rows": len(train_index),
                "test_rows": len(test_index),
                "train_failure_rate": float(y.iloc[train_index].mean()),
                "test_failure_rate": float(y.iloc[test_index].mean()),
            }
        )
    return _evaluate_secom_models(x, y, splits, fold_metadata, "repeated_stratified")


def secom_temporal_model_comparison(data: pd.DataFrame, n_splits: int = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate future-like performance with expanding windows and timestamp-safe boundaries."""
    if "timestamp" not in data:
        raise ValueError("Temporal validation requires a timestamp column")
    ordered = data.sort_values("timestamp", kind="stable").reset_index(drop=True)
    timestamps = pd.to_datetime(ordered["timestamp"], errors="coerce")
    if timestamps.isna().any():
        raise ValueError("Temporal validation requires valid timestamps")
    unique_timestamps = timestamps.drop_duplicates().to_numpy()
    if len(unique_timestamps) <= n_splits:
        raise ValueError("Temporal validation requires more unique timestamps than splits")

    sensor_cols = [c for c in ordered.columns if c.startswith("sensor_")]
    x = ordered[sensor_cols].replace([np.inf, -np.inf], np.nan)
    y = ordered["fail"].astype(int)
    timestamp_splitter = TimeSeriesSplit(n_splits=n_splits)
    splits = []
    fold_metadata = []
    for train_times, test_times in timestamp_splitter.split(unique_timestamps):
        train_index = np.flatnonzero(timestamps.isin(unique_timestamps[train_times]))
        test_index = np.flatnonzero(timestamps.isin(unique_timestamps[test_times]))
        if y.iloc[train_index].nunique() < 2 or y.iloc[test_index].nunique() < 2:
            raise ValueError("Each temporal train and test window must contain both pass and fail labels")
        splits.append((train_index, test_index))
        fold_metadata.append(
            {
                "repeat": 1,
                "split": len(splits),
                "train_rows": len(train_index),
                "test_rows": len(test_index),
                "train_failure_rate": float(y.iloc[train_index].mean()),
                "test_failure_rate": float(y.iloc[test_index].mean()),
                "train_end": timestamps.iloc[train_index].max(),
                "test_start": timestamps.iloc[test_index].min(),
                "test_end": timestamps.iloc[test_index].max(),
            }
        )
    return _evaluate_secom_models(x, y, splits, fold_metadata, "temporal_expanding")


def secom_validation_crosscheck(data: pd.DataFrame, n_splits: int = 5, n_repeats: int = 2) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run IID and future-like validation under the same model and metric definitions."""
    random_summary, random_folds = secom_model_comparison(data, n_splits=n_splits, n_repeats=n_repeats)
    temporal_summary, temporal_folds = secom_temporal_model_comparison(data, n_splits=n_splits)
    summary = pd.concat([random_summary, temporal_summary], ignore_index=True)
    folds = pd.concat([random_folds, temporal_folds], ignore_index=True)
    return summary, folds


def capability(values: pd.Series, lsl: float, usl: float) -> dict[str, float]:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    mean = float(clean.mean())
    std = float(clean.std(ddof=1)) if len(clean) > 1 else 0.0
    cp = (usl - lsl) / (6 * std) if std > 0 else np.nan
    cpk = min((usl - mean) / (3 * std), (mean - lsl) / (3 * std)) if std > 0 else np.nan
    return {"count": int(len(clean)), "mean": mean, "std": std, "cp": cp, "cpk": cpk}


def spc_chart(values: pd.Series, baseline_size: int = 25) -> pd.DataFrame:
    """Build a Phase-I/II Individuals-Moving Range chart with Western Electric rules."""
    series = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).reset_index(drop=True)
    if len(series) < 2:
        raise ValueError("I-MR chart requires at least two observations")
    baseline_size = min(int(baseline_size), len(series))
    if baseline_size < 2 or series.iloc[:baseline_size].isna().any():
        raise ValueError("I-MR baseline requires at least two contiguous numeric observations")
    baseline = series.iloc[:baseline_size]
    baseline_mr = baseline.diff().abs().iloc[1:]
    center = float(baseline.mean())
    mr_center = float(baseline_mr.mean())
    sigma = mr_center / 1.128
    result = pd.DataFrame({"sample": np.arange(1, len(series) + 1), "value": series})
    result["phase"] = np.where(result.index < baseline_size, "baseline", "monitoring")
    result["moving_range"] = series.diff().abs()
    result["center"] = center
    result["ucl"] = center + 3 * sigma
    result["lcl"] = center - 3 * sigma
    result["upper_2sigma"] = center + 2 * sigma
    result["lower_2sigma"] = center - 2 * sigma
    result["upper_1sigma"] = center + sigma
    result["lower_1sigma"] = center - sigma
    result["mr_center"] = mr_center
    result["mr_ucl"] = 3.267 * mr_center
    result["mr_lcl"] = 0.0
    rules: list[set[str]] = [set() for _ in range(len(result))]

    for i in range(baseline_size, len(result)):
        value = series.iloc[i]
        if pd.isna(value):
            continue
        if value > result.at[i, "ucl"] or value < result.at[i, "lcl"]:
            rules[i].add("WE-1")
        if pd.notna(result.at[i, "moving_range"]) and result.at[i, "moving_range"] > result.at[i, "mr_ucl"]:
            rules[i].add("MR-WE-1")
        phase_values = series.iloc[baseline_size : i + 1]
        if len(phase_values) >= 3:
            window = phase_values.iloc[-3:]
            if (window > center + 2 * sigma).sum() >= 2 or (window < center - 2 * sigma).sum() >= 2:
                rules[i].add("WE-2")
        if len(phase_values) >= 5:
            window = phase_values.iloc[-5:]
            if (window > center + sigma).sum() >= 4 or (window < center - sigma).sum() >= 4:
                rules[i].add("WE-3")
        if len(phase_values) >= 8:
            window = phase_values.iloc[-8:]
            if (window > center).all() or (window < center).all():
                rules[i].add("WE-4")
    result["rules"] = [",".join(sorted(item)) for item in rules]
    result["individual_signal"] = [bool(item & {"WE-1", "WE-2", "WE-3", "WE-4"}) for item in rules]
    result["moving_range_signal"] = ["MR-WE-1" in item for item in rules]
    result["out_of_control"] = result["individual_signal"] | result["moving_range_signal"]
    return result


def detect_anomalies(data: pd.DataFrame, contamination: float = 0.05) -> pd.DataFrame:
    result = data.copy()
    numeric = result.select_dtypes(include=np.number).columns.tolist()
    numeric = [c for c in numeric if c not in {"yield_pct", "defect_count"}]
    if not numeric:
        result["anomaly"] = False
        result["anomaly_score"] = 0.0
        return result
    features = result[numeric].fillna(result[numeric].median())
    model = IsolationForest(contamination=contamination, random_state=42)
    result["anomaly"] = model.fit_predict(features) == -1
    result["anomaly_score"] = -model.score_samples(features)
    return result.sort_values("anomaly_score", ascending=False)


def feature_importance(data: pd.DataFrame) -> pd.DataFrame:
    target = "yield_pct"
    candidates = [c for c in data.columns if c != target and c not in {"lot_id", "wafer_id"}]
    usable = data[candidates + [target]].dropna()
    if len(usable) < 20 or not candidates:
        return pd.DataFrame(columns=["feature", "importance"])
    categorical = usable[candidates].select_dtypes(exclude=np.number).columns.tolist()
    numeric = [c for c in candidates if c not in categorical]
    preprocessor = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore"), categorical), ("num", "passthrough", numeric)],
        remainder="drop",
    )
    model = make_pipeline(preprocessor, RandomForestRegressor(n_estimators=160, random_state=42, min_samples_leaf=3))
    model.fit(usable[candidates], usable[target])
    names = model.named_steps["columntransformer"].get_feature_names_out()
    importances = model[-1].feature_importances_
    output = pd.DataFrame({"feature": names, "importance": importances})
    output["feature"] = output["feature"].str.replace(r"^(cat|num)__", "", regex=True)
    return output.groupby("feature", as_index=False)["importance"].sum().sort_values("importance", ascending=False)
