from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, ParameterSampler, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

RANDOM_STATE = 42

CATEGORICAL_FEATURES = [
    "project_domain",
    "project_status",
    "assignee_role",
    "task_type",
    "priority",
    "status",
]

NUMERIC_FEATURES = [
    "project_budget",
    "project_start_days_ago",
    "assignee_experience",
    "assignee_availability",
    "assignee_cost_rate",
    "assignee_recent_hours",
    "assignee_utilization",
    "complexity_score",
    "progress_percentage",
    "estimated_effort_hours",
    "estimated_duration_days",
    "num_dependencies",
    "dependency_depth",
    "num_risks",
    "max_risk_severity_ord",
    "avg_risk_probability",
]

FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES
ID_COLUMNS = ["task_id", "project_id", "assignee_id"]
TARGETS = ["is_delayed", "risk_level", "delay_days", "effort_overrun_ratio"]
OUTCOME_COLUMNS = ["actual_effort_hours", "actual_duration_days"] + TARGETS
REQUIRED_COLUMNS = sorted(set(FEATURES + ID_COLUMNS + TARGETS))
RISK_ORDER = ["None", "Low", "Medium", "High", "Critical"]


def validate_dataset(df: pd.DataFrame, require_targets: bool = True) -> list[str]:
    required = FEATURES + (["project_id"] + TARGETS if require_targets else [])
    missing = sorted(set(required) - set(df.columns))
    errors: list[str] = []
    if missing:
        errors.append(f"Missing required columns: {', '.join(missing)}")
    if df.empty:
        errors.append("The dataset contains no rows.")
    if "is_delayed" in df and not set(df["is_delayed"].dropna().unique()).issubset({0, 1}):
        errors.append("is_delayed must contain only 0 and 1.")
    return errors


def _preprocessor() -> ColumnTransformer:
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", min_frequency=5, sparse_output=True),
            ),
        ]
    )
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    return ColumnTransformer(
        [("cat", categorical, CATEGORICAL_FEATURES), ("num", numeric, NUMERIC_FEATURES)],
        remainder="drop",
    )


def _classifier() -> Pipeline:
    return Pipeline(
        [
            ("prep", _preprocessor()),
            (
                "model",
                RandomForestClassifier(
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                    class_weight="balanced_subsample",
                ),
            ),
        ]
    )


def _regressor() -> Pipeline:
    return Pipeline(
        [
            ("prep", _preprocessor()),
            (
                "model",
                RandomForestRegressor(
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                    criterion="squared_error",
                ),
            ),
        ]
    )


CLASSIFIER_SPACE = {
    "model__n_estimators": [300, 500, 700],
    "model__max_depth": [None, 12, 20, 30],
    "model__min_samples_leaf": [1, 2, 4, 8],
    "model__min_samples_split": [2, 5, 10],
    "model__max_features": ["sqrt", 0.6, 0.8],
}

REGRESSOR_SPACE = {
    "model__n_estimators": [300, 500, 700],
    "model__max_depth": [None, 12, 20, 30],
    "model__min_samples_leaf": [1, 2, 4, 8],
    "model__min_samples_split": [2, 5, 10],
    "model__max_features": ["sqrt", 0.6, 0.8],
}


def _split_indices(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    groups = df["project_id"].to_numpy()
    outer = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=RANDOM_STATE)
    development_idx, test_idx = next(outer.split(df, groups=groups))
    development = df.iloc[development_idx]
    inner = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=RANDOM_STATE + 1)
    train_local, val_local = next(inner.split(development, groups=development["project_id"]))
    return development_idx[train_local], development_idx[val_local], test_idx


def _threshold_from_validation(y_true: pd.Series, probabilities: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    if len(thresholds) == 0:
        return 0.5
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    acceptable = np.where(recall[:-1] >= 0.70)[0]
    idx = acceptable[np.argmax(f1[acceptable])] if len(acceptable) else int(np.argmax(f1))
    return float(np.clip(thresholds[idx], 0.15, 0.85))


@dataclass
class SearchResult:
    best_estimator_: Pipeline
    best_params_: dict[str, Any]
    best_score_: float


def _tune(
    pipeline: Pipeline,
    parameter_space: dict[str, list[Any]],
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    scoring: str,
    iterations: int,
    model_name: str,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> SearchResult:
    cv = GroupKFold(n_splits=3)
    candidates = list(ParameterSampler(parameter_space, n_iter=iterations, random_state=RANDOM_STATE))
    best_score = -np.inf
    best_params: dict[str, Any] = {}
    started = time.perf_counter()
    for index, parameters in enumerate(candidates, start=1):
        candidate = clone(pipeline).set_params(**parameters)
        scores = cross_val_score(candidate, X, y, groups=groups, cv=cv, scoring=scoring, n_jobs=-1)
        score = float(np.mean(scores))
        if score > best_score:
            best_score = score
            best_params = parameters
        if progress_callback:
            progress_callback(
                {
                    "phase": "tuning",
                    "model": model_name,
                    "iteration": index,
                    "total_iterations": len(candidates),
                    "score": score,
                    "best_score": best_score,
                    "elapsed_seconds": time.perf_counter() - started,
                }
            )
    best_estimator = clone(pipeline).set_params(**best_params)
    best_estimator.fit(X, y)
    return SearchResult(best_estimator, best_params, best_score)


def _classification_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
    labels: list[Any],
    probabilities: np.ndarray | None = None,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "labels": [str(item) for item in labels],
        "classification_report": classification_report(
            y_true, y_pred, labels=labels, output_dict=True, zero_division=0
        ),
    }
    if probabilities is not None and len(labels) == 2:
        metrics.update(
            {
                "roc_auc": float(roc_auc_score(y_true, probabilities)),
                "precision": float(precision_score(y_true, y_pred, zero_division=0)),
                "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            }
        )
    return metrics


def _regression_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "r2": float(r2_score(y_true, y_pred)),
    }


def _feature_importance(pipeline: Pipeline) -> list[dict[str, float | str]]:
    transformed = pipeline.named_steps["prep"].get_feature_names_out()
    raw = np.asarray(pipeline.named_steps["model"].feature_importances_, dtype=float)
    aggregated: dict[str, float] = {feature: 0.0 for feature in FEATURES}
    for name, value in zip(transformed, raw):
        clean = name.split("__", 1)[-1]
        source = next(
            (feature for feature in CATEGORICAL_FEATURES if clean == feature or clean.startswith(f"{feature}_")),
            clean,
        )
        aggregated[source] = aggregated.get(source, 0.0) + float(value)
    return [
        {"feature": key, "importance": value}
        for key, value in sorted(aggregated.items(), key=lambda item: item[1], reverse=True)
    ]


def _profile(df: pd.DataFrame) -> dict[str, Any]:
    profile: dict[str, Any] = {"categorical": {}, "numeric": {}}
    for column in CATEGORICAL_FEATURES:
        counts = df[column].dropna().astype(str).value_counts()
        profile["categorical"][column] = {
            "default": str(counts.index[0]),
            "options": sorted(counts.index.tolist()),
        }
    for column in NUMERIC_FEATURES:
        series = pd.to_numeric(df[column], errors="coerce")
        profile["numeric"][column] = {
            "default": float(series.median()),
            "min": float(series.min()),
            "max": float(series.max()),
            "q25": float(series.quantile(0.25)),
            "q75": float(series.quantile(0.75)),
        }
    return profile


def train_all(
    df: pd.DataFrame,
    output_dir: str | Path,
    iterations: int = 8,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    errors = validate_dataset(df, require_targets=True)
    if errors:
        raise ValueError(" ".join(errors))

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    df = df.drop_duplicates().reset_index(drop=True)
    train_idx, val_idx, test_idx = _split_indices(df)
    X = df[FEATURES]
    groups = df["project_id"]

    model_specs = {
        "delay_classifier": (_classifier(), CLASSIFIER_SPACE, "is_delayed", "roc_auc", None),
        "risk_classifier": (_classifier(), CLASSIFIER_SPACE, "risk_level", "f1_macro", None),
        "delay_days_regressor": (_regressor(), REGRESSOR_SPACE, "delay_days", "neg_mean_absolute_error", df["is_delayed"].eq(1)),
        "effort_overrun_regressor": (_regressor(), REGRESSOR_SPACE, "effort_overrun_ratio", "neg_mean_absolute_error", None),
    }

    models: dict[str, Pipeline] = {}
    metrics: dict[str, Any] = {}
    importance: dict[str, Any] = {}
    predictions = df.loc[test_idx, [column for column in ID_COLUMNS if column in df]].copy()

    model_count = len(model_specs)
    for model_index, (name, (pipeline, space, target, scoring, subset_mask)) in enumerate(model_specs.items()):
        train_rows = train_idx if subset_mask is None else train_idx[subset_mask.iloc[train_idx].to_numpy()]
        val_rows = val_idx if subset_mask is None else val_idx[subset_mask.iloc[val_idx].to_numpy()]
        test_rows = test_idx if subset_mask is None else test_idx[subset_mask.iloc[test_idx].to_numpy()]

        def report_iteration(event: dict[str, Any]) -> None:
            if progress_callback:
                event["model_index"] = model_index + 1
                event["model_count"] = model_count
                event["overall_fraction"] = (
                    model_index * iterations + int(event["iteration"])
                ) / (model_count * iterations)
                progress_callback(event)

        search = _tune(
            pipeline,
            space,
            X.iloc[train_rows],
            df[target].iloc[train_rows],
            groups.iloc[train_rows],
            scoring,
            iterations,
            name,
            report_iteration,
        )

        threshold = None
        if name == "delay_classifier":
            val_probability = search.best_estimator_.predict_proba(X.iloc[val_rows])[:, 1]
            threshold = _threshold_from_validation(df[target].iloc[val_rows], val_probability)

        final_rows = np.concatenate([train_rows, val_rows])
        best_model = search.best_estimator_.set_params(**search.best_params_)
        best_model.fit(X.iloc[final_rows], df[target].iloc[final_rows])
        models[name] = best_model
        importance[name] = _feature_importance(best_model)

        y_test = df[target].iloc[test_rows]
        if name == "delay_classifier":
            probability = best_model.predict_proba(X.iloc[test_rows])[:, 1]
            predicted = (probability >= threshold).astype(int)
            result = _classification_metrics(y_test, predicted, [0, 1], probability)
            result["decision_threshold"] = threshold
            predictions.loc[test_rows, "actual_is_delayed"] = y_test
            predictions.loc[test_rows, "delay_probability"] = probability
            predictions.loc[test_rows, "predicted_is_delayed"] = predicted
        elif name == "risk_classifier":
            predicted = best_model.predict(X.iloc[test_rows])
            labels = [label for label in RISK_ORDER if label in set(df[target].astype(str))]
            result = _classification_metrics(y_test, predicted, labels)
            predictions.loc[test_rows, "actual_risk_level"] = y_test
            predictions.loc[test_rows, "predicted_risk_level"] = predicted
        else:
            predicted = np.clip(best_model.predict(X.iloc[test_rows]), 0, None)
            result = _regression_metrics(y_test, predicted)
            predictions.loc[test_rows, f"actual_{target}"] = y_test
            predictions.loc[test_rows, f"predicted_{target}"] = predicted

        result.update(
            {
                "best_cv_score": float(search.best_score_),
                "best_parameters": search.best_params_,
                "train_rows": int(len(train_rows)),
                "validation_rows": int(len(val_rows)),
                "test_rows": int(len(test_rows)),
            }
        )
        metrics[name] = result
        if progress_callback:
            progress_callback(
                {
                    "phase": "model_complete",
                    "model": name,
                    "model_index": model_index + 1,
                    "model_count": model_count,
                    "overall_fraction": (model_index + 1) / model_count,
                    "message": f"Completed {name.replace('_', ' ')}",
                }
            )

    dataset_bytes = df.to_csv(index=False).encode("utf-8")
    bundle = {
        "models": models,
        "features": FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "threshold": metrics["delay_classifier"]["decision_threshold"],
        "risk_order": RISK_ORDER,
        "profile": _profile(df),
        "feature_importance": importance,
        "metadata": {
            "trained_at_utc": datetime.now(timezone.utc).isoformat(),
            "random_state": RANDOM_STATE,
            "rows": int(len(df)),
            "features": len(FEATURES),
            "projects": int(df["project_id"].nunique()),
            "dataset_sha256": hashlib.sha256(dataset_bytes).hexdigest(),
            "split_strategy": "Project-grouped 64/16/20 train/validation/test split",
            "tuning": f"Random parameter search, {iterations} candidates, 3-fold GroupKFold",
            "leakage_controls": "Project grouping; IDs, targets, actual effort, and actual duration excluded",
        },
    }
    joblib.dump(bundle, output / "planwise_models.joblib", compress=3)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (output / "feature_importance.json").write_text(json.dumps(importance, indent=2), encoding="utf-8")
    predictions.to_csv(output / "test_predictions.csv", index=False)
    return {"metrics": metrics, "bundle": bundle}


def load_bundle(path: str | Path) -> dict[str, Any]:
    return joblib.load(path)


def score_tasks(df: pd.DataFrame, bundle: dict[str, Any]) -> pd.DataFrame:
    errors = validate_dataset(df, require_targets=False)
    if errors:
        raise ValueError(" ".join(errors))
    X = df[bundle["features"]]
    scored = df.copy()
    delay = bundle["models"]["delay_classifier"]
    risk = bundle["models"]["risk_classifier"]
    delay_reg = bundle["models"]["delay_days_regressor"]
    overrun_reg = bundle["models"]["effort_overrun_regressor"]
    scored["predicted_delay_probability"] = delay.predict_proba(X)[:, 1]
    scored["predicted_is_delayed"] = (
        scored["predicted_delay_probability"] >= bundle["threshold"]
    ).astype(int)
    risk_probabilities = risk.predict_proba(X)
    scored["predicted_risk_level"] = risk.predict(X)
    scored["predicted_risk_confidence"] = risk_probabilities.max(axis=1)
    scored["predicted_delay_days_if_delayed"] = np.clip(delay_reg.predict(X), 0, None)
    scored["predicted_effort_overrun_ratio"] = np.clip(overrun_reg.predict(X), 0, None)
    scored["project_health_score"] = calculate_health(scored)
    scored["recommended_action"] = scored.apply(recommendation_for_row, axis=1)
    return scored


def calculate_health(df: pd.DataFrame) -> pd.Series:
    risk_penalty = df["predicted_risk_level"].map(
        {"None": 0, "Low": 8, "Medium": 20, "High": 34, "Critical": 48}
    ).fillna(15)
    utilization_penalty = np.clip((df["assignee_utilization"] - 0.75) * 45, 0, 18)
    dependency_penalty = np.clip(df["dependency_depth"] * 1.5, 0, 10)
    raw = 100 - 42 * df["predicted_delay_probability"] - risk_penalty - utilization_penalty - dependency_penalty
    return raw.clip(0, 100).round(1)


def recommendation_for_row(row: pd.Series) -> str:
    actions: list[str] = []
    if row.get("predicted_delay_probability", 0) >= 0.70:
        actions.append("Create a recovery plan and review the deadline")
    elif row.get("predicted_delay_probability", 0) >= 0.45:
        actions.append("Add an early checkpoint and monitor progress")
    if row.get("assignee_utilization", 0) >= 0.85:
        actions.append("Rebalance work from the assigned resource")
    if row.get("assignee_availability", 1) < 0.55:
        actions.append("Confirm backup capacity")
    if row.get("num_dependencies", 0) >= 3 or row.get("dependency_depth", 0) >= 6:
        actions.append("Resolve or parallelize blocking dependencies")
    if row.get("num_risks", 0) >= 2 or row.get("avg_risk_probability", 0) >= 0.60:
        actions.append("Assign a risk owner and mitigation date")
    return "; ".join(actions[:3]) if actions else "Continue current plan and monitor at the next checkpoint"


def health_label(score: float) -> str:
    if score >= 75:
        return "Healthy"
    if score >= 50:
        return "Watch"
    return "Critical"
