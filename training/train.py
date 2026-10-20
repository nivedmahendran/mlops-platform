"""Training job for the MLOps platform.

This is stage 1 of the platform: a reproducible training run that logs to MLflow
and registers a model version. Everything downstream (KServe serving, drift
detection, automated retraining) depends on this producing the same artefacts
every time, so the design decisions here are deliberate.

Design notes worth being able to defend:

* The dataset is generated with a fixed seed and parameterised by
  `--data-version`. Two runs with the same data version produce identical
  training data, which is what makes a metric comparison between runs
  meaningful. A model whose score moved cannot be blamed on the data having
  silently changed.

* The split is stratified and seeded. Without stratification a small dataset can
  put all of one class in a single split, and the accuracy becomes noise.

* Metrics are logged to MLflow, not printed. The whole point of stage 2 is to
  serve "the registered model", which only works if there is a registry
  entry with a metric attached to choose it by.

* `--drift-inject` deliberately shifts the feature distribution, and the shift
  is applied only to the *test* split. This exists so stage 3 has something real
  to detect. Shifting the whole dataset would be pointless as a test signal,
  because the model would simply learn the shifted relationship and score the
  same - which is exactly what the first version of this script did. Shifting
  only the holdout reproduces the real scenario: the world moved, the model did
  not, so the score drops and the drift monitor has a signal to act on.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def generate_data(
    n_samples: int,
    n_features: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate the clean dataset.

    A fixed seed is the reproducibility guarantee: the same (n_samples,
    n_features, seed) always yields the same data, so a change in the metric
    between runs reflects a change in the model or the code, not the data.
    """
    return make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=max(2, n_features // 2),
        n_redundant=0,
        n_classes=2,
        random_state=seed,
        flip_y=0.01,
    )


def inject_drift(
    features: np.ndarray,
    seed: int,
    magnitude: float = 1.5,
) -> np.ndarray:
    """Shift half the feature columns, modelling a changed input distribution.

    Applied to the holdout only, so the model is still trained on the
    distribution it expects and then scored on the moved one. This is the
    condition stage 3 exists to detect, and the reason the flag is useful: the
    score genuinely drops, so the drift monitor is tested against a real signal
    rather than an assertion that it would notice one.
    """
    rng = np.random.default_rng(seed)
    shifted = features.copy()
    shift = rng.normal(loc=magnitude, scale=0.3, size=features.shape[1] // 2)
    shifted[:, : features.shape[1] // 2] += shift
    return shifted


def build_model(max_iter: int, c_value: float, seed: int) -> Pipeline:
    """Scaling plus logistic regression.

    Scaling is inside the pipeline rather than applied beforehand so that the
    transform is fitted on the training split alone and travels with the model.
    A scaler fitted on the full dataset would leak test-set statistics into
    training, which inflates the reported score.
    """
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=max_iter,
                    C=c_value,
                    random_state=seed,
                ),
            ),
        ]
    )


def reference_profile(features: np.ndarray, n_bins: int = 10) -> dict[str, Any]:
    """Summarise the training distribution for drift detection.

    Stored per feature, and that is not an implementation detail. A joint
    histogram over all features would need `n_bins ** n_features` cells: with 20
    features and 10 bins that is 1e20 bins, which numpy rejects outright and
    which would be too sparse to estimate from any real sample count even if it
    did not. Drift is therefore measured one feature at a time, and the profile
    is a list with one entry per feature.

    Stores per-feature decile edges plus the resulting bin proportions. The
    edges matter more than the proportions: the monitor must bin serving data
    with the *training* edges, otherwise it re-derives bins from the new data
    and a shift that moves every value by the same amount scores as no drift at
    all. That failure mode is invisible unless the edges are carried across.

    Quantile edges are used rather than equal-width ones because the synthetic
    features are roughly normal, and equal-width bins put almost every sample in
    the middle bin, where a shift barely moves the proportions. Deciles give
    each bin a meaningful share of the reference mass.
    """
    quantiles = np.linspace(0, 100, n_bins + 1)
    per_feature = []
    for j in range(features.shape[1]):
        column = features[:, j]
        edges = np.percentile(column, quantiles)

        # Guard against duplicated edges on constant or near-constant features,
        # which would make the bin index ambiguous. Widening by a small epsilon
        # keeps every feature binnable.
        for i in range(1, edges.shape[0]):
            if edges[i] <= edges[i - 1]:
                edges[i] = edges[i - 1] + 1e-9

        counts, _ = np.histogram(column, bins=edges)
        total = counts.sum()
        proportions = (counts / total).tolist() if total else []

        per_feature.append(
            {
                "feature": j,
                "edges": edges.tolist(),
                "proportions": proportions,
                "mean": float(column.mean()),
                "std": float(column.std()),
            }
        )

    return {
        "n_features": int(features.shape[1]),
        "n_samples": int(features.shape[0]),
        "n_bins": int(n_bins),
        "features": per_feature,
    }


def evaluate(model: Pipeline, features: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """Compute the metrics the platform reports.

    Accuracy alone is misleading on an unbalanced set, so precision, recall,
    F1, and ROC AUC are recorded as well. Drift-related retraining decisions in
    stage 4 should look at a robust metric, not accuracy, which stays high while
    a minority class is quietly failing.
    """
    predictions = model.predict(features)
    probabilities = model.predict_proba(features)[:, 1]
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and register a model")
    parser.add_argument("--tracking-uri", default=os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
    parser.add_argument("--experiment", default="mlops-platform")
    parser.add_argument("--run-name", default="training")
    parser.add_argument("--n-samples", type=int, default=5000)
    parser.add_argument("--n-features", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--c-value", type=float, default=1.0)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument(
        "--data-version",
        default="v1",
        help="Logical dataset version recorded as a tag, so a metric can be tied to the data it came from",
    )
    parser.add_argument(
        "--drift-inject",
        action="store_true",
        help="Shift the feature distribution, to give stage 3 drift detection a real signal to find",
    )
    parser.add_argument(
        "--register",
        action="store_true",
        help="Register the model in the MLflow registry under --model-name",
    )
    parser.add_argument("--model-name", default="mlops-platform-model")
    parser.add_argument(
        "--metrics-out",
        default="",
        help="Optional path to also write metrics as JSON, for CI to assert on",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment)

    features, labels = generate_data(
        n_samples=args.n_samples,
        n_features=args.n_features,
        seed=args.seed,
    )
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=labels,
    )

    # The shift is applied after the split, so training is untouched and only
    # the holdout moves. See inject_drift for why this ordering is the point.
    x_test_eval = inject_drift(x_test, seed=args.seed) if args.drift_inject else x_test

    # MLflow's client writes its "View run at ..." banner directly to stdout when
    # a run ends. Anything consuming this script's output by piping it (the Argo
    # gate step, a CI assertion) would receive that banner interleaved with the
    # JSON and fail to parse. stdout is diverted to stderr for the duration of
    # the run so stdout stays a single valid JSON document.
    banner_sink = sys.stdout
    sys.stdout = sys.stderr
    try:
        with mlflow.start_run(run_name=args.run_name) as run:
            # Parameters are logged before training so a run that crashes still
            # records what it was attempting.
            mlflow.log_params(
                {
                    "n_samples": args.n_samples,
                    "n_features": args.n_features,
                    "seed": args.seed,
                    "max_iter": args.max_iter,
                    "c_value": args.c_value,
                    "test_size": args.test_size,
                    "drift_inject": args.drift_inject,
                }
            )
            mlflow.set_tags(
                {
                    "data_version": args.data_version,
                    "stage": "training",
                }
            )

            model = build_model(args.max_iter, args.c_value, args.seed)
            model.fit(x_train, y_train)

            # Stage 3's monitor needs the training distribution to compare
            # serving inputs against. Exporting it here, from the same x_train
            # the model was fitted on, is what keeps the reference honest: a
            # profile computed from the full dataset or re-generated later with
            # different parameters would describe a distribution the model never
            # saw, and every drift score would be wrong.
            reference = reference_profile(x_train)
            mlflow.log_dict(reference, "reference_profile.json")

            train_metrics = evaluate(model, x_train, y_train)
            # Scored against the (possibly drifted) holdout. This is the number
            # the quality gate reads, so a drifted run genuinely fails the gate
            # instead of silently passing on data that moved with the model.
            test_metrics = evaluate(model, x_test_eval, y_test)
            clean_metrics = evaluate(model, x_test, y_test)

            for name, value in train_metrics.items():
                mlflow.log_metric(f"train_{name}", value)
            for name, value in test_metrics.items():
                mlflow.log_metric(f"test_{name}", value)
            for name, value in clean_metrics.items():
                mlflow.log_metric(f"clean_{name}", value)

            # How far the score moved when the input distribution changed. Stage
            # 3's monitor looks for this, and it is zero on an unchanged dataset.
            drift_drop = clean_metrics["accuracy"] - test_metrics["accuracy"]
            mlflow.log_metric("drift_accuracy_drop", drift_drop)

            # The gap between train and clean test is the overfitting signal;
            # using the drifted holdout here would conflate drift with
            # overfitting.
            mlflow.log_metric(
                "train_test_accuracy_gap",
                train_metrics["accuracy"] - clean_metrics["accuracy"],
            )

            mlflow.sklearn.log_model(
                sk_model=model,
                name="model",
                # Without a signature, KServe cannot validate input shape and the
                # serving stage fails at request time rather than predict time.
                signature=mlflow.models.infer_signature(x_train, model.predict(x_train)),
                input_example=x_train[:5],
                registered_model_name=args.model_name if args.register else None,
            )

            result: dict[str, Any] = {
                "run_id": run.info.run_id,
                "experiment": args.experiment,
                "data_version": args.data_version,
                "drift_inject": args.drift_inject,
                # test_metrics is the holdout as scored (drifted when requested)
                # and is what the gate acts on. clean_metrics is the same holdout
                # unshifted, kept so a drop can be attributed to drift and not to
                # a worse model.
                "test_metrics": test_metrics,
                "clean_metrics": clean_metrics,
                "drift_accuracy_drop": drift_drop,
                "train_metrics": train_metrics,
            }

            if args.register:
                # `log_model(registered_model_name=...)` creates the entry; the
                # version is read back so downstream stages pin an exact version
                # rather than "latest", which is not reproducible.
                client = mlflow.MlflowClient()
                versions = client.search_model_versions(f"name='{args.model_name}'")
                latest = max(int(v.version) for v in versions)
                result["model_name"] = args.model_name
                result["model_version"] = latest
                mlflow.set_tag("registered_version", str(latest))
    finally:
        sys.stdout = banner_sink

    print(json.dumps(result, indent=2))

    if args.metrics_out:
        with open(args.metrics_out, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
