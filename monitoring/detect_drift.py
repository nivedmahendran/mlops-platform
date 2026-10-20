#!/usr/bin/env python
"""Detect drift between serving-time inputs and the training distribution.

Population Stability Index (PSI) per feature, using the bin edges recorded at
training time. The edges are the load-bearing part: binning new data with
re-derived edges would make a uniform shift score as zero drift, because both
distributions would be split the same way. Stage 1 exports the edges precisely
so this script does not have to guess them.

PSI is read as:

    < 0.10   no meaningful shift
    0.10-0.25  moderate shift, worth watching
    >= 0.25  significant shift, retraining is justified

Those bands are the conventional ones and are exposed as thresholds rather than
hardcoded decisions, so stage 4 can choose the level at which it acts without
this script changing.

The output is both a JSON document (for Argo to read) and a Prometheus text
exposition (for the metric to be scraped). Emitting the metric here rather than
in an HTTP handler keeps the monitor a batch job: it runs on a schedule, exits,
and leaves a time series behind. Nothing has to stay resident to observe drift.

Usage:
    python monitoring/detect_drift.py \
        --reference reference_profile.json \
        --current serving_inputs.npy \
        --metrics-out /tmp/drift.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

# PSI bands. Named rather than inlined so the numbers appear once.
PSI_STABLE = 0.10
PSI_MODERATE = 0.25


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare current inputs against the training reference")
    parser.add_argument(
        "--reference",
        required=True,
        help="reference_profile.json exported by training/train.py",
    )
    parser.add_argument(
        "--current",
        required=True,
        help=".npy file of serving-time feature rows, shape (n, n_features)",
    )
    parser.add_argument("--metrics-out", default=None, help="write the JSON result here")
    parser.add_argument(
        "--prometheus-out",
        default=None,
        help="write Prometheus text exposition here (otherwise stdout)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=PSI_MODERATE,
        help="PSI above which `drift_detected` is true",
    )
    parser.add_argument(
        "--notify-url",
        default=os.environ.get("DRIFT_NOTIFY_URL", ""),
        help=(
            "POST the result here when drift is detected. Stage 4's sensor "
            "listens at this endpoint. Optional: stages 1-3 run without it, so "
            "the monitor has no hard dependency on the retraining path."
        ),
    )
    return parser.parse_args()


def load_reference(path: str) -> dict:
    with open(path, encoding="utf-8") as handle:
        reference = json.load(handle)
    for key in ("features", "n_features"):
        if key not in reference:
            raise ValueError(f"reference profile is missing {key!r}: {sorted(reference)}")
    if len(reference["features"]) != reference["n_features"]:
        raise ValueError(
            "reference profile is inconsistent: "
            f"n_features={reference['n_features']} but {len(reference['features'])} feature entries"
        )
    return reference


def bin_proportions(current: np.ndarray, edges: np.ndarray) -> tuple[np.ndarray, int]:
    """Bin one feature using the training edges, returning per-bin proportions.

    `np.histogram` with explicit edges drops values outside the reference range
    rather than extending the bins. That is intentional: a value beyond every
    observed training value is drift, and folding it into the nearest bin would
    hide exactly that. The outside-range count is returned so it can be reported
    and acted on.
    """
    counts, _ = np.histogram(current, bins=edges)
    total = counts.sum()
    out_of_range = int(current.shape[0] - total)
    if total == 0:
        return np.zeros_like(counts, dtype=np.float64), out_of_range
    return counts / total, out_of_range


def population_stability_index(
    expected: np.ndarray,
    actual: np.ndarray,
    epsilon: float = 1e-6,
) -> float:
    """PSI between two proportion arrays.

    The epsilon floor is what keeps the logarithm finite when a bin is empty on
    one side. An empty bin after perfect separation is maximal drift, not an
    error, so it is pushed to a large finite contribution instead of producing
    `inf` and poisoning the aggregate.
    """
    expected = np.clip(expected, epsilon, None)
    actual = np.clip(actual, epsilon, None)
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def classify(psi: float, threshold: float) -> str:
    if psi >= max(threshold, PSI_MODERATE):
        return "significant"
    if psi >= PSI_STABLE:
        return "moderate"
    return "stable"


def main() -> int:
    args = parse_args()

    reference = load_reference(args.reference)
    n_features = int(reference["n_features"])
    feature_profiles = reference["features"]

    if not os.path.exists(args.current):
        print(f"current inputs not found: {args.current}")
        return 1
    current = np.load(args.current)
    if current.ndim == 1:
        current = current.reshape(1, -1)
    if current.shape[1] != n_features:
        print(f"current inputs have {current.shape[1]} features, reference has {n_features}")
        return 1

    per_feature = []
    out_of_range_total = 0
    for j in range(n_features):
        profile = feature_profiles[j]
        edges = np.asarray(profile["edges"], dtype=np.float64)
        expected = np.asarray(profile["proportions"], dtype=np.float64)

        actual, out_of_range = bin_proportions(current[:, j], edges)
        out_of_range_total += out_of_range

        psi = population_stability_index(expected, actual)
        per_feature.append(
            {"feature": j, "psi": round(psi, 6), "status": classify(psi, args.threshold)}
        )

    max_psi = max(f["psi"] for f in per_feature)
    mean_psi = float(np.mean([f["psi"] for f in per_feature]))
    drifted_features = [f["feature"] for f in per_feature if f["psi"] >= args.threshold]
    drift_detected = bool(drifted_features)

    result = {
        "n_features": n_features,
        "n_samples": int(current.shape[0]),
        "out_of_range_rows": out_of_range_total,
        "max_psi": round(max_psi, 6),
        "mean_psi": round(mean_psi, 6),
        "threshold": args.threshold,
        "drift_detected": drift_detected,
        "drifted_features": drifted_features,
        "per_feature": per_feature,
    }

    # Prometheus text exposition. Metric names carry the platform prefix so they
    # cannot collide with the KServe and kube-prometheus-stack series already in
    # the cluster.
    lines = [
        "# HELP mlops_drift_psi_max Highest per-feature PSI in this batch",
        "# TYPE mlops_drift_psi_max gauge",
        f"mlops_drift_psi_max {max_psi:.6f}",
        "# HELP mlops_drift_psi_mean Mean per-feature PSI in this batch",
        "# TYPE mlops_drift_psi_mean gauge",
        f"mlops_drift_psi_mean {mean_psi:.6f}",
        "# HELP mlops_drift_detected 1 when any feature exceeds the threshold",
        "# TYPE mlops_drift_detected gauge",
        f"mlops_drift_detected {1 if drift_detected else 0}",
        "# HELP mlops_drift_features_total Number of features above the threshold",
        "# TYPE mlops_drift_features_total gauge",
        f"mlops_drift_features_total {len(drifted_features)}",
        "# HELP mlops_drift_out_of_range_rows Rows falling outside the reference range",
        "# TYPE mlops_drift_out_of_range_rows gauge",
        f"mlops_drift_out_of_range_rows {out_of_range_total}",
    ]
    exposition = "\n".join(lines) + "\n"

    if args.prometheus_out:
        with open(args.prometheus_out, "w", encoding="utf-8") as handle:
            handle.write(exposition)

    print(json.dumps(result, indent=2))
    if not args.prometheus_out:
        print(exposition)

    if args.metrics_out:
        with open(args.metrics_out, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)

    if args.notify_url and drift_detected:
        # The notification is best-effort in the sense that a failure here does
        # not invalidate the measurement, but it is reported as a non-zero exit
        # because a drift signal that never reaches the retraining path is an
        # outage of stage 4, not a detail.
        if not notify(args.notify_url, result):
            return 1

    return 0


def notify(url: str, result: dict) -> bool:
    """POST the drift verdict to the retraining sensor.

    Sent only when drift is detected, so the sensor is not woken on every run
    and the event stream stays a record of things that actually happened.
    """
    import urllib.error
    import urllib.request

    payload = json.dumps(
        {
            "drift_detected": result["drift_detected"],
            "max_psi": result["max_psi"],
            "mean_psi": result["mean_psi"],
            "drifted_features": result["drifted_features"],
            "n_features": result["n_features"],
            "n_samples": result["n_samples"],
        }
    ).encode("utf-8")

    request = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            print(f"notified {url}: HTTP {response.status}")
            return 200 <= response.status < 300
    except urllib.error.URLError as exc:
        print(f"failed to notify {url}: {exc}", file=sys.stderr)
        return False


if __name__ == "__main__":
    sys.exit(main())
