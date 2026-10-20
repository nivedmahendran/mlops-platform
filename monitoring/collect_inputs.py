#!/usr/bin/env python
"""Turn recent KServe request logs into the feature array the monitor scores.

KServe's request logger writes one JSON object per request/response pair to
object storage. That log is the only record of what the endpoint was actually
asked, so it is what drift has to be measured against - scoring the training
data against itself would always report "stable".

Two things this script is careful about:

  * It reads the *inputs*, not the predictions. Drift is a property of the
    request distribution. Predictions move for reasons that have nothing to do
    with drift (a threshold change, a new model version), so they are the wrong
    signal.

  * It fails loudly on an empty window instead of returning zeros. A monitor
    that reports "stable" because it read nothing is worse than one that
    reports nothing at all, and an all-zero array would do exactly that while
    looking like a real result.

Usage:
    python monitoring/collect_inputs.py \
        --source s3://mlflow/kserve-logs \
        --window-minutes 15 \
        --out /tmp/inputs/current.npy

    # Local development without object storage:
    python monitoring/collect_inputs.py \
        --source ./local-logs --window-minutes 0 \
        --out /tmp/inputs/current.npy
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

DEFAULT_ENDPOINT = os.environ.get("MLFLOW_S3_ENDPOINT_URL", "http://minio.mlops.svc.cluster.local:9000")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect serving inputs for drift scoring")
    parser.add_argument(
        "--source",
        required=True,
        help="s3://bucket/prefix or a local directory",
    )
    parser.add_argument(
        "--window-minutes",
        type=int,
        default=15,
        help="How far back to look. 0 means 'everything available' (for local runs).",
    )
    parser.add_argument("--out", required=True, help="Where to write the .npy feature array")
    parser.add_argument("--count-out", default=None, help="Where to write the row count")
    parser.add_argument(
        "--max-rows",
        type=int,
        default=20000,
        help="Cap on rows scored in one pass, so a traffic spike cannot make the job unbounded",
    )
    return parser.parse_args()


def is_s3(source: str) -> bool:
    return source.startswith("s3://")


def split_s3(source: str) -> tuple[str, str]:
    without_scheme = source[len("s3://") :]
    bucket, _, prefix = without_scheme.partition("/")
    return bucket, prefix


def s3_client(endpoint: str):
    import boto3

    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
        region_name=os.environ.get("AWS_REGION", "us-east-1"),
    )


def list_objects(client, bucket: str, prefix: str, since: datetime | None) -> list[dict]:
    """List log objects, newest last, optionally filtered by LastModified.

    The window filter is applied here rather than after download so a busy
    bucket does not transfer an entire day of logs to score fifteen minutes.
    """
    paginator = client.get_paginator("list_objects_v2")
    found: list[dict] = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if not (key.endswith(".json") or key.endswith(".json.gz")):
                continue
            if since is not None and obj["LastModified"] < since:
                continue
            found.append(obj)
    found.sort(key=lambda o: o["LastModified"])
    return found


def read_object(client, bucket: str, key: str) -> bytes:
    body = client.get_object(Bucket=bucket, Key=key)["Body"].read()
    if key.endswith(".gz"):
        return gzip.decompress(body)
    return body


def rows_from_log_payload(payload: dict) -> list[list[float]]:
    """Extract feature rows from one logged request.

    KServe's logger wraps the request body under `request` (or `inputs` for a
    v2 payload). Both shapes are handled because the logger's schema depends on
    the protocol version in use, and silently skipping a shape would undercount
    the window without saying so.
    """
    rows: list[list[float]] = []

    request = payload.get("request") or payload.get("inputs") or payload
    if isinstance(request, dict):
        inputs = request.get("inputs") or []
        for entry in inputs:
            data = entry.get("data")
            shape = entry.get("shape") or []
            if data is None:
                continue
            arr = np.asarray(data, dtype=np.float64)
            # A v2 payload is flat with a declared shape; reshape when the shape
            # has more than one dimension so rows stay rows.
            if len(shape) > 1:
                arr = arr.reshape(shape)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            rows.extend(arr.tolist())
    elif isinstance(request, list):
        for item in request:
            arr = np.asarray(item, dtype=np.float64)
            rows.append(arr.tolist())

    return rows


def collect_local(source: str, since: datetime | None, max_rows: int) -> np.ndarray:
    rows: list[list[float]] = []
    root = Path(source)
    files = sorted(p for p in root.rglob("*") if p.suffix in {".json", ".gz"})
    for path in files:
        if since is not None:
            mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
            if mtime < since:
                continue
        raw = path.read_bytes()
        if path.suffix == ".gz":
            raw = gzip.decompress(raw)
        rows.extend(rows_from_log_payload(json.loads(raw)))
        if len(rows) >= max_rows:
            break
    return np.asarray(rows[:max_rows], dtype=np.float64)


def collect_s3(source: str, since: datetime | None, max_rows: int, endpoint: str) -> np.ndarray:
    bucket, prefix = split_s3(source)
    client = s3_client(endpoint)
    rows: list[list[float]] = []
    objects = list_objects(client, bucket, prefix, since)
    for obj in objects:
        rows.extend(rows_from_log_payload(json.loads(read_object(client, bucket, obj["Key"]))))
        if len(rows) >= max_rows:
            break
    return np.asarray(rows[:max_rows], dtype=np.float64)


def main() -> int:
    args = parse_args()

    since = None
    if args.window_minutes > 0:
        since = datetime.now(timezone.utc) - timedelta(minutes=args.window_minutes)

    started = time.time()
    if is_s3(args.source):
        features = collect_s3(args.source, since, args.max_rows, DEFAULT_ENDPOINT)
    else:
        features = collect_local(args.source, since, args.max_rows)

    elapsed = time.time() - started

    if features.size == 0:
        # An empty window is reported as a failure, not as zero drift. Returning
        # an empty array would make the monitor score nothing and report
        # "stable", which is indistinguishable from a healthy endpoint.
        print(
            f"no serving inputs found in the last {args.window_minutes} minutes "
            f"under {args.source}; cannot assess drift",
            file=sys.stderr,
        )
        return 1

    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.save(destination, features)

    count_path = Path(args.count_out) if args.count_out else destination.with_suffix(".count")
    count_path.write_text(str(features.shape[0]), encoding="utf-8")

    print(
        json.dumps(
            {
                "source": args.source,
                "window_minutes": args.window_minutes,
                "rows": int(features.shape[0]),
                "features": int(features.shape[1]),
                "seconds": round(elapsed, 3),
                "out": str(destination),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
