#!/usr/bin/env python
"""Copy a registered model version into the bucket KServe reads from.

KServe's storage initializer is an init container that materialises a
directory; it has no MLflow client and cannot resolve `models:/name/version`.
This script is the bridge. It runs after a version passes the quality gate and
before the InferenceService is rolled, so the serving path never talks to the
registry at request time.

The copy is deliberately explicit rather than an `InferenceService` storageUri
of `models:/...`:
  * The storage initializer cannot authenticate to MLflow, only to object
    storage.
  * Promotion becomes an auditable step. "Which registry version is serving?"
    is answered by a bucket prefix and an annotation, not by the live state of
    the registry, which changes underneath a running endpoint.

Usage:
    python serving/promote.py --tracking-uri http://127.0.0.1:5001
    python serving/promote.py --model-version 3 --dry-run
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path

import mlflow

DEFAULT_MODEL_NAME = "mlops-platform-model"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Promote a registered version to serving storage")
    parser.add_argument(
        "--tracking-uri",
        default=os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000"),
    )
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument(
        "--model-version",
        default=None,
        help="Version to promote. Defaults to the highest registered version.",
    )
    parser.add_argument(
        "--destination",
        default=os.environ.get("SERVING_MODEL_DIR", ""),
        help=(
            "Where to place the model tree. In-cluster this is an S3/MinIO prefix "
            "mounted into the initializer; locally it is a directory, which keeps "
            "the promotion logic testable without object storage."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve and download the version but do not write to the destination",
    )
    return parser.parse_args()


def required_files(model_dir: Path) -> list[str]:
    """Files an MLflow model directory must contain to be loadable.

    Checked before promotion so a half-written artifact cannot be published and
    then fail inside a serving pod, where the failure is far more expensive to
    diagnose.
    """
    return ["MLmodel"]


def resolve_version(client: mlflow.MlflowClient, model_name: str, requested: str | None) -> str | None:
    if requested is not None:
        return requested
    versions = client.search_model_versions(f"name='{model_name}'")
    if not versions:
        return None
    return max(versions, key=lambda v: int(v.version)).version


def main() -> int:
    args = parse_args()

    mlflow.set_tracking_uri(args.tracking_uri)
    client = mlflow.MlflowClient()

    version = resolve_version(client, args.model_name, args.model_version)
    if version is None:
        print(f"no versions registered under {args.model_name!r}; nothing to promote")
        return 1

    uri = f"models:/{args.model_name}/{version}"
    print(f"promoting {uri}")

    with tempfile.TemporaryDirectory() as tmp:
        staged = Path(tmp) / "model"
        mlflow.artifacts.download_artifacts(
            artifact_uri=uri,
            dst_path=str(staged),
        )

        # `download_artifacts` nests under the run's artifact path, so locate
        # the directory that actually holds MLmodel rather than assuming depth.
        candidates = [p for p in [staged, *staged.rglob("*")] if p.is_dir() and (p / "MLmodel").exists()]
        if not candidates:
            print(f"no MLmodel file found under the downloaded artifact: {list(staged.rglob('*'))[:20]}")
            return 1
        model_dir = candidates[0]

        missing = [f for f in required_files(model_dir) if not (model_dir / f).exists()]
        if missing:
            print(f"model directory is missing required files: {missing}")
            return 1

        files = sorted(p.relative_to(model_dir).as_posix() for p in model_dir.rglob("*") if p.is_file())
        print(f"model directory contains {len(files)} files:")
        for name in files:
            print(f"  {name}")

        if args.dry_run or not args.destination:
            reason = "--dry-run" if args.dry_run else "no --destination given"
            print(f"not writing ({reason}); staging verified only")
            return 0

        destination = Path(args.destination) / f"{args.model_name}"
        if destination.exists():
            shutil.rmtree(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(model_dir, destination)

        written = sorted(p.relative_to(destination).as_posix() for p in destination.rglob("*") if p.is_file())
        if written != files:
            print(f"destination does not match source: {written} != {files}")
            return 1

    print(f"promoted version {version} to {destination}")
    # Emitted for the workflow to consume, so the annotation on the
    # InferenceService and the bucket contents cannot disagree.
    print(f"MODEL_VERSION={version}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
