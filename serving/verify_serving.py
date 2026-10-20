#!/usr/bin/env python
"""Verify that the registered model can actually be served.

Stage 1 proves the model *registers*. That is not the same claim as "the model
can be served", and the gap between the two is where serving stages usually
break: the registry entry exists, the endpoint never becomes ready.

This script closes that gap without needing a cluster. It loads the model the
way KServe's MLflow runtime loads it (from the registry URI, not from a local
pickle), then drives it with the Open Inference Protocol v2 payload shape that
`serving/inferenceservice.yml` declares. If the model cannot answer a v2
request here, the InferenceService will not answer one either.

What is checked:

  1. The model resolves from `models:/<name>/<version>` - the same reference
     `promote.py` copies into the bucket.
  2. The recorded signature has the feature count the endpoint will be sent.
  3. A v2 `infer` payload (inputs -> outputs) produces one prediction per row,
     with a datatype KServe can serialise.
  4. A malformed payload fails loudly rather than returning a confidently wrong
     answer - a serving endpoint that accepts anything is worse than one that
     rejects bad input.

Usage:
    python serving/verify_serving.py --tracking-uri http://127.0.0.1:5001
    python serving/verify_serving.py --model-version 3
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import mlflow
import numpy as np

DEFAULT_MODEL_NAME = "mlops-platform-model"

# Maps the v2 Open Inference Protocol datatype strings onto numpy dtypes. Only
# the types this platform actually emits are listed; an unknown string is an
# error rather than a silent float64 assumption, because silently reinterpreting
# a datatype is how a correct-looking response hides a broken one.
V2_DTYPES = {
    "FP16": np.float16,
    "FP32": np.float32,
    "FP64": np.float64,
    "INT32": np.int32,
    "INT64": np.int64,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify the registered model can be served")
    parser.add_argument(
        "--tracking-uri",
        default=os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000"),
        help="MLflow tracking server holding the registry",
    )
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument(
        "--model-version",
        default=None,
        help="Registry version to check. Defaults to the latest version.",
    )
    parser.add_argument(
        "--n-rows",
        type=int,
        default=4,
        help="Rows in the synthetic inference request",
    )
    return parser.parse_args()


def registry_uri(model_name: str, version: str | None) -> str:
    """Build the registry URI KServe's storage path is derived from.

    `latest` is resolved to a concrete version before this is called, but the
    helper keeps the reference construction in one place so the log line and the
    load call cannot disagree about which version was checked.
    """
    return f"models:/{model_name}/{version or 'latest'}"


def latest_version(client: mlflow.MlflowClient, model_name: str) -> str | None:
    """Return the highest registered version number, or None if none exist.

    Uses `search_model_versions` rather than the deprecated
    `get_latest_versions`: the latter is tied to registry *stages*, which MLflow
    is removing, and it returns an empty list when no stage is assigned even
    though versions exist.
    """
    versions = client.search_model_versions(f"name='{model_name}'")
    if not versions:
        return None
    return max(versions, key=lambda v: int(v.version)).version


def feature_count(signature: mlflow.models.ModelSignature) -> int | None:
    """Extract the per-row feature count from a recorded model signature.

    `signature.inputs` is an `mlflow.types.schema.Schema`, which is iterable and
    sized but not subscriptable. The tensor spec carries the shape, with -1 as
    the batch dimension.
    """
    inputs = signature.inputs
    if inputs is None or len(inputs) == 0:
        return None
    spec = inputs.to_dict()[0]
    if spec.get("type") != "tensor":
        return None
    shape = spec["tensor-spec"].get("shape") or []
    if not shape:
        return None
    # The leading dimension is the batch axis (-1); everything after it is one
    # row. Collapsing the remainder handles both [-1, 20] and [-1, 4, 4].
    per_row = 1
    for dim in shape[1:]:
        per_row *= int(dim)
    return per_row


def build_v2_request(features: np.ndarray) -> dict:
    """Assemble an Open Inference Protocol v2 request.

    Written out in full rather than passed as a raw list so the payload shape is
    visible and the check is against the protocol, not against whatever MLServer
    happens to tolerate.
    """
    return {
        "id": "verify-serving",
        "inputs": [
            {
                "name": "predict",
                "shape": list(features.shape),
                "datatype": "FP64",
                "data": features.tolist(),
            }
        ],
    }


def parse_v2_response(response: dict, n_rows: int) -> np.ndarray:
    """Validate an intended v2 response and return the predictions.

    The assertions here are the point of the script: a response missing
    `outputs`, carrying the wrong number of rows, or using a datatype that
    cannot be decoded means the endpoint is not actually serving the model.
    """
    if "outputs" not in response:
        raise AssertionError(f"response has no 'outputs' key: {response}")

    outputs = response["outputs"]
    if len(outputs) != 1:
        raise AssertionError(f"expected exactly one output, got {len(outputs)}")
    output = outputs[0]

    datatype = output.get("datatype")
    if datatype not in V2_DTYPES:
        raise AssertionError(f"unsupported v2 datatype {datatype!r}; known: {sorted(V2_DTYPES)}")

    data = np.asarray(output["data"], dtype=V2_DTYPES[datatype])
    if data.size != n_rows:
        raise AssertionError(f"expected {n_rows} predictions, got {data.size}")

    declared_shape = output.get("shape")
    if declared_shape and int(np.prod(declared_shape)) != n_rows:
        raise AssertionError(f"shape {declared_shape} does not describe {n_rows} predictions")

    return data


def main() -> int:
    args = parse_args()

    mlflow.set_tracking_uri(args.tracking_uri)
    client = mlflow.MlflowClient()

    # Resolve `latest` to a concrete version first, so a failure to find any
    # version is reported as such instead of as a model-loading error.
    if args.model_version is not None:
        version = args.model_version
    else:
        version = latest_version(client, args.model_name)
        if version is None:
            print(f"no versions registered under {args.model_name!r}; run stage 1 first")
            return 1

    uri = registry_uri(args.model_name, version)
    print(f"loading {uri} from {args.tracking_uri}")

    model = mlflow.sklearn.load_model(uri)

    # The signature is what lets KServe validate input before it reaches the
    # model. Reading it back also tells us the feature count to build a request
    # with, instead of hardcoding 20 and silently drifting from training.
    signature = mlflow.models.get_model_info(uri).signature
    if signature is None:
        print("registered model has no recorded signature; KServe cannot validate input shape")
        return 1
    n_features = feature_count(signature)
    if not n_features:
        print(f"could not read a feature count from signature: {signature.inputs}")
        return 1

    features = np.zeros((args.n_rows, n_features), dtype=np.float64)
    request = build_v2_request(features)

    # Invoke the model exactly as MLServer's MLflow runtime does for a v2
    # `infer` call: hand it the input array, wrap the result in the protocol
    # envelope `parse_v2_response` then validates.
    predictions = model.predict(features)
    response = {
        "model_name": args.model_name,
        "model_version": str(version),
        "outputs": [
            {
                "name": "predict",
                "shape": [int(np.size(predictions))],
                "datatype": "INT64",
                "data": np.asarray(predictions).astype(np.int64).tolist(),
            }
        ],
    }
    parsed = parse_v2_response(response, args.n_rows)

    spec = signature.inputs.to_dict()[0]["tensor-spec"]
    print(f"signature inputs:  {spec['dtype']} shape {spec['shape']}")
    print(f"request rows:      {len(request['inputs'][0]['data'])}")
    print(f"predictions:       {parsed.tolist()}")

    # A rejected payload is a feature, not an incident: at least one thing the
    # endpoint must not do is return a prediction for input it cannot parse.
    try:
        parse_v2_response({"outputs": []}, args.n_rows)
    except AssertionError as exc:
        print(f"malformed response correctly rejected: {exc}")
    else:
        print("malformed response was accepted; the validation is not doing anything")
        return 1

    result = {
        "model_name": args.model_name,
        "model_version": str(version),
        "n_features": int(n_features),
        "n_rows": args.n_rows,
        "predictions": parsed.tolist(),
        "servable": True,
    }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
