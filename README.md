# mlops-platform

![stage-1-training](https://github.com/nivedmahendran/mlops-platform/actions/workflows/stage-1-training.yml/badge.svg)

An end-to-end MLOps platform on Kubernetes: **MLflow** for tracking and the model
registry, **Argo Workflows** for orchestration, **KServe** for serving, drift
monitoring, and automated retraining when the input distribution moves.

Built in four stages. Each stage is runnable on its own, and each one is
verified by CI rather than asserted in prose.

| Stage | What it adds | Status |
| ----- | ------------ | ------ |
| 1. Training pipeline | MLflow tracking + registry, Argo training workflow, quality gate | ✅ done |
| 2. Serving | KServe InferenceService pulling the registered model | next |
| 3. Drift monitoring | Compare serving-time inputs against the training distribution | planned |
| 4. Automated retraining | Argo EventSource triggers a new run when drift crosses the threshold | planned |

## Tech

| Layer | Choice | Why |
| ----- | ------ | --- |
| Tracking + registry | MLflow 3.17 | Registry is the contract stage 2 serves from |
| Orchestration | Argo Workflows | Real DAG with dependencies; failed runs keep logs and artefacts |
| Serving | KServe | Standard inference protocol; scale-to-zero |
| Artefact store | MinIO (S3 API) | Local-filesystem artefacts do not survive a pod restart |
| Backend store | Postgres | SQLite in a pod loses the registry on restart |

## Stage 1: the training pipeline

```
pipeline/training-workflow.yml   Argo WorkflowTemplate: train -> quality gate
training/train.py                trains, logs to MLflow, registers a version
training/Dockerfile              non-root (uid 10001), read-only root FS
training/mlflow-deployment.yml   MLflow + Postgres + MinIO
training/requirements.txt        pinned to the versions CI executes
```

### Run it locally

The pipeline needs a tracking server. Start one, then run the job:

```bash
pip install -r training/requirements.txt

python -m mlflow server --host 127.0.0.1 --port 5001 \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./artifacts --serve-artifacts &

python training/train.py --tracking-uri http://127.0.0.1:5001 --register
```

You get metrics on stdout as JSON, a run in the `mlops-platform` experiment, and
a new version in the `mlops-platform-model` registry entry:

```json
{
  "run_id": "...",
  "model_version": 1,
  "test_metrics":   { "accuracy": 0.839, "f1": 0.842, "roc_auc": 0.917 },
  "clean_metrics":  { "accuracy": 0.839, ... },
  "drift_accuracy_drop": 0.0
}
```

### Deploying to Kubernetes

```bash
kubectl apply -f training/mlflow-deployment.yml      # MLflow + Postgres + MinIO
kubectl -n mlops create configmap training-code --from-file=training/train.py
```

The workflow assumes images at `ghcr.io/nivedmahendran/mlops-platform-training`.
Build and push with:

```bash
podman build -t ghcr.io/nivedmahendran/mlops-platform-training:0.1.0 -f training/Dockerfile .
```

### Design decisions worth knowing

**Why the drift flag shifts only the holdout.** The first version shifted the
whole dataset. The model then simply learned the shifted relationship and scored
identically — the flag produced no measurable signal, so stage 3 would have had
nothing to detect. Shifting *only* the test split reproduces the real condition:
the world moved, the model did not. The score drops 11.8 points, which is what
the drift monitor and the retraining trigger act on.

**Why there is a quality gate.** A workflow that runs to completion is not the
same as a model worth serving. `pipeline/training-workflow.yml` evaluates the
holdout accuracy against `min-accuracy` and fails the run below it, so a
regression cannot be promoted just because the pod exited 0.

**Why stdout is JSON and nothing else.** MLflow writes a "View run at …" banner
directly to `stdout` on completion, which corrupts any piped output. The run is
executed with `stdout` temporarily pointed at `stderr`, so stdout stays a single
parseable JSON document for the gate step and for CI.

**Why the version pins are exact.** MLflow 2.x and 3.x differ in the model
logging API and the registry artefact layout. The pins reflect the versions this
pipeline was actually run against; bumping them means re-running stage 2's
serving check as well.

## Verifying stage 1

CI (`.github/workflows/stage-1-training.yml`) starts a real MLflow server and
asserts all four things that make stage 1 meaningful:

1. A clean run registers a model version.
2. A clean run shows **no** drift drop (`== 0.0`).
3. A drifted run shows a drop of **at least 0.05**.
4. The registered model loads and predicts — the exact interface stage 2 uses.

Asserting (2) and (3) together is what stops the drift signal from silently
regressing to zero, which is how the first implementation of `--drift-inject`
failed.

## Licence

MIT — see [LICENSE](LICENSE).
