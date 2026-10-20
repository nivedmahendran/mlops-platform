# mlops-platform

![stage-1-training](https://github.com/nivedmahendran/mlops-platform/actions/workflows/stage-1-training.yml/badge.svg)
![stage-2-serving](https://github.com/nivedmahendran/mlops-platform/actions/workflows/stage-2-serving.yml/badge.svg)
![stage-3-drift-monitoring](https://github.com/nivedmahendran/mlops-platform/actions/workflows/stage-3-drift-monitoring.yml/badge.svg)
![stage-4-automated-retraining](https://github.com/nivedmahendran/mlops-platform/actions/workflows/stage-4-automated-retraining.yml/badge.svg)

An end-to-end MLOps platform on Kubernetes: **MLflow** for tracking and the model
registry, **Argo Workflows** for orchestration, **KServe** for serving, drift
monitoring, and automated retraining when the input distribution moves.

Built in four stages. Each stage is runnable on its own, and each one is
verified by CI rather than asserted in prose.

| Stage | What it adds | Status |
| ----- | ------------ | ------ |
| 1. Training pipeline | MLflow tracking + registry, Argo training workflow, quality gate | ✅ done |
| 2. Serving | KServe InferenceService pulling the registered model | ✅ done |
| 3. Drift monitoring | Compare serving-time inputs against the training distribution | ✅ done |
| 4. Automated retraining | Argo Events sensor triggers a new run when drift crosses the threshold | ✅ done |

## Tech

| Layer | Choice | Why |
| ----- | ------ | --- |
| Tracking + registry | MLflow 3.17 | Registry is the contract stage 2 serves from |
| Orchestration | Argo Workflows | Real DAG with dependencies; failed runs keep logs and artefacts |
| Serving | KServe | Standard inference protocol; scale-to-zero |
| Drift detection | PSI per feature | Compares bin proportions, so shape shifts register, not just mean shifts |
| Eventing | Argo Events | Sensor submits the training workflow; no resident service needed |
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

## Stage 2: serving

```
serving/inferenceservice.yml   KServe InferenceService (mlflow runtime, v2 protocol)
serving/promote.py             registry version -> bucket prefix the initializer reads
serving/verify_serving.py      exercises the inference contract without a cluster
```

The registry is not on the serving path. `promote.py` copies the version's
artifact tree into the bucket prefix the InferenceService reads, so the endpoint
never needs an MLflow client at request time and "which version is serving?" is
answered by the bucket contents rather than by the registry's live state, which
changes underneath a running endpoint.

### Design decisions worth knowing

**Why `modelFormat: mlflow` and `protocolVersion: v2`.** The plain `sklearn`
runtime expects a bare pickled estimator and ignores the recorded signature. The
MLflow runtime reads the model directory MLflow wrote, which is what carries the
signature KServe validates input against.

**Why CI checks the payload shape rather than the cluster.** A cluster is not
available in CI, and a manifest that applies cleanly says nothing about whether
the runtime can answer a request. `verify_serving.py` loads the model the way
the runtime does and drives it with a real v2 payload, so the two failure modes
that matter — an unloadable artifact, and a response the protocol cannot carry —
are caught before deployment.

## Stage 3: drift monitoring

```
monitoring/collect_inputs.py         KServe request logs -> feature array
monitoring/detect_drift.py           per-feature PSI against the training reference
monitoring/drift-cronworkflow.yml    runs it on a schedule
```

`training/train.py` exports `reference_profile.json` alongside the model. Drift
is only meaningful relative to *that* profile: the monitor bins serving data
with the training-time edges, so a shift that moves every value by the same
amount registers as drift instead of being normalised away.

### Design decisions worth knowing

**Why the profile is per feature.** A joint histogram over 20 features and 10
bins needs `10**20` cells, which numpy rejects outright and which no real sample
count could populate. Drift is measured one feature at a time; the profile is a
list, not a grid. (The first version built the joint histogram and crashed.)

**Why PSI and not a mean comparison.** A distribution can shift shape without
its mean moving — variance changes, or mass moves between the tails. PSI
compares bin proportions, so it sees those. The bands are the conventional ones:
below 0.10 stable, 0.10–0.25 moderate, at or above 0.25 significant.

**Why values outside the reference range are dropped, not clipped.** A value
beyond every value seen in training is the clearest drift there is. Folding it
into the nearest bin would hide exactly that signal.

## Stage 4: automated retraining

```
retraining/event-source.yml    EventSource + Sensor that submits the stage-1 template
```

When the monitor finds drift it POSTs the verdict to the sensor, which submits
`training-pipeline` — the *same* WorkflowTemplate a human would run. Drift
justifies a retrain; it does not lower the bar. The retrained model still has to
clear `min-accuracy` in the gate.

### Design decisions worth knowing

**Why one template and not two.** A separate retraining pipeline would drift
from the manual one and eventually publish a model the serving path cannot load.
One template with two callers is what keeps the paths equivalent.

**Why the trigger is the drift signal and not a schedule.** A clock-driven
retrain trains a healthy model on schedule and, worse, misses a model that
drifts just after the schedule fires. The sensor is silent unless there is
something to act on.

**Why the monitor exits non-zero when it cannot notify.** A drift signal that
never reaches the retraining path is an outage of this stage. Reporting the
measurement as fine while the action silently failed is the failure this
guards against.

### Verifying stages 2–4

Each stage has its own workflow, and each asserts against a real MLflow server
rather than mocked state:

- **stage-2-serving** promotes the version, loads it in a clean process, and
  validates the v2 response contract.
- **stage-3-drift-monitoring** asserts *both* directions: a clean batch must not
  be flagged, and a drifted batch must be — with the exact set of shifted
  features identified, so the monitor is shown to locate drift rather than just
  notice that something moved.
- **stage-4-automated-retraining** drives the notification against a real HTTP
  listener, then reconciles the sensor's submitted arguments against the
  WorkflowTemplate's declared parameters, so a sensor that submits a workflow it
  cannot satisfy fails in CI rather than in the cluster.

Manifests are validated with `kubeconform`, including the KServe and Argo Events
custom resources. Those schemas are extracted from the upstream CRDs
(`lint.yml`), because otherwise kubeconform reports every custom resource as
"Skipped" and validates nothing.

## Licence

MIT — see [LICENSE](LICENSE).
