# Training image.
#
# Multi-stage so the final image carries no build tooling. The training job
# runs with a read-only root filesystem (set in the WorkflowTemplate), which
# constrains this Dockerfile: anything the process writes must go somewhere
# explicitly mounted, and Python must not try to write bytecode caches.

FROM python:3.12-slim AS builder

WORKDIR /build
COPY training/requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

FROM python:3.12-slim

# uid 10001 matches the WorkflowTemplate securityContext and the other images
# in this portfolio. A mismatch shows up as an opaque permission failure.
RUN useradd --uid 10001 --create-home --shell /usr/sbin/nologin appuser

COPY --from=builder /install /usr/local
WORKDIR /app
COPY training/train.py .

# PYTHONDONTWRITEBYTECODE matters because the root filesystem is read-only:
# without it Python tries to write __pycache__ next to the source and fails.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

USER 10001

ENTRYPOINT ["python", "train.py"]
