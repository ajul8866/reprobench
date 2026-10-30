# syntax=docker/dockerfile:1
#
# ReproBench scorer image (paper Section 4.4 / plan item 6).
#
# The scorer runs OUTSIDE the agent container. It mounts a run directory
# (read-only) to build the EvidenceBundle, and writes only to the scores
# directory. It never mounts the reference profiles into the agent container.
#
#   docker build -f docker/scorer.Dockerfile -t reprobench/scorer:24.04 .
#   docker run --rm \
#       -v "$PWD/results/runs:/runs:ro" \
#       -v "$PWD/results/scores:/scores" \
#       -v "$PWD/data:/app/data:ro" \
#       -v "$PWD/config:/app/config:ro" \
#       reprobench/scorer:24.04 score --runs /runs --out /scores --judge heuristic
#
FROM ubuntu:24.04

LABEL org.opencontainers.image.title="ReproBench scorer" \
      org.opencontainers.image.description="Evidence-grounded six-phase rubric scorer for ReproBench" \
      org.opencontainers.image.version="1.0.0" \
      org.opencontainers.image.licenses="MIT"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    LC_ALL=C.UTF-8 \
    LANG=C.UTF-8

# ---------------------------------------------------------------------------
# System packages
#   libmagic1  -> python-magic backend used by scoring/evidence.sniff()
#   binutils   -> `file`-adjacent inspection, strings/objdump for ELF checks
#   file       -> fast MIME/magic detection in the evidence manifest
#   qemu-user-static -> optional re-execution checks during judge confirmation
# ---------------------------------------------------------------------------
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-venv \
        libmagic1 \
        file \
        binutils \
        bash \
        ca-certificates \
        git \
        curl \
        qemu-user-static \
 && rm -rf /var/lib/apt/lists/*

# Python stack (mirrors requirements.txt; no model SDKs -> offline default).
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

COPY requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip \
 && pip install -r /tmp/requirements.txt \
 && rm -f /tmp/requirements.txt

# ---------------------------------------------------------------------------
# Project source.  Config + data stay bind-mounted at run time so a scorer
# image is reusable across rubric revisions (rubric_version is recorded in
# every score record, see config/benchmark.yaml judge.rubric_version).
# ---------------------------------------------------------------------------
WORKDIR /app
COPY pyproject.toml README.md /app/
COPY src /app/src
RUN pip install --no-deps -e /app \
 || pip install --no-deps /app

# Non-root scoring user.
RUN useradd --create-home --shell /bin/bash scorer \
 && mkdir -p /runs /scores /app/data /app/config \
 && chown -R scorer:scorer /app /scores
USER scorer

ENV REPROBENCH_LOG_LEVEL=INFO \
    PYTHONPATH=/app/src

# Deterministic, offline default: heuristic judge (no API keys required).
ENTRYPOINT ["reprobench"]
CMD ["score", "--runs", "/runs", "--out", "/scores", "--judge", "heuristic"]

# Offline smoke test build gate: `docker run --rm reprobench/scorer:24.04 verify`
HEALTHCHECK --interval=60s --timeout=20s --start-period=5s --retries=1 \
    CMD reprobench --help >/dev/null || exit 1
