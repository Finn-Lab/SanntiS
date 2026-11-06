# SanntiS 1.0 inference image: the package, CPU-only torch and the model bundle of
# the same release baked in, so `sanntis predict --ips-tsv ...` runs offline (the
# MGnify BGCs ETL runs it on compute nodes without Hub access). Managed
# InterProScan 6 mode (Nextflow + a container engine) is not part of the image.
#
#   docker build --build-arg SANNTIS_REVISION=v1.0.0 -t quay.io/microbiome-informatics/sanntis:1.0.0 .
#   docker run --rm quay.io/microbiome-informatics/sanntis:1.0.0 predict --help
FROM python:3.12-slim AS runtime

ARG SANNTIS_REVISION
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/sanntis/hf \
    HF_HUB_OFFLINE=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    SANNTIS_CACHE=/tmp/sanntis-cache \
    OMP_NUM_THREADS=1

RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates procps tini \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/sanntis/src
COPY pyproject.toml README.md LICENSE CITATION.cff ./
COPY src ./src

RUN python -m pip install --upgrade pip \
 && python -m pip install torch --index-url https://download.pytorch.org/whl/cpu \
 && python -m pip install . \
 && rm -rf /opt/sanntis/src

# Bake the model bundle of this release (sha256-verified on load). The build needs
# network access once; afterwards HF_HUB_OFFLINE=1 serves it from HF_HOME.
RUN HF_HUB_OFFLINE=0 python -c "from sanntis.model.bundle import SanntisModel, default_model_id; \
import os; rev = os.environ.get('SANNTIS_REVISION') or None; \
mid = ('microbiome-informatics/sanntis@' + rev) if rev else default_model_id(); \
m = SanntisModel.from_pretrained(mid); print('baked', m.source)" \
 && python -c "from sanntis.model.bundle import SanntisModel; m = SanntisModel.from_pretrained(); print('offline load ok', m.source)" \
 && mkdir -p /tmp/sanntis-cache && chmod -R a+rX /opt/sanntis/hf

WORKDIR /data
ENTRYPOINT ["/usr/bin/tini", "--", "sanntis"]
CMD ["--help"]
