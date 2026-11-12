# Install

## Conda

```bash
conda create -n sanntis -c conda-forge -c bioconda sanntis
conda activate sanntis
```

The bioconda package includes Nextflow, Java and the `[train]` extras, matplotlib and RDKit.

## pip

SanntiS is published on [PyPI](https://pypi.org/project/sanntis/). On Linux, `pip install torch` installs the
CUDA build of PyTorch (several GB); install the CPU-only build first when there is no GPU.

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu   # optional: CPU-only PyTorch
pip install sanntis              # Python >= 3.11
pip install "sanntis[train]"     # with the training extras
```

A specific tag or an unreleased branch can be installed from GitHub:
`pip install "git+https://github.com/Finn-Lab/SanntiS@v1.2.0"`.

## Container

Every release is published as an image on quay.io with CPU-only PyTorch and the model
bundle of that release baked in, so `predict --ips-tsv` works offline
(`HF_HUB_OFFLINE=1`; `--model` can still point at a local bundle directory). Managed
InterProScan 6 mode is not part of the image.

```bash
docker run --rm -v "$PWD:/data" quay.io/microbiome-informatics/sanntis:1.2.0 \
    predict /data/genome.gbk --ips-tsv /data/genome.ips.tsv --ips-release 107.0 -o /data/out
singularity exec docker://quay.io/microbiome-informatics/sanntis:1.2.0 \
    sanntis predict genome.gbk --ips-tsv genome.ips.tsv --ips-release 107.0 -o out
```

The image is built by the release workflow (`.github/workflows/release.yml`, job
`container`) from the `Dockerfile` in the repository root; `workflow_dispatch` builds any
branch as `quay.io/microbiome-informatics/sanntis:<branch>` for testing.

## Running InterProScan

Without `--ips-tsv`, `sanntis predict` runs InterProScan 6. This requires:

- **Nextflow >= 25.10.4** and Java 17. With Nextflow 25.10.x, InterProScan 6 fails as soon as a configuration
  file is passed with `--nf-config`. **Nextflow 26.x** is recommended; 26.10.3 is the tested version.
- A container engine, selected with `--container-engine`: Docker, Singularity, Apptainer or Podman.

Gene calling and the model run in the Python process on CPU.

With a precomputed InterProScan TSV passed as `--ips-tsv`, Nextflow and a container engine are not needed.

## Caches

| Path | Content |
|---|---|
| `$SANNTIS_CACHE` (default `~/.cache/sanntis`) | root of the sanntis caches |
| `$SANNTIS_CACHE/ips6` | InterProScan 6 data, tens of GB, downloaded once |
| `$SANNTIS_CACHE/annotations` | annotation cache: InterProScan hits per protein md5 |
| `$HF_HOME` | Hugging Face cache with the model releases; `HF_HUB_OFFLINE=1` uses a cached model without network access |
| `$SANNTIS_CACHE/mibig`, `npclassifier`, `ncbi`, `gtdb`, `refseq` | downloads and API answers of `sanntis train` and `sanntis build-negatives` |

## Environment variables

| Variable | Meaning |
|---|---|
| `SANNTIS_CACHE` | cache root, see above |
| `SANNTIS_THREADS` | torch and BLAS threads of `sanntis train` and its units. Default: the CPUs available to the process. `predict` uses `--threads` |
| `SANNTIS_IPS6_CHUNK` | maximum number of proteins per InterProScan 6 run, default 20000. Larger inputs run in sequential chunks |
| `SANNTIS_DATA_REPO` | Hub dataset in which `--negatives <version>` is resolved, default `microbiome-informatics/sanntis-training-data` |
| `NCBI_API_KEY`, `NCBI_EMAIL` | optional credentials for the NCBI taxonomy lookups of `train` and `build-negatives`, which raise the rate limit |
| `SANNTIS_DEBUG_MAX_EPOCHS` | caps the detector epochs, for tests and smoke runs. The cap is recorded in the model manifest and the model card carries a warning |
