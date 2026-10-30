<p align="center">
  <img src="docs/assets/sanntis_logo.svg" alt="SanntiS logo" width="320">
</p>

[![CI](https://github.com/Finn-Lab/SanntiS/actions/workflows/ci.yml/badge.svg)](https://github.com/Finn-Lab/SanntiS/actions/workflows/ci.yml)
[![Bioconda](https://img.shields.io/conda/vn/bioconda/sanntis.svg)](https://anaconda.org/bioconda/sanntis)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![DOI](https://img.shields.io/badge/DOI-10.1101%2F2023.05.23.540769-blue.svg)](https://doi.org/10.1101/2023.05.23.540769)

# SanntiS

> SMBGC Annotation using Neural Networks Trained on Interpro Signatures

Tool for identifying biosynthetic gene clusters (BGCs) in genomic & metagenomic data.

SanntiS takes the InterProScan domain annotations of the proteins of a sequence and reports BGC regions. Each
region carries a biosynthetic class (NRPS, PKS, ribosomal, terpene, saccharide, other) and the NPClassifier
pathway and superclass predicted for its product. Results are written as GFF3, as antiSMASH sideloader JSON and as
per-ORF and per-region tables.

## How to use SanntiS?

SanntiS runs in one of two modes:

1. [With precomputed InterProScan annotations](#1-with-your-own-interproscan-annotations). The input is an
   InterProScan TSV together with the sequence or gene calls. Only the Python package is required.
2. [Running InterProScan through SanntiS](#2-letting-sanntis-run-interproscan). SanntiS runs InterProScan 6 with
   Nextflow and a container engine.

### Installation

#### Conda

```bash
> [!WARNING]
> The Bioconda package is currently out of date and installs **v0.9.x**, which predates the latest major release.
> To get the current version, see **pip** below. This section will be updated once the new release is available on Bioconda.
# conda create -n sanntis -c conda-forge -c bioconda sanntis
# conda activate sanntis
```

The conda package includes Nextflow and Java, which the second mode requires.

#### pip

```bash
pip install "git+https://github.com/Finn-Lab/SanntiS@v1.0.0-beta.1"      # Python >= 3.11
```

On a machine without a GPU, install the CPU build of PyTorch first:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

The model is downloaded from [Hugging Face Hub](https://huggingface.co/microbiome-informatics/sanntis) the first
time `sanntis predict` runs. It is about 85 MB and is cached in `$HF_HOME`.

#### Installation test

`sanntis example` copies an example input to a directory: a protein FASTA, its GFF3 and its InterProScan TSV.

```bash
sanntis example -o sanntis_example/
sanntis predict sanntis_example/ERZ29794225_slice.faa --gff sanntis_example/ERZ29794225_slice.gff \
    --ips-tsv sanntis_example/ERZ29794225_slice.ips.tsv -o sanntis_example/out/
```

The run downloads the model on first use and writes `sanntis_example/out/ERZ29794225_slice.sanntis.gff3`.

### 1. With your own InterProScan annotations

Pass the InterProScan TSV with `--ips-tsv` together with the sequence the proteins come from. Nextflow and a
container engine are not used in this mode.

**GenBank with CDS translations**, prokaryotic or eukaryotic:

```bash
sanntis predict assembly.gbk --ips-tsv assembly.ips.tsv --ips-release 107.0 -o out/
```

**Protein FASTA with the GFF3 of its CDS**, such as the output of a gene caller or an annotation pipeline:

```bash
sanntis predict proteins.faa --gff genes.gff --ips-tsv proteins.ips.tsv --ips-release 107.0 -o out/
```

The `ID` of each `CDS` row in the GFF3 must equal the protein id, which is the first word of the FASTA header.
Rows and proteins may come in any order. Genes are ordered by their coordinates.

Requirements on the TSV:

- InterProScan **5.76 or later, or 6.x**. The version is detected from the file. The TSV must come from a run with
  the 14 member databases used by the model: Pfam, NCBIfam, SFLD, SUPERFAMILY, CATH-Gene3D, HAMAP, Coils, CDD,
  PRINTS, PIRSF, PROSITE profiles, PROSITE patterns, MobiDB-lite and SMART. Hits of other analyses, such as
  PANTHER, are ignored.
- The model was trained on **InterPro 107.0** annotations. `--ips-release` declares the release of the TSV. A
  different release produces a warning, and predictions degrade with the distance between releases.
- Hits are joined to the CDS by protein id, or by the md5 of the sequence when the ids differ. A TSV computed from
  the same proteins under other ids is therefore accepted.

Any input may be gzip-compressed.

### 2. Letting SanntiS run InterProScan

This mode requires **Nextflow >= 26.x** with Java 17 and a container engine: Docker, Singularity, Apptainer or
Podman. SanntiS runs InterProScan 6.0.3.0 with InterPro 107.0 data, the versions the model was trained on. Linux
is the tested platform. The containers are Linux images, so other systems need an engine that runs them, such as
Docker Desktop.

#### Get the InterProScan data

```bash
sanntis setup --container-engine docker
```

`setup` downloads the InterProScan 6 pipeline and the InterPro 107.0 data of the 14 member databases into
`$SANNTIS_CACHE/ips6`, by default `~/.cache/sanntis/ips6`. The data take tens of GB and are downloaded once.
`sanntis predict` runs the same setup when the data are missing, so this step is optional.

#### Run

```bash
sanntis predict genome.fna --container-engine docker -o out/            # prokaryotic nucleotide FASTA
sanntis predict assembly.gbk --container-engine docker -o out/          # GenBank with CDS translations
sanntis predict proteins.faa --gff genes.gff --container-engine docker -o out/
sanntis predict a.fna b.fna c.gbk --container-engine docker -o out/     # batch: one InterProScan run for all
```

| Input | Genes |
|---|---|
| `.fna` `.fa` `.fasta` | called with pyrodigal, `meta` mode by default and `--genecall single` for isolates; prokaryotes only |
| `.gbk` `.gb` `.gbff` | the CDS of the file, which need a `/translation`; any kingdom, with `--kingdom euk` for eukaryotes |
| `.faa` + `--gff` | the `CDS` rows of the GFF3; any kingdom |

Proteins are deduplicated and looked up in the annotation cache in `$SANNTIS_CACHE/annotations` before
InterProScan runs. Proteins seen in an earlier run are not annotated again.

On a cluster, pass a Nextflow configuration with `--nf-config` or `--nf-profile` and run `sanntis` itself as a
small batch job. See [Running on other infrastructure](docs/infrastructure.md).

### Options

```
sanntis predict INPUT... [-o OUTDIR] [--model HUB_ID@REV | DIR] [options]
```

| Option | Default | Description |
|---|---|---|
| `--ips-tsv FILE`, `--ips-release REL` | | precomputed InterProScan TSV and its InterPro release |
| `--gff FILE` | | GFF3 with the CDS of a `.faa` input |
| `--kingdom proka\|euk` | `proka` | kingdom of the input, used by the NPClassifier head |
| `--genecall meta\|single` | `meta` | pyrodigal mode for `.fna` inputs |
| `--npc-precision 0.9\|0.8` | `0.9` | NPClassifier abstention tier |
| `--model` | the Hub release matching the installed version | Hub id `repo@revision` or a local bundle directory |
| `--threads N` | 1 | CPU threads |
| `--write-faa` | off | also write the CDS proteins and their GFF3 |
| `--container-engine`, `--nf-config`, `--nf-profile`, `--nf-args` | | Nextflow options of mode 2 |

`sanntis predict --help` lists every option, and `sanntis -v predict ...` logs progress. The options are described
in [docs/predict.md](docs/predict.md).

## Output

Output files are named after the input file. `<prefix>` is the file name without extensions.

| File | Content |
|---|---|
| `<prefix>.sanntis.gff3` | one `CLUSTER` feature per BGC |
| `<prefix>.sanntis.antismash.json` | the same regions as antiSMASH sideloader JSON |
| `<prefix>.regions.tsv` | one row per region with every attribute |
| `<prefix>.orf_scores.tsv` | the detector score of every CDS |
| `<prefix>.run.json` | run metadata: input, model, InterProScan version, annotation QC flags, timings |

### GFF3

```
seqname:    record id of the contig, as in the input
source:     SanntiS
feature:    CLUSTER
start, end: 1-based, inclusive (first and last CDS of the region)
score:      mean detector score of the CDS of the region
strand, frame: empty
attributes:
  ID                        <record>_sanntis_<n>, n starting at 1 per record
  bgc_class                 biosynthetic class: NRPS, PKS, ribosomal, terpene, saccharide or other
  nearest_MiBIG_class       same value as bgc_class
  p_non_bgc                 post-filter probability that the region is not a BGC
  n_orfs                    number of CDS in the region
  npc_pathway, npc_pathway_score, npc_pathway_confident
                            NPClassifier pathway(s) of the product, score and whether it passes the
                            --npc-precision tier
  npc_superclass, npc_superclass_top, npc_superclass_score
                            NPClassifier superclass(es)
  npc_pathway_key_domains_support, npc_pathway_key_domains_unsupported
                            whether the region carries a key domain of each called pathway
```

Example line, with illustrative values:

```
##gff-version 3
##sanntis-version 1.0.0b1
BGC0002981	SanntiS	CLUSTER	118	27635	0.8118	.	.	ID=BGC0002981_sanntis_1;nearest_MiBIG_class=ribosomal;bgc_class=ribosomal;p_non_bgc=0.0312;n_orfs=22;npc_pathway=Amino acids and Peptides;npc_pathway_score=0.9134;npc_pathway_confident=true;npc_superclass=Oligopeptides;npc_superclass_top=Oligopeptides;npc_superclass_score=0.8710;npc_pathway_key_domains_support=Amino acids and Peptides:IPR003776;npc_pathway_key_domains_unsupported=false
```

The `##` header lines record the model, the InterProScan version and the annotation coverage QC of the run.
`qc-flags` lists warnings such as few CDS with domains or a missing member database. The attributes and the QC
flags are documented in [docs/predict.md](docs/predict.md).

### antiSMASH compatibility

Every run writes `<prefix>.sanntis.antismash.json` in the format of the
[antiSMASH sideloading specification](https://docs.antismash.secondarymetabolites.org/sideloading/). The file can
be uploaded to the [antiSMASH web server](https://antismash.secondarymetabolites.org) under 'Data input' >
'Upload extra annotations', or passed to the command line together with the nucleotide records SanntiS was run on:

```bash
antismash --sideload genome.sanntis.antismash.json genome.gbk
```

SanntiS regions appear as subregions next to the antiSMASH detections.

## Models

The default model is the Hugging Face Hub release whose tag matches the installed SanntiS version, currently
`microbiome-informatics/sanntis@v1.0.0-beta.1`. `--model` selects another Hub revision or a local model directory,
for example one produced by `sanntis train`.

```bash
sanntis predict genome.fna --model microbiome-informatics/sanntis@v1.0.0-beta.1   # explicit Hub revision
sanntis predict genome.fna --model runs/my_run/bundle                             # a retrained local model
HF_HUB_OFFLINE=1 sanntis predict ...                                              # offline, once cached
```

The model directory layout is described in [Model bundle format](docs/bundle.md). SanntiS can retrain a model
from a MIBiG release or from other BGC collections; see [Training a model](docs/train.md) and
[Negative genome pool](docs/negatives.md).

### Training data

The released model was trained on the data published as the Hub dataset
[`microbiome-informatics/sanntis-training-data`](https://huggingface.co/datasets/microbiome-informatics/sanntis-training-data):

| Path (tag) | Content |
|---|---|
| `mibig/5.0rc1-train/` (`mibig-5.0rc1-train`) | **Positives**: the 2,797 `active` and `pending` entries of MIBiG 5.0rc1 as a MIBiG-format archive pair. The JSON entries are the 5.0rc1 files; the GenBank record of each BGC comes from the MIBiG 4.0 archive when its locus is unchanged (2,621) and otherwise from the NCBI nuccore record of the first 5.0rc1 locus (174). `manifest.tsv` describes every record, `excluded.tsv` the 2 entries without a usable record and `provenance.json` the sha256 of the source archives. Training drops the 18 loci longer than 200 kb, leaving 2,777 BGCs |
| `negatives/v1/` (`negatives-v1`) | **Negative genome pool**: 205 GTDB r226 bacterial and archaeal species representatives balanced over phyla and 23 RefSeq fungal and protist genomes, with the regions detected by antiSMASH 8.0.1 removed. 55 prokaryotic and 15 eukaryotic BGC-free runs are held out as test backbones |
| `npclassifier/v1/` (`npclassifier-v1`) | **NPClassifier answers** for the compound structures of the positives, from which the NPC labels are derived |

Positives and negatives are annotated with InterProScan 6.0.3.0 and InterPro 107.0, the 14 member databases listed
above. With the archive pair downloaded from the dataset, the model is reproduced by:

```bash
sanntis train --run runs/release --mibig none \
    --external-gbk mibig_5.0rc1-train_gbk.tar.gz --external-json mibig_5.0rc1-train_json.tar.gz \
    --negatives v1 --npc-cache v1 --container-engine docker
```

`--negatives v1` and `--npc-cache v1` fetch the pool and the NPClassifier answers from the dataset, so no API
request is made. The bundle's `manifest.json` and model card record the sha256 of these inputs. The runbook that
builds the archives and the validation report of the run are in the [`sanntisv1-data-and-analysis`](https://github.com/Finn-Lab/sanntisv1-data-and-analysis) repository.

## Documentation

- [Install](docs/install.md), caches and environment variables
- [InterProScan setup](docs/setup.md)
- [Predict](docs/predict.md): inputs, options, outputs, QC
- [Running on other infrastructure](docs/infrastructure.md): SLURM and other executors
- [Model bundle format](docs/bundle.md), [Training](docs/train.md), [Negative genome pool](docs/negatives.md)

The pages build into a site with `pip install "sanntis[docs]"` and `mkdocs serve`.

## Development

```bash
git clone https://github.com/Finn-Lab/SanntiS.git && cd SanntiS
pip install -e ".[dev,train]"
ruff check src tests && pytest -q -m "not nextflow"
```

Tests marked `nextflow` need Nextflow on `PATH`.

## Citations

If you use SanntiS, please cite:

[Expansion of novel biosynthetic gene clusters from diverse environments using SanntiS](https://www.biorxiv.org/content/10.1101/2023.05.23.540769v1)

```
Expansion of novel biosynthetic gene clusters from diverse environments using SanntiS
Santiago Sanchez, Joel D. Rogers, Alexander B. Rogers, Maaly Nassar, Johanna McEntyre, Martin Welch, Florian Hollfelder, Robert D. Finn
bioRxiv 2023.05.23.540769; doi: https://doi.org/10.1101/2023.05.23.540769
```
