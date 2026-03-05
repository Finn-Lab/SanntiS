# SanntiS - CLAUDE.md

## Project Overview

SanntiS (SMBGC Annotation using Neural Networks Trained on Interpro Signatures) is a bioinformatics tool for identifying biosynthetic gene clusters (BGCs) in genomic and metagenomic data using machine learning.

- Version: 0.9.4.1
- License: Apache 2.0
- Published: "Expansion of novel biosynthetic gene clusters from diverse environments using SanntiS" (bioRxiv 2023)

## Repository Structure

```
sanntis/                  # Main package
├── __init__.py           # Loads version & params
├── __main__.py           # Entry point wrapper
├── _cli.py               # CLI argument parsing and orchestration
├── _params.json          # Model thresholds and configuration
├── build_gb.py           # Utility: builds GenBank from Prodigal output
├── download_data.py      # Downloads InterProScan dependencies
├── models/
│   ├── sanntis.h5        # TensorFlow neural network (~23.5 MB)
│   ├── post_filters.pickle  # Type classification model (~39 MB)
│   └── hmm_lib/sanntis.hmm  # Custom HMM library
└── modules/
    ├── BGCdetection.py   # Core ML inference and cluster detection
    ├── Preproc.py        # External tool execution (Prodigal, IPS, HMMER)
    └── WriteOutput.py    # GFF3 and antiSMASH JSON output generation
test/
├── test_sanntis.yml      # pytest-workflow test definitions
└── files/                # Test data (BGC0001472)
docker/                   # Docker container setup
docs/
└── bgc_class_descriptions.tsv
```

## Key Modules

- `sanntis/modules/BGCdetection.py`: Core logic — parses annotations, runs neural network inference, detects and types BGC clusters
- `sanntis/modules/Preproc.py`: Runs Prodigal (gene prediction), InterProScan, and HMMScan; handles input format detection
- `sanntis/modules/WriteOutput.py`: Writes GFF3 output and optional antiSMASH-compatible JSON
- `sanntis/_cli.py`: CLI entry point (`sanntis` command), orchestrates the full pipeline

## External Dependencies

The tool relies on external bioinformatics software (not Python packages):
- `prodigal`: Gene prediction from nucleotide sequences
- `interproscan`: Functional annotation (v5.52-86.0+, Linux only)
- `hmmer` / `hmmscan`: HMM-based annotation with the in-house sanntis.hmm

Python dependencies: `biopython`, `numpy`, `tensorflow`, `joblib`, `requests`

## Running & Testing

Do NOT run Python scripts directly — use the installed `sanntis` command.

```bash
# Basic usage
sanntis test/files/BGC0001472.fna

# With preprocessed InterProScan output (macOS compatible)
sanntis --is_protein \
        --ip-file test/files/BGC0001472.fna.prodigal.faa.gff3 \
        test/files/BGC0001472.fna.prodigal.faa

# Run tests (cross-platform, no InterProScan needed)
pytest --tag sanntis_with_preprocessed_files

# Run tests (Linux only, full pipeline)
pytest --tag sanntis_full_dependencies
```

## Model Configuration (`_params.json`)

- `thBase`: 0.855 — base detection threshold
- `thBorder`: 0.98 — high-confidence border threshold
- Greediness levels: 0 → 0.980, 1 → 0.855 (default), 2 → 0.675
- Neural network input shape: 200 features
- IPS applications: Pfam, TIGRFAM, PRINTS, ProSitePatterns, Gene3D

## Output Format

GFF3 with custom attributes on CLUSTER features:
- `nearest_MiBIG`: Most similar MiBIG accession
- `nearest_MiBIG_class`: BGC type (NRP, Polyketide, Terpene, etc.)
- `nearest_MiBIG_diceDistance`: Similarity score
- `score`: Detection probability
- `partial`: Edge indicator (5'/3')

Optional: antiSMASH 6.0-compatible JSON for web visualization.

## Notes

- InterProScan only works on Linux; use `--ip-file` with preprocessed outputs on macOS
- The environment is Dockerized — avoid running Python scripts directly
- CI/CD runs on Ubuntu via GitHub Actions (`.github/workflows/test.yml`)
