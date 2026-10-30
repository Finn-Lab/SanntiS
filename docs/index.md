# SanntiS

SanntiS detects and classifies biosynthetic gene clusters (BGCs) from InterProScan domain annotations.

```bash
conda create -n sanntis -c conda-forge -c bioconda sanntis   # or: pip install "git+https://github.com/Finn-Lab/SanntiS@v1.0.0-beta.1"
sanntis setup --container-engine docker            # optional: InterProScan 6.0.3.0 and InterPro 107.0 data
sanntis predict genome.fna --container-engine docker -o out/
```

- Inputs: a prokaryotic nucleotide FASTA, whose genes are called with pyrodigal; a GenBank file with CDS
  translations, of any kingdom; or a protein FASTA with the GFF3 of its CDS, passed with `--gff`. Each input may
  come with a precomputed InterProScan TSV, passed with `--ips-tsv`.
- Outputs: a GFF3 with one `CLUSTER` feature per BGC, per-ORF scores, a region table, antiSMASH sideloader JSON
  and run metadata.
- Models: releases on Hugging Face Hub, selected by tag, or local directories with the same layout.
