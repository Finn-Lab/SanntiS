# Model card template

`save_pretrained` writes a `README.md` from the manifest. The model card of a published model extends it along
these lines:

````markdown
---
library_name: sanntis
license: apache-2.0
tags: [biology, bgc, genomics, interproscan]
---

# <model name>

## Model
- sanntis version, bundle schema, InterProScan version, InterPro release, member databases
- detector seeds, vocabulary size, operating point t*, RF drop threshold, heads enabled

## Training data
- MIBiG release and archive sha256, external MIBiG-format archives and their sha256, NPClassifier labelling,
  negative pool version
- counts: BGCs per class, negative genomes, CDS

## Evaluation
- cross-validation results: tuned epoch count, t*, NPC thresholds
- pooled out-of-fold metrics, see `report/`

## Intended use and limitations
- prokaryotic .fna, GenBank with CDS translations, or protein .faa with a GFF3; the last two also for eukaryotes
- annotations from other InterProScan versions or InterPro releases shift the tokens; check the QC flags

## Usage
```bash
sanntis predict genome.fna --model <repo>@<revision>
```
````
