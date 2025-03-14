[![Docker Repository on Quay](https://quay.io/repository/microbiome-informatics/sanntis/status "Docker Repository on Quay")](https://quay.io/repository/microbiome-informatics/sanntis)

# SanntiS

> SMBGC Annotation using Neural Networks Trained on Interpro Signatures

Tool for identifying biosynthetic gene clusters (BGCs) in genomic & metagenomic data


## How to use SanntiS?

### Conda

Requires:
* Linux OS/Unix-like
* InterProScan > 5.52-86.0: 
  - https://interproscan-docs.readthedocs.io/en/latest/InstallationRequirements.html 
      Non Linux OS can't run InterProScan. InterProScan output must be provided in TSV or GFF3 format sing "--ip-file" and a GBK as SEQUENCE
* Bioconda: https://bioconda.github.io

### Installation

```bash
conda create -n sanntis sanntis
conda activate sanntis
```

#### Installation testing

Tests require [pytest-workflow](https://pypi.org/project/pytest-workflow/).
```bash
pip install pytest-workflow 
```

Verify installation using preprocessed InterProScan outputs (supported on macOS and Linux).

```bash
pytest --tag sanntis_with_preprocessed_files
```

Verify installation and ensure InterProScan is set up correctly (supported on Linux).

```bash
pytest --tag sanntis_full_dependencies
```


### Support of preprocessed InterProScan outputs

SanntiS can be executed using preprocessed InterProScan outputs along with a GenBank (GBK) file specifying the coding sequences (CDSs). This integration increases user flexibility.
```bash
conda activate sanntis
sanntis --ip-file test/files/BGC0001472.fna.prodigal.faa.gff3 test/files/BGC0001472.fna.prodigal.faa.gb
conda deactivate sanntis
```

Additionally, the --ip-file option can be provided with a protein FASTA file containing headers formatted according to Prodigal's convention. In this case, the --is_protein flag must be included to indicate that the sequence file is a proteins FASTA. 
```bash
conda activate sanntis
sanntis --is_protein --ip-file test/files/BGC0001472.fna.prodigal.faa.gff3 test/files/BGC0001472.fna.prodigal.faa
conda deactivate sanntis
```

###  Docker:

#### Get InterProsScan data:
##### The size of download file is ~24G, the final directory is 16G. Be sure to have enough space
```bash
bash ./get_ips_slim.sh
```

#### Docker ready to use script:
##### Only works if "data/" and sanntis_container.py are in the same directory
```bash
sanntis_container.py --help
sanntis_container.py [OPTIONS] ARGUMENTS
```

#### Docker image shell:
```bash
docker -it --entrypoint bash -v <path to SanntiS/docker>/data/:/opt/interproscan quay.io/repository/microbiome-informatics/sanntis
sanntis --help
sanntis [OPTIONS] ARGUMENTS
```


## Ouput

  GFF3 format file

  The fields in this header are as follows:

    seqname: SeqID of contig, as in prodigal output.
    source: sanntis version.
    feature: Feature type name, i.e. CLUSTER, CLUSTER_border, CDS.
    start: Start position of feature
    end: End position of feature
    score: empty
    strand: empty
    frame: empty
    attributes:
      ID: ordinal ID for the cluster, beginning with 1.
      nearest_MiBIG: MiBIG accession of the nearest BGC to the cluster in the MIBIG space, measured in Dice dissimilarity coefficient.
      nearest_MiBIG_class: BGC class of nearest_MiBIG.
      nearest_MiBIG_diceDistance: Dice dissimilarity coefficient between ID and nearest_MiBIG.
      score: Post-processing probability output.
      partial: Indicates if a CLUSTER is at the edge of the contig. First and second digits represent 5' and 3' end, respectively. Same as in prodigal's `partial`. "0" shows the cluster is not at the edge, whereas a "1" indicates is at that edge, (i.e. a partial cluster).

  Sample:

    ##gff-version 3
    DS999642	SanntiSv0.9.0	CLUSTER	1	136970	.	.	.	ID=DS999642_sanntis_1;nearest_MiBIG=BGC0001397;nearest_MiBIG_class=NRP Polyketide;nearest_MiBIG_diceDistance=0.561;partial=10

### antiSMASH compatibility

SanntiS prioritises seamless integration with various downstream analysis tools, leveraging a GFF3 file output for broad compatibility. In addition, one of the key features in this regard is the ability to generate an output compatible with [antiSMASH](https://doi.org/10.1093/nar/gkad344), a widely used tool in the BGC analysis ecosystem.

#### `--antismash_output` Option

SanntiS has an `--antismash_output` option. This option allows you to create a JSON file formatted according to the specifications of [antiSMASH](https://docs.antismash.secondarymetabolites.org/sideloading/). 

```bash
sanntis --antismash_output True test/files/BGC0001472.fna
```
Executing the command above produces a file named with the suffix `antismash.json` facilitating its use in antiSMASH for enriched analysis. Specifically, this file can be uploaded to the [antiSMASH web server](https://antismash.secondarymetabolites.org) under 'Data input' > 'Upload extra annotations', allowing for an integrated analytical approach that leverages external annotation data.

## Cite
  If you use SanntiS make sure to cite the publication 
[Expansion of novel biosynthetic gene clusters from diverse environments using SanntiS](https://www.biorxiv.org/content/10.1101/2023.05.23.540769v1)
```
Expansion of novel biosynthetic gene clusters from diverse environments using SanntiS
Santiago Sanchez, Joel D. Rogers, Alexander B. Rogers, Maaly Nassar, Johanna McEntyre, Martin Welch, Florian Hollfelder, Robert D. Finn
bioRxiv 2023.05.23.540769; doi: https://doi.org/10.1101/2023.05.23.540769
```
