[![Docker Repository on Quay](https://quay.io/repository/microbiome-informatics/emerald-bgc/status "Docker Repository on Quay")](https://quay.io/repository/microbiome-informatics/emerald-bgc)

# emeraldBGC

emeraldBGC - SMBGC detection tool -

## How to use emeraldBGC?

### Conda

Requires:
* Linux OS/Unix-like
* InterProScan > 5.52-86.0: 
  - https://interproscan-docs.readthedocs.io/en/latest/InstallationRequirements.html 
      Non Linux OS can't run InterProScan. InterProScan output must be provided in TSV or GFF3 format sing "--ip-file" and a GBK as SEQUENCE
* Bioconda: https://bioconda.github.io/user/install.html

#### Installation

```bash
conda create -n emeraldbgc emeraldbgc
conda activate emeraldbgc
conda install -c bioconda emeraldbgc
```

#### Basic tests

```bash
$ conda activate emeraldbgc
$ emeraldbgc test/files/BGC0001472.fna
$ conda deactivate emerald
```

 Run with interproscan file:
```bash
$ conda activate emeraldbgc
$ emeraldbgc --ip-file test/files/BGC0001472.fna.prodigal.faa.gff3 test/files/BGC0001472.fna.prodigal.faa.gb
$ conda deactivate emerald
```

###  Docker:

#### Get InterProsScan data:
##### The size of download file is ~24G, the final directory is 16G. Be sure to have enough space
```bash
$ bash ./get_ips_slim.sh
```

#### Docker ready to use script:
##### Only works if "data/" and emeraldbgc_container.py are in the same directory
```bash
$ emeraldbgc_container.py --help
$ emeraldbgc_container.py [OPTIONS] ARGUMENTS
```

#### Docker image shell:
```bash
$ docker -it --entrypoint bash -v <path to emeraldBGC/docker>/data/:/opt/interproscan quay.io/repository/microbiome-informatics/emerald-bgc
$ emeraldbgc --help
$ emeraldbgc [OPTIONS] ARGUMENTS
```


## Ouput

  GFF3 format file

  The fields in this header are as follows:

    seqname: SeqID of contig, as in prodigal output.
    source: emeraldbgc version.
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
    DS999642	EMERALDv0.9.0	CLUSTER	1	136970	.	.	.	ID=DS999642_emrld_1;nearest_MiBIG=BGC0001397;nearest_MiBIG_class=NRP Polyketide;nearest_MiBIG_diceDistance=0.561;partial=10
