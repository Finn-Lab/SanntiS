#!/usr/bin/env bash

## GET INTERPROSCAN-5.52.86.0 DATA
wget ftp://ftp.ebi.ac.uk/pub/databases/metagenomics/genomes-pipeline/interproscan-5.52-86.0_data.tgz
tar -zxvf interproscan-5.52-86.0_data.tgz \
  --exclude=data/cdd \
  --exclude=data/hamap \
  --exclude=data/pirs* \
  --exclude=data/sfld \
  --exclude=data/superfamily \
  --exclude=data/tmhmm \
  --exclude=data/panther \
  --exclude=data/phobius \
  --exclude=data/smart
rm interproscan-5.52-86.0_data.tgz
