# Negative genome pool

`sanntis train --negatives` needs a negative pool: genomes with their native BGCs removed, annotated with the same
InterProScan version as the positives. The released pool is on the Hub. `sanntis build-negatives` builds a new
one.

```bash
sanntis build-negatives --out pools/ --container-engine docker [--genomes genomes.tsv] [--pool-version v1]
```

## Using a pool

`--negatives` takes a pool directory or a version of the training-data dataset on the Hub. The version `v1`
resolves to the tag `negatives-v1` of `microbiome-informatics/sanntis-training-data`. Another dataset is selected
with `repo@v1` or with `$SANNTIS_DATA_REPO`. Every file is checked against the sha256 in the `manifest.json` of
the pool.

## Steps

1. **Sampling** writes `genomes.tsv`: 205 GTDB r226 species representatives balanced over phyla, sampled with
   seed 42, and RefSeq eukaryotes, 3 per phylum for fungi and 2 for protozoa and plants. `--genomes` provides the
   table instead; see below.
2. **Prepare** is a fan-out step with the label `cpu_small`. Each assembly is downloaded and md5-verified, and its
   CDS table is built. Prokaryotic genes are called with pyrodigal in single mode, or in meta mode below 100 kb. A
   local `.gbk` or `.gbff` file contributes its own CDS. Eukaryotic CDS come from the RefSeq GBFF.
3. **antiSMASH 8.0.1** is a fan-out step with the label `antismash`. It runs `--minimal --genefinding-tool none`
   per genome. With a container engine it runs in `quay.io/nf-core/antismash:8.0.1--pyhdfd78af_0`;
   `--antismash-image` selects another image for `--runner inline`. Without a container engine, `antismash` 8.0.1
   must be on `PATH`. `--antismash-databases` binds the antiSMASH databases.
4. **Stripping** masks the CDS that overlap an antiSMASH region. The remaining CDS form BGC-free runs. The longest
   run of at least 50 CDS of 55 prokaryotic and 15 eukaryotic genomes is reserved as an isolate test backbone and
   kept out of the training background.
5. **InterProScan 6** annotates the unique pool proteins and produces the tokens per CDS. `--ips-tsv` provides
   precomputed md5-keyed TSVs instead.
6. **Pool version** writes `<out>/<pool-version>/` with `genomes.parquet`, `cds_tokens.parquet`,
   `clean_runs.parquet`, `native_mask.parquet`, `test_backbones.parquet` and `manifest.json`. The manifest holds
   the pins, the counts, the genome-wide annotation coverage used as the QC reference of the model, and the sha256
   of every file.

Every stage resumes. Finished units have done markers and InterProScan 6 runs with `-resume`. Fan-out steps run
through Nextflow or with `--runner inline --workers N`, as in [training](train.md#steps-resume-and-runners).

## Genome table

`--genomes` takes a TSV and skips sampling. The columns `genome` and `kingdom` (`proka` or `euk`) are required.
Optional columns: `lineage`, `source`, `accession` (default: the `genome` value), `group`, `taxon` (the antiSMASH
`--taxon`, default `bacteria` or `fungi`), `url` (the assembly directory) and `path` (a local file, relative to
the table, which skips the download).

## Options

| Option | Default | Description |
|---|---|---|
| `--out` | | build directory (required) |
| `--pool-version` | `v1` | name of the pool directory written in `--out` |
| `--genomes` | | genome table, skips sampling |
| `--antismash-databases` | | antiSMASH 8.0.1 databases directory |
| `--antismash-image` | the nf-core image | antiSMASH image for `--runner inline` with `--container-engine` |
| `--ips-tsv` | | precomputed md5-keyed InterProScan TSV(s) of the pool proteins (repeatable) |
| `--entries-json` | `$SANNTIS_CACHE/ips6/interpro/<release>/entries.json` | InterPro `entries.json` for the signature map |
| `--runner`, `--workers`, `--dry-run` | `nextflow`, 1 | as for `sanntis train` |
| `--container-engine`, `--nf-config`, `--nf-profile`, `--nf-args` | | see [infrastructure](infrastructure.md) |
