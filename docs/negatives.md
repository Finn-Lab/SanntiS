# Negative genome pool

`sanntis train --negatives` needs a negative pool: genomes with their native BGCs removed, annotated with the same
InterProScan version as the positives. The released pool is on the Hub. `sanntis build-negatives` builds a new
one.

```bash
sanntis build-negatives --out pools/ --container-engine docker [--genomes genomes.tsv] [--exclude exclude.tsv] \
    [--pool-version v1]
```

## Using a pool

`--negatives` takes a pool directory or a version of the training-data dataset on the Hub. The version `v1`
resolves to the tag `negatives-v1` of `microbiome-informatics/sanntis-training-data`. Another dataset is selected
with `repo@v1` or with `$SANNTIS_DATA_REPO`. Every file is checked against the sha256 in the `manifest.json` of
the pool.

## Steps

1. **Sampling** writes `genomes.tsv`: 205 GTDB r226 species representatives balanced over phyla, sampled with
   seed 42, and RefSeq eukaryotes, 3 per phylum for fungi and 2 for protozoa and plants. `--genomes` provides the
   table instead; see below. `--exclude` then removes genomes that must never enter the pool (below); they are
   listed in `excluded.tsv`.
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
Optional columns:

| Column | Meaning |
|---|---|
| `lineage`, `source`, `group` | descriptive |
| `accession` | default: the `genome` value |
| `taxon` | the antiSMASH `--taxon`, default `bacteria` or `fungi` |
| `url` | the assembly directory |
| `path` | a local file, relative to the table; skips the download |
| `quota_cds` | makes a **quota pool** (below) |
| `species` | the species `--exclude` matches; default: the `species__` / `s__` rank of `lineage` |

## Excluded genomes

`--exclude` takes a TSV with the columns `accession` and `species`; either may be empty in a row. A row with an
accession removes that assembly (the `GCA_` / `GCF_` prefix and the version are ignored, so both copies of an
assembly match). A row with a species removes every genome of the species. Use it for the genomes of test sets,
so that a pool never contains them. `plan-negatives --exclude` applies the same table before planning.

## Lineage-matched quota pools

The detector learns P(BGC | tokens). If a lineage holds many positives but few negatives, looking like that
lineage becomes evidence for a BGC.

`sanntis plan-negatives` writes a genome table with `quota_cds` that keeps the ratio of negative to positive CDS
about constant per lineage:
- a floor of `floor_cds` per phylum without positives;
- the rest of the budget in proportion to the positives' CDS per family (falling back to order, then phylum,
  when a family has no candidate genome);
- at most `cap` CDS and one genome per species; candidates are taken by `priority` (lower first), then in a
  seeded random order.

```bash
sanntis train --run runs/plan --mibig none --external-gbk pos_gbk.tar.gz --external-json pos_json.tar.gz \
    --until annotate                                    # the production positives (InterProScan 6 on them)
sanntis plan-negatives --out genomes.tsv --positives runs/plan \
    --taxdump https://ftp.ncbi.nlm.nih.gov/pub/taxonomy/taxdump_archive/taxdmp_2026-07-01.zip \
    --gtdb-release 226 --candidates fungi.tsv --exclude exclude.tsv --budget proka=469554 --budget euk=186849
sanntis build-negatives --out pools/ --genomes genomes.tsv --exclude exclude.tsv --pool-version v2 ...
```

Inputs of `plan-negatives`:

| Option | Meaning |
|---|---|
| `--positives` | a `sanntis train` run stopped after `annotate`: its deduplicated positives, eukaryotes restricted to the recipe's lineages |
| `--taxdump` | one NCBI taxdump (a `taxdmp_<date>.zip` of the NCBI archive, path or URL with `--taxdump-sha256`, or its directory): the lineage of every positive and candidate, by NCBI taxid |
| `--gtdb-release` | candidates: the GTDB species representatives of the release with CheckM2 completeness >= `--gtdb-min-completeness` (90) and contamination <= `--gtdb-max-contamination` (5); `clean_cds` = 0.9 x protein count; priority `--gtdb-priority` (1) |
| `--candidates` | candidate tables (TSV: `genome`, `kingdom`, `accession`, `taxid`, `source`, `priority`, `clean_cds`; optional `url`, `taxon`, `species`). `clean_cds` estimates the BGC-free CDS of the genome |
| `--exclude` | as for `build-negatives` |
| `--budget` | negative CDS per track (`proka=<n>`, `euk=<n>`) |
| `--cap`, `--floor-cds`, `--min-quota`, `--seed` | 1,500, 250, 100, 0 |
| `--breadth`, `--breadth-cds` | a candidate table whose genomes are all added with `min(--breadth-cds, clean_cds)` CDS when they are not planned or excluded |

Candidates below `--min-quota` CDS are dropped, and an assembly listed twice is kept once (lowest priority). The
table carries the plan columns (`phylum`, `order`, `family`, `plan_role`: floor, matched or breadth, ...), and
`<out>.stats.json` the budgets, the shortfalls per track and the options.

In a quota pool each genome keeps `quota_cds` BGC-free CDS:
- its clean runs are concatenated across contigs, so short runs of fragmented assemblies are not lost;
- the CDS are drawn as 3 windows, one per stratum, and written to `portions.parquet`;
- only portion and test-backbone CDS are annotated;
- training cuts its chunks and background pieces from the portions.

## Options

| Option | Default | Description |
|---|---|---|
| `--out` | | build directory (required) |
| `--pool-version` | `v1` | name of the pool directory written in `--out` |
| `--genomes` | | genome table, skips sampling |
| `--exclude` | | genomes never in the pool (accession / species table) |
| `--antismash-databases` | | antiSMASH 8.0.1 databases directory |
| `--antismash-image` | the nf-core image | antiSMASH image for `--runner inline` with `--container-engine` |
| `--ips-tsv` | | precomputed md5-keyed InterProScan TSV(s) of the pool proteins (repeatable) |
| `--entries-json` | `$SANNTIS_CACHE/ips6/interpro/<release>/entries.json` | InterPro `entries.json` for the signature map |
| `--runner`, `--workers`, `--dry-run` | `nextflow`, 1 | as for `sanntis train` |
| `--container-engine`, `--nf-config`, `--nf-profile`, `--nf-args` | | see [infrastructure](infrastructure.md) |
