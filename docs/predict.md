# Predict

```bash
sanntis predict INPUT... [-o OUTDIR] [--model HUB_ID@REV | DIR] [options]
sanntis predict PROTEINS.faa --gff GENES.gff [--ips-tsv IPS.tsv --ips-release 107.0] [options]
```

## Inputs

| Input | Genes | Notes |
|---|---|---|
| `.fna` `.fa` `.fasta`, optionally `.gz` | pyrodigal in `meta` mode by default, or `--genecall single`, which trains on inputs of at least 20 kb | prokaryotes only |
| `.gbk` `.gb` `.gbff`, optionally `.gz` | the CDS of the file | every CDS needs a `/translation`; pseudo genes without one are skipped; joined locations are reported as their span |
| `.faa`, optionally `.gz`, with `--gff` | the `CDS` rows of the GFF3 | see below; one input per run |

**Protein FASTA with GFF3.** Gene calls made elsewhere: the GFF3 gives the coordinates and the FASTA the
translations.

- The `ID` of each `CDS` row must equal a FASTA id, the first word of the header.
- Other feature types, score, phase and source are ignored. Rows and proteins may come in any order. Genes are
  ordered by their GFF coordinates.
- An `ID` that appears on several rows of the same sequence and strand is one CDS spanning them.
- A `##FASTA` section ends the GFF.
- A CDS without a protein stops the run. Proteins without a CDS are skipped with a warning, and their count is
  written to `run.json`.
- Contig lengths are unknown for this input, so `bp` is null in `run.json`.
- Any GFF3 whose CDS rows carry the protein ids in `ID` is accepted, such as the CDS GFF of a gene caller or a
  merged annotation GFF of the same gene calls. A GFF written by `--write-faa` can be read back.

Several `.fna` or `.gbk` inputs run in **batch mode**. The proteins of all inputs are pooled, deduplicated by md5
and annotated in one InterProScan 6 run. Each input is then scored on its own. Output files are named after the
input file name without extensions, so the inputs need distinct names.

## Annotation

By default InterProScan 6 runs through Nextflow; see [setup](setup.md). Proteins are first looked up in the
annotation cache in `$SANNTIS_CACHE/annotations`. The cache is keyed by protein md5, InterProScan version,
InterPro release and application list, so a protein annotated in an earlier run is not annotated again.

`--ips-tsv FILE` uses a precomputed InterProScan TSV instead. InterProScan 5.76 or later and 6.x are accepted, and
the version is detected from the file. Hits are joined to the CDS by protein id, which is the locus tag, or
`<contig>_<n>` for pyrodigal calls. When the ids do not match, hits are joined by the md5 of the sequence. Hits of
other analyses, such as PANTHER, are dropped. If Pfam, Gene3D or SUPERFAMILY have no hit at all in a TSV of 100 or
more proteins, the run stops; `--allow-incomplete-ips` continues instead. The InterPro release cannot be read from
the TSV, so it is declared with `--ips-release`. Without it, the release is recorded as `unknown`. A release that
differs from the model produces a warning. Signatures are mapped to InterPro entries with the table of the model,
not with the IPR column of the TSV.

## Options

| Option | Default | Description |
|---|---|---|
| `--model` | the Hub release matching the installed version | Hub id `repo@revision`, or a local model directory |
| `--gff` | | GFF3 with the CDS of a `.faa` input; required with `.faa` and accepted only with `.faa` |
| `--genecall meta\|single` | `meta` | pyrodigal mode for `.fna` inputs; `single` trains on inputs of at least 20 kb |
| `--ips-tsv`, `--ips-release` | | precomputed InterProScan TSV and its InterPro release, see Annotation |
| `--allow-incomplete-ips` | off | continue although Pfam, Gene3D or SUPERFAMILY have no hit, see Annotation |
| `--kingdom proka\|euk` | `proka` | kingdom of the run, used by the NPClassifier head; `euk` requires GenBank or `.faa` with `--gff` input |
| `--npc-precision 0.9\|0.8` | `0.9` | NPClassifier abstention tier |
| `--threads` | 1 | torch and pyrodigal threads |
| `--allow-ips-mismatch` | off | load a model built with another InterProScan version or InterPro release, with a warning |
| `--write-faa` | off | also write the CDS proteins and their GFF3 |
| `--ips-datadir` | `$SANNTIS_CACHE/ips6` | InterProScan 6 data |
| `--keep-work` | off | keep the Nextflow work directory of the InterProScan 6 run |
| `-o`, `--outdir` | `.` | output directory |
| `--container-engine`, `--nf-config`, `--nf-profile`, `--nf-args` | | forwarded to Nextflow, see [infrastructure](infrastructure.md) |

`sanntis -v predict ...` logs progress and `-vv` logs debug output.

## Outputs (per input)

| File | Content |
|---|---|
| `<prefix>.sanntis.gff3` | one `CLUSTER` feature per region with source `SanntiS`; the `##` header lines are listed below |
| `<prefix>.sanntis.antismash.json` | the same regions as antiSMASH sideloader JSON, see below |
| `<prefix>.orf_scores.tsv` | `record_id, cds_idx, locus_tag, score` |
| `<prefix>.regions.tsv` | one row per region with every attribute |
| `<prefix>.run.json` | run metadata: input, model, InterProScan mode, version and release, cache hits, join statistics, QC, timings |
| `<prefix>.cds.faa`, `<prefix>.cds.gff3` | written with `--write-faa` |

GFF attributes: `ID`, `nearest_MiBIG_class` (equal to `bgc_class`), `bgc_class`, `p_non_bgc`, `n_orfs`,
`npc_pathway`, `npc_pathway_score`, `npc_pathway_confident`, `npc_superclass`, `npc_superclass_top`,
`npc_superclass_score`, `npc_pathway_key_domains_support`, `npc_pathway_key_domains_unsupported`.

### NPC pathway key-domain support

The model carries, for each NPC pathway, a short list of key domains and domain pairs. These are InterPro families
and domains that are specific to the pathway among the labelled training BGCs and rare outside BGCs. Examples are
ACP, PCP and the PKS ketosynthase domain for polyketides, NRPS adenylation with PCP and YcaO for peptides, and
polyprenyl synthetase and terpene cyclases for terpenoids. Superfamilies and unintegrated signatures are not used.
A pair counts when both domains occur anywhere in the region. Each called pathway of a region receives one of
three values:

| Value | Meaning |
|---|---|
| supported | at least one of its key domains or pairs is in the region |
| unsupported | none of them is in the region |
| `na` | the pathway has no rules, or its rules covered less than 50 % of its true regions in cross-validation. In the 1.0 release this applies to Fatty acids and to Shikimates & Phenylpropanoids |

- `npc_pathway_key_domains_unsupported` is `true` when at least one called pathway is unsupported, `false` when
  every judged pathway is supported, and `na` when every called pathway is `na`.
- `npc_pathway_key_domains_support` lists `<pathway>:<rule>` for each key domain or pair present, with pairs
  written as `IPR1+IPR2`, `<pathway>:none` for an unsupported pathway and `<pathway>:na` for an `na` pathway. The
  entries are joined by `|`, for example `Polyketides:IPR020841|Polyketides:IPR014030+IPR014031|Terpenoids:none`.

The flag marks calls that deserve a closer look. It is not a correction of the call. In the out-of-fold
cross-validation of the 1.0 release it was raised on 3 % of the correct pathway calls and on 15 % of the wrong
ones. Flagged calls were wrong about three times as often as unflagged calls, 51 % against 16 %, also among
confident calls. Most wrong calls do carry the key domains, typically in hybrid clusters or through tailoring
enzymes, and Polyketides and Amino acids and Peptides calls are rarely flagged. The training report of each model
has the per-pathway numbers.

Header lines: `sanntis-version`, `input`, `model-source`, `model-sanntis-version`, `model-created`, `model-ips`,
`ips`, `kingdom`, `npc-precision`, `qc-cds-with-tokens`, `qc-unk-token-share`, `qc-flags`.

### antiSMASH sideload

`<prefix>.sanntis.antismash.json` follows the [antiSMASH sideloader
schema](https://github.com/antismash/antismash/tree/master/antismash/detection/sideloader/schemas/general), so the
regions can be shown in antiSMASH next to its own detections:

```
antismash --sideload genome.sanntis.antismash.json genome.gbk
```

- `tool`: the name `SanntiS`, the version and, as `configuration`, the GFF header lines.
- `records`: one per input record. Records without regions have no subregions.
- Each region is one subregion. `start` and `end` are 0-based with an exclusive end, that is GFF `start - 1` and
  `end`. `label` is `bgc_class`, or `SanntiS` when the model has no class head. `details` holds `score` and the GFF
  attributes, unescaped; `npc_pathway` and `npc_pathway_key_domains_support` are lists.
- Regions are subregions, not protoclusters. antiSMASH draws a region around them but runs no product-specific
  analysis on them.

```json
{"tool": {"name": "SanntiS", "version": "1.0.0b1", "description": "...", "configuration": {"kingdom": "proka", "...": "..."}},
 "records": [{"name": "BGC0002981",
              "subregions": [{"start": 117, "end": 27635, "label": "ribosomal",
                              "details": {"score": "0.8118", "ID": "BGC0002981_sanntis_1", "bgc_class": "ribosomal",
                                          "npc_pathway": ["Amino acids and Peptides"], "...": "..."}}]}]}
```

antiSMASH must be given the nucleotide records SanntiS was run on: the same FASTA or GenBank file, or, for `.faa`
input, the genome the proteins come from. antiSMASH matches `name` to its record ids and skips records it does not
know. It stops with "sideloaded area contains no complete CDS" when its own gene calls leave a region without a
complete CDS.

## Coverage QC

Every run reports the share of CDS with at least one domain token, the share of tokens outside the model
vocabulary, written `<unk>`, and the hits per application. When the model has reference values, the flags below
are raised in the log, the GFF header and `run.json`. The reference values are the `coverage_reference` of
`config.json`: genome-wide rates of the negative pool per kingdom, compared with the `--kingdom` of the run.

| Flag | Condition |
|---|---|
| `low_cds_coverage` | CDS with tokens below 0.5 times the reference |
| `high_unk_share` | `<unk>` share above 2 times the reference plus 0.05 |
| `low_<app>_hits` | hits per CDS below 0.25 times the reference, for applications with a reference of at least 0.05 and inputs of at least 100 CDS |
| `no_annotations`, `no_cds` | always checked |
| `missing_apps:<apps>` | applications without any hit |
