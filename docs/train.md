# Training a model

`sanntis train` builds a model from data collection to a validation report. A retrained model is a local directory
with the same layout as the Hub release and loads the same way, with `--model runs/my_run/bundle`.

```bash
pip install "sanntis[train] @ git+https://github.com/Finn-Lab/SanntiS@v1.0.0-beta.1"
sanntis train --run runs/my_run --mibig 4.0 --negatives <pool dir | hub revision> --container-engine docker
```

A run has three parts:

1. A 5-split cross-validation on the training data tunes the epoch count, the operating point t* and the NPC
   thresholds.
2. The production fit trains 3 detector seeds, the RF post-filter and the NPC head.
3. The model directory and the validation report are written. The report is computed from the pooled out-of-fold
   predictions.

Test sets and comparisons with other tools or other SanntiS versions are not part of `sanntis train`. The protocols
of the paper, the MIBiG 3.1 validation on Test A and Test B and the tool comparison, are runbooks of the
`sanntisv1-data-and-analysis` repository. They drive `sanntis train` and score with `sanntis.train.metrics`.

## Inputs

| Option | Description |
|---|---|
| `--mibig <release>` | MIBiG release to train on, for example `4.0`. `mibig_json_<release>.tar.gz` and `mibig_gbk_<release>.tar.gz` are downloaded from `dl.secondarymetabolites.org/mibig/` into `$SANNTIS_CACHE/mibig`, and their sha256 is recorded in the provenance. Active entries are used. An active entry missing from the GBK archive is excluded and listed in `collect/exclusions.tsv`. Unset, `none` or `false` trains on external data only |
| `--external-gbk A_gbk.tar.gz` | BGCs in the MIBiG format described below; repeatable |
| `--external-json A_json.tar.gz` | the JSON archive of each `--external-gbk`, in the same order; give none or one per GBK archive. Needed for the class and NPC heads |
| `--negatives <dir or revision>` | negative genome pool version, built with [`sanntis build-negatives`](negatives.md) or taken from the training-data dataset on the Hub |
| `--npc-cache <dir or version>` | published NPClassifier answers that seed `$SANNTIS_CACHE/npclassifier` before labelling; see [NPC labels](#npc-labels) |
| `--ips-tsv FILE` | precomputed InterProScan TSV(s) of the collected proteins, with the md5 in column 2; skips InterProScan 6 |
| `--ips-datadir DIR` | InterProScan 6 data, default `$SANNTIS_CACHE/ips6`; also holds the InterPro `entries.json` |
| `--entries-json FILE` | InterPro `entries.json` for the signature map and the key-domain entry types; default `<ips-datadir>/interpro/<release>/entries.json` |

Run options: `--runner nextflow|inline`, `--workers N` for the inline pool size, `--gpu` to run the detector fits
with the label `gpu`, `--from STEP`, `--dry-run`, `--config run.yaml`, and the Nextflow options of
[infrastructure](infrastructure.md).

At least one of `--mibig` and `--external-gbk` is required. A `run.yaml` passed with `--config` takes the same
names: `mibig`, `external_gbk` and `external_json` as lists, `negatives`, and so on. Explicit command line values
take precedence.

### MIBiG-format data

MIBiG and external data go through the same reader and validation in `sanntis.train.mibig`:

- `*_gbk*.tar.gz`: one GenBank file per BGC, named `<accession>.gbk`, holding exactly one record. Every CDS
  carries a `/translation`.
- `*_json*.tar.gz`: one MIBiG 4 JSON entry per BGC, named `<accession>.json`. The fields read are:

  | Field | Use |
  |---|---|
  | `accession` | the BGC id, equal to the GBK name; letters, digits, `_ . : -` |
  | `biosynthesis.classes[].class` | at least one of NRPS, PKS, ribosomal, terpene, saccharide, other |
  | `taxonomy.ncbiTaxId` | integer NCBI taxon, from which the kingdom (prokaryote or eukaryote) and the report lineage are derived |
  | `compounds[].structure` | SMILES, optional, used for the NPClassifier labels |
  | `status` | MIBiG only; `active` entries are used. External entries are all used |

External data is validated as a whole before the run starts, and every problem is listed: JSON entries without a GBK,
GBKs without a JSON entry, invalid fields or records, CDS without a `/translation`, and accessions already taken by
the MIBiG release or by another archive. An invalid MIBiG entry is excluded instead and listed in
`collect/exclusions.tsv`. Without a JSON archive, the BGCs train the detector only, with no class and no NPC
labels. Their kingdom comes from the `/db_xref="taxon:N"` of the GBK source feature, and defaults to prokaryote.

### NPC labels

The NPClassifier labels of every BGC with a JSON entry are computed from its compounds:

1. Each SMILES is normalised with RDKit: largest fragment, uncharged, canonical tautomer.
2. The InChIKey skeleton of the normalised structure is sent to the NPClassifier API at `npclassifier.gnps2.org`,
   one request per skeleton.
3. The answer is mapped to the terms of the NPClassifier vocabulary shipped with the package.
4. A BGC takes the union of the labels of its compounds.

API answers are cached in `$SANNTIS_CACHE/npclassifier`, one JSON file per SMILES named by its sha1. MIBiG 5.0rc1
has about 3,800 skeletons, which take about one hour at one request per second the first time. Keeping the cache
makes a run reproducible. The answers of the released models are published in the training-data dataset on the Hub
under `npclassifier/<version>/`, tagged `npclassifier-<version>`; `--npc-cache v1` downloads that snapshot and
copies the answers that are not yet in the cache, so only compounds new since the snapshot reach the API. A
directory of cache files, for example a copy of another machine's `$SANNTIS_CACHE/npclassifier`, works the same
way. Answers already in the cache are never replaced. The API model version, the ontology, the counts and the
seed used are written to `collect/provenance.json`, and the per-compound table to `collect/npc_compounds.tsv`. The
NPC head is trained when at least 50 BGCs are classified.

### NPC pathway key domains

With the NPC head, `tune` also derives the key-domain rules behind `npc_pathway_key_domains_*`; see
[predict](predict.md#npc-pathway-key-domain-support). Candidates are domains and domain pairs. A pair is a
candidate only when it is more specific than both of its domains. Candidates are scored one-vs-rest against the
other labelled BGCs by the 95 % Wilson lower bound of their precision. The recipe block `npc.key_domains` holds
the setting:

| Key | 1.0 value | Meaning |
|---|---|---|
| `tau`, `min_support` | 0.35, 3 | admission: score of at least tau, and present in at least min_support BGCs of the pathway |
| `entry_types` | Family, Domain, Repeat, Conserved_site, Active_site, Binding_site, PTM | candidate tokens are InterPro entries of these types only, taken from the InterPro `entries.json` of the run. Homologous superfamilies, unintegrated signatures and MobiDB are excluded |
| `background_max` | 0.02 | a rule may occur in at most this share of the non-BGC background entities, which are the training entities of the split in `tune` and all of them in `finalize` |
| `min_coverage` | 0.5 | pathways whose rules cover fewer of their true out-of-fold regions are gated and reported as `na` |

Admitted rules are trimmed to a greedy set-cover short list over the BGCs of the pathway. In `tune`, the rules of
each `mc` split are fitted on its training loci and evaluated on its out-of-fold regions with the NPC calls at
cross-split thresholds. This gives the per-pathway coverage, from which the gated pathways follow, and the
false-alarm and catch rates. They are reported in the section "NPC pathway key-domain support" of the report and in
`tune/npc_key_domains_oof.tsv`. `finalize` fits the shipped rules on every labelled locus.

The setting was chosen in a nested-selection cross-validation experiment during development and is fixed in the
recipe. It is not re-tuned per run. The token filters make the rules name domains familiar to a natural-products
scientist, such as ACP and PCP, the PKS ketosynthase, NRPS adenylation, terpene synthases and cyclases, at the same
flagged-to-unflagged error ratio. A GO "catalytic activity" filter was tested and rejected: the KS, ACP and
adenylation domains carry no catalytic GO term, whereas transporters do.

## Steps, resume and runners

The steps run in this order:

```
collect, annotate, entities, cv, tune, finalize, report, package
```

Each step writes `<run>/<step>/` and a `done.json` keyed on its inputs. A re-run skips finished steps. `--from <step>`
forces a step and the ones after it. `--dry-run` prints what would run, including the `nextflow run`
command lines of the fan-out steps.

The fan-out steps are `cv`, with 5 splits, and `finalize`, with 3 seeds. They run either through the bundled
`nf/tasks.nf`, which is the default `--runner nextflow`, or in a local process pool with `--runner inline --workers N`. Both runners execute the same unit code. Resources are requested per **label**, which is the
interface for a site Nextflow configuration:

| Label | Used for | Default request |
|---|---|---|
| `cpu_small` | genome downloads, gene calling (`build-negatives`) | 1 CPU, 4 GB, 4 h |
| `train` | detector fits (CV splits, final seeds) | 8 CPU, 32 GB, 48 h |
| `gpu` | detector fits with `--gpu`, which adds `accelerator 1` | 8 CPU, 32 GB, 48 h |
| `antismash` | antiSMASH 8.0.1 (`build-negatives` only) | 4 CPU, 16 GB, 24 h |

Without `--nf-config`, the requests are capped to the CPUs and the memory of the machine, with the local executor.

```groovy
// SLURM example: one configuration for InterProScan 6 and the training tasks
process {
    executor = 'slurm'
    withLabel: train { queue = 'long'; cpus = 16; memory = '64 GB' }
    withLabel: gpu   { queue = 'gpu'; clusterOptions = '--gres=gpu:1' }
}
```

The run directory and the Python environment must be visible to every job, on a shared filesystem. Object-storage
batch executors are not supported for training. The orchestrator itself should run as a small, long batch job.

## Outputs

| Path | Content |
|---|---|
| `<run>/bundle/` | the production model, with `report/` |
| `<run>/report/report.md` | validation report on the pooled out-of-fold predictions of the `mc` splits: metadata, validation metrics with 95 % CIs, operating-point curve, per-class curves, class and lineage radars, lineage strata, NPC validation |
| `<run>/tune/tune.json` | epochs, t*, NPC thresholds, key-domain CV results under `npc.key_domains`, validation metrics |

A run on MIBiG 4.0 takes about 5 detector CV runs of 10 to 20 min each on 16 CPUs, without a GPU, plus 3 final
seeds of about 10 min each, plus InterProScan 6 for the positives at about 0.5 CPU-hours per 1,000 proteins. The
InterProScan results are cached by protein md5.
