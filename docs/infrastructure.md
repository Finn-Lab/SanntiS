# Running on other infrastructure

!!! warning "Unsupported"
    sanntis is tested on a local workstation with Docker and on one SLURM cluster with Singularity. Other setups
    are configured through standard Nextflow configuration. The examples below are not tested elsewhere.

sanntis sets one Nextflow option itself: `--container-engine` becomes the `-profile` of InterProScan 6. Every
other option is forwarded unchanged to each `nextflow run` it launches:

| Option | Forwarded as |
|---|---|
| `--nf-config FILE` (repeatable) | `-c FILE` |
| `--nf-profile NAME` (repeatable) | appended to `-profile`; the profile must be defined in the configuration |
| `--nf-args "..."` | appended verbatim, for example `-with-trace` |

`sanntis predict` launches one top-level `nextflow run` for InterProScan 6 from the calling process. On a cluster,
run `sanntis` itself in a small batch job rather than on a login node. The Nextflow work directory is
`<outdir>/.sanntis/ips6_run/ips6/work`. It is removed after a successful run unless `--keep-work` is given. Both
`--outdir` and `$SANNTIS_CACHE` must be on a filesystem shared with the compute nodes.

## Example: SLURM + Singularity

```groovy
// slurm.config
executor { name = 'slurm'; queueSize = 100 }
singularity { enabled = true; autoMounts = true; cacheDir = '/shared/singularity-cache' }
process {
    executor = 'slurm'
    queue    = 'standard'
    withName: 'DOWNLOAD_INTERPRO|DOWNLOAD_DATABASE' { time = '12h' }  // first run only
}
```

```bash
sbatch -c 2 --mem 8G --wrap "sanntis predict genome.fna --container-engine singularity --nf-config slurm.config -o out/"
```

Institutional nf-core configurations are passed the same way, with `--nf-config` and `--nf-profile`.

## Large InterProScan 6 runs

Large protein sets (training positives, negative pools, test sets) are annotated in chunks of
`SANNTIS_IPS6_CHUNK` proteins, each its own `nextflow run` with its own work directory. Three settings size the
load on the scheduler:

- `SANNTIS_IPS6_CPUS` (default 4): InterProScan 6 `--cpus`, the cores of each multi-threaded task (HMMER searches,
  Gene3D, SFLD, SUPERFAMILY, SMART, PIRSF, PROSITE profiles, MobiDB-lite); single-threaded tasks take 1;
- `SANNTIS_IPS6_PARALLEL`: chunks run at the same time (one Nextflow head each, in the calling job);
- `SANNTIS_IPS6_QUEUE_SIZE` / `SANNTIS_IPS6_SUBMIT_RATE`: `executor.queueSize` / `submitRateLimit` of each
  InterProScan 6 run, applied after `--nf-config` so they do not change the queue of the other fan-out steps.

At most `PARALLEL` x `QUEUE_SIZE` tasks are queued at once, so about `PARALLEL` x `QUEUE_SIZE` x `CPUS` cores are
busy. For example, about 2,500 cores:

```bash
export SANNTIS_IPS6_PARALLEL=16 SANNTIS_IPS6_QUEUE_SIZE=40 SANNTIS_IPS6_SUBMIT_RATE='100/1min'
export NXF_OPTS="-Xms256m -Xmx2g"   # per Nextflow head: size the calling job for PARALLEL heads
```
