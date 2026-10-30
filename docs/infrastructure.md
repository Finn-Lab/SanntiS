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
