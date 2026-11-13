# InterProScan setup

```bash
sanntis setup --container-engine docker [--nf-config site.config] [--datadir DIR]
```

`setup` downloads the InterProScan 6 pipeline `ebi-pf-team/interproscan6` at revision 6.0.3.0 and its data for
InterPro 107.0 into `$SANNTIS_CACHE/ips6`, or into `--datadir`. It then runs the pipeline twice on one bundled
protein: first without PROSITE patterns, then with all 14 member databases. The two runs work around a problem of
InterProScan 6.0.3.0 on a fresh data directory, where the PROSITE patterns scan can start before the PROSITE data
shared with PROSITE profiles have finished downloading.

`sanntis predict` runs the same setup when the data are missing, so `setup` is optional. It is useful to download
the data once before a series of runs.

The InterProScan version, the InterPro release and the member databases are fixed: InterProScan 6.0.3.0, InterPro
107.0 and the 14 applications Pfam, NCBIfam, SFLD, SUPERFAMILY, CATH-Gene3D, HAMAP, COILS, CDD, PRINTS, PIRSF,
PROSITE profiles, PROSITE patterns, MobiDB-lite and SMART. An `--nf-args` value that would change any of these
(or the lookup and `--cpus` settings below) is refused. The online InterPro compatibility check is skipped
with `--skipInterproVersionCheck`, as the pinned versions are compatible.

## Precalculated matches

InterProScan 6 can fetch the matches of proteins InterPro has already annotated from the Matches API instead of
computing them. The API serves the current InterPro release, and InterProScan 6.0.3.0 does not compare that release
with the one it runs: proteins found in the API would get matches of another release than the proteins computed
locally. SanntiS therefore reads the API's `/info` before each annotation and uses the lookup only when the API
serves the pinned release (`SANNTIS_IPS6_LOOKUP=auto`, the default); otherwise every protein is computed locally
(`--noMatchesApi`). `SANNTIS_IPS6_LOOKUP=off` never uses it, and `SANNTIS_IPS6_MATCHES_API` points at another
Matches API, e.g. a local service for the pinned release. The decision is written to `lookup.json` in the
InterProScan run directory and, for negative pools, to the pool manifest.

