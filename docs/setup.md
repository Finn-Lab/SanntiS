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
PROSITE profiles, PROSITE patterns, MobiDB-lite and SMART. Matches are computed locally with `--noMatchesApi`.
An `--nf-args` value that would change any of these is refused. The online InterPro compatibility check is skipped
with `--skipInterproVersionCheck`, as the pinned versions are compatible.
