# SanntiS Critical Memory Fixes

## Problem

When processing large metagenomic assemblies (~1M contigs) with a preprocessed
InterProScan file, `BGCdetection.py` accumulated all per-contig annotation matrices
simultaneously in `self.annDct` before any inference ran. With a vocab of ~5,000
terms and an average of 5 CDSs per contig, this alone could reach tens of gigabytes
even after switching to `uint8`.

The root cause was a three-pass design: `buildMatrices()` → `predictAnn()` →
`defineLooseClusters()` + `predictType()`, each sweeping the full set of contigs
separately, forcing all intermediate data to remain in memory across all passes.

## Changes

### `sanntis/modules/BGCdetection.py`

**Removed `self.annDct` from `__init__`**

The dict no longer exists. Matrices are never stored on the object.

**Replaced `buildMatrices()` with `_iter_matrices()` generator**

```python
def _iter_matrices(self):
    for contig, cdss in self.contigsDct.items():
        samps = ...
        mat = np.zeros((samps, len(self.vocab)), dtype=np.uint8)
        for ix, (name, _) in enumerate(cdss):
            if name in self.entriesDct:
                modiAnn = [self.vocab[x] for x in self.entriesDct[name] if x in self.vocab]
                mat[ix][modiAnn] = 1
        yield contig, mat
    self.entriesDct.clear()
```

- `mat` is a plain local variable; Python frees it naturally when the generator
  advances, with no manual `del` or dict mutation required
- `entriesDct` is cleared in one call after the generator is exhausted, not
  incrementally inside the loop
- dtype changed from `float64` to `uint8` (8× reduction; values are always 0 or 1)

**Replaced `predictAnn()` + `defineLooseClusters()` + `predictType()` with `predictAndClassify()`**

Single per-contig loop consuming `_iter_matrices()`:

```
for contig, mat in self._iter_matrices():
    TF predict → define clusters → sklearn classify
    # mat goes out of scope here; next iteration begins
```

All three removed methods are gone. The TF model and score thresholds are
initialised once before the loop.

### `sanntis/_cli.py`

- Removed `annotate.buildMatrices()` call
- Replaced three separate calls (`predictAnn`, `defineLooseClusters`, `predictType`)
  with one: `annotate.predictAndClassify(score=args.score, g=args.greed)`

## Memory profile after fixes

| Data structure | Before | After |
|---|---|---|
| `annDct` | All contig matrices in RAM simultaneously (dominant cost) | Does not exist |
| Matrix dtype | `float64` | `uint8` (8× smaller) |
| `entriesDct` | Cleared after `buildMatrices` | Cleared after generator exhausted |
| Peak matrix memory | O(all contigs × CDSs × vocab) | O(largest single contig × vocab) |

## What remains in memory (intentionally)

- `contigsDct`: contig → protein coordinates, needed by `WriteOutput`
- `annResults`, `looseClst`, `borderClst`, `typesClst`: 1D arrays per contig,
  needed by `WriteOutput` — unavoidable until output writing is also streamed

## Next steps

- Stream IPS file line by line in `transformIPS()` instead of `readlines()`
- Batch contigs for TF inference to amortise per-call overhead (currently one
  `model.predict()` call per contig)
