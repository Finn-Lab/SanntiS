import gzip
import os
import shutil


def open_flat_file(path, mode="rt"):
    """Open a plain or gzipped flat file with the requested mode."""
    if str(path).lower().endswith(".gz"):
        return gzip.open(path, mode)
    return open(path, mode)


def materialize_uncompressed(path, dest_dir):
    """Return a path to an uncompressed copy of ``path``.

    If ``path`` has a ``.gz`` suffix it is decompressed into ``dest_dir`` under
    the same basename with the suffix stripped, and that new path is returned.
    Non-gzipped paths are returned unchanged. Used to feed external tools that
    do not read gzip (Prodigal, hmmscan, InterProScan).
    """
    if not str(path).lower().endswith(".gz"):
        return path
    base = os.path.basename(str(path))[:-3]
    out = os.path.join(dest_dir, base)
    with gzip.open(path, "rb") as src, open(out, "wb") as dst:
        shutil.copyfileobj(src, dst)
    return out
