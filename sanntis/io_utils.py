import gzip


def open_flat_file(path, mode="rt"):
    """Open a plain or gzipped flat file with the requested mode."""
    if str(path).lower().endswith(".gz"):
        return gzip.open(path, mode)
    return open(path, mode)
