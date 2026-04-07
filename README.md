# pbzarr

A Python library for PBZ (Per-Base Zarr) — a Zarr v3 convention for storing per-base resolution genomic data such as read depths, methylation levels, and boolean masks.

PBZ is a modern alternative to D4 and bigWig, leveraging the Zarr ecosystem for compression, chunking, and cloud-native access.

## Installation

```bash
pip install pbzarr
```

With optional Dask support:

```bash
pip install pbzarr[dask]
```

## Quick Start

```python
import pbzarr

# Create a store
store = pbzarr.create(
    "sample.pbz.zarr",
    contigs=["chr1", "chr2"],
    contig_lengths=[248_956_422, 242_193_529],
)

# Add a track
track = store.create_track("depths", dtype="uint32", columns=["sample_A", "sample_B"])

# Write data
import numpy as np
track["chr1", 0:1000] = np.random.randint(0, 100, size=(1000, 2), dtype="uint32")

# Query data
data = track.query("chr1:0-1000", columns="sample_A")
```

```python
# Open an existing store
store = pbzarr.open("sample.pbz.zarr")
track = store["depths"]

# Slice-based access
data = track["chr1", 0:1000, "sample_A"]

# Dask backend for lazy/parallel computation
store = pbzarr.open("sample.pbz.zarr", backend="dask")
lazy = store["depths"].query("chr1:0-1000000")
result = lazy.compute()
```

## Features

- **Zarr v3 only** with full codec and storage backend support
- **NumPy and Dask backends** for eager or lazy computation
- **Region query syntax**: `"chr1:1000-2000"`, tuples, or slice notation
- **Column filtering** by name or index
- **Escape hatches** to raw `zarr.Group` and `zarr.Array` objects
- **Self-describing tracks** with independent dtype, chunking, and metadata

## Links

- [PBZ Format Specification](https://github.com/pbzarr/pbzarr-spec)
- [Rust Implementation](https://github.com/pbzarr/pbzarr-rs)
