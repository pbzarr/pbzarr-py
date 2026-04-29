# CLAUDE.md — perbase-zarr (PBZ)

## What This Is

PBZ (Per-Base Zarr) is a Python library and Zarr v3 based format for storing per-base resolution genomic data — depths, methylation, masks, etc. A modern alternative to D4 and bigWig. The spec lives in the `pbzarr-spec` repo (`../pbzarr-spec/SPEC.md`). This implementation targets **spec version 0.1**.

PBZ is a **convention and domain layer** on top of zarr-python, not an array library. It defines where things live in a Zarr hierarchy, parses genomic regions, resolves column names, validates queries, and then calls through to zarr-python for all actual I/O.

## Project State

This is early-stage. The structure is:

```
.
├── pyproject.toml
├── pixi.lock
├── SPEC.md
└── src/
    └── perbase_zarr/
        └── __init__.py
```

There is no CLI. Build the library API first.

## Development with Pixi

This project uses [pixi](https://pixi.sh) for environment and task management. All development happens through pixi — do not use `pip install -e .` or raw `pytest` directly.

```bash
# Set up the environment (installs all deps including dev)
pixi install

# Run tests
pixi run test

# Run a specific test
pixi run pytest tests/test_query.py -k "test_parse_region"

# Add a dependency
pixi add numpy          # conda dependency
pixi add --pypi zarr    # PyPI dependency

# Add a dev-only dependency
pixi add --dev pytest
```

Key points:
- `pixi.lock` is auto-generated and committed. Don't edit it by hand.
- The `.pixi/` directory is the local environment — it's in `.gitignore`.
- Use `pixi run <task>` to run tasks, not bare commands. This ensures the correct environment is active.
- When adding new dependencies, prefer conda packages from `conda-forge` for compiled deps (numpy, zarr). Use `--pypi` for pure-Python packages not on conda-forge.

## Architecture: What PBZ Owns vs Delegates

PBZ is deliberately thin. It owns the domain layer and delegates everything else to zarr-python.

**PBZ owns:**
- Store layout convention — `/tracks/{name}/{contig}`, metadata schema, contig/column arrays
- Region parsing — `"chr1:1000-2000"` → contig name + position slice
- Column name → index resolution and caching
- Validation — contig exists, coordinates in bounds, column exists
- Track metadata schema — `pbz_track` attributes, `_ARRAY_DIMENSIONS`
- Missing value semantics — sentinel values per dtype (SPEC §7)
- Backend dispatch — choosing `zarr_array[slices]` vs `da.from_zarr(zarr_array)[slices]`
- `to_xarray()` — dimension labeling that requires PBZ layout knowledge

**Zarr-python handles:**
- All storage backends (local, S3, GCS, memory, zip)
- Array slicing, compression, codecs
- Chunking and sharding
- Async I/O and concurrency
- Resize, append
- Pickle support
- Advanced indexing (coordinate, mask, orthogonal, block)

**The rule:** if zarr-python already does it, don't reimplement it. Expose the raw zarr objects via escape hatches (SPEC §11) so users can access any zarr feature directly.

## Key Design Decisions

- **Zarr v3 only.** No v2 compatibility (SPEC §14.1).
- **Lean on zarr-python.** PBZ is a convention layer — it constructs the right slice and calls through to zarr. It does not wrap `zarr.Array` in custom classes that re-expose `.shape`, `.dtype`, `.chunks`, etc.
- **0-based, half-open coordinates** everywhere (SPEC §6). Conversion from 1-based formats happens at the boundary via an explicit `one_based=True` flag (SPEC §8.4) — never guess.
- **2D chunking** for columnar tracks: 1M bp position chunks, 16 column chunks (SPEC §4). Cross-column region queries are the common case.
- **Sharding is optional** (SPEC §4.3). Default is no sharding. Use sharding for object stores or large datasets with many files.
- **Sentinel-based missing data** for integer types (dtype max value), NaN for floats (SPEC §7). No separate mask arrays.
- **Tracks are self-describing.** Each track carries its own dtype, chunk config, and column list independently (SPEC §3.1).
- **NumPy by default, dask opt-in.** Dask is an optional dependency. The `backend=` flag is thin dispatch — same region/column resolution, different last-mile call (SPEC §10).
- **Escape hatches always available.** `store.root` gives the raw `zarr.Group`. `track.zarr_array(contig)` gives the raw `zarr.Array`. Users can drop down to zarr at any point (SPEC §11).
- **Storage, compression, and sharding are passthrough.** PBZ passes store paths, codec objects, and shard config directly to zarr-python. No PBZ-specific abstractions for these.
- **Backend is extensible.** v0.1 supports `"numpy"` and `"dask"`. The design allows adding `"cupy"`, `"jax"`, etc. in the future — each backend is just a different last-mile call after region/column resolution (SPEC §10.5). GPU-native I/O will come through the zarr store layer (kvikio/GDS), not the backend layer.
- **Don't hardcode numpy on the return path.** No `isinstance(result, np.ndarray)` checks after `query()`. This keeps the door open for future backends. The [Python Array API Standard](https://data-apis.org/array-api/latest/) is the direction for backend-agnostic downstream code — PBZ doesn't need to adopt it internally, but shouldn't do anything that prevents it.
- **File extension:** `.pbz.zarr` (SPEC §1.2).

## Backend Model

The `backend` is chosen at open time and inherited by all tracks from the store (SPEC §10):

```python
# Default — numpy, eager
store = pbz.open("data.pbz.zarr")
data = store["depths"].query("chr1:1000-2000")  # numpy ndarray

# Dask — lazy, chunked
store = pbz.open("data.pbz.zarr", backend="dask")
data = store["depths"].query("chr1:1000-2000")  # dask.array.Array

# Xarray — convenience method, always dask-backed
ds = store["depths"].to_xarray()  # xr.DataArray
```

The backend is **thin dispatch**. The implementation of `query()` resolves the region and columns identically regardless of backend. The only difference is the final call:

```python
# numpy backend
return zarr_array[start:end, col_slice]

# dask backend
return da.from_zarr(zarr_array)[start:end, col_slice]
```

Dask and xarray are **optional dependencies**. Use lazy imports. If someone uses `backend="dask"` without dask installed, raise a clear `ImportError` pointing to `pip install perbase-zarr[dask]`.

## Dependencies

Core (required): `zarr>=3.0`, `numpy`

Optional:
```toml
[project.optional-dependencies]
dask = ["dask[array]"]
xarray = ["xarray", "dask[array]"]
all = ["dask[array]", "xarray"]
```

## Genomic Query Syntax

PBZ supports a unified region query interface (SPEC §8–9). Summary:

```python
# String syntax
track.query("chr1:1000-2000")

# Tuple syntax
track.query("chr1", 1000, 2000)

# Slice syntax
track["chr1", 1000:2000, :]

# Column filtering
track.query("chr1:1000-2000", columns=["sample_A"])
track["chr1", 1000:2000, "sample_A"]
```

Region strings: `chr1` (whole contig), `chr1:1000-2000` (range), `chr1:1000` (single base). Commas in numbers are stripped. All 0-based half-open unless `one_based=True`.

## Coding Conventions

- Python 3.10+. Use `X | Y` union syntax, not `Union[X, Y]`.
- Type hints on all public functions. Use `numpy.typing.NDArray` for array params.
- `pathlib.Path` for all paths. Accept `str | Path`, convert to `Path` internally.
- Keep everything in `src/perbase_zarr/` — split into submodules as complexity warrants, but don't prematurely create a deep package tree.
- Lazy imports for optional deps (dask, xarray). Pattern:
  ```python
  def _import_dask():
      try:
          import dask.array as da
          return da
      except ImportError:
          raise ImportError(
              "dask is required for this feature. "
              "Install with: pip install perbase-zarr[dask]"
          ) from None
  ```
- Tests with **pytest**. Use `tmp_path` for write tests.
- **Comments should explain *why*, never *what*.** Do not add block-style section dividers (`# -----`). Do not annotate code with SPEC section numbers — the code should be self-explanatory and the SPEC is separate documentation. If a comment restates what the next line of code does, delete it.

## Exception Hierarchy

```
PbzError
├── ContigNotFoundError
├── TrackNotFoundError
├── InvalidRegionError
└── ColumnNotFoundError
```

## Performance Notes

- `query()` resolves to a single zarr or dask call. Don't load extra data and filter in Python.
- Cache column name → index mapping per track instance.
- Write aligned to chunk boundaries when possible to avoid read-modify-write in Zarr.
- PBZ chunk sizes directly determine dask task graph structure (SPEC §10.4). The default 1M × 16 chunking gives good parallelism for typical genomic queries.
- Default compression: Blosc(zstd, level=5, byte shuffle). Don't change without benchmarking.
- Consider sharding for object stores or datasets with many tracks/contigs to reduce file count (SPEC §4.3).

## Known Issue: Metadata Key Naming

The Python code currently uses `"pbz"` and `"pbz_track"` as metadata attribute keys. The canonical spec (in `pbzarr-spec`) uses `"perbase_zarr"` and `"perbase_zarr_track"`. The Rust implementation already uses the correct names.

This needs to be fixed: rename `"pbz"` → `"perbase_zarr"` and `"pbz_track"` → `"perbase_zarr_track"` in the Python code and tests. Until this is done, files written by Python cannot be read by Rust and vice versa.

## What Not to Do

- Don't reimplement zarr functionality. If zarr-python does it, call through to it.
- Don't wrap `zarr.Array` in custom classes that re-expose `.shape`, `.dtype`, `.chunks`. Provide escape hatches instead.
- Don't build a PBZ-specific storage or compression abstraction. Pass through to zarr.
- Don't add Zarr v2 support.
- Don't add a CLI yet — library API first.
- Don't make dask or xarray a core dependency. Always lazy-import.
- Don't store data in any layout other than `(position, column)`.
- Don't add index files alongside the Zarr store — if we need indexing, it goes inside the Zarr hierarchy.
- Don't handle BAM/CRAM reading — PBZ stores pre-computed per-base values. Reading raw alignments is the caller's job.
- Don't do `isinstance(result, np.ndarray)` checks on `query()` return values. The return path must stay backend-agnostic so future backends (cupy, jax) can slot in.
- Don't try to adopt the Python Array API Standard internally for v0.1. PBZ produces arrays from zarr — the array API standard helps *consumers* of PBZ output, not PBZ itself. Just don't do anything that prevents future adoption.