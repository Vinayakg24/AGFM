# CHANGES.md — Modifications from Original AGFM

This file documents all changes made to the original [AGFM pipeline](https://github.com/ayush12gupta/AGFM) by **Vinayak Gupta** as part of M.Tech thesis work at IIT Bombay under Prof. Biplab Banerjee.

The original pipeline is described in:
> Gupta, A., Devaraju, B., Tiwari, A., 2025. Processing pipeline for fully-automated computation of 3D glacier surface flow time series. *Computers & Geosciences* 200, 105918. [doi: 10.1016/j.cageo.2025.105918](https://doi.org/10.1016/j.cageo.2025.105918)

---

## Overview of Changes

Changes fall into two categories:
1. **Scientific extension** — NCC-based uncertainty weighting in the 3D velocity inversion
2. **Bug fixes** — Corrections to offset tracking, interpolation, and DEM download

---

## 1. NCC Uncertainty-Weighted 3D Inversion (`batch3Dinversion.py`, `stack_pipeline.py`)

### Motivation
The original pipeline performs 3D velocity inversion using Helmert Variance Component Estimation (VCE) with equal initial weights, estimating relative ascending/descending pass reliability from residuals. This discards available a priori information about per-pixel measurement quality.

The modified pipeline incorporates pre-computed **De Zan NCC uncertainty maps** (σ_NCC, in m yr⁻¹) to weight each pixel's range and azimuth observations individually before inversion.

### New Functions in `batch3Dinversion.py`

#### `_resolve_sigma_dir(sigma_base, track_path)`
Automatically infers the `{year}_{asc|des}/` subdirectory from the SAR data path using regex year extraction. Handles directory naming conventions like `Chandra_baga_2017_19`.

#### `read_sigma(sigma_dir, stack_id)`
Reads σ_NCC GeoTIFF files (produced by `compute_ncc_uncertainty_v2.py`) for a given epoch.
- Band 1: σ_az (azimuth uncertainty, m yr⁻¹)
- Band 2: σ_rg (range uncertainty, m yr⁻¹)
- Returns `(None, None)` gracefully if file is absent.

#### `sigma_to_weight(sig, floor=0.01)`
Converts σ maps to weight maps: `w = 1 / max(σ, floor)²`  
A minimum floor of 0.01 m yr⁻¹ prevents infinite weights for near-zero σ values.

### Modified: `compute_3DVel_single_hermet` — Dual-Mode Solver

The inversion now operates in two modes depending on data availability:

**Mode 1 — Direct WLS with De Zan weights** *(new, used when `sigma_base` is supplied)*
```
N = Aᵀ P A,   u = Aᵀ P b,   v̂ = N⁻¹ u
```
where `P = diag(1/σ_rg_asc², 1/σ_az_asc², 1/σ_rg_des², 1/σ_az_des²)`

Single solve, no iteration. Rationale: since De Zan σ_NCC already provides all four a priori variances, estimating them iteratively from residuals (VCE) would discard prior information and wash out the range/azimuth weight distinction.

**Mode 2 — Helmert VCE** *(original behaviour, used when `sigma_base` is not supplied)*  
Unchanged logic — iterative ascending/descending group reweighting. Preserved for full backward compatibility.

Both modes now return theoretical uncertainty: `std_nev = sqrt(diag(N⁻¹))`.

### New Output: Uncertainty GeoTIFFs (`_unc.tif`)

For every velocity output `{pair}.tif`, a companion `{pair}_unc.tif` is now saved containing the theoretical 1-σ uncertainty for each of the 3 velocity components (N, E, U). This product is absent in the original pipeline.

### Modified: `stack_pipeline.py` — New CLI Argument

```bash
python stack_pipeline.py -t_asc ASC_TXT -t_des DES_TXT --config CONFIG [--sigma_base SIGMA_DIR]
```

| Argument | Description |
|---|---|
| `--sigma_base` | Base directory of pre-computed De Zan σ_NCC maps. If omitted, equal-weight (original) mode is used. |

---

## 2. Bug Fix — Offset Tracking Array Bounds (`glacier_stacking_autoRIFT.py`)

**Function:** `compute_dx`  
**Issue:** Upscaling window `res[maxLoc[1]-3:maxLoc[1]+4, maxLoc[0]-3:maxLoc[0]+4]` could go out of bounds when the NCC peak (`maxLoc`) is within 3 pixels of the correlation surface edge, causing a crash or silently returning an empty array.

```python
# Before (crashes near edges):
res_up = upscale(res[maxLoc[1]-3:maxLoc[1]+4, maxLoc[0]-3:maxLoc[0]+4], scale, type=interp)

# After (clamped to array bounds):
y1 = max(0, maxLoc[1]-3);  y2 = min(res.shape[0], maxLoc[1]+4)
x1 = max(0, maxLoc[0]-3);  x2 = min(res.shape[1], maxLoc[0]+4)
res_up = upscale(res[y1:y2, x1:x2], scale, type=interp)
```

---

## 3. Bug Fix — Chip Scale Aspect Ratio (`testautoRIFT_ISCE.py`)

**Function:** `runAutorift`  
**Issue:** `ScaleChipSizeY` was hardcoded to `1`, implicitly assuming square pixels in radar geometry. Sentinel-1 IW mode has non-square pixels (range ≠ azimuth pixel spacing), so chips were being stretched incorrectly on the ground.

```python
# Before:
obj.ScaleChipSizeY = 1

# After:
obj.ScaleChipSizeY = pixsizex / pixsizey  # Correct proportion for square chips on ground
```

---

## 4. Bug Fix — BSpline Axis Order (`offset_tracking/util.py`)

**Function:** `BSpline`  
**Issue:** `scipy.interpolate.RectBivariateSpline` expects `(row_coords, col_coords, data)` but the original code passed `(x, y, z)` where `x` was column indices and `y` was row indices — a transposed axis order. This caused the interpolated output to be transposed relative to the input, introducing a directional error in upscaled NCC surfaces.

```python
# Before (axes swapped):
spline = RectBivariateSpline(x, y, z)   # x=cols, y=rows — wrong
z_interp = spline(new_x, new_y)

# After (correct):
spline = RectBivariateSpline(y, x, z)   # y=rows, x=cols — correct
z_interp = spline(new_y, new_x)
```

---

## 5. Bug Fix — DEM Download Robustness (`preprocessing/setup_env.py`)

Three issues corrected:

**a) Wrong `odc.stac.load` usage**  
`odc.stac.load` returns an `xarray.Dataset`, not a `DataArray`. Calling `.squeeze()` directly on a Dataset returned a Dataset, causing `.rio.to_raster()` to fail silently or with an obscure error.
```python
# Before:
data = odc.stac.load(items, bands=["data"], bbox=bound).squeeze()

# After:
ds = odc.stac.load(items, bands=["data"], bbox=bound)
data = ds["data"].squeeze()   # Extract DataArray first
```

**b) Silent failure on download error**  
The original `download_COP30_DEM` returned `None` on failure, which downstream code did not check, causing a silent `gdal.Open(None)` call.
```python
# Before: return None on failure
# After:  raise — so callers detect failure
```

**c) Missing null-check for `gdal.Open`**  
Added explicit `FileNotFoundError` if `gdal.Open` returns `None`, with a clear diagnostic message.

**d) Added `import rioxarray`**  
Required for the `.rio` accessor on `xarray.DataArray` objects. Was missing, causing `AttributeError` at runtime.

**e) Removed dead import**  
`from turtle import down` was an erroneous leftover import.

---

## Files Changed

| File | Type |
|------|------|
| `batch3Dinversion.py` | Scientific extension + improvement |
| `stack_pipeline.py` | New CLI argument |
| `offset_tracking/glacier_stacking_autoRIFT.py` | Bug fix |
| `offset_tracking/testautoRIFT_ISCE.py` | Bug fix |
| `offset_tracking/util.py` | Bug fix |
| `preprocessing/setup_env.py` | Bug fix + robustness |
| `.gitignore` | Added credential/config protection |
| `configs/` | Paths reset to generic placeholders |

## Files Unchanged from Original

`utils.py`, `stack_process.py`, `README.md`, `requirements.txt`, `LICENSE`, `stack/topsStack/` (all files), `offset_tracking/glacier_autoRIFT_ISCE.py`, `offset_tracking/netcdf_output.py`, `notebook/Pre-processing.ipynb`, and all `data/` and `docs/` files are identical to the original repository.

---

## Contact

For questions about these modifications, contact Vinayak Gupta (`vinayakg24@iitb.ac.in`).  
For questions about the original pipeline, contact Ayush Gupta (`ayushg12@uw.edu`).
