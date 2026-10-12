from osgeo import gdal            # GDAL support for reading virtual files
import os, shutil, time                      # To create and remove directories
import multiprocessing as mp
import numpy as np
from glob import glob
import scipy.linalg as la
from tqdm import tqdm
from scipy.linalg import lstsq as sp_lstsq
from offset_tracking.util import numpy_array_to_raster, get_incang, get_flightang, get_matA_3D


# ─────────────────────────────────────────────────────────────────────────────
# Uncertainty directory layout (produced by compute_ncc_uncertainty_v2.py)
# ─────────────────────────────────────────────────────────────────────────────
#   {sigma_base}/{year}_{track}/  e.g.  2017_asc/  2019_des/
#       {epoch_id}_{master}_{slave}_sigma_NCC.tif
#           Band 1  = σ_NCC_az  [m yr⁻¹]
#           Band 2  = σ_NCC_rg  [m yr⁻¹]
#
#   The epoch_id prefix (e.g. "stack_0_1") is extracted from the velocity path;
#   year and track are inferred from the asc/des directory year folder name.
#   Pixel weights entering the normal equations:
#       w_rg = 1 / σ_rg²,   w_az = 1 / σ_az²
#   The weight matrix P is diagonal:  diag(w_az_asc, w_rg_asc, w_az_des, w_rg_des)
#   (observation ordering matches A-matrix rows: rg_asc, az_asc, rg_des, az_des)
# ─────────────────────────────────────────────────────────────────────────────

# Fallback weight used when σ file is missing or a pixel has NaN σ.
# Setting w = 1.0 reproduces the original equal-weight behaviour.
_FALLBACK_WEIGHT = 1.0

# Minimum variance floor: prevents infinite weights for near-zero σ values.
# Any σ below this threshold is clamped to this value before computing 1/σ².
_SIGMA_FLOOR_M_YR = 0.01   # 1 cm yr⁻¹  (physically safe lower bound)


def read_velocity(path):
    ds = gdal.Open(path)
    V_R_asc = ds.GetRasterBand(1).ReadAsArray()
    V_A_asc = ds.GetRasterBand(2).ReadAsArray()
    projs = ds.GetProjection()
    geo = ds.GetGeoTransform()
    ds = None
    return V_R_asc, V_A_asc, projs, geo


def save_raster_outputs(save_dir, V_nev, projs, geo):
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.mkdir(save_dir)
    for i in range(V_nev.shape[0]):
        ds = numpy_array_to_raster(f'{save_dir}/stack_{i}_{i+1}_corrected.tif', V_nev[i], projs, geo, nband=3)
        ds.FlushCache()
        ds = None


# ─────────────────────────────────────────────────────────────────────────────
# Sigma I/O helpers
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_sigma_dir(sigma_base, track_path):
    """
    Infer the year_track subdirectory from the data path.

    The sigma_base layout mirrors the UNCERTAINTY/Final directory:
        {sigma_base}/{year}_{asc|des}/

    We search every path component for an embedded 4-digit year
    (e.g. "Chandra_baga_2017_19" → year "2017") and combine it with the
    track name extracted from the last component ("stack_asc" → "asc").

    Example:
        track_path  = "/DATA/Chandra_baga_2019_20/run.../stack_asc/"
        → year_str  = "2019"  (from "Chandra_baga_2019_20")
        → track_str = "asc"
        → returns   "{sigma_base}/2019_asc"
    """
    import re

    # Normalise: remove trailing slash
    path_clean = track_path.rstrip('/')

    # Extract track name ("asc" or "des")
    basename = os.path.basename(path_clean)          # e.g. "stack_asc"
    track_str = basename.replace('stack_', '')        # e.g. "asc"

    # Extract year: scan components for an embedded 4-digit year (1900-2099).
    # Use digit-boundary lookarounds (not \b) because _ is a word character
    # so \b fails on names like "Chandra_baga_2017_19".
    year_str = None
    for part in path_clean.split('/'):
        m = re.search(r'(?<!\d)((?:19|20)\d{2})(?!\d)', part)
        if m:
            year_str = m.group(1)
            break

    if year_str is None:
        return None  # Cannot resolve; caller will use fallback weights

    sigma_dir = os.path.join(sigma_base, f'{year_str}_{track_str}')
    return sigma_dir if os.path.isdir(sigma_dir) else None



def read_sigma(sigma_dir, stack_id):
    """
    Load σ_az and σ_rg maps for one epoch.

    File naming convention (from compute_ncc_uncertainty_v2.py):
        {epoch_id}_{master}_{slave}_sigma_NCC.tif
    where epoch_id = stack_id  (e.g. "stack_0_1").

    The file is found by globbing  {sigma_dir}/{stack_id}_*_sigma_NCC.tif
    so that it matches regardless of the exact master/slave date stamp.

    Returns
    -------
    sig_az : 2D float32 array, shape (rows, cols)  [m yr⁻¹]   or None
    sig_rg : 2D float32 array, shape (rows, cols)  [m yr⁻¹]   or None
    Returns (None, None) if the file is absent or unreadable.
    """
    if sigma_dir is None:
        return None, None

    pattern = os.path.join(sigma_dir, f'{stack_id}_*_sigma_NCC.tif')
    matches = glob(pattern)
    if not matches:
        return None, None

    sigma_path = matches[0]
    try:
        ds = gdal.Open(sigma_path)
        if ds is None:
            return None, None
        sig_az = ds.GetRasterBand(1).ReadAsArray().astype(np.float32)  # Band 1 = σ_az
        sig_rg = ds.GetRasterBand(2).ReadAsArray().astype(np.float32)  # Band 2 = σ_rg
        nodata = ds.GetRasterBand(1).GetNoDataValue()
        ds = None
        if nodata is not None:
            sig_az[sig_az == nodata] = np.nan
            sig_rg[sig_rg == nodata] = np.nan
        return sig_az, sig_rg
    except Exception:
        return None, None


def sigma_to_weight(sig, floor=_SIGMA_FLOOR_M_YR):
    """
    Convert a sigma map [m yr⁻¹] to a weight map  w = 1 / max(σ, floor)².

    NaN pixels (invalid σ) receive the fallback weight so that no information
    is silently lost — they will be down-weighted later by the IRLS step if
    residuals are large.
    """
    w = np.where(np.isfinite(sig), 1.0 / np.maximum(sig, floor) ** 2, _FALLBACK_WEIGHT)
    return w.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# IRLS / Helmert variance-component re-weighting
# ─────────────────────────────────────────────────────────────────────────────

def initP(P, resi, sigma=None):
    ab_resi = np.abs(resi)
    sigma = np.median(np.abs(resi)) + 2*np.abs(resi).std()
    med = np.median(np.abs(resi))
    std = max(np.abs(resi).std(), 3)
    # print(resi, sigma, med, std)
    if sigma<5:
        return np.diag(P)
        
    P2 = P.copy()
    for i in range(P.shape[0]):
        if (max(med-std, 0)<ab_resi[i])&((med+std)>ab_resi[i]):
            P2[i] = P[i]
        elif (max(med-3*std, 0)<ab_resi[i])&((med+3*std)>ab_resi[i]):
            # P[i] = 1/((2*sigma/np.abs(resi[i])))*P[i]
            if max(med-std, 0)>=ab_resi[i]:
                cnt = max(med-std, 0)-ab_resi[i]
            elif med+std<=ab_resi[i]:
                cnt = ab_resi[i] - (med+std)
                
            P2[i] = ((med/(med+cnt))**0.3)*P[i]
            # P2[i] = (2*np.abs(resi[i])/std)*P[i]
        else:
            P2[i] = 0

    return np.diag(P2)


# ─────────────────────────────────────────────────────────────────────────────
# Core single-pixel 3-D inversion
# ─────────────────────────────────────────────────────────────────────────────

def compute_3DVel_single_hermet(V_aoro, A, w_pixel=None):
    """
    Solve the 3-D velocity inversion for a single pixel.

    Two modes depending on whether a priori De Zan weights are available:

    ── Mode 1: De Zan weights known  (w_pixel supplied) ────────────────────
        Direct Weighted Least Squares (WLS) in a single solve:

            N   = A^T P A
            u   = A^T P b
            v̂  = N⁻¹ u

        where  P = diag(1/σ_rg_asc², 1/σ_az_asc², 1/σ_rg_des², 1/σ_az_des²)

        No iteration, no grouping, no VCE.
        Rationale: Helmert VCE is used to ESTIMATE unknown variance components
        from the data.  De Zan σ_NCC already provides all four a priori
        variances, so estimating them iteratively from residuals would
        discard prior information and wash out the range/azimuth weight
        distinction toward unity.

    ── Mode 2: No weights  (w_pixel is None) ───────────────────────────────
        Original Helmert VCE between ascending and descending pass groups
        (split by pass: A[:2] / A[2:]).  Used when sigma_base is not
        supplied — preserves backward-compatible behaviour.

    Parameters
    ----------
    V_aoro   : 1D array, shape (4,)
               [v_rg_asc, v_az_asc, v_rg_des, v_az_des]  [m yr⁻¹]
    A        : 2D array, shape (4, 3)
               Design matrix (same row ordering as V_aoro).
    w_pixel  : 1D array, shape (4,), optional
               De Zan weights [1/σ²]:
               [w_rg_asc, w_az_asc, w_rg_des, w_az_des]
               If None, Helmert VCE with equal initial weights is used.

    Returns
    -------
    v_nev   : 1D array, shape (3,)   [N, E, U]  [m yr⁻¹]
    std_nev : 1D array, shape (3,)   [σ_N, σ_E, σ_U] [m yr⁻¹] theoretical uncertainty
    res     : 1D array, shape (4,)   post-fit residuals
    """
    if w_pixel is not None:
        # ── MODE 1: Direct WLS with De Zan weights ─────────────────────────
        P    = np.diag(w_pixel)                          # 4×4 diagonal
        N    = A.T @ P @ A                               # 3×3 normal matrix
        u    = A.T @ P @ V_aoro                          # 3×1 right-hand side
        v_nev, _, _, _ = sp_lstsq(N, u, lapack_driver='gelsy', check_finite=False)
        # ── Compute theoretical uncertainty bounds ─────────────────────────
        try:
            cov = np.linalg.inv(N)
            std_nev = np.sqrt(np.diag(cov))
        except np.linalg.LinAlgError:
            std_nev = np.full(3, np.nan)

        res  = V_aoro - A @ v_nev
        return v_nev.reshape(3), std_nev.reshape(3), res

    # ── MODE 2: Helmert VCE — equal initial weights, no a priori info ──────
    # Split by PASS so VCE estimates the between-pass scale factor.
    A1, A2 = A[:2], A[2:]
    b1, b2 = V_aoro[:2], V_aoro[2:]
    P1, P2 = np.eye(2), np.eye(2)

    for _ in range(3):
        N_mat = (A1.T @ P1 @ A1) + (A2.T @ P2 @ A2)
        U_vec = (A1.T @ P1 @ b1) + (A2.T @ P2 @ b2)
        v_nev, _, _, _ = sp_lstsq(N_mat, U_vec, lapack_driver='gelsy',
                                   check_finite=False)
        resi1  = b1 - A1 @ v_nev
        resi2  = b2 - A2 @ v_nev
        sigma01 = (resi1.T @ P1 @ resi1) / 2
        sigma02 = (resi2.T @ P2 @ resi2) / 2
        if sigma02 == 0:
            break
        P2 = (sigma01 / sigma02) * P2
        if np.abs(sigma01 - sigma02) < 1:
            break

    try:
        cov = np.linalg.inv(N_mat)
        std_nev = np.sqrt(np.diag(cov))
    except np.linalg.LinAlgError:
        std_nev = np.full(3, np.nan)

    res = V_aoro - A @ v_nev
    return v_nev.reshape(3), std_nev.reshape(3), res


# ─────────────────────────────────────────────────────────────────────────────
# Parallel column worker
# ─────────────────────────────────────────────────────────────────────────────

def unpacking_column(tup):
    """
    Process one column of pixels.

    tup = (jj, in_shape, A, v_aoro, W_col)
        W_col : 2D array, shape (4, rows)  per-pixel weight vectors,
                or None for equal weights.
    """
    jj, in_shape, A, v_aoro, W_col = tup
    
    v_nev = np.empty(in_shape[:-1],dtype=np.float32)
    v_nev.fill(np.nan)
    std_nev = np.empty(in_shape[:-1],dtype=np.float32)
    std_nev.fill(np.nan)
    
    for ii in range(in_shape[1]):
        if np.isnan(v_aoro[:,ii]).any():
            continue
        
        w_pixel = W_col[:, ii] if W_col is not None else None
        v, std, _ = compute_3DVel_single_hermet(v_aoro[:,ii], A, w_pixel=w_pixel)
        v_nev[:,ii] = v
        std_nev[:,ii] = std

    return v_nev, std_nev


# ─────────────────────────────────────────────────────────────────────────────
# Main 3-D velocity computation (multiprocessing)
# ─────────────────────────────────────────────────────────────────────────────

def compute_3DVel_mp(V_aoro, config_asc_path, config_des_path, W=None):
    """
    Compute 3-D NEU velocity using per-pixel uncertainty weights.

    Parameters
    ----------
    V_aoro          : 4D array, shape (4, rows, cols)
                      [v_rg_asc, v_az_asc, v_rg_des, v_az_des]
    config_asc_path : str   path to ascending testGeogrid.txt
    config_des_path : str   path to descending testGeogrid.txt
    W               : 4D array, shape (4, rows, cols), optional
                      Per-pixel weight maps  [1/σ²].  Row ordering:
                        0 = w_rg_asc, 1 = w_az_asc,
                        2 = w_rg_des, 3 = w_az_des
                      If None, equal weights are used (original behaviour).
    """
    incid_asc = (get_incang(config_asc_path))*(np.pi/180)
    fl_ang_asc = (get_flightang(config_asc_path))*(np.pi/180)
    incid_des = (get_incang(config_des_path))*(np.pi/180)
    fl_ang_des = (get_flightang(config_des_path))*(np.pi/180)

    A = get_matA_3D(fl_ang_asc, incid_asc, fl_ang_des, incid_des)
    
    V_nev = np.empty((3, *V_aoro.shape[1:]),dtype=np.float32)
    V_nev.fill(np.nan)

    V_inp = V_aoro.copy()
    # V_inp[V_inp==0] = np.nan

    in_shape = V_nev.shape
    num_cores = min(mp.cpu_count(), 64)

    # Build per-column weight slices (W[:, :, jj]) or None when no uncertainty
    chunk_inputs = [
        (jj, in_shape, A, V_inp[:, :, jj],
         W[:, :, jj] if W is not None else None)
        for jj in range(in_shape[2])
    ]

    with mp.Pool(processes=num_cores) as pool:
        results = pool.map(unpacking_column, chunk_inputs)

    V_nev = np.transpose(np.array([r[0] for r in results]), (1,2,0))
    V_std = np.transpose(np.array([r[1] for r in results]), (1,2,0))
    
    return V_nev, V_std


# ─────────────────────────────────────────────────────────────────────────────
# Batch inversion entry point
# ─────────────────────────────────────────────────────────────────────────────

def batch_inversion(id_master_slave, asc_path, des_path, out_dir,
                    sigma_base=None):
    """
    Run the 3-D velocity inversion for all matching ascending/descending pairs.

    Parameters
    ----------
    id_master_slave : ndarray, shape (N, 3)
                      Columns: [stack_id, master_date, slave_date]
    asc_path        : str   root directory of the ascending stack
                      (must contain offset_tracking/ subdirectories)
    des_path        : str   root directory of the descending stack
    out_dir         : str   output directory for 3-D velocity GeoTIFFs
    sigma_base      : str or None
                      Base directory of the pre-computed De Zan σ_NCC maps
                      (e.g. "/mnt/Vinayak/UNCERTAINTY/Final").  Subdirectory
                      for each year_track is resolved automatically.
                      If None, equal weights are used (original behaviour).
    """
    asc_imgs = glob(f'{asc_path}/offset_tracking/*/velocity.tif')
    des_imgs = glob(f'{des_path}/offset_tracking/*/velocity.tif')
    stacks = list(set([st.split('/')[-2] for st in asc_imgs]) & set([st.split('/')[-2] for st in des_imgs]))
    # assert len(stacks)==len(des_imgs)
    assert len(stacks)!=0
    
    # cwd = os.getcwd()
    os.makedirs(out_dir, exist_ok=True)

    # Resolve sigma subdirectory once per track
    sigma_dir_asc = _resolve_sigma_dir(sigma_base, asc_path) if sigma_base else None
    sigma_dir_des = _resolve_sigma_dir(sigma_base, des_path) if sigma_base else None

    if sigma_base:
        print(f"[3D Inversion] Using De Zan σ_NCC weights")
        print(f"  σ_base     : {sigma_base}")
        print(f"  σ_dir_asc  : {sigma_dir_asc or 'NOT FOUND — fallback to equal weights'}")
        print(f"  σ_dir_des  : {sigma_dir_des or 'NOT FOUND — fallback to equal weights'}")
    else:
        print("[3D Inversion] No sigma_base supplied — using equal weights (original mode)")

    start = time.time()
    for stack in tqdm(stacks):
        try:
            dt_m, dt_s = id_master_slave[np.where(id_master_slave[:,0]==stack)][0,1:]
        except:
            continue

        va_rg, va_az, projs, geo = read_velocity(f'{asc_path}/offset_tracking/{stack}/velocity.tif')
        vd_rg, vd_az, _, _ = read_velocity(f'{des_path}/offset_tracking/{stack}/velocity.tif')
        V_aoro = np.array([va_rg, va_az, vd_rg, vd_az])

        # ── Build per-pixel weight array from De Zan σ_NCC maps ──────────
        W = None
        if sigma_base:
            sig_az_asc, sig_rg_asc = read_sigma(sigma_dir_asc, stack)
            sig_az_des, sig_rg_des = read_sigma(sigma_dir_des, stack)

            # Only assemble W when at least one track has valid sigmas
            have_asc = sig_rg_asc is not None and sig_az_asc is not None
            have_des = sig_rg_des is not None and sig_az_des is not None

            if have_asc or have_des:
                rows, cols = va_rg.shape

                # Convert σ → weight maps; fall back to unit weight if missing
                w_rg_asc = sigma_to_weight(sig_rg_asc) if have_asc else np.full((rows, cols), _FALLBACK_WEIGHT, dtype=np.float32)
                w_az_asc = sigma_to_weight(sig_az_asc) if have_asc else np.full((rows, cols), _FALLBACK_WEIGHT, dtype=np.float32)
                w_rg_des = sigma_to_weight(sig_rg_des) if have_des else np.full((rows, cols), _FALLBACK_WEIGHT, dtype=np.float32)
                w_az_des = sigma_to_weight(sig_az_des) if have_des else np.full((rows, cols), _FALLBACK_WEIGHT, dtype=np.float32)

                # Stack into (4, rows, cols): rows ordered as A-matrix
                #   row 0: v_rg_asc → w_rg_asc
                #   row 1: v_az_asc → w_az_asc
                #   row 2: v_rg_des → w_rg_des
                #   row 3: v_az_des → w_az_des
                W = np.stack([w_rg_asc, w_az_asc, w_rg_des, w_az_des], axis=0)

        V_nev, V_std = compute_3DVel_mp(V_aoro,
                                 f'{asc_path}/offset_tracking/testGeogrid.txt',
                                 f'{des_path}/offset_tracking/testGeogrid.txt',
                                 W=W)

        ds = numpy_array_to_raster(f'{out_dir}/{dt_m}_{dt_s}_{stack[6:]}.tif', V_nev, projs, geo, nband=3)
        ds.FlushCache()
        ds = None

        ds_std = numpy_array_to_raster(f'{out_dir}/{dt_m}_{dt_s}_{stack[6:]}_unc.tif', V_std, projs, geo, nband=3)
        ds_std.FlushCache()
        ds_std = None

    print("Time Taken:", time.time()-start)
