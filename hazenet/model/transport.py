"""
Wind-advection transport weights — emission grid cell g → receptor station s.

This is the *physics prior* on the transport kernel K of the Level-2 model
(`hazenet/model/pidggnn.py`). It discretises the **advection** term of the
advection–diffusion equation as a directed, distance-penalised projection of the
observed wind field, following:

  - Zhao et al., *Dynamic Geographical GNN*, Environ. Model. & Soft. 2025
    (wind-field edge construction, eqs. 2–4): edge weight = wind projected onto
    the source→receptor direction, distance-penalised.
  - Zhang et al., *Physics-Guided Spatiotemporal Decoupling* (PGSD), 2025:
    advection kernel  W_adv ∝ v · D⁻¹.

For station s, grid cell g, day t:

    a_wind[s, g, t] = relu( ⟨ wind(g, t), dir(g → s) ⟩ ) · exp(−d(g,s)/ℓ)
                      (zeroed when d(g,s) > max_radius_km)

Interpretation: wind at the *source* cell g blowing *toward* receptor s (positive
projection) advects emission from g to s. Negative projection ⇒ no transport
(relu), exactly DGGNN's two-directed-edge idea. The result is a physically
meaningful, non-negative prior that the learnable kernel modulates — not a free
parameter.

Grid-cell ordering MATCHES the model: cell index g = h·W + w, i.e.
lat = LAT[g // W], lon = LON[g % W] — identical to `emission.view(B, H*W)` in
`clno.py`, so a_wind aligns with K and the emission vector φ(E).

All functions are pure (numpy in → numpy out) and computed on CPU; the advection
field is deterministic given observed wind, so it is precomputed once and reused
(no future leakage — same-day wind only).
"""
from __future__ import annotations

import numpy as np

# mean Earth degree → km (equirectangular local approximation)
_KM_PER_DEG_LAT = 110.57
_KM_PER_DEG_LON = 111.32      # scaled by cos(latitude)


def grid_centers(LAT: np.ndarray, LON: np.ndarray) -> np.ndarray:
    """
    (G, 2) array of [lat, lon] for every grid cell, in the model's flatten order
    g = h·W + w  (row-major over (lat, lon)) — must match clno.py's
    emission.view(B, H*W).  LAT has length H, LON has length W.
    """
    LAT = np.asarray(LAT, dtype="float32")
    LON = np.asarray(LON, dtype="float32")
    lat_g, lon_g = np.meshgrid(LAT, LON, indexing="ij")   # (H, W) each
    return np.stack([lat_g.ravel(), lon_g.ravel()], axis=1).astype("float32")


def _local_km(lat1, lon1, lat2, lon2):
    """Equirectangular east/north offset (km) from point 1 → point 2."""
    mean_lat = np.deg2rad((lat1 + lat2) * 0.5)
    east = (lon2 - lon1) * _KM_PER_DEG_LON * np.cos(mean_lat)
    north = (lat2 - lat1) * _KM_PER_DEG_LAT
    return east, north


def advection_weights_day(
    u_day: np.ndarray,            # (H, W) 10 m zonal wind (east+, m/s) — RAW, not normalised
    v_day: np.ndarray,            # (H, W) 10 m meridional wind (north+, m/s)
    grid_xy: np.ndarray,          # (G, 2) lat/lon of cells, from grid_centers()
    station_xy: np.ndarray,       # (S, 2) lat/lon of stations
    length_scale_km: float = 150.0,
    max_radius_km: float = 400.0,
) -> np.ndarray:
    """
    Return a_wind (S, G) for one day: advective transport weight from each grid
    cell g (emission source) to each station s (receptor).
    """
    u = np.asarray(u_day, dtype="float32").ravel()        # (G,)
    v = np.asarray(v_day, dtype="float32").ravel()        # (G,)
    g_lat = grid_xy[:, 0][None, :]                         # (1, G)
    g_lon = grid_xy[:, 1][None, :]
    s_lat = station_xy[:, 0][:, None]                      # (S, 1)
    s_lon = station_xy[:, 1][:, None]

    # offset from source cell g → receptor station s (km, east/north)
    east, north = _local_km(g_lat, g_lon, s_lat, s_lon)   # (S, G) each
    dist = np.sqrt(east * east + north * north) + 1e-6     # (S, G)
    inv = 1.0 / dist
    dir_e = east * inv                                     # unit dir g→s
    dir_n = north * inv

    # wind at source cell projected onto g→s direction (broadcast over stations)
    proj = u[None, :] * dir_e + v[None, :] * dir_n         # (S, G), m/s toward s
    w = np.maximum(proj, 0.0) * np.exp(-dist / length_scale_km)
    w[dist > max_radius_km] = 0.0
    return w.astype("float32")


def precompute_advection(
    u: np.ndarray,                # (T, H, W) RAW zonal wind
    v: np.ndarray,                # (T, H, W) RAW meridional wind
    LAT: np.ndarray,
    LON: np.ndarray,
    station_xy: np.ndarray,       # (S, 2)
    length_scale_km: float = 150.0,
    max_radius_km: float = 400.0,
) -> np.ndarray:
    """
    Precompute a_wind for every day → (T, S, G), float32.

    NOTE memory: T·S·G can be large (e.g. 446·99·11211 ≈ 0.5 B floats ≈ 2 GB).
    The radius mask makes it mostly zeros, so it is a candidate for sparse
    storage later; for the 5-year local cube the dense array is acceptable.
    Caller decides whether to hold it in RAM, memmap, or stream per-batch.
    """
    T = u.shape[0]
    S = station_xy.shape[0]
    G = len(LAT) * len(LON)
    grid_xy = grid_centers(LAT, LON)
    out = np.zeros((T, S, G), dtype="float32")
    for t in range(T):
        out[t] = advection_weights_day(
            u[t], v[t], grid_xy, station_xy,
            length_scale_km=length_scale_km, max_radius_km=max_radius_km)
    return out


def row_normalize(a_wind: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Normalise so each station's incoming weights sum to 1 (mass-conserving, like
    DGGNN's random-walk normalisation).  a_wind: (..., S, G) → same shape.
    Stations with no in-radius wind (all-zero row) are left as zeros.
    """
    s = a_wind.sum(axis=-1, keepdims=True)
    return a_wind / np.maximum(s, eps)
