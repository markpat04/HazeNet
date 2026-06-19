"""Losses — all NaN-masked (stations report on different days).

Imbalanced-regression support (Sprint 1)
----------------------------------------
PM2.5 is heavily right-skewed: severe-haze days (the ones we care about most)
are rare, so a plain masked loss is dominated by ordinary days and the model
systematically UNDER-predicts the extremes (our 2023 problem, bias < 0).

We counter this with Label Distribution Smoothing (LDS) — Yang et al.,
"Delving into Deep Imbalanced Regression", ICML 2021 — which weights each
sample by the inverse of a Gaussian-smoothed empirical label density, so rare
high targets contribute more to the loss. Weights are computed on TRAIN targets
only (per fold) to avoid leakage, normalised to mean≈1, and clipped.
"""
import numpy as np
import torch
import torch.nn.functional as F


def masked_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """MSE over the valid (non-NaN) entries of target. pred,target: (B,S)."""
    mask = ~torch.isnan(target)
    if mask.sum() == 0:
        return pred.sum() * 0.0
    return F.mse_loss(pred[mask], target[mask])


def pinball_loss(preds: torch.Tensor, target: torch.Tensor, quantiles) -> torch.Tensor:
    """
    Quantile (pinball) loss, NaN-masked.

    preds:     (B, S, Q)  — one prediction per quantile
    target:    (B, S)
    quantiles: list of floats, len Q  (e.g. [0.1, 0.5, 0.9])

    Pinball_q(e) = max(q*e, (q-1)*e),  e = target - pred_q
    """
    mask = ~torch.isnan(target)
    if mask.sum() == 0:
        return preds.sum() * 0.0

    tgt = target.unsqueeze(-1)                       # (B,S,1)
    q = torch.tensor(quantiles, device=preds.device, dtype=preds.dtype)  # (Q,)
    err = tgt - preds                                # (B,S,Q)
    loss = torch.maximum(q * err, (q - 1.0) * err)   # (B,S,Q)
    m = mask.unsqueeze(-1).expand_as(loss)
    return loss[m].mean()


# ─────────────────────────────────────────────────────────────────────────
# Label Distribution Smoothing (LDS) — imbalanced regression
# ─────────────────────────────────────────────────────────────────────────
def _gaussian_kernel1d(sigma: float, radius: int | None = None) -> np.ndarray:
    if radius is None:
        radius = max(1, int(round(3 * sigma)))
    x = np.arange(-radius, radius + 1, dtype="float64")
    k = np.exp(-(x ** 2) / (2 * sigma ** 2))
    return k / k.sum()


def compute_lds_weights(train_targets, n_bins: int = 50, sigma: float = 2.0,
                        reweight: str = "sqrt_inv", max_weight: float = 10.0):
    """
    Build a (bin_edges, bin_weights) table for Label Distribution Smoothing.

    train_targets : 1-D array of TRAIN target values (NaNs allowed). Use the
                    same scale that the loss sees (e.g. normalised y), since LDS
                    only depends on the *shape* of the distribution.

    Returns (edges, weights) as float32 arrays; edges has n_bins+1 entries,
    weights has n_bins. Look up a target's weight via bucketize(t, edges)-1.
    """
    t = np.asarray(train_targets, dtype="float64").ravel()
    t = t[np.isfinite(t)]
    if t.size == 0:
        return (np.array([0.0, 1.0], dtype="float32"),
                np.array([1.0], dtype="float32"))

    lo = float(np.percentile(t, 0.5))
    hi = float(np.percentile(t, 99.5))
    if hi <= lo:
        hi = lo + 1e-6
    edges = np.linspace(lo, hi, n_bins + 1)
    hist, _ = np.histogram(np.clip(t, lo, hi), bins=edges)

    # LDS: convolve empirical density with a symmetric Gaussian kernel
    k = _gaussian_kernel1d(sigma)
    eff = np.convolve(hist.astype("float64"), k, mode="same") + 1e-6

    if reweight == "inv":
        w = 1.0 / eff
    else:  # "sqrt_inv" — gentler, the ICML'21 default-style choice
        w = 1.0 / np.sqrt(eff)

    # normalise so the AVERAGE sample weight ≈ 1 (keeps loss scale stable),
    # then clip to avoid a handful of ultra-rare points exploding the gradient
    bin_idx = np.clip(np.digitize(t, edges) - 1, 0, n_bins - 1)
    w = w / w[bin_idx].mean()
    w = np.clip(w, 1.0 / max_weight, max_weight)
    return edges.astype("float32"), w.astype("float32")


def _lookup_weights(target: torch.Tensor, edges, weights) -> torch.Tensor:
    e = torch.as_tensor(edges, device=target.device, dtype=target.dtype)
    w = torch.as_tensor(weights, device=target.device, dtype=target.dtype)
    idx = torch.clamp(torch.bucketize(target, e) - 1, 0, w.numel() - 1)
    return w[idx]


def weighted_pinball_loss(preds, target, quantiles, edges, weights) -> torch.Tensor:
    """Pinball loss with per-sample LDS weights based on the target value."""
    mask = ~torch.isnan(target)
    if mask.sum() == 0:
        return preds.sum() * 0.0
    wmap = _lookup_weights(torch.nan_to_num(target, nan=0.0), edges, weights)  # (B,S)
    tgt = target.unsqueeze(-1)
    q = torch.tensor(quantiles, device=preds.device, dtype=preds.dtype)
    err = tgt - preds
    loss = torch.maximum(q * err, (q - 1.0) * err)              # (B,S,Q)
    wexp = wmap.unsqueeze(-1).expand_as(loss)
    m = mask.unsqueeze(-1).expand_as(loss)
    return (loss * wexp)[m].mean()


def weighted_masked_mse(pred, target, edges, weights) -> torch.Tensor:
    """Masked MSE with per-sample LDS weights based on the target value."""
    mask = ~torch.isnan(target)
    if mask.sum() == 0:
        return pred.sum() * 0.0
    wmap = _lookup_weights(torch.nan_to_num(target, nan=0.0), edges, weights)
    se = (pred - torch.nan_to_num(target, nan=0.0)) ** 2
    return (se * wmap)[mask].mean()
