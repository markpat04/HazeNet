"""Losses — all NaN-masked (stations report on different days)."""
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
