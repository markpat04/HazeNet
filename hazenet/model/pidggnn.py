"""
PI-DGGNN: Physics-Informed wind-advection kernel (Level-2).

Extends CLNO by incorporating a wind-advection physics prior into the
transport kernel K:

    K[b,s,g] = softmax_g( ctx_logit[b,s,g]  +  α · log(a_wind[b,s,g] + ε) )

where:
    ctx_logit  = U(met, sfeats_s) @ V^T           (learned, same as CLNO)
    a_wind     = row-normalised advection weights  (precomputed; see transport.py)
    α          = learnable scalar, init=0  →  PIDGGNN degenerates to CLNO when α=0

In log-probability space, this is a Bayesian physics prior + learned correction:
  a_wind ≈ 0  (wind blows away from s)  →  log-prior ≈ −∞  (strongly suppressed)
  a_wind ≈ 1  (aligned, close)          →  log-prior ≈ 0   (neutral)

α grows during training to exploit physics when it is predictive.

References
----------
Zhao et al. "Dynamic Geographical GNN", Environ. Model. Soft. 2025
    (directed wind-edge weight, eqs. 2–4)
Zhang et al. "Physics-Guided Spatiotemporal Decoupling" (PGSD) 2025
    (advection kernel W_adv ∝ v · D⁻¹)
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .clno import CLNO


class PIDGGNN(CLNO):
    """
    Level-2 physics-informed model.

    Identical interface to CLNO; the only change is in how K is computed.
    Call signature: model(met, emission, station_feats, a_wind=None)

    Parameters
    ----------
    alpha_init : float
        Initial value of the physics-coupling scalar α.
        0.0 → starts as pure CLNO (safe default).
        Set to 1.0 to bias immediately toward physics.
    All other parameters are forwarded to CLNO.__init__.
    """

    def __init__(self, *args, alpha_init: float = 0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = torch.nn.Parameter(torch.tensor(float(alpha_init)))

    def forward(self, met, emission, station_feats, a_wind=None):
        """
        Parameters
        ----------
        met           : (B, in_ch, H, W)   normalised met + static grid
        emission      : (B, H, W)          normalised FRP/emission
        station_feats : (S, n_sfeats)      lat/lon/dem/tpi (normalised)
        a_wind        : (B, S, G) or None  precomputed row-normalised advection
                        weights from transport.precompute_advection().
                        None → identical to CLNO (physics prior disabled).

        Returns
        -------
        (out, K, b) — same shape contract as CLNO.
        """
        B = met.shape[0]
        h = self.encoder(met)                                # (B, hidden)
        sE = self.station_encoder(station_feats)             # (S, sfeat_hidden)

        U = self.U_ctx(h).unsqueeze(1) + self.U_stat(sE).unsqueeze(0)  # (B, S, rank)

        if self.kind == "globalv":
            logit = torch.einsum("bsr,gr->bsg", U, self.V_global)      # (B, S, G)
        else:
            V = self.V_head(h).view(B, self.G, self.rank)
            logit = torch.bmm(U, V.transpose(1, 2))

        # ── physics prior (additive in log-probability space) ──
        if a_wind is not None:
            physics_logit = self.alpha * torch.log(
                a_wind.to(logit.device, logit.dtype) + 1e-4)            # (B, S, G); 1e-4 safe in fp16
            logit = logit + physics_logit

        K = torch.softmax(logit, dim=-1)                                # (B, S, G)

        b = F.softplus(
            self.b_ctx(h).unsqueeze(1) + self.b_stat(sE).unsqueeze(0)
        ).squeeze(-1)                                                    # (B, S)

        E = emission.view(B, self.G)
        if self.curve is not None:
            E = self.curve(E)
        median = torch.bmm(K, E.unsqueeze(-1)).squeeze(-1) + b          # (B, S)

        if not self.quantiles:
            return median, K, b

        # non-crossing quantiles — identical to CLNO
        gaps = F.softplus(
            self.gap_ctx(h).unsqueeze(1) + self.gap_stat(sE).unsqueeze(0))
        Q = len(self.quantiles)
        preds = [None] * Q
        preds[self.q_med] = median
        acc = median
        for i in range(self.q_med + 1, Q):
            acc = acc + gaps[:, :, i - 1]
            preds[i] = acc
        acc = median
        for i in range(self.q_med - 1, -1, -1):
            acc = acc - gaps[:, :, i]
            preds[i] = acc
        return torch.stack(preds, dim=-1), K, b
