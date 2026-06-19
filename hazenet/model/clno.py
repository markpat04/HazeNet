"""
Unified Conditionally-Linear Neural Operator (CLNO) — station-agnostic edition.

Core relation — PM2.5 is LINEAR in the (curved) emission field:

    median_s = Σ_g  K_sg(met, sfeats_s) · φ(E_g)  +  b_s(met, sfeats_s)

  K  = transport kernel (softmax; nonlinear in met AND station features)
  φ  = emission curve   (learnable saturating FRP→emission)
  b  = background PM2.5 (non-fire / inflow)

The key architectural change from the indexed version: b_s and U_s are no longer
looked up by station index.  Instead they are COMPUTED from station features
(lat, lon, dem, tpi) via an additive decomposition:

    U_s(met, sfeats) = U_ctx(met)  +  U_stat(station_enc(sfeats))
    b_s(met, sfeats) = b_ctx(met)  +  b_stat(station_enc(sfeats))

This lets the model predict at ANY point in the domain — not just the 99 training
stations — and is the fix for the spatial-generalisation failure (LOYO new-station
MAE 300 → ~seen-station level).

`kind`:
  lowrank  — K = softmax(U(met,s) @ V(met)ᵀ)    (V context-only)
  globalv  — K = softmax(U(met,s) @ V_globalᵀ)  (V shared learned param; default)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────
class EmissionCurve(nn.Module):
    """
    Learnable FRP→emission map φ(E).  `kind`:

      sat         φ = scale·(1−exp(−E/τ))            (saturating; original)
      power       φ = scale·(E+ε)^γ                  (unbounded; γ learnable)
      sat_linear  φ = scale·(1−exp(−E/τ)) + lin·E    (saturating + linear tail)

    Motivation (ACP 2024, SE-Asia biomass burning): fire PM2.5 is *under*-counted
    (OC emission factor 3–4× too low for peat; up to 54% AOD/PM2.5 deficit). A
    purely SATURATING curve caps the contribution of the biggest fires, which
    worsens the severe-haze (2023) underestimation. `power` (γ can exceed 1) and
    `sat_linear` let large fires contribute proportionally more.
    """
    def __init__(self, kind: str = "sat", tau_init: float = 0.4):
        super().__init__()
        self.kind = kind
        if kind in ("sat", "sat_linear"):
            self.log_tau = nn.Parameter(torch.tensor(float(tau_init)).expm1().clamp_min(1e-3).log())
            self.log_scale = nn.Parameter(torch.zeros(()))
        if kind == "sat_linear":
            self.log_lin = nn.Parameter(torch.tensor(-2.0))   # small initial linear tail
        if kind == "power":
            self.log_scale = nn.Parameter(torch.zeros(()))
            self.log_gamma = nn.Parameter(torch.zeros(()))    # γ = softplus(·)+ε ≈ 0.7 init
        if kind not in ("sat", "power", "sat_linear"):
            raise ValueError(f"unknown emission curve kind: {kind!r}")

    def forward(self, E: torch.Tensor) -> torch.Tensor:
        if self.kind == "power":
            scale = F.softplus(self.log_scale) + 1e-4
            gamma = F.softplus(self.log_gamma) + 1e-2
            return scale * torch.pow(E.clamp_min(0.0) + 1e-6, gamma)
        tau = F.softplus(self.log_tau) + 1e-4
        scale = F.softplus(self.log_scale) + 1e-4
        out = scale * (1.0 - torch.exp(-E / tau))
        if self.kind == "sat_linear":
            lin = F.softplus(self.log_lin) + 1e-6
            out = out + lin * E.clamp_min(0.0)
        return out


# ─────────────────────────────────────────────────────────────────────────
class CLNO(nn.Module):
    def __init__(self, H: int, W: int, in_ch: int,
                 kind: str = "globalv", hidden: int = 64, rank: int = 32,
                 dropout: float = 0.1, n_sfeats: int = 4, sfeat_hidden: int = 16,
                 emission_curve: bool = True, quantiles=None,
                 emission_curve_kind: str = "sat"):
        super().__init__()
        self.H, self.W = H, W
        self.G = H * W
        self.in_ch = in_ch
        self.kind = kind
        self.rank = rank
        self.n_sfeats = n_sfeats
        self.quantiles = list(quantiles) if quantiles else None
        sE = sfeat_hidden

        # ── context encoder (met/static grid → hidden) ──
        self.encoder = nn.Sequential(
            nn.Conv2d(in_ch, 16, 3, padding=1), nn.GELU(),
            nn.Conv2d(16, 32, 3, padding=1), nn.GELU(),
            nn.AdaptiveAvgPool2d((8, 8)),
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, hidden), nn.GELU(),
            nn.Dropout(dropout),
        )

        # ── station-feature encoder (lat/lon/dem/tpi → sE) ──
        self.station_encoder = nn.Sequential(
            nn.Linear(n_sfeats, sE), nn.GELU(),
            nn.Linear(sE, sE),
        )

        # ── transport-kernel: additive context + station decomposition ──
        if kind not in ("lowrank", "globalv"):
            raise ValueError(f"unknown kind: {kind!r}  (use 'globalv' or 'lowrank')")
        self.U_ctx = nn.Linear(hidden, rank)       # (B, rank)
        self.U_stat = nn.Linear(sE, rank)          # (S, rank)
        if kind == "globalv":
            self.V_global = nn.Parameter(torch.randn(self.G, rank) * 0.02)
        else:  # lowrank — V is context-only (source-space projection)
            self.V_head = nn.Linear(hidden, self.G * rank)

        # ── background: additive context + station decomposition ──
        self.b_ctx = nn.Linear(hidden, 1)
        self.b_stat = nn.Linear(sE, 1)

        self.curve = EmissionCurve(kind=emission_curve_kind) if emission_curve else None

        # ── quantile spread (non-crossing) ──
        if self.quantiles:
            assert 0.5 in self.quantiles, "quantiles must include the median 0.5"
            self.q_med = self.quantiles.index(0.5)
            nq = len(self.quantiles) - 1
            self.gap_ctx = nn.Linear(hidden, nq)
            self.gap_stat = nn.Linear(sE, nq)

    # ── forward ──
    def forward(self, met, emission, station_feats):
        """
        met:           (B, in_ch, H, W)   normalised met/static grid
        emission:      (B, H, W)          normalised FRP emission
        station_feats: (S, n_sfeats)      normalised station lat/lon/dem/tpi

        Returns
          out : (B,S) median if no quantiles, else (B,S,Q) sorted by quantile
          K   : (B,S,G)  transport kernel (softmax)
          b   : (B,S)    background PM2.5
        """
        B = met.shape[0]
        h = self.encoder(met)                                 # (B, hidden)
        sE = self.station_encoder(station_feats)              # (S, sfeat_hidden)

        # U: additive context + station → (B, S, rank)
        U = self.U_ctx(h).unsqueeze(1) + self.U_stat(sE).unsqueeze(0)

        # V and kernel K: (B, S, G)
        if self.kind == "globalv":
            logit = torch.einsum("bsr,gr->bsg", U, self.V_global)
        else:
            V = self.V_head(h).view(B, self.G, self.rank)    # (B, G, rank)
            logit = torch.bmm(U, V.transpose(1, 2))
        K = torch.softmax(logit, dim=-1)                      # (B, S, G)

        # background: additive context + station → (B, S)
        b = F.softplus(
            self.b_ctx(h).unsqueeze(1) + self.b_stat(sE).unsqueeze(0)
        ).squeeze(-1)

        # emission curve + kernel integration
        E = emission.view(B, self.G)
        if self.curve is not None:
            E = self.curve(E)
        median = torch.bmm(K, E.unsqueeze(-1)).squeeze(-1) + b   # (B, S)

        if not self.quantiles:
            return median, K, b

        # non-crossing quantiles via cumulative positive gaps around median
        gaps = F.softplus(
            self.gap_ctx(h).unsqueeze(1) + self.gap_stat(sE).unsqueeze(0)
        )                                                         # (B, S, Q-1)
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
        return torch.stack(preds, dim=-1), K, b                  # (B,S,Q)

    def predict_median(self, out: torch.Tensor) -> torch.Tensor:
        if out.dim() == 2:
            return out
        return out[:, :, self.q_med]

    # ── attribution (uses curved emission, consistent with forward) ──
    @torch.no_grad()
    def attribution(self, K, emission):
        E = emission.view(emission.shape[0], self.G)
        if self.curve is not None:
            E = self.curve(E)
        contrib = K * E.unsqueeze(1)                              # (B,S,G)
        total = contrib.sum(dim=-1, keepdim=True).clamp(1e-8)
        frac = contrib / total
        shape = (K.shape[0], K.shape[1], self.H, self.W)
        return contrib.view(shape), frac.view(shape)

    # ── inversion (dual-form Tikhonov; recovers curved emission) ──
    @torch.no_grad()
    def invert(self, K, pm25_obs, b=None, alpha: float = 0.01):
        results = []
        for i in range(K.shape[0]):
            Ki = K[i]; yi = pm25_obs[i].clone()
            if b is not None:
                yi = yi - b[i]
            valid = ~torch.isnan(yi)
            Ki_v, yi_v = Ki[valid], yi[valid]
            Sv = int(valid.sum().item())
            if Sv < 2:
                results.append(torch.zeros(self.G, device=K.device)); continue
            M = Ki_v @ Ki_v.T + alpha * torch.eye(Sv, device=K.device)
            lam = torch.linalg.solve(M, yi_v)
            results.append(F.relu(Ki_v.T @ lam))
        return torch.stack(results).view(-1, self.H, self.W)


def build_model(cfg, in_ch: int) -> CLNO:
    return CLNO(H=cfg.H, W=cfg.W, in_ch=in_ch,
                kind=cfg.model_kind, hidden=cfg.hidden, rank=cfg.rank,
                dropout=cfg.dropout, n_sfeats=4, sfeat_hidden=cfg.sfeat_hidden,
                emission_curve=cfg.emission_curve, quantiles=cfg.quantiles,
                emission_curve_kind=getattr(cfg, "emission_curve_kind", "sat"))
