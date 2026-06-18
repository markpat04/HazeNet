"""
Unified Conditionally-Linear Neural Operator (CLNO).

Core relation — PM2.5 is LINEAR in the (curved) emission field:

    median_s = Σ_g  K_sg(met) · φ(E_g)  +  b_s(met)

  K  = transport kernel   (softmax over G sources; nonlinear in met, linear in E)
  φ  = emission curve      (learnable saturating FRP→emission; identity if disabled)
  b  = background PM2.5     (non-fire / inflow)

One trained model still gives forecast + attribution + inversion for free,
because the prediction is linear in φ(E). The emission curve and the quantile
heads do NOT break that linearity (φ is applied per-cell before K@·; quantiles
are additive offsets around the linear median).

`kind`:
  full     — K_head: hidden → S·G            (small G only)
  lowrank  — K = softmax(U(met) @ V(met)ᵀ)   (U,V predicted per sample)
  globalv  — K = softmax(U(met) @ V_globalᵀ) (V is a shared learned param; ~40× fewer params)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────
class EmissionCurve(nn.Module):
    """Learnable saturating map φ(E)=scale·(1−exp(−E/τ)). Near-linear for small E."""
    def __init__(self, tau_init: float = 0.4):
        super().__init__()
        # softplus(raw)≈tau_init at init
        self.log_tau = nn.Parameter(torch.tensor(float(tau_init)).expm1().clamp_min(1e-3).log())
        self.log_scale = nn.Parameter(torch.zeros(()))

    def forward(self, E: torch.Tensor) -> torch.Tensor:
        tau = F.softplus(self.log_tau) + 1e-4
        scale = F.softplus(self.log_scale) + 1e-4
        return scale * (1.0 - torch.exp(-E / tau))


# ─────────────────────────────────────────────────────────────────────────
class CLNO(nn.Module):
    def __init__(self, H: int, W: int, n_stations: int, in_ch: int,
                 kind: str = "lowrank", hidden: int = 64, rank: int = 32,
                 dropout: float = 0.1, emission_curve: bool = True,
                 quantiles=None):
        super().__init__()
        self.H, self.W = H, W
        self.G = H * W
        self.S = n_stations
        self.in_ch = in_ch
        self.kind = kind
        self.rank = rank
        self.quantiles = list(quantiles) if quantiles else None

        # context encoder (met/static → hidden)
        self.encoder = nn.Sequential(
            nn.Conv2d(in_ch, 16, 3, padding=1), nn.GELU(),
            nn.Conv2d(16, 32, 3, padding=1), nn.GELU(),
            nn.AdaptiveAvgPool2d((8, 8)),
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, hidden), nn.GELU(),
            nn.Dropout(dropout),
        )

        # transport-kernel heads
        if kind == "full":
            self.K_head = nn.Linear(hidden, self.S * self.G)
        elif kind == "lowrank":
            self.U_head = nn.Linear(hidden, self.S * rank)
            self.V_head = nn.Linear(hidden, self.G * rank)
        elif kind == "globalv":
            self.U_head = nn.Linear(hidden, self.S * rank)
            self.V_global = nn.Parameter(torch.randn(self.G, rank) * 0.02)
        else:
            raise ValueError(f"unknown kind: {kind}")

        self.b_head = nn.Linear(hidden, self.S)
        self.curve = EmissionCurve() if emission_curve else None

        # quantile spread: positive gaps between consecutive quantiles
        if self.quantiles:
            assert 0.5 in self.quantiles, "quantiles must include the median 0.5"
            self.q_med = self.quantiles.index(0.5)
            self.gap_head = nn.Linear(hidden, self.S * (len(self.quantiles) - 1))

    # ── kernel ──
    def _build_K(self, h: torch.Tensor) -> torch.Tensor:
        B = h.shape[0]
        if self.kind == "full":
            logit = self.K_head(h).view(B, self.S, self.G)
        elif self.kind == "lowrank":
            U = self.U_head(h).view(B, self.S, self.rank)
            V = self.V_head(h).view(B, self.G, self.rank)
            logit = torch.bmm(U, V.transpose(1, 2))
        else:  # globalv
            U = self.U_head(h).view(B, self.S, self.rank)
            logit = torch.einsum("bsr,gr->bsg", U, self.V_global)
        return torch.softmax(logit, dim=-1)

    # ── forward ──
    def forward(self, met, emission):
        """
        met:      (B, in_ch, H, W)  normalised
        emission: (B, H, W)         normalised to ~[0,1]

        Returns
          out : (B,S) median if no quantiles, else (B,S,Q) sorted by quantile
          K   : (B,S,G)
          b   : (B,S)
        """
        B = met.shape[0]
        h = self.encoder(met)
        K = self._build_K(h)
        b = F.softplus(self.b_head(h))

        E = emission.view(B, self.G)
        if self.curve is not None:
            E = self.curve(E)
        median = torch.bmm(K, E.unsqueeze(-1)).squeeze(-1) + b   # (B,S)

        if not self.quantiles:
            return median, K, b

        # build non-crossing quantiles via cumulative positive gaps around median
        gaps = F.softplus(self.gap_head(h)).view(B, self.S, -1)  # (B,S,Q-1)
        Q = len(self.quantiles)
        preds = [None] * Q
        preds[self.q_med] = median
        # upward from median
        acc = median
        for i in range(self.q_med + 1, Q):
            acc = acc + gaps[:, :, i - 1]
            preds[i] = acc
        # downward from median
        acc = median
        for i in range(self.q_med - 1, -1, -1):
            acc = acc - gaps[:, :, i]
            preds[i] = acc
        out = torch.stack(preds, dim=-1)                         # (B,S,Q)
        return out, K, b

    def predict_median(self, out: torch.Tensor) -> torch.Tensor:
        """Pull the median column from a forward() output (handles both shapes)."""
        if out.dim() == 2:
            return out
        return out[:, :, self.q_med]

    # ── attribution (uses curved emission, consistent with forward) ──
    @torch.no_grad()
    def attribution(self, K, emission):
        E = emission.view(emission.shape[0], self.G)
        if self.curve is not None:
            E = self.curve(E)
        contrib = K * E.unsqueeze(1)                             # (B,S,G)
        total = contrib.sum(dim=-1, keepdim=True).clamp(1e-8)
        frac = contrib / total
        shape = (K.shape[0], self.S, self.H, self.W)
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


def build_model(cfg, n_stations: int, in_ch: int) -> CLNO:
    return CLNO(H=cfg.H, W=cfg.W, n_stations=n_stations, in_ch=in_ch,
                kind=cfg.model_kind, hidden=cfg.hidden, rank=cfg.rank,
                dropout=cfg.dropout, emission_curve=cfg.emission_curve,
                quantiles=cfg.quantiles)
