"""
Conditionally-Linear Neural Operator (CLNO) for PM2.5 haze forecasting.

Key property: PM2.5 prediction is LINEAR in the emission field E:
    PM2.5_s(t) = Σ_g  K_sg(met_t) · E_g(t)  +  b_s(met_t)

K  = transport kernel  — nonlinear function of met (wind, terrain)
                         but LINEAR dependence on E
b  = background PM2.5  — from non-fire sources, boundary inflow

One trained model gives three outputs for free:
  1. Forecast     : PM2.5 = K @ E + b
  2. Attribution  : A_sg = K_sg * E_g / Σ K_sg*E_g   (which fire → which station)
  3. Inversion    : E = lstsq(K, PM2.5 - b)           (recover emission from obs)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CLNO(nn.Module):
    def __init__(self, H: int, W: int, n_stations: int, hidden: int = 32):
        super().__init__()
        self.H, self.W = H, W
        self.G = H * W          # number of source grid cells
        self.S = n_stations

        # Met encoder: (3, H, W) -> (hidden,)
        # Encodes wind + terrain into a context vector for each timestep
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 8,  3, padding=1), nn.GELU(),
            nn.Conv2d(8, 16, 3, padding=1), nn.GELU(),
            nn.AdaptiveAvgPool2d((4, 4)),   # -> (16, 4, 4) = 256
            nn.Flatten(),
            nn.Linear(256, hidden), nn.GELU(),
            nn.Dropout(0.1),
        )

        # Transport kernel: context -> K (S, G)  [softmax over sources]
        # K[s,g] = how strongly source g contributes to receptor s
        self.K_head = nn.Linear(hidden, self.S * self.G)

        # Background PM2.5 per station (non-fire, non-negative)
        self.b_head = nn.Linear(hidden, self.S)

    # ------------------------------------------------------------------
    def forward(self, met, emission):
        """
        met:      (B, 3, H, W)  — [u10, v10, dem], normalised
        emission: (B, H, W)     — FRP field, normalised to [0,1]

        Returns
          pm25 : (B, S)     — predicted PM2.5
          K    : (B, S, G)  — transport kernel
          b    : (B, S)     — background
        """
        B = met.shape[0]
        h = self.encoder(met)                                   # (B, hidden)

        K = torch.softmax(
            self.K_head(h).view(B, self.S, self.G), dim=-1
        )                                                       # (B, S, G)
        b = F.softplus(self.b_head(h))                          # (B, S)

        E = emission.view(B, self.G)                            # (B, G)
        pm25 = torch.bmm(K, E.unsqueeze(-1)).squeeze(-1) + b   # (B, S)
        return pm25, K, b

    # ------------------------------------------------------------------
    @torch.no_grad()
    def attribution(self, K, emission):
        """
        Source contribution map.

        K:        (B, S, G)
        emission: (B, H, W)

        Returns:
          contrib   (B, S, H, W)  — absolute contribution
          fraction  (B, S, H, W)  — fraction of total (sums to 1 over sources)
        """
        E = emission.view(emission.shape[0], self.G)        # (B, G)
        contrib = K * E.unsqueeze(1)                         # (B, S, G)  K_sg * E_g
        total = contrib.sum(dim=-1, keepdim=True).clamp(1e-8)
        fraction = contrib / total
        shape = (K.shape[0], self.S, self.H, self.W)
        return contrib.view(shape), fraction.view(shape)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def invert(self, K, pm25_obs, b=None, alpha: float = 0.01):
        """
        Recover emission field from PM2.5 observations (linear inverse problem).
        Solves:   K @ E  ≈  (PM2.5 - b)   with Tikhonov regularisation.

        K:        (B, S, G)
        pm25_obs: (B, S)     — observed PM2.5 (NaN = missing station)
        b:        (B, S)     — background (optional; subtracted from obs)
        alpha:    float      — regularisation strength

        Returns:  (B, H, W)  — estimated emission field

        Uses dual-form (S×S solve) when G > S, primal (G×G) otherwise.
        Dual form: (K Kᵀ + αI) λ = y,  E = Kᵀλ   → O(S³) instead of O(G³)
        This is critical for M2 where G=11111 >> S~40.
        """
        results = []
        for i in range(K.shape[0]):
            Ki = K[i]                                   # (S, G)
            yi = pm25_obs[i].clone()
            if b is not None:
                yi = yi - b[i]

            valid = ~torch.isnan(yi)
            Ki_v, yi_v = Ki[valid], yi[valid]          # (Sv, G), (Sv,)
            Sv = valid.sum().item()

            if Sv < 2:
                results.append(torch.zeros(self.G, device=K.device))
                continue

            if self.G > Sv:
                # Dual form: (Ki_v @ Ki_v.T + αI) λ = yi_v  → (Sv×Sv)
                M     = Ki_v @ Ki_v.T + alpha * torch.eye(Sv, device=K.device)
                lam   = torch.linalg.solve(M, yi_v)    # (Sv,)
                E_hat = Ki_v.T @ lam                   # (G,)
            else:
                # Primal form: (Ki_v.T @ Ki_v + αI) E = Ki_v.T @ yi_v  → (G×G)
                A     = Ki_v.T @ Ki_v + alpha * torch.eye(self.G, device=K.device)
                E_hat = torch.linalg.solve(A, Ki_v.T @ yi_v)

            results.append(F.relu(E_hat))               # non-negative emission

        return torch.stack(results).view(-1, self.H, self.W)
