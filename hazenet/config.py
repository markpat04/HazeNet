"""
Config system — one YAML drives the whole pipeline.

Load with:  cfg = Config.load("configs/local.yaml")
Everything downstream reads from `cfg`; nothing is hard-coded per-run.

Relative paths in the YAML are resolved against the repository ROOT
(the parent of this package), so a config works regardless of CWD.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

try:
    import yaml
except ImportError as e:  # pragma: no cover
    raise ImportError("pyyaml is required: pip install pyyaml") from e

# repo root = parent of the hazenet/ package
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _abs(p: Optional[str]) -> Optional[str]:
    if p is None:
        return None
    return p if os.path.isabs(p) else os.path.join(ROOT, p)


@dataclass
class Config:
    name: str
    raw: dict                       # full parsed YAML, for anything ad-hoc

    # ── domain ──
    lat0: float; lat1: float; nlat: int
    lon0: float; lon1: float; nlon: int
    step: float

    # ── time ──
    years: list
    test_year: int
    season_months: list             # e.g. [2,3,4] or [11,12,1,2,3,4,5]

    # ── features ──
    channels: list                  # met/static channel names (encoder input)
    precip_accum_window: int
    tpi_radius: int
    enso_csv: Optional[str]

    # ── model ──
    model_kind: str                 # lowrank | globalv
    hidden: int
    rank: int
    dropout: float
    sfeat_hidden: int               # station-feature MLP hidden size
    emission_curve: bool
    quantiles: Optional[list]       # e.g. [0.1,0.5,0.9] or None

    # ── train ──
    epochs: int
    batch_size: int
    lr: float
    weight_decay: float
    grad_clip: float
    amp: bool
    seed: int
    num_workers: int
    patience: int                   # early-stopping patience (0 = disabled)

    # ── paths ──
    grid_nc: str
    pm25_glob: str
    raw_dir: str
    out_dir: str
    models_dir: str
    figures_dir: str

    # ── imbalanced-regression (Sprint 1) — Label Distribution Smoothing ──
    lds: bool = False               # enable LDS-weighted loss
    lds_reweight: str = "sqrt_inv"  # "sqrt_inv" | "inv"
    lds_sigma: float = 2.0          # Gaussian kernel width (bins)
    lds_max_weight: float = 10.0    # clip weights to [1/max, max]

    # ── emission curve shape (Sprint 2) ──
    emission_curve_kind: str = "sat"  # sat | power | sat_linear

    # ── PI-DGGNN (Phase A) ──
    compute_advection_weights: bool = False   # precompute a_wind (T,S,G) — needed for pidggnn
    pidggnn_alpha_init: float = 0.0           # initial physics-coupling α; 0 = starts as CLNO

    # ---------- derived ----------
    @property
    def LAT(self) -> np.ndarray:
        return np.round(np.linspace(self.lat0, self.lat1, self.nlat), 1)

    @property
    def LON(self) -> np.ndarray:
        return np.round(np.linspace(self.lon0, self.lon1, self.nlon), 1)

    @property
    def H(self) -> int: return self.nlat

    @property
    def W(self) -> int: return self.nlon

    @property
    def G(self) -> int: return self.nlat * self.nlon

    @property
    def datacube_zarr(self) -> str:
        return os.path.join(self.out_dir, "datacube.zarr")

    @property
    def target_csv(self) -> str:
        return os.path.join(self.out_dir, "target_pm25.csv")

    @property
    def ckpt_path(self) -> str:
        return os.path.join(self.models_dir, f"clno_{self.name}.pt")

    @property
    def artifacts_path(self) -> str:
        return os.path.join(self.models_dir, f"clno_{self.name}_artifacts.pt")

    def dates(self) -> pd.DatetimeIndex:
        """All daily timestamps where year in `years` and month in `season_months`."""
        out = []
        for y in self.years:
            rng = pd.date_range(f"{y}-01-01", f"{y}-12-31", freq="D")
            out.extend(d for d in rng if d.month in self.season_months)
        return pd.DatetimeIndex(sorted(set(out)))

    @property
    def n_quantiles(self) -> int:
        return len(self.quantiles) if self.quantiles else 0

    # ---------- loader ----------
    @classmethod
    def load(cls, path: str) -> "Config":
        with open(_abs(path), "r", encoding="utf-8") as f:
            y = yaml.safe_load(f)

        dom, tim, feat = y["domain"], y["time"], y["features"]
        mdl, trn, pth = y["model"], y["train"], y["paths"]

        return cls(
            name=y["name"], raw=y,
            lat0=dom["lat"][0], lat1=dom["lat"][1], nlat=dom["lat"][2],
            lon0=dom["lon"][0], lon1=dom["lon"][1], nlon=dom["lon"][2],
            step=dom.get("step", 0.1),
            years=list(tim["years"]), test_year=tim["test_year"],
            season_months=list(tim["season_months"]),
            channels=list(feat["channels"]),
            precip_accum_window=feat.get("precip_accum_window", 3),
            tpi_radius=feat.get("tpi_radius", 5),
            enso_csv=_abs(feat.get("enso_csv")),
            model_kind=mdl.get("kind", "globalv"),
            hidden=mdl.get("hidden", 64), rank=mdl.get("rank", 32),
            dropout=mdl.get("dropout", 0.1),
            sfeat_hidden=mdl.get("sfeat_hidden", 16),
            emission_curve=mdl.get("emission_curve", True),
            quantiles=mdl.get("quantiles") or None,
            emission_curve_kind=mdl.get("emission_curve_kind", "sat"),
            epochs=trn.get("epochs", 200), batch_size=trn.get("batch_size", 32),
            lr=trn.get("lr", 1e-3), weight_decay=trn.get("weight_decay", 1e-4),
            grad_clip=trn.get("grad_clip", 1.0), amp=trn.get("amp", False),
            seed=trn.get("seed", 42), num_workers=trn.get("num_workers", 0),
            patience=trn.get("patience", 0),
            lds=trn.get("lds", False),
            lds_reweight=trn.get("lds_reweight", "sqrt_inv"),
            lds_sigma=trn.get("lds_sigma", 2.0),
            lds_max_weight=trn.get("lds_max_weight", 10.0),
            compute_advection_weights=mdl.get("compute_advection_weights", False),
            pidggnn_alpha_init=mdl.get("pidggnn_alpha_init", 0.0),
            grid_nc=_abs(pth["grid_nc"]), pm25_glob=_abs(pth["pm25_glob"]),
            raw_dir=_abs(pth.get("raw_dir", "data/raw_m2")),
            out_dir=_abs(pth.get("out_dir", "data/processed_hn")),
            models_dir=_abs(pth.get("models_dir", "models")),
            figures_dir=_abs(pth.get("figures_dir", "figures")),
        )

    def summary(self) -> str:
        return (f"[{self.name}] grid {self.H}×{self.W} (G={self.G})  "
                f"years={self.years} test={self.test_year}  "
                f"months={self.season_months}\n"
                f"  channels({len(self.channels)})={self.channels}\n"
                f"  model={self.model_kind} hidden={self.hidden} rank={self.rank} "
                f"sfeat_hidden={self.sfeat_hidden} "
                f"curve={self.emission_curve} quantiles={self.quantiles}")
