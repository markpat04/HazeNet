from .clno import CLNO, EmissionCurve
from .pidggnn import PIDGGNN
from .losses import (masked_mse, pinball_loss, compute_lds_weights,
                     weighted_pinball_loss, weighted_masked_mse)


def build_model(cfg, in_ch: int):
    """Dispatch to CLNO or PIDGGNN based on cfg.model_kind."""
    kind = cfg.model_kind
    common = dict(
        H=cfg.H, W=cfg.W, in_ch=in_ch,
        hidden=cfg.hidden, rank=cfg.rank,
        dropout=cfg.dropout, n_sfeats=4, sfeat_hidden=cfg.sfeat_hidden,
        emission_curve=cfg.emission_curve, quantiles=cfg.quantiles,
        emission_curve_kind=getattr(cfg, "emission_curve_kind", "sat"),
    )
    if kind == "pidggnn":
        return PIDGGNN(kind="globalv", alpha_init=getattr(cfg, "pidggnn_alpha_init", 0.0),
                       **common)
    if kind == "pidggnn_lowrank":
        return PIDGGNN(kind="lowrank", alpha_init=getattr(cfg, "pidggnn_alpha_init", 0.0),
                       **common)
    # Default: CLNO (kind = "globalv" | "lowrank")
    return CLNO(kind=kind, **common)


__all__ = ["CLNO", "PIDGGNN", "EmissionCurve", "build_model",
           "masked_mse", "pinball_loss", "compute_lds_weights",
           "weighted_pinball_loss", "weighted_masked_mse"]
