"""Reconstruct a trained CLNO or PIDGGNN from a checkpoint (used by eval + attribution)."""
from __future__ import annotations

import torch


def load_model(ckpt_path: str, device: str = "cpu"):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    kind = ck.get("model_kind", "globalv")
    kwargs = dict(
        H=ck["H"], W=ck["W"], in_ch=ck["in_ch"],
        hidden=ck["hidden"], rank=ck["rank"],
        dropout=0.0, n_sfeats=ck.get("n_sfeats", 4),
        sfeat_hidden=ck.get("sfeat_hidden", 16),
        emission_curve=ck.get("emission_curve", True),
        quantiles=ck.get("quantiles"),
    )
    if kind.startswith("pidggnn"):
        from .model.pidggnn import PIDGGNN
        kernel = "lowrank" if "lowrank" in kind else "globalv"
        model = PIDGGNN(kind=kernel, alpha_init=0.0, **kwargs)
    else:
        from .model.clno import CLNO
        model = CLNO(kind=kind, **kwargs)
    model.load_state_dict(ck["state_dict"])
    model.to(device)
    model.eval()
    return model, ck
