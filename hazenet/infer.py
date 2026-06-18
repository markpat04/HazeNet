"""Reconstruct a trained CLNO from a checkpoint (used by eval + attribution)."""
from __future__ import annotations

import torch
from .model.clno import CLNO


def load_model(ckpt_path: str, device: str = "cpu"):
    ck = torch.load(ckpt_path, map_location=device)
    model = CLNO(H=ck["H"], W=ck["W"], n_stations=ck["S"], in_ch=ck["in_ch"],
                 kind=ck["model_kind"], hidden=ck["hidden"], rank=ck["rank"],
                 dropout=0.0, emission_curve=ck.get("emission_curve", True),
                 quantiles=ck.get("quantiles")).to(device)
    model.load_state_dict(ck["state_dict"]); model.eval()
    return model, ck
