"""
LOYO experiment harness — sweep loss / feature settings on the SAME datacube,
print a clean comparison table, log each run to the manifest + W&B.
No matplotlib (avoids the Windows Agg DLL crash).

Usage:  python run_experiment.py <base_config.yaml> <out_name> [sweep]
  sweep ∈ { curve, lds, lag }   (default: curve)
"""
import os, sys, json, copy
os.chdir("C:/Users/mark/Desktop/internship")
sys.path.insert(0, "C:/Users/mark/Desktop/internship")
os.environ.setdefault("GDAL_DATA", "C:/Users/mark/miniconda3/envs/hazenet/Library/share/gdal")

import numpy as np
from hazenet.config import Config
from hazenet.loyo import _load_raw, _fit_fold
from hazenet.features import physics
from hazenet.tracking import Experiment

base_cfg_path = sys.argv[1] if len(sys.argv) > 1 else "configs/local_cds.yaml"
out_name = sys.argv[2] if len(sys.argv) > 2 else "experiment_sweep"
sweep = sys.argv[3] if len(sys.argv) > 3 else "curve"

# ── variant definitions ──────────────────────────────────────────────────
# Each value is (cfg_overrides, extra_lag_windows). Lag windows append trailing
# FRP-sum channels to met_raw IN MEMORY (no datacube rebuild needed).
if sweep == "curve":
    VARIANTS = {
        "baseline_sat": (dict(emission_curve=True, emission_curve_kind="sat"), []),
        "linear":       (dict(emission_curve=False), []),
        "power":        (dict(emission_curve=True, emission_curve_kind="power"), []),
        "sat_linear":   (dict(emission_curve=True, emission_curve_kind="sat_linear"), []),
    }
elif sweep == "lds":
    VARIANTS = {
        "baseline":     (dict(lds=False), []),
        "lds_max3":     (dict(lds=True, lds_reweight="sqrt_inv", lds_max_weight=3.0), []),
        "lds_max2":     (dict(lds=True, lds_reweight="sqrt_inv", lds_max_weight=2.0), []),
        "lds_max5":     (dict(lds=True, lds_reweight="sqrt_inv", lds_max_weight=5.0), []),
    }
elif sweep == "lag":
    VARIANTS = {
        "baseline":     (dict(), []),
        "lag3":         (dict(), [3]),
        "lag7":         (dict(), [7]),
        "lag3_7":       (dict(), [3, 7]),
    }
else:
    raise SystemExit(f"unknown sweep: {sweep}")

print(f"Loading datacube from {base_cfg_path} ...  sweep={sweep}", flush=True)
base = Config.load(base_cfg_path)
met_raw, emis_raw, y_raw, times, S, station_feats = _load_raw(base)
import torch
dev = "cuda" if torch.cuda.is_available() else "cpu"
yrs = np.array([t.year for t in times])
years = sorted(set(int(y) for y in yrs))
print(f"device={dev}  years={years}  stations={S}  met_ch={met_raw.shape[1]}", flush=True)


def with_lag(met, windows):
    """Append trailing-FRP-sum channels (computed from emis_raw) to met."""
    if not windows:
        return met
    extra = [physics.emission_accum(emis_raw, window=w)[:, None] for w in windows]
    return np.concatenate([met] + extra, axis=1).astype("float32")


results = {}
for name, (overrides, lag_windows) in VARIANTS.items():
    cfg = copy.copy(base)
    for k, v in overrides.items():
        setattr(cfg, k, v)
    met_v = with_lag(met_raw, lag_windows)
    print(f"\n>>> variant: {name}  {overrides}  lag={lag_windows}  met_ch={met_v.shape[1]}", flush=True)
    exp = Experiment(name=f"{base.name}:{name}", project="hazenet",
                     config={**{k: getattr(cfg, k, None) for k in
                                ("model_kind", "hidden", "rank", "dropout", "seed",
                                 "emission_curve", "emission_curve_kind", "lds",
                                 "lds_reweight", "lds_max_weight")},
                             "lag_windows": lag_windows, "met_ch": int(met_v.shape[1]),
                             **overrides},
                     data_path=base.datacube_zarr, tags=[base.name, sweep],
                     notes=f"{sweep}:{name}")
    rows = []
    for Y in years:
        test = yrs == Y; train = ~test
        r = _fit_fold(cfg, met_v, emis_raw, y_raw, station_feats, train, test, S, dev)
        r["year"] = Y; rows.append(r)
        exp.log({f"MAE/{Y}": r["MAE"], f"bias/{Y}": r["bias"]})
        print(f"    {Y}: MAE={r['MAE']:.2f}  bias={r['bias']:+.2f}", flush=True)
    mean_mae = float(np.mean([r["MAE"] for r in rows]))
    worst = float(np.max([r["MAE"] for r in rows]))
    seen = [r["seen"]["MAE"] for r in rows if r["seen"]["MAE"] is not None]
    new = [r["new"]["MAE"] for r in rows if r["new"]["MAE"] is not None]
    y2023 = [r for r in rows if r["year"] == 2023][0]
    summary = dict(mean_MAE=mean_mae, worst_MAE=worst,
                   seen_MAE=float(np.mean(seen)), new_MAE=float(np.mean(new)),
                   mae_2023=y2023["MAE"], bias_2023=y2023["bias"])
    exp.finish(summary)
    results[name] = dict(**summary, folds=rows)

print("\n\n================ COMPARISON ================", flush=True)
print(f"{'variant':<16} {'mean':>6} {'worst':>6} {'SEEN':>6} {'NEW':>6} {'2023MAE':>8} {'2023bias':>9}", flush=True)
for name, r in results.items():
    print(f"{name:<16} {r['mean_MAE']:>6.2f} {r['worst_MAE']:>6.2f} {r['seen_MAE']:>6.2f} "
          f"{r['new_MAE']:>6.2f} {r['mae_2023']:>8.2f} {r['bias_2023']:>+9.2f}", flush=True)

json.dump(results, open(f"models/{out_name}.json", "w"), indent=2, default=float)
print(f"\nsaved -> models/{out_name}.json", flush=True)
