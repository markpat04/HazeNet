"""
Backfill historical sweeps into W&B (runs that predate the tracking wiring).

Pushes the LDS sweep and the emission-curve sweep as W&B runs so the project has
ALL experiments in one place. The lag sweep is skipped here — it was already
captured offline during its run; sync it with `wandb sync --sync-all`.

Run AFTER `wandb login` so these go straight to the online project.
"""
import os, sys, json
os.chdir("C:/Users/mark/Desktop/internship")
sys.path.insert(0, "C:/Users/mark/Desktop/internship")

from hazenet.tracking import Experiment

PROJECT = "hazenet"


def push(name, sweep, summary, per_year, config):
    exp = Experiment(name=name, project=PROJECT, config=config,
                     data_path="data/processed_cds/datacube.zarr",
                     tags=["local_cds", sweep, "backfill"], notes=f"backfill:{sweep}")
    for yr, (mae, bias) in per_year.items():
        exp.log({f"MAE/{yr}": mae, f"bias/{yr}": bias})
    exp.finish(summary)
    print(f"  pushed {name}", flush=True)


# ── 1. LDS sweep (from JSON) ──────────────────────────────────────────────
print("Backfilling LDS sweep ...", flush=True)
lds = json.load(open("models/experiment_lds_sweep.json"))
for variant, d in lds.items():
    per_year = {f["year"]: (f["MAE"], f["bias"]) for f in d["folds"]}
    summary = {k: d[k] for k in ("mean_MAE", "worst_MAE", "seen_MAE", "new_MAE",
                                 "mae_2023", "bias_2023")}
    cfg = dict(sweep="lds", lds=(variant != "baseline"),
               lds_max_weight={"baseline": None, "lds_sqrt_max3": 3.0,
                               "lds_sqrt_max2": 2.0, "lds_sqrt_max5": 5.0}.get(variant),
               emission_curve_kind="sat")
    push(f"local_cds:{variant}", "lds", summary, per_year, cfg)

# ── 2. Emission-curve sweep (parsed from the run log; sat_linear didn't finish) ─
print("Backfilling curve sweep ...", flush=True)
curve = {
    "baseline_sat": dict(kind="sat", py={2019: (17.88, -9.01), 2020: (16.06, -6.23),
        2021: (14.10, -1.01), 2022: (13.06, -7.45), 2023: (34.19, -13.00)}),
    "linear": dict(kind="linear", py={2019: (17.90, -9.01), 2020: (16.06, -6.08),
        2021: (14.19, -0.73), 2022: (13.03, -7.21), 2023: (34.27, -12.46)}),
    "power": dict(kind="power", py={2019: (17.85, -8.92), 2020: (16.04, -6.32),
        2021: (14.08, -0.77), 2022: (13.35, -5.85), 2023: (34.11, -13.30)}),
}
for variant, d in curve.items():
    py = d["py"]
    maes = [v[0] for v in py.values()]
    summary = dict(mean_MAE=sum(maes) / len(maes), worst_MAE=max(maes),
                   mae_2023=py[2023][0], bias_2023=py[2023][1])
    cfg = dict(sweep="curve", emission_curve=(d["kind"] != "linear"),
               emission_curve_kind=d["kind"])
    push(f"local_cds:curve_{variant}", "curve", summary, py, cfg)

print("\nDONE — backfill complete. Refresh your W&B project to see them.", flush=True)
