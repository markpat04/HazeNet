# HazeNet pipeline (`hazenet/`)

One config-driven package: **fetch → grid → datacube → train → eval → attribution**.
The same code runs the small local experiment and the full RunPod build — only the
YAML changes.

```
hazenet/
  config.py        # Config.load(yaml) — single source of truth
  cli.py           # python -m hazenet.cli --config ... --stage ...
  data/            # fetch_era5 (CDS: blh,t850,winds,precip,t2m,d2m), fetch_firms,
                   #   fetch_pm25, fetch_dem, fetch_enso  (resume-able)
  features/
    grid.py        # raw -> grid_<name>.nc  (CDS path)
    physics.py     # wind_speed, tpi, precip_accum, ventilation, inversion, enso
    datacube.py    # grid -> datacube.zarr + targets + train/test masks
  model/
    clno.py        # CLNO {full,lowrank,globalv} + learnable emission curve + quantile heads
    losses.py      # masked_mse, pinball
  dataset.py       # load + train-only normalisation
  train.py         # AMP + checkpoint/resume + dashboard tracker logging
  evaluate.py      # test MAE/RMSE + per-year bias + Gate W2 + figures
  attribution.py   # per (station, day): heatmap, 8 sectors, near/far, top cells
  infer.py         # rebuild model from checkpoint
```

## Run

```bash
# local experiment (446-day window, reuses existing grid, CPU)
hazenet_run.bat --config configs/local.yaml --stage datacube,train,eval

# full build (needs CDS key for ERA5 blh/t850; bigger window; GPU)
hazenet_run.bat --config configs/runpod.yaml --stage all
```

Direct (if not using the launcher) — activate the env first so matplotlib/GDAL
DLLs load: `conda run -n hazenet python -m hazenet.cli --config ... --stage ...`

## Configs
- `configs/local.yaml` — 2019–2023 Feb–Apr, channels derivable from the existing
  grid, `lowrank` + emission curve + quantiles, CPU.
- `configs/runpod.yaml` — 2016–2024 Nov–May, full physics channels (blh,
  ventilation, inversion, enso), AMP on GPU.

## Outputs
- `data/processed_<name>/datacube.zarr`, `target_pm25.csv`
- `models/clno_<name>.pt` (+ `.resume.pt` while training), `metrics.json`, `eval_<name>.json`
- `figures/<name>_pred_vs_true.png`, `<name>_year_bias.png`

## Gate (Definition of Done)
`evaluate` prints **Gate W2**: worst low-dust-year |bias| must be ≤ 25 µg/m³.
