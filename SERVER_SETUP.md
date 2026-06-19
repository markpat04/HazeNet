# HazeNet — Server Setup & Phase-0 Reproduce (2× RTX 2080 Ti, Linux)

Run these on the server over SSH. Goal of Phase 0: a working env + reproduce the
current pipeline (confirm MAE ≈ 19.1 / LOSO ≈ 17.3, and that the Windows segfaults
are gone on Linux).

## 1. Clone + env

```bash
git clone git@github.com:markpat04/HazeNet.git
cd HazeNet
conda env create -f environment.yml      # creates env "hazenet"
conda activate hazenet
python -c "import torch; print('cuda', torch.cuda.is_available(), 'n_gpu', torch.cuda.device_count())"
# expect: cuda True  n_gpu 2
```

## 2. Get the datacube onto the server (data is NOT in git)

The 5-year datacube + targets live outside git (`*.zarr/`, `data/` are gitignored).
Transfer the existing, verified cube once from your local machine (run on **local**):

```bash
# from local machine — adjust host/path
rsync -avz data/processed_cds/  USER@SERVER:~/HazeNet/data/processed_cds/
# (datacube.zarr + target_pm25.csv + grid_cds.nc live here)
```

## 3. Reproduce the current pipeline (sanity gate)

```bash
conda activate hazenet
# Linux: the torch-before-zarr segfault and matplotlib Agg crash do NOT occur,
# so figures can run; HAZENET_NOFIG=1 is optional now.
python run_pipeline.py local_cds "train,eval,loyo,loso"
```

**Phase-0 pass criteria:**
- runs end-to-end, exit 0
- `models/eval_local_cds.json` → GATE W2 PASS, test MAE ≈ 19 (RMSE ≈ 28–30)
- `models/loyo_local_cds.json` → mean MAE ≈ 19.1, worst (2023) ≈ 34
- `models/loso_local_cds.json` → mean MAE ≈ 17.3
- figures render without crashing (Linux)

Paste those three JSONs (or the W&B run link) back and we compare to the Windows numbers.

## 4. W&B (tracking)

```bash
wandb login        # paste your key once; stored in ~/.netrc, NEVER in the repo
```
Runs auto-log to project `hazenet` via `hazenet/tracking.py`.

## 5. GPU usage convention (2 GPUs)

```bash
CUDA_VISIBLE_DEVICES=0 python run_experiment.py ...   # experiment A on GPU 0
CUDA_VISIBLE_DEVICES=1 python run_experiment.py ...   # experiment B on GPU 1  (parallel)
# heavy single run (9-yr / pretrain): torchrun --nproc_per_node=2 ...
```

---

When Phase 0 passes, we move to **Phase A**: build the Level-2 model
(`hazenet/model/transport.py` → `pidggnn.py`) and run LOYO/LOSO + ablation.
