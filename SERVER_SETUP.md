# HazeNet — Server Setup & Runbook (2x RTX 2080 Ti, Linux)

You SSH into the server yourself; these are the commands to run there. Code/env are
fully scripted; data + API keys are manual (never committed).

> **Security:** rotate any password shared in plaintext, and prefer SSH-key auth.
> API keys go ONLY in `~/.cdsapirc` / env vars on the server — never in the repo.

---

## 1. One-shot setup (code + conda env + GPU check)

SSH in, then:

```bash
git clone git@github.com:markpat04/HazeNet.git ~/HazeNet && cd ~/HazeNet
git checkout v2-phase0
bash setup_server.sh
```

`setup_server.sh` is idempotent: checks `nvidia-smi`, installs miniconda if missing,
clones/pulls `v2-phase0`, creates/updates the `hazenet` env, and prints
`torch … cuda True … n_gpu 2`. Re-run any time to pull latest + update env.

---

## 2. Phase 0 — reproduce the 5-year pipeline (sanity gate)

Data is NOT in git. Transfer the existing 5-yr datacube once, from your **LOCAL** machine:

```bash
rsync -avz data/processed_cds/  makufff@10.50.3.22:~/HazeNet/data/processed_cds/
```

Then on the server:

```bash
conda activate hazenet
python run_pipeline.py local_cds "train,eval,loyo,loso"
```

**Pass criteria** (paste these JSONs / W&B link back):
- exit 0, no segfault (Linux is clean)
- `models/eval_local_cds.json` → GATE W2 PASS, test MAE ≈ 19
- `models/loyo_local_cds.json` → mean ≈ 19.1, 2023 ≈ 34
- `models/loso_local_cds.json` → mean ≈ 17.3

---

## 3. W&B tracking

```bash
wandb login          # key stored in ~/.netrc, NEVER in the repo
```

---

## 4. 9-year build (Phase C, stretch) — run in the background, in parallel

NOT on the MVP critical path. Kick it off and let it run while Phase 0/A proceed.

**4a. API keys (on the server only — never commit):**
```bash
printf 'url: https://cds.climate.copernicus.eu/api\nkey: <UID>:<APIKEY>\n' > ~/.cdsapirc
export FIRMS_MAP_KEY=<your_firms_map_key>     # add to ~/.bashrc to persist
df -h .                                        # confirm tens of GB free
```

**4b. Fetch + build (auto-pulls DEM/ERA5/FIRMS/PM2.5/ENSO, resume-able):**
```bash
nohup python run_pipeline.py server9yr "fetch,grid,datacube" > fetch9yr.log 2>&1 &
tail -f fetch9yr.log        # watch progress; Ctrl-C just stops the tail
```

**Expectations (honest):**
- ERA5 = ~9yr×7mo of CDS requests → slow queue (hours, sometimes a day).
- ⚠️ PM2.5 2016–2018: OpenAQ coverage in N. Thailand was thin in early years —
  check how many stations/year landed before trusting the 9-yr cube.
- FIRMS (VIIRS) fire covers 2016–2024 fine.

**4c. When the cube is built, train:**
```bash
CUDA_VISIBLE_DEVICES=0 python run_pipeline.py server9yr "train,eval,loyo,loso"
```
(Multi-GPU DDP needs train.py changes — not wired yet; one 2080 Ti fits the model
fine. Use the 2nd GPU for a parallel experiment, not DDP, for now.)

---

## 5. GPU convention (2 GPUs)

```bash
CUDA_VISIBLE_DEVICES=0 python run_experiment.py ...   # experiment A on GPU 0
CUDA_VISIBLE_DEVICES=1 python run_experiment.py ...   # experiment B on GPU 1 (parallel)
```

---

When Phase 0 passes → **Phase A**: build the Level-2 model
(`hazenet/model/transport.py` → `pidggnn.py`) + LOYO/LOSO + ablation.
