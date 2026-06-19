#!/usr/bin/env bash
# HazeNet — one-shot server bootstrap.
# Run this AFTER you SSH into the server yourself. It sets up CODE + conda ENV +
# verifies the GPUs. It does NOT touch credentials or data (those are separate,
# manual steps below). Safe to re-run (idempotent).
#
#   curl -fsSL <raw-url>/setup_server.sh | bash      # or: bash setup_server.sh
set -euo pipefail

echo "==================== 1. GPU check ===================="
nvidia-smi --query-gpu=name,memory.total --format=csv \
  || { echo "!! nvidia-smi not found — install NVIDIA drivers first"; exit 1; }

echo "==================== 2. conda ===================="
if ! command -v conda >/dev/null 2>&1; then
  echo ".. installing miniconda"
  wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/mc.sh
  bash /tmp/mc.sh -b -p "$HOME/miniconda3"
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
  conda init bash
else
  source "$(conda info --base)/etc/profile.d/conda.sh"
fi

echo "==================== 3. repo (branch v2-phase0) ===================="
if [ ! -d "$HOME/HazeNet/.git" ]; then
  git clone git@github.com:markpat04/HazeNet.git "$HOME/HazeNet"
fi
cd "$HOME/HazeNet"
git fetch origin
git checkout v2-phase0
git pull --ff-only origin v2-phase0

echo "==================== 4. conda env 'hazenet' ===================="
if conda env list | grep -qE '^hazenet[[:space:]]'; then
  conda env update -n hazenet -f environment.yml
else
  conda env create -f environment.yml
fi

echo "==================== 5. verify torch + GPUs ===================="
conda run -n hazenet python -c \
  "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), '| n_gpu', torch.cuda.device_count())"

cat <<'EOF'

==================== DONE — code + env ready ====================
NEXT (manual):

  conda activate hazenet

  # (A) Phase-0 reproduce — needs the 5-yr datacube. From your LOCAL machine:
  #     rsync -avz data/processed_cds/  makufff@10.50.3.22:~/HazeNet/data/processed_cds/
  python run_pipeline.py local_cds "train,eval,loyo,loso"

  # (B) 9-year build (stretch, background) — needs API keys (NEVER commit them):
  #     printf 'url: https://cds.climate.copernicus.eu/api\nkey: <UID>:<APIKEY>\n' > ~/.cdsapirc
  #     export FIRMS_MAP_KEY=<your_firms_key>
  #     wandb login
  nohup python run_pipeline.py server9yr "fetch,grid,datacube" > fetch9yr.log 2>&1 &
================================================================
EOF
