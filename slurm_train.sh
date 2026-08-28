#!/bin/bash
#SBATCH --job-name=ml_jazz
#SBATCH --time=10:00:00
#SBATCH --mem=16GB
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu-gen
#SBATCH --gres=gpu:rtx6000:1
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j-err.out

set -euo pipefail

# Create logs/ before calling sbatch; Slurm opens the log files before this
# script begins. Run: mkdir -p logs && sbatch slurm_train.sh

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

# Load cluster-specific modules here if required.
# module purge
# module load python/3.11 cuda/12.1

VENV_DIR="venv"
TRAIN_CONFIG="configs/train.json"

if [ ! -x "${VENV_DIR}/bin/python" ]; then
    echo "[$(date --iso-8601=seconds)] Creating virtual environment"
    python3 -u -m venv "${VENV_DIR}"
    "${VENV_DIR}/bin/python" -u -m pip install --upgrade pip
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

echo "[$(date --iso-8601=seconds)] Installing/verifying dependencies"
python -u -m pip install -r requirements.txt

echo "[$(date --iso-8601=seconds)] Environment"
python -u --version
python -u -m pip list
python -u -c 'import torch; print("torch:", torch.__version__); print("CUDA available:", torch.cuda.is_available()); print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"); raise SystemExit(0 if torch.cuda.is_available() else "CUDA is required for this job")'

echo "[$(date --iso-8601=seconds)] Preparing audio clips and manifest"
python -u -m scripts.prepare_data --config "${TRAIN_CONFIG}"

echo "[$(date --iso-8601=seconds)] Caching VAE latents on GPU"
python -u -m scripts.cache_latents --config "${TRAIN_CONFIG}" --device cuda

echo "[$(date --iso-8601=seconds)] Starting LoRA training"
# The training module creates its own Accelerator, so a second launcher is not
# needed for this single-GPU allocation.
python -u -m scripts.train_lora --config "${TRAIN_CONFIG}"

echo "[$(date --iso-8601=seconds)] Training job completed"
