#!/bin/bash
#SBATCH --job-name=ml_jazz_eval
#SBATCH --time=10:00:00
#SBATCH --mem=16GB
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu-gen
#SBATCH --gres=gpu:rtx6000:1
#SBATCH --output=logs/slurm-fad-%j.out
#SBATCH --error=logs/slurm-fad-%j-err.out

set -euo pipefail

# Create logs/ before calling sbatch; Slurm opens these files before the
# script begins. Submit with: mkdir -p logs && sbatch slurm_fad.sh

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

# Load cluster-specific modules here if required.
# module purge
# module load python/3.11 cuda/12.1

VENV_DIR="venv"

if [ ! -x "${VENV_DIR}/bin/python" ]; then
    echo "[$(date --iso-8601=seconds)] Creating virtual environment"
    python3 -u -m venv "${VENV_DIR}"
    "${VENV_DIR}/bin/python" -u -m pip install --upgrade pip
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

echo "[$(date --iso-8601=seconds)] Installing evaluation dependencies"
python -u -m pip install -r requirements-eval.txt

echo "[$(date --iso-8601=seconds)] Verifying CUDA"
python -u -c 'import torch; print("torch:", torch.__version__); print("CUDA available:", torch.cuda.is_available()); print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"); raise SystemExit(0 if torch.cuda.is_available() else "CUDA is required for this job")'

echo "[$(date --iso-8601=seconds)] Generating 30 samples per condition and computing FAD"
python -u -m scripts.evaluate \
    --config configs/train.json \
    --adapter output/jazz_cafe_lora \
    --device cuda \
    --count 30 \
    --steps 100 \
    --fad

echo "[$(date --iso-8601=seconds)] FAD evaluation completed"
