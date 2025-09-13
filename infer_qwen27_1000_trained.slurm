#!/bin/bash
#SBATCH --job-name=gen_qwen27_1000_trained
#SBATCH --gres=gpu:1              # request 1 GPU
#SBATCH --cpus-per-task=8         # number of CPU cores
#SBATCH --mem=48G                 # RAM
#SBATCH --time=0-08:00            # walltime (D-HH:MM)
#SBATCH --output=/home/pranon/scratch/def-tahmedge/pranon/experiments/logs/%x-%j.out
#SBATCH --error=/home/pranon/scratch/def-tahmedge/pranon/experiments/logs/%x-%j.err
#SBATCH --mail-user=raianrahman@iut-dhaka.edu
#SBATCH --mail-type=ALL

# -----------------------------
# 1. Load required modules
# -----------------------------

module load python/3.13
module load cuda/12.6
module load scipy-stack
module load arrow/21.0.0

# -----------------------------
# 2. Create virtual environment
# -----------------------------
VENV_DIR=$SLURM_TMPDIR/venv
# python -m venv $VENV_DIR
source /home/pranon/projects/def-tahmedge/pranon/pyenv/py13/bin/activate

# Upgrade pip (no index so it uses ComputeCanada wheels if available)
pip install --no-index --upgrade pip setuptools wheel

cd /home/pranon/projects/def-tahmedge/pranon/llama-factory-training

# -----------------------------
# 5. Run training
# -----------------------------
accelerate launch -m src.main --model_name "Qwen2-VL-7B-Instruct-1000-trained"
