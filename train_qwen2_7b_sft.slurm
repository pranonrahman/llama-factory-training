#!/bin/bash
#SBATCH --job-name=qwen_27_sft
#SBATCH --gres=gpu:1              # request 1 GPU
#SBATCH --cpus-per-task=8         # number of CPU cores
#SBATCH --mem=48G                 # RAM
#SBATCH --time=01-00:00           # walltime (DD-HH:MM)
#SBATCH --output=/home/pranon/scratch/def-tahmedge/pranon/experiments/logs/%x-%j.out
#SBATCH --error=/home/pranon/scratch/def-tahmedge/pranon/experiments/logs/%x-%j.err
#SBATCH --mail-user=raianrahman@iut-dhaka.edu
#SBATCH --mail-type=ALL

# -----------------------------
# 1. Load required modules
# -----------------------------

module load python/3.11.5
module load cuda/12.6
module load scipy-stack/2023b
module load arrow/21.0.0

# -----------------------------
# 2. Create virtual environment
# -----------------------------
source /home/pranon/projects/def-tahmedge/pranon/LLaMA-Factory/llama-factory/bin/activate

# Upgrade pip (no index so it uses ComputeCanada wheels if available)
pip install --no-index --upgrade pip setuptools wheel

# -----------------------------
# 3. Clone your GitHub project
# -----------------------------
# cd $SLURM_TMPDIR
# git clone git@github.com:pranonrahman/sm-trainer.git
# cd sm-trainer

cd /home/pranon/projects/def-tahmedge/pranon/llama-factory-training

# -----------------------------
# 5. Run training
# -----------------------------
export DISABLE_VERSION_CHECK=1

llamafactory-cli train /home/pranon/projects/def-tahmedge/pranon/llama-factory-training/sft/cc_qwen27_train.yaml