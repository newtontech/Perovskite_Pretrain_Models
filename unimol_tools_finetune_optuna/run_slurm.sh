#!/bin/bash
#SBATCH -o ./log/job.%j.out
#SBATCH -e ./log/job.%j.err
#SBATCH --partition=GPU40G
#SBATCH --qos=low
#SBATCH --gres=gpu:1
#SBATCH -J UniMol_finetune
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

# ================================================================
# SLURM Script for UniMol Fine-tuning with Optuna
# ================================================================

# Load environment
source ~/.bashrc
conda activate unimol_tools

# Set UniMol weights directory
export UNIMOL_WEIGHT_DIR='/lustre/home/2101110149/.local/lib/python3.10/site-packages/unimol_tools/weights'

# Create directories
mkdir -p ./log
mkdir -p ./output

# Set paths
DATA_PATH="../baselines/data/split_seed_0/train.csv"
TEST_PATH="../baselines/data/split_seed_0/test.csv"
SAVE_PATH="./output/unimol_model"

# Print job info
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "=========================================="

# Run training
echo "Starting UniMol fine-tuning..."
python unimol_finetune.py \
    --data_path "$DATA_PATH" \
    --test_path "$TEST_PATH" \
    --save_path "$SAVE_PATH" \
    --model_name unimolv2 \
    --model_size 84m \
    --epochs 100 \
    --batch_size 16 \
    --learning_rate 1e-4 \
    --early_stopping 20 \
    --kfold 5

# Check exit status
if [ $? -eq 0 ]; then
    echo "Training completed successfully!"
else
    echo "Training failed with error code $?"
fi

echo "End time: $(date)"
