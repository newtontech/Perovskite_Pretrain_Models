#!/bin/bash
#SBATCH -o ./log/job.%j.out
#SBATCH --partition=C064M0256G
#SBATCH --qos=low
#SBATCH -J Gaussian_DFT
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4

# ================================================================
# SLURM Script for Gaussian 16 + Multiwfn Calculations
# ================================================================

# Load environment
source ~/.bashrc
conda activate base  # or your environment with Gaussian/Multiwfn

# Set indices (override with: sbatch --export=i=0 run_gaussian_slurm.sh)
export i=0

# Create log directory
mkdir -p ./log

# Run Gaussian calculation
echo "Running Gaussian calculation for molecule $i"
g16 < "$i".gjf > "$i".log

# Check if Gaussian finished successfully
if grep -q "Normal termination" "$i".log; then
    echo "Gaussian finished successfully"

    # Convert checkpoint to formatted checkpoint
    formchk "$i".chk "$i".fchk

    # Run Multiwfn feature extraction
    echo "Extracting features with Multiwfn..."
    python extract_features.py "$i"
else
    echo "Gaussian calculation failed for molecule $i"
    exit 1
fi
