#!/bin/bash
#SBATCH -o ./log/job.%j.out
#SBATCH --partition=C064M0256G
#SBATCH --qos=low
#SBATCH -J Multiwfn_analysis
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4

# ================================================================
# SLURM Script for Multiwfn Feature Extraction
# ================================================================

# Load environment
source ~/.bashrc
conda activate unimol_tools  # or your preferred environment

# Set molecule index
export i=0

# Create log directory
mkdir -p ./log

# Run feature extraction script
echo "Extracting features for molecule $i"
python /path/to/generate_features.py "$i"
