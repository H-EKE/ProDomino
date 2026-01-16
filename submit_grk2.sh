#!/bin/bash
#SBATCH --job-name=prodomino_grk2
#SBATCH --output=logs/prodomino_grk2_%j.out
#SBATCH --error=logs/prodomino_grk2_%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

# ============================================================
# ProDomino - GRK2 Insertion Site Prediction
# ============================================================
# Adjust the SBATCH parameters above for your cluster:
#   --partition: your GPU partition name (e.g., gpu, a100, v100)
#   --gres: GPU resource specification
#   --mem: 32GB should be sufficient for ESM-2 3B model
#   --time: typically completes in <30 minutes
# ============================================================

# Create logs directory if it doesn't exist
mkdir -p logs

# Load modules (adjust for your cluster)
# module load anaconda3
# module load cuda/12.1

# Activate conda environment
# conda activate prodomino
# OR: source /path/to/your/venv/bin/activate

# Set working directory
cd $SLURM_SUBMIT_DIR

# Print job info
echo "============================================================"
echo "ProDomino - GRK2 Prediction"
echo "============================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "Working directory: $(pwd)"
echo "============================================================"

# Check GPU availability
nvidia-smi

# Run prediction for human GRK2 (UniProt: P21146)
python predict_insertion_sites.py \
    --uniprot P21146 \
    --output results/grk2 \
    --name GRK2_human

# Alternative: use a local FASTA file
# python predict_insertion_sites.py \
#     --fasta data/grk2.fasta \
#     --output results/grk2 \
#     --name GRK2_human

# Alternative: with PDB structure for B-factor mapping
# python predict_insertion_sites.py \
#     --uniprot P21146 \
#     --pdb data/grk2_structure.pdb \
#     --chain A \
#     --output results/grk2 \
#     --name GRK2_human

echo "============================================================"
echo "End time: $(date)"
echo "============================================================"
