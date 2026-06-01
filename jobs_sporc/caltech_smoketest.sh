#!/bin/bash
#SBATCH --account=nlagent
#SBATCH --partition=sporc-cpu
#SBATCH --job-name=caltech_loader
#SBATCH --output=/home/%u/logs/caltech256_smoke.out
#SBATCH --error=/home/%u/logs/caltech256_smoke.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --mail-user=slack:@pn2694
#SBATCH --mail-type=ALL
#SBATCH --time=00:30:00

# Activate environment
source ~/ghn3_env/bin/activate

# Go to repo
cd ~/ghn3

echo "Starting smoke test for Caltech256"

# Run the smoke test
python caltech_loader.py caltech256

