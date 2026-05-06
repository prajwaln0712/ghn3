#!/bin/bash
#SBATCH --account=nlagent
#SBATCH --partition=sporc-gpu
#SBATCH --job-name=ghn3_test
#SBATCH --output=/home/%u/logs/ghn3_test_%j.out
#SBATCH --error=/home/%u/logs/ghn3_test_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mail-user=slack:@pn2694
#SBATCH --mail-type=ALL
#SBATCH --mem=32G
#SBATCH --time=00:30:00

# Activate environment
source ~/ghn3_env/bin/activate

# Go to repo
cd ~/ghn3

# Run the smoke test
python ghn3_smoketest.py