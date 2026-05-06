#!/bin/bash
#SBATCH --account=nlagent
#SBATCH --partition=sporc-cpu
#SBATCH --job-name=ghn3_test
#SBATCH --output=/home/%u/logs/ghn3_test_%j.out
#SBATCH --error=/home/%u/logs/ghn3_test_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00

# Activate environment
source ~/ghn3_env/bin/activate

# Go to repo
cd ~/ghn3

# Run the smoke test
python eval.py -d imagenet -D ./data --arch resnet50 --ckpt ./checkpoints/resnet50_ghn3_predicted.pt