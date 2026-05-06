#!/bin/bash
#SBATCH --account=nlagent
#SBATCH --partition=sporc-gpu
#SBATCH --job-name=medmnist_smoke
#SBATCH --output=/home/%u/logs/dermamnist_test.out
#SBATCH --error=/home/%u/logs/dermamnist_test.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --mail-user=slack:@pn2694
#SBATCH --mail-type=ALL
#SBATCH --time=01:00:00

# Activate environment
source ~/ghn3_env/bin/activate

# Go to repo
cd ~/ghn3

echo "Starting DermaMNIST run with GHN-3 XL/M16 init (50 epochs)"

# Run the smoke test
python train_ddp.py -d dermamnist -D ./data --arch resnet50 --name dermamnist-ghn3 -e 50 \
    -b 64 -i 224 --lr 0.025 --ckpt ghn3xlm16.pt