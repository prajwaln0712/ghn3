#!/bin/bash
#SBATCH --account=nlagent
#SBATCH --partition=sporc-gpu
#SBATCH --job-name=derma_rand
#SBATCH --output=/home/%u/logs/derma_rand.out
#SBATCH --error=/home/%u/logs/derma_rand.err
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

echo "Starting DermaMNIST run with random init (50 epochs)"

python train_ddp.py -d dermamnist -D ./data --arch resnet50 --name resnet50-derma-rand -e 50 \
    -b 64 -i 224 --wd 1e-4 --lr 0.1
