#!/bin/bash
#SBATCH --account=nlagent
#SBATCH --partition=sporc-gpu
#SBATCH --job-name=path_ghn3
#SBATCH --output=/home/%u/logs/path_ghn3.out
#SBATCH --error=/home/%u/logs/path_ghn3.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --mail-user=slack:@pn2694
#SBATCH --mail-type=ALL
#SBATCH --time=08:00:00

# Activate environment
source ~/ghn3_env/bin/activate

# Go to repo
cd ~/ghn3

echo "Starting PathMNIST run with GHN-3 XL/M16 init (50 epochs)"

python train_ddp.py -d pathmnist -D ./data --arch resnet50 --name resnet50-path-ghn3 -e 50 \
    -b 64 -i 224 --wd 1e-4 --lr 0.025 --ckpt ghn3xlm16.pt
