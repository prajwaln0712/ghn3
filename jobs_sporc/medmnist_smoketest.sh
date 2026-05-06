#!/bin/bash
#SBATCH --account=nlagent
#SBATCH --partition=sporc-gpu
#SBATCH --job-name=medmnist_smoke
#SBATCH --output=/home/%u/logs/medmnist_test_gpu.out
#SBATCH --error=/home/%u/logs/medmnist_test_gpu.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --mail-user=slack:@pn2694
#SBATCH --mail-type=ALL
#SBATCH --time=00:30:00

# Activate environment
source ~/ghn3_env/bin/activate

# Go to repo
cd ~/ghn3

nvidia-smi

echo "Starting smoke test for MedMNIST with GHN3-XL-M16 checkpoint"

# Run the smoke test
python train_ddp.py -d dermamnist -D ./data --arch resnet50 --name dermamnist-smoke -e 5 \
    -b 64 -i 224 --lr 0.025 --ckpt ghn3xlm16.pt