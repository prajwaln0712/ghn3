#!/bin/bash
#SBATCH --account=nlagent
#SBATCH --partition=sporc-gpu
#SBATCH --job-name=cal101_ghn3
#SBATCH --output=/home/%u/logs/cal101_ghn3.out
#SBATCH --error=/home/%u/logs/cal101_ghn3.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=10G
#SBATCH --mail-user=slack:@pn2694
#SBATCH --mail-type=ALL
#SBATCH --time=02:00:00

source ~/ghn3_env/bin/activate
cd ~/ghn3

echo "Starting Caltech 101 run with GHN-3 XL/M16 init (50 epochs)"

python train_ddp.py -d caltech101 -D ./data --arch resnet50 --name resnet50-cal101-ghn3 -e 50 \
    -b 64 -i 224 --wd 1e-4 --lr 0.025 --ckpt ghn3xlm16.pt
