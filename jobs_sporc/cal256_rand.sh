#!/bin/bash
#SBATCH --account=nlagent
#SBATCH --partition=sporc-gpu
#SBATCH --job-name=cal256_rand
#SBATCH --output=/home/%u/logs/cal256_rand.out
#SBATCH --error=/home/%u/logs/cal256_rand.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=10G
#SBATCH --mail-user=slack:@pn2694
#SBATCH --mail-type=ALL
#SBATCH --time=04:00:00

source ~/ghn3_env/bin/activate
cd ~/ghn3

echo "Starting Caltech 256 run with random init (50 epochs)"

python train_ddp.py -d caltech256 -D ./data --arch resnet50 --name resnet50-cal256-rand -e 50 \
    -b 64 -i 224 --wd 1e-4 --lr 0.1
