#!/bin/bash

#SBATCH --job-name=CelebA
#SBATCH --nodes=1
#SBATCH -p GPU
#SBATCH -t 0:05:00
#SBATCH --gpus=v100-16:8
#SBATCH --output=/jet/home/cmiyamot/upscaling-ddpm/sbatch/logging/FashionMNIST_%j.out
#SBATCH --error=/jet/home/cmiyamot/upscaling-ddpm/sbatch/logging/FashionMNIST_%j.err

set -x  # Echo commands to stdout

module activate AI/pytorch_23.02-1.13.1-py3 # Activate PyTorch

cd /jet/home/cmiyamot/upscaling-ddpm


# Run code
python train.py --dataset CelebA --batch_size 64
