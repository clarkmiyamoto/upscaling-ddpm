#!/bin/bash

#SBATCH --job-name=train
#SBATCH --nodes=1
#SBATCH -p GPU
#SBATCH -t 2:00:00
#SBATCH --gpus=v100-16:8 

set -x  # Echo commands to stdout

module activate AI/pytorch_23.02-1.13.1-py3 # Activate PyTorch

cd /jet/home/cmiyamot/upscaling-ddpm


# Run code
python train_FashionMNIST.py
