#!/bin/bash
module load anaconda/2023a-pytorch
conda activate jax

python train_classphasecond.py
#python allclass_classphasecond.py
