#!/bin/bash
module load anaconda/2023a-pytorch
conda activate jax

python make_training.py