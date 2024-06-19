#!/bin/bash
module load anaconda/2023a-pytorch
conda activate jax

python test_photometrycond.py