#!/bin/bash
module load anaconda/2023a-pytorch
conda activate jax

python using_classphasecond.py
