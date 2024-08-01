#!/bin/bash
module load anaconda/2023a-pytorch
conda activate jax

python train_classphasecond_crossattn.py
python allclass_classphasecond_crossattn.py
