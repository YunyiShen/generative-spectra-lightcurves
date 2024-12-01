#!/bin/bash
module load anaconda/2023a-pytorch
conda activate jax

python inferring_spect_from_LC.py
python visual.py