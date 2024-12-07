#!/bin/bash
module load anaconda/2023a-pytorch
conda activate jax

#python prepare_goldstein_LSST_randomphase.py
python prepare_goldstein_realisticLSST_withphase.py true