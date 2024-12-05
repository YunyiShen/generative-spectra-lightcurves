#!/bin/bash
module load anaconda/2023a-pytorch
conda activate jax

python inferring_spect_from_Ia_Goldstein_LSST_batch.py
python visual_Ia_Goldstein_LSST_batch.py