#!/bin/bash
module load anaconda/2023a-pytorch
conda activate jax

python inferring_spect_from_Ia_Goldstein_LSST.py 3 false true
python visual_Ia_Goldstein_LSST.py 3 false true