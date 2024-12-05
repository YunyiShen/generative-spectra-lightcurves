#!/bin/bash
module load anaconda/2023a-pytorch
conda activate jax

python train_photocond_Ia_Goldstein_LSST.py $1 $2 $3 # midfilter, centering, realistic
cd ../experiments
python inferring_spect_from_Ia_Goldstein_LSST.py $1 $2 $3
python visual_Ia_Goldstein_LSST.py $1 $2 $3
