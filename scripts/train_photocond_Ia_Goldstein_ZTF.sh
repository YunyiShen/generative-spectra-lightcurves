#!/bin/bash
module load anaconda/2023a-pytorch
conda activate jax

python train_photocond_Ia_Goldstein_ZTF.py
cd ../experiments
python inferring_spect_from_Ia_Goldstein.py
python visual_Ia_Goldstein.py
