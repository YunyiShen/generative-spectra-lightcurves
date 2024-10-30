#!/bin/bash
module load anaconda/2023a-pytorch
conda activate jax

python train_photocond_Ia_ZTF.py
cd ../experiments
python inferring_spect_from_Ia_ZTF.py
python visual_Ia_ZTF.py
