#!/bin/bash
module load anaconda/2023a-pytorch
conda activate jax

echo "My task ID: " $LLSUB_RANK
echo "Number of Tasks: " $LLSUB_SIZE

python inferring_spect_from_Ia_Goldstein_LSST_batch.py $LLSUB_RANK $LLSUB_SIZE