#!/bin/bash
module load anaconda/2023a-pytorch
conda activate jax

python fit_salt3.py
python evaluation.py
python evaluate_covering.py
python make_plot.py
python plot_couple_examples.py