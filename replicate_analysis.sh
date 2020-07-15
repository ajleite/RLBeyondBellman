#!/bin/bash

# Script to replicate our statistics and figures for ALIFE 2020.

# You should either extract alife-results.tar.xz directly in the root
# folder of the repository  or run replicate_experiments.sh first.

# Some of the figures in the paper were re-arranged in Inkscape after
# being created here.

echo "New Actor Training Statistics:"
python3 -m alife.stats1_new_actor_training
echo "Q-map / Gradient Correlation Statistics:"
python3 -m alife.stats2_Qmaps_gradients

echo "Generating and showing figures..."
python3 -m alife.figure1_critic_development
python3 -m alife.figure2_new_actor_training
python3 -m alife.figure3_spectra
python3 -m alife.figure4_clustering_analysis

