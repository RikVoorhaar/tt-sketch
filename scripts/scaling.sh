#!/bin/bash
#SBATCH --job-name=tt-sketch-timings
#SBATCH --partition=private-math-cpu
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem-per-cpu=4000 # in MB
#SBATCH --mail-user=rik.voorhaar@unige.ch
#SBATCH --mail-type=ALL
srun python plot_dimension_scaling.py
