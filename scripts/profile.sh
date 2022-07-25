#!/bin/bash
#SBATCH --job-name=tt-sketch-timings
#SBATCH --partition=private-math-cpu
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem-per-cpu=4000 # in MB
#SBATCH --mail-user=rik.voorhaar@unige.ch
#SBATCH --mail-type=ALL
python -m cProfile -o hmt-$HOSTNAME.pstats simple_timing.py --method="HMT"
python -m cProfile -o stta-$HOSTNAME.pstats simple_timing.py --method="STTA"
