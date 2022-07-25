#!/bin/bash
python -m cProfile -o hmt-$HOSTNAME.pstats simple_timing.py --method="HMT"
python -m cProfile -o stta-$HOSTNAME.pstats simple_timing.py --method="STTA"
