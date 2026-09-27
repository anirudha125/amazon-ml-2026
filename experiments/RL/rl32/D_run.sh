#!/bin/bash
cd /teamspace/studios/this_studio/amazon-ml-challenge-2026/experiments/RL/rl32
export LGB_THREADS=6 OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 MKL_NUM_THREADS=6
nice -n 10 python D_loco.py prep && nice -n 10 python D_loco.py fit US && nice -n 10 python D_loco.py st && nice -n 10 python D_loco.py fit IN MIX
