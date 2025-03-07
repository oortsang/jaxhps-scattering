#!/bin/bash



# iterate over l values
for l in 2 3 4 5 6 7 8 9; do
    python driver_time_solver.py -l $l\
        -p 16\
        -n 5 \
        -out_fp data/timing/timing_gpu_our_recomp_l_$l.mat \
        --recomputation Ours
done
