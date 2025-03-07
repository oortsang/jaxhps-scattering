#!/bin/bash



# iterate over l values
for l in 4 5 6 7; do
    python driver_time_solver.py -l $l\
        -p 16\
        -n 5 \
        -out_fp data/timing/timing_multicore_cpu_l_$l.mat
done
