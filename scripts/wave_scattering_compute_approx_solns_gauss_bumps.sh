#!/bin/bash



# Nested loops to iterate over all combinations
for l in 2 3 4 5 6 7; do
    # Loop through p values
    for p in 8 12 16; do
        echo "Running with l=$l, p=$p"
        python wave_scattering_compute_approx_soln.py -l $l\
         -p $p\
         --n_time_samples 5 \
         -k 100 \
         --scattering_potential gauss_bumps
    done
done

# Nested loops to iterate over all combinations
for l in 2 3 4 5 6 7; do
    # Loop through p values
    for p in 8 12 16; do
        echo "Running with l=$l, p=$p"
        python wave_scattering_compute_approx_soln.py -l $l\
         -p $p\
         --n_time_samples 5 \
         -k 200 \
         --scattering_potential gauss_bumps
    done
done