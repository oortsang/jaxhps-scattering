#!/bin/bash



python wave_scattering_compute_reference_soln.py --scattering_potential GBM_1 -k 10 -l 7 --plot_utot
python wave_scattering_compute_reference_soln.py --scattering_potential luneburg -k 200 -l 7 --plot_utot
python wave_scattering_compute_reference_soln.py --scattering_potential GBM_1 -k 200 -l 7 --plot_utot
python wave_scattering_compute_reference_soln.py --scattering_potential luneburg -k 100 -l 7 --plot_utot
python wave_scattering_compute_reference_soln.py --scattering_potential GBM_1 -k 100 -l 7 --plot_utot
python wave_scattering_compute_reference_soln.py --scattering_potential gauss_bumps -k 100 -l 7 --plot_utot
python wave_scattering_compute_reference_soln.py --scattering_potential gauss_bumps -k 200 -l 7 --plot_utot